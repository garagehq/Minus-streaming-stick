#!/usr/bin/env python3
"""Clip farm: keep LeBron-heavy, captioned YouTube videos playing so Minus's
training-clip collector (src/asr_clips.py) builds a labelled dataset.

Loop:
  1. Find candidate videos with yt-dlp searches. Keep 5-90 min videos whose
     caption track mentions LeBron/King James at least MIN_MENTIONS times,
     and save that caption track (the labels).
  2. Play one on the Android TV over ADB. Skip it if the player never
     reaches PLAYING (Restricted Mode, region lock) or the ASR hears
     nothing.
  3. While it plays, log the video position against the box's monotonic
     clock every SYNC_EVERY_S (playlog/<date>.jsonl), and keep Minus's
     random-clip interval at RANDOM_INTERVAL_S: with a caption track every
     clip is labelable, not just the ones around a name.
  4. Every LABEL_EVERY_S, run tools/label_clips.py: clip capture time ->
     video position -> caption words -> trimmed WAV + manifest line in
     /home/radxa/asr_dataset.

Run as root (ADB key in /root/.android). Installed as minus-clipfarm.service.

usage: sudo python3 tools/clip_farm.py [--ip 192.168.1.203] [--once VIDEO_ID]
"""
import argparse
import datetime
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_clips  # noqa: E402

FARM_DIR = label_clips.FARM_DIR
STATE = FARM_DIR / 'state.json'
MINUS = os.environ.get('MINUS_URL', 'http://localhost')
ADBKEY = os.environ.get('ADBKEY', '/root/.android/adbkey')

SYNC_EVERY_S = 5.0
AD_DROP_S = 10          # position this far behind the furthest seen: ad or new video
AD_MAX_S = 240          # ...and still behind after this long: autoplay moved on
END_SLACK_S = 20        # a drop this close to the end is the video ending
START_WAIT_S = 30       # every video that played reported PLAYING within ~6 s; Restricted Mode blocks never do
MISMATCH_EVERY_S = 60   # how often ASR is compared with the captions at the current position
MISMATCH_MAX = 0.15     # share of heard words found in the captions; normal videos ~0.6-0.8
MISMATCH_RUNS = 5       # this many low checks in a row: something else is playing (ads run up to ~3 min)
PREROLL_S = 180         # a drop this early is the pre-roll ad giving way to the video
RANDOM_INTERVAL_S = float(os.environ.get('CLIP_FARM_RANDOM_S', '20'))
LABEL_EVERY_S = 600
MAX_PER_VIDEO_S = float(os.environ.get('CLIP_FARM_MAX_PER_VIDEO_S', '2400'))
MIN_MENTIONS = 3
MIN_DUR, MAX_DUR = 300, 5400

QUERIES = [
    'lebron james full game highlights commentary',
    'lebron james postgame interview',
    'lebron james mic\'d up',
    'lebron james career retrospective',
    'lebron james documentary',
    'nba finals lebron james full game',
    'lebron james first take debate',
    'lebron james undisputed skip bayless',
    'lebron james high school game espn',
    'lebron james heat 2012 finals highlights',
    'cavaliers 2016 finals game 7 full game',
    'lebron james breaks scoring record broadcast',
    'lebron james 40000 points',
    'lebron james olympics 2024 highlights commentary',
    'lebron james podcast interview',
    'lebron james mind the game podcast',
    'lebron james vs celtics full game highlights',
    'lebron james lakers full game highlights',
    'nba today lebron james',
    'the decision lebron james 2010',
]
HELD_OUT = label_clips.HELD_OUT
NAME_RE = re.compile(r'\blebron\b|\bking james\b', re.I)
POS_RE = re.compile(r'state=PLAYING\(3\), position=(\d+), buffered position=\d+, '
                    r'speed=([0-9.]+), updated=(\d+)')

_stop = False


def log(msg):
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------- state

def load_state():
    try:
        with open(STATE) as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {'seen': [], 'queue': [], 'played': {}, 'query_idx': 0}
    st['seen'] = sorted(set(st['seen']) | HELD_OUT)
    st['queue'] = [q for q in st['queue'] if q['id'] not in HELD_OUT]
    return st


def save_state(st):
    tmp = STATE.with_suffix('.tmp')
    json.dump(st, open(tmp, 'w'), indent=1)
    os.replace(tmp, STATE)


# ---------------------------------------------------------------- minus API

def api(path, body=None, timeout=5):
    req = urllib.request.Request(MINUS + path, method='POST' if body is not None else 'GET',
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def set_random_interval(seconds):
    try:
        api('/api/name-mute/clips', {'random_interval_s': seconds})
    except Exception as e:
        log(f"could not set clip interval: {e}")


def request_nickname_clips(vid, hits, lo, pos, board_t):
    """Ask Minus to keep a clip for each caption nickname (label_clips.NICKNAMES)
    played since the last sync. Not searched for; just kept when they occur.
    Returns the position checked up to."""
    for key, p in hits:
        if lo < p <= pos and pos - p < 20:      # still inside the 30 s tap buffer
            try:
                api('/api/name-mute/clips', {'clip_at': board_t - (pos - p),
                                             'label': f"{key} {vid}@{p:.1f}"})
                log(f"{vid}: nickname '{key}' at {p:.0f}s, clip requested")
            except Exception as e:
                log(f"nickname clip request failed: {e}")
    return max(lo, pos)


def asr_transcript():
    try:
        return api('/api/status')['asr'].get('last_transcript', '') or ''
    except Exception:
        return ''


# ---------------------------------------------------------------- discovery

def ytdlp(args, timeout=120):
    # yt-dlp is pip-installed for the radxa user; point root's Python at that
    # user site (runuser would log a PAM session per call to the journal).
    env = dict(os.environ, PYTHONUSERBASE='/home/radxa/.local')
    if os.geteuid() == 0:
        env['XDG_CACHE_HOME'] = '/root/.cache'
    return subprocess.run([sys.executable, '-m', 'yt_dlp'] + args, capture_output=True,
                          text=True, timeout=timeout, env=env)


def fetch_captions(vid):
    """Download the English caption track; returns its path or None."""
    cap_dir = FARM_DIR / 'captions'
    for f in cap_dir.glob(f'{vid}.*'):
        if f.suffix in ('.json3', '.vtt') and f.stem == vid:
            return f
    tmp = cap_dir / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    ytdlp(['-q', '--skip-download', '--write-subs', '--write-auto-subs',
           '--sub-langs', 'en,en-US,en-GB,en-orig', '--sub-format', 'json3/vtt/best',
           '-o', str(tmp / '%(id)s'), f'https://www.youtube.com/watch?v={vid}'])
    got = sorted(tmp.glob(f'{vid}.*'))
    # Prefer word-timed json3, then the plain English track.
    got.sort(key=lambda f: (f.suffix != '.json3', '.en.' not in f.name))
    best = None
    for f in got:
        if f.suffix in ('.json3', '.vtt') and best is None:
            best = cap_dir / f'{vid}{f.suffix}'
            os.replace(f, best)
        else:
            f.unlink()
    return best


def count_mentions(path):
    text = path.read_text(errors='ignore')
    if path.suffix == '.json3':
        try:
            text = ' '.join(sg.get('utf8', '') for ev in json.loads(text).get('events', [])
                            for sg in ev.get('segs') or [])
        except ValueError:
            return 0
    return len(NAME_RE.findall(text))


def discover(st, want=4):
    """Add up to `want` new videos to the queue."""
    added = 0
    for _ in range(len(QUERIES)):
        if added >= want or _stop:
            break
        q = QUERIES[st['query_idx'] % len(QUERIES)]
        st['query_idx'] += 1
        r = ytdlp(['--flat-playlist', '--print', '%(id)s|%(duration)s|%(title)s',
                   f'ytsearch25:{q}'])
        for line in r.stdout.splitlines():
            parts = line.split('|', 2)
            if len(parts) != 3 or parts[0] in st['seen']:
                continue
            vid, dur, title = parts
            st['seen'].append(vid)
            try:
                dur = float(dur)
            except ValueError:
                continue
            if not MIN_DUR <= dur <= MAX_DUR:
                continue
            cap = fetch_captions(vid)
            n = count_mentions(cap) if cap else 0
            if n < MIN_MENTIONS:
                continue
            st['queue'].append({'id': vid, 'duration': dur, 'title': title[:80],
                                'mentions': n, 'captions': cap.suffix})
            log(f"queued {vid} ({dur / 60:.0f} min, {n} mentions, {cap.suffix}): {title[:60]}")
            added += 1
            if added >= want:
                break
        save_state(st)
    random.shuffle(st['queue'])
    return added


# ---------------------------------------------------------------- playback

def connect(ip):
    from adb_shell.adb_device import AdbDeviceTcp
    from adb_shell.auth.sign_pythonrsa import PythonRSASigner
    d = AdbDeviceTcp(ip, 5555, default_transport_timeout_s=10)
    signer = PythonRSASigner(open(ADBKEY + '.pub').read(), open(ADBKEY).read())
    d.connect(rsa_keys=[signer], auth_timeout_s=5)
    return d


def sync_sample(dev):
    t0 = time.monotonic()
    out = dev.shell("cat /proc/uptime; dumpsys media_session | grep -m1 'state=PLAYING'",
                    read_timeout_s=10)
    t1 = time.monotonic()
    m = POS_RE.search(out)
    if not m:
        return None
    return {'rtt': round(t1 - t0, 4), 'board_mid': round((t0 + t1) / 2, 4),
            'device_uptime': float(out.split()[0]),
            'position': int(m.group(1)) / 1000.0, 'speed': float(m.group(2)),
            'updated': int(m.group(3)) / 1000.0}


def best_sync(dev, n=3):
    s = [x for x in (sync_sample(dev) for _ in range(n)) if x]
    return min(s, key=lambda x: x['rtt']) if s else None


def caption_match(heard, captions):
    """Share of the content words ASR heard that appear in `captions`
    (caption words near the current position). None if too little was heard."""
    words = {w for t in heard for w in label_clips.normalize(t).split() if len(w) >= 4}
    if len(words) < 6:
        return None
    cap = {label_clips.normalize(w) for w in captions}
    return len(words & cap) / len(words)


def playlog(rec):
    d = FARM_DIR / 'playlog'
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{datetime.date.today():%Y%m%d}.jsonl", 'a') as f:
        f.write(json.dumps(rec) + '\n')


def play(dev_box, ip, item, last_label):
    """Play one video until it ends; returns (status, seconds, last_label)."""
    vid, duration = item['id'], item['duration']
    dev = dev_box[0]
    t_launch = time.monotonic()
    dev.shell(f"am start -a android.intent.action.VIEW -d "
              f"'https://www.youtube.com/watch?v={vid}&t=0s' com.google.android.youtube.tv")
    set_random_interval(RANDOM_INTERVAL_S)
    t_start = time.monotonic()
    playlog({'type': 'start', 'video': vid, 't': t_start, 'wall': time.time()})

    def fresh(x):
        # The previous video's session keeps reporting PLAYING for a moment;
        # only trust a session whose last state change came after the launch
        # (logged: old sessions changed state >=0.43 s before it, new ones >=0.66 s after).
        return x and x['updated'] >= x['device_uptime'] - (x['board_mid'] - t_launch) + 0.2

    first = None
    while time.monotonic() - t_start < START_WAIT_S and not _stop:   # loading; blocked videos never start
        first = best_sync(dev, n=3)
        if fresh(first):
            break
        first = None
        time.sleep(5)
    if not first:
        return ('interrupted' if _stop else 'never_playing'), 0, last_label

    # Media-session 'position' is the position at 'updated' (the last state
    # change), so it barely moves while playing; extrapolate to the sync time.
    def cur(x):
        return label_clips.vpos(x, x['board_mid'])

    heard = False
    max_pos, last_playing, dropped_at = cur(first), time.monotonic(), None
    cap_words = label_clips.caption_words(FARM_DIR, vid)
    nick_hits, nick_from = label_clips.nickname_hits(cap_words), cur(first)
    window, last_check, low_runs = [], time.monotonic(), 0   # [(pos, transcript)]
    playlog({'type': 'sync', 'video': vid, **first})
    while not _stop:
        now = time.monotonic()
        if now - t_start > MAX_PER_VIDEO_S:
            status = 'time_cap'
            break
        if not heard:
            heard = bool(asr_transcript())
            if not heard and now - t_start > 75:
                status = 'silent'
                break
        try:
            s = best_sync(dev)
        except Exception as e:
            log(f"adb error: {e}; reconnecting")
            try:
                dev_box[0] = dev = connect(ip)
            except Exception:
                time.sleep(10)
            continue
        if s:
            last_playing = now
            pos = cur(s)
            if pos < max_pos - AD_DROP_S and now - t_start < PREROLL_S:
                max_pos = nick_from = pos      # pre-roll ad over; the video starts
            if pos < max_pos - AD_DROP_S:
                # A mid-roll ad (content resumes where it paused) or autoplay
                # moved on to another video (it never comes back).
                if max_pos >= duration - END_SLACK_S:
                    status = 'ended'           # autoplay right after the end; syncs can miss the last seconds
                    break
                dropped_at = dropped_at or now
                if now - dropped_at > AD_MAX_S:
                    status = 'next_video'
                    break
            else:
                dropped_at = None
                max_pos = max(max_pos, pos)
                playlog({'type': 'sync', 'video': vid, **s})
                nick_from = request_nickname_clips(vid, nick_hits, nick_from, pos, s['board_mid'])
                window.append((pos, asr_transcript()))
                if cap_words and now - last_check >= MISMATCH_EVERY_S:
                    lo, hi = min(p for p, _ in window) - 15, max(p for p, _ in window) + 5
                    m = caption_match([t for _, t in window],
                                      [w for w, a, _ in cap_words if lo <= a <= hi])
                    window, last_check = [], now
                    if m is not None:
                        low_runs = low_runs + 1 if m < MISMATCH_MAX else 0
                        if low_runs >= MISMATCH_RUNS:
                            status = 'mismatch'    # audio isn't this video (long ad, wrong video)
                            break
                if pos >= duration - 3:
                    status = 'ended'
                    break
        elif now - last_playing > 90:
            status = 'stopped'             # ended screen, error, or long ad
            break
        if now - last_label > LABEL_EVERY_S:
            label()
            last_label = time.monotonic()
        time.sleep(SYNC_EVERY_S)
    else:
        status = 'interrupted'
    playlog({'type': 'end', 'video': vid, 't': time.monotonic(), 'status': status})
    return status, round(time.monotonic() - t_start), last_label


def label():
    try:
        label_clips.label_all(log=log)
        log(f"dataset: {label_clips.summary()}")
    except Exception as e:
        log(f"labeling failed: {e}")


# ---------------------------------------------------------------- main

def main():
    global _stop
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', default=os.environ.get('CLIP_FARM_TV_IP', '192.168.1.203'))
    ap.add_argument('--once', help='play just this video id, then exit')
    args = ap.parse_args()

    def on_signal(*_):
        global _stop
        _stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    (FARM_DIR / 'captions').mkdir(parents=True, exist_ok=True)
    st = load_state()
    # Started alongside minus.service (boot, restarts): its API takes ~30 s.
    for _ in range(60):
        try:
            api('/api/status', timeout=3)
            break
        except Exception:
            if _stop:
                return
            time.sleep(5)
    try:
        api('/api/asr/enable', {})
    except Exception as e:
        log(f"could not enable ASR: {e}")
    dev_box = [connect(args.ip)]
    last_label = time.monotonic()
    log(f"clip farm started (TV {args.ip}, random clips every {RANDOM_INTERVAL_S:.0f}s)")
    try:
        while not _stop:
            if args.once:
                if args.once in HELD_OUT:
                    log(f"note: {args.once} is a held-out evaluation video")
                cap = fetch_captions(args.once)
                item = {'id': args.once, 'duration': 1e9, 'title': '', 'mentions': 0}
                log(f"playing {args.once} (captions: {cap})")
            else:
                if label_clips.dataset_bytes(label_clips.DATASET_DIR) >= label_clips.DATASET_MAX_BYTES:
                    label()
                    log(f"dataset at its {label_clips.DATASET_MAX_BYTES / 1e6:.0f} MB cap "
                        f"(ASR_DATASET_MAX_MB); not playing. Checking again in 30 min")
                    for _ in range(360):
                        if _stop:
                            break
                        time.sleep(5)
                    continue
                if len(st['queue']) < 2:
                    discover(st)
                if not st['queue']:
                    log("no candidates found; retrying in 10 min")
                    time.sleep(600)
                    continue
                item = st['queue'].pop(0)
                save_state(st)
                log(f"playing {item['id']} ({item['duration'] / 60:.0f} min, "
                    f"{item['mentions']} mentions): {item['title']}")
            status, secs, last_label = play(dev_box, args.ip, item, last_label)
            log(f"{item['id']}: {status} after {secs}s")
            st['played'][item['id']] = {'status': status, 'secs': secs, 'when': time.time()}
            if status == 'interrupted' and secs < 60 and not args.once:
                st['queue'].insert(0, item)    # stopped right at its start: play it next run
            save_state(st)
            if args.once:
                break
    finally:
        set_random_interval(300)
        label()
        log("clip farm stopped")


if __name__ == '__main__':
    main()
