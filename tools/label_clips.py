#!/usr/bin/env python3
"""Label training clips from YouTube caption tracks.

tools/clip_farm.py plays captioned videos on the TV and logs, every few
seconds, which video was playing at which position (on the box's monotonic
clock). Minus's clip collector (src/asr_clips.py) stamps each clip with the
monotonic time of its first sample. This script joins the two:

  clip capture time -> video position (only inside stretches where
  consecutive syncs agree, so mid-roll ads that stop the video clock are
  excluded) -> caption words spoken during the clip -> the audio trimmed to
  those words + the text, appended to a NeMo-style manifest.

Output (DATASET_DIR, default /home/radxa/asr_dataset):
  wavs/<clip>.wav     16 kHz mono, trimmed to the labelled words
  manifest.jsonl      {"audio_filepath", "duration", "text", "text_raw",
                       "video_id", "video_start", "kind", "asr_overlap",
                       "has_name", "nicknames"}
  labeled.txt         clip names already processed (labelled or rejected)

usage: python3 tools/label_clips.py [CLIPS_DIR] [FARM_DIR] [DATASET_DIR]
"""
import bisect
import json
import os
import re
import sys
import wave
from pathlib import Path

CLIPS_DIR = Path(os.environ.get('MINUS_ASR_CLIPS_DIR', '/home/radxa/Minus/screenshots/asr_clips'))
FARM_DIR = Path(os.environ.get('CLIP_FARM_DIR', '/home/radxa/clip_farm'))
DATASET_DIR = Path(os.environ.get('ASR_DATASET_DIR', '/home/radxa/asr_dataset'))
DATASET_MAX_BYTES = int(float(os.environ.get('ASR_DATASET_MAX_MB', '10000')) * 1e6)

SR = 16000
SYNC_AGREE_S = 0.3      # consecutive syncs this close are one steady stretch
MAX_SYNC_GAP_S = 20.0   # a stretch breaks if syncs are further apart than this
EDGE_MARGIN_S = 0.1     # words must lie this far inside the clip
LEAD_S, TAIL_S = 0.15, 0.25   # audio kept around the labelled words
MIN_WORDS, MIN_SPAN_S = 3, 1.5
# Share of label words that Minus's own ASR also heard somewhere in the clip.
# Correct labels score ~0.7-0.9; labels from the wrong video (autoplay, an ad,
# a resumed position) score ~0-0.3. Clips with no ASR text are rejected too.
MIN_ASR_OVERLAP = float(os.environ.get('ASR_LABEL_MIN_OVERLAP', '0.45'))
NAME_RE = re.compile(r"\blebron\b|\bking\b(?!s)", re.I)
# Nicknames kept for possible later training (not muted). Matched on
# normalize()d text, so hyphens and dots are already spaces.
NICKNAMES = {
    'king james': r"king james",
    'lbj': r"l b j|lbj",
    'kid from cleveland': r"kid from cleveland",
    'chosen one': r"chosen one",
    'captain lemerica': r"captain le ?merica",
    'bron bron': r"bron ?bron",
    'benjamin buckets': r"benjamin buckets",
    'l train': r"l ?train",
    'akron hammer': r"akron hammer",
    'little emperor': r"little emperor",
}
NICKNAME_RE = re.compile(r"\b(?:" + "|".join(f"(?P<n{k}>{rx})" for k, rx in
                                             enumerate(NICKNAMES.values())) + r")\b")
_NICK_KEYS = list(NICKNAMES)


def find_nicknames(text):
    """Nickname keys found in text (normalized first)."""
    return sorted({_NICK_KEYS[int(m.lastgroup[1:])] for m in NICKNAME_RE.finditer(normalize(text))})


def nickname_hits(words):
    """[(key, start_s)] for nicknames in a caption word list [(word, start, end)]."""
    toks = [(normalize(w), s) for w, s, _ in words]
    text, starts = '', []
    for t, s in toks:
        if not t:
            continue
        if text:
            text += ' '
        starts.append((len(text), s))
        text += t
    hits = []
    for m in NICKNAME_RE.finditer(text):
        at = max(s for off, s in starts if off <= m.start())
        hits.append((_NICK_KEYS[int(m.lastgroup[1:])], at))
    return hits

# Videos used to evaluate the muter (docs/ASR_BENCHMARKS.md, tools/roku_lebron.py
# held-out runs, false-mute runs). Never train on them, or the benchmarks stop
# meaning anything. CLIP_FARM_EXCLUDE adds more (comma-separated ids).
HELD_OUT = {
    '2nC9z57MuaI', 'MIWYB7qtq6c',                      # live videos A and B
    'W-KQ8rG2DRU', 'fnghnofk_y8', 'koplcs7BMIc', 'LDw0gfoJGPU',
    '7a6gnRvQqHQ', 'FzeIZLOrHic', 'EfWlOmGEfvU',       # roku_lebron.py list
    'wARQnIHvg14',                                     # offline NBA clip
    'NWXkL7x1RQQ', 'JAcD4zFHcTI', 'Rmjz1iGXPeU', 'AOYACk7m7Fk',
    'm0nr13xnRzU', 'CQF37Z4CEWg', '218gMy1KfyQ',       # false-mute runs
} | {v for v in os.environ.get('CLIP_FARM_EXCLUDE', '').split(',') if v}


def vpos(s, t):
    """Video position at box-monotonic time t, extrapolated from sync s."""
    return s['position'] + (t + s['device_uptime'] - s['board_mid'] - s['updated']) * s['speed']


def load_syncs(farm_dir: Path):
    """All play-log syncs, sorted by box time: [(t, video_id, sync_dict)]."""
    out = []
    for f in sorted((farm_dir / 'playlog').glob('*.jsonl')):
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('type') == 'sync':
                    out.append((r['board_mid'], r['video'], r))
    out.sort(key=lambda x: x[0])
    return out


def position_span(syncs, t0, t1):
    """(video_id, v0, v1) if [t0, t1] lies in one steady stretch, else None."""
    times = [s[0] for s in syncs]
    i = bisect.bisect_right(times, t0) - 1     # last sync at/before t0
    j = bisect.bisect_left(times, t1)          # first sync at/after t1
    if i < 0 or j >= len(syncs):
        return None
    chain = syncs[i:j + 1]
    vid = chain[0][1]
    for (ta, va, a), (tb, vb, b) in zip(chain, chain[1:]):
        if vb != vid or tb - ta > MAX_SYNC_GAP_S:
            return None
        if abs(vpos(a, tb) - vpos(b, tb)) > SYNC_AGREE_S:
            return None                         # clock jumped: ad, seek, stall
    a = chain[0][2]
    return vid, vpos(a, t0), vpos(a, t1)


_caption_cache = {}


def caption_words(farm_dir: Path, vid: str):
    """[(word, start, end)] for a video, from its json3 (word-level) or vtt
    (cue-level, words spread evenly over the cue) caption file."""
    if vid in _caption_cache:
        return _caption_cache[vid]
    words = []
    j = farm_dir / 'captions' / f'{vid}.json3'
    v = farm_dir / 'captions' / f'{vid}.vtt'
    if j.exists():
        starts = []
        with open(j) as fh:
            events = json.load(fh).get('events', [])
        for ev in events:
            t = ev.get('tStartMs')
            if t is None:
                continue
            for sg in ev.get('segs') or []:
                txt = (sg.get('utf8') or '').strip()
                if txt:
                    starts.append(((t + sg.get('tOffsetMs', 0)) / 1000.0,
                                   (t + ev.get('dDurationMs', 0)) / 1000.0, txt))
        starts.sort()
        for k, (s, ev_end, txt) in enumerate(starts):
            nxt = starts[k + 1][0] if k + 1 < len(starts) else ev_end
            for w in txt.split():
                words.append((w, s, min(max(nxt, s + 0.1), s + 1.0)))
    elif v.exists():
        cue_re = re.compile(r'(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)')
        cur = None
        lines = []
        with open(v) as fh:
            vtt_lines = list(fh)
        for line in vtt_lines + ['']:
            m = cue_re.match(line)
            if m:
                g = [int(x) for x in m.groups()]
                cur = (g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                       g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000)
                lines = []
            elif cur and line.strip():
                lines.append(re.sub(r'<[^>]+>', '', line.strip()))
            elif cur and not line.strip():
                ws = ' '.join(lines).split()
                if ws:
                    d = (cur[1] - cur[0]) / len(ws)
                    words += [(w, cur[0] + k * d, cur[0] + (k + 1) * d) for k, w in enumerate(ws)]
                cur = None
    # Drop sound tags like [Music] / [Applause].
    words = [w for w in words if not re.fullmatch(r'\[.*\]|\(.*\)|>>', w[0])]
    _caption_cache[vid] = words
    return words


def normalize(text):
    t = text.lower().replace('’', "'")
    t = re.sub(r"[^a-z0-9' ]+", ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def asr_overlap(meta, text):
    """Fraction of label words found in the clip's ASR windows (None if no ASR text)."""
    heard = set()
    for w in meta.get('asr_windows') or []:
        heard |= set(normalize(w.get('text', '')).split())
    words = text.split()
    if not heard or not words:
        return None
    return sum(w in heard for w in words) / len(words)


def dataset_bytes(ds: Path):
    return sum(f.stat().st_size for f in (ds / 'wavs').glob('*.wav')) if (ds / 'wavs').exists() else 0


def label_all(clips_dir=CLIPS_DIR, farm_dir=FARM_DIR, ds=DATASET_DIR, log=print):
    ds.mkdir(parents=True, exist_ok=True)
    (ds / 'wavs').mkdir(exist_ok=True)
    done_file = ds / 'labeled.txt'
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    syncs = load_syncs(farm_dir)
    if not syncs:
        log("[label] no play-log syncs yet")
        return {'labeled': 0, 'rejected': 0, 'pending': 0}
    last_sync = syncs[-1][0]
    size = dataset_bytes(ds)
    stats = {'labeled': 0, 'rejected': 0, 'pending': 0, 'full': False}
    new_done = []
    with open(ds / 'manifest.jsonl', 'a') as man:
        for meta_path in sorted(clips_dir.glob('*.json')):
            name = meta_path.stem
            if name in done:
                continue
            wav_path = meta_path.with_suffix('.wav')
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
            except (ValueError, OSError):
                continue
            t0 = meta.get('clip_start_mono')
            if t0 is None or not wav_path.exists():
                new_done.append(name)       # pre-farm clip: nothing to align
                stats['rejected'] += 1
                continue
            t1 = t0 + meta.get('duration_s', 10.0)
            if t1 > last_sync:
                stats['pending'] += 1       # play log not written that far yet
                continue
            new_done.append(name)
            span = position_span(syncs, t0, t1)
            if span is None:
                stats['rejected'] += 1
                continue
            vid, v0, v1 = span
            if vid in HELD_OUT:
                stats['rejected'] += 1         # evaluation video: never train on it
                continue
            ws = [w for w in caption_words(farm_dir, vid)
                  if w[1] >= v0 + EDGE_MARGIN_S and w[2] <= v1 - EDGE_MARGIN_S]
            if len(ws) < MIN_WORDS or ws[-1][2] - ws[0][1] < MIN_SPAN_S:
                stats['rejected'] += 1
                continue
            raw = ' '.join(x[0] for x in ws)
            ov = asr_overlap(meta, normalize(raw))
            if ov is None or ov < MIN_ASR_OVERLAP:
                stats['rejected'] += 1         # label doesn't match what was playing
                continue
            if size >= DATASET_MAX_BYTES:
                stats['full'] = True
                new_done.pop()              # retry once space is freed
                break
            a = max(0.0, ws[0][1] - LEAD_S - v0)
            b = min(v1 - v0, ws[-1][2] + TAIL_S - v0)
            with wave.open(str(wav_path)) as w:
                w.setpos(int(a * SR))
                frames = w.readframes(int((b - a) * SR))
            out = ds / 'wavs' / f'{name}.wav'
            with wave.open(str(out), 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(frames)
            if os.geteuid() == 0:
                os.chown(out, 1000, 1000)      # the farm runs as root; data is radxa's
            size += out.stat().st_size
            man.write(json.dumps({
                'audio_filepath': str(out), 'duration': round(b - a, 3),
                'text': normalize(raw), 'text_raw': raw, 'video_id': vid,
                'video_start': round(v0 + a, 3), 'kind': meta.get('kind'),
                'asr_overlap': round(ov, 2), 'has_name': bool(NAME_RE.search(raw)),
                'nicknames': find_nicknames(raw)}) + '\n')
            stats['labeled'] += 1
    if new_done:
        with open(done_file, 'a') as fh:
            fh.write('\n'.join(new_done) + '\n')
    if os.geteuid() == 0:
        for f in (ds, ds / 'wavs', ds / 'manifest.jsonl', done_file):
            if f.exists():
                os.chown(f, 1000, 1000)
    if stats['full']:
        log(f"[label] dataset at its {DATASET_MAX_BYTES / 1e6:.0f} MB cap; not adding more")
    log(f"[label] labeled {stats['labeled']}, rejected {stats['rejected']}, "
        f"waiting on play log {stats['pending']}")
    return stats


def summary(ds=DATASET_DIR):
    man = ds / 'manifest.jsonl'
    if not man.exists():
        return {'clips': 0, 'hours': 0.0, 'with_name': 0}
    with open(man) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return {'clips': len(rows), 'hours': round(sum(r['duration'] for r in rows) / 3600, 2),
            'with_name': sum(r['has_name'] for r in rows),
            'with_nickname': sum(bool(find_nicknames(r['text_raw'])) for r in rows),
            'videos': len({r['video_id'] for r in rows}),
            'bytes': dataset_bytes(ds)}


if __name__ == '__main__':
    a = sys.argv[1:]
    label_all(Path(a[0]) if a else CLIPS_DIR, Path(a[1]) if len(a) > 1 else FARM_DIR,
              Path(a[2]) if len(a) > 2 else DATASET_DIR)
    print(summary(Path(a[2]) if len(a) > 2 else DATASET_DIR))
