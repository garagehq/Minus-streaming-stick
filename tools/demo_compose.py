#!/usr/bin/env python3
"""Compose a demo video from a tools/demo_take.py capture.

What a viewer of Minus saw and heard: the main picture is the live input
shifted by the A/V delay (the output shows each frame DELAY seconds late),
with Minus's recorded output audio, so mutes land where they really did.
A small live-input inset shows how far behind the output runs. Overlays:
a MUTED badge (output audio is exact digital silence), a mute timeline,
CPU temperature and big-core clock caps (throttling), and what Minus's ASR
was hearing at that moment.

usage: python3 tools/demo_compose.py TAKE_DIR AUDIO_DIR OUT.mp4 "TAG" "TITLE" "SUBTITLE"
"""
import bisect
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1280, 720, 30, 48000
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def live_time_of(syncs, start):
    """Box-monotonic time at which the live input reached video position `start`."""
    est = [s['board_mid'] + (start - s['pos']) for s in syncs if abs(s['pos'] - start) < 40]
    if not est:
        sys.exit('no syncs near the segment start')
    return float(np.median(est))


def output_audio(audio_dir, t0, seconds):
    for idx_path in sorted(glob.glob(os.path.join(audio_dir, '*.idx'))):
        if os.path.getsize(idx_path) == 0:
            continue
        rows = np.loadtxt(idx_path, ndmin=2)
        if len(rows) < 2 or not (rows[0, 0] <= t0 and rows[-1, 0] >= t0 + seconds):
            continue
        f0 = int(np.interp(t0, rows[:, 0], rows[:, 1]))
        n = int(seconds * SR)
        raw = np.fromfile(idx_path[:-4] + '.s16', dtype=np.int16, count=n * 2, offset=f0 * 4)
        return raw.reshape(-1, 2)
    sys.exit(f'no output audio in {audio_dir} covers {t0:.1f}..{t0 + seconds:.1f}')


def mute_mask(audio, frames):
    """Per video frame: True when the output audio is digital silence (muted)."""
    per = SR // FPS
    return [bool(len(audio[k * per:(k + 1) * per]) and
                 not np.any(audio[k * per:(k + 1) * per])) for k in range(frames)]


def throttle_info(entry):
    """(lowest big-core clock cap in GHz, any big cluster capped below its own max)."""
    clocks = entry.get('clocks') or {}
    big = [c for c in clocks.values() if c.get('max') and c['max'] > 2000000]
    if not big:
        return None, False
    capped = [c for c in big if c.get('cap') and c['cap'] < c['max']]
    cap = min((c['cap'] for c in capped), default=max(c['max'] for c in big))
    return cap / 1e6, bool(capped)


def main():
    take, audio_dir, out, tag, title, subtitle = sys.argv[1:7]
    meta = json.load(open(os.path.join(take, 'take.json')))
    seconds = meta['seconds']
    syncs = load_jsonl(os.path.join(take, 'syncs.jsonl'))
    tl = load_jsonl(os.path.join(take, 'timeline.jsonl'))
    tl_t = [e['t'] for e in tl]
    delays = [e['delay_s'] for e in tl if e.get('delay_s')]
    delay = float(np.median(delays)) if delays else meta['delay_s']
    fidx = np.loadtxt(os.path.join(take, 'frames.idx'), ndmin=2)
    f_t = fidx[:, 1]

    t_live = live_time_of(syncs, meta['start_s'])
    t0 = t_live + delay
    nframes = int(seconds * FPS)
    audio = output_audio(audio_dir, t0, seconds)
    muted = mute_mask(audio, nframes)
    # Mute intervals for the timeline strip.
    spans, k = [], 0
    while k < nframes:
        if muted[k]:
            j = k
            while j < nframes and muted[j]:
                j += 1
            if j - k >= 2:
                spans.append((k / nframes, j / nframes))
            k = j
        else:
            k += 1

    wav = out[:-4] + '.s16'
    audio.tofile(wav)
    ff = subprocess.Popen(
        ['ffmpeg', '-y', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
         '-f', 's16le', '-ar', str(SR), '-ac', '2', '-i', wav,
         '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p',
         '-c:a', 'aac', '-b:a', '160k', '-t', str(seconds), '-movflags', '+faststart', out],
        stdin=subprocess.PIPE)

    cache = {}

    def frame(i, size):
        key = (i, size)
        if key not in cache:
            if len(cache) > 12:
                cache.clear()
            im = Image.open(os.path.join(take, 'frames', f'{int(fidx[i, 0]):06d}.jpg'))
            im.draft('RGB', size)
            cache[key] = im.convert('RGB').resize(size, Image.BILINEAR)
        return cache[key]

    def nearest(t):
        j = bisect.bisect_left(f_t, t)
        if j <= 0:
            return 0
        if j >= len(f_t):
            return len(f_t) - 1
        return j if f_t[j] - t < t - f_t[j - 1] else j - 1

    f_tag, f_title, f_sub = font(30, True), font(24, True), font(18)
    f_stat, f_small, f_mute = font(17, True), font(16), font(44, True)
    pip_w, pip_h = 352, 198
    for k in range(nframes):
        t = t0 + k / FPS
        img = frame(nearest(t - delay), (W, H)).copy()
        d = ImageDraw.Draw(img, 'RGBA')
        # Title bar.
        d.rectangle([0, 0, W, 74], fill=(0, 0, 0, 185))
        tag_w = d.textlength(tag, font=f_tag) + 28
        d.rounded_rectangle([14, 14, 14 + tag_w, 60], 8,
                            fill=(200, 40, 40, 255) if tag.startswith('BEFORE') else (30, 130, 70, 255))
        d.text((28, 19), tag, font=f_tag, fill='white')
        d.text((tag_w + 34, 12), title, font=f_title, fill='white')
        d.text((tag_w + 34, 44), subtitle, font=f_sub, fill=(210, 210, 210))
        # Live inset.
        px, py = W - pip_w - 16, 90
        img.paste(frame(nearest(t), (pip_w, pip_h)), (px, py))
        d.rectangle([px - 2, py - 2, px + pip_w + 1, py + pip_h + 1], outline=(255, 255, 255, 220), width=2)
        d.rectangle([px, py + pip_h - 26, px + pip_w, py + pip_h], fill=(0, 0, 0, 170))
        d.text((px + 8, py + pip_h - 23), f'LIVE INPUT  ·  output is {delay:.1f} s behind',
               font=f_small, fill='white')
        # Stats.
        e = tl[max(0, bisect.bisect_right(tl_t, t) - 1)]
        temps = e.get('temps') or {}
        hot = max(temps.values()) if temps else None
        cap, throttled = throttle_info(e)
        lines = []
        if hot is not None:
            s = f'CPU {hot:.0f}°C (throttles at 85°C)'
            if cap:
                s += f'   big cores capped at {cap:.2f} GHz' if throttled else f'   big cores {cap:.2f} GHz'
            lines.append((s + ('   THROTTLING' if throttled else ''),
                          (255, 90, 90) if throttled or hot >= 85 else (255, 255, 255)))
        heard = (e.get('asr_text') or '').strip()
        eng = e.get('asr_engine') or 'ASR'
        lines.append((f'{eng} hears: “{heard[-40:]}”' if heard else f'{eng} hears: …', (230, 230, 230)))
        load = []
        if e.get('cpu_percent') is not None:
            load.append(f'CPU load {e["cpu_percent"]:.0f}%')
        if e.get('asr_latency_s'):
            load.append(f'speech recognition takes {e["asr_latency_s"]:.1f} s per window')
        if load:
            busy = (e.get('cpu_percent') or 0) >= 75
            lines.append(('   ·   '.join(load), (255, 170, 80) if busy else (200, 200, 200)))
        # Stats sit top-left, clear of the TV's own captions at the bottom.
        sw = 560
        d.rectangle([0, 80, sw, 88 + 26 * len(lines)], fill=(0, 0, 0, 160))
        for i, (s, c) in enumerate(lines):
            d.text((12, 84 + 26 * i), s, font=f_stat, fill=c)
        # Mute timeline strip with playhead.
        sx0, sx1, sy = 18, W - 18, H - 22
        d.rectangle([0, sy - 26, W, H], fill=(0, 0, 0, 140))
        d.rectangle([sx0, sy, sx1, sy + 14], fill=(70, 70, 70, 255))
        for a, b in spans:
            d.rectangle([sx0 + a * (sx1 - sx0), sy, sx0 + b * (sx1 - sx0), sy + 14], fill=(220, 40, 40, 255))
        hx = sx0 + (k / nframes) * (sx1 - sx0)
        d.rectangle([hx - 2, sy - 4, hx + 2, sy + 18], fill='white')
        d.text((sx0, sy - 22), 'audio muted (red) over this clip', font=f_small, fill=(200, 200, 200))
        # MUTED badge.
        if muted[k]:
            bx = 622
            d.rounded_rectangle([bx, 90, bx + 280, 160], 12, fill=(210, 30, 30, 235))
            d.text((bx + 38, 99), 'MUTED', font=f_mute, fill='white')
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print(json.dumps({'out': out, 'delay_s': round(delay, 2), 'muted_frac': round(sum(muted) / nframes, 3),
                      'mutes': len(spans)}))


if __name__ == '__main__':
    main()
