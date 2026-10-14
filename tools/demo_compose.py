#!/usr/bin/env python3
"""Compose a demo video from a tools/demo_take.py capture.

What a viewer of Minus saw and heard: the picture is the captured HDMI input
and the sound is Minus's recorded output audio (MINUS_RECORD_OUTPUT_DIR),
shown in sync with each other, so mutes land on the words they really cut.
Overlays: a title bar, a MUTED badge (output audio is exact digital silence)
and a mute timeline.

Both streams are lined up against the original video (SOURCE, e.g. from
yt-dlp, covering the segment with some margin): the output audio by
cross-correlating it with the source audio, the captured frames by
cross-correlating frame-to-frame change with the source's. That gives the
box-clock time at which each source position was captured and played out,
independent of media-session timing, and the real A/V delay is the
difference.

usage: python3 tools/demo_compose.py TAKE_DIR AUDIO_DIR OUT.mp4 "TAG" "TITLE" "SUBTITLE" \\
           SOURCE.mp4 SOURCE_START_S

SOURCE_START_S is the video position of SOURCE's first frame. "{delay}" in
SUBTITLE is replaced by the measured delay (e.g. "2.4").
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
ASR = 16000                # alignment sample rate
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
SIG_W, SIG_H = 64, 36


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def xcorr(x, y):
    """Valid-mode cross-correlation of y slid along x."""
    n = len(x) + len(y) - 1
    N = 1 << (n - 1).bit_length()
    r = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(y[::-1], N), N)[:n]
    return r[len(y) - 1:len(x)]


def source_audio(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-ac', '1', '-ar', str(ASR),
                          '-f', 's16le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32)


def source_signature(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', path, '-vf',
                          f'fps={FPS},scale={SIG_W}:{SIG_H},format=gray', '-f', 'rawvideo', '-'],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, SIG_H, SIG_W).astype(np.float32)
    return np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))


def load_output(audio_dir, t_lo, t_hi):
    """(stereo int16 samples, idx rows) of the recording covering [t_lo, t_hi]."""
    for idx_path in sorted(glob.glob(os.path.join(audio_dir, '*.idx'))):
        if os.path.getsize(idx_path) == 0:
            continue
        rows = np.loadtxt(idx_path, ndmin=2)
        if len(rows) > 1 and rows[0, 0] <= t_lo and rows[-1, 0] >= t_hi:
            return np.fromfile(idx_path[:-4] + '.s16', dtype=np.int16).reshape(-1, 2), rows
    sys.exit(f'no output audio in {audio_dir} covers {t_lo:.1f}..{t_hi:.1f}')


def align_audio(out, src, src_start, start, seconds):
    """Index (48 kHz frames) of the output sample where video position
    `start` plays. Found by content, not by the recorder's arrival times
    (those are stamped in Python and lag when the GIL is busy); checked
    against a second match near the end for lost or gained samples."""
    mono = out.mean(axis=1)
    mono = mono[:len(mono) // 3 * 3].reshape(-1, 3).mean(axis=1)     # 48k -> 16k
    found = []
    for p in (start, start + seconds - 6):
        y = src[int((p - src_start) * ASR):int((p - src_start + 6) * ASR)]
        c = xcorr(mono, y)
        k = int(np.argmax(np.abs(c)))
        q = abs(c[k]) / (np.linalg.norm(y) * np.linalg.norm(mono[k:k + len(y)]) + 1e-9)
        found.append((k / ASR - (p - start), q))
    (s0, q0), (s1, q1) = found
    print(f'audio: match {q0:.2f}/{q1:.2f}, samples over the segment {1000 * (s1 - s0):+.0f} ms off')
    # Lost samples pull later audio earlier (negative). A positive offset is
    # the TV's own playback stalling, in the capture as well as the output.
    if min(q0, q1) < 0.3 or not -0.03 <= s1 - s0 <= 0.3:
        sys.exit('output audio does not line up with the source (lost samples?)')
    return int(round(s0 * SR))


def align_video(take, f_t, sig_src, src_start, start, t_out):
    """Box time at which video position `start` was captured."""
    # Captured frames resampled onto a FPS grid around the expected time
    # (the output delay is a few seconds at most).
    grid = np.arange(t_out - 8.0, t_out + 22.0, 1.0 / FPS)
    prev, sig = None, []
    for t in grid:
        j = min(max(bisect.bisect_left(f_t, t), 0), len(f_t) - 1)
        im = Image.open(os.path.join(take, 'frames', f'{j:06d}.jpg'))
        im.draft('L', (SIG_W * 4, SIG_H * 4))
        a = np.asarray(im.convert('L').resize((SIG_W, SIG_H), Image.BILINEAR), np.float32)
        sig.append(0.0 if prev is None else float(np.abs(a - prev).mean()))
        prev = a
    sig = np.array(sig[1:])
    # Source change signal over the segment, slid along the capture's.
    i0 = int(round((start - src_start) * FPS))
    y = sig_src[i0:i0 + 20 * FPS]
    y = (y - y.mean()) / (y.std() + 1e-9)
    best = None
    for k in range(0, len(sig) - len(y)):
        x = sig[k:k + len(y)]
        r = float(np.dot((x - x.mean()) / (x.std() + 1e-9), y)) / len(y)
        if best is None or r > best[0]:
            best = (r, k)
    r, k = best
    print(f'video: frame-change match {r:.2f}')
    if r < 0.5:
        sys.exit('captured frames do not line up with the source')
    return float(grid[k + 1])


def main():
    take, audio_dir, out, tag, title, subtitle, src_path = sys.argv[1:8]
    src_start = float(sys.argv[8])
    meta = json.load(open(os.path.join(take, 'take.json')))
    start, seconds = meta['start_s'], meta['seconds']
    fidx = np.loadtxt(os.path.join(take, 'frames.idx'), ndmin=2)
    f_t = fidx[:, 1]

    out_audio, rows = load_output(audio_dir, f_t[0], f_t[-1])
    f0 = align_audio(out_audio, source_audio(src_path), src_start, start, seconds)
    t_out = float(np.interp(f0, rows[:, 1], rows[:, 0]))      # approximate
    t_cap = align_video(take, f_t, source_signature(src_path), src_start, start, t_out - 3.0)
    delay = t_out - t_cap
    print(f'delay about {delay:.2f} s (recorder timestamps)')
    subtitle = subtitle.replace('{delay}', f'{delay:.1f}')

    nframes = int(seconds * FPS)
    audio = out_audio[f0:f0 + int(seconds * SR)]
    per = SR // FPS
    muted = [bool(len(audio[k * per:(k + 1) * per]) and not np.any(audio[k * per:(k + 1) * per]))
             for k in range(nframes)]
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

    f_tag, f_title, f_sub = font(30, True), font(24, True), font(18)
    f_small, f_mute = font(16), font(44, True)
    for k in range(nframes):
        j = min(max(bisect.bisect_left(f_t, t_cap + k / FPS), 0), len(f_t) - 1)
        im = Image.open(os.path.join(take, 'frames', f'{int(fidx[j, 0]):06d}.jpg'))
        im.draft('RGB', (W, H))
        img = im.convert('RGB').resize((W, H), Image.BILINEAR)
        d = ImageDraw.Draw(img, 'RGBA')
        # Title bar.
        d.rectangle([0, 0, W, 74], fill=(0, 0, 0, 185))
        tag_w = d.textlength(tag, font=f_tag) + 28
        d.rounded_rectangle([14, 14, 14 + tag_w, 60], 8, fill=(30, 130, 70, 255))
        d.text((28, 19), tag, font=f_tag, fill='white')
        d.text((tag_w + 34, 12), title, font=f_title, fill='white')
        d.text((tag_w + 34, 44), subtitle, font=f_sub, fill=(210, 210, 210))
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
            bx = W - 300
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
