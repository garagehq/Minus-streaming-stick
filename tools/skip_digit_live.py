#!/usr/bin/env python3
"""Does re-reading the skip badge at FULL resolution recover the countdown?

The offline probe (skip_digit_probe.py) showed that cropping the badge out of
the 960x540 frame the main OCR pass uses does not work: it "recovers" a digit
on about a third of misses, but those digits are mostly wrong -- values outside
any plausible skip gate, and a run reading 6, 25, 19, 4 across 16 seconds,
which is not a countdown. Upscaling a downscaled image cannot invent detail.

So the question is whether the detail survives in the ORIGINAL frame. The main
pass downscales the 4K capture to 960x540 before OCR (src/capture.py), which is
a 4x linear reduction -- a countdown digit maybe 30px tall on the source is
~7px by the time OCR sees it. This fetches the same frame at full resolution,
locates the badge using the label box found in the downscaled pass (scaled up),
and recognises that region from the ORIGINAL pixels.

Both readings come from ONE fetched frame, so the comparison is like-for-like.

Correctness is judged the same way as offline: a real countdown decrements in
step with the wall clock, and noise does not.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from ocr import PaddleOCR                      # noqa: E402
from config import resolve_ocr_models          # noqa: E402
from ad_countdown import _SKIP_LABEL_RE        # noqa: E402

LONE = re.compile(r'^[0-9]{1,2}$')
SNAP = 'http://localhost:9090/snapshot/raw'

# Learned offline from 40 frames where the digit WAS read, as multiples of the
# label box: the badge sits just to the right, on the same line.
GAP_LO, GAP_HI = 0.02, 0.40
V_PAD = 0.7


def rect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def load_ocr():
    m = resolve_ocr_models()
    ocr = PaddleOCR(m['det'], m['rec'], m['dict'], db_params=m['db_params'])
    if not ocr.load_models():
        raise SystemExit('failed to load OCR models')
    return ocr


def read_region(ocr, img, r, scale=1.0):
    x0, y0, x1, y1 = (int(v) for v in r)
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, 0.0
    crop = img[y0:y1, x0:x1]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    t, c, _ = ocr.recognize(crop)
    return t.strip(), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=600)
    ap.add_argument('--interval', type=float, default=1.0)
    args = ap.parse_args()

    ocr = load_ocr()
    sess = requests.Session()
    seen = base_hits = full_hits = 0
    rows = []

    end = time.time() + args.seconds
    while time.time() < end:
        loop = time.time()
        try:
            resp = sess.get(SNAP, timeout=5)
            full = cv2.imdecode(np.frombuffer(resp.content, np.uint8),
                                cv2.IMREAD_COLOR)
        except Exception:
            time.sleep(args.interval)
            continue
        if full is None:
            time.sleep(args.interval)
            continue

        fh, fw = full.shape[:2]
        small = (cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)
                 if (fw > 960 or fh > 540) else full)
        sx, sy = fw / small.shape[1], fh / small.shape[0]

        res = ocr.ocr(small)
        lab = next((r for r in res
                    if _SKIP_LABEL_RE.search(r['text'].lower())), None)
        if lab is None:
            time.sleep(max(0, args.interval - (time.time() - loop)))
            continue

        seen += 1
        lx0, ly0, lx1, ly1 = rect(lab['box'])
        lw, lh = max(1, lx1 - lx0), max(1, ly1 - ly0)

        # What the pipeline gets today: a digit box the detector proposed.
        base = None
        for r in res:
            if not LONE.match(r['text'].strip()):
                continue
            dx0, dy0, dx1, dy1 = rect(r['box'])
            if dy1 < ly0 - lh or dy0 > ly1 + lh:
                continue
            if dx0 - lx1 > 2.0 * lw or lx0 - dx1 > 2.0 * lw:
                continue
            base = r['text'].strip()
            break
        if base:
            base_hits += 1

        # Same badge, read from the ORIGINAL pixels.
        reg = (lx1 + GAP_LO * lw, ly0 - V_PAD * lh,
               lx1 + GAP_HI * lw, ly1 + V_PAD * lh)
        big = tuple(v * s for v, s in zip(reg, (sx, sy, sx, sy)))
        ftext, fconf = read_region(ocr, full, big, 1.0)
        fhit = ftext if (ftext and LONE.match(ftext)) else None
        if fhit:
            full_hits += 1

        rows.append((time.time(), base, fhit, fconf))
        print(f'  {time.strftime("%H:%M:%S")}  downscaled={base or "-":>3}  '
              f'fullres={fhit or "-":>3} (conf {fconf:.2f})  [{fw}x{fh}]',
              flush=True)
        time.sleep(max(0, args.interval - (time.time() - loop)))

    print(f'\nframes with a "Skip in" label : {seen}')
    if not seen:
        return
    print(f'  digit via current downscale : {base_hits} '
          f'({100*base_hits/seen:.0f}%)')
    print(f'  digit via full-res re-read  : {full_hits} '
          f'({100*full_hits/seen:.0f}%)')

    def consistency(idx):
        pairs = ok = 0
        vals = [(t, v) for t, *rest in
                [(r[0], r[1], r[2]) for r in rows] for v in [rest[idx]] if v]
        for i in range(1, len(vals)):
            dt = vals[i][0] - vals[i - 1][0]
            if not 0 < dt <= 6:
                continue
            a, b = int(vals[i - 1][1]), int(vals[i][1])
            pairs += 1
            if b <= a and abs((a - b) - dt) <= 3:
                ok += 1
        return ok, pairs

    for name, idx in (('downscaled', 0), ('full-res', 1)):
        ok, pairs = consistency(idx)
        pct = f'{100*ok/pairs:.0f}%' if pairs else 'n/a'
        print(f'  {name:11s} clock-consistent: {ok}/{pairs} ({pct})')


if __name__ == '__main__':
    main()
