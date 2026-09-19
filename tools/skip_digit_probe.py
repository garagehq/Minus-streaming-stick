#!/usr/bin/env python3
"""Measure whether a targeted re-OCR recovers the ad's skip countdown digit.

The ad clock can only count down a real number when OCR actually returns one.
Measured live, it did not: of 150 frames showing a "Skip in" label, only 12%
carried a recoverable digit, so the clock ran on a flat assumption the rest of
the time. The digit is plainly on the TV -- the question this answers is
whether the main OCR pass is failing to DETECT it (a box is never proposed) or
failing to RECOGNISE it (the box is proposed and decoded wrong).

That distinction decides the fix. A detection failure is repairable cheaply: we
already know where the label is, so we can crop the badge next to it and force
recognition on that region without waiting for the detector to propose it. A
recognition failure is not repairable this way and would need better input
pixels (the main pass runs on a 960x540 downscale of a 4K frame).

Stage 1 learns the badge geometry from frames where the digit WAS read, so the
crop is positioned from data rather than a guess. Stage 2 applies it to frames
where the digit was missed and reports the recovery rate.

NOTE ON THE CORPUS: screenshots/ads/ is saved at 960x540, already downscaled.
So this measures the targeted-crop effect ALONE and cannot measure the benefit
of re-reading at full 4K resolution -- that needs a live capture path.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from ocr import PaddleOCR                      # noqa: E402
from config import resolve_ocr_models          # noqa: E402
from ad_countdown import _SKIP_LABEL_RE        # noqa: E402

LONE_DIGITS = re.compile(r'^[0-9]{1,2}$')


def box_rect(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def find_label(results):
    for r in results:
        if _SKIP_LABEL_RE.search(r['text'].lower()):
            return r
    return None


def find_lone_digit(results, label_rect=None):
    """A standalone 1-2 digit element.

    When a label rect is given, only digits actually ADJACENT to it count. Any
    number anywhere in the frame is not the countdown -- ad creatives are full
    of prices, ratings and runtimes, and counting those inflates the hit rate
    and poisons the learned geometry (a first pass put the 10th-percentile gap
    at -10.75 label widths, i.e. the other side of the screen).
    """
    best = None
    for r in results:
        if not LONE_DIGITS.match(r['text'].strip()):
            continue
        if label_rect is None:
            return r
        lx0, ly0, lx1, ly1 = label_rect
        dx0, dy0, dx1, dy1 = box_rect(r['box'])
        lw = max(1, lx1 - lx0)
        lh = max(1, ly1 - ly0)
        # Same text line, within a couple of label widths either side.
        if dy1 < ly0 - lh or dy0 > ly1 + lh:
            continue
        gap = dx0 - lx1 if dx0 >= lx1 else lx0 - dx1
        if gap > 2.0 * lw:
            continue
        if best is None or gap < best[0]:
            best = (gap, r)
    return best[1] if best else None


def load_ocr():
    m = resolve_ocr_models()
    if not m:
        raise SystemExit('OCR models not found')
    ocr = PaddleOCR(m['det'], m['rec'], m['dict'], db_params=m['db_params'])
    if not ocr.load_models():
        raise SystemExit('failed to load OCR models')
    return ocr


def recognise_region(ocr, img, rect, scale):
    """Force recognition on a region the detector may never have proposed."""
    x0, y0, x1, y1 = rect
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, 0.0
    crop = img[y0:y1, x0:x1]
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    text, conf, _ = ocr.recognize(crop)
    return text.strip(), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--scale', type=float, default=4.0)
    ap.add_argument('--pad', type=float, default=0.6,
                    help='badge search width as a multiple of label width')
    args = ap.parse_args()

    files = sorted(ROOT.glob('screenshots/ads/*.png'))[-args.limit:]
    ocr = load_ocr()

    have_label = 0
    with_digit = []      # (label_rect, digit_rect) for geometry
    missing = []         # (path, img, label_rect)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        res = ocr.ocr(img)
        lab = find_label(res)
        if lab is None:
            continue
        have_label += 1
        dig = find_lone_digit(res, box_rect(lab['box']))
        if dig is not None:
            with_digit.append((box_rect(lab['box']), box_rect(dig['box']),
                               dig['text'].strip()))
        else:
            missing.append((f, img, box_rect(lab['box'])))

    print(f'frames scanned                : {len(files)}')
    print(f'frames showing a "Skip in"    : {have_label}')
    print(f'  digit already read          : {len(with_digit)}'
          f'  ({100*len(with_digit)/max(have_label,1):.0f}%)')
    print(f'  digit MISSING               : {len(missing)}'
          f'  ({100*len(missing)/max(have_label,1):.0f}%)')

    if not with_digit:
        print('\nno geometry samples; cannot position a crop')
        return

    # --- stage 1: learn where the badge sits relative to the label ---
    dxs, dys, ws, hs = [], [], [], []
    for (lx0, ly0, lx1, ly1), (dx0, dy0, dx1, dy1), _ in with_digit:
        lw = max(1, lx1 - lx0)
        lh = max(1, ly1 - ly0)
        dxs.append((dx0 - lx1) / lw)      # gap after label, in label widths
        dys.append((dy0 - ly0) / lh)      # vertical offset, in label heights
        ws.append((dx1 - dx0) / lw)
        hs.append((dy1 - dy0) / lh)
    med = lambda a: float(np.median(a))
    print(f'\nbadge geometry from {len(with_digit)} samples (label-relative):')
    print(f'  gap after label  median {med(dxs):+.2f}  '
          f'p10 {np.percentile(dxs,10):+.2f}  p90 {np.percentile(dxs,90):+.2f}')
    print(f'  vertical offset  median {med(dys):+.2f}')
    print(f'  digit width      median {med(ws):.2f}   height {med(hs):.2f}')

    # --- stage 2: force recognition where the digit was missed ---
    gap_lo = float(np.percentile(dxs, 5))
    gap_hi = float(np.percentile(dxs, 95))
    recovered = 0
    samples = []
    all_recovered = []
    for f, img, (lx0, ly0, lx1, ly1) in missing:
        lw = max(1, lx1 - lx0)
        lh = max(1, ly1 - ly0)
        pad_y = lh * 0.6
        best = None
        # Search both sides: element order in the main pass is detection
        # order, so the badge is not reliably on one side.
        for lo, hi in ((lx1 + gap_lo * lw, lx1 + (gap_hi + args.pad) * lw),
                       (lx0 - (gap_hi + args.pad) * lw, lx0 - gap_lo * lw)):
            rect = (lo, ly0 - pad_y, hi, ly1 + pad_y)
            text, conf = recognise_region(ocr, img, rect, args.scale)
            if text and LONE_DIGITS.match(text) and (best is None or conf > best[1]):
                best = (text, conf)
        if best:
            recovered += 1
            all_recovered.append((f.name, best[0], best[1]))
            if len(samples) < 10:
                samples.append((f.name, best[0], best[1]))

    # --- stage 3: are the recovered digits actually RIGHT? ---
    # No ground truth exists, but a countdown has a property noise does not:
    # consecutive frames of the same ad must DECREASE, roughly in step with
    # the wall clock. Frames are named with their capture time, so a run of
    # recoveries from one ad can be checked for that shape. A recovered value
    # that jumps around is a misread however confident the model was.
    def stamp(name):
        m = re.search(r'_(\d{8})_(\d{6})_', name)
        if not m:
            return None
        d, t = m.group(1), m.group(2)
        return (int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:]), d)

    print(f'\ntargeted re-OCR on the {len(missing)} misses:')
    print(f'  recovered a digit           : {recovered}'
          f'  ({100*recovered/max(len(missing),1):.0f}%)')
    total_now = len(with_digit) + recovered
    print(f'  digit availability overall  : '
          f'{100*len(with_digit)/max(have_label,1):.0f}% -> '
          f'{100*total_now/max(have_label,1):.0f}%')
    if samples:
        print('\n  samples:')
        for n, t, c in samples:
            print(f'    {n}  -> {t!r} (conf {c:.2f})')

    if all_recovered:
        seq = []
        for name, val, conf in all_recovered:
            st = stamp(name)
            if st:
                seq.append((st[1], st[0], int(val), conf))
        seq.sort()
        pairs = ok_pairs = 0
        for i in range(1, len(seq)):
            d0, t0, v0, _ = seq[i - 1]
            d1, t1, v1, _ = seq[i]
            dt = t1 - t0
            if d0 != d1 or not (0 < dt <= 6):
                continue          # not the same ad moment
            pairs += 1
            # Should have counted DOWN by roughly the elapsed seconds.
            if v1 <= v0 and abs((v0 - v1) - dt) <= 3:
                ok_pairs += 1
        print(f'\n  temporal check on consecutive recoveries:')
        print(f'    comparable pairs          : {pairs}')
        if pairs:
            print(f'    consistent with a clock   : {ok_pairs} '
                  f'({100*ok_pairs/pairs:.0f}%)')
        plaus = sum(1 for _, v, _ in all_recovered if 0 < int(v) <= 60)
        print(f'    within a plausible gate   : {plaus}/{len(all_recovered)}'
              f'  (<=60s)')
        for floor in (0.3, 0.5):
            keep = [x for x in all_recovered if x[2] >= floor]
            good = sum(1 for _, v, _ in keep if 0 < int(v) <= 60)
            print(f'    conf >= {floor}: kept {len(keep):3d}, '
                  f'plausible {good}')


if __name__ == '__main__':
    main()
