#!/usr/bin/env python3
"""Audit the normalization baked into each shipped OCR model.

PaddleOCR uses DIFFERENT input normalization for detection and recognition, and
in RKNN that choice is frozen into the model at conversion time (`rknn.config`
mean_values/std_values). Get it wrong and there is no runtime knob to fix it:
`inputs_pass_through=1` segfaults the RK3588 runtime, and pre-compensating the
uint8 input costs ~55% of the input precision (mean +/- std spans only ~117 of
256 levels).

Reference (PaddleOCR):
  detection    NormalizeImage with ImageNet statistics
                 mean [123.675, 116.28, 103.53] std [58.395, 57.12, 57.375]
  recognition  RecResizeImg.resize_norm_img
                 resized_image = resized_image / 255; -= 0.5; /= 0.5
                 i.e. mean [127.5]*3 std [127.5]*3

Known deviation, Sep 2026: the v3 recognition model was converted with the
DETECTION (ImageNet) statistics. It is flagged below rather than asserted
correct, so the deviation stays visible and a corrected drop-in is noticed.

Measured impact before deciding not to hot-patch it, 40 real ad frames from
screenshots/ads/, production keyword matcher, frames flagged as an ad:

    v3 as shipped (ImageNet)          39/40
    v3 grey-padded                    37/40
    v3 pre-compensated to 127.5       38/40
    v6 as shipped (127.5, correct)    38/40

So correcting it on the host measured no better. Caveat that keeps this open
rather than closed: screenshots/ads/ only contains frames the CURRENT pipeline
already detected, so the corpus is biased toward the shipped settings. The
honest reading is "no evidence of harm", not "the deviation is fine". Settling
it needs a re-converted v3 rec at full precision, which needs the converter
toolchain (torch<=2.2, onnx==1.16.1, protobuf==3.20.3, onnxoptimizer -- the
last has no aarch64 wheel), so it belongs on the machine that built v6.
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

MODELS = ROOT / 'models' / 'paddleocr'

IMAGENET = ([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
PADDLE_REC = ([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])

# Models whose baked normalization is known to deviate from the reference.
# Remove an entry when a corrected model is dropped in; the test then enforces
# the correct value for it.
KNOWN_DEVIATIONS = {
    'ppocrv3_rec': 'converted with detection (ImageNet) stats instead of rec 127.5',
}


def baked_norm(path):
    """Read the mean/std the RKNN runtime will apply, from model metadata."""
    blob = path.read_bytes()
    m = re.search(rb"'mean': \[([^\]]+)\], 'std': \[([^\]]+)\]", blob)
    if not m:
        return None
    f = lambda b: [float(x) for x in b.decode().split(',')]
    return f(m.group(1)), f(m.group(2))


def find(prefix):
    hits = sorted(MODELS.glob(f'{prefix}_*.rknn'))
    return hits[0] if hits else None


class TestBakedNormalization(unittest.TestCase):

    def _check(self, prefix, expected, label):
        path = find(prefix)
        if path is None:
            self.skipTest(f'{prefix} not present')
        got = baked_norm(path)
        self.assertIsNotNone(got, f'no normalization metadata in {path.name}')
        if prefix in KNOWN_DEVIATIONS:
            if got != tuple(map(list, expected)):
                self.skipTest(
                    f'{path.name}: KNOWN DEVIATION ({KNOWN_DEVIATIONS[prefix]}) '
                    f'-- expected {label} {expected}, has {got}')
            self.fail(
                f'{path.name} now matches the reference. Remove it from '
                f'KNOWN_DEVIATIONS so the correct value is enforced from here on.')
        self.assertEqual(got, tuple(map(list, expected)),
                         f'{path.name} must use {label} normalization')

    def test_v3_det_uses_imagenet(self):
        self._check('ppocrv3_det', IMAGENET, 'ImageNet (detection)')

    def test_v6_det_uses_imagenet(self):
        self._check('ppocrv6_det', IMAGENET, 'ImageNet (detection)')

    def test_v6_rec_uses_paddle_rec_stats(self):
        self._check('ppocrv6_rec', PADDLE_REC, 'PaddleOCR rec (127.5)')

    def test_v3_rec_deviation_is_tracked(self):
        """Fails once v3 rec is re-converted, prompting the flag to be cleared."""
        self._check('ppocrv3_rec', PADDLE_REC, 'PaddleOCR rec (127.5)')

    def test_det_and_rec_conventions_actually_differ(self):
        """Guards the whole point: these are not the same numbers."""
        self.assertNotEqual(IMAGENET, PADDLE_REC)


class TestNoLosslessRuntimeWorkaround(unittest.TestCase):
    """Documents why this cannot simply be corrected at runtime."""

    def test_precompensation_loses_input_precision(self):
        """mean +/- std spans well under the full uint8 range."""
        mean, std = IMAGENET
        lo = min(m - s for m, s in zip(mean, std))
        hi = max(m + s for m, s in zip(mean, std))
        self.assertGreater(lo, 0)
        self.assertLess(hi, 255)
        levels = hi - lo
        self.assertLess(levels / 255.0, 0.75,
                        'pre-compensation would be lossless, so prefer it')


class TestRecPaddingConvention(unittest.TestCase):
    """Second deviation found in the same audit, affecting BOTH generations.

    PaddleOCR pads AFTER normalizing, with 0.0 in normalized space, which is
    pixel 127.5 (mid grey). src/ocr.py pads BEFORE normalizing with pixel 0
    (black). For short ad UI text the padding dominates the model input --
    a 60x22 "Skip" crop is 59% padding at 48x320.

    Measured neutral on the product metric (v6: 38/40 ad frames either way), so
    it is documented rather than changed. Same corpus-bias caveat as above.
    """

    def test_padding_share_of_a_short_crop_is_large(self):
        w, h, target_w, target_h = 60, 22, 320, 48
        used = min(int(w * target_h / h), target_w)
        self.assertGreater((target_w - used) / target_w, 0.5)

    def test_normalized_zero_corresponds_to_mid_grey(self):
        """Under the rec convention, PaddleOCR's pad value is 127.5, not 0."""
        mean, std = PADDLE_REC
        pad_pixel = mean[0] + std[0] * 0.0
        self.assertEqual(pad_pixel, 127.5)
        black_normalized = (0.0 - mean[0]) / std[0]
        self.assertEqual(black_normalized, -1.0)

    def test_current_behaviour_is_documented(self):
        """src/ocr.py still pads with 0; change this test when that changes."""
        src = (ROOT / 'src' / 'ocr.py').read_text()
        self.assertIn('constant_values=0', src,
                      'padding changed -- update this audit and re-measure')


if __name__ == '__main__':
    unittest.main(verbosity=2)
