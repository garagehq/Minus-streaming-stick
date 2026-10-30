"""Faster ad detection: cheaper snapshot decode, faster capture cadence,
and 'Send to phone' (YouTube TV ad-only button) blocking on the first frame."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


def _jpeg(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (w // 2, h // 2), (255, 255, 255), -1)
    ok, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


class TestSnapshotDecode(unittest.TestCase):

    def test_jpeg_size_reads_the_header(self):
        import capture
        self.assertEqual(capture.jpeg_size(_jpeg(3840, 2160)), (3840, 2160))
        self.assertEqual(capture.jpeg_size(_jpeg(1280, 720)), (1280, 720))
        self.assertIsNone(capture.jpeg_size(b'not a jpeg'))

    def test_4k_and_1080p_decode_straight_to_960x540(self):
        import capture
        for w, h in ((3840, 2160), (1920, 1080)):
            self.assertEqual(capture.decode_for_detection(_jpeg(w, h)).shape, (540, 960, 3))

    def test_other_sizes_decode_in_full(self):
        import capture
        self.assertEqual(capture.decode_for_detection(_jpeg(1280, 720)).shape, (720, 1280, 3))

    def test_reduced_decode_matches_full_decode_plus_resize(self):
        import capture
        data = _jpeg(3840, 2160)
        full = cv2.resize(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
                          (960, 540), interpolation=cv2.INTER_AREA)
        red = capture.decode_for_detection(data)
        self.assertLess(np.abs(full.astype(int) - red.astype(int)).mean(), 2.0)

    def test_garbage_returns_none(self):
        import capture
        self.assertIsNone(capture.decode_for_detection(b'garbage'))


class TestSendToPhone(unittest.TestCase):

    def _kw(self, texts):
        from ocr import PaddleOCR
        r = PaddleOCR.check_ad_keywords([{'text': t} for t in texts])
        return [k for k, _ in (r[1] if isinstance(r, tuple) else r)]

    def test_matches_with_and_without_spaces(self):
        self.assertIn('send to phone', self._kw(['Send to phone']))
        self.assertIn('send to phone', self._kw(['Sendto phone']))

    def test_not_on_unrelated_phone_text(self):
        self.assertEqual(self._kw(['Phone settings', 'Send feedback']), [])

    def test_is_definitive_so_blocks_on_first_frame(self):
        from minus import Minus
        src = (ROOT / 'minus.py').read_text()
        block = src[src.index('self.STRONG_AD_KEYWORD_NAMES = frozenset({'):]
        block = block[:block.index('})')]
        self.assertIn("'send to phone'", block)
        self.assertTrue(hasattr(Minus, '__init__'))


class TestDisplayLatencyInAvSync(unittest.TestCase):
    """Lip sync counts the display pipeline's measured latency."""

    def test_measured_pipeline_replaces_the_constant(self):
        import av_sync
        base = av_sync.video_latency_ms(40.0, 60.0)
        meas = av_sync.video_latency_ms(40.0, 60.0, 30.0)
        self.assertAlmostEqual(base, 1000 / 60 + 40 + av_sync.DISPLAY_SIDE_MS)
        self.assertAlmostEqual(meas, 1000 / 60 + 40 + 30 + av_sync.SCANOUT_MS)

    def test_blocker_reports_median_only_when_running(self):
        import time
        from collections import deque
        from ad_blocker import DRMAdBlocker
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        b._display_latency_samples = deque([30.0] * 25 + [80.0], maxlen=120)
        b._display_latency_time = time.time()
        b.parked = False
        self.assertEqual(b.get_display_latency_ms(), 30.0)
        b.parked = True
        self.assertIsNone(b.get_display_latency_ms())
        b.parked = False
        b._display_latency_time = time.time() - 60
        self.assertIsNone(b.get_display_latency_ms())
        b._display_latency_time = time.time()
        b._display_latency_samples = deque([30.0] * 5)
        self.assertIsNone(b.get_display_latency_ms())


if __name__ == '__main__':
    unittest.main()
