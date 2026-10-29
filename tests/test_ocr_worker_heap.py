#!/usr/bin/env python3
"""The OCR worker must not grow without bound.

minus, 2026-10-29 overnight soak: the OCR worker's RSS grew ~110 MB/h. Most of
it was freed-but-unreturned glibc heap from the RKNN inference calls
(malloc_trim(0) on the live worker gave back 109 MB); ~30 MB/h survived the
trim. The worker now trims every TRIM_EVERY requests, before that reply, and
OCRProcess replaces it between requests if its RSS is still over RSS_CAP_MB.
"""

import queue
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import ocr_worker as ow  # noqa: E402


class FakeOCR:
    def __init__(self, *a, **k):
        self.released = False

    def load_models(self):
        return True

    def ocr(self, img):
        return [{'text': 'x', 'confidence': 1.0, 'box': []}]

    def release(self):
        self.released = True


class TestWorkerTrims(unittest.TestCase):

    def run_worker(self, n_requests):
        events = []
        requests = queue.Queue()
        for _ in range(n_requests):
            requests.put((b'frame', 'ocr'))
        requests.put(None)
        responses = MagicMock()
        responses.put.side_effect = lambda r: events.append('reply')
        fake_ocr = types.ModuleType('ocr')
        fake_ocr.PaddleOCR = FakeOCR
        fake_config = types.ModuleType('config')
        fake_config.OCR_MODEL_DIR = '/nonexistent'
        fake_config.resolve_ocr_models = lambda: {
            'version': 'test', 'base_dir': '/nonexistent', 'det': 'd', 'rec': 'r',
            'dict': 'k', 'db_params': {}}
        with patch.dict(sys.modules, {'ocr': fake_ocr, 'config': fake_config}), \
                patch.object(ow, '_trim_heap', side_effect=lambda: events.append('trim')), \
                patch('signal.signal'):
            ow._ocr_worker_main(requests, responses, threading.Event(), threading.Event())
        return events

    def test_trims_every_trim_every_requests_before_the_reply(self):
        events = self.run_worker(2 * ow.TRIM_EVERY + 3)
        self.assertEqual(events.count('trim'), 2)
        self.assertEqual(events.count('reply'), 2 * ow.TRIM_EVERY + 3)
        replies_before_trim = [events[:i].count('reply') for i, e in enumerate(events) if e == 'trim']
        # The trim for request N runs before reply N is sent.
        self.assertEqual(replies_before_trim, [ow.TRIM_EVERY - 1, 2 * ow.TRIM_EVERY - 1])

    def test_no_trim_below_trim_every(self):
        self.assertNotIn('trim', self.run_worker(ow.TRIM_EVERY - 1))

    def test_trim_heap_and_rss_never_raise(self):
        ow._trim_heap()
        self.assertIsNone(ow._rss_mb(-1))
        self.assertGreater(ow._rss_mb('self'), 0)


class TestParentReplacesOversizedWorker(unittest.TestCase):

    def process(self, rss):
        p = ow.OCRProcess()
        p.is_ready = True
        p.process = MagicMock(pid=4242)
        p.process.is_alive.return_value = True
        p.request_queue = MagicMock()
        p.response_queue = MagicMock()
        p.response_queue.get.return_value = ('ok', [])
        p.kill = MagicMock()
        p.start = MagicMock(return_value=True)
        self.rss = patch.object(ow, '_rss_mb', return_value=rss)
        self.rss.start()
        self.addCleanup(self.rss.stop)
        return p

    def test_restarts_once_over_the_cap_without_counting_a_restart(self):
        p = self.process(ow.RSS_CAP_MB + 200)
        for _ in range(ow.TRIM_EVERY - 1):
            self.assertEqual(p.ocr(b'f'), [])
        p.kill.assert_not_called()
        self.assertEqual(p.ocr(b'f'), [])
        p.kill.assert_called_once()
        p.start.assert_called_once()
        self.assertEqual(p.restart_count, 0)
        self.assertEqual(p._consecutive_timeouts, 0)

    def test_leaves_a_worker_under_the_cap_alone(self):
        p = self.process(ow.RSS_CAP_MB - 100)
        for _ in range(3 * ow.TRIM_EVERY):
            p.ocr(b'f')
        p.kill.assert_not_called()
        p.start.assert_not_called()

    def test_unreadable_rss_does_nothing(self):
        p = self.process(None)
        for _ in range(ow.TRIM_EVERY):
            p.ocr(b'f')
        p.kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
