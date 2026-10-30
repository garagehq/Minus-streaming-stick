#!/usr/bin/env python3
"""A skipped ad's block ends through the normal end path.

minus, 2026-10-30: "[SKIP] Forcing unblock after skip" cleared the flags and
blocking_source itself. The VLM vote history still held the ad's "yes"
votes, so the next decision logged "VLM detected ad ... No (p=0.0004)", and
the block's end was logged "stopped by unknown" (3x in 25 min).
"""

import logging
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import test_static_resume_release as tsr  # noqa: E402


class TestSkipRelease(unittest.TestCase):

    def _blocking(self):
        m = tsr.TestStaticResumeRelease._minus(self)
        m._skip_release = False
        m.ocr_ad_detected = True
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)
        m.blocking_start_time = time.time()          # inside the min duration
        m._current_min_blocking_duration = lambda: 5.0
        m.vlm_decision_history.extend([True] * 6)    # the ad's own votes
        return m

    def _skip(self, m):
        # What _unblock_after_skip leaves behind.
        m.ocr_ad_detected = False
        m.vlm_ad_detected = False
        m._skip_release = True

    def test_skip_ends_the_block_and_clears_vlm_history(self):
        m = self._blocking()
        self._skip(m)
        with self.assertLogs(level=logging.WARNING) as logs:
            m._update_blocking_state()
        self.assertFalse(m.ad_detected)
        self.assertIsNone(m.blocking_source)
        self.assertEqual(len(m.vlm_decision_history), 0)
        self.assertFalse(m._skip_release)
        self.assertTrue(any('stopped by SKIP' in l for l in logs.output), logs.output)

    def test_skip_overrides_the_ad_clock(self):
        m = self._blocking()
        m._ad_clock_says_playing = lambda now: True
        self._skip(m)
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)

    def test_skip_thread_keeps_blocking_source_for_the_tick(self):
        src = (ROOT / 'minus.py').read_text()
        body = src[src.index('def _unblock_after_skip'):src.index('threading.Thread(target=_unblock_after_skip')]
        self.assertIn('self._skip_release = True', body)
        self.assertNotIn('self.blocking_source = None', body)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestSkipVerified(unittest.TestCase):
    """minus, 2026-10-30 04:49: the skip press landed while "Skip in 1" was
    still counting, the forced unblock ran anyway, and the ad's sound played
    for ~2 s until it re-blocked. The unblock now waits for an OCR read taken
    after the press and keeps the block if the ad is still there."""

    def test_still_showing(self):
        import minus as M
        f = M.Minus._ad_still_showing_after_skip
        self.assertTrue(f(['Sponsored • Join Us Today', '90+', 'Skip I']))
        self.assertTrue(f(['Skip in', '3']))
        self.assertFalse(f(['CLEAN BANDIT']))
        self.assertFalse(f([]))
        self.assertFalse(f(None))

    def test_unblock_thread_checks_before_releasing(self):
        src = (ROOT / 'minus.py').read_text()
        body = src[src.index('def _unblock_after_skip'):src.index('threading.Thread(target=_unblock_after_skip')]
        check = body.index('_ad_still_showing_after_skip')
        self.assertLess(check, body.index('self._skip_release = True'))
        self.assertLess(check, body.index('self.audio.unmute()'))
        self.assertIn('last_ocr_texts_time', body)

    def test_retry_flag_resets_when_the_block_ends(self):
        m = TestSkipRelease._blocking(self)
        m._skip_retry_used = True
        m.ocr_ad_detected = False
        m.vlm_ad_detected = False
        m._skip_release = True
        m._update_blocking_state()
        self.assertFalse(m._skip_retry_used)
