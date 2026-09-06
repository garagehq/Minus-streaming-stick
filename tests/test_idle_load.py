#!/usr/bin/env python3
"""Tests for the Sep-2026 idle-load work.

Context: a user paused their video for 15 minutes and came back to an 85C
throttling box at ~15fps. Investigation showed pausing the video reduces
Minus's workload by almost nothing:

  - HDMI-RX captures 60fps of 4K whether the picture moves or not, so
    ustreamer encodes (46% CPU) and GStreamer decodes (58%+12% CPU)
    continuously.
  - The scene-change "skip" avoids OCR/VLM *inference* but NOT the capture
    that precedes it: both loops call frame_capture.capture() (4K JPEG over
    HTTP + imdecode + resize) every iteration, then slept only 0.1s (OCR) /
    0.5s (VLM) before doing it again.
  - Measured during static-suppressed periods, OCR still ran on 85% of
    cycles and VLM on 70%.

Covered here:
 1. _idle_backoff_sleep: geometric backoff of the inter-capture sleep once
    the screen has been unchanged for a while, collapsing instantly on any
    change.
 2. HealthMonitor.get_process_memory: per-process memory in /api/health
    (previously system-wide only, so a process climb was invisible).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


class TestIdleBackoff(unittest.TestCase):

    def _minus(self, after=8, max_s=1.5):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.IDLE_BACKOFF_AFTER = after
        m.IDLE_BACKOFF_MAX_S = max_s
        return m

    def test_no_backoff_below_threshold(self):
        """Brief static stretches must behave exactly as before the change."""
        m = self._minus()
        for n in range(0, 9):
            self.assertEqual(m._idle_backoff_sleep(n, 0.1), 0.1)
            self.assertEqual(m._idle_backoff_sleep(n, 0.5), 0.5)

    def test_backoff_grows_geometrically(self):
        m = self._minus()
        self.assertAlmostEqual(m._idle_backoff_sleep(9, 0.1), 0.2)
        self.assertAlmostEqual(m._idle_backoff_sleep(10, 0.1), 0.4)
        self.assertAlmostEqual(m._idle_backoff_sleep(11, 0.1), 0.8)

    def test_backoff_is_capped(self):
        m = self._minus()
        for n in (12, 20, 100, 10_000):
            self.assertLessEqual(m._idle_backoff_sleep(n, 0.1), 1.5)
            self.assertLessEqual(m._idle_backoff_sleep(n, 0.5), 1.5)

    def test_cap_never_exceeded_for_large_base(self):
        """A base sleep already above the cap must not be inflated."""
        m = self._minus()
        self.assertLessEqual(m._idle_backoff_sleep(50, 2.0), 2.0)

    def test_reset_returns_to_base(self):
        """The loops reset skip_count to 0 on any scene change; backoff must
        collapse immediately rather than decaying."""
        m = self._minus()
        self.assertEqual(m._idle_backoff_sleep(30, 0.1), 1.5)
        self.assertEqual(m._idle_backoff_sleep(0, 0.1), 0.1)

    def test_env_overridable(self):
        m = self._minus(after=2, max_s=0.5)
        self.assertEqual(m._idle_backoff_sleep(2, 0.1), 0.1)
        self.assertLessEqual(m._idle_backoff_sleep(20, 0.1), 0.5)

    def test_idle_capture_rate_drops_substantially(self):
        """The point of the change: far fewer 4K captures while frozen.

        Models the OCR loop over a 60s freeze, counting captures. Capture
        itself costs ~0.9s (measured cap= in production logs).
        """
        m = self._minus()
        CAPTURE_S = 0.9

        def captures_in(window_s, backoff):
            t = 0.0
            n = 0
            skips = 0
            while t < window_s:
                n += 1
                skips += 1
                t += CAPTURE_S
                t += m._idle_backoff_sleep(skips, 0.1) if backoff else 0.1
            return n

        before = captures_in(60, backoff=False)
        after = captures_in(60, backoff=True)
        self.assertLess(after, before)
        self.assertLessEqual(after / before, 0.75,
                             f"expected a meaningful drop, got {before}->{after}")

    def test_force_run_cap_still_reachable(self):
        """Backoff must not starve the safety net that catches an ad
        appearing without a scene change."""
        m = self._minus()
        total = sum(m._idle_backoff_sleep(n, 0.1) for n in range(1, 30))
        self.assertLess(total, 60.0, "force-run should still fire within a minute")


class TestProcessMemoryReporting(unittest.TestCase):

    def _hm(self):
        from health import HealthMonitor
        hm = HealthMonitor.__new__(HealthMonitor)
        return hm

    def test_returns_expected_keys(self):
        hm = self._hm()
        hm._get_memory_percent = MagicMock(return_value=33.3)
        out = hm.get_process_memory()
        for k in ('rss_mb', 'pss_anon_mb', 'system_percent'):
            self.assertIn(k, out)

    def test_reads_real_process_values(self):
        hm = self._hm()
        hm._get_memory_percent = MagicMock(return_value=10.0)
        out = hm.get_process_memory()
        self.assertIsInstance(out['rss_mb'], float)
        self.assertGreater(out['rss_mb'], 0)
        self.assertEqual(out['system_percent'], 10.0)

    def test_survives_unreadable_proc(self):
        """Must degrade to None rather than raising into the health check."""
        hm = self._hm()
        hm._get_memory_percent = MagicMock(side_effect=OSError('nope'))
        with patch('builtins.open', side_effect=OSError('nope')):
            out = hm.get_process_memory()
        self.assertIsNone(out['rss_mb'])
        self.assertIsNone(out['pss_anon_mb'])
        self.assertIsNone(out['system_percent'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
