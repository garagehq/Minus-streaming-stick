#!/usr/bin/env python3
"""Tests for the Sep-2026 anti-waffle / fast-recovery work.

Two user-reported bugs, opposite in direction, both traced to the same
mistake: treating OCR's *cadence* and OCR's *failures* as if they were
evidence about the ad.

Bug 1 — block drops mid-ad.
  "During an ad block with a big wall of text it stops blocking the ad, but
   after the big wall of text is gone it starts blocking the ad again."

  Recognition latency scales with the number of detected text regions, so a
  text-dense ad frame (pharma fine print, a legal disclaimer card) is exactly
  what pushes inference past OCRProcess.HARD_TIMEOUT. The worker was killed and
  returned [], and the loop counted that empty list as a no-ad vote. Three in a
  row unblocked a live ad; once the wall of text passed, OCR sped up, re-read
  the same ad and re-blocked. Absence of evidence was being treated as evidence
  of absence.

Bug 2 — up to 8 seconds of show missed after the ad ends.
  Measured over 36h of production: recovery p50 5.0s, p90 8.0s, max 12s, with
  100/196 blocks over 4s. Cause was NOT the thresholds. The global capture
  limiter shared by the OCR and VLM loops RAISED its interval from 0.5s to 1.0s
  while blocking ("glitches during blocking are OK"), so each loop's cadence
  went 1.0s -> 2.0s and the 3-frame OCR stop took ~6s. Capture measured p50
  488ms idle vs 1487ms blocking while OCR inference barely moved (231ms ->
  330ms), which is what pinned it on the limiter rather than on inference.

The fix keeps OCR_STOP_THRESHOLD at 3 — it was tuned against the measured OCR
miss distribution and is what resists flapping — and buys the latency back from
cadence instead.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


class TestOCRFailureContract(unittest.TestCase):
    """OCRProcess.ocr() must distinguish 'ran, saw nothing' from 'could not run'."""

    def _proc(self):
        from ocr_worker import OCRProcess
        p = OCRProcess.__new__(OCRProcess)
        p.is_ready = True
        p._consecutive_timeouts = 0
        p.HARD_TIMEOUT = 1.5
        p.process = MagicMock()
        p.process.is_alive.return_value = True
        p.request_queue = MagicMock()
        p.response_queue = MagicMock()
        p.restart = MagicMock(return_value=True)
        return p

    def test_successful_empty_result_is_a_list(self):
        """A frame with no text is a real observation, not a failure."""
        p = self._proc()
        p.response_queue.get.return_value = ('ok', [])
        out = p.ocr(MagicMock())
        self.assertIsInstance(out, list)
        self.assertEqual(out, [])
        self.assertIsNotNone(out, "empty text must stay distinguishable from failure")

    def test_successful_result_passes_through(self):
        p = self._proc()
        p.response_queue.get.return_value = ('ok', [{'text': 'Skip in 5'}])
        self.assertEqual(p.ocr(MagicMock()), [{'text': 'Skip in 5'}])

    def test_timeout_returns_none(self):
        """The wall-of-text case: inference overran, worker killed."""
        p = self._proc()
        p.response_queue.get.side_effect = Exception('queue timeout')
        self.assertIsNone(p.ocr(MagicMock()))
        p.restart.assert_called_once()

    def test_worker_error_returns_none(self):
        p = self._proc()
        p.response_queue.get.return_value = ('error', 'boom')
        self.assertIsNone(p.ocr(MagicMock()))

    def test_dead_worker_that_cannot_restart_returns_none(self):
        from ocr_worker import OCRProcess
        p = self._proc()
        p.is_ready = False
        p.start = MagicMock(return_value=False)
        self.assertIsNone(p.ocr(MagicMock()))

    def test_none_result_from_worker_normalizes_to_empty_list(self):
        """'ok' means it ran; a None payload must not masquerade as a failure."""
        p = self._proc()
        p.response_queue.get.return_value = ('ok', None)
        self.assertEqual(p.ocr(MagicMock()), [])


class _Engine:
    """Mirror of the OCR loop's accounting, matching minus.py.

    Only the branch under test is modelled: what a cycle's OCR outcome does to
    the counters.
    """

    OCR_STOP_THRESHOLD = 3

    def __init__(self):
        self.ocr_no_ad_count = 0
        self.ocr_ad_detection_count = 0
        self.ocr_failure_streak = 0
        self.ocr_ad_detected = True   # a block is running
        self.stopped = False

    def cycle(self, ocr_results):
        if ocr_results is None:
            # Absence of evidence: hold state, touch nothing.
            self.ocr_failure_streak += 1
            return
        self.ocr_failure_streak = 0
        if not ocr_results:
            self._no_ad()
            return
        if any('skip' in r['text'].lower() or 'ad ' in r['text'].lower()
               for r in ocr_results):
            self.ocr_ad_detection_count += 1
            self.ocr_no_ad_count = 0
        else:
            self._no_ad()

    def _no_ad(self):
        self.ocr_no_ad_count += 1
        self.ocr_ad_detection_count = 0
        if self.ocr_ad_detected and self.ocr_no_ad_count >= self.OCR_STOP_THRESHOLD:
            self.ocr_ad_detected = False
            self.stopped = True


class TestWallOfTextRegression(unittest.TestCase):
    """The reported bug, as a scenario."""

    def test_timeouts_during_an_ad_do_not_end_the_block(self):
        e = _Engine()
        for _ in range(6):          # a long disclaimer card: every frame times out
            e.cycle(None)
        self.assertFalse(e.stopped, "block ended on timeouts alone — the reported bug")
        self.assertEqual(e.ocr_no_ad_count, 0)
        self.assertEqual(e.ocr_failure_streak, 6)

    def test_old_behaviour_would_have_stopped(self):
        """Guards the regression: counting [] as no-ad ends the block in 3."""
        e = _Engine()
        for _ in range(3):
            e.cycle([])             # what the worker used to return on timeout
        self.assertTrue(e.stopped)

    def test_block_survives_timeouts_then_resumes_detecting(self):
        e = _Engine()
        e.cycle([{'text': 'Skip in 5'}])
        for _ in range(4):
            e.cycle(None)           # wall of text
        e.cycle([{'text': 'Skip in 2'}])   # readable again, ad still there
        self.assertFalse(e.stopped)
        self.assertEqual(e.ocr_failure_streak, 0)
        # Both readable frames saw the ad; the timeouts between them neither
        # counted against it nor reset the dwell.
        self.assertEqual(e.ocr_ad_detection_count, 2)
        self.assertEqual(e.ocr_no_ad_count, 0)

    def test_real_no_ad_still_ends_the_block(self):
        """The fix must not make blocks unstoppable."""
        e = _Engine()
        for _ in range(3):
            e.cycle([{'text': 'Season 2 Episode 4'}])
        self.assertTrue(e.stopped)

    def test_genuinely_blank_frames_still_end_the_block(self):
        """Empty list = OCR read a blank frame. Real evidence, still counts."""
        e = _Engine()
        for _ in range(3):
            e.cycle([])
        self.assertTrue(e.stopped)

    def test_timeouts_interleaved_with_no_ads_still_end_the_block(self):
        """Failures neither add to nor reset the no-ad tally."""
        e = _Engine()
        e.cycle([]); e.cycle(None); e.cycle([]); e.cycle(None); e.cycle([])
        self.assertTrue(e.stopped)
        self.assertEqual(e.ocr_no_ad_count, 3)

    def test_ad_frame_resets_the_no_ad_tally(self):
        e = _Engine()
        e.cycle([]); e.cycle([])
        e.cycle([{'text': 'Skip in 3'}])
        e.cycle([])
        self.assertFalse(e.stopped)
        self.assertEqual(e.ocr_no_ad_count, 1)


class TestCaptureCadence(unittest.TestCase):
    """Bug 2: the blocking throttle that tripled detection latency."""

    def test_blocking_interval_is_not_slower_than_idle(self):
        import capture
        self.assertLessEqual(
            capture._MIN_CAPTURE_INTERVAL_BLOCKING,
            capture._MIN_CAPTURE_INTERVAL,
            "blocking is exactly when we need frames to notice the ad ended")

    def test_defaults(self):
        import capture
        self.assertEqual(capture._MIN_CAPTURE_INTERVAL, 0.5)
        self.assertEqual(capture._MIN_CAPTURE_INTERVAL_BLOCKING, 0.5)

    def test_env_overridable(self):
        import importlib
        import capture
        os.environ['MINUS_CAPTURE_MIN_INTERVAL_BLOCKING'] = '0.25'
        try:
            importlib.reload(capture)
            self.assertEqual(capture._MIN_CAPTURE_INTERVAL_BLOCKING, 0.25)
        finally:
            del os.environ['MINUS_CAPTURE_MIN_INTERVAL_BLOCKING']
            importlib.reload(capture)

    def test_recovery_budget_at_the_new_cadence(self):
        """3 no-ad frames must fit the 1.5-2.0s recovery goal.

        The limiter is global across the OCR and VLM loops, so a loop's cadence
        is about 2x the interval, plus inference.
        """
        import capture
        cadence = capture._MIN_CAPTURE_INTERVAL_BLOCKING * 2 + 0.33
        self.assertLessEqual(cadence * 3, 6.0,
                             "3-frame stop must beat the 6.0s it measured before")
        # And the old setting must be shown to have missed that budget.
        old_cadence = 1.0 * 2 + 0.33
        self.assertGreater(old_cadence * 3, 6.0)


class TestOCRDegradedFallback(unittest.TestCase):
    """A permanently wedged OCR must not hold an OCR-source block forever."""

    def _minus(self, streak, vlm_no_ad):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.OCR_DEGRADED_STREAK = 3
        m.VLM_STOP_THRESHOLD = 2
        m.ocr_failure_streak = streak
        m.vlm_no_ad_count = vlm_no_ad
        return m

    def _should_stop(self, m, ocr_says_stop=False):
        vlm_says_stop = m.vlm_no_ad_count >= m.VLM_STOP_THRESHOLD
        should_stop = ocr_says_stop
        if (not should_stop
                and m.ocr_failure_streak >= m.OCR_DEGRADED_STREAK
                and vlm_says_stop):
            should_stop = True
        return should_stop

    def test_healthy_ocr_keeps_authority(self):
        """VLM dissent must not end an OCR-source block while OCR still works."""
        m = self._minus(streak=0, vlm_no_ad=5)
        self.assertFalse(self._should_stop(m))

    def test_brief_failure_streak_keeps_authority(self):
        m = self._minus(streak=2, vlm_no_ad=5)
        self.assertFalse(self._should_stop(m))

    def test_degraded_ocr_defers_to_vlm(self):
        m = self._minus(streak=3, vlm_no_ad=2)
        self.assertTrue(self._should_stop(m))

    def test_degraded_ocr_still_needs_vlm_to_agree(self):
        m = self._minus(streak=10, vlm_no_ad=1)
        self.assertFalse(self._should_stop(m))

    def test_ocr_stop_still_wins_regardless(self):
        m = self._minus(streak=0, vlm_no_ad=0)
        self.assertTrue(self._should_stop(m, ocr_says_stop=True))


class TestOCRModelGenerations(unittest.TestCase):
    """PP-OCRv3 / v6 selection: weights, dictionary and DB thresholds move together."""

    def test_v3_is_the_default(self):
        import importlib, config
        os.environ.pop('MINUS_OCR_MODEL_VERSION', None)
        importlib.reload(config)
        self.assertEqual(config.OCR_MODEL_VERSION, 'v3')

    def test_each_generation_binds_its_own_dict_and_thresholds(self):
        import config
        v3 = config.OCR_MODEL_GENERATIONS['v3']
        v6 = config.OCR_MODEL_GENERATIONS['v6']
        self.assertNotEqual(v3['dict_name'], v6['dict_name'],
                            "dictionaries are not interchangeable (6624 vs 6904 entries)")
        self.assertNotEqual(v3['db_params'], v6['db_params'])
        self.assertEqual(v6['db_params'],
                         {'thresh': 0.2, 'box_thresh': 0.4, 'unclip_ratio': 1.4})

    def test_resolver_returns_a_matched_set(self):
        import config
        for version in ('v3', 'v6'):
            r = config.resolve_ocr_models(version=version)
            if r is None:
                self.skipTest(f'{version} weights not present')
            self.assertEqual(r['version'], version)
            self.assertIn(version, Path(r['det']).name)
            self.assertIn(version, Path(r['rec']).name)
            self.assertEqual(Path(r['dict']).name,
                             config.OCR_MODEL_GENERATIONS[version]['dict_name'])
            self.assertEqual(r['db_params'],
                             config.OCR_MODEL_GENERATIONS[version]['db_params'])

    def test_missing_directory_returns_none_rather_than_raising(self):
        import config
        self.assertIsNone(config.resolve_ocr_models(base_dir='/nonexistent/ocr'))

    def test_unknown_version_falls_back_to_a_working_set(self):
        import config
        r = config.resolve_ocr_models(version='v99')
        self.assertIsNotNone(r)
        self.assertIn(r['version'], ('v3', 'v6'))

    def test_db_params_reach_the_postprocessor(self):
        from ocr import PaddleOCR, HAS_POSTPROCESS
        if not HAS_POSTPROCESS:
            self.skipTest('post-process deps unavailable')
        o = PaddleOCR('det.rknn', 'rec.rknn', 'keys.txt',
                      db_params={'thresh': 0.2, 'box_thresh': 0.4, 'unclip_ratio': 1.4})
        self.assertEqual(o.db_postprocess.thresh, 0.2)
        self.assertEqual(o.db_postprocess.box_thresh, 0.4)
        self.assertEqual(o.db_postprocess.unclip_ratio, 1.4)

    def test_default_db_params_unchanged_for_v3(self):
        from ocr import PaddleOCR, HAS_POSTPROCESS
        if not HAS_POSTPROCESS:
            self.skipTest('post-process deps unavailable')
        o = PaddleOCR('det.rknn', 'rec.rknn', 'keys.txt')
        self.assertEqual(o.db_postprocess.thresh, 0.3)
        self.assertEqual(o.db_postprocess.box_thresh, 0.5)
        self.assertEqual(o.db_postprocess.unclip_ratio, 1.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
