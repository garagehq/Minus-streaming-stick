#!/usr/bin/env python3
"""Resuming from a static ad must not block the real content.

Observed live (Oct 2026), paused on a Prime Video "Press up to shop" ad:

    22:16:14 [Static] Screen became dynamic - cooldown 1.5s
    22:16:15 VLM: NO-AD p=0.0005            <- show content
    22:16:15 [SAFEGUARD] Stream resumed -- clearing freeze suppression
    22:16:15 AD BLOCKING STARTED (OCR+VLM)  <- from the PAUSED ad's flags
    22:16:16 [Static] Cooldown complete; clearing stale detection state
    22:16:16 Starting blocking (both)       <- overlay now visible on the show
    22:16:21 AD BLOCKING ENDED after 6.1s

Two defects: a block could latch (ad_detected=True) while static
suppression hid it, and the resume cleanup reset the detector flags but not
the latched block, which then surfaced on content until the normal stop
counters caught up.
"""

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import test_asr  # noqa: E402


class TestStaticResumeRelease(unittest.TestCase):

    def _minus(self):
        m = test_asr.TestDecisionEngineASRGate._make_minus(self)
        m._static_resume_release = False
        m.OCR_STOP_THRESHOLD_MAX = 6
        m.flap_escalation = 0
        m.FLAP_REBLOCK_WINDOW_S = 5.0
        m.AD_COUNTDOWN_ENABLED = False
        m._ad_clock_says_playing = lambda now: False
        m._asr_verdict = lambda: 'unknown'
        m.ad_clock_stats = {'holds': 0, 'early_release': 0, 'pause_override': 0}
        m.vlm_min_decisions = 3
        m.OCR_STOP_MIN_SECONDS = 5.0
        m.OCR_STOP_MAX_SECONDS = 9.0
        m.ocr_failure_streak = 0
        m.OCR_DEGRADED_STREAK = 3
        m.OCR_TRIANGULATION_MIN_BLOCK_S = 4.0
        m.OCR_TRIANGULATION_VLM_NOAD_RATIO = 0.80
        m.OCR_TRUSTED_DWELL_FRAMES = 3
        m.FROZEN_EARLY_SECONDS = 30.0
        m._ocr_text_frozen_for = 0.0
        return m

    def test_no_block_starts_while_static_suppressed(self):
        """Paused on an ad: OCR+VLM still see it, but no block may latch."""
        m = self._minus()
        m.static_blocking_suppressed = True
        m.ocr_ad_detected = True
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)

    def test_strong_override_still_blocks(self):
        """The strong-keyword path lifts suppression first, so it still fires."""
        m = self._minus()
        m.static_blocking_suppressed = False  # lifted by the OCR loop
        m.ocr_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_resume_ends_block_held_from_static_period(self):
        """A block active when the screen resumes ends at once, even though
        it is inside its minimum duration and the counters have not moved."""
        m = self._minus()
        m.ocr_ad_detected = True
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)
        m.blocking_start_time = time.time()  # well inside min duration
        m._current_min_blocking_duration = lambda: 5.0
        # Cooldown complete: detector flags cleared, release flagged.
        m.ocr_ad_detected = False
        m.vlm_ad_detected = False
        m._static_resume_release = True
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)
        self.assertFalse(m._static_resume_release)
        m.ad_blocker.hide.assert_called()

    def test_release_overrides_ad_clock(self):
        """A frozen countdown on the paused ad must not hold the block."""
        m = self._minus()
        m.ocr_ad_detected = True
        m._update_blocking_state()
        m._ad_clock_says_playing = lambda now: True
        m._static_resume_release = True
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)

    def test_without_release_block_is_held(self):
        """Control: the same block without a resume keeps its min duration."""
        m = self._minus()
        m.ocr_ad_detected = True
        m._update_blocking_state()
        m.blocking_start_time = time.time()
        m._current_min_blocking_duration = lambda: 5.0
        m.ocr_ad_detected = False
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_cooldown_path_flags_release_when_blocking(self):
        """The OCR loop's cooldown-complete branch must arm the release."""
        src = (ROOT / 'minus.py').read_text()
        seg = src[src.index('[Static] Clearing stale detection state'):]
        seg = seg[:seg.index('self._update_blocking_state()')]
        self.assertIn('self._static_resume_release = True', seg)
        head = src[src.index('had_state = ('):src.index('if had_state:')]
        self.assertIn('self.ad_detected', head)


if __name__ == '__main__':
    unittest.main(verbosity=2)
