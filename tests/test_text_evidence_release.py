#!/usr/bin/env python3
"""Stopping a block needs evidence, and silence is not evidence.

When an ad ends, OCR starts reading whatever replaced it. When OCR merely
loses an ad it is still looking at, it returns nothing. Both used to count
identically toward ending a block, and both then had to wait out the same
wall-clock floor.

Measured over 153 production stops:

    stop taken while OCR returned TEXT   ->  re-blocked within 5s   0 / 9
    stop taken on EMPTY reads            ->  re-blocked within 5s  33 / 56 (59%)

So the floor is worth waiting out when all we have is silence, and is pure
latency once the content is actually visible -- on stops that had text, it had
been available a median of 5s before the block released.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import minus as m  # noqa: E402


def engine(has_text, no_ad_count=3, last_ad_age=0.5):
    o = object.__new__(m.Minus)
    o.ocr_no_ad_count = no_ad_count
    o._ocr_noad_run_has_text = has_text
    o.last_ocr_ad_time = time.time() - last_ad_age
    o.OCR_STOP_THRESHOLD = 3
    o.OCR_STOP_THRESHOLD_MAX = 5
    o.OCR_STOP_MIN_SECONDS = 5.0
    o.OCR_STOP_MAX_SECONDS = 9.0
    o.AD_COUNTDOWN_ENABLED = True
    o.ad_clock_stats = {'holds': 0, 'early_release': 0, 'text_release': 0}
    o.flap_escalation = 0
    clock = MagicMock()
    clock.expired.return_value = False
    o.ad_countdown = clock
    o._ad_clock_says_playing = lambda now=None: False
    return o


class TestTextEvidenceRelease(unittest.TestCase):

    def test_text_run_releases_without_waiting_out_the_floor(self):
        o = engine(has_text=True, last_ad_age=0.5)
        self.assertTrue(o._ocr_says_stop())

    def test_empty_run_still_waits_out_the_floor(self):
        o = engine(has_text=False, last_ad_age=0.5)
        self.assertFalse(o._ocr_says_stop())

    def test_empty_run_releases_once_the_floor_has_passed(self):
        o = engine(has_text=False, last_ad_age=6.0)
        self.assertTrue(o._ocr_says_stop())

    def test_text_evidence_does_not_bypass_the_frame_threshold(self):
        """A single stray text frame mid-ad must not end a block."""
        o = engine(has_text=True, no_ad_count=1, last_ad_age=0.5)
        self.assertFalse(o._ocr_says_stop())

    def test_running_ad_clock_still_outranks_text_evidence(self):
        o = engine(has_text=True, last_ad_age=0.5)
        o._ad_clock_says_playing = lambda now=None: True
        self.assertFalse(o._ocr_says_stop())

    def test_text_release_is_counted(self):
        o = engine(has_text=True, last_ad_age=0.5)
        o._ocr_says_stop()
        self.assertEqual(o.ad_clock_stats['text_release'], 1)

    def test_empty_release_is_not_counted_as_text(self):
        o = engine(has_text=False, last_ad_age=6.0)
        o._ocr_says_stop()
        self.assertEqual(o.ad_clock_stats['text_release'], 0)


class TestRunFlagLifecycle(unittest.TestCase):
    """The flag describes the CURRENT run, so it must clear with the counter."""

    def test_flag_is_declared_false(self):
        o = object.__new__(m.Minus)
        self.assertFalse(getattr(o, '_ocr_noad_run_has_text', False))

    def test_every_counter_reset_also_clears_the_flag(self):
        src = (ROOT / 'minus.py').read_text().split('\n')
        resets = [i for i, l in enumerate(src)
                  if l.strip() == 'self.ocr_no_ad_count = 0']
        self.assertGreater(len(resets), 1)
        for i in resets:
            window = '\n'.join(src[i:i + 5])
            self.assertIn('_ocr_noad_run_has_text = False', window,
                          f'counter reset at line {i+1} leaves the run flag stale')


if __name__ == "__main__":
    unittest.main(verbosity=2)
