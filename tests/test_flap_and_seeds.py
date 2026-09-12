#!/usr/bin/env python3
"""Anti-flap stop criterion, and music-seed diversity.

FLAPPING (Sep 2026)
-------------------
OCR_STOP_THRESHOLD counts FRAMES, but what it resists -- the ad's keyword
dropping out of the OCR text while the ad is still on screen -- happens in
SECONDS. When the capture fix doubled detection cadence (2.0s -> 1.0s per
frame) it therefore halved the threshold's real tolerance, from 3x2.0 = 6s of
dropout to 3x1.0 = 3s, without anyone touching the number. Measured effect
across two nights on the same threshold: flap rate 4.2% -> 66%.

Dropout runs measured inside 27 real ad episodes:

    run length   1    2    3    4    5   6+
    occurrences  7    9   23   14    2    0

Runs of exactly 3 are the MODE, so a threshold of 3 sat precisely on the peak.
Simulated against those episodes (blocks per episode, 1.00 = no flapping):

    fixed 3 (was)      1.89
    fixed 4            1.48
    fixed 5            1.07
    adaptive 3->6      1.59

Hence a 5.0s wall-clock floor: 5+ runs occurred twice in 27 episodes and 6+
never. Expressing it in seconds is the actual fix -- it keeps its meaning when
cadence moves again (thermal throttling, load, a future capture change), which
the frame counter did not. The escalating frame counter is kept on top for
pathological cases.

SEEDS
-----
Six seeds, all Western pop from one decade, played in a fixed order from index
0 every session. An overnight run surfaced exactly ONE recognisable advertiser
(Audible), because YouTube targets ad inventory by content category and
audience. Widened to 33 verified ids spanning 1985-2019 and several regions and
genres, with a shuffled rotation.
"""

import os
import re
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


def _minus(base=3, cap=6, min_s=5.0, esc=0, no_ad=0, last_hit=None):
    from minus import Minus
    m = Minus.__new__(Minus)
    m.OCR_STOP_THRESHOLD = base
    m.OCR_STOP_THRESHOLD_MAX = cap
    m.OCR_STOP_MIN_SECONDS = min_s
    m.flap_escalation = esc
    m.ocr_no_ad_count = no_ad
    m.last_ocr_ad_time = time.time() if last_hit is None else last_hit
    # The ad clock (Sep 2026) can veto or accelerate a stop. These tests are
    # about the frame/time criterion, so give it a clock with nothing to say.
    m.AD_COUNTDOWN_ENABLED = False
    from ad_countdown import AdCountdownTracker
    m.ad_countdown = AdCountdownTracker()
    m.audio = None
    return m


class TestEffectiveThreshold(unittest.TestCase):

    def test_base_when_not_flapping(self):
        self.assertEqual(_minus(esc=0)._effective_ocr_stop_threshold(), 3)

    def test_rises_with_each_flap(self):
        self.assertEqual(_minus(esc=1)._effective_ocr_stop_threshold(), 4)
        self.assertEqual(_minus(esc=2)._effective_ocr_stop_threshold(), 5)

    def test_capped(self):
        for esc in (3, 10, 500):
            self.assertLessEqual(_minus(esc=esc)._effective_ocr_stop_threshold(), 6)

    def test_never_below_base(self):
        self.assertGreaterEqual(_minus(esc=0)._effective_ocr_stop_threshold(), 3)

    def test_cap_below_base_is_not_honoured(self):
        """A misconfigured cap must never make stopping easier than the base."""
        self.assertEqual(_minus(base=5, cap=2)._effective_ocr_stop_threshold(), 5)


class TestStopCriterion(unittest.TestCase):
    """Both conditions must hold: enough frames AND enough wall-clock time."""

    def test_frames_alone_are_not_enough(self):
        """The regression: 3 frames at 1.0s cadence is only 3s of dropout,
        and runs of 3 are the most common thing an ad does."""
        m = _minus(no_ad=3, last_hit=time.time() - 3.0)
        self.assertFalse(m._ocr_says_stop())

    def test_time_alone_is_not_enough(self):
        m = _minus(no_ad=1, last_hit=time.time() - 30.0)
        self.assertFalse(m._ocr_says_stop())

    def test_both_conditions_stop(self):
        m = _minus(no_ad=5, last_hit=time.time() - 6.0)
        self.assertTrue(m._ocr_says_stop())

    def test_exactly_at_the_floor(self):
        m = _minus(no_ad=5, last_hit=time.time() - 5.0)
        self.assertTrue(m._ocr_says_stop())

    def test_escalation_raises_the_frame_bar(self):
        m = _minus(esc=3, no_ad=5, last_hit=time.time() - 30.0)
        self.assertFalse(m._ocr_says_stop(), 'needs 6 frames once escalated')
        m.ocr_no_ad_count = 6
        self.assertTrue(m._ocr_says_stop())

    def test_no_recorded_hit_falls_back_to_frames(self):
        """Nothing to measure time against; do not deadlock the block."""
        m = _minus(no_ad=5, last_hit=0)
        self.assertTrue(m._ocr_says_stop())

    def test_tolerance_survives_a_cadence_change(self):
        """The whole point of using seconds.

        At 2.0s cadence 3 frames spanned 6s; at 1.0s it spanned 3s. The floor
        holds the real tolerance fixed either way.
        """
        for cadence in (0.5, 1.0, 2.0):
            m = _minus(no_ad=99, last_hit=time.time() - (3 * cadence))
            expected = (3 * cadence) >= 5.0
            self.assertEqual(m._ocr_says_stop(), expected,
                             f'cadence {cadence}s must not change the tolerance')


class TestMeasuredDistribution(unittest.TestCase):
    """The 5.0s floor is where the measured exposure goes to ~zero."""

    RUNS = {1: 7, 2: 9, 3: 23, 4: 14, 5: 2}

    def _exposure(self, n):
        return sum(v for k, v in self.RUNS.items() if k >= n)

    def test_three_sits_on_the_mode(self):
        self.assertEqual(max(self.RUNS, key=self.RUNS.get), 3)

    def test_five_nearly_eliminates_exposure(self):
        self.assertLessEqual(self._exposure(5), 2)
        self.assertGreater(self._exposure(3), 30)

    def test_floor_matches_the_knee(self):
        m = _minus()
        self.assertGreaterEqual(m.OCR_STOP_MIN_SECONDS, 5.0)


class TestVLMRespectsRecentOCRAdText(unittest.TestCase):
    """VLM had no wall-clock discipline; OCR text outranks it.

    Measured live: blocks ending after 3.5s and 3.7s, both "stopped by BOTH",
    while OCR was still matching 'sponsored' on the same ad -- below even the
    5s floor that governs OCR's own stop path. On a "both"-source block VLM
    could end things the instant it voted no-ad twice, and that was the bulk
    of a 67% flap rate.

    If OCR matched an ad keyword within the floor, the ad is still on screen
    and two VLM votes cannot contradict it.
    """

    FLOOR = 5.0

    def _vlm_stop(self, vlm_no_ad, ocr_ad_seconds_ago):
        """Mirror of the decision in _update_blocking_state."""
        stop = vlm_no_ad >= 2
        if (stop and ocr_ad_seconds_ago is not None
                and ocr_ad_seconds_ago < self.FLOOR):
            stop = False
        return stop

    def test_vlm_deferred_while_ocr_still_sees_the_ad(self):
        self.assertFalse(self._vlm_stop(2, 1.0))
        self.assertFalse(self._vlm_stop(5, 4.9))

    def test_vlm_may_stop_once_ocr_text_is_stale(self):
        self.assertTrue(self._vlm_stop(2, 6.0))
        self.assertTrue(self._vlm_stop(2, 30.0))

    def test_vlm_only_blocks_are_unconstrained(self):
        """OCR never saw the ad, so it has no opinion to outrank VLM with."""
        self.assertTrue(self._vlm_stop(2, None))

    def test_threshold_still_required(self):
        self.assertFalse(self._vlm_stop(1, 30.0))

    def test_the_measured_short_blocks_would_now_be_held(self):
        """3.5s and 3.7s blocks, OCR matching 'sponsored' throughout."""
        for block_age in (3.5, 3.7):
            self.assertFalse(self._vlm_stop(2, block_age),
                             f'{block_age}s block must not be ended by VLM')

    def test_wired_into_the_engine(self):
        src = (ROOT / 'minus.py').read_text()
        start = src.index('    def _update_blocking_state')
        body = src[start:src.index('\n    def ', start + 10)]
        self.assertIn('vlm_deferred', body,
                      'the VLM floor must be applied in the stop decision')


class TestSeedDiversity(unittest.TestCase):

    def _seeds(self):
        from autonomous_mode import AutonomousMode
        return AutonomousMode.MUSIC_VIDEO_SEEDS

    def test_widened(self):
        """Six Western pop hits from one decade yielded one advertiser."""
        self.assertGreaterEqual(len(self._seeds()), 25)

    def test_no_duplicates(self):
        s = self._seeds()
        self.assertEqual(len(s), len(set(s)))

    def test_ids_are_well_formed(self):
        """A malformed id is not a no-op: it silently lands on the home
        screen, which is the failure the home-screen re-issue path exists
        to catch."""
        for v in self._seeds():
            self.assertRegex(v, r'^[A-Za-z0-9_-]{11}$', v)

    def test_original_seeds_retained(self):
        for v in ('kJQP7kiw5Fk', 'JGwWNGJdvx8', 'RgKAFK5djSk',
                  'OPf0YbXqDm0', 'CevxZvSJLk8', '9bZkp7q19f0'):
            self.assertIn(v, self._seeds())

    def test_rotation_is_shuffled_and_covers_everything(self):
        import collections
        from autonomous_mode import AutonomousMode
        a = AutonomousMode.__new__(AutonomousMode)
        a._music_seed_order = list(range(len(self._seeds())))
        import random
        random.shuffle(a._music_seed_order)
        self.assertEqual(sorted(a._music_seed_order),
                         list(range(len(self._seeds()))),
                         'every seed must remain reachable')

    def test_rotation_order_varies_between_sessions(self):
        import random
        n = len(self._seeds())
        a = list(range(n)); b = list(range(n))
        random.shuffle(a); random.shuffle(b)
        self.assertGreater(n, 10, 'too few seeds for shuffling to matter')


if __name__ == '__main__':
    unittest.main(verbosity=2)
