#!/usr/bin/env python3
"""Ad countdown timestamps as a duration signal.

Ads state their own length. "Ad 0:15", "Skip in 5", "Ad 10" are on screen
every frame and OCR already reads them; the decision engine kept only the
boolean "a keyword matched" and discarded the number.

That number is the strongest evidence available about whether the ad is
still running. The consecutive-miss counters were inferring it indirectly:
a keyword that drops out for three frames looks exactly like the end of an
ad, which is why flapping was so hard to tune. At 0:12-remaining a missed
keyword is obviously a misread.

It is advisory, never a command:

  * corroboration required -- one reading cannot hold a block, because a
    misread digit is common and a misread that lands on a plausible
    successor of the previous value is not;
  * it goes stale, so an old number cannot outlive its ad;
  * "skip in N" bounds the start of the skip window, not the end of the ad,
    so it may hold a block but may never end one;
  * a FROZEN clock means a pause. Holding the overlay across a pause is the
    specific failure the user called out, so a frozen clock never holds and
    silent audio independently overrides it.

Every existing safeguard (max duration, frozen-stream, VLM dissent) still
runs on top.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from ad_countdown import (AdCountdownTracker, parse_ad_remaining,  # noqa: E402
                          parse_skip_in)


class TestParsing(unittest.TestCase):

    def test_plain_timestamp(self):
        self.assertEqual(parse_ad_remaining(['Ad 0:15']), 15)
        self.assertEqual(parse_ad_remaining(['Ad0:30']), 30)
        self.assertEqual(parse_ad_remaining(['Ad 1:20']), 80)

    def test_ocr_misreads(self):
        """The exact shapes CLAUDE.md documents, plus the user's '0:1s'."""
        for text, expect in (('Ado:15', 15), ('Ad0:1s', 15), ('Ado:3o', 30),
                             ('Adl:lo', 70), ('Ad0;30', 30), ('Ad0.30', 30)):
            self.assertEqual(parse_ad_remaining([text]), expect, text)

    def test_bare_seconds_countdown(self):
        """Netflix style 'Ad 10'."""
        self.assertEqual(parse_ad_remaining(['Ad 10']), 10)
        self.assertEqual(parse_ad_remaining(['Ad5']), 5)

    def test_pod_counter_is_not_a_duration(self):
        """'Ad 1 of 2' is a position in the pod, not one second remaining."""
        self.assertIsNone(parse_ad_remaining(['Ad 1 of 2']))
        self.assertIsNone(parse_ad_remaining(['Ad2of2']))

    def test_pod_counter_alongside_a_real_timer(self):
        self.assertEqual(parse_ad_remaining(['Ad 1 of 2', 'Ad 0:20']), 20)

    def test_picks_the_smallest_plausible_timer(self):
        """Video runtime and ad timer both on screen: the ad is the shorter."""
        self.assertEqual(parse_ad_remaining(['Ad 0:12', '4:35']), 12)

    def test_rejects_implausible_and_absent(self):
        for texts in ([], None, ['no numbers here'], ['Ad 0:00'], ['99:99']):
            self.assertIsNone(parse_ad_remaining(texts), texts)

    def test_countdown_split_across_ocr_elements(self):
        """The live shape: OCR returns the label and the digit separately.

        Observed on the device as "Sponsored | Skip in | 5".
        """
        self.assertEqual(parse_skip_in(['Sponsored', 'Skip in', '5']), 5)
        self.assertEqual(parse_skip_in(['Sponsored | Skip in | 5']), 5)
        self.assertEqual(parse_ad_remaining(['Ad', '0:15']), 15)
        self.assertEqual(parse_ad_remaining(['Ad | 10']), 10)

    def test_skip_intro_is_not_a_countdown(self):
        """'Skip Intro' is a show control, not an ad timer."""
        self.assertIsNone(parse_skip_in(['Skip Intro']))
        self.assertIsNone(parse_ad_remaining(['Skip Intro']))

    def test_skip_in(self):
        self.assertEqual(parse_skip_in(['Skip in 5']), 5)
        self.assertEqual(parse_skip_in(['Skip Ad in 12s']), 12)
        self.assertIsNone(parse_skip_in(['Skip Ad']))


class TestTracker(unittest.TestCase):

    def test_single_reading_is_not_enough_to_hold(self):
        """One misread digit must not be able to pin a block."""
        t = AdCountdownTracker()
        now = time.time()
        t.observe(15, now)
        self.assertFalse(t.should_hold(now))

    def test_two_consistent_readings_hold(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(15, now)
        t.observe(13, now + 2)
        self.assertTrue(t.should_hold(now + 2))
        self.assertAlmostEqual(t.remaining(now + 2), 13, delta=0.5)

    def test_inconsistent_reading_restarts_corroboration(self):
        """A jump means a different ad (or a misread), not the same clock."""
        t = AdCountdownTracker()
        now = time.time()
        t.observe(15, now)
        t.observe(58, now + 1)          # next ad in the pod
        self.assertFalse(t.should_hold(now + 1))

    def test_hold_ends_when_the_clock_runs_out(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(4, now)
        t.observe(2, now + 2)
        self.assertTrue(t.should_hold(now + 2))
        self.assertFalse(t.should_hold(now + 6))
        self.assertTrue(t.expired(now + 6))

    def test_stale_countdown_stops_holding(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(300, now)
        t.observe(298, now + 2)
        self.assertTrue(t.should_hold(now + 2))
        self.assertFalse(t.should_hold(now + t.STALE_AFTER_S + 5),
                         'an old reading must not outlive its ad')

    def test_reset_clears_everything(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(20, now); t.observe(18, now + 2)
        t.reset()
        self.assertFalse(t.should_hold(now + 2))
        self.assertEqual(t.remaining(now + 2), 0.0)

    def test_none_readings_are_ignored(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(None, now)
        self.assertFalse(t.should_hold(now))


class TestPauseIsTheDangerousCase(unittest.TestCase):
    """A pause freezes the countdown. Holding through it is the failure."""

    def test_frozen_countdown_never_holds(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(20, now)
        t.observe(18, now + 2)
        self.assertTrue(t.should_hold(now + 2))
        # Viewer pauses: the same number keeps coming back.
        for i in range(6):
            t.observe(18, now + 3 + i)
        self.assertTrue(t.is_frozen(now + 9))
        self.assertFalse(t.should_hold(now + 9),
                         'a paused ad must release the block')

    def test_clock_resumes_after_the_pause_ends(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(20, now); t.observe(18, now + 2)
        for i in range(6):
            t.observe(18, now + 3 + i)      # paused
        self.assertFalse(t.should_hold(now + 9))
        t.observe(17, now + 10)             # resumed, clock moves again
        self.assertFalse(t.is_frozen(now + 10))
        self.assertTrue(t.should_hold(now + 10))


class TestSkipBoundIsWeaker(unittest.TestCase):
    """'Skip in N' times the skip button, not the ad."""

    def test_skip_bound_can_hold(self):
        t = AdCountdownTracker()
        now = time.time()
        t.observe(8, now, is_skip_bound=True)
        t.observe(6, now + 2, is_skip_bound=True)
        self.assertTrue(t.should_hold(now + 2))

    def test_skip_bound_never_declares_the_ad_over(self):
        """The ad keeps running after the skip button appears."""
        t = AdCountdownTracker()
        now = time.time()
        t.observe(4, now, is_skip_bound=True)
        t.observe(2, now + 2, is_skip_bound=True)
        self.assertFalse(t.expired(now + 10))


class TestEngineIntegration(unittest.TestCase):
    """The stop decision, with the clock wired in."""

    def _minus(self, remaining=0.0, hold=False, expired=False,
               no_ad=99, last_hit_ago=99.0, level=0.5, state='playing'):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.AD_COUNTDOWN_ENABLED = True
        m.OCR_STOP_THRESHOLD = 3
        m.OCR_STOP_THRESHOLD_MAX = 6
        m.OCR_STOP_MIN_SECONDS = 5.0
        m.flap_escalation = 0
        m.ocr_no_ad_count = no_ad
        m.last_ocr_ad_time = time.time() - last_hit_ago
        m.ad_countdown = MagicMock()
        m.ad_countdown.should_hold.return_value = hold
        m.ad_countdown.expired.return_value = expired
        m.ad_countdown.remaining.return_value = remaining
        m.audio = MagicMock()
        m.audio.get_status.return_value = {'state': state, 'recent_level': level}
        return m

    def test_clock_holds_the_block_against_the_counters(self):
        """Counters say stop, clock says 12s left. The clock wins."""
        m = self._minus(hold=True, remaining=12.0)
        self.assertFalse(m._ocr_says_stop())

    def test_silent_audio_overrides_the_clock(self):
        """Paused mid-ad: audio drops, so the hold is abandoned."""
        m = self._minus(hold=True, remaining=12.0, level=0.0)
        self.assertTrue(m._playback_looks_paused())
        self.assertTrue(m._ocr_says_stop(), 'a pause must release the block')

    def test_expired_clock_releases_without_the_wall_clock_floor(self):
        """Ad is over by its own clock: no need to wait out the 5s floor."""
        m = self._minus(expired=True, no_ad=3, last_hit_ago=1.0)
        self.assertTrue(m._ocr_says_stop())

    def test_expired_clock_still_needs_the_frame_count(self):
        """Never stop on the clock alone -- OCR must agree it sees no ad."""
        m = self._minus(expired=True, no_ad=1, last_hit_ago=1.0)
        self.assertFalse(m._ocr_says_stop())

    def test_unchanged_when_the_clock_has_nothing_to_say(self):
        m = self._minus(hold=False, expired=False, no_ad=3, last_hit_ago=9.0)
        self.assertTrue(m._ocr_says_stop())
        m2 = self._minus(hold=False, expired=False, no_ad=3, last_hit_ago=1.0)
        self.assertFalse(m2._ocr_says_stop(), 'wall-clock floor still applies')

    def test_feature_flag_disables_every_effect(self):
        m = self._minus(hold=True, remaining=30.0)
        m.AD_COUNTDOWN_ENABLED = False
        self.assertFalse(m._ad_clock_says_playing())
        self.assertTrue(m._ocr_says_stop())

    def test_audio_failures_fail_open(self):
        """An unreadable audio path must never pin the overlay up."""
        m = self._minus(hold=True)
        m.audio = None
        self.assertFalse(m._playback_looks_paused())
        m2 = self._minus(hold=True)
        m2.audio.get_status.side_effect = RuntimeError('boom')
        self.assertFalse(m2._playback_looks_paused())

    def test_stopped_audio_pipeline_is_not_a_pause(self):
        """TV off routes playback to fakesink; that is not the viewer pausing."""
        m = self._minus(hold=True, state='null', level=0.0)
        self.assertFalse(m._playback_looks_paused())


class TestScenario(unittest.TestCase):
    """A 30-second ad read once a second, with realistic OCR dropouts."""

    def test_clock_carries_the_block_through_missed_frames(self):
        t = AdCountdownTracker()
        start = time.time()
        held = []
        for elapsed in range(0, 30):
            now = start + elapsed
            # OCR loses the keyword for frames 8-12 -- the classic dropout
            # that used to end the block and cause a re-block.
            if not (8 <= elapsed <= 12):
                t.observe(30 - elapsed, now)
            held.append(t.should_hold(now))
        self.assertTrue(all(held[2:28]), 'block must hold across the dropout')
        self.assertFalse(t.should_hold(start + 31), 'and release once elapsed')


if __name__ == "__main__":
    unittest.main(verbosity=2)
