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

from ad_countdown import (AdCountdownTracker, SKIP_LABEL_ASSUMED_S,  # noqa: E402
                          has_skip_countdown_label, parse_ad_remaining,
                          parse_skip_in, parse_skip_in_candidates)


class TestParsing(unittest.TestCase):

    def test_plain_timestamp(self):
        self.assertEqual(parse_ad_remaining(['Ad 0:15']), 15)
        self.assertEqual(parse_ad_remaining(['Ad0:30']), 30)
        self.assertEqual(parse_ad_remaining(['Ad 1:20']), 80)

    def test_ocr_misreads(self):
        """The exact shapes CLAUDE.md documents, plus the user's '0:1s'."""
        for text, expect in (('Ado:15', 15), ('Ad0:1s', 15), ('Ado:3o', 30),
                             ('Ad0;30', 30), ('Ad0.30', 30)):
            self.assertEqual(parse_ad_remaining([text]), expect, text)

    def test_fully_misread_timer_is_skipped_not_guessed(self):
        """'Adl:lo' is 1:10 with every character misread.

        We decline it. The same shape without a real digit also matches
        ordinary words -- 'hello:so' would parse as 10:50 -- and a fabricated
        deadline pins a block. Skipping costs nothing because the countdown
        is advisory; guessing wrong costs a held overlay.
        """
        self.assertIsNone(parse_ad_remaining(['Adl:lo']))
        self.assertIsNone(parse_ad_remaining(['hello:so']))

    def test_plural_ads_is_not_five_seconds(self):
        """Caught live: 'Free with ads PG' read as ad + s, and s->5."""
        self.assertIsNone(parse_ad_remaining(['Free with ads PG']))
        self.assertIsNone(parse_ad_remaining(['Press and hold for ad options']))
        self.assertIsNone(parse_skip_in(['skip ads']))

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

    def test_rejects_a_wall_clock_as_an_ad_timer(self):
        """Caught live: "CIil | 12:49 | a" parsed as 769 seconds.

        12:49 is a clock, not an ad timer, and a fabricated 12-minute
        deadline is exactly what could pin a block. Real ad breaks top out
        around 2-3 minutes.
        """
        self.assertIsNone(parse_ad_remaining(['CIil', '12:49', 'a']))
        self.assertIsNone(parse_ad_remaining(['10:30']))

    def test_still_accepts_a_long_but_plausible_ad(self):
        self.assertEqual(parse_ad_remaining(['Ad 2:30']), 150)

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


class TestUnreadableSkipDigit(unittest.TestCase):
    """OCR drops the countdown digit constantly.

    Measured live on a Disney+ pre-roll: 0 of 36 in-block frames yielded a
    number while "Skip in" / "Skip I" was plainly on screen, and 20 of those
    36 frames read pure garbage ("O07"). The label alone still states a fact --
    the skip gate has not opened, so the ad is definitely still running -- so
    it is worth a bounded hold even without a number.
    """

    REAL_FRAMES = [
        ['Sponsored • Now streaming', 'Send to phone', '90+', 'Skip I',
         'disneyplus.com', 'The Devil Wears'],
        ['Sponsored · Now streaming', 'Send to phone', 'Skip in',
         'disneyplus.com', 'TheDevilWarsPrada'],
    ]

    def test_detects_the_label_on_real_frames(self):
        for f in self.REAL_FRAMES:
            self.assertTrue(has_skip_countdown_label(f), f)

    def test_a_readable_digit_takes_precedence(self):
        """When the number IS readable, use it rather than the assumption."""
        self.assertFalse(has_skip_countdown_label(['Skip in 5']))
        self.assertEqual(parse_skip_in(['Skip in 5']), 5)

    def test_skip_intro_never_matches(self):
        """A show control, not an ad timer."""
        for t in (['Skip Intro'], ['Sk1p 1ntro'], ['skip intro']):
            self.assertFalse(has_skip_countdown_label(t), t)

    def test_ready_skip_button_is_not_a_countdown(self):
        """'Skip' alone means the gate is already open."""
        for t in (['Skip'], ['skip ad'], ['Sponsored']):
            self.assertFalse(has_skip_countdown_label(t), t)

    def test_hold_persists_across_the_real_read_pattern(self):
        """The label is legible on roughly one frame in three."""
        t = AdCountdownTracker()
        n = 1000.0
        reads = {0, 3, 6, 8, 11}
        held = []
        for i in range(17):
            if i in reads:
                t.observe(SKIP_LABEL_ASSUMED_S, n + i,
                          is_skip_bound=True, synthetic=True)
            held.append(t.should_hold(n + i))
        self.assertTrue(all(held[4:13]),
                        'the hold must bridge the frames OCR cannot read')

    def test_hold_lapses_after_the_label_goes(self):
        t = AdCountdownTracker()
        n = 1000.0
        for i in (0, 2, 4):
            t.observe(SKIP_LABEL_ASSUMED_S, n + i, is_skip_bound=True, synthetic=True)
        self.assertTrue(t.should_hold(n + 4))
        self.assertFalse(t.should_hold(n + 4 + SKIP_LABEL_ASSUMED_S + 1))

    def test_synthetic_readings_are_exempt_from_freeze(self):
        """The assumed value is constant by construction.

        Without the exemption it trips the frozen-clock detector on every ad
        and kills the hold it exists to provide. Pause protection for this
        path comes from audio instead.
        """
        t = AdCountdownTracker()
        n = 1000.0
        for i in range(10):
            t.observe(SKIP_LABEL_ASSUMED_S, n + i, is_skip_bound=True, synthetic=True)
        self.assertFalse(t.is_frozen(n + 9))
        self.assertTrue(t.should_hold(n + 9))

    def test_a_real_repeating_number_still_freezes(self):
        """The exemption must not disarm pause detection for real readings."""
        t = AdCountdownTracker()
        n = 1000.0
        for i in range(10):
            t.observe(18, n + i)
        self.assertTrue(t.is_frozen(n + 9))
        self.assertFalse(t.should_hold(n + 9))

    def test_label_hold_never_declares_the_ad_over(self):
        t = AdCountdownTracker()
        n = 1000.0
        for i in (0, 2):
            t.observe(SKIP_LABEL_ASSUMED_S, n + i, is_skip_bound=True, synthetic=True)
        self.assertFalse(t.expired(n + 30))


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


class TestRetargetAndQuarantine(unittest.TestCase):
    """Agreeing readings retarget; contradicting ones are quarantined.

    A reading that agrees with the running clock re-anchors on it, so 0:15
    followed by 0:12 leaves a 12-second deadline. A reading that contradicts
    an established clock (0:15 -> 0:99) is far more likely a misread than a
    real ad, so it must not reassign on the spot -- that would both fabricate
    a 99-second hold and discard the corroboration already earned, collapsing
    a legitimate hold mid-ad. It is held aside and adopted only once a later
    reading agrees with IT (0:99 then 0:97), which is also what a genuine
    resync into the next ad of a pod looks like.
    """

    def _established(self, base=1000.0, values=(30, 28, 26, 24)):
        t = AdCountdownTracker()
        for i, v in enumerate(values):
            t.observe(v, base + i * 2)
        return t, base + (len(values) - 1) * 2

    def test_agreeing_reading_retargets(self):
        t = AdCountdownTracker(); n = 1000.0
        t.observe(15, n)
        t.observe(12, n + 3)
        self.assertAlmostEqual(t.remaining(n + 3), 12, delta=0.5)
        self.assertTrue(t.should_hold(n + 3))

    def test_it_keeps_retargeting_as_the_ad_runs_down(self):
        t = AdCountdownTracker(); n = 1000.0
        for i, v in enumerate((20, 18, 16, 14, 12)):
            t.observe(v, n + i * 2)
        self.assertAlmostEqual(t.remaining(n + 8), 12, delta=0.5)

    def test_single_outlier_does_not_reassign_the_deadline(self):
        t, now = self._established()
        before = t.remaining(now)
        t.observe(99, now + 2)
        self.assertLess(t.remaining(now + 2), before,
                        'a wild reading must not become the new deadline')
        self.assertLess(t.remaining(now + 2), 30)

    def test_single_outlier_does_not_collapse_an_earned_hold(self):
        """The regression this guards: one bad read used to end a live hold."""
        t, now = self._established()
        self.assertTrue(t.should_hold(now))
        t.observe(99, now + 2)
        self.assertTrue(t.should_hold(now + 2))

    def test_good_readings_resume_after_an_outlier(self):
        t, now = self._established()
        t.observe(99, now + 2)
        t.observe(22, now + 4)
        self.assertAlmostEqual(t.remaining(now + 4), 22, delta=1.0)
        self.assertTrue(t.should_hold(now + 4))

    def test_corroborated_jump_is_adopted(self):
        """0:99 then 0:97 -- a real resync, one reading later."""
        t, now = self._established()
        t.observe(99, now + 2)
        t.observe(97, now + 4)
        self.assertAlmostEqual(t.remaining(now + 4), 97, delta=1.0)
        self.assertTrue(t.should_hold(now + 4))

    def test_next_ad_in_a_pod_is_picked_up(self):
        """A 30s ad ends and a 60s one starts: adopted after corroboration."""
        t, now = self._established(values=(6, 4, 2))
        t.observe(60, now + 2)
        self.assertLess(t.remaining(now + 2), 10, 'not adopted on one reading')
        t.observe(58, now + 4)
        self.assertAlmostEqual(t.remaining(now + 4), 58, delta=1.0)

    def test_two_unrelated_outliers_never_adopt(self):
        """Noise that never agrees with itself must not take over."""
        t, now = self._established()
        t.observe(99, now + 2)
        t.observe(41, now + 4)
        t.observe(77, now + 6)
        self.assertLess(t.remaining(now + 6), 30)


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
        m.OCR_STOP_MAX_SECONDS = 9.0
        m.flap_escalation = 0
        m.ocr_no_ad_count = no_ad
        m.last_ocr_ad_time = time.time() - last_hit_ago
        m.ad_countdown = MagicMock()
        m.ad_countdown.should_hold.return_value = hold
        m.ad_countdown.expired.return_value = expired
        m.ad_countdown.remaining.return_value = remaining
        m.audio = MagicMock()
        m.audio.get_status.return_value = {'state': state, 'recent_level': level}
        # Observability counters the engine bumps as it decides.
        m.ad_clock_stats = {'parsed': 0, 'holds': 0, 'early_release': 0,
                            'pause_override': 0, 'last_value': None}
        m._ad_clock_log_last = 0.0
        return m

    def test_clock_holds_the_block_against_the_counters(self):
        """Counters say stop, clock says 12s left. The clock wins."""
        m = self._minus(hold=True, remaining=12.0)
        self.assertFalse(m._ocr_says_stop())
        self.assertEqual(m.ad_clock_stats['holds'], 1, 'the hold must be counted')

    def test_counters_record_what_the_clock_did(self):
        """The signal has to be observable, or a soak can only guess."""
        m = self._minus(expired=True, no_ad=3, last_hit_ago=1.0)
        m._ocr_says_stop()
        self.assertEqual(m.ad_clock_stats['early_release'], 1)
        m2 = self._minus(hold=True, remaining=9.0, level=0.0)
        m2._ocr_says_stop()
        self.assertEqual(m2.ad_clock_stats['pause_override'], 1)

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


class TestVetoAppliesToEverySource(unittest.TestCase):
    """The clock must outrank VLM too, not just OCR.

    Observed live: an ad flapped through 6 blocks in 27 seconds and every one
    ended "stopped by BOTH" -- VLM was ending them. A veto living only inside
    _ocr_says_stop never got a vote on those, so it has to sit above the
    per-source branches in _update_blocking_state.
    """

    def _body(self):
        """The whole _update_blocking_state method."""
        src = (ROOT / 'minus.py').read_text()
        start = src.index('    def _update_blocking_state')
        nxt = src.index('\n    def ', start + 10)
        return src[start:nxt]

    def test_veto_is_consulted_in_update_blocking_state(self):
        self.assertIn('_ad_clock_says_playing', self._body(),
                      'the clock must gate the stop decision for every source')

    def test_veto_runs_before_the_per_source_branches(self):
        body = self._body()
        veto = body.index('_ad_clock_says_playing')
        branch = body.index('should_stop = vlm_says_stop')
        self.assertLess(veto, branch,
                        'the veto must precede the VLM/OCR/both branches')


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

class TestCrossElementSkipCountdown(unittest.TestCase):
    """OCR returns each text box separately, so the skip digit arrives as its
    own element and rarely next to the label.

    Measured over a 5h soak: the digit was present on 46 of 107 "Skip in"
    frames but immediately after the label on 2, so the adjacency regex saw
    almost none of them and the tracker ran on the flat assumed value instead.
    """

    def test_digit_several_elements_from_the_label(self):
        els = ['Sponsored - Women Preferences', 'Skip in', 'Uber',
               'Send to phone', '20', 'uber.com']
        self.assertEqual(parse_skip_in(els), 20)

    def test_digit_before_the_label(self):
        els = ['Sponsored', '19', 'Skip in', 'Uber', 'Send to phone']
        self.assertEqual(parse_skip_in(els), 19)

    def test_adjacent_form_still_wins(self):
        self.assertEqual(parse_skip_in(['Skip in 5']), 5)

    def test_no_label_means_no_countdown(self):
        """A number on screen is not a countdown without the label."""
        self.assertIsNone(parse_skip_in(['Uber', '20', 'uber.com']))

    def test_skip_intro_is_not_a_countdown(self):
        self.assertIsNone(parse_skip_in(['Sponsored', 'Skip Intro', '12']))

    def test_implausible_gate_rejected(self):
        """A skip gate is never 90s; that number is something else on screen."""
        self.assertIsNone(parse_skip_in(['Skip in', '90']))
        self.assertEqual(parse_skip_in(['Skip in', '30']), 30)

    def test_candidates_are_ordered_nearest_label_first(self):
        els = ['Sponsored', 'Send to phone', '49', 'Skip I(', 'doordash.com',
               '5', '29']
        self.assertEqual(parse_skip_in_candidates(els)[0], 49)
        self.assertIn(5, parse_skip_in_candidates(els))

    def test_label_without_digit_still_reports_no_number(self):
        self.assertIsNone(parse_skip_in(['Sponsored', 'Skip in', 'uber.com']))
        self.assertTrue(has_skip_countdown_label(['Sponsored', 'Skip in']))


class TestCandidateDisambiguation(unittest.TestCase):
    """The creative renders numbers too, so more than one candidate is normal.

    A Disney+ ad parks a static "33" where the countdown sits and a recurring
    "2" shows up mid-countdown. Position cannot separate those from the real
    reading; the running clock can.
    """

    def test_projection_picks_the_real_countdown_over_noise(self):
        t = AdCountdownTracker()
        now = 1000.0
        self.assertEqual(t.observe_candidates([25, 2], now, is_skip_bound=True), 25)
        # Noise listed FIRST from here on; the clock should still track truth.
        self.assertEqual(t.observe_candidates([2, 24], now + 1, is_skip_bound=True), 24)
        self.assertEqual(t.observe_candidates([2, 23], now + 2, is_skip_bound=True), 23)
        self.assertAlmostEqual(t.remaining(now + 2), 23.0, places=1)
        self.assertTrue(t.is_confident(now + 2))

    def test_first_candidate_taken_with_no_clock_running(self):
        t = AdCountdownTracker()
        self.assertEqual(t.observe_candidates([15, 3], 1000.0, is_skip_bound=True), 15)

    def test_single_reading_cannot_hold_a_block(self):
        """MIN_READINGS still applies to a candidate-chosen value."""
        t = AdCountdownTracker()
        t.observe_candidates([33], 1000.0, is_skip_bound=True)
        self.assertFalse(t.is_confident(1000.0))

    def test_static_number_does_not_masquerade_as_a_countdown(self):
        """A creative's fixed 33 repeats instead of counting down, so it reads
        as frozen rather than as time left to hold on."""
        t = AdCountdownTracker()
        now = 1000.0
        for i in range(8):
            t.observe_candidates([33], now + i, is_skip_bound=True)
        self.assertTrue(t.is_frozen(now + 8))

    def test_empty_candidates_are_a_no_op(self):
        t = AdCountdownTracker()
        self.assertIsNone(t.observe_candidates([], 1000.0))


class TestLongFormRuntimeIsNotACountdown(unittest.TestCase):
    """An H:MM:SS runtime contains a well-formed M:SS.

    A 70-minute music mix renders "1:10:55"; its leading "1:10" parsed as a
    70-second countdown on 286 non-ad frames in a single night. The plausible-
    seconds ceiling cannot catch it, because 70 seconds is a perfectly ordinary
    ad length -- the tell is the trailing colon, not the magnitude.
    """

    def test_hour_long_runtime_is_rejected(self):
        self.assertIsNone(parse_ad_remaining(['1:10:55']))
        self.assertIsNone(parse_ad_remaining(['2:05:00']))

    def test_runtime_amongst_real_video_text_is_rejected(self):
        els = ['X', 'Book Club Radio', 'That Funky House Mix', '1:10:55',
               'Next in 19']
        self.assertIsNone(parse_ad_remaining(els))

    def test_real_ad_timers_still_parse(self):
        self.assertEqual(parse_ad_remaining(['Ad 0:30']), 30)
        self.assertEqual(parse_ad_remaining(['Ad 1:45']), 105)
        self.assertEqual(parse_ad_remaining(['Ado:15']), 15)
        self.assertEqual(parse_ad_remaining(['Ad0:30']), 30)

    def test_wall_clock_still_rejected(self):
        self.assertIsNone(parse_ad_remaining(['CIil', '12:49', 'a']))




class TestAssumedGateLength(unittest.TestCase):
    """The fallback used when "Skip in" is readable but its digit is not.

    The original 5s came from an assumption about streaming skip gates that a
    soak disproved: measured gates ran p50 21.5s and up to 53s, and 5s was
    short enough to release mid-ad and re-block -- 60% of blocks that night
    ended at 5-6s, the constant's own length.
    """

    def test_assumption_outlasts_a_typical_ocr_label_dropout(self):
        """The label is legible on roughly one frame in three, so the hold has
        to bridge the frames where it is missed."""
        self.assertGreaterEqual(SKIP_LABEL_ASSUMED_S, 8)

    def test_assumption_stays_well_inside_the_recovery_budget(self):
        """It is also the worst-case over-hold once the ad really has ended."""
        self.assertLessEqual(SKIP_LABEL_ASSUMED_S, 15)

    def test_hold_survives_a_gap_that_used_to_end_the_block(self):
        t = AdCountdownTracker()
        n = 1000.0
        for i in range(3):
            t.observe(SKIP_LABEL_ASSUMED_S, n + i, is_skip_bound=True,
                      synthetic=True)
        # 6s after the last sighting the old constant had already lapsed.
        self.assertTrue(t.should_hold(n + 2 + 6))

    def test_hold_still_lapses_rather_than_pinning_the_overlay(self):
        t = AdCountdownTracker()
        n = 1000.0
        for i in range(3):
            t.observe(SKIP_LABEL_ASSUMED_S, n + i, is_skip_bound=True,
                      synthetic=True)
        self.assertFalse(t.should_hold(n + 2 + SKIP_LABEL_ASSUMED_S + 1))

    def test_assumption_never_ends_a_block(self):
        """Skip-bound is a lower bound: it may hold, never release."""
        t = AdCountdownTracker()
        n = 1000.0
        for i in range(3):
            t.observe(SKIP_LABEL_ASSUMED_S, n + i, is_skip_bound=True,
                      synthetic=True)
        self.assertFalse(t.expired(n + 2 + SKIP_LABEL_ASSUMED_S + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
