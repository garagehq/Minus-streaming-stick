#!/usr/bin/env python3
"""Music-mode long-form steering: continuous sighting capture.

The bug, measured over a 5.67h production soak:

The H:MM:SS runtime marker that identifies an hour-long video (a DJ mix,
podcast or stream, as opposed to a music video's M:SS) only appears while the
player overlay is up. It was present in 0.21% of OCR frames -- 54 of 21,363 --
and those 54 frames came from just FIVE distinct overlay appearances, 29 to 124
minutes apart, each lasting 15-19s and carrying up to 24 frames.

`_check_music_drift` runs about every 34s and asked for the marker to be on
screen AT its own sampling instant, three times in a row, resetting the streak
on any sample without it. Odds of roughly 0.21% cubed. Over the whole soak it
logged "long-form runtime on screen (1/3)" twice and never fired once. The box
sat on hour-long mixes yielding 27-28 ad-keyword hits/hour against 118-177 in
good stretches, until the blind 34-minute backstop rescued it -- in one case 72
minutes after the content had changed.

The evidence is transient; the fact it attests to is not. So sightings are now
recorded from EVERY OCR frame by `observe_ocr_text`, and the check asks whether
enough of them landed inside one overlay appearance.

Replaying the real trace through the new rule at the real check moments: it
steers at 07:06:30 and 09:10:46, beating the backstop by 72 and 47 minutes,
and ignores the two single-frame sightings that were most likely OCR misreads.
"""

import collections
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


def _am(music_mode=True, window=60.0, need=3, cooldown=120.0, blocking=False):
    from autonomous_mode import AutonomousMode
    a = AutonomousMode.__new__(AutonomousMode)
    a._music_mode = music_mode
    a._ad_blocker = MagicMock()
    a._ad_blocker.is_visible = blocking
    a._longform_sightings = collections.deque(maxlen=128)
    a._LONGFORM_CONFIRM_WINDOW_S = window
    a._LONGFORM_CONFIRM_FRAMES = need
    a._LONGFORM_STEER_COOLDOWN_S = cooldown
    a._last_longform_steer = 0.0
    return a


class TestObserveOCRText(unittest.TestCase):

    def test_records_long_form_runtime(self):
        a = _am()
        a.observe_ocr_text(['Lofi mix 1:23:45', 'Subscribe'])
        self.assertEqual(len(a._longform_sightings), 1)

    def test_ignores_short_form_runtime(self):
        """A music video shows M:SS. That is the content we WANT."""
        a = _am()
        for _ in range(10):
            a.observe_ocr_text(['Artist - Song 3:45'])
        self.assertEqual(len(a._longform_sightings), 0)

    def test_ignores_everything_when_music_mode_off(self):
        a = _am(music_mode=False)
        a.observe_ocr_text(['Podcast 2:10:00'])
        self.assertEqual(len(a._longform_sightings), 0)

    def test_ignores_text_while_an_ad_is_blocking(self):
        """Mid-block the OCR text is the ad's, not the underlying video's."""
        a = _am(blocking=True)
        a.observe_ocr_text(['Ad 1:30:00'])
        self.assertEqual(len(a._longform_sightings), 0)

    def test_survives_junk_input(self):
        a = _am()
        for bad in (None, [], [None], [123], ['']):
            a.observe_ocr_text(bad)
        self.assertEqual(len(a._longform_sightings), 0)

    def test_deque_is_bounded(self):
        a = _am()
        for _ in range(500):
            a.observe_ocr_text(['x 1:00:00'])
        self.assertLessEqual(len(a._longform_sightings), 128)


class TestLongFormRuntimeShapes(unittest.TestCase):
    """H:MM:SS is not the only long-form runtime.

    Observed 2026-09-11: the hour-only rule matched 0 of 3616 OCR frames all
    night while the autoplay panel queued "Club 1BD | Hip Hop, RnB, Edits,
    Dancehall | 40:18 | DJ Miss Milan". A 40-minute mix renders as MM:SS and
    was invisible. A wall clock renders the same way, so clock/date context
    vetoes it -- measured, 67 frames carried MM:SS >= 10 and only 2 were a
    clock, each alongside a weekday or month.
    """

    def setUp(self):
        self.a = _am()

    def test_hour_plus_runtime(self):
        self.assertTrue(self.a._looks_long_form('Now playing 1:23:45 mix'))

    def test_forty_minute_mix_from_the_autoplay_panel(self):
        self.assertTrue(self.a._looks_long_form(
            'X | Club 1BD | 2:26 | Hip Hop, RnB, Edits, Dancehall | 40:18 | DJ Miss Milan'))

    def test_ten_minute_boundary_is_long_form(self):
        self.assertTrue(self.a._looks_long_form('set 10:00'))

    def test_normal_music_video_is_not_long_form(self):
        for t in ('Artist - Song 3:45', 'Official Video 4:24', '0:00 / 2:26'):
            self.assertFalse(self.a._looks_long_form(t), t)

    def test_wall_clock_is_vetoed(self):
        for t in ('11:48 | WED, SEP 9', '10:30 PM', 'FRI, OCT 3 | 22:15'):
            self.assertFalse(self.a._looks_long_form(t), t)

    def test_ad_countdown_is_not_long_form(self):
        self.assertFalse(self.a._looks_long_form('Ad 0:30 left | Skip in 5'))

    def test_observer_records_the_new_shape(self):
        a = _am()
        for _ in range(3):
            a.observe_ocr_text(['Club 1BD | 40:18 | DJ Miss Milan'])
        self.assertGreaterEqual(a._longform_confirmed(), a._LONGFORM_CONFIRM_FRAMES)

    def test_observer_ignores_a_clock(self):
        a = _am()
        for _ in range(6):
            a.observe_ocr_text(['11:48 | WED, SEP 9'])
        self.assertEqual(a._longform_confirmed(), 0)


class TestConfirmationWindow(unittest.TestCase):

    def test_single_sighting_does_not_confirm(self):
        """Guards against a one-frame OCR misread fabricating H:MM:SS.

        Two of the five real appearances were single frames; both should be
        ignored.
        """
        a = _am()
        a.observe_ocr_text(['x 1:00:00'])
        self.assertLess(a._longform_confirmed(), a._LONGFORM_CONFIRM_FRAMES)

    def test_one_overlay_appearance_confirms(self):
        """15-19s of overlay carrying many frames is the real signal."""
        a = _am()
        for _ in range(5):
            a.observe_ocr_text(['DJ set 2:04:11'])
        self.assertGreaterEqual(a._longform_confirmed(), a._LONGFORM_CONFIRM_FRAMES)

    def test_sightings_expire_out_of_the_window(self):
        a = _am(window=1.0)
        for _ in range(5):
            a.observe_ocr_text(['mix 1:11:11'])
        time.sleep(1.1)
        self.assertEqual(a._longform_confirmed(), 0)

    def test_window_does_not_span_separate_appearances(self):
        """Real appearances are 29-124 min apart; the window must not merge them."""
        a = _am()
        self.assertLess(a._LONGFORM_CONFIRM_WINDOW_S, 29 * 60)


class TestTraceReplay(unittest.TestCase):
    """Replay the real overlay appearances from the production soak."""

    # (label, frames in the appearance, span seconds)
    APPEARANCES = [
        ('04:40 mix', 6, 15),
        ('05:09 single frame', 1, 0),
        ('05:38 single frame', 1, 0),
        ('07:06 DJ mix', 22, 17),
        ('09:10 DJ mix', 24, 19),
    ]

    def _confirms(self, frames, span):
        a = _am()
        now = time.time()
        step = (span / max(frames - 1, 1)) if frames > 1 else 0
        for i in range(frames):
            a._longform_sightings.append(now - span + i * step)
        return a._longform_confirmed() >= a._LONGFORM_CONFIRM_FRAMES

    def test_multi_frame_appearances_confirm(self):
        for label, frames, span in self.APPEARANCES:
            if frames >= 3:
                self.assertTrue(self._confirms(frames, span),
                                f'{label} should confirm ({frames} frames)')

    def test_single_frame_appearances_do_not_confirm(self):
        for label, frames, span in self.APPEARANCES:
            if frames == 1:
                self.assertFalse(self._confirms(frames, span),
                                 f'{label} is a likely misread, must not steer')

    def test_old_rule_could_not_fire(self):
        """The regression: 3 CONSECUTIVE samples at a 0.21% hit rate.

        Streak resets on any sample without the marker, so reaching 3 needs
        three consecutive hits.
        """
        p_hit = 0.0021
        self.assertLess(p_hit ** 3, 1e-8,
                        'old rule was effectively unreachable -- that was the bug')


class TestSteerCooldown(unittest.TestCase):
    """One appearance must not steer twice.

    An appearance lasts 15-19s and the check runs every ~34s, so without a
    cooldown the tail of the same overlay can re-confirm on the next check.
    """

    def test_cooldown_blocks_immediate_resteer(self):
        a = _am()
        a._last_longform_steer = time.time()
        self.assertLess(time.time() - a._last_longform_steer,
                        a._LONGFORM_STEER_COOLDOWN_S)

    def test_cooldown_outlasts_one_appearance(self):
        a = _am()
        self.assertGreater(a._LONGFORM_STEER_COOLDOWN_S, 19,
                           'cooldown must exceed the longest observed appearance')


class TestThermalStopThresholdUnchanged(unittest.TestCase):
    """Follow-up 4: analysed, deliberately NOT lowered.

    Measured OCR miss-run distribution over 24h / 101 ad breaks: runs of
    exactly 1,2,3,4,5+ occurred 15, 118, 4, 6, 0 times. False-stop exposure is
    the count of runs >= threshold, so 5 is the first threshold with ZERO
    observed exposure while 4 has six. Thermal-degraded mode is where OCR is
    slowest and misses are most likely, so trading that for ~1s is a bad deal.
    The capture fix already halved its absolute cost, 10.0s to 5.0s.
    """

    MISS_RUNS = {1: 15, 2: 118, 3: 4, 4: 6, 5: 0}

    def _exposure(self, n):
        return sum(v for k, v in self.MISS_RUNS.items() if k >= n)

    def test_five_has_no_observed_exposure(self):
        self.assertEqual(self._exposure(5), 0)

    def test_four_would_admit_false_stops(self):
        self.assertGreater(self._exposure(4), 0)

    def test_degraded_threshold_is_still_five(self):
        from minus import Minus
        self.assertEqual(Minus.THERMAL_DEGRADED_PARAMS['OCR_STOP_THRESHOLD'], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
