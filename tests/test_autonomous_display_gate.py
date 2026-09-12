#!/usr/bin/env python3
"""Autonomous mode stands down when the TV is actually being watched.

Autonomous mode drives the streaming device's remote to keep YouTube playing
overnight. The moment a person turns the TV on they are trying to use that
device, and a background process pressing buttons is not a background process
any more -- it changes what they are watching out from under them.

The signal is deliberately BOTH "the pipeline is up" AND "the TV is live",
because each half alone lies in a different direction: the pipeline flag stays
True until a retry loop notices a disconnect, and a TV can be plugged in while
our pipeline is down.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from autonomous_mode import AutonomousMode  # noqa: E402


class TestDisplayInUsePredicate(unittest.TestCase):

    def setUp(self):
        self.am = AutonomousMode()

    def test_absent_predicate_means_not_in_use(self):
        """Unwired, behaviour is exactly as before."""
        self.assertFalse(self.am._display_in_use())

    def test_predicate_result_is_honoured(self):
        self.am.set_display_in_use_predicate(lambda: True)
        self.assertTrue(self.am._display_in_use())
        self.am.set_display_in_use_predicate(lambda: False)
        self.assertFalse(self.am._display_in_use())

    def test_predicate_can_be_removed(self):
        self.am.set_display_in_use_predicate(lambda: True)
        self.am.set_display_in_use_predicate(None)
        self.assertFalse(self.am._display_in_use())

    def test_raising_predicate_fails_closed(self):
        """A broken probe must never permanently disable autonomous mode."""
        def boom():
            raise RuntimeError("sysfs unreadable")
        self.am.set_display_in_use_predicate(boom)
        self.assertFalse(self.am._display_in_use())

    def test_truthiness_is_coerced(self):
        self.am.set_display_in_use_predicate(lambda: 1)
        self.assertTrue(self.am._display_in_use())


class TestSuppressionOverridesEverything(unittest.TestCase):
    """The gate has to beat both the schedule and a manual start."""

    def setUp(self):
        self.am = AutonomousMode()
        self.am._activate = MagicMock(
            side_effect=lambda: setattr(self.am, '_active', True))
        self.am._deactivate = MagicMock(
            side_effect=lambda: setattr(self.am, '_active', False))

    def _tick(self):
        """One iteration of the run loop's activation decision."""
        should_be_active = self.am._manual_override or self.am.is_scheduled_time()
        display_busy = self.am._display_in_use()
        if display_busy:
            should_be_active = False
        if should_be_active and not self.am._active:
            self.am._activate()
        elif not should_be_active and self.am._active and (
                not self.am._manual_override or display_busy):
            self.am._deactivate()

    def test_scheduled_session_stops_when_tv_turns_on(self):
        self.am.is_scheduled_time = lambda: True
        self.am.set_display_in_use_predicate(lambda: False)
        self._tick()
        self.assertTrue(self.am._active)
        # TV comes on mid-window.
        self.am.set_display_in_use_predicate(lambda: True)
        self._tick()
        self.assertFalse(self.am._active)

    def test_manual_override_also_yields_to_the_tv(self):
        """A manual start is still the user's intent, but turning the TV on is
        a newer and more specific one."""
        self.am._manual_override = True
        self.am.set_display_in_use_predicate(lambda: False)
        self._tick()
        self.assertTrue(self.am._active)
        self.am.set_display_in_use_predicate(lambda: True)
        self._tick()
        self.assertFalse(self.am._active)

    def test_does_not_start_while_the_tv_is_already_on(self):
        self.am.is_scheduled_time = lambda: True
        self.am.set_display_in_use_predicate(lambda: True)
        self._tick()
        self.assertFalse(self.am._active)
        self.am._activate.assert_not_called()

    def test_resumes_once_the_tv_goes_off_again(self):
        self.am.is_scheduled_time = lambda: True
        self.am.set_display_in_use_predicate(lambda: True)
        self._tick()
        self.assertFalse(self.am._active)
        self.am.set_display_in_use_predicate(lambda: False)
        self._tick()
        self.assertTrue(self.am._active)

    def test_manual_override_still_survives_leaving_the_schedule(self):
        """The pre-existing behaviour must not regress: outside the window a
        manual session keeps running, as long as the TV is off."""
        self.am.is_scheduled_time = lambda: False
        self.am._manual_override = True
        self.am.set_display_in_use_predicate(lambda: False)
        self._tick()
        self.assertTrue(self.am._active)
        self._tick()
        self.assertTrue(self.am._active)


class TestMinusPredicateSemantics(unittest.TestCase):
    """Minus._display_in_use requires both halves."""

    def _probe(self, pipeline_up, tv_live):
        import minus as m
        obj = object.__new__(m.Minus)
        obj.display_connected = pipeline_up
        obj.is_display_connected_live = lambda: tv_live
        return m.Minus._display_in_use(obj)

    def test_both_true_is_in_use(self):
        self.assertTrue(self._probe(True, True))

    def test_stale_pipeline_flag_with_tv_off_is_not_in_use(self):
        self.assertFalse(self._probe(True, False))

    def test_tv_present_but_pipeline_down_is_not_in_use(self):
        self.assertFalse(self._probe(False, True))

    def test_neither_is_not_in_use(self):
        self.assertFalse(self._probe(False, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
