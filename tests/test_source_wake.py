#!/usr/bin/env python3
"""Autonomous mode has to wake a source that has powered its HDMI output down.

A Fire TV left idle long enough stops driving HDMI. Minus correctly shows NO
SIGNAL, but autonomous mode carried on as if nothing were wrong: it has no
frames, so every screen check classifies nothing, and the recovery it reaches
for is an ADB deep-link. ADB accepts those while the output stays asleep, so it
logged success indefinitely.

Observed live 2026-09-17: the input sat dark for 45871s (12.7 hours) while
music seeds were dispatched into it. A single `home` keypress restored it.

An intent is not input. Only a key event wakes the output.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import autonomous_mode as am  # noqa: E402
from autonomous_mode import AutonomousMode  # noqa: E402


class TestSignalPredicate(unittest.TestCase):

    def setUp(self):
        self.am = AutonomousMode()

    def test_unwired_predicate_assumes_signal(self):
        """Behaviour is unchanged where nothing installs the check."""
        self.assertTrue(self.am._signal_present())

    def test_predicate_is_honoured(self):
        self.am.set_signal_present_predicate(lambda: False)
        self.assertFalse(self.am._signal_present())

    def test_broken_probe_fails_open(self):
        """A failing probe must not make us hammer the remote forever."""
        def boom():
            raise RuntimeError('v4l2 unreadable')
        self.am.set_signal_present_predicate(boom)
        self.assertTrue(self.am._signal_present())


class TestWakeBehaviour(unittest.TestCase):

    def setUp(self):
        self.am = AutonomousMode()
        self.ctrl = MagicMock()
        self.ctrl.is_connected.return_value = True
        self.am._device_controller = self.ctrl
        self.am._log_event = lambda *a, **k: None

    def test_sends_home_not_an_intent(self):
        """The whole bug is that ADB intents do not wake the output."""
        self.assertTrue(self.am._wake_sleeping_source())
        self.ctrl.send_command.assert_called_once_with('home')

    def test_rate_limited(self):
        self.assertTrue(self.am._wake_sleeping_source())
        self.assertFalse(self.am._wake_sleeping_source())
        self.assertEqual(self.ctrl.send_command.call_count, 1)

    def test_retries_after_the_interval(self):
        self.am._wake_sleeping_source()
        self.am._last_source_wake = time.time() - am.SOURCE_WAKE_INTERVAL - 1
        self.assertTrue(self.am._wake_sleeping_source())
        self.assertEqual(self.ctrl.send_command.call_count, 2)

    def test_no_controller_is_a_no_op(self):
        self.am._device_controller = None
        self.assertFalse(self.am._wake_sleeping_source())

    def test_disconnected_controller_is_a_no_op(self):
        self.ctrl.is_connected.return_value = False
        self.assertFalse(self.am._wake_sleeping_source())

    def test_keypress_failure_is_reported(self):
        self.ctrl.send_command.side_effect = RuntimeError('adb gone')
        self.assertFalse(self.am._wake_sleeping_source())


class TestKeepaliveShortCircuits(unittest.TestCase):
    """With no picture, the frame-based path must not run at all."""

    def setUp(self):
        self.am = AutonomousMode()
        self.ctrl = MagicMock()
        self.ctrl.is_connected.return_value = True
        self.am._device_controller = self.ctrl
        self.am._log_event = lambda *a, **k: None
        self.am._check_roku_active_app = MagicMock(return_value=True)
        self.am._check_android_active_app = MagicMock(return_value=True)

    def test_no_signal_wakes_and_skips_screen_checks(self):
        self.am.set_signal_present_predicate(lambda: False)
        result = self.am._ensure_youtube_playing()
        self.assertTrue(result)
        self.ctrl.send_command.assert_called_once_with('home')
        self.am._check_android_active_app.assert_not_called()

    def test_signal_present_runs_the_normal_path(self):
        self.am.set_signal_present_predicate(lambda: True)
        self.am._ensure_youtube_playing()
        self.am._check_android_active_app.assert_called()


class TestMinusPredicate(unittest.TestCase):

    def _probe(self, lost):
        import minus as m
        o = object.__new__(m.Minus)
        o._hdmi_signal_lost = lost
        return m.Minus._hdmi_signal_present(o)

    def test_signal_lost_reports_absent(self):
        self.assertFalse(self._probe(True))

    def test_signal_ok_reports_present(self):
        self.assertTrue(self._probe(False))

    def test_missing_attribute_assumes_present(self):
        import minus as m
        o = object.__new__(m.Minus)
        self.assertTrue(m.Minus._hdmi_signal_present(o))


if __name__ == "__main__":
    unittest.main(verbosity=2)
