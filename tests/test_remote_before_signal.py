#!/usr/bin/env python3
"""The remote must be reachable while the source is asleep.

Waking a streaming stick that has powered its HDMI output down takes a
keypress, and a keypress needs the remote connection. Device setup used to run
only after Minus saw a picture on the input, so starting with the stick asleep
left nothing able to wake it. Seen live: after a restart the Fire TV's ADB
answered in one second once Minus tried, but Minus did not try for 5.5 minutes
because it was waiting for the picture that only a keypress could bring back.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import minus as m  # noqa: E402


def run_setup(cfg, **kw):
    o = object.__new__(m.Minus)
    o.running = True
    mgr = MagicMock()
    mgr.get_config.return_value = cfg
    with patch('src.device_config.get_device_config_manager', return_value=mgr), \
         patch('minus.threading.Thread') as T:
        o._start_device_setup_delayed(**kw)
        return o, T


PAIRED = {'device_type': 'fire_tv', 'device_ip': '192.168.1.112', 'setup_complete': True}
UNPAIRED = {'device_type': 'fire_tv', 'device_ip': '', 'setup_complete': False}


class TestEarlyConnect(unittest.TestCase):

    def test_paired_device_connects_with_no_picture(self):
        o, T = run_setup(PAIRED, delay_seconds=0, known_only=True)
        T.assert_called_once()
        self.assertTrue(o._device_setup_started)

    def test_paired_roku_connects_with_no_picture(self):
        cfg = dict(PAIRED, device_type='roku')
        o, T = run_setup(cfg, delay_seconds=0, known_only=True)
        T.assert_called_once()

    def test_unpaired_device_waits_for_a_picture(self):
        """First-time pairing needs the screen for the ADB 'Allow' dialog."""
        o, T = run_setup(UNPAIRED, delay_seconds=0, known_only=True)
        T.assert_not_called()
        self.assertFalse(getattr(o, '_device_setup_started', False),
                         'must leave the later, picture-gated setup free to run')

    def test_no_device_type_waits(self):
        o, T = run_setup({'device_type': 'none'}, delay_seconds=0, known_only=True)
        T.assert_not_called()

    def test_unreadable_config_waits_rather_than_guessing(self):
        o = object.__new__(m.Minus)
        o.running = True
        with patch('src.device_config.get_device_config_manager',
                   side_effect=RuntimeError('corrupt')), \
             patch('minus.threading.Thread') as T:
            o._start_device_setup_delayed(delay_seconds=0, known_only=True)
        T.assert_not_called()


class TestRunsOnce(unittest.TestCase):

    def test_later_picture_gated_call_does_not_start_a_second_setup(self):
        """Two setup flows would fight over the same ADB connection."""
        o, T = run_setup(PAIRED, delay_seconds=0, known_only=True)
        with patch('minus.threading.Thread') as T2:
            o._start_device_setup_delayed(delay_seconds=0)
        T2.assert_not_called()

    def test_normal_path_unchanged_when_nothing_started_early(self):
        o, T = run_setup(UNPAIRED, delay_seconds=0)
        T.assert_called_once()


class TestWiredIntoStartup(unittest.TestCase):

    def test_early_connect_happens_before_waiting_for_signal(self):
        src = (ROOT / 'minus.py').read_text()
        wait = src.index('# Poll for HDMI signal every 2 seconds')
        early = src.index('_start_device_setup_delayed(delay_seconds=5.0, known_only=True)')
        loop = src.index('while self.running:', wait)
        self.assertLess(wait, early)
        self.assertLess(early, loop, 'must start before the blocking wait loop')


class TestAutonomousStartsBeforeSignal(unittest.TestCase):
    """The remote alone wakes nothing: autonomous mode is what presses Home.

    With only the remote moved ahead of the wait, a boot with the stick asleep
    would sit with a connected controller and nobody to use it, because
    autonomous mode still started after the picture appeared.
    """

    def test_autonomous_starts_before_the_wait_loop(self):
        src = (ROOT / 'minus.py').read_text()
        wait = src.index('# Poll for HDMI signal every 2 seconds')
        loop = src.index('while self.running:', wait)
        block = src[wait:loop]
        self.assertIn('self.autonomous_mode.start_if_enabled()', block)
        self.assertIn('self.autonomous_mode.set_ad_blocker(self)', block)

    def test_normal_start_still_attaches_vlm_and_capture(self):
        src = (ROOT / 'minus.py').read_text()
        tail = src[src.index('# Start night mode if it was enabled'):]
        tail = tail[:tail.index('Minus running - press Ctrl+C')]
        for needed in ('set_vlm(self.vlm)', 'set_frame_capture(self.frame_capture)',
                       'start_if_enabled()'):
            self.assertIn(needed, tail)

    def test_second_start_is_a_no_op(self):
        from autonomous_mode import AutonomousMode
        am = AutonomousMode()
        am._enabled = True
        started = []
        with patch('autonomous_mode.threading.Thread') as T:
            t = MagicMock(); t.is_alive.return_value = True
            T.return_value = t
            am.start_if_enabled()
            am.start_if_enabled()
            started = T.call_count
        self.assertEqual(started, 1, 'a second start must not spawn a second loop')

    def test_screen_query_survives_missing_vlm(self):
        from autonomous_mode import AutonomousMode
        am = AutonomousMode()
        am._vlm = None
        am._frame_capture = None
        self.assertIsNone(am._query_screen())


if __name__ == "__main__":
    unittest.main(verbosity=2)
