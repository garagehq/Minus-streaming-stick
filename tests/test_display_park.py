"""Display pipeline parks while the TV is off, and comes back with it.

With the TV off the display pipeline kept decoding 4K for nobody (~80% of a
core plus the VPU), holding minus at 78-84 C vs ~65 C with no display. All
hardware/GStreamer interaction is mocked.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


def _blocker():
    from ad_blocker import DRMAdBlocker
    b = DRMAdBlocker.__new__(DRMAdBlocker)
    b._build_lock = threading.RLock()
    b._restart_lock = threading.Lock()
    b._pipeline_restarting = False
    b.parked = False
    b.pipeline = MagicMock()
    b.bus = MagicMock()
    b.current_source = None
    b._last_buffer_time = 123.0
    b._stop_watchdog_thread = MagicMock()
    b._stop_loading_animation = MagicMock()
    b._stop_no_signal_animation = MagicMock()
    return b


class TestAdBlockerPark(unittest.TestCase):

    def test_park_tears_down_and_stops_watchdog(self):
        b = _blocker()
        pipeline = b.pipeline
        self.assertTrue(b.park())
        self.assertTrue(b.parked)
        self.assertIsNone(b.pipeline)
        b._stop_watchdog_thread.assert_called_once()
        pipeline.set_state.assert_called_once()

    def test_park_twice_is_a_noop(self):
        b = _blocker()
        b.park()
        self.assertFalse(b.park())

    def test_watchdog_restart_skipped_while_parked(self):
        b = _blocker()
        b.park()
        b._restart_count = 0
        b._consecutive_failures = 0
        b._base_restart_delay = 0.0
        b._max_restart_delay = 0.0
        b._restart_pipeline(penalize=False)
        self.assertIsNone(b.pipeline)
        self.assertTrue(b.parked)
        self.assertFalse(b._pipeline_restarting)

    def test_start_rebuilds_a_parked_pipeline_with_hdmi_reinit(self):
        b = _blocker()
        b.park()
        order = []
        b._force_hdmi_reinit_if_connected = lambda: order.append('reinit')

        def init():
            order.append('init')
            b.pipeline = MagicMock()
        b._init_pipeline = init
        b._start_watchdog = MagicMock()
        b._set_led_state = MagicMock()
        b._update_frame_resolution = MagicMock()
        b._apply_saved_color_settings = MagicMock()
        b._consecutive_failures = 0
        self.assertTrue(b._start_locked())
        self.assertEqual(order, ['reinit', 'init'])
        self.assertFalse(b.parked)
        b._start_watchdog.assert_called_once()


class TestMinusPark(unittest.TestCase):

    def _minus(self, tv_connected=False, display_connected=True):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.ad_blocker = MagicMock()
        m.ad_blocker.park.return_value = True
        m.display_connected = display_connected
        m.display_error = None
        m._any_hdmi_output_connected = MagicMock(return_value=tv_connected)
        m._start_display_retry_loop = MagicMock()
        return m

    def test_parks_and_arms_retry_loop(self):
        m = self._minus()
        self.assertTrue(m.park_display_for_tv_off())
        self.assertFalse(m.display_connected)
        m._start_display_retry_loop.assert_called_once()

    def test_not_when_tv_came_back(self):
        m = self._minus(tv_connected=True)
        self.assertFalse(m.park_display_for_tv_off())
        m.ad_blocker.park.assert_not_called()
        self.assertTrue(m.display_connected)

    def test_not_when_display_already_down(self):
        m = self._minus(display_connected=False)
        self.assertFalse(m.park_display_for_tv_off())
        m.ad_blocker.park.assert_not_called()


class TestRecoveryWithoutDisplay(unittest.TestCase):
    """OCR, VLM and autonomous mode read ustreamer, so it must be kept
    alive and on the right format while the TV is off (parked or never on)."""

    def test_format_change_restarts_ustreamer_without_display(self):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.display_connected = False
        m.ad_blocker = MagicMock()
        m.audio = MagicMock()
        m._restart_ustreamer = MagicMock()
        m._on_format_change('NV12@1920x1080')
        m._restart_ustreamer.assert_called_once_with(even_without_display=True)
        m.ad_blocker.start_loading_mode.assert_not_called()
        m.ad_blocker.start.assert_not_called()

    def test_ustreamer_alive_ignores_commands_that_mention_it(self):
        import re
        pat = re.compile(r'^\S*ustreamer')
        self.assertTrue(pat.search('/home/radxa/ustreamer-patched --device=/dev/video0'))
        self.assertFalse(pat.search("/bin/bash -c journalctl | grep 'ustreamer not responding'"))
        self.assertFalse(pat.search('ugrep -c ustreamer'))


class TestHealthParkTimer(unittest.TestCase):

    def _hm(self):
        from health import HealthMonitor
        hm = HealthMonitor.__new__(HealthMonitor)
        hm.DISPLAY_PARK_AFTER_S = 60.0
        hm._hdmi_output_off_since = 1000.0
        hm.minus = MagicMock()
        hm.minus.park_display_for_tv_off.return_value = True
        return hm

    def test_waits_for_the_delay(self):
        hm = self._hm()
        self.assertFalse(hm._maybe_park_display(False, now=1059.0))
        hm.minus.park_display_for_tv_off.assert_not_called()

    def test_parks_after_the_delay_once(self):
        hm = self._hm()
        self.assertTrue(hm._maybe_park_display(False, now=1061.0))
        self.assertFalse(hm._maybe_park_display(False, now=1200.0))
        hm.minus.park_display_for_tv_off.assert_called_once()

    def test_tv_back_resets_timer(self):
        hm = self._hm()
        hm._maybe_park_display(True, now=1030.0)
        self.assertIsNone(hm._hdmi_output_off_since)
        self.assertFalse(hm._maybe_park_display(False, now=1100.0))

    def test_no_timer_without_an_observed_disconnect(self):
        hm = self._hm()
        hm._hdmi_output_off_since = None
        self.assertFalse(hm._maybe_park_display(False, now=99999.0))


if __name__ == '__main__':
    unittest.main()
