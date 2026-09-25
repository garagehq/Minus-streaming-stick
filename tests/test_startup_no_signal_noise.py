#!/usr/bin/env python3
"""Starting with no picture should be quiet and should not fight itself.

Found while testing startup with the streaming stick asleep:

1. Every overlay attempt logged an ERROR while ustreamer was not running --
   eight on one startup -- though that is the normal state with no input.
2. With no display attached, the loading screen still built a kmssink
   pipeline that was guaranteed to fail, logging ERRORs each time, while the
   no-signal screen already skipped cleanly in the same situation.
3. The health monitor judges signal only through ustreamer. Minus starts
   ustreamer after it has seen a picture, so for a few seconds after the source
   woke the monitor read "no signal" and forced the NO SIGNAL screen over a
   startup that was succeeding (picture back 19:22:55, declared lost 19:22:58).
"""

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import overlay as ov  # noqa: E402
import health  # noqa: E402
from health import HealthMonitor  # noqa: E402


def refused():
    return urllib.error.URLError(ConnectionRefusedError(111, 'Connection refused'))


class TestOverlayQuietWhenUstreamerDown(unittest.TestCase):

    def _overlay(self):
        o = object.__new__(ov.NotificationOverlay)
        o._api_base = 'http://localhost:9090/overlay/set'
        return o

    def test_refused_logs_one_warning_not_repeated_errors(self):
        o = self._overlay()
        with patch('overlay.urllib.request.urlopen', side_effect=refused()), \
             patch.object(ov.logger, 'error') as err, \
             patch.object(ov.logger, 'warning') as warn:
            for _ in range(8):
                self.assertFalse(o._call_api({'text': 'Connecting...'}))
        err.assert_not_called()
        self.assertEqual(warn.call_count, 1)

    def test_recovery_is_announced_and_rearms_the_warning(self):
        o = self._overlay()
        ok = MagicMock(); ok.__enter__.return_value.status = 200
        with patch('overlay.urllib.request.urlopen', side_effect=[refused(), ok, refused()]), \
             patch.object(ov.logger, 'warning') as warn, patch.object(ov.logger, 'info') as info:
            o._call_api({}); o._call_api({}); o._call_api({})
        self.assertEqual(warn.call_count, 2)
        self.assertTrue(any('back' in str(c) for c in info.call_args_list))

    def test_other_connection_failures_are_still_errors(self):
        o = self._overlay()
        with patch('overlay.urllib.request.urlopen',
                   side_effect=urllib.error.URLError(TimeoutError('timed out'))), \
             patch.object(ov.logger, 'error') as err:
            o._call_api({})
        err.assert_called_once()


class TestNoDisplayNoLoadingPipeline(unittest.TestCase):

    def test_loading_mode_skips_without_a_display(self):
        from ad_blocker import DRMAdBlocker
        b = object.__new__(DRMAdBlocker)
        b.pipeline = None; b.bus = None; b.connector_id = 215; b.plane_id = 192
        b._stop_watchdog_thread = lambda: None
        b._stop_loading_animation = lambda: None
        b._stop_no_signal_animation = lambda: None
        with patch('ad_blocker.probe_drm_output', return_value={}), \
             patch('ad_blocker.Gst') as G:
            self.assertFalse(b.start_loading_mode())
        G.parse_launch.assert_not_called()

    def test_start_without_pipeline_is_a_warning_when_no_display(self):
        from ad_blocker import DRMAdBlocker
        import ad_blocker as ab
        b = object.__new__(DRMAdBlocker)
        b.pipeline = None; b.current_source = 'normal'
        b._stop_loading_animation = lambda: None
        b._stop_no_signal_animation = lambda: None
        with patch.object(DRMAdBlocker, '_hdmi_output_attached', return_value=False), \
             patch.object(ab.logger, 'error') as err:
            self.assertFalse(b.start())
        err.assert_not_called()

    def test_start_without_pipeline_stays_an_error_with_a_display(self):
        from ad_blocker import DRMAdBlocker
        import ad_blocker as ab
        b = object.__new__(DRMAdBlocker)
        b.pipeline = None; b.current_source = 'normal'
        b._stop_loading_animation = lambda: None
        b._stop_no_signal_animation = lambda: None
        with patch.object(DRMAdBlocker, '_hdmi_output_attached', return_value=True), \
             patch.object(ab.logger, 'error') as err:
            b.start()
        err.assert_called_once()

    def test_display_probe_fails_open(self):
        from ad_blocker import DRMAdBlocker
        with patch('pathlib.Path.glob', side_effect=OSError('sysfs')):
            self.assertTrue(DRMAdBlocker._hdmi_output_attached())


class TestNoFalseSignalLossDuringStartup(unittest.TestCase):

    def _hm(self, startup_complete, receiver):
        hm = object.__new__(HealthMonitor)
        hm.minus = MagicMock()
        hm.minus.config.ustreamer_port = 9090
        hm.minus._startup_complete = startup_complete
        hm.minus.check_hdmi_signal.return_value = receiver
        hm._hdmi_fps_zero_since = 0
        return hm

    def test_picture_present_while_ustreamer_starts_is_not_signal_loss(self):
        hm = self._hm(startup_complete=False, receiver=(3840, 2160, 59.94))
        with patch('urllib.request.urlopen', side_effect=refused()):
            self.assertEqual(hm._check_hdmi_signal(), (True, '3840x2160'))

    def test_genuinely_no_picture_during_startup_is_still_loss(self):
        hm = self._hm(startup_complete=False, receiver=None)
        with patch('urllib.request.urlopen', side_effect=refused()):
            self.assertEqual(hm._check_hdmi_signal(), (False, ''))

    def test_after_startup_behaviour_is_unchanged(self):
        """Once running, an unreachable ustreamer means what it always did."""
        hm = self._hm(startup_complete=True, receiver=(3840, 2160, 59.94))
        with patch('urllib.request.urlopen', side_effect=refused()):
            self.assertEqual(hm._check_hdmi_signal(), (False, ''))
        hm.minus.check_hdmi_signal.assert_not_called()

    def test_hung_ustreamer_does_not_trigger_the_receiver_query(self):
        """A timeout may mean an active stream the ioctl could disturb."""
        hm = self._hm(startup_complete=False, receiver=(3840, 2160, 59.94))
        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.URLError(TimeoutError('timed out'))):
            self.assertEqual(hm._check_hdmi_signal(), (False, ''))
        hm.minus.check_hdmi_signal.assert_not_called()

    def test_startup_flag_lifecycle(self):
        src = (ROOT / 'minus.py').read_text()
        self.assertIn('self._startup_complete = False', src)
        self.assertLess(src.index('self._startup_complete = True'),
                        src.index('logger.info("Minus running - press Ctrl+C to stop")'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
