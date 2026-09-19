#!/usr/bin/env python3
"""Audio must not stay on fakesink once the TV is back.

The playback branch falls back to fakesink when HDMI-TX is absent, so HDMI-RX
capture and the ASR tap keep working with the TV off. Nothing rebuilt it when
the TV returned.

This is invisible to every other health check, which is why it survived: the
pipeline is genuinely healthy. Buffers flow, state is PLAYING, the source is
not silent, nothing is muted -- it is playing perfectly into /dev/null. Seen
live after 63h uptime with the TV connected (jack on, ELD ok) and video running
at 51fps, while /api/audio/check reported playback_fakesink=true.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from audio import AudioPassthrough  # noqa: E402
import minus as m  # noqa: E402


def watchdog_decision(on_fakesink, tx_up):
    """The condition the watchdog uses, in isolation."""
    a = object.__new__(AudioPassthrough)
    a._playback_fakesink = on_fakesink
    a._hdmi_tx_connected = lambda: tx_up
    return bool(getattr(a, '_playback_fakesink', False) and a._hdmi_tx_connected())


class TestWatchdogDetectsStrandedFakesink(unittest.TestCase):

    def test_fakesink_with_tv_connected_triggers_restart(self):
        self.assertTrue(watchdog_decision(on_fakesink=True, tx_up=True))

    def test_fakesink_with_tv_absent_is_correct_and_left_alone(self):
        """This is the state fakesink exists for -- rebuilding would loop."""
        self.assertFalse(watchdog_decision(on_fakesink=True, tx_up=False))

    def test_real_sink_is_never_touched(self):
        self.assertFalse(watchdog_decision(on_fakesink=False, tx_up=True))
        self.assertFalse(watchdog_decision(on_fakesink=False, tx_up=False))

    def test_condition_is_present_in_the_watchdog(self):
        src = (ROOT / 'src' / 'audio.py').read_text()
        loop = src[src.index('def _watchdog_loop'):]
        loop = loop[:loop.index('def ', 10)] if 'def ' in loop[10:] else loop
        self.assertIn('_playback_fakesink', loop,
                      'watchdog cannot detect a stranded fakesink')
        self.assertIn('_hdmi_tx_connected', loop)


class TestReprepareRebuildsOffFakesink(unittest.TestCase):
    """The re-prepare used to skip whenever it saw fakesink, which was the
    exact case needing a rebuild: there is no alsasink in the pipeline at all,
    so re-preparing a PCM cannot help -- only reconstruction can."""

    def _run(self, on_fakesink, tx_up):
        o = object.__new__(m.Minus)
        audio = MagicMock()
        audio.is_running = True
        audio._playback_fakesink = on_fakesink
        audio._hdmi_tx_connected.return_value = tx_up
        o.audio = audio
        with patch('minus.threading.Thread') as T:
            o._schedule_audio_reprepare(delay=0, reason='test')
            target = T.call_args.kwargs['target']
        with patch('minus.time.sleep'):
            target()
        return audio.restart.called

    def test_rebuilds_when_on_fakesink_and_tv_is_back(self):
        self.assertTrue(self._run(on_fakesink=True, tx_up=True))

    def test_skips_when_tv_is_genuinely_absent(self):
        self.assertFalse(self._run(on_fakesink=True, tx_up=False))

    def test_still_reprepares_a_real_sink(self):
        """The original purpose: reopen the PCM after a modeset."""
        self.assertTrue(self._run(on_fakesink=False, tx_up=True))

    def test_no_audio_object_is_a_no_op(self):
        o = object.__new__(m.Minus)
        o.audio = None
        o._schedule_audio_reprepare(delay=0, reason='test')

    def test_stopped_audio_is_not_restarted(self):
        o = object.__new__(m.Minus)
        audio = MagicMock()
        audio.is_running = False
        o.audio = audio
        with patch('minus.threading.Thread') as T:
            o._schedule_audio_reprepare(delay=0, reason='test')
            target = T.call_args.kwargs['target']
        with patch('minus.time.sleep'):
            target()
        self.assertFalse(audio.restart.called)


class TestTXProbeFailsOpen(unittest.TestCase):
    def test_unreadable_sysfs_reports_connected(self):
        """Fails open so a probe glitch never downgrades working playback."""
        a = object.__new__(AudioPassthrough)
        with patch('pathlib.Path.glob', side_effect=OSError('boom')):
            self.assertTrue(AudioPassthrough._hdmi_tx_connected(a))


class TestTXReinitOnNormalStartup(unittest.TestCase):
    """The transmitter must be re-initialised before audio opens its PCM.

    The rockchip HDMI-TX driver programs its audio infoframe and clock
    regeneration at PCM prepare time. If the link was not freshly trained the
    lane comes up DEAD, and nothing above ALSA can tell: the PCM runs, hw_ptr
    advances, state is RUNNING, jack and ELD read fine, and the sink is silent.

    Reopening the PCM does not fix it -- measured directly on the box, hw_ptr
    reset and advanced with the sink still silent. Only a DPMS cycle recovers
    it. The reconnect and no-signal paths already did this; normal start-up did
    not, so booting with the display attached gave silent HDMI-TX audio that
    survived every service restart.
    """

    def _blocker(self, attached):
        from ad_blocker import DRMAdBlocker
        b = object.__new__(DRMAdBlocker)
        b._force_hdmi_reinit = MagicMock()
        paths = []
        if attached:
            c = MagicMock()
            c.__truediv__ = lambda self, o: MagicMock(
                exists=lambda: True, read_text=lambda: 'connected\n')
            paths.append(c)
        else:
            c = MagicMock()
            c.__truediv__ = lambda self, o: MagicMock(
                exists=lambda: True, read_text=lambda: 'disconnected\n')
            paths.append(c)
        return b, paths

    def test_reinit_runs_when_a_display_is_attached(self):
        b, paths = self._blocker(attached=True)
        with patch('pathlib.Path.glob', return_value=paths):
            b._force_hdmi_reinit_if_connected()
        b._force_hdmi_reinit.assert_called_once()

    def test_reinit_skipped_with_nothing_attached(self):
        """No link to train, and it would fight the no-signal path."""
        b, paths = self._blocker(attached=False)
        with patch('pathlib.Path.glob', return_value=paths):
            b._force_hdmi_reinit_if_connected()
        b._force_hdmi_reinit.assert_not_called()

    def test_probe_failure_is_not_fatal(self):
        """Video must come up even if the re-init cannot be attempted."""
        from ad_blocker import DRMAdBlocker
        b = object.__new__(DRMAdBlocker)
        b._force_hdmi_reinit = MagicMock(side_effect=RuntimeError('modetest gone'))
        with patch('pathlib.Path.glob', side_effect=OSError('sysfs gone')):
            b._force_hdmi_reinit_if_connected()

    def test_startup_path_calls_it_before_building_the_pipeline(self):
        src = (ROOT / 'src' / 'ad_blocker.py').read_text()
        body = src[src.index('    def start(self):'):]
        body = body[:body.index('\n    def ', 10)]
        self.assertIn('_force_hdmi_reinit_if_connected', body,
                      'normal start-up no longer re-inits the transmitter')
        self.assertLess(body.index('_force_hdmi_reinit_if_connected'),
                        body.index('self._init_pipeline()'),
                        're-init must happen while DRM is still free')


if __name__ == "__main__":
    unittest.main(verbosity=2)
