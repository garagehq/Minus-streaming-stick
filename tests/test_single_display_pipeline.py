#!/usr/bin/env python3
"""There must only ever be one display pipeline.

When the TV came on, the health monitor's reconnect restart and the display
retry loop rebuilt the pipeline in the same second. The loser was overwritten
in self.pipeline while still PLAYING, and nothing referenced it again, so
nothing ever stopped it. ustreamer showed two clients each pulling 61fps.

Two costs: both decoded 4K at 60fps (double VPU/CPU on a SoC at its thermal
trip), and the thermal frame gate keeps its token bucket on the blocker rather
than per pipeline, so in degraded mode they split the 30fps budget and the
screen showed ~15fps.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import ad_blocker as ab  # noqa: E402
from ad_blocker import DRMAdBlocker  # noqa: E402


def blocker():
    b = object.__new__(DRMAdBlocker)
    b.pipeline = None
    b.bus = None
    b._build_lock = threading.RLock()
    b._framegate_credit = 1.7
    b._framegate_last_ts = 123.0
    built = []

    def fake_build():
        p = MagicMock(name=f'pipeline{len(built)}')
        built.append(p)
        b.pipeline = p
        b.bus = MagicMock()
        return True
    b._build_pipeline = fake_build
    b._built = built
    return b


class TestSinglePipelineInvariant(unittest.TestCase):

    def test_first_build_has_nothing_to_tear_down(self):
        b = blocker()
        b._init_pipeline()
        self.assertEqual(len(b._built), 1)
        b._built[0].set_state.assert_not_called()

    def test_rebuild_stops_the_live_pipeline_first(self):
        b = blocker()
        b._init_pipeline()
        old = b.pipeline
        old_bus = b.bus
        b._init_pipeline()
        old.set_state.assert_called_with(ab.Gst.State.NULL)
        old_bus.remove_signal_watch.assert_called_once()
        self.assertIsNot(b.pipeline, old)

    def test_concurrent_builders_leave_exactly_one_live(self):
        """The production race: two recovery paths building at once."""
        b = blocker()
        barrier = threading.Barrier(8)

        def go():
            barrier.wait()
            b._init_pipeline()
        ts = [threading.Thread(target=go) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        stopped = [p for p in b._built
                   if any(c.args == (ab.Gst.State.NULL,) for c in p.set_state.call_args_list)]
        live = [p for p in b._built if p not in stopped]
        self.assertEqual(len(b._built), 8)
        self.assertEqual(live, [b.pipeline],
                         'every pipeline except the current one must be stopped')

    def test_new_pipeline_gets_a_fresh_frame_budget(self):
        b = blocker()
        b._init_pipeline()
        self.assertEqual(b._framegate_credit, 0.0)
        self.assertIsNone(b._framegate_last_ts)

    def test_teardown_survives_a_failing_pipeline(self):
        b = blocker()
        b._init_pipeline()
        b.pipeline.set_state.side_effect = RuntimeError('gst')
        b._init_pipeline()
        self.assertEqual(len(b._built), 2)


def standalone_blocker():
    """A blocker whose NO SIGNAL / loading screens build mock pipelines."""
    b = object.__new__(DRMAdBlocker)
    b.pipeline = None
    b.bus = None
    b._build_lock = threading.RLock()
    b._restart_lock = threading.Lock()
    b._pipeline_restarting = False
    b.current_source = None
    b.is_visible = False
    b.plane_id, b.connector_id = 192, 231
    for name in ('_set_led_state', '_stop_watchdog_thread', '_stop_loading_animation',
                 '_stop_no_signal_animation', '_start_no_signal_animation',
                 '_start_loading_animation', '_force_hdmi_reinit'):
        setattr(b, name, MagicMock())
    return b


class TestStandaloneScreensAreSerialised(unittest.TestCase):
    """minus, 2026-10-18: the TV came on and three paths started NO SIGNAL
    within a second. One pipeline fell out of self.pipeline still PLAYING: the
    text froze mid-screen, and once the source came up the video pipeline
    stalled on every restart because the plane still showed videotestsrc."""

    def run_concurrently(self, b, calls):
        built = []
        lock = threading.Lock()
        real = ab.Gst.parse_launch

        def fake_parse(desc):
            # Widen the race window the way a real parse/PLAYING does.
            import time
            time.sleep(0.01)
            p = MagicMock(name=f'p{len(built)}')
            p.set_state.return_value = ab.Gst.StateChangeReturn.SUCCESS
            with lock:
                built.append(p)
            return p
        # Don't fork the real modetest: six concurrent probes under full-suite
        # load occasionally fail, and a probe with no connector makes the
        # screen skip its build, so the test then saw no pipeline at all.
        real_probe = ab.probe_drm_output
        ab.probe_drm_output = lambda: {'connector_id': b.connector_id, 'plane_id': b.plane_id}
        ab.Gst.parse_launch = fake_parse
        try:
            barrier = threading.Barrier(len(calls))

            def go(fn):
                barrier.wait()
                fn()
            ts = [threading.Thread(target=go, args=(fn,)) for fn in calls]
            for t in ts: t.start()
            for t in ts: t.join()
        finally:
            ab.Gst.parse_launch = real
            ab.probe_drm_output = real_probe
        return built

    def assert_one_live(self, b, built):
        live = [p for p in built
                if not any(c.args == (ab.Gst.State.NULL,) for c in p.set_state.call_args_list)]
        self.assertEqual(live, [b.pipeline], 'an orphaned pipeline is still PLAYING')

    def test_concurrent_no_signal_starts_leave_one_pipeline(self):
        b = standalone_blocker()
        built = self.run_concurrently(
            b, [lambda: b.start_no_signal_mode(skip_dpms=True)] * 6)
        self.assertEqual(len(built), 6)
        self.assert_one_live(b, built)

    def test_no_signal_racing_loading_leaves_one_pipeline(self):
        b = standalone_blocker()
        built = self.run_concurrently(
            b, [lambda: b.start_no_signal_mode(skip_dpms=True), b.start_loading_mode] * 3)
        self.assert_one_live(b, built)

    def test_restart_does_not_replace_the_no_signal_screen(self):
        b = standalone_blocker()
        b._restart_count = 0
        b._consecutive_failures = 0
        b._base_restart_delay, b._max_restart_delay = 0.0, 0.0
        b._init_pipeline = MagicMock()
        b.current_source = 'no_hdmi_device'
        b._restart_pipeline(penalize=False)
        b._init_pipeline.assert_not_called()


class TestRestoreDuringStartup(unittest.TestCase):

    def test_startup_owns_the_first_display_start(self):
        import minus as m
        mm = object.__new__(m.Minus)
        mm._startup_waiting_for_signal = True
        mm._hdmi_signal_lost = True
        mm.ad_blocker = MagicMock()
        mm._set_led_state = MagicMock()
        mm._restart_ustreamer = MagicMock()
        mm._on_hdmi_restored()
        mm.ad_blocker.start.assert_not_called()
        mm._restart_ustreamer.assert_not_called()
        self.assertFalse(mm._hdmi_signal_lost)


class TestCaptureResumesWithoutATV(unittest.TestCase):
    """minus 2026-10-19: the source slept and woke with the projector off;
    recovery couldn't start the display, kept the audio watchdog paused, and
    capture (ASR, audio gate) stayed stopped for the rest of the night."""

    def test_signal_back_without_a_tv_restarts_capture(self):
        import minus as m
        from unittest.mock import patch
        mm = object.__new__(m.Minus)
        mm._startup_waiting_for_signal = False
        mm._hdmi_signal_lost = True
        mm.ad_blocker = MagicMock()
        mm.ad_blocker.start.return_value = False        # no TV attached
        mm.audio = MagicMock()
        mm._set_led_state = MagicMock()
        mm._restart_ustreamer = MagicMock()
        mm._start_display_retry_loop = MagicMock()
        with patch('minus.time.sleep'):
            mm._on_hdmi_restored()
        mm._start_display_retry_loop.assert_called_once()
        mm.audio.resume_watchdog.assert_called_once()
        mm.audio.unmute.assert_not_called()

class TestCpuPercent(unittest.TestCase):

    def _hm(self):
        from health import HealthMonitor
        return object.__new__(HealthMonitor)

    def _read(self, hm, line):
        from unittest.mock import mock_open, patch
        with patch('builtins.open', mock_open(read_data=line + '\n')):
            return hm._get_cpu_percent()

    def test_first_read_is_baseline_only(self):
        hm = self._hm()
        self.assertEqual(self._read(hm, 'cpu  100 0 100 800 0 0 0 0 0 0'), 0.0)

    def test_delta_between_reads(self):
        hm = self._hm()
        self._read(hm, 'cpu  100 0 100 800 0 0 0 0 0 0')
        # +300 busy, +100 idle -> 75% busy
        self.assertAlmostEqual(self._read(hm, 'cpu  300 0 200 900 0 0 0 0 0 0'), 75.0)

    def test_iowait_counts_as_idle(self):
        hm = self._hm()
        self._read(hm, 'cpu  0 0 0 0 0 0 0 0 0 0')
        self.assertAlmostEqual(self._read(hm, 'cpu  50 0 0 0 50 0 0 0 0 0'), 50.0)

    def test_live_value_is_a_percentage(self):
        hm = self._hm()
        hm._get_cpu_percent()
        import time; time.sleep(0.2)
        v = hm._get_cpu_percent()
        self.assertTrue(0.0 <= v <= 100.0)

    def test_badge_wired(self):
        html = (ROOT / 'src' / 'templates' / 'index.html').read_text()
        self.assertIn('id="cpu-status"', html)
        self.assertIn('status.cpu_percent', html)
        self.assertIn("'cpu_percent'", (ROOT / 'minus.py').read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
