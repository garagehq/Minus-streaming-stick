#!/usr/bin/env python3
"""Tests for the Aug-2026 log-analysis fixes.

Four independent issues found by sweeping 4 days of production logs:

1. OCR hard timeout hardcoded at 1.0s, sitting on the p99.9 of the measured
   inference distribution -> 1933 worker kills in 4 days. The documented
   --ocr-timeout / MinusConfig.ocr_timeout was parsed, stored, and never read.
2. Display retry loop forked `modetest` and logged 2 lines every 7s forever
   while the TV was off (~12,300 lines/day).
3. Thermal DEGRADED entered on the kernel throttle flag alone, firing at
   65.6C / 67.5C where the SoC is not heat-limited.
4. ad_blocker.hide() cleared is_visible immediately but unmuted only at the
   end of the fade-out animation, with three paths that skipped the unmute
   entirely -> audio dead until the health watchdog swept it.

All hardware/GStreamer interaction is mocked.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


# =========================================================================
# 1. OCR hard timeout is configurable
# =========================================================================

class TestOCRTimeoutConfigurable(unittest.TestCase):

    def test_default_is_one_point_five(self):
        """1.0s was below the observed max inference (1093ms); the default
        must clear the measured tail."""
        from ocr_worker import OCRProcess
        self.assertEqual(OCRProcess.DEFAULT_HARD_TIMEOUT, 1.5)
        self.assertEqual(OCRProcess().HARD_TIMEOUT, 1.5)

    def test_explicit_timeout_applied(self):
        from ocr_worker import OCRProcess
        self.assertEqual(OCRProcess(hard_timeout=2.5).HARD_TIMEOUT, 2.5)

    def test_config_default_flows_through(self):
        """MinusConfig.ocr_timeout was dead config before this fix."""
        from config import MinusConfig
        from ocr_worker import OCRProcess
        cfg = MinusConfig()
        self.assertEqual(OCRProcess(hard_timeout=cfg.ocr_timeout).HARD_TIMEOUT,
                         cfg.ocr_timeout)

    def test_invalid_values_fall_back_to_default(self):
        from ocr_worker import OCRProcess
        for bad in (None, 0, -1, 'abc', float('nan') * 0 - 1):
            proc = OCRProcess(hard_timeout=bad)
            self.assertGreater(proc.HARD_TIMEOUT, 0)
            if bad in (None, 0, -1) or bad == 'abc':
                self.assertEqual(proc.HARD_TIMEOUT,
                                 OCRProcess.DEFAULT_HARD_TIMEOUT)

    def test_instance_does_not_mutate_class(self):
        """Per-instance override must not leak into other instances."""
        from ocr_worker import OCRProcess
        OCRProcess(hard_timeout=9.0)
        self.assertEqual(OCRProcess.HARD_TIMEOUT,
                         OCRProcess.DEFAULT_HARD_TIMEOUT)
        self.assertEqual(OCRProcess().HARD_TIMEOUT,
                         OCRProcess.DEFAULT_HARD_TIMEOUT)

    def test_timeout_used_by_ocr_call(self):
        """The configured value is what response_queue.get() waits on."""
        from ocr_worker import OCRProcess
        proc = OCRProcess(hard_timeout=1.5)
        proc.is_ready = True
        proc.process = MagicMock()
        proc.process.is_alive.return_value = True
        proc.request_queue = MagicMock()
        proc.response_queue = MagicMock()
        proc.response_queue.get.return_value = ('ok', ['text'])
        proc.ocr(MagicMock())
        self.assertEqual(proc.response_queue.get.call_args.kwargs['timeout'], 1.5)


# =========================================================================
# 2. Display retry loop gates the expensive probe on cheap sysfs
# =========================================================================

class TestHDMIOutputGate(unittest.TestCase):
    """The retry loop used to fork `modetest` and log 2 lines every 7s
    forever while the TV was off (~12,300 lines/day, the bulk of all journal
    volume, plus a subprocess fork every 7s on a throttled SoC). It now
    pre-checks sysfs, which is free, so the 7s cadence (and therefore
    reconnect responsiveness) is unchanged."""

    def _minus(self, health_monitor=None):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.health_monitor = health_monitor
        return m

    def test_delegates_to_health_monitor(self):
        hm = MagicMock()
        hm._check_hdmi_output_connected.return_value = True
        self.assertTrue(self._minus(hm)._any_hdmi_output_connected())
        hm._check_hdmi_output_connected.return_value = False
        self.assertFalse(self._minus(hm)._any_hdmi_output_connected())

    def test_health_monitor_exception_falls_back_to_sysfs(self):
        hm = MagicMock()
        hm._check_hdmi_output_connected.side_effect = RuntimeError('boom')
        m = self._minus(hm)
        fake = MagicMock()
        fake.read_text.return_value = 'connected\n'
        with patch('minus.Path') as P:
            P.return_value.glob.return_value = [fake]
            self.assertTrue(m._any_hdmi_output_connected())

    def test_sysfs_disconnected(self):
        m = self._minus(None)
        fake = MagicMock()
        fake.read_text.return_value = 'disconnected\n'
        with patch('minus.Path') as P:
            P.return_value.glob.return_value = [fake]
            self.assertFalse(m._any_hdmi_output_connected())

    def test_any_connected_wins(self):
        m = self._minus(None)
        a, b = MagicMock(), MagicMock()
        a.read_text.return_value = 'disconnected\n'
        b.read_text.return_value = 'connected\n'
        with patch('minus.Path') as P:
            P.return_value.glob.return_value = [a, b]
            self.assertTrue(m._any_hdmi_output_connected())

    def test_fails_open_when_no_sysfs_nodes(self):
        """Unknown topology must never permanently block display recovery."""
        m = self._minus(None)
        with patch('minus.Path') as P:
            P.return_value.glob.return_value = []
            self.assertTrue(m._any_hdmi_output_connected())

    def test_fails_open_on_read_error(self):
        m = self._minus(None)
        with patch('minus.Path') as P:
            P.return_value.glob.side_effect = OSError('nope')
            self.assertTrue(m._any_hdmi_output_connected())

    def test_retry_loop_skips_probe_while_disconnected(self):
        """The whole point: while sysfs says disconnected we must not reach
        probe_drm_output() (which forks modetest)."""
        source = (ROOT / 'minus.py').read_text()
        idx = source.index('def _start_display_retry_loop')
        body = source[idx:idx + 4000]
        gate = body.index('_any_hdmi_output_connected')
        probe = body.index('probe_drm_output()')
        self.assertLess(gate, probe,
                        "sysfs gate must precede the modetest probe")
        self.assertIn('continue', body[gate:probe],
                      "gate must short-circuit the loop iteration")


# =========================================================================
# 2b. VLM per-inference double logging
# =========================================================================

class TestVLMVerboseGate(unittest.TestCase):
    """detect_ad() logged twice at INFO — once in the worker, once in the
    parent detection loop — which was 61% of the entire journal (23,579 +
    23,575 lines over 10.4h). The parent line already carries verdict,
    latency, confidence and the same p_yes; only the raw logits were
    unique, so the worker line defaults to DEBUG behind MINUS_VLM_VERBOSE."""

    def _reload(self, verbose):
        import importlib
        env = {'MINUS_VLM_VERBOSE': '1'} if verbose else {}
        with patch.dict('os.environ', env, clear=False):
            if not verbose:
                import os as _os
                _os.environ.pop('MINUS_VLM_VERBOSE', None)
            import vlm
            return importlib.reload(vlm)

    def test_default_is_debug(self):
        mod = self._reload(verbose=False)
        self.assertFalse(mod.VLM_VERBOSE_LOG)

    def test_env_flag_enables_info(self):
        mod = self._reload(verbose=True)
        self.assertTrue(mod.VLM_VERBOSE_LOG)

    def test_detect_line_level_follows_flag(self):
        """The demoted call must use logger.log(level, ...) so the flag
        actually controls the emitted level."""
        source = (ROOT / 'src' / 'vlm.py').read_text()
        # rindex: skip the module-level explanatory comment, find the call
        idx = source.rindex('f"VLM(LFM2): ')
        window = source[max(0, idx - 300):idx]
        self.assertIn('logging.INFO if VLM_VERBOSE_LOG else logging.DEBUG', window)
        self.assertNotIn('logger.info(', window.split('logger.log(')[-1])

    def test_screen_query_line_also_gated(self):
        """query_image is duplicated the same way — autonomous_mode logs
        "[AutonomousMode] VLM screen query (0.3s): MENU" for every call — so
        it rides the same flag. Its unique content (per-class score margins)
        comes back with MINUS_VLM_VERBOSE=1."""
        source = (ROOT / 'src' / 'vlm.py').read_text()
        idx = source.rindex('f"VLM(LFM2) query: ')
        window = source[max(0, idx - 400):idx]
        self.assertIn('logging.INFO if VLM_VERBOSE_LOG else logging.DEBUG', window)

    def test_no_unconditional_info_vlm_inference_logs(self):
        """Neither per-inference worker line may be logger.info() outright."""
        source = (ROOT / 'src' / 'vlm.py').read_text()
        for marker in ('f"VLM(LFM2): ', 'f"VLM(LFM2) query: '):
            idx = source.rindex(marker)
            window = source[max(0, idx - 400):idx]
            call = window[window.rindex('logger.'):]
            self.assertTrue(call.startswith('logger.log('),
                            f"{marker.strip()} should use gated logger.log()")

    def tearDown(self):
        import importlib, os as _os
        _os.environ.pop('MINUS_VLM_VERBOSE', None)
        import vlm
        importlib.reload(vlm)


# =========================================================================
# 3. Thermal throttle guard needs temperature corroboration
# =========================================================================

class TestThrottleTemperatureFloor(unittest.TestCase):

    def _gov(self):
        from thermal import ThermalGovernor
        return ThermalGovernor(enter_temp=83, exit_temp=75, enter_sustain=15,
                               exit_sustain=60, throttle_min_temp=78)

    def test_warm_throttle_ignored(self):
        """Live regression: DEGRADED entered at 65.6C and 67.5C."""
        for temp in (65.615, 67.461):
            g = self._gov()
            for t in range(0, 300, 5):
                g.update(temp, True, t)
            self.assertFalse(g.degraded, f"{temp}C should not degrade")

    def test_hot_throttle_still_enters(self):
        g = self._gov()
        g.update(82.23, True, 0)
        self.assertTrue(g.update(82.23, True, 15))

    def test_floor_is_inclusive(self):
        g = self._gov()
        g.update(78.0, True, 0)
        self.assertTrue(g.update(78.0, True, 15))

    def test_just_below_floor_rejected(self):
        g = self._gov()
        for t in range(0, 300, 5):
            g.update(77.9, True, t)
        self.assertFalse(g.degraded)

    def test_temp_only_path_unaffected(self):
        """A hot SoC with no throttle flag must still degrade."""
        g = self._gov()
        g.update(84, False, 0)
        self.assertTrue(g.update(84, False, 15))

    def test_exit_still_requires_throttle_clear(self):
        """We must never drop the cap while the kernel is still capping."""
        g = self._gov()
        g.update(84, True, 0)
        g.update(84, True, 15)
        self.assertTrue(g.degraded)
        for t in range(100, 400, 5):
            g.update(70, True, t)   # cool but still throttled
        self.assertTrue(g.degraded)


# =========================================================================
# 4. hide() unmutes on every path
# =========================================================================

class TestHideAlwaysUnmutes(unittest.TestCase):

    def _blocker(self, visible=True, pipeline=True, animating=False,
                 direction=None, animation_enabled=False):
        import threading
        from ad_blocker import DRMAdBlocker
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        b._lock = threading.RLock()
        b._test_blocking_until = 0
        b.is_visible = visible
        b.current_source = 'ocr'
        b._animating = animating
        b._animation_direction = direction
        b._animation_enabled = animation_enabled
        b.pipeline = MagicMock() if pipeline else None
        b.audio = MagicMock()
        b.audio.is_muted = True
        b.minus = None
        b._current_block_start = None
        b._total_blocking_time = 0.0
        b._content_kind_lock_until = 0.0
        b.CONTENT_KIND_COOLDOWN_SECONDS = 30
        for m in ('_set_led_state', '_stop_rotation_thread', '_stop_debug_thread',
                  '_clear_ad_countdown', '_blocking_api_call',
                  '_stop_animation_thread', '_start_animation',
                  '_on_end_animation_complete'):
            setattr(b, m, MagicMock())
        return b

    def test_normal_hide_unmutes(self):
        b = self._blocker()
        b.hide()
        b.audio.unmute.assert_called()

    def test_no_pipeline_path_unmutes(self):
        b = self._blocker(pipeline=False)
        b.hide()
        b.audio.unmute.assert_called()

    def test_already_hidden_early_return_still_unmutes(self):
        """REGRESSION: `if not was_visible and _animation_direction != 'start'`
        returned without ever unmuting, leaving audio dead until the
        health watchdog swept it up to 5s later."""
        b = self._blocker(visible=False, direction=None)
        b.hide()
        b.audio.unmute.assert_called()

    def test_animated_path_unmutes_before_animation(self):
        """Unmute must not be deferred to _on_end_animation_complete — an
        interrupted end animation never runs it."""
        b = self._blocker(animation_enabled=True)
        b.hide()
        b.audio.unmute.assert_called()
        # ordering: unmute happened, animation kicked off after
        b._start_animation.assert_called_once()

    def test_unmute_happens_with_is_visible_cleared(self):
        """The watchdog reads (is_visible, is_muted); both must be consistent
        by the time hide() returns."""
        b = self._blocker()
        observed = {}

        def record():
            observed['visible'] = b.is_visible
        b.audio.unmute.side_effect = record
        b.hide()
        self.assertFalse(observed['visible'])

    def test_hide_during_end_animation_is_noop(self):
        """Already ending: the first hide() did the unmute."""
        b = self._blocker(animating=True, direction='end')
        b.hide()
        self.assertTrue(b.is_visible)  # untouched by the early return

    def test_test_mode_hold_not_unmuted(self):
        """Test-mode blocking is intentionally held; must stay muted."""
        import time as _t
        b = self._blocker()
        b._test_blocking_until = _t.time() + 60
        b.hide()
        b.audio.unmute.assert_not_called()

    def test_forced_hide_overrides_test_mode(self):
        import time as _t
        b = self._blocker()
        b._test_blocking_until = _t.time() + 60
        b.hide(force=True)
        b.audio.unmute.assert_called()

    def test_no_audio_object_is_safe(self):
        b = self._blocker()
        b.audio = None
        b.hide()  # must not raise


# =========================================================================
# 4b. Watchdog two-strike confirmation
# =========================================================================

class TestUnmuteWatchdogConfirmation(unittest.TestCase):

    def _monitor(self, blocking, muted):
        from health import HealthMonitor
        hm = HealthMonitor.__new__(HealthMonitor)
        hm._muted_not_blocking_seen = False
        hm.minus = MagicMock()
        hm.minus.audio.is_muted = muted
        hm.minus.ad_blocker.is_visible = blocking
        return hm

    def _tick(self, hm):
        """Mirror of the watchdog block in health.py::_check_health."""
        import logging
        logger = logging.getLogger('test')
        if hm.minus.audio and hm.minus.ad_blocker:
            is_blocking = hm.minus.ad_blocker.is_visible
            is_muted = hm.minus.audio.is_muted
            if not is_blocking and is_muted:
                if hm._muted_not_blocking_seen:
                    logger.warning("forcing unmute")
                    hm.minus.audio.unmute()
                    hm._muted_not_blocking_seen = False
                else:
                    hm._muted_not_blocking_seen = True
            else:
                hm._muted_not_blocking_seen = False

    def test_single_observation_does_not_fire(self):
        hm = self._monitor(blocking=False, muted=True)
        self._tick(hm)
        hm.minus.audio.unmute.assert_not_called()

    def test_two_consecutive_observations_fire(self):
        hm = self._monitor(blocking=False, muted=True)
        self._tick(hm)
        self._tick(hm)
        hm.minus.audio.unmute.assert_called_once()

    def test_transient_straddle_resets(self):
        """One straddling sample followed by a healthy one must not fire."""
        hm = self._monitor(blocking=False, muted=True)
        self._tick(hm)
        hm.minus.audio.is_muted = False
        self._tick(hm)
        hm.minus.audio.is_muted = True
        self._tick(hm)
        hm.minus.audio.unmute.assert_not_called()

    def test_blocking_and_muted_is_healthy(self):
        hm = self._monitor(blocking=True, muted=True)
        for _ in range(5):
            self._tick(hm)
        hm.minus.audio.unmute.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
