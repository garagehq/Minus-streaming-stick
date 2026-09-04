#!/usr/bin/env python3
"""Tests for thermal-adaptive degradation (src/thermal.py + wiring).

Covers:
- ThermalGovernor hysteresis: sustained-enter, sustained-exit, flapping
  samples don't flap the mode
- sysfs readers (tolerant of missing files; validated against the real
  sysfs when present)
- Minus._on_thermal_change parameter swap/restore round-trip
- ad_blocker framegate: pass-through when off, stable ~30fps rate limit
  when on, instant toggle
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from thermal import ThermalGovernor, read_max_temp_c, read_cpu_throttled


# =========================================================================
# ThermalGovernor hysteresis
# =========================================================================

class TestGovernorEnter(unittest.TestCase):

    def _gov(self):
        return ThermalGovernor(enter_temp=83, exit_temp=75,
                               enter_sustain=15, exit_sustain=60,
                               throttle_min_temp=78)

    def test_starts_normal(self):
        self.assertFalse(self._gov().degraded)

    def test_throttled_must_sustain_before_entering(self):
        g = self._gov()
        self.assertFalse(g.update(80, True, 0))    # hot sample, timer starts
        self.assertFalse(g.degraded)
        self.assertFalse(g.update(80, True, 10))   # 10s < 15s sustain
        self.assertFalse(g.degraded)
        self.assertTrue(g.update(80, True, 15))    # sustained -> enter
        self.assertTrue(g.degraded)

    def test_throttled_but_cool_does_not_enter(self):
        """Regression: RK3588 reports cpufreq cooling state > 0 while merely
        warm. Live logs showed DEGRADED entered at 65.6°C and 67.5°C, which
        is nowhere near heat-limited and just churned the fps cap and the
        blocking thresholds."""
        g = self._gov()
        for t in range(0, 200, 5):
            g.update(66, True, t)
        self.assertFalse(g.degraded)

    def test_throttled_at_floor_enters(self):
        g = self._gov()
        g.update(78, True, 0)
        self.assertTrue(g.update(78, True, 15))

    def test_hot_without_throttle_still_enters(self):
        """The standalone temperature path must still catch a hot SoC whose
        cooling state we failed to read."""
        g = self._gov()
        g.update(84, False, 0)
        self.assertTrue(g.update(84, False, 15))

    def test_throttled_with_unreadable_temp_enters(self):
        """No temperature reading: trust the kernel's throttle signal."""
        g = self._gov()
        g.update(None, True, 0)
        self.assertTrue(g.update(None, True, 15))

    def test_temp_alone_can_enter(self):
        g = self._gov()
        g.update(84, False, 0)
        self.assertTrue(g.update(84, False, 15))
        self.assertTrue(g.degraded)

    def test_brief_throttle_blip_does_not_enter(self):
        g = self._gov()
        g.update(70, True, 0)
        self.assertFalse(g.update(70, False, 10))  # recovered before sustain
        self.assertFalse(g.update(70, True, 20))   # new blip restarts timer
        self.assertFalse(g.update(70, True, 30))   # only 10s into new streak
        self.assertFalse(g.degraded)

    def test_none_temp_with_throttle_still_enters(self):
        g = self._gov()
        g.update(None, True, 0)
        self.assertTrue(g.update(None, True, 15))

    def test_none_temp_not_throttled_stays_normal(self):
        g = self._gov()
        g.update(None, False, 0)
        self.assertFalse(g.update(None, False, 100))
        self.assertFalse(g.degraded)


class TestGovernorExit(unittest.TestCase):

    def _degraded_gov(self):
        g = ThermalGovernor(enter_temp=83, exit_temp=75,
                            enter_sustain=15, exit_sustain=60)
        g.update(85, True, 0)
        g.update(85, True, 15)
        assert g.degraded
        return g

    def test_exit_requires_cool_and_unthrottled_sustained(self):
        g = self._degraded_gov()
        self.assertFalse(g.update(74, False, 100))   # cool, timer starts
        self.assertFalse(g.update(74, False, 150))   # 50s < 60s
        self.assertTrue(g.update(74, False, 160))    # sustained -> exit
        self.assertFalse(g.degraded)

    def test_cool_but_still_throttled_does_not_exit(self):
        g = self._degraded_gov()
        g.update(74, True, 100)
        self.assertFalse(g.update(74, True, 300))
        self.assertTrue(g.degraded)

    def test_unthrottled_but_hot_does_not_exit(self):
        """Cooling states drop to 0 quickly once load shrinks but the SoC is
        still hot — recovery must wait for real cooling."""
        g = self._degraded_gov()
        g.update(80, False, 100)
        self.assertFalse(g.update(80, False, 300))
        self.assertTrue(g.degraded)

    def test_reheat_during_cooldown_restarts_exit_timer(self):
        g = self._degraded_gov()
        g.update(74, False, 100)
        g.update(84, True, 130)     # reheat mid-cooldown
        g.update(74, False, 140)    # cool again, timer restarts
        self.assertFalse(g.update(74, False, 190))  # only 50s into new streak
        self.assertTrue(g.degraded)
        self.assertTrue(g.update(74, False, 200))
        self.assertFalse(g.degraded)

    def test_borderline_flapping_samples_do_not_flap_mode(self):
        """Alternating hot/cool samples every 5s must never toggle the mode."""
        g = self._degraded_gov()
        changed = []
        t = 100
        for i in range(40):
            changed.append(g.update(74 if i % 2 else 84, bool(i % 2), t))
            t += 5
        self.assertTrue(g.degraded)
        self.assertFalse(any(changed))


# =========================================================================
# sysfs readers
# =========================================================================

class TestSysfsReaders(unittest.TestCase):

    def test_read_max_temp_returns_float_or_none(self):
        t = read_max_temp_c()
        if t is not None:
            self.assertIsInstance(t, float)
            self.assertGreater(t, 0)
            self.assertLess(t, 150)

    def test_read_throttled_returns_bool(self):
        self.assertIn(read_cpu_throttled(), (True, False))

    def test_readers_tolerate_missing_sysfs(self):
        with patch('thermal._THERMAL_ZONE_GLOB', '/nonexistent/zone*'), \
             patch('thermal._COOLING_DEVICE_GLOB', '/nonexistent/cool*'):
            self.assertIsNone(read_max_temp_c())
            self.assertFalse(read_cpu_throttled())


# =========================================================================
# Minus parameter swap/restore
# =========================================================================

class TestMinusThermalSwap(unittest.TestCase):

    def _make_minus(self):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.thermal_degraded = False
        m._thermal_saved_params = None
        m.ad_blocker = MagicMock()
        # Production defaults (mirrors __init__)
        m.OCR_STOP_THRESHOLD = 2
        m.VLM_STOP_THRESHOLD = 2
        m.MIN_BLOCKING_DURATION_BASE = 3.0
        m.MIN_BLOCKING_DURATION_FLOOR_OCR = 1.0
        m.MIN_BLOCKING_DURATION_FLOOR_BOTH = 1.5
        m.MIN_BLOCKING_DURATION_FLOOR_VLM = 0.5
        return m

    def test_enter_swaps_all_params(self):
        m = self._make_minus()
        m._on_thermal_change(True, {'temp_c': 85, 'throttled': True})
        self.assertTrue(m.thermal_degraded)
        # Degraded value tracks the base (base 3 + 2) so the "stickier under
        # throttle" offset survived the base moving 2 -> 3 for the flap fix.
        self.assertEqual(m.OCR_STOP_THRESHOLD,
                         type(m).THERMAL_DEGRADED_PARAMS['OCR_STOP_THRESHOLD'])
        self.assertGreater(m.OCR_STOP_THRESHOLD, 3)
        self.assertEqual(m.VLM_STOP_THRESHOLD, 3)
        self.assertEqual(m.MIN_BLOCKING_DURATION_BASE, 5.0)
        self.assertEqual(m.MIN_BLOCKING_DURATION_FLOOR_VLM, 1.5)
        m.ad_blocker.set_thermal_fps_cap.assert_called_once_with(True)

    def test_exit_restores_exact_originals(self):
        m = self._make_minus()
        m._on_thermal_change(True, {'temp_c': 85, 'throttled': True})
        m._on_thermal_change(False, {'temp_c': 70, 'throttled': False})
        self.assertFalse(m.thermal_degraded)
        self.assertEqual(m.OCR_STOP_THRESHOLD, 2)
        self.assertEqual(m.VLM_STOP_THRESHOLD, 2)
        self.assertEqual(m.MIN_BLOCKING_DURATION_BASE, 3.0)
        self.assertEqual(m.MIN_BLOCKING_DURATION_FLOOR_OCR, 1.0)
        self.assertEqual(m.MIN_BLOCKING_DURATION_FLOOR_BOTH, 1.5)
        self.assertEqual(m.MIN_BLOCKING_DURATION_FLOOR_VLM, 0.5)
        self.assertIsNone(m._thermal_saved_params)
        m.ad_blocker.set_thermal_fps_cap.assert_called_with(False)

    def test_double_enter_does_not_clobber_saved_originals(self):
        """Two consecutive enter events must not save the degraded values as
        'originals' (which would make recovery restore degraded params)."""
        m = self._make_minus()
        m._on_thermal_change(True, {})
        m._on_thermal_change(True, {})
        m._on_thermal_change(False, {})
        self.assertEqual(m.OCR_STOP_THRESHOLD, 2)
        self.assertEqual(m.MIN_BLOCKING_DURATION_BASE, 3.0)

    def test_exit_without_enter_is_safe(self):
        m = self._make_minus()
        m._on_thermal_change(False, {})
        self.assertEqual(m.OCR_STOP_THRESHOLD, 2)

    def test_no_ad_blocker_is_safe(self):
        m = self._make_minus()
        m.ad_blocker = None
        m._on_thermal_change(True, {})
        self.assertTrue(m.thermal_degraded)

    def test_params_dict_covers_only_existing_attrs(self):
        """Every key in THERMAL_DEGRADED_PARAMS must be a real Minus
        blocking parameter (guards against typos going silent via setattr)."""
        m = self._make_minus()
        for key in type(m).THERMAL_DEGRADED_PARAMS:
            self.assertTrue(hasattr(m, key), f"unknown param {key}")


# =========================================================================
# ad_blocker framegate
# =========================================================================

class TestFramegate(unittest.TestCase):

    def _make_blocker(self, rate=30.0):
        from ad_blocker import DRMAdBlocker
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        b._thermal_fps_cap = False
        b._framegate_rate = rate
        b._framegate_credit = 0.0
        b._framegate_last_ts = None
        return b

    def _run_stream(self, blocker, fps, seconds):
        """Feed simulated buffers at `fps` for `seconds`; return forwarded count."""
        from ad_blocker import Gst
        forwarded = 0
        n = int(fps * seconds)
        for i in range(n):
            t = i / fps
            with patch('ad_blocker.time.monotonic', return_value=t):
                if blocker._framegate_probe(None, None, None) == Gst.PadProbeReturn.OK:
                    forwarded += 1
        return forwarded

    def test_passthrough_when_cap_disabled(self):
        b = self._make_blocker()
        self.assertEqual(self._run_stream(b, 60, 2), 120)  # every frame passes

    def test_60fps_input_capped_to_30(self):
        b = self._make_blocker()
        b.set_thermal_fps_cap(True)
        out = self._run_stream(b, 60, 10)
        self.assertAlmostEqual(out / 10.0, 30.0, delta=1.5)

    def test_40fps_input_still_yields_30(self):
        """Token bucket converges on the cap even from a 40fps source
        (forwards 3 of every 4 frames) — the fixed-interval gate this
        replaced collapsed to ~20fps here."""
        b = self._make_blocker()
        b.set_thermal_fps_cap(True)
        out = self._run_stream(b, 40, 10)
        self.assertAlmostEqual(out / 10.0, 30.0, delta=1.5)

    def test_jittery_burst_input_still_capped(self):
        """souphttpsrc delivers MJPEG in bursts; average rate must hold ~30
        with no burst amplification (bucket capped at 1 credit)."""
        from ad_blocker import Gst
        b = self._make_blocker()
        b.set_thermal_fps_cap(True)
        forwarded = 0
        t = 0.0
        n = 0
        # 60fps average delivered as pairs: two frames 1ms apart, then a 32.3ms gap
        while t < 10.0:
            for dt in (0.001, 0.0323):
                with patch('ad_blocker.time.monotonic', return_value=t):
                    if b._framegate_probe(None, None, None) == Gst.PadProbeReturn.OK:
                        forwarded += 1
                t += dt
                n += 1
        self.assertAlmostEqual(forwarded / 10.0, 30.0, delta=2.0)

    def test_slow_input_passes_untouched(self):
        """Input already below the cap (25fps) must not be reduced further."""
        b = self._make_blocker()
        b.set_thermal_fps_cap(True)
        out = self._run_stream(b, 25, 10)
        self.assertAlmostEqual(out / 10.0, 25.0, delta=1.0)

    def test_toggle_restores_full_rate(self):
        b = self._make_blocker()
        b.set_thermal_fps_cap(True)
        b.set_thermal_fps_cap(False)
        self.assertEqual(self._run_stream(b, 60, 1), 60)

    def test_set_cap_idempotent(self):
        b = self._make_blocker()
        b.set_thermal_fps_cap(True)
        b._framegate_last_ts = 123.0
        b.set_thermal_fps_cap(True)  # same value: must not reset gate state
        self.assertEqual(b._framegate_last_ts, 123.0)

    def test_cap_rate_configurable(self):
        """MINUS_THERMAL_FPS_CAP sets _framegate_rate at construction; the
        bucket honors whatever rate is configured."""
        b = self._make_blocker(rate=15.0)
        b.set_thermal_fps_cap(True)
        out = self._run_stream(b, 60, 10)
        self.assertAlmostEqual(out / 10.0, 15.0, delta=1.5)

    def test_pipeline_string_contains_framegate(self):
        """The framegate identity must sit between jpegparse and mppjpegdec
        so drops skip VPU decode."""
        from ad_blocker import DRMAdBlocker
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        b._saved_color_settings = {'saturation': 1.0, 'brightness': 0.0,
                                   'contrast': 1.0, 'hue': 0.0}
        b.ustreamer_port = 9090
        b.plane_id = 192
        b.connector_id = 231
        with patch('ad_blocker.Gst') as mock_gst:
            b._init_pipeline()
            launch = mock_gst.parse_launch.call_args[0][0]
        self.assertIn('identity name=framegate', launch)
        self.assertLess(launch.index('jpegparse'), launch.index('framegate'))
        self.assertLess(launch.index('framegate'), launch.index('mppjpegdec'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
