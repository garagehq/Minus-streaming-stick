#!/usr/bin/env python3
"""Comprehensive tests for the color-settings pipeline (saturation /
brightness / contrast / hue).

Covers the Aug 2026 realtime fix end to end:
- _merge_color_settings clamping + persistence
- _color_settings_neutral epsilon behavior
- set_color_settings: all four branches (no pipeline, no element + neutral,
  no element + non-neutral, element present live-update)
- get_color_settings fallback order (element -> saved -> defaults)
- _init_pipeline launch-string baking: saved values baked into videobalance,
  element OMITTED when neutral (zero-copy 60fps path), skip-vsync=true kept
- _restart_pipeline penalize semantics: color rebuilds must not increment
  the failure counter, sleep the backoff, or trip the pkill-ustreamer reset

All GStreamer / subprocess / thread interaction is mocked - no hardware.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from ad_blocker import DRMAdBlocker

NEUTRAL = {'saturation': 1.0, 'brightness': 0.0, 'contrast': 1.0, 'hue': 0.0}


def make_blocker(saved=None, current_source=None, element=None, pipeline='auto'):
    """Bare DRMAdBlocker with just the attrs the color path touches."""
    b = DRMAdBlocker.__new__(DRMAdBlocker)
    b._saved_color_settings = dict(saved) if saved else dict(NEUTRAL)
    b._save_color_settings = MagicMock()
    b.current_source = current_source
    b.restart = MagicMock()
    if pipeline == 'auto':
        b.pipeline = MagicMock()
        b.pipeline.get_by_name.return_value = element
    else:
        b.pipeline = pipeline
    return b


def make_element(values=None):
    """Mock videobalance element whose get_property returns real floats."""
    vals = dict(values or NEUTRAL)
    el = MagicMock()
    el.get_property.side_effect = lambda prop: vals[prop]
    el.set_property.side_effect = lambda prop, v: vals.__setitem__(prop, v)
    return el


# =========================================================================
# _merge_color_settings
# =========================================================================

class TestMergeColorSettings(unittest.TestCase):

    def test_saturation_clamped_high_and_low(self):
        b = make_blocker()
        self.assertEqual(b._merge_color_settings(saturation=3.0)['saturation'], 2.0)
        self.assertEqual(b._merge_color_settings(saturation=-1.0)['saturation'], 0.0)

    def test_brightness_clamped_high_and_low(self):
        b = make_blocker()
        self.assertEqual(b._merge_color_settings(brightness=2.0)['brightness'], 1.0)
        self.assertEqual(b._merge_color_settings(brightness=-2.0)['brightness'], -1.0)

    def test_contrast_clamped_high_and_low(self):
        b = make_blocker()
        self.assertEqual(b._merge_color_settings(contrast=5.0)['contrast'], 2.0)
        self.assertEqual(b._merge_color_settings(contrast=-0.5)['contrast'], 0.0)

    def test_hue_clamped_high_and_low(self):
        b = make_blocker()
        self.assertEqual(b._merge_color_settings(hue=1.5)['hue'], 1.0)
        self.assertEqual(b._merge_color_settings(hue=-1.5)['hue'], -1.0)

    def test_partial_update_preserves_other_values(self):
        b = make_blocker(saved={'saturation': 1.3, 'brightness': 0.1,
                                'contrast': 0.9, 'hue': 0.05})
        merged = b._merge_color_settings(saturation=1.5)
        self.assertEqual(merged['saturation'], 1.5)
        self.assertEqual(merged['brightness'], 0.1)
        self.assertEqual(merged['contrast'], 0.9)
        self.assertEqual(merged['hue'], 0.05)

    def test_all_none_is_noop_merge(self):
        b = make_blocker(saved={'saturation': 1.2, 'brightness': 0.0,
                                'contrast': 1.0, 'hue': 0.0})
        merged = b._merge_color_settings()
        self.assertEqual(merged['saturation'], 1.2)
        b._save_color_settings.assert_called_once()

    def test_persists_and_updates_saved(self):
        b = make_blocker()
        merged = b._merge_color_settings(contrast=0.8)
        b._save_color_settings.assert_called_once_with(merged)
        self.assertEqual(b._saved_color_settings['contrast'], 0.8)

    def test_saved_copy_not_aliased_to_return_value(self):
        b = make_blocker()
        merged = b._merge_color_settings(saturation=1.4)
        merged['saturation'] = 99.0  # caller mutation must not corrupt saved
        self.assertEqual(b._saved_color_settings['saturation'], 1.4)

    def test_missing_saved_falls_back_to_defaults(self):
        b = make_blocker()
        b._saved_color_settings = None
        merged = b._merge_color_settings(hue=0.2)
        self.assertEqual(merged['saturation'], 1.0)
        self.assertEqual(merged['hue'], 0.2)

    def test_string_input_coerced_to_float(self):
        b = make_blocker()
        merged = b._merge_color_settings(saturation="1.25")
        self.assertEqual(merged['saturation'], 1.25)


# =========================================================================
# _color_settings_neutral
# =========================================================================

class TestColorSettingsNeutral(unittest.TestCase):

    def test_exact_neutral_true(self):
        b = make_blocker()
        self.assertTrue(b._color_settings_neutral(dict(NEUTRAL)))

    def test_sub_epsilon_deviation_still_neutral(self):
        b = make_blocker()
        s = dict(NEUTRAL)
        s['saturation'] = 1.0 + 1e-9
        self.assertTrue(b._color_settings_neutral(s))

    def test_each_param_individually_non_neutral(self):
        b = make_blocker()
        for key, value in [('saturation', 1.1), ('brightness', 0.05),
                           ('contrast', 0.9), ('hue', -0.1)]:
            s = dict(NEUTRAL)
            s[key] = value
            self.assertFalse(b._color_settings_neutral(s), f"{key} should break neutrality")

    def test_uses_saved_when_no_arg(self):
        b = make_blocker(saved={'saturation': 1.3, 'brightness': 0.0,
                                'contrast': 1.0, 'hue': 0.0})
        self.assertFalse(b._color_settings_neutral())
        b._saved_color_settings = dict(NEUTRAL)
        self.assertTrue(b._color_settings_neutral())

    def test_empty_settings_default_neutral(self):
        b = make_blocker()
        self.assertTrue(b._color_settings_neutral({}))


# =========================================================================
# set_color_settings - branch: no pipeline at all
# =========================================================================

class TestSetColorNoPipeline(unittest.TestCase):

    def test_persists_and_succeeds(self):
        b = make_blocker(pipeline=None)
        result = b.set_color_settings(saturation=1.2)
        self.assertTrue(result['success'])
        self.assertEqual(result['saturation'], 1.2)
        b._save_color_settings.assert_called_once()

    def test_no_restart_attempted(self):
        b = make_blocker(pipeline=None)
        b.set_color_settings(brightness=0.3)
        b.restart.assert_not_called()

    def test_values_available_at_next_init(self):
        """The persisted values are what _init_pipeline will bake in."""
        b = make_blocker(pipeline=None)
        b.set_color_settings(saturation=1.2, contrast=0.8)
        self.assertEqual(b._saved_color_settings['saturation'], 1.2)
        self.assertEqual(b._saved_color_settings['contrast'], 0.8)


# =========================================================================
# set_color_settings - branch: pipeline without colorbalance element
# =========================================================================

class TestSetColorNoElement(unittest.TestCase):

    def test_still_neutral_no_rebuild(self):
        b = make_blocker()
        result = b.set_color_settings(saturation=1.0)
        self.assertTrue(result['success'])
        b.restart.assert_not_called()

    def test_non_neutral_rebuilds_without_penalty(self):
        b = make_blocker()
        result = b.set_color_settings(saturation=1.3)
        self.assertTrue(result['success'])
        b.restart.assert_called_once_with(penalize=False)

    def test_persist_happens_before_rebuild(self):
        """Ordering matters: _init_pipeline reads _saved_color_settings, so
        the save must land before restart is kicked off."""
        b = make_blocker()
        order = []
        b._save_color_settings.side_effect = lambda s: order.append('save')
        b.restart.side_effect = lambda **kw: order.append('restart')
        b.set_color_settings(saturation=1.3)
        self.assertEqual(order, ['save', 'restart'])

    def test_no_signal_mode_persist_only(self):
        b = make_blocker(current_source='no_hdmi_device')
        result = b.set_color_settings(contrast=0.7)
        self.assertTrue(result['success'])
        b.restart.assert_not_called()
        b._save_color_settings.assert_called_once()

    def test_loading_mode_persist_only(self):
        b = make_blocker(current_source='loading')
        result = b.set_color_settings(hue=0.1)
        self.assertTrue(result['success'])
        b.restart.assert_not_called()

    def test_response_contains_merged_values(self):
        b = make_blocker()
        result = b.set_color_settings(saturation=1.3, brightness=0.1)
        self.assertEqual(result['saturation'], 1.3)
        self.assertEqual(result['brightness'], 0.1)
        self.assertEqual(result['contrast'], 1.0)

    def test_clamping_applies_in_rebuild_branch(self):
        b = make_blocker()
        result = b.set_color_settings(saturation=99.0)
        self.assertEqual(result['saturation'], 2.0)
        b.restart.assert_called_once_with(penalize=False)


# =========================================================================
# set_color_settings - branch: element present (true realtime path)
# =========================================================================

class TestSetColorLive(unittest.TestCase):

    def _blocker_with_element(self, values=None):
        el = make_element(values)
        b = make_blocker(element=el)
        return b, el

    def test_sets_each_property_live(self):
        b, el = self._blocker_with_element()
        b.set_color_settings(saturation=1.3, brightness=0.1,
                             contrast=0.9, hue=-0.05)
        self.assertEqual(el.get_property('saturation'), 1.3)
        self.assertEqual(el.get_property('brightness'), 0.1)
        self.assertEqual(el.get_property('contrast'), 0.9)
        self.assertEqual(el.get_property('hue'), -0.05)

    def test_no_pipeline_rebuild_in_live_path(self):
        b, el = self._blocker_with_element()
        b.set_color_settings(saturation=1.3)
        b.restart.assert_not_called()

    def test_partial_update_leaves_other_properties(self):
        b, el = self._blocker_with_element(
            {'saturation': 1.3, 'brightness': 0.1, 'contrast': 0.9, 'hue': 0.0})
        b.set_color_settings(saturation=1.5)
        self.assertEqual(el.get_property('saturation'), 1.5)
        self.assertEqual(el.get_property('brightness'), 0.1)

    def test_live_values_persisted(self):
        b, el = self._blocker_with_element()
        result = b.set_color_settings(saturation=1.3)
        self.assertTrue(result['success'])
        b._save_color_settings.assert_called_once()
        self.assertEqual(b._saved_color_settings['saturation'], 1.3)

    def test_live_clamping(self):
        b, el = self._blocker_with_element()
        b.set_color_settings(saturation=3.0, brightness=-9, contrast=100, hue=2)
        self.assertEqual(el.get_property('saturation'), 2.0)
        self.assertEqual(el.get_property('brightness'), -1.0)
        self.assertEqual(el.get_property('contrast'), 2.0)
        self.assertEqual(el.get_property('hue'), 1.0)

    def test_set_property_exception_returns_error(self):
        el = MagicMock()
        el.set_property.side_effect = RuntimeError('gst boom')
        b = make_blocker(element=el)
        result = b.set_color_settings(saturation=1.3)
        self.assertFalse(result['success'])
        self.assertIn('error', result)


# =========================================================================
# get_color_settings fallback order
# =========================================================================

class TestGetColorSettings(unittest.TestCase):

    def test_reads_element_when_present(self):
        el = make_element({'saturation': 1.4, 'brightness': 0.2,
                           'contrast': 1.1, 'hue': 0.0})
        b = make_blocker(element=el)
        self.assertEqual(b.get_color_settings()['saturation'], 1.4)

    def test_falls_back_to_saved_without_element(self):
        b = make_blocker(saved={'saturation': 1.2, 'brightness': 0.0,
                                'contrast': 1.0, 'hue': 0.0})
        self.assertEqual(b.get_color_settings()['saturation'], 1.2)

    def test_falls_back_to_saved_without_pipeline(self):
        b = make_blocker(saved={'saturation': 0.9, 'brightness': 0.0,
                                'contrast': 1.0, 'hue': 0.0}, pipeline=None)
        self.assertEqual(b.get_color_settings()['saturation'], 0.9)

    def test_defaults_when_nothing_saved(self):
        b = make_blocker(pipeline=None)
        b._saved_color_settings = None
        self.assertEqual(b.get_color_settings(), NEUTRAL)

    def test_returned_saved_dict_is_a_copy(self):
        b = make_blocker(pipeline=None)
        got = b.get_color_settings()
        got['saturation'] = 42
        self.assertEqual(b._saved_color_settings['saturation'], 1.0)


# =========================================================================
# _init_pipeline launch-string baking (the core of the realtime fix)
# =========================================================================

class TestInitPipelineLaunchString(unittest.TestCase):

    def _run_init(self, saved):
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        b._saved_color_settings = dict(saved)
        b.ustreamer_port = 9090
        b.plane_id = 192
        b.connector_id = 231
        with patch('ad_blocker.Gst') as mock_gst:
            b._init_pipeline()
            launch_str = mock_gst.parse_launch.call_args[0][0]
        return b, launch_str

    def test_neutral_omits_videobalance_entirely(self):
        """Zero-copy 60fps path: neutral settings must produce a pipeline
        with NO videobalance element at all."""
        b, launch = self._run_init(NEUTRAL)
        self.assertNotIn('videobalance', launch)
        self.assertFalse(b._pipeline_has_colorbalance)

    def test_non_neutral_bakes_all_saved_values(self):
        b, launch = self._run_init({'saturation': 1.3, 'brightness': 0.15,
                                    'contrast': 0.85, 'hue': -0.05})
        self.assertIn('videobalance', launch)
        self.assertIn('saturation=1.3000', launch)
        self.assertIn('brightness=0.1500', launch)
        self.assertIn('contrast=0.8500', launch)
        self.assertIn('hue=-0.0500', launch)
        self.assertIn('name=colorbalance', launch)
        self.assertTrue(b._pipeline_has_colorbalance)

    def test_regression_element_never_built_at_identity_when_non_neutral(self):
        """The original bug: element was created at 1.0/0.0/1.0/0.0 and the
        saved values were never applied on restart paths."""
        b, launch = self._run_init({'saturation': 0.8, 'brightness': 0.05,
                                    'contrast': 0.5, 'hue': 0.0})
        self.assertNotIn('saturation=1.0000', launch)
        self.assertIn('saturation=0.8000', launch)
        self.assertIn('contrast=0.5000', launch)

    def test_skip_vsync_kept_in_both_modes(self):
        """60fps guard: skip-vsync=true (the double-vsync fix) must survive
        in both pipeline shapes."""
        _, launch_neutral = self._run_init(NEUTRAL)
        _, launch_color = self._run_init({'saturation': 1.3, 'brightness': 0.0,
                                          'contrast': 1.0, 'hue': 0.0})
        self.assertIn('skip-vsync=true', launch_neutral)
        self.assertIn('skip-vsync=true', launch_color)

    def test_pipeline_shape_preserved(self):
        """Decode -> (videobalance) -> queue -> identity -> kmssink order and
        the leaky low-latency queue must be unchanged by the fix."""
        _, launch = self._run_init({'saturation': 1.3, 'brightness': 0.0,
                                    'contrast': 1.0, 'hue': 0.0})
        self.assertLess(launch.index('mppjpegdec'), launch.index('videobalance'))
        self.assertLess(launch.index('videobalance'), launch.index('videoqueue'))
        self.assertLess(launch.index('videoqueue'), launch.index('kmssink'))
        self.assertIn('queue max-size-buffers=3 leaky=downstream', launch)

    def test_element_lookup_name_matches_setter(self):
        """set_color_settings looks up 'colorbalance' - the baked element must
        carry exactly that name so live updates keep working after a rebuild."""
        _, launch = self._run_init({'saturation': 1.3, 'brightness': 0.0,
                                    'contrast': 1.0, 'hue': 0.0})
        self.assertIn('name=colorbalance !', launch)


# =========================================================================
# _restart_pipeline penalize semantics
# =========================================================================

class TestRestartPenalize(unittest.TestCase):

    def _make_restartable(self, failures=0):
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        b._restart_lock = threading.Lock()
        b._pipeline_restarting = False
        b._restart_count = 0
        b._consecutive_failures = failures
        b._base_restart_delay = 1.0
        b._max_restart_delay = 30.0
        b._last_buffer_time = 0.0
        b._last_restart_time = 0.0
        b.pipeline = None
        b.bus = None
        b._init_pipeline = MagicMock()   # leaves b.pipeline None -> no PLAYING
        b._force_hdmi_reinit = MagicMock()
        return b

    def test_non_penalized_keeps_failure_counter(self):
        b = self._make_restartable(failures=2)
        with patch('ad_blocker.time.sleep'):
            b._restart_pipeline(penalize=False)
        self.assertEqual(b._consecutive_failures, 2)
        b._init_pipeline.assert_called_once()

    def test_penalized_increments_failure_counter(self):
        b = self._make_restartable(failures=2)
        with patch('ad_blocker.time.sleep'):
            b._restart_pipeline(penalize=True)
        self.assertEqual(b._consecutive_failures, 3)

    def test_non_penalized_has_zero_delay(self):
        b = self._make_restartable(failures=4)  # would be a long backoff
        with patch('ad_blocker.time.sleep') as mock_sleep:
            b._restart_pipeline(penalize=False)
        mock_sleep.assert_called_once_with(0.0)

    def test_penalized_backoff_delay(self):
        b = self._make_restartable(failures=2)  # -> 3rd failure, 1.0 * 2^2 = 4s
        with patch('ad_blocker.time.sleep') as mock_sleep, \
             patch('subprocess.run') as mock_run:
            b._restart_pipeline(penalize=True)
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        self.assertIn(4.0, delays)

    def test_non_penalized_never_pkills_ustreamer(self):
        """Even with an already-elevated failure counter, a color rebuild
        must not trigger the MPP pkill escalation."""
        b = self._make_restartable(failures=5)
        with patch('ad_blocker.time.sleep'), patch('subprocess.run') as mock_run:
            b._restart_pipeline(penalize=False)
        mock_run.assert_not_called()

    def test_penalized_pkills_after_three_failures(self):
        b = self._make_restartable(failures=2)  # increments to 3 -> pkill
        with patch('ad_blocker.time.sleep'), patch('subprocess.run') as mock_run:
            b._restart_pipeline(penalize=True)
        self.assertTrue(mock_run.called)
        self.assertIn('ustreamer', mock_run.call_args[0][0])

    def test_restart_thread_passes_penalize_through(self):
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        with patch('ad_blocker.threading.Thread') as mock_thread:
            b.restart(penalize=False)
        kwargs = mock_thread.call_args.kwargs
        self.assertEqual(kwargs['args'], (False, False))
        self.assertEqual(kwargs['target'], b._restart_pipeline)

    def test_restart_defaults_stay_penalized(self):
        """Watchdog/health callers use restart() bare - that must remain the
        penalized (backoff-protected) path."""
        b = DRMAdBlocker.__new__(DRMAdBlocker)
        with patch('ad_blocker.threading.Thread') as mock_thread:
            b.restart()
        self.assertEqual(mock_thread.call_args.kwargs['args'], (False, True))

    def test_reentrancy_guard_still_works(self):
        b = self._make_restartable()
        b._pipeline_restarting = True
        with patch('ad_blocker.time.sleep'):
            b._restart_pipeline(penalize=False)
        b._init_pipeline.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
