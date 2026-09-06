#!/usr/bin/env python3
"""Tests for the A/V drift detector + in-place resync in src/audio.py.

The HDMI source's audio clock and the RK3588's HDMI-TX audio clock are
independent crystals; with alsasink sync=false a slightly-faster source
piles samples into `syncqueue`, which creeps from its 300ms floor toward its
500ms cap — audio lags video, then the queue saturates. These tests cover
the pure decision logic and the drain procedure with a mocked queue.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from audio import AudioPassthrough


def make_audio(level_ms=300.0, muted=False, silent=True):
    a = AudioPassthrough.__new__(AudioPassthrough)
    a.SYNC_BASELINE_MS = 300.0
    a.SYNC_MAX_MS = 500.0
    a.DRIFT_RESYNC_MS = 100.0
    a.DRIFT_HARD_MS = 170.0
    a.DRIFT_SUSTAIN_CHECKS = 3
    a.RESYNC_MIN_INTERVAL_S = 120.0
    a._drift_enabled = True
    a._drift_over_checks = 0
    a._resync_count = 0
    a._last_resync_time = 0.0
    a._last_resync_reason = ''
    a._last_sync_level_ms = None
    a._max_sync_level_ms = 0.0
    a.is_running = True
    a.is_muted = muted
    a._restart_in_progress = False
    a._playback_fakesink = False
    a._level_history = [] if silent else [0.3]
    # mocked pipeline with a stateful syncqueue
    q = MagicMock()
    state = {'level': level_ms * 1e6, 'max': 500_000_000, 'leaky': 0}
    def getp(name):
        return {'current-level-time': state['level'], 'max-size-time': state['max'],
                'leaky': state['leaky']}[name]
    def setp(name, val):
        if name == 'max-size-time': state['max'] = val
        if name == 'leaky':
            state['leaky'] = val
            # emulate GstQueue: leaky downstream drains to max-size on next buffer
            if val == 2 and state['level'] > state['max']:
                state['level'] = state['max']
    q.get_property.side_effect = getp
    q.set_property.side_effect = setp
    a.pipeline = MagicMock()
    a.pipeline.get_by_name.side_effect = lambda n: q if n == 'syncqueue' else None
    a._q = q; a._qstate = state
    return a


class TestShouldResync(unittest.TestCase):
    def test_below_threshold_never(self):
        a = make_audio()
        for t in range(0, 60, 3):
            self.assertIsNone(a._should_resync(50.0, False, False, t))
        self.assertEqual(a._drift_over_checks, 0)

    def test_none_drift_resets(self):
        a = make_audio(); a._drift_over_checks = 2
        self.assertIsNone(a._should_resync(None, True, True, 0))
        self.assertEqual(a._drift_over_checks, 0)

    def test_disabled(self):
        a = make_audio(); a._drift_enabled = False
        self.assertIsNone(a._should_resync(180.0, True, True, 1000))

    def test_hard_cap_fires_immediately_even_when_loud(self):
        a = make_audio()
        self.assertEqual(a._should_resync(180.0, False, False, 1000), 'hard')

    def test_sustained_then_quiet(self):
        a = make_audio()
        self.assertIsNone(a._should_resync(120.0, False, True, 1000))
        self.assertIsNone(a._should_resync(120.0, False, True, 1003))
        self.assertEqual(a._should_resync(120.0, False, True, 1006), 'quiet')

    def test_muted_counts_as_quiet(self):
        a = make_audio()
        for t in (0, 3): a._should_resync(120.0, True, False, t)
        self.assertEqual(a._should_resync(120.0, True, False, 6), 'quiet')

    def test_loud_waits_then_goes_anyway(self):
        """Not quiet: hold until 4x the sustain window, then resync regardless."""
        a = make_audio()
        results = [a._should_resync(120.0, False, False, t) for t in range(0, 3 * 12, 3)]
        self.assertTrue(all(r is None for r in results[:11]))
        self.assertEqual(results[11], 'sustained')

    def test_dip_below_threshold_resets_streak(self):
        a = make_audio()
        a._should_resync(120.0, False, True, 0)
        a._should_resync(120.0, False, True, 3)
        a._should_resync(50.0, False, True, 6)   # dips
        self.assertEqual(a._drift_over_checks, 0)
        self.assertIsNone(a._should_resync(120.0, False, True, 9))

    def test_rate_limited(self):
        a = make_audio(); a._last_resync_time = 1000.0
        for t in (1003, 1006, 1009):
            r = a._should_resync(120.0, True, True, t)
        self.assertIsNone(r)  # within 120s of last resync
        a._drift_over_checks = 3
        self.assertEqual(a._should_resync(120.0, True, True, 1000 + 121), 'quiet')

    def test_hard_has_shorter_rate_limit(self):
        a = make_audio(); a._last_resync_time = 1000.0
        self.assertIsNone(a._should_resync(180.0, False, False, 1020))
        self.assertEqual(a._should_resync(180.0, False, False, 1031), 'hard')


class TestThresholdsReachable(unittest.TestCase):
    def test_thresholds_inside_queue_range(self):
        """Drift can never exceed SYNC_MAX - baseline (the queue cap); both
        thresholds must be strictly inside that range or they never fire."""
        a = make_audio()
        rng = a.SYNC_MAX_MS - a.SYNC_BASELINE_MS
        self.assertLess(a.DRIFT_RESYNC_MS, a.DRIFT_HARD_MS)
        self.assertLess(a.DRIFT_HARD_MS, rng)
        self.assertGreater(a.DRIFT_RESYNC_MS, 0)


class TestResyncAv(unittest.TestCase):
    def test_drains_to_baseline_and_restores(self):
        a = make_audio(level_ms=480.0)
        with patch('audio.time.sleep'):
            r = a.resync_av('quiet')
        self.assertTrue(r['success'])
        self.assertAlmostEqual(r['before_ms'], 480.0)
        self.assertAlmostEqual(r['after_ms'], 300.0)
        # restored: not leaky, original cap
        self.assertEqual(a._qstate['leaky'], 0)
        self.assertEqual(a._qstate['max'], 500_000_000)
        self.assertEqual(a._resync_count, 1)
        self.assertEqual(a._last_resync_reason, 'quiet')
        self.assertEqual(a._drift_over_checks, 0)

    def test_property_sequence(self):
        a = make_audio(level_ms=480.0)
        with patch('audio.time.sleep'):
            a.resync_av()
        calls = [c.args for c in a._q.set_property.call_args_list]
        self.assertEqual(calls[0], ('max-size-time', 300_000_000))
        self.assertEqual(calls[1], ('leaky', 2))
        self.assertEqual(calls[-2], ('leaky', 0))
        self.assertEqual(calls[-1], ('max-size-time', 500_000_000))

    def test_restores_even_if_drain_raises(self):
        a = make_audio(level_ms=480.0)
        orig = a._q.set_property.side_effect
        def boom(name, val):
            if name == 'leaky' and val == 2:
                raise RuntimeError('gst')
            return orig(name, val)
        a._q.set_property.side_effect = boom
        with patch('audio.time.sleep'):
            with self.assertRaises(RuntimeError):
                a.resync_av()
        self.assertEqual(a._qstate['leaky'], 0)
        self.assertEqual(a._qstate['max'], 500_000_000)

    def test_refuses_during_restart(self):
        a = make_audio(); a._restart_in_progress = True
        self.assertFalse(a.resync_av()['success'])

    def test_refuses_without_pipeline(self):
        a = make_audio(); a.pipeline = None
        self.assertFalse(a.resync_av()['success'])


class TestWatchdogHook(unittest.TestCase):
    def test_check_resyncs_on_hard_drift(self):
        a = make_audio(level_ms=480.0)
        with patch('audio.time.sleep'):
            a._check_av_drift()
        self.assertEqual(a._resync_count, 1)
        self.assertEqual(a._last_resync_reason, 'hard')
        self.assertAlmostEqual(a._max_sync_level_ms, 480.0)

    def test_check_skips_on_fakesink(self):
        a = make_audio(level_ms=480.0); a._playback_fakesink = True
        a._check_av_drift()
        self.assertEqual(a._resync_count, 0)

    def test_check_no_action_at_baseline(self):
        a = make_audio(level_ms=305.0)
        a._check_av_drift()
        self.assertEqual(a._resync_count, 0)
        self.assertAlmostEqual(a._last_sync_level_ms, 305.0)

    def test_get_sync_levels_reports_drift(self):
        a = make_audio(level_ms=420.0)
        lv = a.get_sync_levels()
        self.assertAlmostEqual(lv['sync_level_ms'], 420.0)
        self.assertAlmostEqual(lv['sync_drift_ms'], 120.0)
        self.assertIsNone(lv['audioqueue_level_ms'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
