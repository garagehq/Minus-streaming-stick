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


# ---------------------------------------------------------------------------
# Audio delay follows the measured video latency (av_sync.py).
#
# minus, 2026-10-18: the delay was a constant 300 ms calibrated in April for a
# ustreamer that took ~110 ms; the July 60 fps change made video ~83 ms end to
# end and the sound arrived ~300 ms after the picture.
# ---------------------------------------------------------------------------
import time  # noqa: E402
import unittest.mock  # noqa: E402
import av_sync  # noqa: E402

FRAME = (b'--boundarydonotcross\r\nContent-Type: image/jpeg\r\n'
         b'X-UStreamer-Grab-Time: %.6f\r\nX-UStreamer-Send-Time: %.6f\r\n\r\n' + b'\xff' * 64)


def stream(*pairs):
    return b''.join(FRAME % p for p in pairs)


def make_calibratable(level_ms=300.0, aq_ms=65.0, ustreamer_ms=38.0, fps=60.0,
                      display=('opt', 'Optoma UHD', 72, 0), offsets=None, legacy=None):
    a = make_audio(level_ms=level_ms)
    a.SYNC_HEADROOM_MS = 200.0
    a.SYNC_AUTO = True
    a.CALIBRATE_INTERVAL_S = 300.0
    a._calibrate_at = 0.0
    a._last_calibration = None
    a.sync_offsets = dict(offsets or {})
    a.legacy_trim_ms = legacy
    a.saved = []
    a.on_sync_settings = a.saved.append
    a._display_id = 'unknown'
    a._display_name = 'unknown'
    a._calibrated_once = False
    a._smoother = av_sync.DelaySmoother(band=40.0)
    a.m = {'ustreamer_ms': ustreamer_ms, 'source_fps': fps, 'display_id': display[0],
           'display_name': display[1], 'display_video_ms': display[2], 'display_audio_ms': display[3]}
    a.video_latency_fn = lambda: a.m
    sq = a._q
    aq = MagicMock()
    aq_state = {'level': aq_ms * 1e6, 'max': 100_000_000}
    aq.get_property.side_effect = lambda n: {'current-level-time': aq_state['level'],
                                             'max-size-time': aq_state['max']}[n]

    def aq_set(n, v):
        if n == 'max-size-time':
            aq_state['max'] = v
            aq_state['level'] = min(aq_state['level'], v)   # leaky downstream
    aq.set_property.side_effect = aq_set
    a.pipeline.get_by_name.side_effect = lambda n: {'syncqueue': sq, 'audioqueue': aq}.get(n)
    a._aq_state = aq_state
    return a


# The real Optoma UHD EDID read on minus 2026-10-18 (HDMI VSDB: video 72 ms, audio 0 ms).
OPTOMA_VSDB = bytes.fromhex('72030c002000b83ce025013701808001020304')


def edid_with(vsdb=OPTOMA_VSDB, name=b'Optoma UHD'):
    base = bytearray(128)
    base[0:8] = b'\x00\xff\xff\xff\xff\xff\xff\x00'
    base[8:16] = bytes.fromhex('3e8d7c0004000000')
    base[54:57] = b'\x00\x00\x00'
    base[57] = 0xFC
    base[59:59 + 13] = (name + b'\n').ljust(13, b' ')
    ext = bytearray(128)
    ext[0], ext[1] = 0x02, 0x03
    ext[4:4 + len(vsdb)] = vsdb
    ext[2] = 4 + len(vsdb)
    return bytes(base + ext)


class TestAvSyncMath(unittest.TestCase):
    def test_parses_grab_to_send_per_frame(self):
        lat = av_sync.frame_latencies_ms(stream((100.000, 100.038), (100.016, 100.050)))
        self.assertEqual([round(x) for x in lat], [38, 34])

    def test_a_frame_without_both_times_is_skipped(self):
        data = stream((1.0, 1.04)) + b'--boundarydonotcross\r\nX-UStreamer-Grab-Time: 2.0\r\n'
        self.assertEqual(len(av_sync.frame_latencies_ms(data)), 1)

    def test_trim_is_clamped(self):
        self.assertEqual(av_sync.clamp_trim(9999), av_sync.TRIM_LIMIT_MS)
        self.assertEqual(av_sync.clamp_trim('junk'), 0.0)

    def test_display_audio_latency_is_subtracted(self):
        self.assertEqual(av_sync.target_delay_ms(300.0, 0.0, 0.0, display_audio_ms=50.0),
                         300.0 - 50.0 - av_sync.AUDIO_PATH_MS)


class TestEdid(unittest.TestCase):
    def test_reads_the_optoma_on_minus(self):
        info = av_sync.parse_edid(edid_with())
        self.assertEqual(info['name'], 'Optoma UHD')
        self.assertEqual((info['video_ms'], info['audio_ms']), (72, 0))
        self.assertEqual(info['id'], '3e8d7c0004000000')

    def test_display_without_latency_fields(self):
        vsdb = bytearray(OPTOMA_VSDB)
        vsdb[8] &= 0x7F                    # Latency_Fields_Present off
        info = av_sync.parse_edid(edid_with(bytes(vsdb)))
        self.assertEqual((info['video_ms'], info['audio_ms']), (None, None))

    def test_garbage(self):
        self.assertIsNone(av_sync.parse_edid(b''))
        self.assertIsNone(av_sync.parse_edid(b'\x00' * 10))

    def test_connected_display_from_sysfs(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            for name, status, edid in (('card0-HDMI-A-1', 'disconnected', b''),
                                       ('card0-HDMI-A-2', 'connected', edid_with())):
                os.makedirs(os.path.join(d, name))
                Path(d, name, 'status').write_text(status + '\n')
                Path(d, name, 'edid').write_bytes(edid)
            self.assertEqual(av_sync.connected_display(d)['video_ms'], 72)


class TestSmoother(unittest.TestCase):
    def test_small_moves_are_ignored(self):
        s = av_sync.DelaySmoother(40)
        self.assertIsNone(s.propose(330, 300))

    def test_a_big_move_needs_two_agreeing_readings(self):
        s = av_sync.DelaySmoother(40)
        self.assertIsNone(s.propose(200, 300))
        self.assertEqual(s.propose(220, 300), 210.0)

    def test_one_outlier_never_moves_it(self):
        s = av_sync.DelaySmoother(40)
        self.assertIsNone(s.propose(150, 300))     # outlier
        self.assertIsNone(s.propose(310, 300))     # back to normal: pending dropped
        self.assertIsNone(s.propose(150, 300))
        self.assertIsNone(s.propose(260, 300))     # disagrees with the last one

    def test_force_applies_at_once(self):
        self.assertEqual(av_sync.DelaySmoother(40).propose(120, 300, force=True), 120)


class TestLiveDelay(unittest.TestCase):
    def test_first_calibration_applies_and_includes_the_display(self):
        a = make_calibratable(level_ms=300.0, aq_ms=0.0, offsets={'opt': 0.0})
        a.calibrate_sync('test')
        # 16.7 + 38 + 28 (box) + 72 (Optoma) + 20 (AUDIO_PATH_MS = -20) - 15 (typical audioqueue) = 159.7
        self.assertAlmostEqual(a.SYNC_BASELINE_MS, 159.7, places=1)
        self.assertEqual(a.get_sync_info()['display'], 'Optoma UHD')

    def test_old_global_trim_migrates_without_moving_the_delay(self):
        """10-18: the owner set the old trim to 325 by ear; after the update
        the delay must come out the same."""
        before = av_sync.target_delay_ms(av_sync.video_latency_ms(38.0, 60.0),
                                         av_sync.AUDIOQUEUE_TYPICAL_MS, 325.0)
        a = make_calibratable(level_ms=300.0, aq_ms=0.0, legacy=325.0)
        a.calibrate_sync('test')
        self.assertAlmostEqual(a.SYNC_BASELINE_MS, before, places=1)
        self.assertEqual(a.sync_offsets['opt'], 325.0 - 72)
        self.assertIsNone(a.legacy_trim_ms)
        self.assertEqual(a.saved[-1]['av_sync_offsets'], {'opt': 253.0})

    def test_later_noise_does_not_move_it(self):
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, offsets={'opt': 253.0})
        a.calibrate_sync('start')
        settled = a.SYNC_BASELINE_MS
        for us in (75.0, 20.0, 60.0):          # ustreamer reads that jump around
            a.m['ustreamer_ms'] = us
            a.calibrate_sync('periodic')
        self.assertEqual(a.SYNC_BASELINE_MS, settled)

    def test_a_real_change_is_followed_after_two_readings(self):
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, offsets={'opt': 0.0})
        a.calibrate_sync('start')
        start = a.SYNC_BASELINE_MS
        a.m['ustreamer_ms'] = 138.0            # e.g. a slower capture path: +100 ms
        a.calibrate_sync('periodic')
        self.assertEqual(a.SYNC_BASELINE_MS, start)
        a.calibrate_sync('periodic')
        self.assertAlmostEqual(a.SYNC_BASELINE_MS, start + 100.0, places=0)

    def test_each_display_keeps_its_own_offset(self):
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, offsets={'opt': 253.0, 'tv': -20.0})
        a.calibrate_sync('start')
        projector = a.SYNC_BASELINE_MS
        a.m.update(display_id='tv', display_name='Living room TV', display_video_ms=20, display_audio_ms=0)
        a.calibrate_sync('periodic')          # a new display applies at once
        self.assertAlmostEqual(projector - a.SYNC_BASELINE_MS, (253 + 72) - (-20 + 20), places=0)

    def test_trim_sets_this_displays_offset_at_once(self):
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, offsets={'opt': 0.0})
        a.calibrate_sync('start')
        base = a.SYNC_BASELINE_MS
        info = a.set_sync_trim(20)
        self.assertEqual(info['trim_ms'], 20)
        self.assertAlmostEqual(a.SYNC_BASELINE_MS, base + 20, places=1)
        self.assertEqual(a.sync_offsets, {'opt': 20.0})

    def test_lowering_drains_but_never_to_unlimited(self):
        a = make_calibratable(level_ms=300.0, aq_ms=0.0, display=('x', 'TV', None, None), offsets={'x': -200.0})
        a.calibrate_sync('start')
        self.assertEqual(a.SYNC_BASELINE_MS, 0.0)
        a._q.set_property.assert_any_call('min-threshold-time', 0)
        self.assertLessEqual(a._qstate['level'], 5e6)       # 5 ms floor, not max-size-time=0

    def test_audioqueue_level_does_not_move_the_delay(self):
        """04:02: the delay dropped 63 ms because audioqueue read 60 ms at
        that moment; it emptied soon after and the sound came out early."""
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, offsets={'opt': 259.0})
        a.calibrate_sync('start')
        settled = a.SYNC_BASELINE_MS
        for aq in (60.0, 70.0, 0.0, 60.0):
            a._aq_state['level'] = aq * 1e6
            a.calibrate_sync('periodic')
        self.assertEqual(a.SYNC_BASELINE_MS, settled)

    def test_audioqueue_is_capped_small_live(self):
        a = make_calibratable(level_ms=0.0, aq_ms=60.0, offsets={'opt': 0.0})
        a.calibrate_sync('start')
        self.assertEqual(a._aq_state['max'], int(av_sync.AUDIOQUEUE_MAX_MS * 1e6))
        self.assertLessEqual(a._aq_state['level'], av_sync.AUDIOQUEUE_MAX_MS * 1e6)

    def test_pipeline_builds_audioqueue_with_the_cap(self):
        src = (Path(__file__).resolve().parent.parent / 'src' / 'audio.py').read_text()
        self.assertIn('max-size-time={int(av_sync.AUDIOQUEUE_MAX_MS * 1e6)} leaky=downstream name=audioqueue', src)

    def test_raising_the_delay_does_not_drain(self):
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, ustreamer_ms=400.0, offsets={'opt': 0.0})
        a.SYNC_BASELINE_MS = 0.0
        a.calibrate_sync('start')
        self.assertGreater(a.SYNC_BASELINE_MS, 300.0)
        self.assertNotIn(unittest.mock.call('leaky', 2), a._q.set_property.call_args_list)

    def test_no_measurement_keeps_the_delay(self):
        a = make_calibratable()
        a.video_latency_fn = lambda: None
        self.assertFalse(a.calibrate_sync('test')['success'])
        self.assertEqual(a.SYNC_BASELINE_MS, 300.0)

    def test_watchdog_calibrates_when_due_and_reschedules(self):
        a = make_calibratable(level_ms=300.0, offsets={'opt': 0.0})
        a._check_av_drift()
        self.assertTrue(a._calibrated_once)
        self.assertGreater(a._calibrate_at, time.time() + 200)

    def test_pipeline_string_uses_the_current_delay(self):
        src = (Path(__file__).resolve().parent.parent / 'src' / 'audio.py').read_text()
        self.assertIn('min-threshold-time={int(self.SYNC_BASELINE_MS * 1e6)}', src)
        self.assertNotIn('min-threshold-time=300000000', src)


class TestAvSettingsSurviveRestart(unittest.TestCase):
    """_load_system_settings used to keep only keys it had defaults for, so
    the A/V trim (and anything newer) vanished on every restart (10-18)."""

    def test_unknown_and_av_keys_round_trip(self):
        import json, tempfile
        import minus as m
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, 'settings.json')
            f.write_text(json.dumps({'av_sync_offsets': {'3e8d7c0004000000': 259.0},
                                     'av_sync_last_delay_ms': 362.7,
                                     'some_future_key': 'kept', 'vlm_preload': False}))
            with patch.object(m, 'SYSTEM_SETTINGS_FILE', f):
                obj = m.Minus.__new__(m.Minus)
                st = obj._load_system_settings()
        self.assertEqual(st['av_sync_offsets'], {'3e8d7c0004000000': 259.0})
        self.assertEqual(st['av_sync_last_delay_ms'], 362.7)
        self.assertEqual(st['some_future_key'], 'kept')
        self.assertFalse(st['vlm_preload'])
        self.assertEqual(st['capture_fps'], 30)          # default for a new key

    def test_restart_starts_from_the_saved_delay_and_offset(self):
        import minus as m
        obj = m.Minus.__new__(m.Minus)
        obj._system_settings = {'av_sync_offsets': {'opt': 259.0}, 'av_sync_last_delay_ms': 362.7}
        obj.audio = make_calibratable(level_ms=0.0, aq_ms=0.0)
        obj._init_av_sync()
        self.assertEqual(obj.audio.SYNC_BASELINE_MS, 362.7)
        self.assertEqual(obj.audio.sync_offsets, {'opt': 259.0})
        obj.audio.video_latency_fn = lambda: obj.audio.m
        obj.audio.calibrate_sync('start')
        # 16.7 + 38 + 28 + 72 + 20 - 15 + 259 = 418.7: where the owner set it
        self.assertAlmostEqual(obj.audio.SYNC_BASELINE_MS, 418.7, places=1)


class TestNoDisplayNoCalibration(unittest.TestCase):
    """minus 2026-10-19: the projector went off, the next calibration set the
    delay to the box alone (17.7 ms) and saved it; the Optoma needs ~347."""

    def test_no_display_keeps_the_delay_and_saves_nothing(self):
        a = make_calibratable(level_ms=0.0, aq_ms=0.0, offsets={'opt': 259.0})
        a.calibrate_sync('start')
        good = a.SYNC_BASELINE_MS
        saved = list(a.saved)
        a.m.update(display_id=None, display_name=None, display_video_ms=None, display_audio_ms=None)
        r = a.calibrate_sync('periodic')
        self.assertFalse(r['success'])
        self.assertEqual(a.SYNC_BASELINE_MS, good)
        self.assertEqual(a.saved, saved, 'nothing saved without a display')
        self.assertNotIn('unknown', a.sync_offsets)

    def test_reconnect_requests_a_calibration(self):
        import minus as minus_mod
        from unittest.mock import MagicMock
        m = minus_mod.Minus.__new__(minus_mod.Minus)
        m.HDMI_RECONNECT_GRACE_SECONDS = 90.0
        m.audio = MagicMock()
        m.notify_hdmi_reconnect()
        m.audio.request_calibration.assert_called_once()

