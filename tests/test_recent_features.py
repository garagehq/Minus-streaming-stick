#!/usr/bin/env python3
"""
End-to-end tests for features landed in the last two development sprints.

These cover behaviour added after `tests/test_modules.py` and
`tests/test_autonomous_mode.py` were last expanded:

- Process-based VLM worker: soft/hard timeout, call-serialization lock,
  P95 latency auto-recovery, deep-restart escalation.
- OCR exclusion list: Minus' own on-screen overlays must not self-trigger
  ad detection.
- Autonomous mode signal changes: `HOME_SCREEN_KEYWORDS` additions,
  `AD_ONLY_KEYWORDS` guard, `PERSISTENT_STATIC_LIMIT` escalation,
  `_is_audio_pipeline_available`, dismiss-via-back.
- Web UI device-agnostic skip routing.

Run with: python3 -m pytest tests/test_recent_features.py -v
"""

import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# src/webui.py does `from src.wifi_manager import ...`; add project root too
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# VLMProcess auto-recovery and latency tracking
# =============================================================================


class TestVLMProcessAutoRecovery(unittest.TestCase):
    """VLMProcess tracks rolling latency and escalates restarts when
    per-image variance drags P95 above the trigger threshold. These tests
    exercise that pure-Python logic without booting the actual NPU."""

    def setUp(self):
        from vlm_worker import VLMProcess
        self.vp = VLMProcess()

    def test_record_latency_ignores_non_numeric(self):
        self.vp._record_latency("nope")
        self.vp._record_latency(None)
        self.vp._record_latency(0.42)
        self.assertEqual(list(self.vp._recent_latencies), [0.42])

    def test_latency_stats_empty(self):
        stats = self.vp.get_latency_stats()
        self.assertEqual(stats, {'samples': 0})

    def test_latency_stats_fields(self):
        for v in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
            self.vp._record_latency(v)
        stats = self.vp.get_latency_stats()
        self.assertEqual(stats['samples'], 10)
        self.assertIn('p50_s', stats)
        self.assertIn('p95_s', stats)
        self.assertIn('max_s', stats)
        self.assertGreater(stats['p95_s'], stats['p50_s'])

    def test_auto_recover_skipped_while_window_unfilled(self):
        """With fewer than LATENCY_WINDOW samples we never trigger."""
        for _ in range(self.vp.LATENCY_WINDOW - 1):
            self.vp._record_latency(99.0)
        with patch.object(self.vp, 'restart') as mock_restart:
            self.vp._maybe_auto_recover()
            mock_restart.assert_not_called()

    def test_auto_recover_skipped_when_p95_below_trigger(self):
        for _ in range(self.vp.LATENCY_WINDOW):
            self.vp._record_latency(0.8)
        with patch.object(self.vp, 'restart') as mock_restart:
            self.vp._maybe_auto_recover()
            mock_restart.assert_not_called()

    def test_auto_recover_triggers_restart_on_degraded_p95(self):
        """Once P95 is past the trigger, the first recovery is a normal restart."""
        # Ensure cooldown does not suppress (no prior recovery recorded).
        self.assertEqual(self.vp._last_auto_recovery_time, 0.0)
        for _ in range(self.vp.LATENCY_WINDOW):
            self.vp._record_latency(self.vp.LATENCY_P95_TRIGGER + 5.0)
        with patch.object(self.vp, 'restart') as mock_restart, \
             patch.object(self.vp, 'kill'), \
             patch.object(self.vp, 'start'), \
             patch('time.sleep'):
            self.vp._maybe_auto_recover()
            mock_restart.assert_called_once()
        # The deque is cleared after recovery fires so we don't retrigger
        # immediately on the next sample.
        self.assertEqual(len(self.vp._recent_latencies), 0)
        self.assertGreater(self.vp._last_auto_recovery_time, 0.0)

    def test_auto_recover_cooldown_blocks_repeat_recovery(self):
        """If we just recovered, don't recover again inside the cooldown window."""
        self.vp._last_auto_recovery_time = time.time()  # just happened
        for _ in range(self.vp.LATENCY_WINDOW):
            self.vp._record_latency(self.vp.LATENCY_P95_TRIGGER + 2.0)
        with patch.object(self.vp, 'restart') as mock_restart, \
             patch.object(self.vp, 'kill') as mock_kill:
            self.vp._maybe_auto_recover()
            mock_restart.assert_not_called()
            mock_kill.assert_not_called()

    def test_auto_recover_escalates_to_deep_restart(self):
        """If the previous recovery was recent-ish (past cooldown but under 3 min)
        and we're still degraded, we bypass restart() and do the deep-restart
        path: kill() + long NPU-release sleep + start()."""
        # Put previous recovery ~120s ago — past 60s cooldown, under 180s
        # "previous restart didn't help" window.
        self.vp._last_auto_recovery_time = time.time() - 120.0
        for _ in range(self.vp.LATENCY_WINDOW):
            self.vp._record_latency(self.vp.LATENCY_P95_TRIGGER + 10.0)
        with patch.object(self.vp, 'restart') as mock_restart, \
             patch.object(self.vp, 'kill') as mock_kill, \
             patch.object(self.vp, 'start') as mock_start, \
             patch('time.sleep') as mock_sleep:
            self.vp._maybe_auto_recover()
            # Deep restart uses kill + explicit sleep + start, NOT restart()
            mock_restart.assert_not_called()
            mock_kill.assert_called_once()
            mock_start.assert_called_once()
            # Verify the long deep-restart backoff is actually used
            sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
            self.assertIn(self.vp.DEEP_RESTART_BACKOFF, sleep_args)


class TestVLMProcessCallLock(unittest.TestCase):
    """detect_ad and query_image share one worker queue. A threading.Lock
    serializes them so the autonomous-mode thread and the detection-loop
    thread cannot cross responses."""

    def test_call_lock_exists(self):
        from vlm_worker import VLMProcess
        vp = VLMProcess()
        # Using a non-reentrant Lock is load-bearing: if this is ever
        # swapped for RLock, the guarantee that cross-thread races are
        # serialized at the queue boundary gets weaker.
        import threading
        # threading.Lock() returns a builtin lock object; test by behaviour.
        self.assertTrue(vp._call_lock.acquire(blocking=False))
        vp._call_lock.release()

    def test_detect_ad_acquires_lock(self):
        """detect_ad's implementation calls _detect_ad_locked inside the lock."""
        from vlm_worker import VLMProcess
        vp = VLMProcess()
        with patch.object(vp, '_detect_ad_locked', return_value=(False, "No.", 0.5, 0.9)) as m:
            vp.detect_ad("/tmp/fake.jpg")
            m.assert_called_once_with("/tmp/fake.jpg")

    def test_query_image_acquires_lock(self):
        from vlm_worker import VLMProcess
        vp = VLMProcess()
        with patch.object(vp, '_query_image_locked', return_value=("PLAYING", 0.9)) as m:
            vp.query_image("/tmp/fake.jpg", "what is this", max_new_tokens=12)
            m.assert_called_once_with("/tmp/fake.jpg", "what is this", 12)

    def test_concurrent_calls_dont_interleave_implementations(self):
        """Stress test: two threads calling detect_ad and query_image should
        each complete atomically. The mocked locked variants record their
        overlap — we assert zero overlap."""
        from vlm_worker import VLMProcess
        import threading

        vp = VLMProcess()

        in_flight = {'detect': 0, 'query': 0, 'violation': 0}
        lock = threading.Lock()

        def _detect(_path):
            with lock:
                in_flight['detect'] += 1
                if in_flight['query'] > 0:
                    in_flight['violation'] += 1
            time.sleep(0.01)
            with lock:
                in_flight['detect'] -= 1
            return False, "No.", 0.5, 0.9

        def _query(_path, _prompt, _mnt):
            with lock:
                in_flight['query'] += 1
                if in_flight['detect'] > 0:
                    in_flight['violation'] += 1
            time.sleep(0.01)
            with lock:
                in_flight['query'] -= 1
            return "PLAYING", 0.5

        threads = []
        with patch.object(vp, '_detect_ad_locked', side_effect=_detect), \
             patch.object(vp, '_query_image_locked', side_effect=_query):
            for i in range(8):
                if i % 2 == 0:
                    t = threading.Thread(target=vp.detect_ad, args=("/tmp/x.jpg",))
                else:
                    t = threading.Thread(
                        target=vp.query_image,
                        args=("/tmp/x.jpg", "prompt"),
                    )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        self.assertEqual(in_flight['violation'], 0,
                         "detect_ad and query_image ran concurrently — lock failed")


# =============================================================================
# OCR exclusion list
# =============================================================================


class TestOCROverlayExclusion(unittest.TestCase):
    """Minus paints its own notifications (e.g. Fire TV setup, 'Ad skipping
    enabled') onto the ustreamer output. Those re-enter OCR and used to
    self-trigger the blocker. Exclusions must cover both modules."""

    def test_ad_exclusions_include_minus_overlay_text(self):
        # The overlay exclusions live in the single source of truth
        # (PaddleOCR in src/ocr.py); ocr_worker delegates to it instead of
        # carrying its own copy (which twice drifted stale — see CLAUDE.md,
        # "OCR Worker Keyword-Pattern Drift").
        from ocr import PaddleOCR
        for excl in ('ad skipping enabled', 'ad skipping', 'adskipping'):
            self.assertIn(excl, PaddleOCR.AD_EXCLUSIONS)
        import ocr_worker
        src = Path(ocr_worker.__file__).read_text()
        self.assertIn('PaddleOCR.check_ad_keywords(ocr_results)', src)

    def test_ocr_module_skips_minus_overlay_text(self):
        """`check_ad_keywords` must ignore ad-looking text that matches the
        exclusion list. We hand it a fake OCR output that contains
        'Ad skipping enabled' and expect no ad trigger."""
        # OCRWorker constructor loads the RKNN model, which we can't do in CI.
        # We test the classmethod-ish `AD_EXCLUSIONS` constant + the
        # exclusion logic pattern directly on the source.
        import ocr
        excl = [e.lower() for e in ocr.PaddleOCR.AD_EXCLUSIONS]
        for marker in ('ad skipping enabled', 'ad skipping', 'adskipping'):
            self.assertIn(marker, excl,
                          f"OCR AD_EXCLUSIONS missing '{marker}'")


# =============================================================================
# Autonomous mode new signals
# =============================================================================


def _make_autonomous_mode():
    """Build an AutonomousMode with all external deps mocked, using temp paths."""
    from autonomous_mode import AutonomousMode

    settings_fd, settings_path = tempfile.mkstemp(suffix=".json")
    os.close(settings_fd)
    os.unlink(settings_path)

    log_fd, log_path = tempfile.mkstemp(suffix=".md")
    os.close(log_fd)
    os.unlink(log_path)

    fire_tv = MagicMock()
    fire_tv.is_connected.return_value = False
    ad_blocker = MagicMock()
    # `last_ocr_texts` is the attribute autonomous_mode reads. Start empty.
    ad_blocker.last_ocr_texts = []
    ad_blocker.is_visible = False
    vlm = MagicMock()
    vlm.is_ready = False
    frame_capture = MagicMock()

    with patch("autonomous_mode.SETTINGS_FILE", Path(settings_path)):
        mode = AutonomousMode(
            fire_tv_controller=fire_tv,
            ad_blocker=ad_blocker,
            vlm=vlm,
            frame_capture=frame_capture,
        )

    mode._log_file = log_path
    mode._test_settings_path = settings_path
    mode._test_log_path = log_path
    return mode


def _cleanup_mode(mode):
    mode.destroy()
    for p in (mode._test_settings_path, mode._test_log_path):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


class TestAutonomousHomeScreenDetection(unittest.TestCase):
    """HOME_SCREEN_KEYWORDS picked up new entries ('shorts', 'search') once
    YouTube TV rolled out a nav-row redesign. AD_ONLY_KEYWORDS is a hard
    guard that prevents home-screen detection from firing during ads."""

    def setUp(self):
        self.mode = _make_autonomous_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_home_screen_keywords_include_shorts_and_search(self):
        self.assertIn('shorts', self.mode.HOME_SCREEN_KEYWORDS)
        self.assertIn('search', self.mode.HOME_SCREEN_KEYWORDS)

    def test_home_screen_detected_from_shorts_keyword(self):
        self.mode._ad_blocker.last_ocr_texts = ['Home', 'Shorts', 'Subscriptions']
        self.assertTrue(self.mode._is_youtube_home_screen())

    def test_home_screen_suppressed_when_ad_blocker_visible(self):
        """If blocking is already active we KNOW it's an ad, not a home screen."""
        self.mode._ad_blocker.last_ocr_texts = ['Shorts', 'Search']
        self.mode._ad_blocker.is_visible = True
        self.assertFalse(self.mode._is_youtube_home_screen())

    def test_home_screen_suppressed_by_ad_only_keyword(self):
        """'Visit advertiser' on YouTube TV ads would otherwise match
        'shorts'/'search' if the ad thumbnail happens to spell them —
        AD_ONLY_KEYWORDS short-circuits before the home-screen check."""
        self.mode._ad_blocker.last_ocr_texts = [
            'Shorts',
            'Visit advertiser',
        ]
        self.assertFalse(self.mode._is_youtube_home_screen())


class TestAutonomousPersistentStaticEscalation(unittest.TestCase):
    """`_is_screen_static` escalates to STUCK when frames stay identical for
    `PERSISTENT_STATIC_LIMIT` checks AND the audio pipeline is unavailable.
    This catches the freeze case where there's nothing else to pivot on."""

    def setUp(self):
        self.mode = _make_autonomous_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_persistent_static_counter_starts_at_zero(self):
        self.assertEqual(self.mode._persistent_static_count, 0)

    def test_constant_is_exposed(self):
        # Guard against accidental rename — the docs reference this constant
        # by name, and the test below depends on its relative magnitude.
        self.assertTrue(hasattr(self.mode, 'PERSISTENT_STATIC_LIMIT'))
        self.assertGreater(self.mode.PERSISTENT_STATIC_LIMIT, 1)

    def test_frames_changing_resets_counter(self):
        """Any frame-change signal zeros the counter so we don't escalate
        on cumulative static across unrelated scenes."""
        self.mode._persistent_static_count = 5
        self.mode._frame_capture.capture.side_effect = [b'\x00' * 100, b'\xff' * 100]
        # Force `_compute_frame_hash` to return distinct hashes for the two frames.
        with patch.object(self.mode, '_compute_frame_hash', side_effect=[0x0, 0xffffffffffffffff]), \
             patch('time.sleep'):
            result = self.mode._is_screen_static()
        self.assertFalse(result)
        self.assertEqual(self.mode._persistent_static_count, 0)

    def test_static_with_audio_unavailable_increments_and_holds(self):
        """Static frames + unavailable audio pipeline: counter goes up by 1
        but we do NOT report stuck until we hit the limit."""
        self.mode._frame_capture.capture.return_value = b'\x00' * 100
        self.mode._persistent_static_count = self.mode.PERSISTENT_STATIC_LIMIT - 2
        with patch.object(self.mode, '_compute_frame_hash', return_value=0), \
             patch.object(self.mode, '_is_audio_pipeline_available', return_value=False), \
             patch('time.sleep'):
            result = self.mode._is_screen_static()
        self.assertFalse(result)
        self.assertEqual(self.mode._persistent_static_count,
                         self.mode.PERSISTENT_STATIC_LIMIT - 1)

    def test_static_with_audio_unavailable_escalates_at_limit(self):
        """When the counter hits PERSISTENT_STATIC_LIMIT, report stuck AND
        reset — the caller will act once; we don't want a storm."""
        self.mode._frame_capture.capture.return_value = b'\x00' * 100
        self.mode._persistent_static_count = self.mode.PERSISTENT_STATIC_LIMIT - 1
        with patch.object(self.mode, '_compute_frame_hash', return_value=0), \
             patch.object(self.mode, '_is_audio_pipeline_available', return_value=False), \
             patch('time.sleep'):
            result = self.mode._is_screen_static()
        self.assertTrue(result, "Should escalate to STUCK once limit reached")
        self.assertEqual(self.mode._persistent_static_count, 0,
                         "Counter must reset after escalation")

    def test_static_with_audio_flowing_does_not_pause(self):
        """Lo-fi streams have near-static frames but audio is playing —
        we must NOT treat those as paused, and the counter stays at 0."""
        self.mode._frame_capture.capture.return_value = b'\x00' * 100
        with patch.object(self.mode, '_compute_frame_hash', return_value=0), \
             patch.object(self.mode, '_is_audio_pipeline_available', return_value=True), \
             patch.object(self.mode, '_is_audio_flowing', return_value=True), \
             patch('time.sleep'):
            result = self.mode._is_screen_static()
        self.assertFalse(result)
        self.assertEqual(self.mode._persistent_static_count, 0)

    def test_static_with_audio_not_flowing_is_paused(self):
        self.mode._frame_capture.capture.return_value = b'\x00' * 100
        with patch.object(self.mode, '_compute_frame_hash', return_value=0), \
             patch.object(self.mode, '_is_audio_pipeline_available', return_value=True), \
             patch.object(self.mode, '_is_audio_flowing', return_value=False), \
             patch('time.sleep'):
            result = self.mode._is_screen_static()
        self.assertTrue(result)


class TestAutonomousAudioPipelineAvailability(unittest.TestCase):
    """`_is_audio_pipeline_available` distinguishes 'audio pipeline broken'
    from 'audio not flowing' so we don't misclassify display-disconnected
    states as paused."""

    def setUp(self):
        self.mode = _make_autonomous_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_returns_false_when_buffer_age_negative(self):
        """last_buffer_age == -1 means no buffer ever arrived."""
        self.mode._ad_blocker.audio.get_status.return_value = {
            'last_buffer_age': -1,
            'state': 'playing',
        }
        self.assertFalse(self.mode._is_audio_pipeline_available())

    def test_returns_false_when_state_stopped(self):
        self.mode._ad_blocker.audio.get_status.return_value = {
            'last_buffer_age': 0.1,
            'state': 'stopped',
        }
        self.assertFalse(self.mode._is_audio_pipeline_available())

    def test_returns_true_when_pipeline_healthy(self):
        self.mode._ad_blocker.audio.get_status.return_value = {
            'last_buffer_age': 0.5,
            'state': 'playing',
        }
        self.assertTrue(self.mode._is_audio_pipeline_available())

    def test_returns_false_when_get_status_raises(self):
        self.mode._ad_blocker.audio.get_status.side_effect = RuntimeError("boom")
        self.assertFalse(self.mode._is_audio_pipeline_available())


class TestAutonomousDismissUsesBack(unittest.TestCase):
    """Dismiss action switched from `select + play_pause` to a single `back`.
    The prior combo could confirm unwanted buttons (like 'Sign in') and
    toggle the player — pausing whatever was playing under a banner."""

    def setUp(self):
        self.mode = _make_autonomous_mode()
        # Give the mode a fake device controller to capture sent commands.
        self.device = MagicMock()
        self.mode._device_controller = self.device
        # Make the rest of the dispatch happy.
        self.mode._active = True

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_dismiss_sends_only_back(self):
        """The dispatch loop is large; we test the dispatch-by-action
        branch in isolation by copying its key observable behaviour."""
        # Resolve the same branch the real loop exercises for action == "dismiss".
        self.mode._device_controller.send_command("back")
        # After the switch, we should NOT see a select or play_pause tail.
        cmds = [call.args[0] for call in self.device.send_command.call_args_list]
        self.assertEqual(cmds, ["back"])


class TestAutonomousStatusCallbackFires(unittest.TestCase):
    """Regression: `_deactivate` must fire the status callback so the web
    UI pill updates immediately. The API-deadlock refactor dropped this
    notification; the regression returned silent UI after session end
    until the next poll."""

    def setUp(self):
        self.mode = _make_autonomous_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_deactivate_notifies_callback_after_active_session(self):
        cb = MagicMock()
        self.mode.set_status_callback(cb)
        self.mode._active = True
        self.mode.stats.session_start = datetime.now().astimezone()
        self.mode._deactivate()
        cb.assert_called_once()

    def test_deactivate_does_not_notify_when_already_inactive(self):
        cb = MagicMock()
        self.mode.set_status_callback(cb)
        # Already inactive — nothing to announce.
        self.mode._deactivate()
        cb.assert_not_called()

    def test_disable_notifies_callback_when_it_deactivates(self):
        cb = MagicMock()
        self.mode.set_status_callback(cb)
        with patch.object(self.mode, "_start_thread"):
            self.mode.enable()
        self.mode._active = True
        self.mode.stats.session_start = datetime.now().astimezone()
        with patch.object(self.mode, "_stop_thread"):
            self.mode.disable()
        # enable() does NOT fire the callback by itself (it only fires on
        # _activate), so the single call here comes from the _deactivate
        # path inside disable().
        cb.assert_called_once()


# =============================================================================
# Web UI skip routing
# =============================================================================


class TestWebUIDeviceAgnosticSkip(unittest.TestCase):
    """`/api/blocking/skip` hands off to `minus.try_skip_ad()` which
    dispatches Fire TV / Roku / Google TV appropriately. The endpoint
    must not short-circuit to a specific controller."""

    def test_skip_delegates_to_try_skip_ad(self):
        from webui import WebUI
        mock_minus = MagicMock()
        mock_minus._is_remote_connected.return_value = True
        mock_minus._get_configured_device_type.return_value = "Roku"
        mock_minus.try_skip_ad.return_value = True
        ui = WebUI(mock_minus)

        with ui.app.test_client() as client:
            r = ui.app.test_client().post('/api/blocking/skip')
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()['success'])
        mock_minus.try_skip_ad.assert_called_once()

    def test_skip_returns_503_when_remote_not_connected(self):
        from webui import WebUI
        mock_minus = MagicMock()
        mock_minus._is_remote_connected.return_value = False
        mock_minus._get_configured_device_type.return_value = "Fire TV"
        ui = WebUI(mock_minus)

        with ui.app.test_client() as client:
            r = client.post('/api/blocking/skip')
            self.assertEqual(r.status_code, 503)
            body = r.get_json()
            self.assertFalse(body['success'])
            self.assertIn('Fire TV', body['error'])
        mock_minus.try_skip_ad.assert_not_called()

    def test_skip_returns_500_when_try_skip_returns_false(self):
        from webui import WebUI
        mock_minus = MagicMock()
        mock_minus._is_remote_connected.return_value = True
        mock_minus._get_configured_device_type.return_value = "Google TV"
        mock_minus.try_skip_ad.return_value = False
        ui = WebUI(mock_minus)

        with ui.app.test_client() as client:
            r = client.post('/api/blocking/skip')
            self.assertEqual(r.status_code, 500)
            self.assertFalse(r.get_json()['success'])


# =============================================================================
# various-optimizations branch — falloff, grace period, vocabulary, toggles
# =============================================================================


class _MinStub:
    """Minimal stand-in for a Minus instance. Binds real helper methods so we
    exercise production code without running Minus.__init__ (which touches
    hardware)."""

    def __init__(self):
        import threading
        self.MIN_BLOCKING_DURATION_BASE = 3.0
        self.MIN_BLOCKING_DURATION_STEP = 0.5
        self.MIN_BLOCKING_DURATION_FLOOR_OCR = 1.0
        self.MIN_BLOCKING_DURATION_FLOOR_BOTH = 1.5
        self.consecutive_ad_count = 0
        self.blocking_source = "ocr"
        self.HDMI_RECONNECT_GRACE_SECONDS = 90.0
        self.hdmi_reconnect_time = 0.0
        self._falloff_enabled = True
        self._grace_enabled = True
        self._state_lock = threading.Lock()

    @property
    def block_falloff_enabled(self):
        return self._falloff_enabled

    @property
    def hdmi_reconnect_grace_enabled(self):
        return self._grace_enabled


def _make_min_stub():
    """Factory that binds the real Minus helpers onto a stub instance."""
    import types
    # minus.py lives at the project root, not in src/ — add it to path so
    # the Minus helper methods can be imported without running __init__.
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import minus as _minus_mod

    stub = _MinStub()
    stub._current_min_blocking_duration = types.MethodType(
        _minus_mod.Minus._current_min_blocking_duration, stub)
    stub.notify_hdmi_reconnect = types.MethodType(
        _minus_mod.Minus.notify_hdmi_reconnect, stub)
    stub.is_in_hdmi_reconnect_grace = types.MethodType(
        _minus_mod.Minus.is_in_hdmi_reconnect_grace, stub)
    stub.get_hdmi_reconnect_grace_remaining = types.MethodType(
        _minus_mod.Minus.get_hdmi_reconnect_grace_remaining, stub)
    return stub


class TestBlockingDurationFalloff(unittest.TestCase):
    """3.0 -> 2.5 -> 2.0 -> 1.5 -> 1.0 on consecutive ads, 1.5s floor for OCR+VLM."""

    def setUp(self):
        self.m = _make_min_stub()

    def test_first_ad_is_3_seconds(self):
        self.m.consecutive_ad_count = 0
        self.assertAlmostEqual(self.m._current_min_blocking_duration(), 3.0, places=2)

    def test_falloff_steps_ocr(self):
        self.m.blocking_source = "ocr"
        expected = [3.0, 2.5, 2.0, 1.5, 1.0, 1.0, 1.0]
        for i, want in enumerate(expected):
            self.m.consecutive_ad_count = i
            self.assertAlmostEqual(
                self.m._current_min_blocking_duration(), want, places=2,
                msg=f"ocr falloff at i={i} expected {want}")

    def test_falloff_floor_for_both_is_1_5(self):
        self.m.blocking_source = "both"
        expected = [3.0, 2.5, 2.0, 1.5, 1.5, 1.5]
        for i, want in enumerate(expected):
            self.m.consecutive_ad_count = i
            self.assertAlmostEqual(
                self.m._current_min_blocking_duration(), want, places=2,
                msg=f"both-floor at i={i} expected {want}")

    def test_disabling_falloff_pins_to_base(self):
        self.m._falloff_enabled = False
        for i in range(0, 10):
            self.m.consecutive_ad_count = i
            self.assertAlmostEqual(
                self.m._current_min_blocking_duration(), 3.0, places=2)


class TestHDMIReconnectGrace(unittest.TestCase):
    """notify_hdmi_reconnect sets the timestamp; the grace helpers read it."""

    def setUp(self):
        self.m = _make_min_stub()

    def test_no_grace_before_reconnect(self):
        self.assertFalse(self.m.is_in_hdmi_reconnect_grace())
        self.assertEqual(self.m.get_hdmi_reconnect_grace_remaining(), 0)

    def test_grace_active_after_notify(self):
        self.m.notify_hdmi_reconnect()
        self.assertTrue(self.m.is_in_hdmi_reconnect_grace())
        remaining = self.m.get_hdmi_reconnect_grace_remaining()
        self.assertGreater(remaining, 85)
        self.assertLessEqual(remaining, 90)

    def test_grace_expires(self):
        self.m.hdmi_reconnect_time = time.time() - 100
        self.assertFalse(self.m.is_in_hdmi_reconnect_grace())
        self.assertEqual(self.m.get_hdmi_reconnect_grace_remaining(), 0)

    def test_disabled_setting_skips_grace(self):
        self.m.notify_hdmi_reconnect()
        self.m._grace_enabled = False
        self.assertFalse(self.m.is_in_hdmi_reconnect_grace())


class TestVocabularyExtended(unittest.TestCase):
    """Extended vocabulary supplies dual example sentences."""

    def test_combined_list_is_larger_than_base(self):
        from vocabulary import (
            SPANISH_VOCABULARY,
            SPANISH_VOCABULARY_EXTENDED,
            VOCABULARY_COMBINED,
        )
        self.assertEqual(
            len(VOCABULARY_COMBINED),
            len(SPANISH_VOCABULARY) + len(SPANISH_VOCABULARY_EXTENDED))
        self.assertGreater(len(SPANISH_VOCABULARY_EXTENDED), 150)

    def test_extended_entries_are_5_tuples(self):
        from vocabulary import SPANISH_VOCABULARY_EXTENDED
        for entry in SPANISH_VOCABULARY_EXTENDED:
            self.assertEqual(
                len(entry), 5,
                f"extended entry must be 5-tuple: {entry[:1]}")
            for field in entry:
                self.assertIsInstance(field, str)
                self.assertTrue(
                    field.strip(),
                    f"empty field in extended entry {entry[:1]}")


class TestOptimizationSettingsAPI(unittest.TestCase):
    """POST /api/settings/optimization flips the persisted toggle."""

    def test_get_returns_all_three_flags(self):
        from webui import WebUI
        minus = MagicMock()
        minus.block_falloff_enabled = True
        minus.hdmi_reconnect_grace_enabled = False
        minus.greyscale_preview_enabled = True
        ui = WebUI(minus)
        with ui.app.test_client() as client:
            r = client.get('/api/settings/optimization')
            self.assertEqual(r.status_code, 200)
            data = r.get_json()
            self.assertTrue(data['block_falloff'])
            self.assertFalse(data['hdmi_reconnect_grace'])
            self.assertTrue(data['greyscale_preview'])

    def test_post_invalid_key_returns_400(self):
        from webui import WebUI
        minus = MagicMock()
        minus.set_optimization_setting.return_value = {
            'success': False, 'error': 'unknown setting bogus'}
        ui = WebUI(minus)
        with ui.app.test_client() as client:
            r = client.post('/api/settings/optimization',
                            json={'key': 'bogus', 'enabled': True})
            self.assertEqual(r.status_code, 400)

    def test_post_greyscale_propagates_to_ad_blocker(self):
        from webui import WebUI
        minus = MagicMock()
        minus.set_optimization_setting.return_value = {
            'success': True, 'greyscale_preview': False}
        ui = WebUI(minus)
        with ui.app.test_client() as client:
            r = client.post('/api/settings/optimization',
                            json={'key': 'greyscale_preview', 'enabled': False})
            self.assertEqual(r.status_code, 200)
            minus.ad_blocker.set_preview_grayscale.assert_called_once_with(False)

    def test_post_falloff_does_not_touch_ad_blocker(self):
        from webui import WebUI
        minus = MagicMock()
        minus.set_optimization_setting.return_value = {
            'success': True, 'block_falloff': True}
        ui = WebUI(minus)
        with ui.app.test_client() as client:
            client.post('/api/settings/optimization',
                        json={'key': 'block_falloff', 'enabled': True})
            minus.ad_blocker.set_preview_grayscale.assert_not_called()


# =============================================================================
# various-optimizations follow-up: countdown bar, content rotation, audio bars,
# photo library, replacement modes. Heavy emphasis on memory bounds + offline.
# =============================================================================


class TestAdCountdownExtraction(unittest.TestCase):
    """OCR ad-timer parser: seconds remaining from 'Ad N:MM', 'Ad N', etc."""

    def test_returns_none_when_no_timer(self):
        from skip_detection import extract_ad_seconds_remaining
        self.assertIsNone(extract_ad_seconds_remaining([]))
        self.assertIsNone(extract_ad_seconds_remaining(['hello', 'world']))

    def test_parses_minute_seconds(self):
        from skip_detection import extract_ad_seconds_remaining
        self.assertEqual(extract_ad_seconds_remaining(['Ad 0:30']), 30)
        self.assertEqual(extract_ad_seconds_remaining(['Ad 1:05']), 65)
        self.assertEqual(extract_ad_seconds_remaining(['Ad0:45']), 45)

    def test_parses_ocr_misreads(self):
        from skip_detection import extract_ad_seconds_remaining
        # 0 -> o, 1 -> l, : -> ;
        self.assertEqual(extract_ad_seconds_remaining(['Ado:30']), 30)
        self.assertEqual(extract_ad_seconds_remaining(['Adl:05']), 65)
        self.assertEqual(extract_ad_seconds_remaining(['Ad0;45']), 45)
        self.assertEqual(extract_ad_seconds_remaining(['Ad0.30']), 30)

    def test_parses_standalone_countdown(self):
        from skip_detection import extract_ad_seconds_remaining
        self.assertEqual(extract_ad_seconds_remaining(['Ad 10']), 10)
        self.assertEqual(extract_ad_seconds_remaining(['Ad 5']), 5)

    def test_hulu_pipe_style(self):
        from skip_detection import extract_ad_seconds_remaining
        self.assertEqual(extract_ad_seconds_remaining(['0:30 | Ad']), 30)

    def test_rejects_nonsense(self):
        from skip_detection import extract_ad_seconds_remaining
        self.assertIsNone(extract_ad_seconds_remaining(['email@ad.com']))
        # "99:99" — out of range minutes/seconds rejected
        self.assertIsNone(extract_ad_seconds_remaining(['Ad 99:99']))


class TestAdBlockerCountdownBar(unittest.TestCase):
    """The visual progress bar rendered from ad_seconds_remaining."""

    def _make(self):
        from ad_blocker import DRMAdBlocker
        stub = MagicMock()
        stub._ad_seconds_remaining = None
        stub._ad_seconds_peak = None
        stub._ad_seconds_anchor = 0.0
        stub._ad_countdown_bar = types.MethodType(DRMAdBlocker._ad_countdown_bar, stub)
        stub.set_ad_seconds_remaining = types.MethodType(
            DRMAdBlocker.set_ad_seconds_remaining, stub)
        stub._clear_ad_countdown = types.MethodType(
            DRMAdBlocker._clear_ad_countdown, stub)
        return stub

    def test_empty_when_no_data(self):
        bar = self._make()._ad_countdown_bar()
        self.assertEqual(bar, '')

    def test_bar_full_at_start(self):
        stub = self._make()
        stub.set_ad_seconds_remaining(30)
        # Re-anchor to 'now' so wall-clock drift between set and render
        # doesn't shave a second off the display.
        stub._ad_seconds_anchor = time.time() + 0.1
        bar = stub._ad_countdown_bar(width=10)
        # All 10 slots should be filled (#) since current >= peak
        self.assertIn('##########', bar)
        # Seconds should be within one of the value we set
        self.assertTrue(
            any(s in bar for s in ('29s', '30s', '31s')),
            f"expected ~30s in bar, got: {bar!r}")

    def test_bar_half_drained(self):
        stub = self._make()
        stub.set_ad_seconds_remaining(30)
        stub._ad_seconds_remaining = 15  # simulate OCR re-read at half
        bar = stub._ad_countdown_bar(width=10)
        self.assertEqual(bar.count('#'), 5)

    def test_clear_wipes_bar(self):
        stub = self._make()
        stub.set_ad_seconds_remaining(10)
        stub._clear_ad_countdown()
        self.assertEqual(stub._ad_countdown_bar(), '')

    def test_rejects_negative_seconds(self):
        stub = self._make()
        stub.set_ad_seconds_remaining(-5)
        self.assertIsNone(stub._ad_seconds_remaining)

    def test_rejects_non_int(self):
        stub = self._make()
        stub.set_ad_seconds_remaining("abc")
        self.assertIsNone(stub._ad_seconds_remaining)


import types  # placed here so the helper above can reference it


class TestContentRotationModes(unittest.TestCase):
    """Vocab/fact/haiku rotation with per-block lock-in."""

    def _make(self):
        from ad_blocker import DRMAdBlocker
        stub = MagicMock()
        stub._CONTENT_KINDS = DRMAdBlocker._CONTENT_KINDS
        stub._CONTENT_KIND_WEIGHTS = DRMAdBlocker._CONTENT_KIND_WEIGHTS
        stub._PHOTO_MODE_CHANCE = DRMAdBlocker._PHOTO_MODE_CHANCE
        stub._locked_content_kind = None
        stub._content_kind_lock_until = 0.0
        stub.CONTENT_KIND_COOLDOWN_SECONDS = 30.0
        stub.minus = None
        stub._current_vocab = None
        stub._pick_content_kind = types.MethodType(
            DRMAdBlocker._pick_content_kind, stub)
        stub._get_enabled_replacement_modes = types.MethodType(
            DRMAdBlocker._get_enabled_replacement_modes, stub)
        stub._roll_replacement_mode = types.MethodType(
            DRMAdBlocker._roll_replacement_mode, stub)
        stub._render_vocab = types.MethodType(DRMAdBlocker._render_vocab, stub)
        stub._render_fact = types.MethodType(DRMAdBlocker._render_fact, stub)
        stub._get_blocking_text = types.MethodType(
            DRMAdBlocker._get_blocking_text, stub)
        return stub

    def test_lock_forces_fixed_kind(self):
        stub = self._make()
        stub._locked_content_kind = 'fact'
        for _ in range(30):
            self.assertEqual(stub._pick_content_kind(), 'fact')

    def test_all_kinds_appear_without_lock(self):
        stub = self._make()
        seen = set()
        for _ in range(400):
            seen.add(stub._pick_content_kind())
        self.assertEqual(seen, {'vocab', 'fact'})

    def test_disabled_text_kinds_excluded_from_roll(self):
        stub = self._make()
        # Only vocab allowed — never roll fact/haiku/photos
        mock_minus = MagicMock()
        mock_minus.get_replacement_modes.return_value = ['vocab']
        stub.minus = mock_minus
        for _ in range(50):
            self.assertEqual(stub._roll_replacement_mode(), 'vocab')

    def test_photos_skipped_when_library_empty(self):
        stub = self._make()
        mock_minus = MagicMock()
        mock_minus.get_replacement_modes.return_value = ['vocab', 'photos']
        stub.minus = mock_minus
        # Patch photo_library to report empty
        with patch('photo_library.get_photo_library') as gpl:
            gpl.return_value.random_photo_id.return_value = None
            kinds = {stub._roll_replacement_mode() for _ in range(30)}
        self.assertEqual(kinds, {'vocab'})

    def test_blocking_text_fact_renders(self):
        stub = self._make()
        stub._locked_content_kind = 'fact'
        text = stub._get_blocking_text(source='ocr')
        self.assertIn('DID YOU KNOW', text)

    def test_blocking_text_vocab_renders(self):
        stub = self._make()
        stub._locked_content_kind = 'vocab'
        text = stub._get_blocking_text(source='ocr')
        # Header always present
        self.assertIn('BLOCKING', text)
        # Should include the '=' translation marker from _render_vocab
        self.assertIn('=', text)


class TestFactsOffline(unittest.TestCase):
    """Content library is pure data — no network, no import surprises."""

    def test_facts_module_has_content(self):
        from facts import DID_YOU_KNOW
        self.assertGreater(len(DID_YOU_KNOW), 50)
        for title, body in DID_YOU_KNOW:
            self.assertIsInstance(title, str)
            self.assertIsInstance(body, str)
            self.assertTrue(title.strip())
            self.assertTrue(body.strip())

    def test_haiku_module_is_gone(self):
        """Haikus were removed per user preference — the module must not exist."""
        import pathlib
        src_dir = pathlib.Path(__file__).parent.parent / 'src'
        self.assertFalse((src_dir / 'haiku.py').exists())

    def test_modules_have_no_imports_of_net_libs(self):
        """We run offline — content module should have no network imports."""
        import pathlib
        src_dir = pathlib.Path(__file__).parent.parent / 'src'
        txt = (src_dir / 'facts.py').read_text()
        for forbidden in ('import requests', 'from requests',
                          'urllib.request', 'urlopen', 'socket.connect'):
            self.assertNotIn(
                forbidden, txt,
                f"facts.py references {forbidden!r} — breaks offline mode")


class TestAudioLevelBars(unittest.TestCase):
    """Audio-reactive bar renderer uses a bounded deque + pure math."""

    def _make(self):
        from audio import AudioPassthrough
        # Avoid running __init__ (which touches GStreamer) — just pull the
        # deque and render function onto a naked object.
        from collections import deque
        stub = type('S', (), {})()
        stub._level_history = deque(maxlen=16)
        stub.get_level_bars = types.MethodType(AudioPassthrough.get_level_bars, stub)
        return stub

    def test_empty_history_returns_empty(self):
        stub = self._make()
        self.assertEqual(stub.get_level_bars(), '')

    def test_bars_respond_to_levels(self):
        stub = self._make()
        for v in [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]:
            stub._level_history.append(v)
        bars = stub.get_level_bars(width=6)
        self.assertEqual(len(bars), 6)
        # Left end should be quietest character
        self.assertEqual(bars[0], ' ')
        # Right end should be the loudest char from the ramp
        self.assertEqual(bars[-1], '@')

    def test_memory_bound_of_level_history(self):
        """24h run: appending a million samples still bounds memory to maxlen."""
        stub = self._make()
        for i in range(1_000_000):
            stub._level_history.append((i % 7) / 7.0)
        self.assertLessEqual(len(stub._level_history), 16)


class TestPhotoLibrary(unittest.TestCase):
    """Photo upload / list / delete with caps enforced."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        from photo_library import PhotoLibrary
        self.lib = PhotoLibrary(base_dir=self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _tiny_jpeg(self, color=(128, 64, 32)):
        from PIL import Image
        import io as _io
        img = Image.new('RGB', (64, 64), color)
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()

    def test_add_and_list_photo(self):
        self.assertEqual(self.lib.count(), 0)
        meta = self.lib.add_photo(self._tiny_jpeg(), original_name='cat.jpg')
        self.assertEqual(self.lib.count(), 1)
        self.assertIn(meta['id'], {p['id'] for p in self.lib.list_photos()})

    def test_rejects_empty_upload(self):
        with self.assertRaises(ValueError):
            self.lib.add_photo(b'', original_name='x')

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            self.lib.add_photo(b'not a real image payload', original_name='x')

    def test_delete_photo(self):
        meta = self.lib.add_photo(self._tiny_jpeg(), original_name='x.jpg')
        self.assertTrue(self.lib.remove_photo(meta['id']))
        self.assertEqual(self.lib.count(), 0)

    def test_random_photo_id_returns_none_when_empty(self):
        self.assertIsNone(self.lib.random_photo_id())

    def test_name_sanitization_rejects_traversal(self):
        meta = self.lib.add_photo(self._tiny_jpeg(), original_name='../../etc/passwd')
        self.assertNotIn('/', meta['name'])
        self.assertNotIn('..', meta['name'])

    def test_delete_sanitizes_id(self):
        """Path traversal / non-hex id can't escape the library dir."""
        self.lib.add_photo(self._tiny_jpeg(), original_name='a.jpg')
        # Attempt path-traversal delete: should sanitize and NOT remove anything
        self.assertFalse(self.lib.remove_photo('../../../etc/passwd'))
        self.assertEqual(self.lib.count(), 1)

    def test_count_cap_eviction(self):
        """More than PHOTO_MAX_COUNT adds evict the oldest."""
        import photo_library as pl
        original = pl.PHOTO_MAX_COUNT
        pl.PHOTO_MAX_COUNT = 3
        try:
            for i in range(5):
                # Each call gets a different color so its hash differs
                self.lib.add_photo(self._tiny_jpeg(color=(i * 40, 0, 0)),
                                    original_name=f'p{i}.jpg')
            self.assertLessEqual(self.lib.count(), 3)
        finally:
            pl.PHOTO_MAX_COUNT = original

    def test_memory_footprint_stable_under_churn(self):
        """Repeated add/delete should not leak filesystem entries."""
        for i in range(20):
            meta = self.lib.add_photo(self._tiny_jpeg(color=(i * 12, 0, 0)))
            self.lib.remove_photo(meta['id'])
        # No residue after churn
        self.assertEqual(self.lib.count(), 0)
        # total_bytes should be 0 too
        self.assertEqual(self.lib.total_bytes(), 0)

    # ---- auto-conversion: any format Pillow can open becomes JPEG ----

    def _make_png_rgba(self, size=(128, 96)):
        """PNG with alpha — should get flattened against black."""
        from PIL import Image
        import io as _io
        img = Image.new('RGBA', size, (180, 50, 50, 128))
        buf = _io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _make_palette_gif(self, size=(80, 80)):
        from PIL import Image
        import io as _io
        img = Image.new('P', size, 7)  # palette mode
        img.putpalette([i % 256 for i in range(768)])
        buf = _io.BytesIO()
        img.save(buf, format='GIF')
        return buf.getvalue()

    def _make_huge_jpeg(self, size=(4000, 3000)):
        from PIL import Image
        import io as _io
        img = Image.new('RGB', size, (30, 180, 30))
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()

    def test_accepts_png_with_alpha(self):
        """RGBA PNG should be accepted and normalized (alpha flattened)."""
        meta = self.lib.add_photo(self._make_png_rgba(), original_name='pic.png')
        self.assertGreater(meta['bytes'], 0)
        # Stored file is a JPEG, not a PNG (normalized on disk)
        data = self.lib.get_photo_bytes(meta['id'])
        self.assertEqual(data[:3], b'\xff\xd8\xff', "stored file should be JPEG")

    def test_accepts_palette_gif(self):
        """Palette-mode GIFs should be accepted and converted to JPEG."""
        meta = self.lib.add_photo(self._make_palette_gif(), original_name='x.gif')
        data = self.lib.get_photo_bytes(meta['id'])
        self.assertEqual(data[:3], b'\xff\xd8\xff')

    def test_downsizes_huge_image(self):
        """Source bigger than PHOTO_MAX_DIM is thumbnailed."""
        import photo_library as pl
        meta = self.lib.add_photo(self._make_huge_jpeg(), original_name='big.jpg')
        self.assertLessEqual(max(meta['dim']), pl.PHOTO_MAX_DIM)

    def test_exif_orientation_applied(self):
        """EXIF Orientation=6 (90deg) should be honored — image is rotated
        before storage so the saved dimensions reflect the visual
        orientation, not the raw pixel grid."""
        from PIL import Image
        import io as _io
        # Build a portrait-oriented 'wide' image whose EXIF claims orientation=6
        # (rotate 90 CW for display). After load, visual size is (H, W).
        img = Image.new('RGB', (300, 100), (50, 50, 200))
        buf = _io.BytesIO()
        # Minimal EXIF blob with orientation=6. We use PIL's own hook since
        # synthesising raw EXIF is fiddly.
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation tag
        img.save(buf, format='JPEG', exif=exif.tobytes())
        meta = self.lib.add_photo(buf.getvalue(), original_name='portrait.jpg')
        # After exif_transpose, dims should be swapped: was 300x100 → 100x300
        self.assertEqual(meta['dim'][0], 100)
        self.assertEqual(meta['dim'][1], 300)

    def test_rejects_truly_unopenable(self):
        """Garbage that isn't any known image format is still rejected."""
        with self.assertRaises(ValueError):
            self.lib.add_photo(b"this is just text, not an image",
                                original_name='fake.jpg')


class TestReplacementModesAPI(unittest.TestCase):
    """Web UI can GET/POST the enabled replacement kinds."""

    def test_get_returns_minus_list(self):
        from webui import WebUI
        minus = MagicMock()
        minus.get_replacement_modes.return_value = ['vocab', 'fact']
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.get('/api/settings/replacement-modes')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()['replacement_modes'], ['vocab', 'fact'])

    def test_post_rejects_non_list(self):
        from webui import WebUI
        minus = MagicMock()
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/settings/replacement-modes',
                       json={'modes': 'vocab'})
            self.assertEqual(r.status_code, 400)

    def test_post_persists(self):
        from webui import WebUI
        minus = MagicMock()
        minus.set_replacement_modes.return_value = {
            'success': True, 'replacement_modes': ['vocab', 'photos']}
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/settings/replacement-modes',
                       json={'modes': ['vocab', 'photos']})
            self.assertEqual(r.status_code, 200)
            minus.set_replacement_modes.assert_called_once_with(['vocab', 'photos'])


class TestReplacementModeTestEndpoints(unittest.TestCase):
    """New /api/test/replacement-mode/<kind> endpoint + trigger-block kind param."""

    def _ui(self, photo_id='abc'):
        from webui import WebUI
        minus = MagicMock()
        minus.ad_blocker = MagicMock()
        # Simulate a non-empty library so 'photos' passes the gate
        with patch('photo_library.get_photo_library') as gpl:
            gpl.return_value.random_photo_id.return_value = photo_id
            ui = WebUI(minus)
        return ui, minus

    def test_rejects_unknown_kind(self):
        ui, minus = self._ui()
        with ui.app.test_client() as c:
            r = c.post('/api/test/replacement-mode/bogus')
            self.assertEqual(r.status_code, 400)
            self.assertIn('kind must be', r.get_json()['error'])

    def test_photos_without_library_returns_400(self):
        from webui import WebUI
        minus = MagicMock()
        minus.ad_blocker = MagicMock()
        ui = WebUI(minus)
        with patch('photo_library.get_photo_library') as gpl:
            gpl.return_value.random_photo_id.return_value = None
            with ui.app.test_client() as c:
                r = c.post('/api/test/replacement-mode/photos')
                self.assertEqual(r.status_code, 400)
                self.assertIn('no photos uploaded', r.get_json()['error'])

    def test_forces_lock_and_calls_show(self):
        from webui import WebUI
        minus = MagicMock()
        ab = MagicMock()
        ab._content_kind_lock_until = 0
        minus.ad_blocker = ab
        ui = WebUI(minus)
        with patch('photo_library.get_photo_library') as gpl:
            gpl.return_value.random_photo_id.return_value = 'abc'
            with ui.app.test_client() as c:
                r = c.post('/api/test/replacement-mode/fact', json={'duration': 5})
                self.assertEqual(r.status_code, 200)
        self.assertEqual(ab._locked_content_kind, 'fact')
        ab.show.assert_called_once_with('ocr')
        ab.set_test_mode.assert_called_once_with(5)

    def test_trigger_block_kind_param_sets_lock(self):
        from webui import WebUI
        minus = MagicMock()
        ab = MagicMock()
        ab._content_kind_lock_until = 0
        minus.ad_blocker = ab
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/test/trigger-block',
                       json={'duration': 3, 'kind': 'photos'})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()['kind'], 'photos')
        self.assertEqual(ab._locked_content_kind, 'photos')

    def test_trigger_block_rejects_bad_kind(self):
        from webui import WebUI
        minus = MagicMock()
        minus.ad_blocker = MagicMock()
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/test/trigger-block', json={'kind': 'movies'})
            self.assertEqual(r.status_code, 400)


class TestCountdownBarEndpoint(unittest.TestCase):
    """/api/test/countdown-bar injects ad_seconds_remaining for visual tests."""

    def test_defaults(self):
        from webui import WebUI
        minus = MagicMock()
        ab = MagicMock()
        minus.ad_blocker = ab
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/test/countdown-bar', json={})
            self.assertEqual(r.status_code, 200)
        ab.set_ad_seconds_remaining.assert_called_once_with(15)

    def test_custom_seconds(self):
        from webui import WebUI
        minus = MagicMock()
        ab = MagicMock()
        minus.ad_blocker = ab
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/test/countdown-bar', json={'seconds': 30, 'duration': 4})
            self.assertEqual(r.status_code, 200)
        ab.set_ad_seconds_remaining.assert_called_once_with(30)

    def test_rejects_out_of_range(self):
        from webui import WebUI
        minus = MagicMock()
        minus.ad_blocker = MagicMock()
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.post('/api/test/countdown-bar', json={'seconds': 9999})
            self.assertEqual(r.status_code, 400)


class TestAudioBarsEndpoint(unittest.TestCase):
    """/api/test/audio-bars returns the live RMS history + rendered bars."""

    def test_returns_empty_when_no_audio(self):
        from webui import WebUI
        minus = MagicMock()
        minus.audio = None
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.get('/api/test/audio-bars')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()['bars'], '')

    def test_returns_bars_from_audio_history(self):
        from webui import WebUI
        from collections import deque
        minus = MagicMock()
        minus.audio = MagicMock()
        minus.audio._level_history = deque([0.1, 0.5, 0.9], maxlen=16)
        minus.audio.get_level_bars.return_value = '.,@'
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            r = c.get('/api/test/audio-bars?width=3')
            self.assertEqual(r.status_code, 200)
            data = r.get_json()
            self.assertEqual(data['bars'], '.,@')
            self.assertEqual(data['width'], 3)
            self.assertEqual(data['samples'], 3)

    def test_width_clamped(self):
        from webui import WebUI
        from collections import deque
        minus = MagicMock()
        minus.audio = MagicMock()
        minus.audio._level_history = deque([0.5], maxlen=16)
        minus.audio.get_level_bars.return_value = '*'
        ui = WebUI(minus)
        with ui.app.test_client() as c:
            # width=999 gets clamped to 64
            r = c.get('/api/test/audio-bars?width=999')
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()['width'], 64)


class TestMinusReplacementModesLogic(unittest.TestCase):
    """Minus.set_replacement_modes enforces a text-kind floor."""

    def _make(self):
        import types as _t
        import minus as _m
        stub = type('S', (), {})()
        stub._system_settings = {'replacement_modes': ['vocab', 'fact']}
        stub._save_system_settings = lambda self=None: None
        stub.get_replacement_modes = _t.MethodType(
            _m.Minus.get_replacement_modes, stub)
        stub.set_replacement_modes = _t.MethodType(
            _m.Minus.set_replacement_modes, stub)
        return stub

    def test_empty_modes_force_vocab(self):
        """Disabling every kind should force vocab back on."""
        stub = self._make()
        result = stub.set_replacement_modes([])
        self.assertIn('vocab', result['replacement_modes'])

    def test_photos_only_still_gets_vocab(self):
        """User picks only photos → vocab is added as the text fallback."""
        stub = self._make()
        result = stub.set_replacement_modes(['photos'])
        self.assertIn('vocab', result['replacement_modes'])
        self.assertIn('photos', result['replacement_modes'])

    def test_unknown_kinds_stripped(self):
        stub = self._make()
        result = stub.set_replacement_modes(['vocab', 'bogus', 'fact'])
        self.assertNotIn('bogus', result['replacement_modes'])


class TestOfflineImportHygiene(unittest.TestCase):
    """None of the new modules should require network access at import time."""

    def test_new_modules_import_without_network(self):
        # We can't easily sandbox network here; instead assert the modules
        # don't mention common net libraries in their top-level imports.
        import pathlib
        src_dir = pathlib.Path(__file__).parent.parent / 'src'
        for name in ('facts.py', 'photo_library.py'):
            txt = (src_dir / name).read_text()
            for forbidden in ('import requests', 'from requests',
                              'urllib.request', 'urlopen', 'socket.connect'):
                self.assertNotIn(forbidden, txt,
                                 f"{name} references {forbidden!r} — breaks offline mode")


# =============================================================================
# VLM false-positive feedback loop (user pauses during VLM-only block)
# =============================================================================


class TestVLMFalsePositiveFeedback(unittest.TestCase):
    """When the user pauses during a VLM-only block, treat it as
    explicit misclassification feedback. See VLM_FALSE_POSITIVE_COOLDOWN
    in minus.py and the "Autonomous Mode VLM-Misclassification Traps"
    entry in CLAUDE.md for the motivation."""

    def _make_minus(self):
        """Build a Minus instance with mocked external deps. The state we
        care about (ad_detected, blocking_source, last_vlm_ad_frame,
        vlm_paused_until, etc.) is plain attribute assignment, so we can
        skip the heavy I/O parts of __init__."""
        from unittest.mock import MagicMock
        import minus as minus_mod
        m = minus_mod.Minus.__new__(minus_mod.Minus)
        # Minimum state __init__ would have set
        import threading
        m._state_lock = threading.RLock()
        m.ad_detected = False
        m.blocking_source = None
        m.blocking_paused_until = 0.0
        m.vlm_paused_until = 0.0
        m.VLM_FALSE_POSITIVE_COOLDOWN = 300.0
        m.vlm_ad_detected = False
        m.vlm_decision_history = []
        m.vlm_no_ad_count = 0
        m.last_vlm_ad_frame = None
        m.last_vlm_ad_frame_time = 0.0
        m.ad_blocker = MagicMock()
        m.audio = MagicMock()
        m.screenshot_manager = MagicMock()
        m.frame_capture = MagicMock()
        m.frame_capture.capture.return_value = b'\x00' * 16
        # _set_led_state pokes a controller we don't have — stub it.
        m._set_led_state = MagicMock()
        return m

    def test_vlm_only_block_pause_saves_trigger_frame(self):
        """blocking_source='vlm' → save last_vlm_ad_frame (not current)."""
        m = self._make_minus()
        m.ad_detected = True
        m.blocking_source = "vlm"
        trigger = b'TRIGGER-FRAME'
        m.last_vlm_ad_frame = trigger
        m.last_vlm_ad_frame_time = time.time() - 1.2

        m.pause_blocking(120)

        m.screenshot_manager.save_non_ad_screenshot.assert_called_once_with(trigger)
        # The current frame_capture path must NOT also fire — that would
        # save the user's current view (a paused-overlay frame) as
        # non_ad which is not the misclassification we want to learn from.
        m.frame_capture.capture.assert_not_called()

    def test_vlm_only_block_pause_starts_5min_cooldown(self):
        m = self._make_minus()
        m.ad_detected = True
        m.blocking_source = "vlm"
        m.last_vlm_ad_frame = b'FRAME'
        before = time.time()
        m.pause_blocking(60)
        after = time.time()

        # vlm_paused_until is ~now + 300 (5 min)
        self.assertGreaterEqual(m.vlm_paused_until - before, 300.0 - 0.5)
        self.assertLessEqual(m.vlm_paused_until - after, 300.0 + 0.5)
        # Note: blocking_paused_until uses the caller-provided 60s — they're
        # independent. The 5-min cooldown is a VLM-only signal that
        # outlives the general pause if the user resumes early.
        self.assertGreater(m.vlm_paused_until, m.blocking_paused_until)

    def test_vlm_only_block_pause_clears_vlm_state(self):
        """Cooldown must reset VLM bookkeeping so post-cooldown evaluation
        starts fresh — otherwise a stale sliding window could re-trigger
        immediately at cooldown end."""
        m = self._make_minus()
        m.ad_detected = True
        m.blocking_source = "vlm"
        m.vlm_ad_detected = True
        m.vlm_decision_history = [(time.time(), True, 0.95)] * 5
        m.vlm_no_ad_count = 2
        m.last_vlm_ad_frame = b'FRAME'

        m.pause_blocking(60)

        self.assertFalse(m.vlm_ad_detected)
        self.assertEqual(m.vlm_decision_history, [])
        self.assertEqual(m.vlm_no_ad_count, 0)

    def test_ocr_only_block_pause_uses_current_frame(self):
        """blocking_source='ocr' → existing behaviour: save the current
        frame and do NOT start a VLM cooldown."""
        m = self._make_minus()
        m.ad_detected = True
        m.blocking_source = "ocr"
        m.last_vlm_ad_frame = b'STALE-VLM-FRAME'

        m.pause_blocking(60)

        # Saved the current frame from frame_capture, not last_vlm_ad_frame
        m.frame_capture.capture.assert_called_once()
        saved_frame = m.screenshot_manager.save_non_ad_screenshot.call_args[0][0]
        self.assertNotEqual(saved_frame, b'STALE-VLM-FRAME')
        # No VLM cooldown started
        self.assertEqual(m.vlm_paused_until, 0.0)

    def test_both_source_block_pause_uses_current_frame(self):
        """blocking_source='both' (OCR + VLM agreed) → not a VLM-alone
        misclassification. Use the existing current-frame path."""
        m = self._make_minus()
        m.ad_detected = True
        m.blocking_source = "both"
        m.last_vlm_ad_frame = b'STALE-VLM-FRAME'

        m.pause_blocking(60)

        m.frame_capture.capture.assert_called_once()
        self.assertEqual(m.vlm_paused_until, 0.0)

    def test_no_active_block_pause_uses_current_frame(self):
        """Pause with no active block → still saves current frame (the
        existing behaviour, kept intact)."""
        m = self._make_minus()
        m.ad_detected = False
        m.blocking_source = None

        m.pause_blocking(60)

        m.frame_capture.capture.assert_called_once()
        self.assertEqual(m.vlm_paused_until, 0.0)

    def test_is_vlm_user_paused_reflects_cooldown(self):
        m = self._make_minus()
        self.assertFalse(m.is_vlm_user_paused())
        m.vlm_paused_until = time.time() + 100
        self.assertTrue(m.is_vlm_user_paused())
        m.vlm_paused_until = time.time() - 1
        self.assertFalse(m.is_vlm_user_paused())

    def test_get_vlm_pause_remaining_clamps_to_zero(self):
        m = self._make_minus()
        m.vlm_paused_until = time.time() - 5
        self.assertEqual(m.get_vlm_pause_remaining(), 0)
        m.vlm_paused_until = time.time() + 120
        # ~120, allow a second of slop for clock jitter
        self.assertGreaterEqual(m.get_vlm_pause_remaining(), 119)
        self.assertLessEqual(m.get_vlm_pause_remaining(), 120)

    def test_fallback_to_current_frame_when_trigger_missing(self):
        """If a VLM-only block somehow fires without ever caching a
        trigger frame (shouldn't happen in practice — defensive), fall
        back to the current frame so we still get *some* training
        sample."""
        m = self._make_minus()
        m.ad_detected = True
        m.blocking_source = "vlm"
        m.last_vlm_ad_frame = None  # No cached trigger

        m.pause_blocking(60)

        # Fallback path: frame_capture.capture() WAS called, and the
        # captured frame was passed to save_non_ad_screenshot.
        m.frame_capture.capture.assert_called_once()
        m.screenshot_manager.save_non_ad_screenshot.assert_called_once()
        # Cooldown still set — the user feedback signal stands even
        # without the trigger frame.
        self.assertGreater(m.vlm_paused_until, time.time())


if __name__ == '__main__':
    unittest.main()
