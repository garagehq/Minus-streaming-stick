#!/usr/bin/env python3
"""
Comprehensive test suite for AutonomousMode and AutonomousModeStats.

Run with: python3 tests/test_autonomous_mode.py
Or:       python3 -m pytest tests/test_autonomous_mode.py -v
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from autonomous_mode import AutonomousMode, AutonomousModeStats, YOUTUBE_PACKAGES, ET


def _make_mode(**kwargs):
    """Create an AutonomousMode with mocked dependencies and temp files.

    Returns (mode, settings_tmpfile_path, log_tmpfile_path).
    Caller should clean up the temp files.
    """
    fire_tv = kwargs.pop("fire_tv", MagicMock())
    fire_tv.is_connected.return_value = kwargs.pop("fire_tv_connected", False)

    vlm = kwargs.pop("vlm", MagicMock())
    vlm.is_ready = kwargs.pop("vlm_ready", False)

    frame_capture = kwargs.pop("frame_capture", MagicMock())

    # Use temp files for settings and log so tests don't touch real paths
    settings_fd, settings_path = tempfile.mkstemp(suffix=".json")
    os.close(settings_fd)
    os.unlink(settings_path)  # start with no file

    log_fd, log_path = tempfile.mkstemp(suffix=".md")
    os.close(log_fd)
    os.unlink(log_path)  # start with no file

    with patch("autonomous_mode.SETTINGS_FILE", Path(settings_path)):
        mode = AutonomousMode(
            fire_tv_controller=fire_tv,
            ad_blocker=MagicMock(),
            vlm=vlm,
            frame_capture=frame_capture,
        )

    mode._log_file = log_path
    # Store paths for cleanup and for assertions
    mode._test_settings_path = settings_path
    mode._test_log_path = log_path
    return mode


def _cleanup_mode(mode):
    """Destroy mode and remove temp files."""
    mode.destroy()
    for p in (mode._test_settings_path, mode._test_log_path):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


# =============================================================================
# AutonomousModeStats Tests
# =============================================================================


class TestAutonomousModeStats(unittest.TestCase):
    """Tests for AutonomousModeStats."""

    def test_initial_state(self):
        stats = AutonomousModeStats()
        self.assertIsNone(stats.session_start)
        self.assertIsNone(stats.session_end)
        self.assertEqual(stats.videos_played, 0)
        self.assertEqual(stats.ads_detected, 0)
        self.assertEqual(stats.ads_skipped, 0)
        self.assertEqual(stats.errors, 0)
        self.assertIsNone(stats.last_activity)

    def test_to_dict_initial(self):
        stats = AutonomousModeStats()
        d = stats.to_dict()
        self.assertIsNone(d["session_start"])
        self.assertIsNone(d["session_end"])
        self.assertEqual(d["videos_played"], 0)
        self.assertEqual(d["ads_detected"], 0)
        self.assertEqual(d["ads_skipped"], 0)
        self.assertEqual(d["errors"], 0)
        self.assertIsNone(d["last_activity"])
        self.assertEqual(d["duration_minutes"], 0)

    def test_to_dict_with_session(self):
        stats = AutonomousModeStats()
        start = datetime(2026, 4, 8, 1, 0, 0, tzinfo=ET)
        end = datetime(2026, 4, 8, 2, 30, 0, tzinfo=ET)
        stats.session_start = start
        stats.session_end = end
        stats.videos_played = 5
        stats.ads_detected = 10
        stats.ads_skipped = 8

        d = stats.to_dict()
        self.assertEqual(d["session_start"], start.isoformat())
        self.assertEqual(d["session_end"], end.isoformat())
        self.assertEqual(d["videos_played"], 5)
        self.assertEqual(d["ads_detected"], 10)
        self.assertEqual(d["ads_skipped"], 8)
        self.assertEqual(d["duration_minutes"], 90)

    def test_duration_no_session_start(self):
        stats = AutonomousModeStats()
        self.assertEqual(stats._get_duration_minutes(), 0)

    def test_duration_ongoing_session(self):
        stats = AutonomousModeStats()
        stats.session_start = datetime.now(ET) - timedelta(minutes=45)
        # No session_end => uses now
        duration = stats._get_duration_minutes()
        self.assertGreaterEqual(duration, 44)
        self.assertLessEqual(duration, 46)

    def test_duration_completed_session(self):
        stats = AutonomousModeStats()
        stats.session_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=ET)
        stats.session_end = datetime(2026, 1, 1, 3, 15, 0, tzinfo=ET)
        self.assertEqual(stats._get_duration_minutes(), 195)

    def test_reset(self):
        stats = AutonomousModeStats()
        stats.session_start = datetime.now(ET)
        stats.videos_played = 10
        stats.ads_detected = 5
        stats.ads_skipped = 3
        stats.errors = 2
        stats.last_activity = datetime.now(ET)

        stats.reset()

        self.assertIsNone(stats.session_start)
        self.assertIsNone(stats.session_end)
        self.assertEqual(stats.videos_played, 0)
        self.assertEqual(stats.ads_detected, 0)
        self.assertEqual(stats.ads_skipped, 0)
        self.assertEqual(stats.errors, 0)
        self.assertIsNone(stats.last_activity)

    def test_to_dict_last_activity(self):
        stats = AutonomousModeStats()
        now = datetime.now(ET)
        stats.last_activity = now
        d = stats.to_dict()
        self.assertEqual(d["last_activity"], now.isoformat())


# =============================================================================
# Schedule Management Tests
# =============================================================================


class TestScheduleManagement(unittest.TestCase):
    """Tests for schedule-related methods."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_default_schedule(self):
        self.assertEqual(self.mode._start_hour, 0)
        self.assertEqual(self.mode._end_hour, 8)
        self.assertFalse(self.mode._always_on)

    def test_set_schedule_normal_range(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            result = self.mode.set_schedule(9, 17)
        self.assertEqual(self.mode._start_hour, 9)
        self.assertEqual(self.mode._end_hour, 17)
        self.assertFalse(self.mode._always_on)
        self.assertIn("schedule", result)
        self.assertEqual(result["schedule"], "09:00-17:00")

    def test_set_schedule_overnight_range(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            result = self.mode.set_schedule(22, 6)
        self.assertEqual(self.mode._start_hour, 22)
        self.assertEqual(self.mode._end_hour, 6)
        self.assertEqual(result["schedule"], "22:00-06:00")

    def test_set_schedule_always_on(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            result = self.mode.set_schedule(0, 0, always_on=True)
        self.assertTrue(self.mode._always_on)
        self.assertEqual(result["schedule"], "24/7")

    def test_set_schedule_same_hour(self):
        """Same start and end hour means a zero-length window."""
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            self.mode.set_schedule(12, 12)
        self.assertEqual(self.mode._start_hour, 12)
        self.assertEqual(self.mode._end_hour, 12)

    def test_set_schedule_clamps_negative(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            self.mode.set_schedule(-5, 25)
        self.assertEqual(self.mode._start_hour, 0)
        self.assertEqual(self.mode._end_hour, 23)

    def test_set_schedule_clamps_over_23(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            self.mode.set_schedule(100, 200)
        self.assertEqual(self.mode._start_hour, 23)
        self.assertEqual(self.mode._end_hour, 23)

    # -- is_scheduled_time --

    def test_is_scheduled_time_always_on(self):
        self.mode._always_on = True
        self.assertTrue(self.mode.is_scheduled_time())

    def test_is_scheduled_time_normal_range_inside(self):
        """9:00-17:00, current time is 12:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 12, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertTrue(self.mode.is_scheduled_time())

    def test_is_scheduled_time_normal_range_before(self):
        """9:00-17:00, current time is 7:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 7, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertFalse(self.mode.is_scheduled_time())

    def test_is_scheduled_time_normal_range_after(self):
        """9:00-17:00, current time is 18:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 18, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertFalse(self.mode.is_scheduled_time())

    def test_is_scheduled_time_normal_range_at_start(self):
        """9:00-17:00, current time is exactly 9:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 9, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertTrue(self.mode.is_scheduled_time())

    def test_is_scheduled_time_normal_range_at_end(self):
        """9:00-17:00, current time is exactly 17:00 -> should be False (< end, not <=)."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 17, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertFalse(self.mode.is_scheduled_time())

    def test_is_scheduled_time_overnight_range_late_night(self):
        """22:00-06:00, current time is 23:00."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 23, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertTrue(self.mode.is_scheduled_time())

    def test_is_scheduled_time_overnight_range_early_morning(self):
        """22:00-06:00, current time is 3:00."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 3, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertTrue(self.mode.is_scheduled_time())

    def test_is_scheduled_time_overnight_range_daytime(self):
        """22:00-06:00, current time is 14:00 -> outside."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 14, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertFalse(self.mode.is_scheduled_time())

    def test_is_scheduled_time_overnight_at_start(self):
        """22:00-06:00, current time is 22:00 -> in window."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 22, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertTrue(self.mode.is_scheduled_time())

    def test_is_scheduled_time_overnight_at_end(self):
        """22:00-06:00, current time is 06:00 -> outside (< 6 is false)."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 6, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertFalse(self.mode.is_scheduled_time())

    def test_is_scheduled_time_same_hour(self):
        """12:00-12:00 -> zero-length window, always False."""
        self.mode._start_hour = 12
        self.mode._end_hour = 12
        with patch("autonomous_mode.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 8, 12, 0, 0, tzinfo=ET)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            self.assertFalse(self.mode.is_scheduled_time())

    # -- get_next_window --

    def test_get_next_window_always_on(self):
        self.mode._always_on = True
        start, end = self.mode.get_next_window()
        # end should be ~365 days from now
        self.assertGreater(end - start, timedelta(days=364))

    def test_get_next_window_normal_before_window(self):
        """9:00-17:00, now is 7:00 -> window is today 9:00-17:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 7, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = self.mode.get_next_window()
            self.assertEqual(start.hour, 9)
            self.assertEqual(end.hour, 17)
            self.assertEqual(start.day, 8)

    def test_get_next_window_normal_during_window(self):
        """9:00-17:00, now is 12:00 -> window is today 9:00-17:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 12, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = self.mode.get_next_window()
            self.assertEqual(start.hour, 9)
            self.assertEqual(end.hour, 17)
            self.assertEqual(start.day, 8)

    def test_get_next_window_normal_after_window(self):
        """9:00-17:00, now is 20:00 -> window is tomorrow 9:00-17:00."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 20, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = self.mode.get_next_window()
            self.assertEqual(start.hour, 9)
            self.assertEqual(end.hour, 17)
            self.assertEqual(start.day, 9)

    def test_get_next_window_overnight_during_late(self):
        """22:00-06:00, now is 23:00 -> start today 22:00, end tomorrow 06:00."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 23, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = self.mode.get_next_window()
            self.assertEqual(start.hour, 22)
            self.assertEqual(start.day, 8)
            self.assertEqual(end.hour, 6)
            self.assertEqual(end.day, 9)

    def test_get_next_window_overnight_during_early_morning(self):
        """22:00-06:00, now is 3:00 -> start yesterday 22:00, end today 06:00."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 3, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = self.mode.get_next_window()
            self.assertEqual(start.hour, 22)
            self.assertEqual(start.day, 7)
            self.assertEqual(end.hour, 6)
            self.assertEqual(end.day, 8)

    def test_get_next_window_overnight_between_windows(self):
        """22:00-06:00, now is 14:00 -> next window starts today 22:00, ends tomorrow 06:00."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 14, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            start, end = self.mode.get_next_window()
            self.assertEqual(start.hour, 22)
            self.assertEqual(start.day, 8)
            self.assertEqual(end.hour, 6)
            self.assertEqual(end.day, 9)

    # -- get_time_until_window --

    def test_time_until_window_returns_none_when_inside(self):
        """During scheduled time, should return None."""
        self.mode._always_on = True
        result = self.mode.get_time_until_window()
        self.assertIsNone(result)

    def test_time_until_window_returns_timedelta_when_outside(self):
        """Outside scheduled time, should return a timedelta."""
        self.mode._start_hour = 9
        self.mode._end_hour = 17
        self.mode._always_on = False
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 7, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.mode.get_time_until_window()
            self.assertIsNotNone(result)
            self.assertIsInstance(result, timedelta)
            # Should be about 2 hours
            self.assertGreater(result.total_seconds(), 3600)
            self.assertLessEqual(result.total_seconds(), 7200 + 60)

    def test_time_until_window_overnight_daytime(self):
        """22:00-06:00, now is 14:00 -> ~8 hours until window."""
        self.mode._start_hour = 22
        self.mode._end_hour = 6
        self.mode._always_on = False
        with patch("autonomous_mode.datetime") as mock_dt:
            now = datetime(2026, 4, 8, 14, 0, 0, tzinfo=ET)
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = self.mode.get_time_until_window()
            self.assertIsNotNone(result)
            # About 8 hours
            self.assertGreater(result.total_seconds(), 7 * 3600)
            self.assertLessEqual(result.total_seconds(), 8 * 3600 + 60)


# =============================================================================
# VLM Screen Classification Tests
# =============================================================================


class TestDetermineAction(unittest.TestCase):
    """Tests for _determine_action() VLM response parsing."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    # -- Structured responses --

    def test_playing(self):
        self.assertEqual(self.mode._determine_action("PLAYING"), "none")

    def test_playing_lowercase(self):
        self.assertEqual(self.mode._determine_action("playing"), "none")

    def test_playing_with_extra_text(self):
        self.assertEqual(self.mode._determine_action("PLAYING - a video is actively playing"), "none")

    def test_paused(self):
        self.assertEqual(self.mode._determine_action("PAUSED"), "play")

    def test_paused_mixed_case(self):
        self.assertEqual(self.mode._determine_action("Paused"), "play")

    def test_dialog(self):
        self.assertEqual(self.mode._determine_action("DIALOG"), "dismiss")

    def test_dialog_with_description(self):
        self.assertEqual(self.mode._determine_action("DIALOG - are you still watching?"), "dismiss")

    def test_menu(self):
        self.assertEqual(self.mode._determine_action("MENU"), "select")

    def test_screensaver(self):
        self.assertEqual(self.mode._determine_action("SCREENSAVER"), "launch")

    def test_screensaver_lowercase(self):
        self.assertEqual(self.mode._determine_action("screensaver"), "launch")

    # -- Fallback keyword matching --

    def test_still_watching_keyword(self):
        self.assertEqual(
            self.mode._determine_action("I see a dialog saying 'Are you still watching?'"),
            "dismiss",
        )

    def test_still_there_keyword(self):
        self.assertEqual(
            self.mode._determine_action("The TV shows 'Are you still there?'"),
            "dismiss",
        )

    def test_home_screen_keyword(self):
        self.assertEqual(
            self.mode._determine_action("This is the YouTube home screen with recommended videos"),
            "select",
        )

    def test_browse_keyword(self):
        self.assertEqual(
            self.mode._determine_action("A browse screen with categories"),
            "select",
        )

    def test_thumbnail_keyword(self):
        self.assertEqual(
            self.mode._determine_action("Multiple thumbnail images of videos"),
            "select",
        )

    def test_paused_keyword(self):
        self.assertEqual(
            self.mode._determine_action("The video appears to be paused with a play button visible"),
            "play",
        )

    def test_not_paused_keyword(self):
        """'not paused' should NOT trigger play."""
        self.assertEqual(
            self.mode._determine_action("The video is not paused, it is actively playing"),
            "none",
        )

    def test_playing_keyword(self):
        self.assertEqual(
            self.mode._determine_action("A video is currently playing on screen"),
            "none",
        )

    def test_black_screen_keyword(self):
        self.assertEqual(
            self.mode._determine_action("I see a black screen with no content"),
            "launch",
        )

    def test_screensaver_keyword_in_sentence(self):
        self.assertEqual(
            self.mode._determine_action("The device appears to have a screensaver active"),
            "launch",
        )

    # -- None / empty / unknown --

    def test_none_input(self):
        self.assertEqual(self.mode._determine_action(None), "none")

    def test_empty_string(self):
        self.assertEqual(self.mode._determine_action(""), "none")

    def test_unknown_response(self):
        self.assertEqual(self.mode._determine_action("I can't tell what this is"), "none")

    def test_whitespace_only(self):
        self.assertEqual(self.mode._determine_action("   "), "none")

    # -- Priority tests --

    def test_dismiss_takes_priority_over_select(self):
        """'still watching' in a menu-like response should dismiss."""
        self.assertEqual(
            self.mode._determine_action("Home screen with 'still watching?' dialog"),
            "dismiss",
        )

    def test_dialog_structured_takes_priority(self):
        """Structured DIALOG prefix takes priority over fallback keywords."""
        self.assertEqual(
            self.mode._determine_action("DIALOG with still watching prompt"),
            "dismiss",
        )

    def test_screensaver_structured_over_playing_keyword(self):
        """SCREENSAVER prefix beats 'playing' keyword in text."""
        self.assertEqual(
            self.mode._determine_action("SCREENSAVER was playing before going dark"),
            "launch",
        )


# =============================================================================
# YouTube Package Detection Tests
# =============================================================================


class TestIsYoutubeApp(unittest.TestCase):
    """Tests for _is_youtube_app()."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_fire_tv_youtube(self):
        self.assertTrue(self.mode._is_youtube_app("com.amazon.firetv.youtube"))

    def test_google_youtube_tv(self):
        self.assertTrue(self.mode._is_youtube_app("com.google.android.youtube.tv"))

    def test_google_youtube(self):
        self.assertTrue(self.mode._is_youtube_app("com.google.android.youtube"))

    def test_plain_youtube(self):
        self.assertTrue(self.mode._is_youtube_app("youtube"))

    def test_youtube_case_insensitive(self):
        self.assertTrue(self.mode._is_youtube_app("com.Amazon.FireTV.YouTube"))

    def test_non_youtube_app(self):
        self.assertFalse(self.mode._is_youtube_app("com.netflix.ninja"))

    def test_none_input(self):
        self.assertFalse(self.mode._is_youtube_app(None))

    def test_empty_string(self):
        self.assertFalse(self.mode._is_youtube_app(""))

    def test_partial_match(self):
        """Package name containing youtube somewhere."""
        self.assertTrue(self.mode._is_youtube_app("com.custom.youtube.player"))

    def test_all_known_packages(self):
        for pkg in YOUTUBE_PACKAGES:
            self.assertTrue(self.mode._is_youtube_app(pkg), f"Failed for package: {pkg}")


# =============================================================================
# State Management Tests
# =============================================================================


class TestStateManagement(unittest.TestCase):
    """Tests for enable/disable/toggle/start_now/get_status/destroy."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_initial_state(self):
        self.assertFalse(self.mode._enabled)
        self.assertFalse(self.mode._active)
        self.assertFalse(self.mode._manual_override)

    def test_enable(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                result = self.mode.enable()
        self.assertTrue(self.mode._enabled)
        self.assertFalse(self.mode._manual_override)
        self.assertTrue(result["enabled"])

    def test_enable_idempotent(self):
        """Enabling twice without manual flag returns early on second call."""
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                self.mode.enable()
                result = self.mode.enable()
        self.assertTrue(result["enabled"])

    def test_enable_manual(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                result = self.mode.enable(manual=True)
        self.assertTrue(self.mode._enabled)
        self.assertTrue(self.mode._manual_override)
        self.assertTrue(result["manual_override"])

    def test_disable(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                self.mode.enable()
            with patch.object(self.mode, "_stop_thread"):
                result = self.mode.disable()
        self.assertFalse(self.mode._enabled)
        self.assertFalse(self.mode._manual_override)
        self.assertFalse(result["enabled"])

    def test_disable_idempotent(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_stop_thread"):
                result = self.mode.disable()
        self.assertFalse(result["enabled"])

    def test_disable_stops_active_session(self):
        """disable() deactivates the session when _active is True.

        Since the API-deadlock fix, `disable()` calls `_deactivate_unlocked()`
        directly (it already holds `self._lock`), not the public `_deactivate()`
        wrapper. We mock the unlocked variant and assert it runs.
        """
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                self.mode.enable()
            self.mode._active = True
            self.mode.stats.session_start = datetime.now(ET)
            with patch.object(self.mode, "_deactivate_unlocked", return_value=True) as mock_deact, \
                 patch.object(self.mode, "_stop_thread"):
                self.mode.disable()
            mock_deact.assert_called_once()

    def test_toggle_on(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                result = self.mode.toggle()
        self.assertTrue(self.mode._enabled)
        self.assertTrue(result["enabled"])

    def test_toggle_off(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                self.mode.enable()
            with patch.object(self.mode, "_stop_thread"):
                result = self.mode.toggle()
        self.assertFalse(self.mode._enabled)
        self.assertFalse(result["enabled"])

    def test_start_now(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                result = self.mode.start_now()
        self.assertTrue(self.mode._enabled)
        self.assertTrue(self.mode._manual_override)
        self.assertTrue(result["manual_override"])

    def test_get_status_fields(self):
        status = self.mode.get_status()
        expected_keys = {
            "enabled", "active", "manual_override", "is_scheduled_time",
            "always_on", "music_mode", "start_hour", "end_hour", "schedule",
            "current_time_et", "next_window_start", "next_window_end",
            "time_until_window", "fire_tv_connected", "device_type",
            "device_connected", "stats",
        }
        self.assertEqual(set(status.keys()), expected_keys)

    def test_get_status_fire_tv_connected(self):
        self.mode._device_type = 'fire_tv'
        self.mode._fire_tv.is_connected.return_value = True
        status = self.mode.get_status()
        self.assertTrue(status["fire_tv_connected"])

    def test_get_status_fire_tv_disconnected(self):
        self.mode._fire_tv.is_connected.return_value = False
        status = self.mode.get_status()
        self.assertFalse(status["fire_tv_connected"])

    def test_get_status_no_fire_tv(self):
        self.mode._fire_tv = None
        status = self.mode.get_status()
        self.assertFalse(status["fire_tv_connected"])

    def test_get_status_stats_dict(self):
        status = self.mode.get_status()
        self.assertIsInstance(status["stats"], dict)
        self.assertIn("videos_played", status["stats"])

    def test_destroy_cleans_up(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.mode._test_settings_path)):
            with patch.object(self.mode, "_start_thread"):
                self.mode.enable()
        self.mode._active = True
        self.mode.stats.session_start = datetime.now(ET)
        self.mode.destroy()
        self.assertFalse(self.mode._active)
        self.assertFalse(self.mode._running)
        self.assertIsNone(self.mode._thread)

    def test_destroy_sets_session_end(self):
        self.mode._active = True
        self.mode.stats.session_start = datetime.now(ET)
        self.mode.destroy()
        self.assertIsNotNone(self.mode.stats.session_end)

    def test_destroy_no_active_session(self):
        """Destroy when not active should not set session_end."""
        self.mode.destroy()
        self.assertIsNone(self.mode.stats.session_end)


# =============================================================================
# Settings Persistence Tests
# =============================================================================


class TestSettingsPersistence(unittest.TestCase):
    """Tests for _load_settings / _save_settings round-trip."""

    def setUp(self):
        fd, self.settings_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.settings_path)  # start clean

    def tearDown(self):
        try:
            os.unlink(self.settings_path)
        except FileNotFoundError:
            pass

    def test_save_and_load_round_trip(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode1 = AutonomousMode()
            mode1._log_file = "/dev/null"
            mode1._enabled = True
            mode1._start_hour = 22
            mode1._end_hour = 6
            mode1._always_on = True
            mode1._save_settings()
            mode1.destroy()

        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode2 = AutonomousMode()
            mode2._log_file = "/dev/null"

        self.assertTrue(mode2._enabled)
        self.assertEqual(mode2._start_hour, 22)
        self.assertEqual(mode2._end_hour, 6)
        self.assertTrue(mode2._always_on)
        mode2.destroy()

    def test_missing_settings_file_uses_defaults(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode = AutonomousMode()
            mode._log_file = "/dev/null"

        self.assertFalse(mode._enabled)
        self.assertEqual(mode._start_hour, AutonomousMode.DEFAULT_START_HOUR)
        self.assertEqual(mode._end_hour, AutonomousMode.DEFAULT_END_HOUR)
        self.assertFalse(mode._always_on)
        mode.destroy()

    def test_corrupted_json_uses_defaults(self):
        with open(self.settings_path, "w") as f:
            f.write("NOT VALID JSON {{{")

        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode = AutonomousMode()
            mode._log_file = "/dev/null"

        self.assertFalse(mode._enabled)
        self.assertEqual(mode._start_hour, AutonomousMode.DEFAULT_START_HOUR)
        mode.destroy()

    def test_partial_settings_fills_defaults(self):
        with open(self.settings_path, "w") as f:
            json.dump({"enabled": True}, f)

        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode = AutonomousMode()
            mode._log_file = "/dev/null"

        self.assertTrue(mode._enabled)
        self.assertEqual(mode._start_hour, AutonomousMode.DEFAULT_START_HOUR)
        self.assertEqual(mode._end_hour, AutonomousMode.DEFAULT_END_HOUR)
        self.assertFalse(mode._always_on)
        mode.destroy()

    def test_save_settings_contains_last_updated(self):
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode = AutonomousMode()
            mode._log_file = "/dev/null"
            mode._save_settings()

        with open(self.settings_path, "r") as f:
            data = json.load(f)
        self.assertIn("last_updated", data)
        mode.destroy()

    def test_save_on_disable(self):
        """Disabling should persist the disabled state."""
        with patch("autonomous_mode.SETTINGS_FILE", Path(self.settings_path)):
            mode = AutonomousMode()
            mode._log_file = "/dev/null"
            mode._enabled = True
            with patch.object(mode, "_stop_thread"):
                mode.disable()

        with open(self.settings_path, "r") as f:
            data = json.load(f)
        self.assertFalse(data["enabled"])
        mode.destroy()


# =============================================================================
# Statistics Recording Tests
# =============================================================================


class TestStatisticsRecording(unittest.TestCase):
    """Tests for record_ad_detected / record_ad_skipped."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_record_ad_detected(self):
        self.assertEqual(self.mode.stats.ads_detected, 0)
        self.mode.record_ad_detected()
        self.assertEqual(self.mode.stats.ads_detected, 1)
        self.assertIsNotNone(self.mode.stats.last_activity)

    def test_record_ad_detected_multiple(self):
        for _ in range(5):
            self.mode.record_ad_detected()
        self.assertEqual(self.mode.stats.ads_detected, 5)

    def test_record_ad_skipped(self):
        self.assertEqual(self.mode.stats.ads_skipped, 0)
        self.mode.record_ad_skipped()
        self.assertEqual(self.mode.stats.ads_skipped, 1)
        self.assertIsNotNone(self.mode.stats.last_activity)

    def test_record_ad_skipped_multiple(self):
        for _ in range(3):
            self.mode.record_ad_skipped()
        self.assertEqual(self.mode.stats.ads_skipped, 3)

    def test_record_updates_last_activity(self):
        before = datetime.now(ET)
        self.mode.record_ad_detected()
        after = datetime.now(ET)
        self.assertGreaterEqual(self.mode.stats.last_activity, before)
        self.assertLessEqual(self.mode.stats.last_activity, after)


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging(unittest.TestCase):
    """Tests for _log_event and get_log_tail."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_log_event_creates_file(self):
        self.assertFalse(os.path.exists(self.mode._test_log_path))
        self.mode._log_event("Test event")
        self.assertTrue(os.path.exists(self.mode._test_log_path))

    def test_log_event_format(self):
        self.mode._log_event("Hello world")
        with open(self.mode._test_log_path, "r") as f:
            content = f.read()
        self.assertIn("Hello world", content)
        self.assertIn("- [", content)
        self.assertIn("ET]", content)

    def test_log_event_appends(self):
        self.mode._log_event("First event")
        self.mode._log_event("Second event")
        with open(self.mode._test_log_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("First event", lines[0])
        self.assertIn("Second event", lines[1])

    def test_get_log_tail_returns_content(self):
        for i in range(10):
            self.mode._log_event(f"Event {i}")
        tail = self.mode.get_log_tail(5)
        self.assertIn("Event 5", tail)
        self.assertIn("Event 9", tail)
        self.assertNotIn("Event 4", tail)

    def test_get_log_tail_missing_file(self):
        result = self.mode.get_log_tail()
        self.assertEqual(result, "No autonomous mode logs yet.")

    def test_get_log_tail_default_lines(self):
        for i in range(100):
            self.mode._log_event(f"Event {i}")
        tail = self.mode.get_log_tail()
        # Default is 50 lines
        lines = tail.strip().split("\n")
        self.assertEqual(len(lines), 50)

    def test_get_log_tail_fewer_lines_than_requested(self):
        self.mode._log_event("Only one event")
        tail = self.mode.get_log_tail(50)
        lines = [l for l in tail.strip().split("\n") if l]
        self.assertEqual(len(lines), 1)


# =============================================================================
# Setters and Callbacks Tests
# =============================================================================


class TestSettersAndCallbacks(unittest.TestCase):
    """Tests for set_fire_tv, set_ad_blocker, set_vlm, set_frame_capture, set_status_callback."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_set_fire_tv(self):
        new_ft = MagicMock()
        self.mode.set_fire_tv(new_ft)
        self.assertIs(self.mode._fire_tv, new_ft)

    def test_set_ad_blocker(self):
        new_ab = MagicMock()
        self.mode.set_ad_blocker(new_ab)
        self.assertIs(self.mode._ad_blocker, new_ab)

    def test_set_vlm(self):
        new_vlm = MagicMock()
        self.mode.set_vlm(new_vlm)
        self.assertIs(self.mode._vlm, new_vlm)

    def test_set_frame_capture(self):
        new_fc = MagicMock()
        self.mode.set_frame_capture(new_fc)
        self.assertIs(self.mode._frame_capture, new_fc)

    def test_set_status_callback(self):
        cb = MagicMock()
        self.mode.set_status_callback(cb)
        self.assertIs(self.mode._on_status_change, cb)


# =============================================================================
# Activation / Deactivation Tests
# =============================================================================


class TestActivationDeactivation(unittest.TestCase):
    """Tests for _activate and _deactivate."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    @patch.object(AutonomousMode, "_launch_youtube", return_value=True)
    def test_activate_sets_state(self, mock_launch):
        self.mode._activate()
        self.assertTrue(self.mode._active)
        self.assertIsNotNone(self.mode.stats.session_start)
        mock_launch.assert_called_once()

    @patch.object(AutonomousMode, "_launch_youtube", return_value=True)
    def test_activate_idempotent(self, mock_launch):
        self.mode._activate()
        self.mode._activate()
        mock_launch.assert_called_once()

    @patch.object(AutonomousMode, "_launch_youtube", return_value=True)
    def test_activate_calls_status_callback(self, mock_launch):
        cb = MagicMock()
        self.mode.set_status_callback(cb)
        self.mode._activate()
        cb.assert_called_once()
        status = cb.call_args[0][0]
        self.assertTrue(status["active"])

    def test_deactivate_sets_state(self):
        self.mode._active = True
        self.mode.stats.session_start = datetime.now(ET)
        self.mode._deactivate()
        self.assertFalse(self.mode._active)
        self.assertIsNotNone(self.mode.stats.session_end)

    def test_deactivate_idempotent(self):
        self.mode._deactivate()  # no-op when not active
        self.assertFalse(self.mode._active)

    def test_deactivate_calls_status_callback(self):
        cb = MagicMock()
        self.mode.set_status_callback(cb)
        self.mode._active = True
        self.mode.stats.session_start = datetime.now(ET)
        self.mode._deactivate()
        cb.assert_called_once()

    @patch.object(AutonomousMode, "_launch_youtube", return_value=True)
    def test_activate_resets_stats(self, mock_launch):
        self.mode.stats.ads_detected = 10
        self.mode._activate()
        self.assertEqual(self.mode.stats.ads_detected, 0)


# =============================================================================
# start_if_enabled Tests
# =============================================================================


class TestStartIfEnabled(unittest.TestCase):
    """Tests for start_if_enabled."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_start_if_enabled_when_disabled(self):
        self.mode._enabled = False
        self.mode.start_if_enabled()
        # Should not start thread
        self.assertFalse(self.mode._running)

    @patch.object(AutonomousMode, "_start_thread")
    def test_start_if_enabled_when_enabled(self, mock_start):
        self.mode._enabled = True
        self.mode.start_if_enabled()
        mock_start.assert_called_once()


# =============================================================================
# Thread Management Tests
# =============================================================================


class TestThreadManagement(unittest.TestCase):
    """Tests for _start_thread / _stop_thread."""

    def setUp(self):
        self.mode = _make_mode()

    def tearDown(self):
        _cleanup_mode(self.mode)

    def test_start_thread(self):
        self.mode._start_thread()
        self.assertTrue(self.mode._running)
        self.assertIsNotNone(self.mode._thread)
        self.assertTrue(self.mode._thread.is_alive())
        # Clean up
        self.mode._stop_thread()

    def test_start_thread_idempotent(self):
        self.mode._start_thread()
        first_thread = self.mode._thread
        self.mode._start_thread()
        self.assertIs(self.mode._thread, first_thread)
        self.mode._stop_thread()

    def test_stop_thread(self):
        self.mode._start_thread()
        self.mode._stop_thread()
        self.assertFalse(self.mode._running)
        self.assertIsNone(self.mode._thread)

    def test_stop_thread_when_not_started(self):
        # Should not raise
        self.mode._stop_thread()
        self.assertFalse(self.mode._running)

    def test_thread_is_daemon(self):
        self.mode._start_thread()
        self.assertTrue(self.mode._thread.daemon)
        self.mode._stop_thread()


# =============================================================================
# SCREEN_QUERY_PROMPT Tests
# =============================================================================


class TestScreenQueryPrompt(unittest.TestCase):
    """Verify the VLM prompt constant."""

    def test_prompt_exists(self):
        self.assertIsInstance(AutonomousMode.SCREEN_QUERY_PROMPT, str)

    def test_prompt_mentions_categories(self):
        for cat in ("PLAYING", "PAUSED", "DIALOG", "MENU", "SCREENSAVER"):
            self.assertIn(cat, AutonomousMode.SCREEN_QUERY_PROMPT)


# =============================================================================
# Roku Device Controller Tests
# =============================================================================


class TestAutonomousModeRoku(unittest.TestCase):
    """Tests for Roku-specific autonomous mode features."""

    def test_set_device_controller_roku(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.__class__.__name__ = "RokuController"
            mode.set_device_controller(roku, 'roku')
            self.assertEqual(mode._device_type, 'roku')
            self.assertIs(mode._device_controller, roku)
        finally:
            _cleanup_mode(mode)

    def test_detect_device_type_roku(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.__class__.__name__ = "RokuController"
            detected = mode._detect_device_type(roku)
            self.assertEqual(detected, 'roku')
        finally:
            _cleanup_mode(mode)

    def test_detect_device_type_fire_tv(self):
        mode = _make_mode()
        try:
            ftv = MagicMock()
            ftv.__class__.__name__ = "FireTVController"
            detected = mode._detect_device_type(ftv)
            self.assertEqual(detected, 'fire_tv')
        finally:
            _cleanup_mode(mode)

    def test_detect_device_type_google_tv(self):
        mode = _make_mode()
        try:
            gtv = MagicMock()
            gtv.__class__.__name__ = "GoogleTVController"
            detected = mode._detect_device_type(gtv)
            self.assertEqual(detected, 'google_tv')
        finally:
            _cleanup_mode(mode)

    def test_launch_youtube_roku(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.is_connected.return_value = True
            roku.launch_app.return_value = True
            mode.set_device_controller(roku, 'roku')
            result = mode._launch_youtube()
            self.assertTrue(result)
            roku.launch_app.assert_called_once_with('youtube')
        finally:
            _cleanup_mode(mode)

    def test_launch_youtube_roku_fails(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.is_connected.return_value = True
            roku.launch_app.return_value = False
            mode.set_device_controller(roku, 'roku')
            result = mode._launch_youtube()
            self.assertFalse(result)
        finally:
            _cleanup_mode(mode)

    def test_launch_youtube_device_disconnected(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.is_connected.return_value = False
            mode.set_device_controller(roku, 'roku')
            result = mode._launch_youtube()
            self.assertFalse(result)
        finally:
            _cleanup_mode(mode)

    def test_status_includes_device_type(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.is_connected.return_value = True
            mode.set_device_controller(roku, 'roku')
            status = mode.get_status()
            self.assertEqual(status['device_type'], 'roku')
            self.assertTrue(status['device_connected'])
        finally:
            _cleanup_mode(mode)


class TestAutonomousModeRokuActiveApp(unittest.TestCase):
    """Tests for Roku ECP active app checking."""

    def test_check_roku_active_app_youtube_running(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.get_active_app_id.return_value = '837'
            roku.is_screensaver_active.return_value = False
            mode.set_device_controller(roku, 'roku')
            result = mode._check_roku_active_app()
            self.assertTrue(result)
        finally:
            _cleanup_mode(mode)

    def test_check_roku_active_app_not_youtube(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.get_active_app_id.return_value = '562859'  # Home
            roku.get_active_app.return_value = 'Home'
            roku.is_screensaver_active.return_value = False
            mode.set_device_controller(roku, 'roku')
            result = mode._check_roku_active_app()
            self.assertFalse(result)
        finally:
            _cleanup_mode(mode)

    def test_check_roku_screensaver_active(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.is_screensaver_active.return_value = True
            roku.send_command.return_value = True
            mode.set_device_controller(roku, 'roku')
            result = mode._check_roku_active_app()
            self.assertTrue(result)  # Returns True because screensaver was dismissed
            roku.send_command.assert_called_with('select')
        finally:
            _cleanup_mode(mode)

    def test_check_roku_skipped_for_fire_tv(self):
        mode = _make_mode()
        try:
            ftv = MagicMock()
            mode.set_device_controller(ftv, 'fire_tv')
            result = mode._check_roku_active_app()
            self.assertTrue(result)  # Non-Roku devices always return True
        finally:
            _cleanup_mode(mode)

    def test_check_roku_query_fails_gracefully(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.is_screensaver_active.return_value = False
            roku.get_active_app_id.return_value = None  # Query failed
            mode.set_device_controller(roku, 'roku')
            result = mode._check_roku_active_app()
            self.assertTrue(result)  # Don't interfere on failure
        finally:
            _cleanup_mode(mode)


# =============================================================================
# Static Detection + Audio Flow Tests
# =============================================================================


class TestAutonomousModeStaticDetection(unittest.TestCase):
    """Tests for frame-change and audio-aware static detection."""

    def test_compute_frame_hash(self):
        """Frame hash should return an integer."""
        import numpy as np
        mode = _make_mode()
        try:
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            h = mode._compute_frame_hash(frame)
            self.assertIsInstance(h, int)
        finally:
            _cleanup_mode(mode)

    def test_compute_frame_hash_different_frames(self):
        """Different frames should produce different hashes."""
        import numpy as np
        mode = _make_mode()
        try:
            # Use frames with actual gradient differences (not solid colors)
            frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
            frame2 = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            h1 = mode._compute_frame_hash(frame1)
            h2 = mode._compute_frame_hash(frame2)
            # Hashes should differ for visually different frames
            hamming = bin(h1 ^ h2).count('1')
            self.assertGreater(hamming, 0)
        finally:
            _cleanup_mode(mode)

    def test_compute_frame_hash_identical_frames(self):
        """Identical frames should produce identical hashes."""
        import numpy as np
        mode = _make_mode()
        try:
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            h1 = mode._compute_frame_hash(frame)
            h2 = mode._compute_frame_hash(frame.copy())
            self.assertEqual(h1, h2)
        finally:
            _cleanup_mode(mode)

    def test_is_audio_flowing_with_ad_blocker(self):
        """Audio flowing check via ad_blocker audio module."""
        mode = _make_mode()
        try:
            mode._ad_blocker = MagicMock()
            mode._ad_blocker.audio = MagicMock()
            # Fresh buffer + real content level (above AUDIO_SILENCE_THRESHOLD)
            mode._ad_blocker.audio.get_status.return_value = {
                'last_buffer_age': 0.5, 'recent_level': 0.15}
            self.assertTrue(mode._is_audio_flowing())

            # Fresh buffer but digital silence (paused source keeps the
            # HDMI carrier alive) — must NOT count as flowing
            mode._ad_blocker.audio.get_status.return_value = {
                'last_buffer_age': 0.5, 'recent_level': 0.0}
            self.assertFalse(mode._is_audio_flowing())

            mode._ad_blocker.audio.get_status.return_value = {
                'last_buffer_age': 10.0, 'recent_level': 0.15}
            self.assertFalse(mode._is_audio_flowing())
        finally:
            _cleanup_mode(mode)

    def test_consecutive_static_counter_resets_on_action(self):
        """Static counter should reset when an action is taken."""
        mode = _make_mode()
        try:
            mode._consecutive_static = 5
            roku = MagicMock()
            roku.is_connected.return_value = True
            roku.send_command.return_value = True
            mode.set_device_controller(roku, 'roku')

            # Simulate VLM returning PAUSED
            mode._vlm = MagicMock()
            mode._vlm.is_ready = True
            mode._query_screen = MagicMock(return_value="PAUSED")

            mode._ensure_youtube_playing()
            self.assertEqual(mode._consecutive_static, 0)
        finally:
            _cleanup_mode(mode)

    def test_determine_action_playing(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("PLAYING"), "none")
        finally:
            _cleanup_mode(mode)

    def test_determine_action_paused(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("PAUSED"), "play")
        finally:
            _cleanup_mode(mode)

    def test_determine_action_dialog(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("DIALOG"), "dismiss")
        finally:
            _cleanup_mode(mode)

    def test_determine_action_screensaver(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("SCREENSAVER"), "launch")
        finally:
            _cleanup_mode(mode)

    def test_determine_action_menu(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("MENU"), "select")
        finally:
            _cleanup_mode(mode)

    def test_determine_action_unknown(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("UNKNOWN_STATE"), "none")
        finally:
            _cleanup_mode(mode)

    def test_determine_action_still_watching(self):
        mode = _make_mode()
        try:
            self.assertEqual(mode._determine_action("Are you still watching?"), "dismiss")
        finally:
            _cleanup_mode(mode)


class TestAutonomousModeWakeDevice(unittest.TestCase):
    """Tests for device wake functionality."""

    def test_wake_roku(self):
        mode = _make_mode()
        try:
            roku = MagicMock()
            roku.send_command.return_value = True
            mode.set_device_controller(roku, 'roku')
            mode._wake_device()
            # Roku wake sends power then home
            calls = [c[0][0] for c in roku.send_command.call_args_list]
            self.assertIn('power', calls)
            self.assertIn('home', calls)
        finally:
            _cleanup_mode(mode)

    def test_wake_fire_tv(self):
        mode = _make_mode()
        try:
            ftv = MagicMock()
            ftv.send_command.return_value = True
            mode.set_device_controller(ftv, 'fire_tv')
            mode._wake_device()
            ftv.send_command.assert_called_with('wakeup')
        finally:
            _cleanup_mode(mode)


# =============================================================================
# YouTube TV Prompt Detection Tests
# =============================================================================


def _set_ocr_texts(mode, texts):
    """Point the mode's mocked ad_blocker at the given OCR texts."""
    mode._ad_blocker.last_ocr_texts = texts
    mode._ad_blocker.is_visible = False


class TestYouTubeTVPromptDetection(unittest.TestCase):
    """Tests for _is_youtube_tv_prompt() — avoiding YouTube TV upsell screens."""

    def test_exact_promo_phrase(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["YouTube TV", "Cable-free live TV", "Try it free"])
            self.assertTrue(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_try_youtube_tv(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["Try YouTube TV", "Learn more"])
            self.assertTrue(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_youtube_tv_plus_marker(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["YouTube TV", "$72.99 per month", "Sign up"])
            self.assertTrue(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_youtube_tv_free_trial_merged(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["youtubetv", "free trial", "New users only"])
            self.assertTrue(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_video_title_mentioning_youtube_tv_not_detected(self):
        """A video titled 'YouTube TV review' must NOT trigger the escape."""
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["YouTube TV review 2026", "347K views", "3 days ago"])
            self.assertFalse(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_normal_home_screen_not_detected(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["Recommended", "Trending", "Subscriptions"])
            self.assertFalse(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_suppressed_while_ad_blocking(self):
        """A YouTube TV *ad* is the ad blocker's problem — no Back press."""
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, ["Try YouTube TV", "free trial"])
            mode._ad_blocker.is_visible = True
            self.assertFalse(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)

    def test_empty_ocr_not_detected(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, [])
            self.assertFalse(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)


class TestYouTubeTVActivationScreen(unittest.TestCase):
    """The 'Sign in to YouTube TV' activation screen (live OCR capture
    2026-07-02) stalled autonomous mode for 40 min because no detector
    recognized it. It must trip BOTH the keyboard-stuck check (activation
    code markers) and the YouTube TV prompt detector."""

    # Verbatim OCR output from the live incident
    LIVE_OCR = ["finalthemeu", "Or scan with your phone", "STEP3", "Suw",
                "YFW-XVR-STL", "STEP2", "Enterthis code:",
                "tv.youtube.com/start", "STEP1", "Orm",
                "Sign in to YouTube TV"]

    def test_keyboard_stuck_detects_activation_screen(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, self.LIVE_OCR)
            self.assertTrue(mode._is_keyboard_stuck_screen())
        finally:
            _cleanup_mode(mode)

    def test_yttv_prompt_detects_activation_screen(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, self.LIVE_OCR)
            self.assertTrue(mode._is_youtube_tv_prompt())
        finally:
            _cleanup_mode(mode)


class TestMenuSkipWatchdog(unittest.TestCase):
    """The MENU-select skip guard must escalate to an escape after N
    consecutive no-op skips instead of stalling forever on an
    unrecognized dead-end screen."""

    def _stalled_mode(self):
        """Mode where every cycle produces a skipped MENU verdict."""
        mode = _make_mode()
        roku = MagicMock()
        roku.is_connected.return_value = True
        roku.send_command.return_value = True
        mode.set_device_controller(roku, 'roku')
        _set_ocr_texts(mode, ["finalthemeu", "STEP1"])  # nothing recognizable
        mode._ad_blocker.display_connected = False  # skip audio-recovery section
        # Force the dead-end path: MENU verdict, no audio, no overlay,
        # no recognizable screen of any kind
        mode._check_roku_active_app = MagicMock(return_value=True)
        mode._is_roku_home_screen = MagicMock(return_value=False)
        mode._is_keyboard_stuck_screen = MagicMock(return_value=False)
        mode._is_youtube_tv_prompt = MagicMock(return_value=False)
        mode._is_survey_screen = MagicMock(return_value=False)
        mode._is_signed_out_screen = MagicMock(return_value=False)
        mode._is_youtube_login_screen = MagicMock(return_value=False)
        mode._is_youtube_shorts = MagicMock(return_value=False)
        mode._is_youtube_home_screen = MagicMock(return_value=False)
        mode._has_live_content_indicator = MagicMock(return_value=False)
        mode._query_screen = MagicMock(return_value="MENU")
        mode._is_audio_flowing = MagicMock(return_value=False)
        mode._is_video_player_overlay = MagicMock(return_value=False)
        mode._escape_stuck_state = MagicMock(return_value=True)
        return mode, roku

    def test_escapes_after_threshold_consecutive_skips(self):
        mode, roku = self._stalled_mode()
        try:
            for i in range(mode._MENU_SKIP_ESCAPE_AT - 1):
                self.assertFalse(mode._ensure_youtube_playing())
            mode._escape_stuck_state.assert_not_called()

            # Nth consecutive skip triggers the escape
            self.assertTrue(mode._ensure_youtube_playing())
            mode._escape_stuck_state.assert_called_once()
            self.assertEqual(mode._menu_skip_count, 0)
        finally:
            _cleanup_mode(mode)

    def test_playing_screen_resets_skip_counter(self):
        mode, roku = self._stalled_mode()
        try:
            for _ in range(mode._MENU_SKIP_ESCAPE_AT - 1):
                mode._ensure_youtube_playing()

            # A genuinely-playing cycle resets the watchdog
            mode._query_screen = MagicMock(return_value="PLAYING")
            mode._is_screen_static = MagicMock(return_value=False)
            self.assertFalse(mode._ensure_youtube_playing())
            self.assertEqual(mode._menu_skip_count, 0)

            # Back to stalled MENU: counts from 1 again, no premature escape
            mode._query_screen = MagicMock(return_value="MENU")
            mode._ensure_youtube_playing()
            mode._escape_stuck_state.assert_not_called()
            self.assertEqual(mode._menu_skip_count, 1)
        finally:
            _cleanup_mode(mode)


class TestKeyboardStuckAudioGuard(unittest.TestCase):
    """The keyboard-stuck escape must not fire while audio is flowing —
    real sign-in/keyboard screens are silent. Observed live 2026-07-02
    03:31: the character-pattern heuristic false-positived on a playing
    video and the Back-escape exited YouTube."""

    def _keyboardish_mode(self):
        mode = _make_mode()
        roku = MagicMock()
        roku.is_connected.return_value = True
        roku.send_command.return_value = True
        mode.set_device_controller(roku, 'roku')
        # OCR that trips the character-pattern heuristic: mostly 1-2 char
        # fragments including digits
        _set_ocr_texts(mode, ["1", "2", "a", "b", "c", "longer text"])
        mode._check_roku_active_app = MagicMock(return_value=True)
        mode._is_roku_home_screen = MagicMock(return_value=False)
        mode._escape_stuck_state = MagicMock(return_value=True)
        mode._full_reset_to_youtube = MagicMock(return_value=True)
        return mode, roku

    def test_audio_flowing_vetoes_keyboard_escape(self):
        mode, roku = self._keyboardish_mode()
        try:
            self.assertTrue(mode._is_keyboard_stuck_screen())  # heuristic trips
            mode._is_audio_flowing = MagicMock(return_value=True)
            self.assertFalse(mode._ensure_youtube_playing())
            mode._escape_stuck_state.assert_not_called()
            mode._full_reset_to_youtube.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_silent_keyboard_screen_still_escapes(self):
        mode, roku = self._keyboardish_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            self.assertTrue(mode._ensure_youtube_playing())
            mode._escape_stuck_state.assert_called_once()
        finally:
            _cleanup_mode(mode)


class TestRokuHomeOCRGuards(unittest.TestCase):
    """The Roku-home OCR fallback must never relaunch over playing video:
    observed live 2026-07-02, an ad's 'Watch now' CTA OCR-merged to
    'watchnow' and relaunched YouTube mid-ad-break."""

    def _roku_mode(self, ocr_texts):
        mode = _make_mode()
        roku = MagicMock()
        roku.is_connected.return_value = True
        mode.set_device_controller(roku, 'roku')
        _set_ocr_texts(mode, ocr_texts)
        return mode, roku

    def test_watch_now_cta_no_longer_matches(self):
        mode, roku = self._roku_mode(["WatchNow", "Get 3 months free"])
        try:
            self.assertFalse(mode._is_roku_home_screen())
        finally:
            _cleanup_mode(mode)

    def test_distinctive_keywords_still_match(self):
        mode, roku = self._roku_mode(["Roku Channel", "Frndly TV", "hulu"])
        try:
            self.assertTrue(mode._is_roku_home_screen())
        finally:
            _cleanup_mode(mode)

    def test_suppressed_while_ad_blocking(self):
        mode, roku = self._roku_mode(["Roku Channel"])
        try:
            mode._ad_blocker.is_visible = True
            self.assertFalse(mode._is_roku_home_screen())
        finally:
            _cleanup_mode(mode)

    def test_audio_flowing_vetoes_ocr_relaunch(self):
        """OCR says Roku home but audio flows → no relaunch."""
        mode, roku = self._roku_mode(["Roku Channel"])
        try:
            mode._check_roku_active_app = MagicMock(return_value=True)
            mode._is_audio_flowing = MagicMock(return_value=True)
            mode._launch_youtube = MagicMock()
            mode._is_keyboard_stuck_screen = MagicMock(return_value=False)
            mode._is_youtube_tv_prompt = MagicMock(return_value=False)
            mode._is_survey_screen = MagicMock(return_value=False)
            mode._is_signed_out_screen = MagicMock(return_value=False)
            mode._is_youtube_login_screen = MagicMock(return_value=False)
            mode._is_youtube_shorts = MagicMock(return_value=False)
            mode._is_youtube_home_screen = MagicMock(return_value=False)
            mode._ad_blocker.display_connected = False
            mode._query_screen = MagicMock(return_value="PLAYING")
            mode._is_screen_static = MagicMock(return_value=False)
            self.assertFalse(mode._ensure_youtube_playing())
            mode._launch_youtube.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_silent_roku_home_still_relaunches(self):
        mode, roku = self._roku_mode(["Roku Channel"])
        try:
            mode._check_roku_active_app = MagicMock(return_value=True)
            mode._is_audio_flowing = MagicMock(return_value=False)
            mode._launch_youtube = MagicMock(return_value=True)
            self.assertTrue(mode._ensure_youtube_playing())
            mode._launch_youtube.assert_called_once()
        finally:
            _cleanup_mode(mode)


class TestDialogDismissAudioGuard(unittest.TestCase):
    """DIALOG→back must be vetoed while audio flows: VLM misreads playing
    frames as DIALOG and back EXITS the video (observed live 2026-07-02,
    twice, each costing a ~3-min watchdog recovery)."""

    def _dialog_mode(self):
        mode = _make_mode()
        roku = MagicMock()
        roku.is_connected.return_value = True
        roku.send_command.return_value = True
        mode.set_device_controller(roku, 'roku')
        _set_ocr_texts(mode, [])
        mode._ad_blocker.display_connected = False
        mode._check_roku_active_app = MagicMock(return_value=True)
        mode._is_roku_home_screen = MagicMock(return_value=False)
        mode._is_keyboard_stuck_screen = MagicMock(return_value=False)
        mode._is_youtube_tv_prompt = MagicMock(return_value=False)
        mode._is_survey_screen = MagicMock(return_value=False)
        mode._is_signed_out_screen = MagicMock(return_value=False)
        mode._is_youtube_login_screen = MagicMock(return_value=False)
        mode._is_youtube_shorts = MagicMock(return_value=False)
        mode._is_youtube_home_screen = MagicMock(return_value=False)
        mode._query_screen = MagicMock(return_value="DIALOG")
        return mode, roku

    def test_audio_flowing_vetoes_dismiss(self):
        mode, roku = self._dialog_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=True)
            self.assertFalse(mode._ensure_youtube_playing())
            roku.send_command.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_silent_dialog_still_dismissed(self):
        """Real blocking dialogs pause playback (no audio) → back sent."""
        mode, roku = self._dialog_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            self.assertTrue(mode._ensure_youtube_playing())
            roku.send_command.assert_called_with("back")
        finally:
            _cleanup_mode(mode)

    def test_recent_audio_vetoes_dismiss_during_ad_gap(self):
        """Silent gap between pre-roll ads: audio flowed seconds ago —
        the recency window must veto the back press (live incident
        2026-07-02 17:21: DIALOG misfire 4s after an ad block ended
        back-exited the just-started video)."""
        mode, roku = self._dialog_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            mode._last_audio_flowing_time = time.time() - 5.0
            self.assertFalse(mode._ensure_youtube_playing())
            roku.send_command.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_stale_audio_does_not_veto_dismiss(self):
        """Audio last flowed beyond the window → real silent dialog,
        dismiss proceeds."""
        mode, roku = self._dialog_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            mode._last_audio_flowing_time = time.time() - 60.0
            self.assertTrue(mode._ensure_youtube_playing())
            roku.send_command.assert_called_with("back")
        finally:
            _cleanup_mode(mode)


class TestScreensaverLaunchGuards(unittest.TestCase):
    """The SCREENSAVER→launch action must consult authoritative playback
    signals (audio, Roku ECP) before wake+relaunch. Observed live
    2026-07-02: dark playing scenes drew SCREENSAVER verdicts and the
    unguarded launch killed playback every ~2 min."""

    def _screensaver_mode(self, device='roku'):
        mode = _make_mode()
        controller = MagicMock()
        controller.is_connected.return_value = True
        controller.send_command.return_value = True
        mode.set_device_controller(controller, device)
        _set_ocr_texts(mode, [])
        mode._ad_blocker.display_connected = False  # skip audio-recovery section
        mode._check_roku_active_app = MagicMock(return_value=True)
        mode._is_roku_home_screen = MagicMock(return_value=False)
        mode._is_keyboard_stuck_screen = MagicMock(return_value=False)
        mode._is_youtube_tv_prompt = MagicMock(return_value=False)
        mode._is_survey_screen = MagicMock(return_value=False)
        mode._is_signed_out_screen = MagicMock(return_value=False)
        mode._is_youtube_login_screen = MagicMock(return_value=False)
        mode._is_youtube_shorts = MagicMock(return_value=False)
        mode._is_youtube_home_screen = MagicMock(return_value=False)
        mode._query_screen = MagicMock(return_value="SCREENSAVER")
        mode._wake_device = MagicMock()
        mode._launch_youtube = MagicMock(return_value=True)
        return mode, controller

    def test_audio_flowing_vetoes_launch(self):
        mode, controller = self._screensaver_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=True)
            self.assertFalse(mode._ensure_youtube_playing())
            mode._wake_device.assert_not_called()
            mode._launch_youtube.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_roku_ecp_youtube_active_vetoes_launch(self):
        """No audio (quiet dark scene) but ECP says YouTube active with no
        screensaver — VLM misread a dark frame; don't wake+relaunch."""
        mode, controller = self._screensaver_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            controller.is_screensaver_active.return_value = False
            controller.get_active_app_id.return_value = '837'
            self.assertFalse(mode._ensure_youtube_playing())
            mode._wake_device.assert_not_called()
            mode._launch_youtube.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_genuine_screensaver_still_launches(self):
        """No audio + ECP confirms screensaver → wake+launch proceeds."""
        mode, controller = self._screensaver_mode()
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            controller.is_screensaver_active.return_value = True
            with patch("autonomous_mode.time.sleep"):
                self.assertTrue(mode._ensure_youtube_playing())
            mode._wake_device.assert_called_once()
            mode._launch_youtube.assert_called_once()
        finally:
            _cleanup_mode(mode)

    def test_fire_tv_no_audio_still_launches(self):
        """Non-Roku device with no audio: no ECP to consult, launch runs."""
        mode, controller = self._screensaver_mode(device='fire_tv')
        try:
            mode._is_audio_flowing = MagicMock(return_value=False)
            with patch("autonomous_mode.time.sleep"):
                self.assertTrue(mode._ensure_youtube_playing())
            mode._wake_device.assert_called_once()
            mode._launch_youtube.assert_called_once()
        finally:
            _cleanup_mode(mode)


# =============================================================================
# YouTube Shorts Detection Tests
# =============================================================================


class TestShortsDetection(unittest.TestCase):
    """Tests for the improved _is_youtube_shorts() and vertical-frame check."""

    @staticmethod
    def _pillarboxed_frame(width=640, height=360, center_val=120):
        """Synthetic Shorts-style frame: black side bars, lit vertical center."""
        import numpy as np
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Centered vertical video ~32% of width
        x0 = int(width * 0.34)
        x1 = int(width * 0.66)
        frame[:, x0:x1] = center_val
        return frame

    def test_vertical_frame_detected(self):
        mode = _make_mode()
        try:
            frame = self._pillarboxed_frame()
            self.assertTrue(mode._is_vertical_video_frame(frame))
        finally:
            _cleanup_mode(mode)

    def test_full_width_frame_not_vertical(self):
        import numpy as np
        mode = _make_mode()
        try:
            frame = np.full((360, 640, 3), 120, dtype=np.uint8)
            self.assertFalse(mode._is_vertical_video_frame(frame))
        finally:
            _cleanup_mode(mode)

    def test_letterboxed_frame_not_vertical(self):
        """Cinematic content (dark top/bottom bars, bright full width)."""
        import numpy as np
        mode = _make_mode()
        try:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[60:300, :] = 120  # bright band across full width
            self.assertFalse(mode._is_vertical_video_frame(frame))
        finally:
            _cleanup_mode(mode)

    def test_dark_scene_with_textured_sides_not_vertical(self):
        """A dark movie shot with a lit center subject: sides have texture
        (std too high for true pillarbox bars) so it must not trigger."""
        import numpy as np
        mode = _make_mode()
        try:
            rng = np.random.default_rng(42)
            frame = rng.integers(0, 45, (360, 640, 3), dtype=np.uint8).astype(np.uint8)
            frame[:, 256:384] = 150  # lit center
            self.assertFalse(mode._is_vertical_video_frame(frame))
        finally:
            _cleanup_mode(mode)

    def test_none_frame_not_vertical(self):
        mode = _make_mode()
        try:
            self.assertFalse(mode._is_vertical_video_frame(None))
        finally:
            _cleanup_mode(mode)

    def test_ocr_signature_detected(self):
        mode = _make_mode()
        try:
            mode._frame_capture = None  # isolate OCR path
            _set_ocr_texts(mode, ["@somecreator", "Subscribe", "#shorts"])
            self.assertTrue(mode._is_youtube_shorts())
        finally:
            _cleanup_mode(mode)

    def test_ocr_signature_with_duration_not_shorts(self):
        """A full video's player overlay (@handle + Subscribe + 10:23) is
        NOT Shorts — the time marker distinguishes it. The old tautological
        duration check regressed here."""
        mode = _make_mode()
        try:
            mode._frame_capture = None
            _set_ocr_texts(mode, ["@somecreator", "Subscribe", "10:23"])
            self.assertFalse(mode._is_youtube_shorts())
        finally:
            _cleanup_mode(mode)

    def test_vertical_frames_detected_as_shorts(self):
        """Overlay-less Shorts playback: OCR empty, both frames pillarboxed."""
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, [])
            frame = self._pillarboxed_frame()
            mode._frame_capture.capture.return_value = frame
            with patch("autonomous_mode.time.sleep"):
                self.assertTrue(mode._is_youtube_shorts())
        finally:
            _cleanup_mode(mode)

    def test_single_vertical_frame_not_enough(self):
        """Double-confirmation: one pillarboxed frame + one normal frame
        (scene cut in a dark movie) must not trigger."""
        import numpy as np
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, [])
            normal = np.full((360, 640, 3), 120, dtype=np.uint8)
            mode._frame_capture.capture.side_effect = [self._pillarboxed_frame(), normal]
            with patch("autonomous_mode.time.sleep"):
                self.assertFalse(mode._is_youtube_shorts())
        finally:
            _cleanup_mode(mode)

    def test_vertical_check_skipped_while_blocking(self):
        mode = _make_mode()
        try:
            _set_ocr_texts(mode, [])
            mode._ad_blocker.is_visible = True
            mode._frame_capture.capture.return_value = self._pillarboxed_frame()
            self.assertFalse(mode._is_youtube_shorts())
            mode._frame_capture.capture.assert_not_called()
        finally:
            _cleanup_mode(mode)


class TestMusicMode(unittest.TestCase):
    """Tests for the music-videos toggle: persistence, seed deep-link
    launching, and OCR-evidence drift steering."""

    def test_default_off(self):
        mode = _make_mode()
        try:
            self.assertFalse(mode._music_mode)
            self.assertFalse(mode.get_status()["music_mode"])
        finally:
            _cleanup_mode(mode)

    def test_set_music_mode_persists(self):
        mode = _make_mode()
        try:
            with patch("autonomous_mode.SETTINGS_FILE", Path(mode._test_settings_path)):
                result = mode.set_music_mode(True)
                self.assertTrue(result["music_mode"])
                with open(mode._test_settings_path) as f:
                    saved = json.load(f)
                self.assertTrue(saved["music_mode"])

                result = mode.set_music_mode(False)
                self.assertFalse(result["music_mode"])
        finally:
            _cleanup_mode(mode)

    def test_launch_uses_seed_when_music_mode_on(self):
        """_launch_youtube deep-links a rotating seed when music mode is on."""
        mode = _make_mode(fire_tv_connected=True)
        try:
            ctrl = mode._device_controller
            ctrl.launch_app_with_content = MagicMock(return_value=True)
            mode._music_mode = True
            with patch("autonomous_mode.time.sleep"):
                self.assertTrue(mode._launch_youtube())
                self.assertTrue(mode._launch_youtube())
            calls = ctrl.launch_app_with_content.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].args[0], 'youtube')
            # Seeds rotate — consecutive launches use different videos
            self.assertNotEqual(calls[0].args[1], calls[1].args[1])
            self.assertIn(calls[0].args[1], mode.MUSIC_VIDEO_SEEDS)
        finally:
            _cleanup_mode(mode)

    def test_launch_falls_back_without_deep_link_support(self):
        """Controllers without launch_app_with_content fall through to the
        plain launch path (spec'd mock has no such attr)."""
        mode = _make_mode(fire_tv_connected=True)
        try:
            ctrl = MagicMock(spec=['is_connected', 'send_command', 'launch_app'])
            ctrl.is_connected.return_value = True
            ctrl.launch_app.return_value = True
            mode.set_device_controller(ctrl, 'roku')
            mode._music_mode = True
            with patch("autonomous_mode.time.sleep"):
                self.assertTrue(mode._launch_youtube())
            ctrl.launch_app.assert_called_once_with('youtube')
        finally:
            _cleanup_mode(mode)

    def test_launch_plain_when_music_mode_off(self):
        mode = _make_mode(fire_tv_connected=True)
        try:
            ctrl = mode._device_controller
            ctrl.launch_app_with_content = MagicMock(return_value=True)
            ctrl.launch_app = MagicMock(return_value=True)
            mode.set_device_controller(ctrl, 'roku')
            with patch("autonomous_mode.time.sleep"):
                mode._launch_youtube()
            ctrl.launch_app_with_content.assert_not_called()
        finally:
            _cleanup_mode(mode)

    def test_drift_ignores_no_text_cycles(self):
        """Cycles with no meaningful OCR text are neutral, not misses."""
        mode = _make_mode()
        try:
            mode._music_mode = True
            mode._ad_blocker.is_visible = False
            mode._ad_blocker.last_ocr_texts = []
            for _ in range(10):
                self.assertFalse(mode._check_music_drift())
            self.assertEqual(mode._music_no_evidence_checks, 0)
        finally:
            _cleanup_mode(mode)

    def test_drift_resets_on_music_evidence(self):
        mode = _make_mode()
        try:
            mode._music_mode = True
            mode._ad_blocker.is_visible = False
            mode._music_no_evidence_checks = 3
            self.assertFalse(mode._check_music_drift(
                force_text="Luis Fonsi - Despacito ft. Daddy Yankee (Official Video)"))
            self.assertEqual(mode._music_no_evidence_checks, 0)
        finally:
            _cleanup_mode(mode)

    def test_drift_steers_after_threshold(self):
        """Repeated info-bearing non-music text triggers a seed re-launch."""
        mode = _make_mode(fire_tv_connected=True)
        try:
            ctrl = mode._device_controller
            ctrl.launch_app_with_content = MagicMock(return_value=True)
            mode._music_mode = True
            mode._ad_blocker.is_visible = False
            non_music = "Inside The Ocean's Greatest Engineering Marvels 69K views"
            with patch("autonomous_mode.time.sleep"):
                for i in range(mode._MUSIC_STEER_AFTER - 1):
                    self.assertFalse(mode._check_music_drift(force_text=non_music))
                self.assertTrue(mode._check_music_drift(force_text=non_music))
            ctrl.launch_app_with_content.assert_called_once()
            self.assertEqual(mode._music_no_evidence_checks, 0)
        finally:
            _cleanup_mode(mode)

    def test_drift_skipped_during_ad_block(self):
        """Ad copy on screen says nothing about the underlying video."""
        mode = _make_mode()
        try:
            mode._music_mode = True
            mode._ad_blocker.is_visible = True
            self.assertFalse(mode._check_music_drift(
                force_text="Shop now at example.com limited time offer"))
            self.assertEqual(mode._music_no_evidence_checks, 0)
        finally:
            _cleanup_mode(mode)

    def test_drift_noop_when_music_mode_off(self):
        mode = _make_mode()
        try:
            mode._ad_blocker.is_visible = False
            self.assertFalse(mode._check_music_drift(
                force_text="Some documentary title 1M views"))
            self.assertEqual(mode._music_no_evidence_checks, 0)
        finally:
            _cleanup_mode(mode)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Use unittest runner so file is self-contained
    unittest.main(verbosity=2)
