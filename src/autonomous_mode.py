"""
Autonomous Mode for Minus - Automated YouTube playback for training data collection.

Configurable schedule with support for 24/7 operation. Keeps YouTube playing
on streaming devices (Fire TV, Roku, Google TV) to collect ad detection training data.
Uses VLM to understand screen state and take intelligent actions.

Device-agnostic design supports any streaming device with remote control capability.
"""

import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from zoneinfo import ZoneInfo

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Settings file for persistence (use absolute path to work regardless of running user)
SETTINGS_FILE = Path("/home/radxa/.minus_autonomous_mode.json")

# Eastern timezone (default, but schedule hours are timezone-agnostic for simplicity)
ET = ZoneInfo("America/New_York")

# YouTube package names (Fire TV/Android TV use Android packages)
YOUTUBE_PACKAGES = [
    "com.amazon.firetv.youtube",
    "com.google.android.youtube.tv",
    "com.google.android.youtube",
    "youtube",
]

# Supported device types for autonomous mode
DEVICE_TYPE_FIRE_TV = 'fire_tv'
DEVICE_TYPE_ROKU = 'roku'
DEVICE_TYPE_GOOGLE_TV = 'google_tv'

# Timing constants - adaptive based on state
CHECK_INTERVAL = 15.0              # Base check interval
KEEPALIVE_INTERVAL_PLAYING = 20.0  # When video is playing, check every 20s (catches unexpected exits)
KEEPALIVE_INTERVAL_NAV = 10.0      # On navigation screens (home/login), check every 10s


class AutonomousModeStats:
    """Statistics for autonomous mode session."""

    def __init__(self):
        self.session_start: Optional[datetime] = None
        self.session_end: Optional[datetime] = None
        self.videos_played = 0
        self.ads_detected = 0
        self.ads_skipped = 0
        self.errors = 0
        self.last_activity: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "videos_played": self.videos_played,
            "ads_detected": self.ads_detected,
            "ads_skipped": self.ads_skipped,
            "errors": self.errors,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "duration_minutes": self._get_duration_minutes(),
        }

    def _get_duration_minutes(self) -> int:
        if not self.session_start:
            return 0
        end = self.session_end or datetime.now(ET)
        return int((end - self.session_start).total_seconds() / 60)

    def reset(self):
        """Reset stats for new session."""
        self.__init__()


class AutonomousMode:
    """
    Autonomous Mode controller for automated operation.

    Features:
    - Configurable schedule (start/end hours, or 24/7 mode)
    - Manual enable/disable toggle
    - Keeps YouTube playing on streaming device (Fire TV, Roku, Google TV)
    - Uses VLM for intelligent screen understanding
    - Tracks statistics
    - Integrates with ad blocking system

    Device-agnostic: works with any controller that has is_connected() and send_command().
    """

    # Default schedule
    DEFAULT_START_HOUR = 0   # Midnight
    DEFAULT_END_HOUR = 8     # 8 AM

    # Music mode: seed videos to deep-link into when steering toward music
    # content. All are popular official music videos — YouTube autoplay from
    # any of them keeps recommending music, which carries a much higher ad
    # load than general content (the point of the mode: more ad training
    # data per hour). Rotated round-robin per launch so one dead/aged seed
    # can't wedge the mode.
    MUSIC_VIDEO_SEEDS = [
        'kJQP7kiw5Fk',  # Luis Fonsi - Despacito ft. Daddy Yankee
        'JGwWNGJdvx8',  # Ed Sheeran - Shape of You
        'RgKAFK5djSk',  # Wiz Khalifa - See You Again ft. Charlie Puth
        'OPf0YbXqDm0',  # Mark Ronson - Uptown Funk ft. Bruno Mars
        'CevxZvSJLk8',  # Katy Perry - Roar
        '9bZkp7q19f0',  # PSY - Gangnam Style
    ]

    # OCR text markers that indicate the current video is music. These only
    # appear when title text is on screen (player overlay, end cards), so
    # most checks carry no information — see _check_music_drift for how
    # "no text" cycles are treated as neutral rather than as misses.
    MUSIC_EVIDENCE_KEYWORDS = [
        'vevo', 'official video', 'official music video', 'music video',
        'lyric', 'official audio', 'remix', 'feat.', 'ft.', 'visualizer',
        'live performance', 'live session', 'acoustic',
    ]

    # When the audio pipeline is unavailable (display off / alsasink can't
    # open), we normally abstain from pause detection to avoid false positives
    # on music streams with static album art. But a genuinely-frozen video
    # (e.g. a live stream that froze on the source side) will show hamming=0
    # indefinitely and the user gets stuck. If we see that many consecutive
    # hamming=0 observations in a row, we escalate to "stuck" regardless of
    # audio state. At ~22s per _is_screen_static() call, 15 observations =
    # ~5.5 min of a truly-static screen before we act.
    PERSISTENT_STATIC_LIMIT = 15

    def __init__(self, device_controller=None, ad_blocker=None, vlm=None, frame_capture=None,
                 fire_tv_controller=None):
        """
        Initialize autonomous mode.

        Args:
            device_controller: Generic device controller (FireTV, Roku, GoogleTV)
            ad_blocker: DRMAdBlocker instance for ad detection stats
            vlm: VLMManager instance for screen understanding
            frame_capture: UstreamerCapture instance for grabbing frames
            fire_tv_controller: Deprecated, use device_controller instead
        """
        # Support both new device_controller and legacy fire_tv_controller param
        self._device_controller = device_controller or fire_tv_controller
        self._device_type: Optional[str] = None  # Detected at runtime
        self._ad_blocker = ad_blocker
        self._vlm = vlm
        self._frame_capture = frame_capture

        # Legacy alias for backwards compatibility
        self._fire_tv = self._device_controller

        # State
        self._enabled = False          # User toggle
        self._active = False           # Currently in active window
        self._running = False          # Thread running
        self._manual_override = False  # User manually started outside schedule

        # Schedule (configurable)
        self._start_hour = self.DEFAULT_START_HOUR
        self._end_hour = self.DEFAULT_END_HOUR
        self._always_on = False        # 24/7 mode

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Stats
        self.stats = AutonomousModeStats()

        # Callbacks
        self._on_status_change: Optional[Callable[[dict], None]] = None

        # Frame change detection for pause verification
        self._prev_frame_hash: Optional[int] = None
        self._consecutive_static: int = 0
        self._STATIC_PAUSE_THRESHOLD = 2  # Consecutive static checks before forcing play

        # Persistent-static tracking for the audio-pipeline-unavailable case:
        # if hamming=0 keeps recurring for PERSISTENT_STATIC_LIMIT checks in a
        # row we escalate from "abstain" to "genuinely stuck". See
        # _is_screen_static() and PERSISTENT_STATIC_LIMIT.
        self._persistent_static_count: int = 0

        # No-audio timeout for stuck state recovery
        self._no_audio_start_time: Optional[float] = None
        self._NO_AUDIO_TIMEOUT = 30.0  # Seconds without audio before recovery attempt
        self._last_recovery_time: Optional[float] = None
        self._RECOVERY_COOLDOWN = 60.0  # Minimum seconds between recovery attempts

        # Escalating recovery - tracks failed attempts to try different strategies
        self._recovery_attempt_count = 0
        self._last_successful_audio_time: Optional[float] = None
        self._RECOVERY_ESCALATION_THRESHOLD = 3  # After N failed attempts, escalate

        # Stuck detection - state machine for detecting and escaping stuck states
        self._last_screen_state: Optional[str] = None  # Track last detected screen type
        self._stuck_count: int = 0                     # Consecutive times we've seen same stuck state
        self._STUCK_THRESHOLD = 3                      # After N stuck detections, reset with Home
        self._last_action_time: Optional[float] = None # Track when we last took an action
        self._ACTION_TIMEOUT = 45.0                    # If no progress in N seconds, consider stuck

        # Consecutive MENU verdicts vetoed by _is_video_player_overlay().
        # Reset when any other action runs or the overlay disappears.
        # Escalation: 3 vetoes -> single 'back' press to dismiss overlay,
        # 6 vetoes -> full YouTube relaunch.
        self._overlay_veto_count: int = 0
        self._OVERLAY_VETO_BACK_AT = 3
        self._OVERLAY_VETO_RESET_AT = 6

        # Consecutive MENU verdicts whose select-dispatch was skipped
        # because OCR showed no home-screen keywords. That guard prevents
        # sign-in traps, but with no escalation it can no-op FOREVER on an
        # unrecognized dead-end screen (observed live 2026-07-02: the
        # "Sign in to YouTube TV" activation screen stalled the session
        # 40 min with a MENU-skip every ~33s). After N consecutive skips
        # (~3 min) with no audio, no overlay, and no recognizable screen,
        # escape with Back presses + content selection.
        self._menu_skip_count: int = 0
        # Consecutive audio-vetoed Roku-home OCR matches (see dispatch escalation)
        self._roku_home_veto_streak: int = 0
        self._ROKU_HOME_VETO_ESCAPE_AT = 6  # ~3 min at the ~33s monitor cycle
        self._MENU_SKIP_ESCAPE_AT = 5

        # Music mode: steer content toward music videos (higher ad density).
        # When on, every YouTube (re)launch deep-links to a rotating seed
        # from MUSIC_VIDEO_SEEDS, and the PLAYING branch watches OCR text
        # for music evidence, re-steering when the autoplay chain drifts to
        # non-music content. Persisted with the other autonomous settings.
        self._music_mode = False
        self._music_seed_index: int = 0
        self._music_no_evidence_checks: int = 0
        self._MUSIC_STEER_AFTER = 5  # info-bearing checks w/o music evidence

        # Timestamp of the last _is_audio_flowing() == True observation.
        # Destructive guards use _audio_recently_flowing() instead of the
        # instantaneous check: pre-roll ad pods have silent gaps of a few
        # seconds between ads, and a VLM misfire inside such a gap slipped
        # past the instantaneous guard (observed live 2026-07-02 17:21:
        # DIALOG verdict 4s after an ad block ended → back exited the
        # just-started video → home → re-select churn, 7 videos in 15 min).
        self._last_audio_flowing_time: Optional[float] = None
        self._AUDIO_RECENT_WINDOW = 20.0

        # Logging
        self._log_file = "/home/radxa/Minus/autonomous-mode-logs.md"

        # Load persisted settings
        self._load_settings()

    def _load_settings(self):
        """Load persisted autonomous mode settings."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    self._enabled = settings.get("enabled", False)
                    self._start_hour = settings.get("start_hour", self.DEFAULT_START_HOUR)
                    self._end_hour = settings.get("end_hour", self.DEFAULT_END_HOUR)
                    self._always_on = settings.get("always_on", False)
                    self._music_mode = settings.get("music_mode", False)
                    logger.info(f"[AutonomousMode] Loaded settings: enabled={self._enabled}, "
                               f"schedule={self._start_hour}:00-{self._end_hour}:00, "
                               f"always_on={self._always_on}, music_mode={self._music_mode}")
        except Exception as e:
            logger.warning(f"[AutonomousMode] Could not load settings: {e}")

    def _save_settings(self):
        """Save autonomous mode settings to disk."""
        try:
            settings = {
                "enabled": self._enabled,
                "start_hour": self._start_hour,
                "end_hour": self._end_hour,
                "always_on": self._always_on,
                "music_mode": self._music_mode,
                "last_updated": datetime.now(ET).isoformat(),
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f)
            logger.debug(f"[AutonomousMode] Settings saved")
        except Exception as e:
            logger.warning(f"[AutonomousMode] Could not save settings: {e}")

    def set_device_controller(self, controller, device_type: Optional[str] = None):
        """Set device controller reference.

        Args:
            controller: Device controller (FireTV, Roku, GoogleTV)
            device_type: Optional device type hint ('fire_tv', 'roku', 'google_tv')
                        If not provided, will be detected from controller class name.
        """
        self._device_controller = controller
        self._fire_tv = controller  # Legacy alias

        if device_type:
            self._device_type = device_type
        else:
            # Auto-detect device type from controller class name
            self._device_type = self._detect_device_type(controller)

        logger.info(f"[AutonomousMode] Device controller set: {self._device_type}")

    def set_fire_tv(self, controller):
        """Set Fire TV controller reference (legacy, use set_device_controller)."""
        self.set_device_controller(controller, DEVICE_TYPE_FIRE_TV)

    def set_roku(self, controller):
        """Set Roku controller reference."""
        self.set_device_controller(controller, DEVICE_TYPE_ROKU)

    def _detect_device_type(self, controller) -> str:
        """Detect device type from controller class name."""
        if controller is None:
            return DEVICE_TYPE_FIRE_TV  # Default

        class_name = controller.__class__.__name__.lower()
        if 'roku' in class_name:
            return DEVICE_TYPE_ROKU
        elif 'google' in class_name or 'android' in class_name:
            return DEVICE_TYPE_GOOGLE_TV
        else:
            return DEVICE_TYPE_FIRE_TV  # Default to Fire TV for backwards compatibility

    def set_ad_blocker(self, blocker):
        """Set ad blocker reference."""
        self._ad_blocker = blocker

    def set_vlm(self, vlm):
        """Set VLM reference for screen understanding."""
        self._vlm = vlm

    def set_frame_capture(self, capture):
        """Set frame capture reference."""
        self._frame_capture = capture

    def set_status_callback(self, callback: Callable[[dict], None]):
        """Set callback for status changes."""
        self._on_status_change = callback

    def start_if_enabled(self):
        """Start monitoring thread if autonomous mode was enabled (called on startup)."""
        if self._enabled:
            logger.info("[AutonomousMode] Autonomous mode was enabled, starting monitoring thread")
            self._start_thread()

    def set_schedule(self, start_hour: int, end_hour: int, always_on: bool = False) -> dict:
        """
        Set the autonomous mode schedule.

        Args:
            start_hour: Hour to start (0-23)
            end_hour: Hour to end (0-23)
            always_on: If True, run 24/7 regardless of hours

        Returns:
            Status dict
        """
        with self._lock:
            # Validate hours
            start_hour = max(0, min(23, start_hour))
            end_hour = max(0, min(23, end_hour))

            self._start_hour = start_hour
            self._end_hour = end_hour
            self._always_on = always_on

            self._save_settings()

            schedule_desc = "24/7" if always_on else f"{start_hour}:00-{end_hour}:00"
            logger.info(f"[AutonomousMode] Schedule set to {schedule_desc}")
            self._log_event(f"Schedule changed to {schedule_desc}")

        # Return status OUTSIDE lock (get_status may be slow due to device checks)
        return self.get_status()

    def set_music_mode(self, enabled: bool) -> dict:
        """Enable/disable music mode (steer playback toward music videos).

        Music videos carry a much higher ad load on YouTube than general
        content, so this mode maximizes ad training data per hour. See
        MUSIC_VIDEO_SEEDS / _check_music_drift for the mechanism.
        """
        with self._lock:
            self._music_mode = bool(enabled)
            self._music_no_evidence_checks = 0
            self._save_settings()

        state = "enabled" if self._music_mode else "disabled"
        logger.info(f"[AutonomousMode] Music mode {state}")
        self._log_event(f"Music mode {state}")
        return self.get_status()

    def is_scheduled_time(self) -> bool:
        """Check if current time is within the scheduled window."""
        if self._always_on:
            return True

        now = datetime.now(ET)
        current_hour = now.hour

        if self._start_hour <= self._end_hour:
            # Normal range (e.g., 9:00 to 17:00)
            return self._start_hour <= current_hour < self._end_hour
        else:
            # Overnight range (e.g., 22:00 to 6:00)
            return current_hour >= self._start_hour or current_hour < self._end_hour

    def get_next_window(self) -> tuple[datetime, datetime]:
        """Get the next autonomous mode window (start, end) in ET."""
        now = datetime.now(ET)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if self._always_on:
            # Always on - window is now to forever
            return now, now + timedelta(days=365)

        start_today = today.replace(hour=self._start_hour)
        end_today = today.replace(hour=self._end_hour)

        if self._start_hour <= self._end_hour:
            # Normal range
            if now < start_today:
                return start_today, end_today
            elif now < end_today:
                return start_today, end_today
            else:
                # Next window is tomorrow
                tomorrow = today + timedelta(days=1)
                return tomorrow.replace(hour=self._start_hour), tomorrow.replace(hour=self._end_hour)
        else:
            # Overnight range (e.g., 22:00 to 6:00)
            if now.hour >= self._start_hour:
                # Currently after start, end is tomorrow
                tomorrow = today + timedelta(days=1)
                return start_today, tomorrow.replace(hour=self._end_hour)
            elif now.hour < self._end_hour:
                # Currently before end (early morning)
                yesterday = today - timedelta(days=1)
                return yesterday.replace(hour=self._start_hour), end_today
            else:
                # Between end and start, next window starts today
                return start_today, (today + timedelta(days=1)).replace(hour=self._end_hour)

    def get_time_until_window(self) -> Optional[timedelta]:
        """Get time until next window starts. None if currently in window."""
        if self.is_scheduled_time():
            return None

        start, _ = self.get_next_window()
        now = datetime.now(ET)
        if start > now:
            return start - now
        return None

    def enable(self, manual: bool = False) -> dict:
        """
        Enable autonomous mode.

        Args:
            manual: If True, start immediately regardless of schedule

        Returns:
            Status dict
        """
        with self._lock:
            if self._enabled and not manual:
                pass  # Will return status outside lock
            else:
                self._enabled = True
                self._manual_override = manual

                # Persist setting
                self._save_settings()

                logger.info(f"[AutonomousMode] Enabled (manual={manual})")
                self._log_event("Autonomous mode ENABLED" + (" (manual)" if manual else " (scheduled)"))

                # Start the monitoring thread
                self._start_thread()

        # Return status OUTSIDE lock (get_status may be slow due to device checks)
        return self.get_status()

    def disable(self) -> dict:
        """Disable autonomous mode."""
        did_deactivate = False
        with self._lock:
            if not self._enabled:
                return self.get_status()

            self._enabled = False
            self._manual_override = False

            # Persist setting
            self._save_settings()

            # Stop if running (use unlocked version since we hold the lock)
            if self._active:
                did_deactivate = self._deactivate_unlocked()

            self._stop_thread()

            logger.info("[AutonomousMode] Disabled")
            self._log_event("Autonomous mode DISABLED")

        # Notify status change OUTSIDE lock (mirrors _activate; get_status may be slow)
        if did_deactivate and self._on_status_change:
            self._on_status_change(self.get_status())

        # Return status OUTSIDE lock (get_status may be slow due to device checks)
        return self.get_status()

    def toggle(self) -> dict:
        """Toggle autonomous mode on/off."""
        if self._enabled:
            return self.disable()
        else:
            return self.enable()

    def start_now(self) -> dict:
        """Start autonomous mode immediately, regardless of schedule."""
        return self.enable(manual=True)

    def get_status(self) -> dict:
        """Get current autonomous mode status."""
        is_scheduled = self.is_scheduled_time()
        next_start, next_end = self.get_next_window()
        time_until = self.get_time_until_window()

        schedule_str = "24/7" if self._always_on else f"{self._start_hour:02d}:00-{self._end_hour:02d}:00"

        # Check device connection (works for any device type)
        device_connected = False
        if self._device_controller:
            try:
                device_connected = self._device_controller.is_connected()
            except Exception:
                device_connected = False

        return {
            "enabled": self._enabled,
            "active": self._active,
            "manual_override": self._manual_override,
            "is_scheduled_time": is_scheduled,
            "always_on": self._always_on,
            "music_mode": self._music_mode,
            "start_hour": self._start_hour,
            "end_hour": self._end_hour,
            "schedule": schedule_str,
            "current_time_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "next_window_start": next_start.strftime("%Y-%m-%d %H:%M:%S") if not self._always_on else None,
            "next_window_end": next_end.strftime("%Y-%m-%d %H:%M:%S") if not self._always_on else None,
            "time_until_window": str(time_until).split(".")[0] if time_until else None,
            "device_type": self._device_type,
            "device_connected": device_connected,
            # Legacy field for backwards compatibility
            "fire_tv_connected": device_connected if self._device_type == DEVICE_TYPE_FIRE_TV else False,
            "stats": self.stats.to_dict(),
        }

    def _start_thread(self):
        """Start the autonomous mode monitoring thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="AutonomousMode",
            daemon=True
        )
        self._thread.start()

    def _stop_thread(self):
        """Stop the monitoring thread."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self):
        """Main autonomous mode loop."""
        logger.info("[AutonomousMode] Monitoring thread started")

        last_keepalive = 0
        on_nav_screen = True  # Start assuming we need fast checks

        while self._running and not self._stop_event.is_set():
            try:
                should_be_active = self._manual_override or self.is_scheduled_time()

                if should_be_active and not self._active:
                    # Activate autonomous mode
                    self._activate()
                elif not should_be_active and self._active and not self._manual_override:
                    # Deactivate (only if not manual override)
                    self._deactivate()

                if self._active:
                    # Adaptive keepalive: shorter on nav screens, longer when playing
                    keepalive_interval = KEEPALIVE_INTERVAL_NAV if on_nav_screen else KEEPALIVE_INTERVAL_PLAYING

                    # Keep YouTube running
                    now = time.time()
                    if now - last_keepalive > keepalive_interval:
                        took_action = self._ensure_youtube_playing()
                        last_keepalive = now
                        # If we took an action, we're probably on a nav screen - use fast checks
                        # If no action needed, video is playing - use slower checks
                        on_nav_screen = took_action

                # Update stats
                if self._active:
                    self.stats.last_activity = datetime.now(ET)

            except Exception as e:
                logger.error(f"[AutonomousMode] Loop error: {e}")
                self.stats.errors += 1

            # Wait for next check
            self._stop_event.wait(CHECK_INTERVAL)

        logger.info("[AutonomousMode] Monitoring thread stopped")

    def _activate(self):
        """Activate autonomous mode session."""
        # Quick state update inside lock
        with self._lock:
            if self._active:
                return

            self._active = True
            self.stats.reset()
            self.stats.session_start = datetime.now(ET)

            logger.info("[AutonomousMode] Session STARTED")
            self._log_event("Session STARTED")

        # Slow operations OUTSIDE lock to prevent blocking API calls
        self._launch_youtube()

        # Notify status change (also outside lock)
        if self._on_status_change:
            self._on_status_change(self.get_status())

    def _deactivate(self):
        """Deactivate autonomous mode session (acquires lock)."""
        with self._lock:
            did_deactivate = self._deactivate_unlocked()

        # Notify status change OUTSIDE lock (mirrors _activate; get_status may be slow)
        if did_deactivate and self._on_status_change:
            self._on_status_change(self.get_status())

    def _deactivate_unlocked(self):
        """Deactivate autonomous mode session (caller must hold lock).

        Returns True if a session was actually ended, False if it was already inactive.
        Callers holding the lock must fire the status callback themselves outside the lock;
        see `_deactivate` for the standalone path.
        """
        if not self._active:
            return False

        self._active = False
        self.stats.session_end = datetime.now(ET)

        duration = self.stats._get_duration_minutes()
        logger.info(f"[AutonomousMode] Session ENDED after {duration} minutes")
        self._log_event(f"Session ENDED - Duration: {duration}min, Videos: {self.stats.videos_played}, Ads: {self.stats.ads_detected}")
        return True

    def _is_youtube_app(self, app_name: str) -> bool:
        """Check if the app name matches any known YouTube package."""
        if not app_name:
            return False
        app_lower = app_name.lower()
        return any(pkg in app_lower for pkg in YOUTUBE_PACKAGES)

    # VLM prompt for screen-state classification. The LFM2.5-VL model
    # in `vlm.py` has a fixed 320-token prefill window, and after the
    # chat template + 256 image tokens there is only ~40 tokens of
    # headroom for the user question. The previous, longer phrasing
    # tokenised to 326 (over by 6) and silently truncated the
    # [IM_START] assistant\n suffix → garbage logits.
    # `vlm.py.query_image` runs prefill-only and picks the class whose
    # first-token logit is highest (max over no-leading-space and
    # leading-space spellings); there is no autoregressive decode, so
    # only the FIRST emitted token's logits matter. The prompt prefix
    # "Classify this TV screen" is also used by query_image to reuse a
    # cached token-id sequence — keep that prefix intact when editing.
    SCREEN_QUERY_PROMPT = (
        "Classify this TV screen: PLAYING, PAUSED, DIALOG, MENU, or SCREENSAVER?"
    )

    def _query_screen(self) -> Optional[str]:
        """Use VLM to understand what's currently on screen.

        Returns:
            VLM response (should be one of: PLAYING, PAUSED, DIALOG, MENU, SCREENSAVER),
            or None if unavailable.
        """
        if not self._vlm or not self._vlm.is_ready or not self._frame_capture:
            return None

        try:
            frame = self._frame_capture.capture()
            if frame is None:
                return None

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, frame)

            try:
                response, elapsed = self._vlm.query_image(tmp_path, self.SCREEN_QUERY_PROMPT)
                logger.info(f"[AutonomousMode] VLM screen query ({elapsed:.1f}s): {response}")
                return response
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"[AutonomousMode] VLM screen query failed: {e}")
            return None

    def _determine_action(self, screen_desc: str) -> str:
        """Determine what action to take based on VLM screen classification.

        Expects structured response (PLAYING/PAUSED/DIALOG/MENU/SCREENSAVER).
        Falls back to keyword matching if VLM gives a longer response.

        Returns one of: 'none', 'play', 'dismiss', 'select', 'launch'
        """
        if not screen_desc:
            return "none"

        desc = screen_desc.strip().upper()

        # Check for structured single-word responses first
        if desc.startswith("PLAYING"):
            return "none"
        if desc.startswith("DIALOG"):
            return "dismiss"
        if desc.startswith("SCREENSAVER"):
            return "launch"
        if desc.startswith("MENU"):
            return "select"
        if desc.startswith("PAUSED"):
            return "play"

        # Fallback: keyword matching on longer responses
        desc_lower = screen_desc.lower()

        # "still watching" is a strong signal for dialog regardless of context
        if "still watching" in desc_lower or "still there" in desc_lower:
            return "dismiss"

        if "screensaver" in desc_lower or "black screen" in desc_lower:
            return "launch"

        if "home screen" in desc_lower or "browse" in desc_lower or "thumbnail" in desc_lower:
            return "select"

        # Only match "paused" as a positive statement, not "not paused"
        if "paused" in desc_lower and "not paused" not in desc_lower:
            return "play"

        if "playing" in desc_lower:
            return "none"

        # Unknown state - do nothing to avoid disruption
        return "none"

    def _launch_youtube(self) -> bool:
        """Launch YouTube app on the connected streaming device."""
        if not self._device_controller or not self._device_controller.is_connected():
            logger.warning(f"[AutonomousMode] {self._device_type or 'Device'} not connected, cannot launch YouTube")
            return False

        try:
            # Music mode: every (re)launch path funnels through here, so a
            # successful seed deep-link keeps sessions starting in music
            # land (YouTube autoplay then chains more music). Falls through
            # to the plain launch when unsupported or the deep-link fails.
            if self._music_mode and self._launch_music_seed():
                return True

            # Device-specific YouTube launch
            if self._device_type == DEVICE_TYPE_ROKU:
                return self._launch_youtube_roku()
            elif self._device_type in (DEVICE_TYPE_FIRE_TV, DEVICE_TYPE_GOOGLE_TV):
                return self._launch_youtube_android()
            else:
                # Fallback: try Android method
                return self._launch_youtube_android()

        except Exception as e:
            logger.error(f"[AutonomousMode] Failed to launch YouTube: {e}")
            self.stats.errors += 1
            return False

    def _check_music_drift(self, force_text: str = None) -> bool:
        """Music mode: watch OCR text for music evidence while playing.

        Called from the verified-PLAYING branch (~2 min cycle). OCR only
        sees title text occasionally (player overlay, end cards), so most
        cycles carry no information: a miss is only counted when there IS
        meaningful text on screen (≥12 chars), the ad blocker is not
        mid-block (ad copy says nothing about the underlying video), and
        none of MUSIC_EVIDENCE_KEYWORDS match. After _MUSIC_STEER_AFTER
        such misses the autoplay chain has demonstrably drifted off music —
        re-steer with the next seed deep-link. Note this deliberately
        interrupts a playing video: that is the mode's policy (the video is
        playing but it isn't music), not a misclassification recovery, and
        the info-bearing-text requirement is what keeps it from firing on
        an actual music video that simply shows no title text.

        Returns True if a steer was triggered (for tests).
        """
        if not self._music_mode:
            return False
        try:
            if self._ad_blocker is None:
                return False
            if getattr(self._ad_blocker, 'is_visible', False):
                return False  # mid ad-block: OCR text is the ad, not the video
            if force_text is not None:
                combined = force_text.lower()
            else:
                texts = getattr(self._ad_blocker, 'last_ocr_texts', None) or []
                combined = ' '.join(str(t) for t in texts).lower()
            if len(combined.strip()) < 12:
                return False  # no information this cycle
            if any(k in combined for k in self.MUSIC_EVIDENCE_KEYWORDS):
                self._music_no_evidence_checks = 0
                return False
            self._music_no_evidence_checks += 1
            logger.info(f"[AutonomousMode] Music mode: no music evidence in OCR "
                        f"({self._music_no_evidence_checks}/{self._MUSIC_STEER_AFTER})")
            if self._music_no_evidence_checks >= self._MUSIC_STEER_AFTER:
                logger.info("[AutonomousMode] Music mode: drifted off music - steering to seed")
                self._log_event("Music mode: drifted off music - steering to seed")
                self._music_no_evidence_checks = 0
                self._launch_music_seed()
                return True
        except Exception as e:
            logger.debug(f"[AutonomousMode] Music drift check error: {e}")
        return False

    def _launch_music_seed(self) -> bool:
        """Deep-link YouTube to the next seed music video (music mode).

        Roku only for now: ECP launch supports ?contentId=<video id>.
        Controllers without launch_app_with_content return False so the
        caller proceeds with the plain launch. Rotates MUSIC_VIDEO_SEEDS
        so one dead/aged seed can't wedge the mode.
        """
        ctrl = self._device_controller
        if not ctrl or not ctrl.is_connected():
            return False
        if not hasattr(ctrl, 'launch_app_with_content'):
            return False

        seed = self.MUSIC_VIDEO_SEEDS[self._music_seed_index % len(self.MUSIC_VIDEO_SEEDS)]
        self._music_seed_index += 1
        try:
            if ctrl.launch_app_with_content('youtube', seed):
                time.sleep(3)
                self._music_no_evidence_checks = 0
                logger.info(f"[AutonomousMode] Music seed launched: {seed}")
                self._log_event(f"Music seed launched ({seed})")
                return True
        except Exception as e:
            logger.warning(f"[AutonomousMode] Music seed launch failed: {e}")
        return False

    def _launch_youtube_roku(self) -> bool:
        """Launch YouTube on Roku using ECP launch API."""
        try:
            logger.info("[AutonomousMode] Launching YouTube on Roku...")

            # Roku controller has launch_app method
            if hasattr(self._device_controller, 'launch_app'):
                result = self._device_controller.launch_app('youtube')
                if result:
                    time.sleep(3)
                    logger.info("[AutonomousMode] YouTube launched on Roku")
                    self._log_event("YouTube launched (Roku)")
                    return True
                else:
                    logger.error("[AutonomousMode] Roku launch_app returned False")
                    return False
            else:
                logger.error("[AutonomousMode] Roku controller missing launch_app method")
                return False

        except Exception as e:
            logger.error(f"[AutonomousMode] Roku YouTube launch error: {e}")
            return False

    def _launch_youtube_android(self) -> bool:
        """Launch YouTube on Fire TV / Android TV / Google TV using ADB."""
        try:
            # Check current app if the controller supports it
            if hasattr(self._device_controller, 'get_current_app'):
                current = self._device_controller.get_current_app()
                logger.debug(f"[AutonomousMode] Current app: {current}")
                if self._is_youtube_app(current):
                    logger.debug("[AutonomousMode] YouTube already running")
                    return True

            # Launch YouTube via ADB intent
            logger.info(f"[AutonomousMode] Launching YouTube on {self._device_type}...")

            # Access internal _device for ADB shell command
            if hasattr(self._device_controller, '_lock') and hasattr(self._device_controller, '_device'):
                with self._device_controller._lock:
                    if self._device_controller._device:
                        # Try multiple package names
                        for pkg in YOUTUBE_PACKAGES:
                            try:
                                self._device_controller._device.adb_shell(
                                    f"am start -a android.intent.action.MAIN -c android.intent.category.LEANBACK_LAUNCHER {pkg}"
                                )
                                break
                            except Exception:
                                continue

            time.sleep(3)
            logger.info("[AutonomousMode] YouTube launched")
            self._log_event(f"YouTube launched ({self._device_type})")
            return True

        except Exception as e:
            logger.error(f"[AutonomousMode] Android YouTube launch error: {e}")
            return False

    def _compute_frame_hash(self, frame) -> int:
        """Compute a perceptual hash (dHash) of a frame for change detection.

        Returns a 64-bit integer hash. Frames that look similar will have
        hashes with low Hamming distance.
        """
        small = cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        diff = gray[:, 1:] > gray[:, :-1]
        return int(np.packbits(diff.flatten())[:8].view(np.uint64)[0])

    def _is_audio_pipeline_available(self) -> bool:
        """Check if the audio pipeline is actually functional.

        When HDMI-TX is disconnected, the alsasink can't open and the pipeline
        never receives buffers. In that case audio-based pause detection is
        unreliable — treat as unavailable rather than "not flowing".
        """
        if self._ad_blocker and hasattr(self._ad_blocker, 'audio') and self._ad_blocker.audio:
            try:
                status = self._ad_blocker.audio.get_status()
                buffer_age = status.get('last_buffer_age', -1)
                state = status.get('state', 'stopped')
                # Pipeline is available if it's playing AND has received a buffer
                # buffer_age == -1 means no buffer ever received → pipeline broken
                if buffer_age < 0 or state in ('stopped', 'unknown'):
                    return False
                return True
            except Exception:
                return False
        # Fallback: check ALSA capture; if it's not open at all, pipeline isn't up
        try:
            with open("/proc/asound/card4/pcm0c/sub0/status", 'r') as f:
                content = f.read().strip()
            return content != 'closed'
        except Exception:
            return False

    # Minimum recent RMS level to count as "real audio content" rather than
    # silent HDMI carrier. HDMI sources keep emitting silence buffers when
    # the connected device is paused (Roku/Fire TV/Google TV all do this) —
    # buffer-age alone returns True for silence and would make every paused
    # MENU verdict get vetoed by `audio flowing`. The visual-curved RMS in
    # audio.py rests near 0.0 in true silence and rises into the 0.02-0.30+
    # range with speech/music. 0.01 is comfortably below speech but above
    # the noise floor measured on the HDMI-RX device.
    AUDIO_SILENCE_THRESHOLD = 0.01

    def _is_audio_flowing(self) -> bool:
        """Check if audio is currently flowing with REAL content (not just
        the HDMI carrier emitting silence on a paused source).

        Uses the ad_blocker's audio module if available, otherwise falls
        back to ALSA capture device status (less reliable — RUNNING is
        true even during silence). When the audio module is available we
        require BOTH: a fresh buffer (<3s old) AND a recent RMS level
        above the silence threshold. This was added 2026-06-06 after
        the conservative MENU guard was vetoing every action because
        Roku's HDMI output emits a silent 48kHz stream while paused.
        """
        # Method 1: Check via ad_blocker's audio module
        if self._ad_blocker and hasattr(self._ad_blocker, 'audio') and self._ad_blocker.audio:
            try:
                status = self._ad_blocker.audio.get_status()
                buffer_age = status.get('last_buffer_age', 999)
                recent_level = status.get('recent_level', 0.0)
                buffer_fresh = 0 <= buffer_age < 3.0
                has_content = recent_level >= self.AUDIO_SILENCE_THRESHOLD
                is_flowing = buffer_fresh and has_content
                logger.debug(
                    f"[AutonomousMode] Audio buffer age={buffer_age:.1f}s "
                    f"recent_level={recent_level:.3f} "
                    f"(threshold={self.AUDIO_SILENCE_THRESHOLD}) "
                    f"flowing={is_flowing}"
                )
                if is_flowing:
                    self._last_audio_flowing_time = time.time()
                return is_flowing
            except Exception:
                pass

        # Method 2: Check ALSA capture device status directly
        try:
            alsa_status_path = "/proc/asound/card4/pcm0c/sub0/status"
            with open(alsa_status_path, 'r') as f:
                content = f.read()
            is_running = 'state: RUNNING' in content
            logger.debug(f"[AutonomousMode] ALSA capture: {'RUNNING' if is_running else 'not running'}")
            if is_running:
                self._last_audio_flowing_time = time.time()
            return is_running
        except Exception:
            return False

    def _audio_recently_flowing(self) -> bool:
        """True if audio is flowing now OR flowed within the last
        _AUDIO_RECENT_WINDOW seconds.

        Used by DESTRUCTIVE guards (dismiss/back, keyboard escape) where a
        few-second silent gap between pre-roll ads must not count as "not
        playing". A real blocking dialog / sign-in screen sits silent for
        minutes, so it still gets handled one cycle after the window
        expires (~1 min at the 33s cycle cadence).
        """
        if self._is_audio_flowing():
            return True
        if self._last_audio_flowing_time is None:
            return False
        return (time.time() - self._last_audio_flowing_time) < self._AUDIO_RECENT_WINDOW

    # Keywords that indicate YouTube login/account selection screen
    # Includes variants to handle OCR noise (missing/merged spaces)
    LOGIN_SCREEN_KEYWORDS = [
        'watch as guest',
        'watchas guest',     # OCR sometimes merges "watch as"
        'add a kid account',
        'add akid account',  # OCR sometimes merges "a kid"
        'kid account',       # Specific to account selection
        'choose account',
        'choose an account',
        'switch account',
    ]

    # Keywords that indicate we're signed out and need to sign in
    # "Make YouTube your own" is the sign-out state prompt
    SIGNED_OUT_KEYWORDS = [
        'make youtube your own',
        'makeyoutube your own',  # OCR sometimes merges
        'you are in guest mode',
        'guest mode',
        'sign in to see the latest',
    ]

    # Keywords that indicate a survey/dialog that should be skipped
    SURVEY_KEYWORDS = [
        'skip survey',
        'skipsurvey',
        'advertiser survey',
        'submit answers',
    ]

    # Keywords that indicate we're on the Roku home screen (not YouTube)
    # These are app names and UI elements only visible on Roku home.
    # NOTE: 'watchnow' was removed 2026-07-02 — "Watch now" is a generic
    # ad CTA (observed live: a YouTube ad's "Watch now" button OCR-merged
    # to 'watchnow' and triggered a YouTube relaunch mid-video). The ECP
    # active-app check is the authoritative Roku-home detector; this OCR
    # fallback must only contain strings that CANNOT appear inside apps.
    ROKU_HOME_KEYWORDS = [
        'rokuchannel',
        'roku channel',
        'ad-free tv',
        'frndly',            # Frndly TV app on Roku
        'press for more',    # Roku UI prompt
    ]

    # Phrases that only appear on YouTube TV (the paid live-TV service)
    # promo/upsell prompts that autonomous mode sometimes lands on from
    # the YouTube home screen or after a mis-navigation. These screens are
    # dead ends: every selectable option leads to sign-up/payment flows.
    # Each phrase is specific to the promo layout and never appears during
    # normal video playback or on the regular home rows.
    YOUTUBE_TV_PROMPT_KEYWORDS = [
        'cable-free live tv',
        'cablefree live tv',
        'cable free live tv',
        'try youtube tv',
        'tryyoutube tv',     # OCR sometimes merges spaces
        'youtube tv free trial',
        'new users only. terms apply',
        # The signup flow itself ("Sign in to YouTube TV" + activation
        # code + tv.youtube.com/start). Observed live 2026-07-02: this
        # screen stalled autonomous mode for 40 min because no detector
        # knew it. The keyboard-stuck check also catches it (activation
        # code markers) and runs first; this entry is belt-and-braces.
        'sign in to youtube tv',
        'signin to youtube tv',
    ]

    # Generic promo markers: any of these *together with* "youtube tv" in
    # the OCR text marks the screen as a YouTube TV upsell. "youtube tv"
    # alone is NOT enough — video titles and search results legitimately
    # contain it (e.g. "YouTube TV review").
    YOUTUBE_TV_PROMPT_MARKERS = [
        'free trial',
        'try it free',
        'try free',
        'start trial',
        'per month',
        '/month',
        '/mo',
        'sign up',
        'live tv',
    ]

    # Keywords that indicate we're on a keyboard/sign-in screen (STUCK - need to escape)
    # These screens require manual input we can't provide - press Back to escape
    KEYBOARD_STUCK_KEYWORDS = [
        '12#',               # Keyboard symbol toggle
        'qwerty',            # Keyboard layout
        'enter email',
        'enter password',
        'phone number',
        'verification code',
        'yt.be/activate',    # YouTube sign-in with code screen
        'enter the code',    # Sign-in code entry
        'enter this code',   # YouTube TV activation screen wording
        'enterthis code',    # OCR-merged variant (observed live 2026-07-02)
        'scan qr code',      # QR code sign-in
        'scan with your phone',  # "Or scan with your phone" (YouTube TV activation)
        'youtube.com/start', # tv.youtube.com/start activation URL
        'sign in with your phone',  # Mobile sign-in prompt
        'add your google',   # Google account addition
    ]

    # Keywords that indicate the YouTube VIDEO PLAYER overlay is visible
    # (the floating UI that appears over a *playing* video — Description /
    # Subscribe / cc / Up next / "Sign in" CTA, alongside time markers like
    # "0:42 | 0:12"). VLM misclassifies these frames as MENU because the
    # overlay buttons look like a menu, but the underlying video is actually
    # PLAYING and the overlay auto-dismisses in a few seconds.
    #
    # If we treat this as a MENU and send `down + select`, focus walks
    # through the overlay buttons and reliably lands on "Sign in" — which
    # opens the Google sign-in flow (keyboard / QR code / yt.be/activate),
    # an unrecoverable trap for autonomous mode. Observed live on minus-2
    # 2026-05-23: OCR text "Sign in | Description | cc | Subscribe | 0:42 |
    # 0:12 | ... 875K views · 3d ago" → MENU verdict → down+select → sign-in
    # flow → ~2 min before stuck-detection escapes.
    #
    # Veto signature: at least TWO of these terms present in OCR text.
    # Single-term match (e.g. just "subscribe") is too weak — YouTube
    # Shorts and other states show it alone.
    VIDEO_PLAYER_OVERLAY_KEYWORDS = [
        'description',       # Player overlay "Description" button
        'up next',           # Right-side "Up next" panel header
        'upnext',            # OCR-merged variant
        'autoplay',          # Autoplay toggle in player overlay
        'cc',                # Closed-captions toggle (very short — see check)
    ]

    # Keywords that indicate YouTube home/browse screen (need to select a video)
    # NOTE: "subscribe" and "description" removed - they appear on paused videos too
    HOME_SCREEN_KEYWORDS = [
        'new to you',
        'newtoyou',          # OCR sometimes merges spaces
        'recommended',       # Main home screen section
        'trending',
        'subscriptions',
        'library',
        # 'views' removed: too common (any playing video info panel shows "347M views")
        'year ago',
        'month ago',
        'day ago',
        'hour ago',
        # New YouTube TV browse layout (observed Apr 2026): top-of-page nav
        # row reads "Search" + "Shorts", with category rows like
        # "Food processing and more". A playing video's info panel does not
        # show these terms, so they're safe home-screen markers.
        'shorts',
        'search',
    ]

    # Keywords that strongly indicate a playing AD (not home screen).
    # When these are present, skip home-screen detection — "Sponsored" tiles
    # on YouTube home will coexist with the keywords above, but during an ad
    # these appear alone alongside "Sponsored".
    AD_ONLY_KEYWORDS = [
        'visit advertiser',
        'send to phone',
        'sendtophone',
        'skip in',
        'skip ad',
    ]

    # Indicators that the currently-highlighted tile (or its row) is a LIVE
    # YouTube stream. Live streams are bad picks for ad-training autonomous
    # mode because: (a) they often have long unskippable mid-roll pods,
    # (b) audio/video are stream-paced (drift causes pause-detection FPs),
    # (c) "live chat" overlays confuse VLM screen classification.
    # Phrases here are specific enough to avoid common false matches
    # ("olive", "delivery", "alive", etc. don't contain these tokens).
    LIVE_CONTENT_KEYWORDS = [
        'live now',
        'livenow',           # OCR merges spaces
        'streaming now',
        'streamingnow',
        'started streaming',
        'watching now',
        'watchingnow',
        'is live',
        'islive',
        ' live ·',           # YouTube "· LIVE ·" badge separator pattern
        '· live',
        'live ·',
    ]

    def _is_youtube_login_screen(self) -> bool:
        """Check if we're on the YouTube login/account selection screen using OCR keywords.

        The login screen shows:
        - "Watch as guest"
        - "Add account"
        - "Add a kid account"
        - User profile names

        VLM often misclassifies this static screen as PLAYING.
        Uses the last_ocr_texts from the ad_blocker (most recent OCR results).
        """
        try:
            # Check the most recent OCR texts from ad_blocker
            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    for keyword in self.LOGIN_SCREEN_KEYWORDS:
                        if keyword in combined:
                            logger.info(f"[AutonomousMode] YouTube login screen detected: '{keyword}'")
                            return True

            # Fallback: High consecutive static count suggests stuck on login screen
            if self._consecutive_static >= 4:
                logger.info("[AutonomousMode] High static count - might be login screen")
                return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Login screen check failed: {e}")
            return False

    def _has_live_content_indicator(self) -> bool:
        """Detect that the currently-visible YouTube screen contains a LIVE
        stream indicator (a "LIVE" badge or "Watching now" count). When this
        is true on a home screen, the highlighted tile is most likely the
        first live row — autonomous mode should push selection further down
        to skip past it and pick recorded content instead.
        """
        try:
            if not self._ad_blocker or not hasattr(self._ad_blocker, 'last_ocr_texts'):
                return False
            texts = self._ad_blocker.last_ocr_texts
            if not texts:
                return False
            combined = ' '.join(str(t) for t in texts).lower()
            for kw in self.LIVE_CONTENT_KEYWORDS:
                if kw in combined:
                    logger.info(f"[AutonomousMode] Live-content indicator detected ('{kw}') — will skip past live tile")
                    return True
            return False
        except Exception as e:
            logger.debug(f"[AutonomousMode] Live-indicator check failed: {e}")
            return False

    def _is_youtube_home_screen(self) -> bool:
        """Check if we're on YouTube home/browse screen showing video thumbnails.

        The home screen shows video recommendations with:
        - "New to you", "Trending", "Subscriptions", "Library" tabs
        - Video thumbnails with view counts ("3.3M views · 1 year ago")

        When VLM misclassifies this as PLAYING, we need to select a video
        instead of sending play_pause.
        """
        try:
            # Don't treat as home screen if ad blocker is actively blocking
            # (we know it's an ad, not home screen).
            if self._ad_blocker and getattr(self._ad_blocker, 'is_visible', False):
                return False

            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    # If ad-specific keywords are present, this is an ad, not home
                    for ad_kw in self.AD_ONLY_KEYWORDS:
                        if ad_kw in combined:
                            return False

                    for keyword in self.HOME_SCREEN_KEYWORDS:
                        if keyword in combined:
                            logger.info(f"[AutonomousMode] YouTube home screen detected: '{keyword}'")
                            return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Home screen check failed: {e}")
            return False

    def _is_video_player_overlay(self) -> bool:
        """Detect that the YouTube video player overlay is visible on a
        *playing* video — the floating UI with Description / Subscribe / cc /
        Up next, plus a time marker like "0:42 | 0:12".

        This veto exists because VLM classifies these frames as MENU and the
        downstream `down + select` MENU action navigates the overlay UI and
        lands on "Sign in", trapping autonomous mode in the Google sign-in
        flow. See VIDEO_PLAYER_OVERLAY_KEYWORDS comment block for the live
        incident that prompted this.

        Signature requires BOTH:
          - at least one overlay-specific keyword (description / up next /
            autoplay / cc-as-whole-token)
          - a time marker like "0:42" or "10:23" (current-time / duration
            display — never present on a static menu or home screen)

        Requiring both prevents false positives from the home page (which
        also has "subscribe" sometimes), Shorts, and other UI states.
        """
        try:
            if not self._ad_blocker or not hasattr(self._ad_blocker, 'last_ocr_texts'):
                return False
            texts = self._ad_blocker.last_ocr_texts
            if not texts:
                return False
            combined = ' '.join(str(t) for t in texts).lower()

            # Ad blocker actively blocking — definitely not a benign overlay.
            if getattr(self._ad_blocker, 'is_visible', False):
                return False

            # 1) at least one overlay-specific keyword
            overlay_hits = []
            for kw in self.VIDEO_PLAYER_OVERLAY_KEYWORDS:
                if kw == 'cc':
                    # 'cc' is short and substring-matches in countless words
                    # (e.g. "accept", "success"). Require whole-token form.
                    if re.search(r'\bcc\b', combined):
                        overlay_hits.append('cc')
                elif kw in combined:
                    overlay_hits.append(kw)
            if not overlay_hits:
                return False

            # 2) time marker like "0:42" or "10:23". \b\d{1,2}:\d{2}\b is
            # tight enough to avoid OCR garbage (single colons in normal text)
            # while catching both current-time and duration tokens.
            if not re.search(r'\b\d{1,2}:\d{2}\b', combined):
                return False

            logger.info(f"[AutonomousMode] Video player overlay visible "
                        f"(keywords={overlay_hits}) — vetoing MENU action")
            return True

        except Exception as e:
            logger.debug(f"[AutonomousMode] Overlay check failed: {e}")
            return False

    def _is_vertical_video_frame(self, frame) -> bool:
        """Detect a pillarboxed vertical video (Shorts/Reels format) frame.

        A 9:16 video centered on a 16:9 display fills only ~32% of the
        width; the black bars are ~34% on each side. We sample the outer
        22% on each side (safely inside the bars) and the middle 20%
        (safely inside the video) and require:
          - both side bands dark AND uniform (low std) — true pillarbox
            bars are flat black; a dark movie scene has texture
          - the center clearly brighter than the sides

        Callers double-confirm across two frames a couple of seconds apart
        so a single unlucky movie shot can't false-trigger a Back press.
        """
        try:
            if frame is None:
                return False
            h, w = frame.shape[:2]
            if h == 0 or w <= h:
                return False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            side_w = max(1, int(w * 0.22))
            left = gray[:, :side_w].astype(np.float32)
            right = gray[:, w - side_w:].astype(np.float32)
            center = gray[:, int(w * 0.40):int(w * 0.60)].astype(np.float32)

            left_mean, right_mean = float(left.mean()), float(right.mean())
            sides_dark = left_mean < 25 and right_mean < 25
            # True pillarbox bars are FLAT (std ~1-3 from JPEG noise on
            # video-range black). Keep this tight: even dim random noise
            # grays out to std ~8 after BGR→gray conversion, and a dark
            # movie scene with a lit center subject must not trigger a
            # Back press on a playing video.
            sides_flat = float(left.std()) < 6 and float(right.std()) < 6
            center_mean = float(center.mean())
            center_lit = center_mean > 40 and center_mean > 4 * max(left_mean, right_mean, 1.0)
            return sides_dark and sides_flat and center_lit
        except Exception as e:
            logger.debug(f"[AutonomousMode] Vertical-frame check error: {e}")
            return False

    def _is_youtube_shorts(self) -> bool:
        """Check if we're watching YouTube Shorts (short-form vertical video).

        We want to exit Shorts and find full-length videos. Two
        complementary signals:

        1. OCR signature — "@handle" + "subscribe" visible with no video
           duration/progress time marker (Shorts have no progress bar).
           Catches the moments the Shorts UI overlay is on screen.
        2. Vertical-format frame — pillarboxed 9:16 video (dark uniform
           side bars + lit center), confirmed on TWO frames ~2s apart.
           Catches overlay-less playback where OCR sees nothing.
        """
        try:
            # Signal 1: OCR signature
            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    has_handle = '@' in combined
                    has_subscribe = 'subscribe' in combined
                    # Full videos show a time marker like "10:23"; Shorts
                    # never do. (The old check here was a tautology — it
                    # matched any digit anywhere + any colon anywhere, so
                    # it almost always suppressed detection.)
                    has_duration = re.search(r'\b\d{1,2}:\d{2}\b', combined) is not None

                    if has_handle and has_subscribe and not has_duration:
                        logger.info("[AutonomousMode] Shorts detected via OCR: "
                                    "@handle + subscribe, no duration")
                        return True

            # Signal 2: vertical-format (pillarboxed) frames, double-checked.
            # Skip while the blocking overlay is active — an ad is being
            # handled and a Back press could exit the underlying video.
            if self._frame_capture and not (
                self._ad_blocker and getattr(self._ad_blocker, 'is_visible', False)
            ):
                frame = self._frame_capture.capture()
                if self._is_vertical_video_frame(frame):
                    time.sleep(2)
                    frame2 = self._frame_capture.capture()
                    if self._is_vertical_video_frame(frame2):
                        logger.info("[AutonomousMode] Shorts detected via "
                                    "vertical-format (pillarboxed) frames")
                        return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Shorts check failed: {e}")
            return False

    def _is_signed_out_screen(self) -> bool:
        """Check if we're on the signed-out "Make YouTube your own" screen.

        This screen appears when YouTube is launched without a signed-in account.
        Shows "Make YouTube your own" with a "Sign in" button.
        We need to click Sign in to get to account selection.
        """
        try:
            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    for keyword in self.SIGNED_OUT_KEYWORDS:
                        if keyword in combined:
                            logger.info(f"[AutonomousMode] Signed-out screen detected: '{keyword}'")
                            return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Signed-out screen check failed: {e}")
            return False

    def _has_accounts_visible(self) -> bool:
        """Check if account names are visible on the signed-out screen.

        When "Make YouTube your own" shows with existing accounts (e.g., @username),
        we should navigate down to select an account instead of clicking Sign in.
        """
        try:
            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()
                    # Look for @ symbols indicating account names
                    # Also check for "add account" which appears when accounts exist
                    if '@' in combined or 'add account' in combined or 'addaccount' in combined:
                        return True
            return False
        except Exception:
            return False

    def _is_survey_screen(self) -> bool:
        """Check if there's a survey dialog that should be skipped.

        YouTube shows advertiser surveys with "Skip survey" button.
        We need to navigate to and click the skip button.
        """
        try:
            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    for keyword in self.SURVEY_KEYWORDS:
                        if keyword in combined:
                            logger.info(f"[AutonomousMode] Survey dialog detected: '{keyword}'")
                            return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Survey screen check failed: {e}")
            return False

    def _is_youtube_tv_prompt(self) -> bool:
        """Check if we've landed on a YouTube TV (live-TV service) upsell prompt.

        These promo screens ("Try YouTube TV", "Cable-free live TV", free
        trial offers) are dead ends for autonomous mode: every option leads
        to a sign-up/payment flow. Detected via OCR — either an exact promo
        phrase, or "youtube tv" combined with a promo marker (free trial /
        pricing / sign up). The escape action is Back, escalating to a full
        YouTube relaunch if the prompt keeps reappearing.
        """
        try:
            # If the ad blocker is actively blocking, this is an *ad* for
            # YouTube TV — let ad handling deal with it. Pressing Back
            # mid-ad could exit the underlying video.
            if self._ad_blocker and getattr(self._ad_blocker, 'is_visible', False):
                return False

            if not self._ad_blocker or not hasattr(self._ad_blocker, 'last_ocr_texts'):
                return False
            texts = self._ad_blocker.last_ocr_texts
            if not texts:
                return False
            combined = ' '.join(str(t) for t in texts).lower()

            for kw in self.YOUTUBE_TV_PROMPT_KEYWORDS:
                if kw in combined:
                    logger.info(f"[AutonomousMode] YouTube TV prompt detected: '{kw}'")
                    return True

            if 'youtube tv' in combined or 'youtubetv' in combined:
                for marker in self.YOUTUBE_TV_PROMPT_MARKERS:
                    if marker in combined:
                        logger.info(f"[AutonomousMode] YouTube TV prompt detected: "
                                    f"'youtube tv' + '{marker}'")
                        return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] YouTube TV prompt check failed: {e}")
            return False

    def _is_roku_home_screen(self) -> bool:
        """Check if we're on the Roku home screen (not YouTube) using OCR.

        The Roku home screen shows app tiles like "Roku Channel", "Frndly TV", "hulu"
        that are never visible inside YouTube. This serves as a fallback when the
        ECP active-app query doesn't work or is slow.
        """
        if self._device_type != DEVICE_TYPE_ROKU:
            return False  # Only for Roku

        try:
            # Ad blocker actively blocking → the OCR text is from an AD,
            # not the Roku home screen (observed live 2026-07-02: an ad's
            # "Watch now" CTA matched this check mid-block and relaunched
            # YouTube during a playing video).
            if self._ad_blocker and getattr(self._ad_blocker, 'is_visible', False):
                return False

            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    for keyword in self.ROKU_HOME_KEYWORDS:
                        if keyword in combined:
                            logger.info(f"[AutonomousMode] Roku home screen detected via OCR: '{keyword}'")
                            return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Roku home screen check failed: {e}")
            return False

    def _is_keyboard_stuck_screen(self) -> bool:
        """Check if we're on a keyboard/sign-in screen that requires manual input.

        These screens (email entry, password, verification code) can't be automated.
        We need to press Back to escape and try a different path.
        """
        try:
            if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                texts = self._ad_blocker.last_ocr_texts
                if texts:
                    combined = ' '.join(str(t) for t in texts).lower()

                    for keyword in self.KEYBOARD_STUCK_KEYWORDS:
                        if keyword in combined:
                            logger.info(f"[AutonomousMode] Keyboard/stuck screen detected: '{keyword}'")
                            return True

                    # Also detect if OCR only shows single characters (keyboard keys)
                    # If most texts are 1-2 chars and include numbers, it's likely a keyboard
                    if len(texts) >= 4:
                        short_texts = [t for t in texts if len(str(t)) <= 2]
                        if len(short_texts) >= len(texts) * 0.6:  # 60%+ are short
                            has_numbers = any(c.isdigit() for t in texts for c in str(t))
                            if has_numbers:
                                logger.info("[AutonomousMode] Keyboard detected via character pattern")
                                return True

            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Keyboard screen check failed: {e}")
            return False

    def _escape_stuck_state(self) -> bool:
        """Attempt to escape a stuck state by pressing Back and navigating.

        Returns True if escape was attempted, False if not needed.
        """
        if not self._device_controller or not self._device_controller.is_connected():
            return False

        logger.info("[AutonomousMode] Attempting to escape stuck state with Back + navigation")
        self._log_event("Escaping stuck state - Back + navigate")

        # Press Back multiple times to exit dialogs/keyboards/sign-in flows
        for _ in range(4):
            self._device_controller.send_command("back")
            time.sleep(0.4)

        # After escaping, try to navigate to content
        # Press Down and Right to navigate away from sign-in options
        time.sleep(0.5)
        for _ in range(2):
            self._device_controller.send_command("down")
            time.sleep(0.3)
        self._device_controller.send_command("select")

        return True

    def _full_reset_to_youtube(self) -> bool:
        """Full reset: go to Home and relaunch YouTube from scratch.

        This is the nuclear option when we're completely stuck.
        """
        if not self._device_controller or not self._device_controller.is_connected():
            return False

        logger.warning("[AutonomousMode] Full reset - going Home and relaunching YouTube")
        self._log_event("FULL RESET - Home + relaunch YouTube")

        # Go to home screen
        self._device_controller.send_command("home")
        time.sleep(2)

        # Launch YouTube
        self._launch_youtube()
        time.sleep(3)

        # Reset stuck counters - let normal OCR/VLM detection handle the rest
        self._stuck_count = 0
        self._last_screen_state = None
        self._recovery_attempt_count = 0
        self._consecutive_static = 0
        self._persistent_static_count = 0
        self._menu_skip_count = 0

        return True

    def _is_screen_static(self) -> bool:
        """Check if screen is truly paused by combining frame analysis with audio state.

        A truly paused screen has:
        - Static frames (identical between captures)
        - No audio flowing (music stopped)

        A music stream with a static image has:
        - Static or near-static frames
        - Audio still flowing (music playing)

        Returns True only if screen is static AND audio is not flowing (truly paused).
        """
        if not self._frame_capture:
            return False

        try:
            frame1 = self._frame_capture.capture()
            if frame1 is None:
                return False

            time.sleep(3)

            frame2 = self._frame_capture.capture()
            if frame2 is None:
                return False

            hash1 = self._compute_frame_hash(frame1)
            hash2 = self._compute_frame_hash(frame2)

            # Hamming distance - low distance means nearly identical frames
            # Truly paused screens: hamming = 0 (identical JPEG captures)
            # Slow animations (lo-fi streams): hamming = 3-10 (subtle changes)
            # Active video: hamming = 15-40 (clear changes)
            hamming = bin(hash1 ^ hash2).count('1')
            frames_static = hamming < 3  # Only truly frozen screens

            if not frames_static:
                # Video is moving — reset the persistent-static counter.
                self._persistent_static_count = 0
                logger.info(f"[AutonomousMode] Frame change check: hamming={hamming}, video is changing")
                return False

            # Frames are static - check if audio is still playing
            # If the audio pipeline itself is unavailable (e.g. display disconnected
            # so alsasink can't open), the "no audio" signal is meaningless and we
            # must not treat static frames as paused — that would pause a live
            # music stream with static album art.
            # NOTE: HDMI-RX `audio_present` was tried as a workaround but proved
            # unreliable on Roku/YouTube — the source sends a continuous 48kHz
            # silence stream even when paused, so audio_present is always 1.
            if not self._is_audio_pipeline_available():
                self._persistent_static_count += 1
                if self._persistent_static_count >= self.PERSISTENT_STATIC_LIMIT:
                    # Frames have been truly frozen for ~5-7 minutes with no
                    # signal from the output audio pipeline. This is past the
                    # point where a real music stream would have updated *any*
                    # pixels (album-art fades, equalizer animations, etc.), so
                    # we treat it as stuck and let the caller take action.
                    logger.warning(
                        f"[AutonomousMode] Frame change check: hamming={hamming}, "
                        f"persistently static for {self._persistent_static_count} checks "
                        f"({self._persistent_static_count * 22}s approx) — escalating to STUCK"
                    )
                    self._persistent_static_count = 0
                    return True
                logger.info(
                    f"[AutonomousMode] Frame change check: hamming={hamming}, "
                    f"frames static but audio pipeline unavailable "
                    f"(persistent-static {self._persistent_static_count}/"
                    f"{self.PERSISTENT_STATIC_LIMIT}) — not treating as paused"
                )
                return False

            audio_flowing = self._is_audio_flowing()

            if audio_flowing:
                # Static image but audio playing = music stream (lo-fi, etc.) - NOT paused
                self._persistent_static_count = 0
                logger.info(f"[AutonomousMode] Frame change check: hamming={hamming}, "
                           f"frames static but audio flowing (music stream, not paused)")
                return False
            else:
                # Static image AND no audio = truly paused
                self._persistent_static_count = 0
                logger.info(f"[AutonomousMode] Frame change check: hamming={hamming}, "
                           f"frames static + no audio = PAUSED")
                return True

        except Exception as e:
            logger.debug(f"[AutonomousMode] Frame change check error: {e}")
            return False

    def _check_roku_active_app(self) -> bool:
        """For Roku devices, check if YouTube is the active app via ECP.

        This is more reliable than VLM because the Roku ECP definitively
        reports which app is running. VLM can confuse the Roku City screensaver
        with a playing video.

        Returns True if YouTube is running (or if not a Roku device).
        Returns False if Roku is on home/screensaver (YouTube needs relaunch).
        """
        if self._device_type != DEVICE_TYPE_ROKU:
            return True  # Not a Roku, skip this check

        if not hasattr(self._device_controller, 'get_active_app_id'):
            return True  # Controller doesn't support active app query

        try:
            # Check for screensaver overlay first — this can happen even when
            # YouTube is the "active" app (screensaver overlays it)
            if hasattr(self._device_controller, 'is_screensaver_active'):
                if self._device_controller.is_screensaver_active():
                    logger.info("[AutonomousMode] Roku screensaver active — dismissing")
                    self._device_controller.send_command('select')  # Wake from screensaver
                    self._log_event("Roku screensaver dismissed")
                    time.sleep(1)
                    return True  # Screensaver dismissed, YouTube should resume

            app_id = self._device_controller.get_active_app_id()
            if app_id is None:
                return True  # Query failed, don't interfere

            youtube_app_id = '837'  # Roku YouTube app ID
            if app_id == youtube_app_id:
                return True

            # Not YouTube — check what's running
            app_name = self._device_controller.get_active_app() or f"app_id={app_id}"
            logger.info(f"[AutonomousMode] Roku active app is '{app_name}' (not YouTube) — relaunching")
            self._log_event(f"Roku not on YouTube (active: {app_name}), relaunching")
            return False

        except Exception as e:
            logger.debug(f"[AutonomousMode] Roku active app check error: {e}")
            return True  # On error, don't interfere

    def _ensure_youtube_playing(self):
        """Use VLM to understand screen state and take appropriate action.

        For Roku: first checks active app via ECP (definitive) before VLM.
        VLM can confuse the Roku City screensaver with a playing video.

        Includes frame-change verification: if VLM says PLAYING but the screen
        is actually static (not changing), the video is likely paused. VLM is
        unreliable at distinguishing paused from playing states.
        Returns:
            True if an action was taken (we're on a navigation screen)
            False if no action needed (video is playing)
        """
        if not self._device_controller or not self._device_controller.is_connected():
            return False

        try:
            # For Roku: check active app via ECP before VLM
            # This catches the case where Roku exits YouTube to screensaver/home
            # and VLM misclassifies the animated screensaver as "PLAYING"
            if not self._check_roku_active_app():
                self._launch_youtube()
                self._consecutive_static = 0
                return True

            # OCR-based Roku home screen fallback (if ECP missed it).
            # AUDIO GUARD: the Roku home screen is essentially silent;
            # continuous audio means an app is playing video and the OCR
            # match came from in-app content (e.g. an ad CTA) — never
            # relaunch over playing video on OCR evidence alone. The ECP
            # active-app check above is authoritative and stays unguarded.
            if self._is_roku_home_screen():
                if self._is_audio_flowing():
                    # Roku home AUTOPLAYS promo/ad audio in its side pane, so
                    # "audio flowing" is not proof a video is playing when the
                    # home-tile text persists. Observed live 2026-07-12: Roku
                    # OS 15.2 reported home as <app id="native-ui"> which the
                    # digits-only ECP regex failed to parse (ECP stood down)
                    # and home promo audio vetoed this fallback for 50+ min.
                    # Escalate after enough consecutive vetoed matches —
                    # home-tile OCR persisting ~3 minutes IS the home screen.
                    self._roku_home_veto_streak += 1
                    if self._roku_home_veto_streak >= self._ROKU_HOME_VETO_ESCAPE_AT:
                        logger.warning(
                            f"[AutonomousMode] Roku-home OCR persisted through "
                            f"{self._roku_home_veto_streak} audio vetoes — relaunching anyway")
                        self._log_event("Roku home persisted despite audio - launching YouTube")
                        self._roku_home_veto_streak = 0
                        self._launch_youtube()
                        self._consecutive_static = 0
                        self._stuck_count = 0
                        return True
                    logger.info("[AutonomousMode] Roku-home OCR match vetoed: "
                                "audio flowing — video is playing")
                    self._log_event("Roku-home OCR vetoed: audio flowing")
                else:
                    logger.info("[AutonomousMode] Roku home detected via OCR - launching YouTube")
                    self._log_event("Roku home (OCR fallback) - launching YouTube")
                    self._roku_home_veto_streak = 0
                    self._launch_youtube()
                    self._consecutive_static = 0
                    self._stuck_count = 0
                    return True
            else:
                self._roku_home_veto_streak = 0

            # STUCK DETECTION: Check for keyboard/sign-in screens we can't automate
            # This must come BEFORE other checks to escape stuck states quickly
            if self._is_keyboard_stuck_screen():
                # AUDIO GUARD (per the ARCHITECTURAL INVARIANT below):
                # real keyboard/sign-in screens are static and SILENT.
                # Observed live 2026-07-02 03:31: the character-pattern
                # heuristic (many short OCR fragments + digits) false-
                # positived on a playing video frame and the 4-Back
                # escape exited YouTube to the Roku home screen. Audio
                # flowing means a video is playing — never escape.
                # Uses the 20s-recent window so ad-pod silence gaps don't
                # count as "not playing" (see the dismiss guard).
                if self._audio_recently_flowing():
                    logger.info("[AutonomousMode] Keyboard-screen verdict vetoed: "
                                "audio flowing — video is playing")
                    self._log_event("Keyboard verdict vetoed: audio flowing")
                    self._stuck_count = 0
                    return False

                self._stuck_count += 1
                logger.warning(f"[AutonomousMode] Keyboard/stuck screen detected ({self._stuck_count}/{self._STUCK_THRESHOLD})")

                if self._stuck_count >= self._STUCK_THRESHOLD:
                    # We've been stuck too long - full reset
                    self._full_reset_to_youtube()
                    return True
                else:
                    # Try to escape with Back presses
                    self._escape_stuck_state()
                    return True

            # YouTube TV upsell prompt — a dead end (every option leads to
            # sign-up/payment flows). Press Back to dismiss; escalate to a
            # full YouTube relaunch if it keeps reappearing.
            if self._is_youtube_tv_prompt():
                if self._last_screen_state == 'yttv_prompt':
                    self._stuck_count += 1
                else:
                    self._stuck_count = 1
                    self._last_screen_state = 'yttv_prompt'

                if self._stuck_count >= self._STUCK_THRESHOLD:
                    logger.warning(f"[AutonomousMode] Stuck on YouTube TV prompt "
                                   f"({self._stuck_count}x) - full reset")
                    self._full_reset_to_youtube()
                    return True

                logger.info("[AutonomousMode] YouTube TV prompt detected - pressing Back to dismiss")
                self._log_event("YouTube TV prompt detected - pressing Back")
                self._device_controller.send_command("back")
                time.sleep(0.5)
                self._consecutive_static = 0
                return True

            # OCR-based screen detection (VLM often misclassifies static screens as PLAYING)

            # Check for survey dialog - need to skip it
            if self._is_survey_screen():
                logger.info("[AutonomousMode] Survey dialog detected - skipping")
                self._log_event("Survey dialog detected - skipping")
                # Navigate right to "Skip survey" button and press select
                for _ in range(3):  # Move right a few times to find skip button
                    self._device_controller.send_command("right")
                    time.sleep(0.3)
                self._device_controller.send_command("select")
                self._consecutive_static = 0
                return True

            # Check for signed-out screen OR login screen - need to select account or watch as guest
            is_signed_out = self._is_signed_out_screen()
            is_login = self._is_youtube_login_screen()

            if is_signed_out or is_login:
                # Track consecutive auth screen detections
                if self._last_screen_state in ('signed_out', 'login'):
                    self._stuck_count += 1
                else:
                    self._stuck_count = 0
                    self._last_screen_state = 'signed_out' if is_signed_out else 'login'

                if self._stuck_count >= self._STUCK_THRESHOLD:
                    logger.warning(f"[AutonomousMode] Stuck on auth screen ({self._stuck_count}x) - full reset")
                    self._full_reset_to_youtube()
                    return True

                # Get OCR text to understand the screen layout
                combined = ''
                if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                    texts = self._ad_blocker.last_ocr_texts
                    if texts:
                        combined = ' '.join(str(t) for t in texts).lower()

                # Count accounts visible (@ symbols indicate account names)
                account_count = combined.count('@')
                logger.info(f"[AutonomousMode] Auth screen - {account_count} accounts visible, navigating to guest option")
                self._log_event(f"Auth screen - {account_count} accounts, selecting guest")

                # Navigate DOWN past all accounts to reach "Watch as guest" / guest mode option
                # Layout: [Sign in] → [Add account] → [Account 1] → [Account 2] → [Guest] → [Sign in bottom]
                # Need to go past accounts (2-3) plus header items (1-2), so 5-6 downs should reach guest
                down_count = max(4, account_count + 3)  # At least 4, more if many accounts
                for _ in range(down_count):
                    self._device_controller.send_command("down")
                    time.sleep(0.2)

                # Now go UP once in case we overshot past guest to "Add account" or "Sign in" at bottom
                self._device_controller.send_command("up")
                time.sleep(0.2)
                self._device_controller.send_command("select")

                self._consecutive_static = 0
                return True

            # Check if we're stuck in YouTube Shorts - exit back to home
            if self._is_youtube_shorts():
                if self._last_screen_state == 'shorts':
                    self._stuck_count += 1
                else:
                    self._stuck_count = 1
                    self._last_screen_state = 'shorts'

                if self._stuck_count >= self._STUCK_THRESHOLD:
                    # Back isn't getting us out (or home keeps feeding us
                    # back into the Shorts row) - full reset
                    logger.warning(f"[AutonomousMode] Stuck in Shorts "
                                   f"({self._stuck_count}x) - full reset")
                    self._full_reset_to_youtube()
                    return True

                logger.info("[AutonomousMode] YouTube Shorts detected - exiting to find full video")
                self._log_event("YouTube Shorts detected - pressing Back")
                self._device_controller.send_command("back")
                time.sleep(0.5)
                return True

            # Check for YouTube home screen - need to select a video
            if self._is_youtube_home_screen():
                # Reaching home screen is progress - reset stuck counters
                self._stuck_count = 0
                self._last_screen_state = 'home'

                logger.info("[AutonomousMode] YouTube home screen detected via OCR - selecting a video")
                self._log_event("YouTube home screen detected - selecting a video")

                # Vary navigation to find different videos (some require sign-in).
                # If the page shows a live-stream indicator, push the selection
                # further down to skip past the live row entirely (live streams
                # are unsuitable for autonomous ad-training, see LIVE_CONTENT_KEYWORDS).
                # Same treatment when 'shorts' OCR keyword is present — YouTube
                # TV's Shorts row sits near the top of home, so the default
                # 3-6 downs lands on it ~half the time; the Shorts-exit
                # recovery succeeds but wastes a screensaver/relaunch cycle.
                # Pattern observed 2026-06-06: 5/7/6 Shorts exits in 3
                # consecutive 15-min windows. Push past Shorts row when its
                # keyword is visible.
                import random
                ocr_combined = ''
                try:
                    if self._ad_blocker and hasattr(self._ad_blocker, 'last_ocr_texts'):
                        ocr_combined = ' '.join(
                            str(t) for t in (self._ad_blocker.last_ocr_texts or [])
                        ).lower()
                except Exception:
                    pass
                if self._has_live_content_indicator():
                    down_count = random.randint(6, 9)  # Push past live row
                    self._log_event("Live content detected — skipping past live row")
                elif 'shorts' in ocr_combined:
                    down_count = random.randint(7, 10)  # Push past Shorts row
                    self._log_event("Shorts row detected — skipping past it")
                else:
                    # Default bumped from (3,6) to (5,8) on 2026-06-06: the
                    # narrow `'shorts' in ocr_combined` check above only
                    # caught a fraction of Shorts hits because the home-
                    # screen-detection frame typically OCRs `'recommended'`
                    # or `'search'` without `'shorts'` even when the Shorts
                    # row IS on the page (below the focused tile). YouTube
                    # TV layout puts Shorts in row 1-2 consistently, so a
                    # higher floor avoids landing on it. 5-8 stays below
                    # the (6,9) live-row push and well below (7,10) Shorts-
                    # explicit push, so behavioral overlap is intentional.
                    down_count = random.randint(5, 8)
                right_first = random.choice([True, False])  # Sometimes skip the right press

                if right_first:
                    self._device_controller.send_command("right")
                    time.sleep(0.3)
                for _ in range(down_count):
                    self._device_controller.send_command("down")
                    time.sleep(0.3)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1
                self._consecutive_static = 0
                return True

            # 30-second no-audio timeout recovery
            # If we've been without audio for 30+ seconds, something is stuck
            # BUT skip this check if display is disconnected (audio pipeline isn't running)
            display_ok = True
            if self._ad_blocker and hasattr(self._ad_blocker, 'display_connected'):
                display_ok = self._ad_blocker.display_connected
            elif self._ad_blocker and hasattr(self._ad_blocker, 'video_ok'):
                display_ok = self._ad_blocker.video_ok

            if not display_ok:
                # Display disconnected - audio pipeline isn't running, skip audio recovery
                logger.debug("[AutonomousMode] Display disconnected - skipping audio recovery check")
                self._no_audio_start_time = None
                self._recovery_attempt_count = 0
                # Still do VLM check below for screen state
            else:
                audio_flowing = self._is_audio_flowing()
                current_time = time.time()

                if audio_flowing:
                    # Audio is working, reset the timer and recovery count
                    self._no_audio_start_time = None
                    self._last_successful_audio_time = current_time
                    if self._recovery_attempt_count > 0:
                        logger.info(f"[AutonomousMode] Audio recovered after {self._recovery_attempt_count} attempts")
                        self._recovery_attempt_count = 0
                else:
                    # No audio - track how long
                    if self._no_audio_start_time is None:
                        self._no_audio_start_time = current_time
                        logger.debug("[AutonomousMode] No audio detected, starting timer")
                    else:
                        no_audio_duration = current_time - self._no_audio_start_time
                        # Check if we've exceeded the timeout and not in cooldown
                        in_cooldown = (self._last_recovery_time is not None and
                                       current_time - self._last_recovery_time < self._RECOVERY_COOLDOWN)

                        if no_audio_duration >= self._NO_AUDIO_TIMEOUT and not in_cooldown:
                            self._recovery_attempt_count += 1
                            strategy = self._get_recovery_strategy(self._recovery_attempt_count)
                            logger.warning(f"[AutonomousMode] No audio for {no_audio_duration:.1f}s - "
                                          f"recovery attempt #{self._recovery_attempt_count} ({strategy})")
                            self._log_event(f"No audio {no_audio_duration:.0f}s - attempt #{self._recovery_attempt_count} ({strategy})")

                            self._execute_recovery_strategy(strategy)

                            # Reset timers
                            self._no_audio_start_time = None
                            self._last_recovery_time = current_time
                            self._consecutive_static = 0
                            logger.info(f"[AutonomousMode] Recovery strategy '{strategy}' completed")
                            return True

            # Use VLM to understand what's on screen
            screen_desc = self._query_screen()
            action = self._determine_action(screen_desc)

            if action == "none":
                # VLM says PLAYING - verify with frame change detection
                if self._is_screen_static():
                    self._consecutive_static += 1
                    logger.info(f"[AutonomousMode] VLM says PLAYING but screen is static "
                               f"({self._consecutive_static}/{self._STATIC_PAUSE_THRESHOLD})")

                    if self._consecutive_static >= self._STATIC_PAUSE_THRESHOLD:
                        # Static for 2 checks - try play_pause first (most common: paused video)
                        logger.info("[AutonomousMode] Static screen detected - sending play_pause")
                        self._device_controller.send_command("play_pause")
                        self._log_event("Static screen: sent play_pause")
                        # Don't reset counter - if still static next check, we'll escalate

                    if self._consecutive_static >= 4:
                        # Still static after play_pause didn't help - must be home/end screen
                        # Try selecting a video
                        logger.info("[AutonomousMode] play_pause didn't help - selecting a video")
                        self._device_controller.send_command("down")
                        time.sleep(0.5)
                        self._device_controller.send_command("select")
                        self._log_event("Escalated: selected video (play_pause failed)")
                        self.stats.videos_played += 1
                        self._consecutive_static = 0

                    # Action taken due to static screen
                    return True
                else:
                    # Screen is changing - truly playing! Reset all stuck counters.
                    self._consecutive_static = 0
                    self._stuck_count = 0
                    self._menu_skip_count = 0
                    self._last_screen_state = 'playing'
                    logger.debug("[AutonomousMode] Screen looks good, video is playing")
                    self._check_music_drift()
                    return False

            # Taking an action - reset static counter
            self._consecutive_static = 0

            # Snapshot overlay-veto count: any non-veto action resets it
            # (handled by `self._overlay_veto_count = 0` below). The veto
            # path itself re-installs prev+1 so consecutive vetoes still
            # escalate as intended.
            _prev_overlay_veto = self._overlay_veto_count
            self._overlay_veto_count = 0

            # Same snapshot pattern for the MENU-skip watchdog: any real
            # action below resets it; the skip path re-installs prev+1.
            _prev_menu_skip = self._menu_skip_count
            self._menu_skip_count = 0

            # ARCHITECTURAL INVARIANT (see "Autonomous Mode VLM-
            # Misclassification Traps" in CLAUDE.md Known Issues):
            # any action below that interrupts playback (down+select,
            # back, play_pause, launch) MUST consult an authoritative
            # playback signal BEFORE acting. The hierarchy is:
            #   1. _is_audio_flowing() — HDMI-RX audio is ground truth
            #      when HDMI-TX is connected.
            #   2. _is_video_player_overlay() — disambiguates overlay-
            #      on-video from real menu when audio is unavailable.
            # VLM verdicts alone are NOT sufficient. The screen-state
            # classifier (query_image) regularly misclassifies playing
            # videos with overlay UI as MENU, and paused videos with
            # overlay as MENU as well. Acting on the verdict alone
            # produces the three production traps documented in
            # CLAUDE.md (Sign-in trap, exit-paused-video, audio-blind
            # interruption). Future actions added below must follow the
            # same pattern.
            logger.info(f"[AutonomousMode] Action needed: {action} (screen: {screen_desc})")
            self._log_event(f"VLM action: {action}")

            if action == "play":
                # Video is paused per VLM - send play_pause to resume.
                # AUDIO GUARD: if HDMI-RX is receiving audio, the video
                # is genuinely playing and VLM misclassified. Skip.
                # (Same precedent as _is_screen_static's
                # PLAYING-static-no-audio branch.)
                if self._is_audio_flowing():
                    logger.info("[AutonomousMode] PAUSED verdict vetoed: "
                                "audio flowing — video is actually playing")
                    self._log_event("PAUSED vetoed: audio flowing")
                    self._last_screen_state = 'playing'
                    return False
                self._device_controller.send_command("play_pause")
                logger.info("[AutonomousMode] Sent play_pause command (video was paused)")

            elif action == "dismiss":
                # Dismiss overlays/banners (e.g. YouTube's persistent
                # "Sign in to subscribe" banner that overlays the player).
                # Use BACK rather than select+play_pause: select can confirm
                # an unwanted button (e.g. the Sign in button itself), and
                # play_pause toggles the player — which previously paused
                # any video that happened to be playing under the banner.
                # If the dialog actually paused the video (e.g. "Are you
                # still watching?"), the next pause-detection cycle will
                # send play_pause via the "play" action.
                #
                # AUDIO GUARD (the LAST interrupting action to get one):
                # observed live 2026-07-02 04:36 + 05:07 — audio was
                # flowing 30-60s before each DIALOG verdict; VLM had
                # misread a playing-video frame as DIALOG and the `back`
                # press EXITED the video, costing a ~3-min watchdog
                # recovery each time. Real blocking dialogs ("Are you
                # still watching?") pause playback → no audio → still
                # dismissed. A banner over a playing video merely lingers
                # (video keeps playing underneath) — acceptable.
                # RECENCY (added same day, 17:21 incident): uses the
                # 20s-recent window, not the instantaneous check — a
                # DIALOG misfire inside a silent gap BETWEEN pre-roll ads
                # slipped past the instant guard and back-exited the
                # just-started video. Real dialogs sit silent for minutes,
                # so they're still dismissed one cycle after the window.
                if self._audio_recently_flowing():
                    logger.info("[AutonomousMode] DIALOG verdict vetoed: "
                                "audio flowing — video is playing")
                    self._log_event("DIALOG vetoed: audio flowing")
                    self._last_screen_state = 'playing'
                    return False
                self._device_controller.send_command("back")
                logger.info("[AutonomousMode] Dismissed dialog with back")

            elif action == "select":
                # On home/menu screen - navigate to a video.
                # AUDIO GUARD: if HDMI-RX is receiving audio from the
                # streaming device, the video is genuinely playing. Never
                # interrupt — VLM misclassified the frame (lo-fi music,
                # talking-head, slow scene, player overlay during
                # playback all confuse it). Audio is authoritative when
                # it's available.
                if self._is_audio_flowing():
                    logger.info("[AutonomousMode] MENU vetoed: audio "
                                "flowing — video is playing")
                    self._log_event("MENU vetoed: audio flowing")
                    self._overlay_veto_count = 0
                    self._last_screen_state = 'playing'
                    return False

                # No audio + VLM says MENU. Per the user's diagnostic
                # ("audio not playing and being on Menu is a HUGE clue
                # we are paused", 2026-05-23): this is most likely a
                # PAUSED video showing its player overlay. Send
                # play_pause — universally safe:
                #   - paused video → resumes (the case we want to handle)
                #   - playing-but-silent video → pauses (recoverable next
                #     iteration; vanishingly rare in practice — VLM
                #     virtually never says MENU on a playing-no-overlay
                #     video, and a real such case will get re-toggled)
                #   - real menu → no harm (play_pause does nothing on a
                #     menu; the next iteration will retry select)
                # play_pause does NOT navigate UI, so the Sign-in trap
                # the overlay-veto was built to prevent is structurally
                # impossible from this action. The overlay-veto `back`
                # path was removed because `back` EXITS the paused
                # video on Roku/YouTube TV — the opposite of resume.
                if self._is_video_player_overlay():
                    logger.info("[AutonomousMode] MENU + no audio + "
                                "overlay visible → likely paused; "
                                "sending play_pause to resume")
                    self._log_event(
                        "MENU + overlay (likely paused) → play_pause")
                    self._overlay_veto_count = 0
                    self._device_controller.send_command("play_pause")
                    return True

                # No overlay, no audio: VLM says MENU. Before pressing
                # down+select, require OCR-confirmation that we're actually
                # on a home/browse screen (HOME_SCREEN_KEYWORDS). LFM2's
                # screen classifier is markedly noisy on overlay-heavy
                # frames — a single MENU verdict without OCR home keywords
                # is most often the briefly-overlay-less middle of a
                # playing video, and down+select there navigates the
                # player UI to Sign-in or exits to home. Skipping the
                # action when home isn't OCR-confirmed lets the next
                # iteration's audio/overlay signals catch up, eliminating
                # the home-flap loop observed 2026-06-06.
                if not self._is_youtube_home_screen():
                    # WATCHDOG: this skip is correct for a transient
                    # overlay-less playing frame, but a screen NO detector
                    # recognizes (observed live 2026-07-02: the "Sign in
                    # to YouTube TV" activation screen) produces this skip
                    # every cycle forever — a silent stall. No audio + no
                    # overlay + no recognizable screen for N consecutive
                    # cycles (~3 min) means nothing is playing and nothing
                    # is progressing: escape with Back presses + content
                    # selection. A genuinely-playing quiet video never
                    # accumulates N — any audio blip or overlay resets us
                    # via the paths above.
                    self._menu_skip_count = _prev_menu_skip + 1
                    if self._menu_skip_count >= self._MENU_SKIP_ESCAPE_AT:
                        logger.warning(
                            f"[AutonomousMode] MENU skip-loop stuck "
                            f"({self._menu_skip_count} consecutive skips, "
                            f"no audio/overlay/home) — escaping")
                        self._log_event(
                            "MENU skip-loop stuck — escaping with Back")
                        self._menu_skip_count = 0
                        self._overlay_veto_count = 0
                        self._escape_stuck_state()
                        return True

                    logger.info(f"[AutonomousMode] MENU select dispatch "
                                f"skipped — OCR shows no home-screen "
                                f"keywords (likely overlay-less playing "
                                f"frame, not a real menu) "
                                f"({self._menu_skip_count}/"
                                f"{self._MENU_SKIP_ESCAPE_AT})")
                    self._log_event(
                        "MENU select skipped — home OCR not confirmed")
                    self._overlay_veto_count = 0
                    return False

                # OCR confirms home screen. Apply live-skip logic if
                # the visible row is a live-stream tile.
                if self._has_live_content_indicator():
                    for _ in range(5):  # Push past live row
                        self._device_controller.send_command("down")
                        time.sleep(0.3)
                    self._log_event("MENU + home + live → skipped past live row")
                else:
                    self._device_controller.send_command("down")
                    time.sleep(0.5)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1
                logger.info("[AutonomousMode] Selected video from menu")
                self._log_event("Selected video from menu")

            elif action == "launch":
                # Screensaver/sleep per VLM - wake up and launch YouTube.
                # GUARDS (per the ARCHITECTURAL INVARIANT above — this was
                # the one interrupting action without any): VLM regularly
                # misclassifies dark playing scenes as SCREENSAVER.
                # Observed live 2026-07-02: a video that was demonstrably
                # playing (audio-flowing vetoes 30s earlier, ECP reporting
                # YouTube active with no screensaver) drew a SCREENSAVER
                # verdict → _wake_device (power+home) + relaunch killed
                # playback, bounced through the Roku home screen, and
                # churned to a new video every ~2 min.
                #
                # AUDIO GUARD: real screensavers are silent; audio flowing
                # means a video is playing and VLM misread a dark frame.
                if self._is_audio_flowing():
                    logger.info("[AutonomousMode] SCREENSAVER verdict vetoed: "
                                "audio flowing — video is playing")
                    self._log_event("SCREENSAVER vetoed: audio flowing")
                    self._last_screen_state = 'playing'
                    return False

                # ROKU ECP GUARD: ECP is authoritative for screensaver
                # state. The active-app check at the top of this cycle
                # already dismissed a real screensaver overlay, so if ECP
                # still reports YouTube active with no screensaver, this
                # is a dark/quiet playing frame — don't wake+relaunch.
                if (self._device_type == DEVICE_TYPE_ROKU
                        and hasattr(self._device_controller, 'is_screensaver_active')
                        and hasattr(self._device_controller, 'get_active_app_id')):
                    try:
                        if (not self._device_controller.is_screensaver_active()
                                and self._device_controller.get_active_app_id() == '837'):
                            logger.info("[AutonomousMode] SCREENSAVER verdict vetoed: "
                                        "ECP reports YouTube active, no screensaver")
                            self._log_event("SCREENSAVER vetoed: ECP shows YouTube")
                            return False
                    except Exception as e:
                        logger.debug(f"[AutonomousMode] ECP screensaver re-check failed: {e}")

                self._wake_device()
                time.sleep(2)
                self._launch_youtube()
                time.sleep(2)
                self._device_controller.send_command("down")
                time.sleep(0.5)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1
                logger.info("[AutonomousMode] Woke up and launched YouTube")
                self._log_event("Woke up device and launched YouTube")

            elif action == "back":
                self._device_controller.send_command("back")
                time.sleep(1)
                self._device_controller.send_command("play")

            # VLM action taken - we're on a navigation screen
            return True

        except Exception as e:
            logger.error(f"[AutonomousMode] Error in ensure_youtube_playing: {e}")
            self.stats.errors += 1
            return True  # Error occurred, assume we need fast checks

    def _wake_device(self):
        """Wake up the device from screensaver/sleep."""
        try:
            if self._device_type == DEVICE_TYPE_ROKU:
                # Roku: power on or home button
                if hasattr(self._device_controller, 'send_command'):
                    # Try power first, then home as fallback
                    self._device_controller.send_command("power")
                    time.sleep(0.5)
                    self._device_controller.send_command("home")
            else:
                # Fire TV / Android TV: wakeup command
                if hasattr(self._device_controller, 'send_command'):
                    self._device_controller.send_command("wakeup")
        except Exception as e:
            logger.warning(f"[AutonomousMode] Wake device error: {e}")

    def _get_recovery_strategy(self, attempt: int) -> str:
        """Get recovery strategy based on attempt number.

        Escalates through increasingly aggressive strategies:
        1-2: Basic navigation (back + select video)
        3-4: Play/pause attempts
        5-6: Multiple navigation attempts
        7+: Full relaunch YouTube
        """
        if attempt <= 2:
            return "navigate_select"
        elif attempt <= 4:
            return "play_pause_navigate"
        elif attempt <= 6:
            return "deep_navigate"
        else:
            return "relaunch_youtube"

    def _execute_recovery_strategy(self, strategy: str):
        """Execute the specified recovery strategy."""
        try:
            if strategy == "navigate_select":
                # Basic: back, navigate down, select
                self._device_controller.send_command("back")
                time.sleep(1.5)
                self._device_controller.send_command("down")
                time.sleep(0.5)
                self._device_controller.send_command("down")
                time.sleep(0.5)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1

            elif strategy == "play_pause_navigate":
                # Try play_pause first, then navigate
                self._device_controller.send_command("play_pause")
                time.sleep(2)
                # If still no audio, navigate to a new video
                self._device_controller.send_command("back")
                time.sleep(1.5)
                for _ in range(3):
                    self._device_controller.send_command("down")
                    time.sleep(0.3)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1

            elif strategy == "deep_navigate":
                # Go back multiple times and try to find content
                for _ in range(2):
                    self._device_controller.send_command("back")
                    time.sleep(1)
                # Navigate around more
                for _ in range(4):
                    self._device_controller.send_command("down")
                    time.sleep(0.3)
                self._device_controller.send_command("right")
                time.sleep(0.3)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1

            elif strategy == "relaunch_youtube":
                # Nuclear option: go home and relaunch YouTube
                logger.info("[AutonomousMode] Executing full YouTube relaunch")
                self._log_event("Full YouTube relaunch (escalation)")

                if self._device_type == DEVICE_TYPE_ROKU:
                    self._device_controller.send_command("home")
                    time.sleep(2)
                    self._launch_youtube()
                    time.sleep(4)
                else:
                    self._device_controller.send_command("home")
                    time.sleep(2)
                    self._launch_youtube()
                    time.sleep(3)

                # Navigate to a video
                for _ in range(3):
                    self._device_controller.send_command("down")
                    time.sleep(0.3)
                self._device_controller.send_command("select")
                self.stats.videos_played += 1

                # Reset attempt count after relaunch
                self._recovery_attempt_count = 0

        except Exception as e:
            logger.error(f"[AutonomousMode] Recovery strategy '{strategy}' failed: {e}")
            self.stats.errors += 1

    def play_next_video(self):
        """Skip to next video in YouTube."""
        if not self._device_controller or not self._device_controller.is_connected():
            return False

        try:
            self._device_controller.send_command("right")
            time.sleep(0.3)
            self._device_controller.send_command("right")
            time.sleep(0.3)
            self._device_controller.send_command("select")

            self.stats.videos_played += 1
            return True

        except Exception as e:
            logger.error(f"[AutonomousMode] Failed to play next: {e}")
            return False

    def record_ad_detected(self):
        """Record that an ad was detected."""
        self.stats.ads_detected += 1
        self.stats.last_activity = datetime.now(ET)

    def record_ad_skipped(self):
        """Record that an ad was skipped."""
        self.stats.ads_skipped += 1
        self.stats.last_activity = datetime.now(ET)

    def _log_event(self, message: str):
        """Log event to autonomous mode log file."""
        try:
            timestamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
            entry = f"- [{timestamp}] {message}\n"

            with open(self._log_file, "a") as f:
                f.write(entry)

        except Exception as e:
            logger.error(f"[AutonomousMode] Failed to write log: {e}")

    def get_log_tail(self, lines: int = 50) -> str:
        """Get last N lines of autonomous mode log."""
        try:
            with open(self._log_file, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except FileNotFoundError:
            return "No autonomous mode logs yet."
        except Exception as e:
            return f"Error reading logs: {e}"

    def destroy(self):
        """Clean up resources without changing persisted settings."""
        self._running = False
        self._stop_event.set()
        if self._active:
            self._active = False
            self.stats.session_end = datetime.now(ET)
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
