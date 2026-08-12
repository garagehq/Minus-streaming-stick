"""
Ad Blocker Overlay for Minus.

Displays a blocking overlay when ads are detected on screen.
Uses ustreamer's native blocking mode for smooth 60fps overlays and animations.

Architecture:
- Simple GStreamer pipeline with queue element for smooth video display
- All overlay compositing done in ustreamer's MPP encoder (60fps preview!)
- Control via HTTP API to ustreamer's /blocking endpoints

Features:
- 60fps live preview window (vs ~4fps with GStreamer gdkpixbufoverlay)
- Smooth animations via rapid API updates
- Spanish vocabulary practice during ad blocks
- Pixelated background from pre-ad content
"""

import os
import threading
import time
import random
import logging
import urllib.request
import urllib.parse
import urllib.error
import json
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Import vocabulary from extracted module
from vocabulary import SPANISH_VOCABULARY, VOCABULARY_COMBINED
from facts import DID_YOU_KNOW
from config import MinusConfig
from drm import (
    get_color_format, set_color_format, is_connector_connected,
    check_hdmi_i2c_errors, COLOR_FORMAT_YCBCR420, COLOR_FORMAT_NAMES,
    probe_drm_output
)

# Set up logging
logger = logging.getLogger(__name__)



class DRMAdBlocker:
    """
    DRM-based ad blocker using ustreamer's native blocking mode.

    Uses a simple GStreamer pipeline for display with queue element for smooth playback.
    All overlay compositing (background, preview, text) done in ustreamer's MPP encoder.
    """

    def __init__(self, connector_id=215, plane_id=72, minus_instance=None, ustreamer_port=9090,
                 output_width=1920, output_height=1080, config=None):
        self.is_visible = False
        self.current_source = None
        self.connector_id = connector_id
        self.plane_id = plane_id
        self.ustreamer_port = ustreamer_port
        self.minus = minus_instance
        self.output_width = output_width or 1920
        self.output_height = output_height or 1080
        self._lock = threading.Lock()

        # GStreamer pipeline
        self.pipeline = None
        self.bus = None

        # Audio passthrough reference
        self.audio = None

        # Pipeline health tracking
        self._pipeline_errors = 0
        self._last_error_time = 0
        self._pipeline_restarting = False
        self._restart_lock = threading.Lock()

        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        self._fps_lock = threading.Lock()

        # Video buffer watchdog
        self._last_buffer_time = 0
        self._watchdog_thread = None
        self._stop_watchdog = threading.Event()
        self._watchdog_interval = 3.0
        self._stall_threshold = 10.0
        self._restart_count = 0
        self._last_restart_time = 0
        self._consecutive_failures = 0
        self._base_restart_delay = 1.0
        self._max_restart_delay = 30.0
        self._success_reset_time = 10.0

        # Text rotation
        self._rotation_thread = None
        self._stop_rotation = threading.Event()

        # Debug overlay
        self._debug_overlay_enabled = True
        self._debug_thread = None
        self._stop_debug = threading.Event()
        self._debug_interval = 2.0
        self._total_blocking_time = 0.0
        self._current_block_start = None
        self._total_ads_blocked = 0
        # Top-right OCR trigger snippet, e.g. `(Ad) 0:30 left`. Populated when
        # OCR fires the block; rendered only while debug overlay is enabled.
        # Skipped/cleared for VLM-only blocks. OCR reads /snapshot/raw which
        # excludes the blocking composite, so this can't recurse.
        self._ocr_trigger_text = ""

        # Preview settings - use actual capture resolution for positioning
        self._preview_enabled = True
        # When True, the corner preview is desaturated during blocking so the
        # ad looks less appealing than the Spanish overlay. Toggled via the
        # web UI (`greyscale_preview` in ~/.minus_system_settings.json).
        self._preview_grayscale = True

        # Ad-remaining countdown state for the stats progress bar. Set from
        # minus.py when OCR reads an "Ad 0:NN" timer. Decays client-side
        # between OCR samples so the bar moves even at sub-second resolution.
        self._ad_seconds_remaining = None
        self._ad_seconds_anchor = 0.0  # time.time() when we received the value
        self._ad_seconds_peak = None   # Largest value seen this ad (for bar %)

        # Replacement-mode lock-in: at the start of each ad block we roll once
        # for a content kind (vocab / fact / haiku) and stick with it for the
        # whole break, plus a cooldown afterwards. Prevents flip-flopping
        # between styles mid-ad, which felt visually chaotic in testing.
        self._locked_content_kind = None
        self._content_kind_lock_until = 0.0
        self.CONTENT_KIND_COOLDOWN_SECONDS = 30.0

        # Pixelated pre-ad background — heavy pixelation (20x downscale) plus
        # 60% darken so the previous content reads as "where I was" without
        # competing with the Spanish overlay for attention. Kept fully offline
        # — JPEG comes from the local ustreamer /snapshot poll, not any remote
        # source. Can be turned off if glitches resurface.
        self._pixelated_background_enabled = True
        self._frame_width, self._frame_height = self._detect_frame_resolution()
        self._preview_w = int(self._frame_width * 0.20)
        self._preview_h = int(self._frame_height * 0.20)
        self._preview_padding = int(self._frame_height * 0.02)

        # Skip status
        self._skip_available = False
        self._skip_text = None

        # Time saved tracking
        self._total_time_saved = 0.0

        # Animation settings (use config values if provided)
        self._config = config or MinusConfig()
        self._animation_thread = None
        self._stop_animation = threading.Event()
        self._animation_duration_start = self._config.animation_start_duration
        self._animation_duration_end = self._config.animation_end_duration
        self._animating = False
        self._animation_direction = None
        self._animation_source = None
        # Animation can be disabled to prevent glitches (rapid API calls cause stream hiccups)
        self._animation_enabled = True  # Re-enabled with 10fps to reduce HTTP calls

        # Text background box opacity (0=transparent, 255=opaque)
        # Default was 180, increased for better readability
        self._box_alpha = 220

        # Text color in YUV (white - clean and readable, doesn't distract from vocabulary)
        # White: Y=235, U=128, V=128
        self._text_y = 235
        self._text_u = 128
        self._text_v = 128

        # Color settings persistence
        self._color_settings_file = Path.home() / '.minus_color_settings.json'
        self._saved_color_settings = self._load_color_settings()

        # Current vocabulary word tracking
        self._current_vocab = None  # (spanish, pronunciation, english, example)

        # Test mode
        self._test_blocking_until = 0

        # Snapshot buffer
        self._snapshot_buffer = deque(maxlen=3)
        self._snapshot_buffer_thread = None
        self._stop_snapshot_buffer = threading.Event()
        self._snapshot_interval = 2.0

        # Adaptive bandwidth fallback for problematic HDMI cables
        # Uses i2c error detection: when HDMI signal fails at high bandwidth,
        # the dwhdmi driver floods dmesg with "i2c read err!" messages.
        # This is more reliable than FPS detection since the pipeline can report
        # frames flowing even when the TV shows "No Signal".
        self._bandwidth_fallback_attempted = False
        self._bandwidth_fallback_applied = False  # True if currently running with fallback
        self._i2c_error_check_interval = 3.0  # How often to check for i2c errors
        self._last_i2c_error_check = 0.0
        self._i2c_error_threshold = 10  # Number of errors indicating failure
        self._i2c_error_window = 5.0  # Time window for error counting

        # Initialize GStreamer
        Gst.init(None)
        self._init_pipeline()
        self._start_snapshot_buffer()

    def _detect_frame_resolution(self):
        """Detect actual encoder output resolution from ustreamer snapshot.

        Note: This gets the actual JPEG output dimensions, not the source resolution.
        With --encode-scale=native, 4K NV12 input is downscaled to 1080p output.
        We need the output resolution for correct font scaling in blocking mode.
        """
        try:
            # Get actual JPEG dimensions by downloading snapshot
            # This is more reliable than /state which only shows source resolution
            url = f"http://localhost:{self.ustreamer_port}/snapshot"
            with urllib.request.urlopen(url, timeout=3.0) as response:
                jpeg_data = response.read()
                # Decode JPEG to get actual dimensions
                img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # Grayscale is faster
                if img is not None:
                    height, width = img.shape[:2]
                    logger.info(f"[DRMAdBlocker] Detected encoder output resolution: {width}x{height}")
                    return width, height
                else:
                    logger.warning("[DRMAdBlocker] Failed to decode snapshot, using 1920x1080")
                    return 1920, 1080
        except Exception as e:
            logger.warning(f"[DRMAdBlocker] Could not detect frame resolution: {e}, using 1920x1080")
            return 1920, 1080

    def _load_color_settings(self):
        """Load saved color settings from disk."""
        defaults = {
            'saturation': 1.0,
            'brightness': 0.0,
            'contrast': 1.0,
            'hue': 0.0
        }
        try:
            if self._color_settings_file.exists():
                with open(self._color_settings_file, 'r') as f:
                    saved = json.load(f)
                    # Merge with defaults to handle missing keys
                    for key in defaults:
                        if key in saved:
                            defaults[key] = saved[key]
                    logger.info(f"[DRMAdBlocker] Loaded color settings: sat={defaults['saturation']:.2f}")
                    return defaults
        except Exception as e:
            logger.warning(f"[DRMAdBlocker] Could not load color settings: {e}")
        return defaults

    def _save_color_settings(self, settings):
        """Save color settings to disk."""
        try:
            with open(self._color_settings_file, 'w') as f:
                json.dump(settings, f)
            logger.debug(f"[DRMAdBlocker] Saved color settings to {self._color_settings_file}")
        except Exception as e:
            logger.warning(f"[DRMAdBlocker] Could not save color settings: {e}")

    def _apply_saved_color_settings(self):
        """Apply saved color settings to the pipeline."""
        if self._saved_color_settings and self.pipeline:
            self.set_color_settings(
                saturation=self._saved_color_settings.get('saturation'),
                brightness=self._saved_color_settings.get('brightness'),
                contrast=self._saved_color_settings.get('contrast'),
                hue=self._saved_color_settings.get('hue')
            )

    def _blocking_api_call(self, endpoint, params=None, data=None, method='GET', timeout=0.1):
        """Make an API call to ustreamer blocking endpoint."""
        try:
            url = f"http://localhost:{self.ustreamer_port}{endpoint}"
            if params:
                url += '?' + urllib.parse.urlencode(params)

            if method == 'POST' and data:
                req = urllib.request.Request(url, data=data, method='POST')
                req.add_header('Content-Type', 'image/jpeg')
            else:
                req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            # Network errors - log at INFO since these indicate connectivity issues
            logger.info(f"[DRMAdBlocker] API connection error ({endpoint}): {e}")
            return None
        except Exception as e:
            # Other errors (JSON parse, etc.) - log at DEBUG
            logger.debug(f"[DRMAdBlocker] API call error ({endpoint}): {e}")
            return None

    def _color_settings_neutral(self, settings=None):
        """True when color settings are identity (no adjustment needed)."""
        s = settings or getattr(self, '_saved_color_settings', None) or {}
        return (abs(s.get('saturation', 1.0) - 1.0) < 1e-6
                and abs(s.get('brightness', 0.0)) < 1e-6
                and abs(s.get('contrast', 1.0) - 1.0) < 1e-6
                and abs(s.get('hue', 0.0)) < 1e-6)

    def _init_pipeline(self):
        """Initialize simple GStreamer display pipeline with queue element."""
        try:
            # Simple pipeline with small queue for low latency
            # - 3 buffer queue provides minimal latency while absorbing brief hiccups
            # - leaky=downstream drops oldest frames if queue fills (prevents latency buildup)
            #
            # videobalance is CPU per-pixel processing and the measured 4K
            # ceiling with non-neutral settings is ~56fps (vs 120fps decode
            # headroom without it). When the saved color settings are
            # NEUTRAL, omit the element entirely so the DMABuf path from
            # mppjpegdec to kmssink stays zero-copy. Crossing between
            # neutral and non-neutral at runtime restarts the pipeline
            # (see set_color_settings).
            self._pipeline_has_colorbalance = not self._color_settings_neutral()
            colorbalance_part = (
                f"videobalance saturation=1.0 brightness=0.0 contrast=1.0 hue=0.0 name=colorbalance ! "
                if self._pipeline_has_colorbalance else ""
            )
            pipeline_str = (
                f"souphttpsrc location=http://localhost:{self.ustreamer_port}/stream "
                f"is-live=true blocksize=524288 timeout=10 retries=-1 keep-alive=true ! "
                f"multipartdemux ! jpegparse ! mppjpegdec ! video/x-raw,format=NV12 ! "
                f"{colorbalance_part}"
                f"queue max-size-buffers=3 leaky=downstream name=videoqueue ! "
                f"identity name=fpsprobe ! "
                f"kmssink plane-id={self.plane_id} connector-id={self.connector_id} sync=false skip-vsync=true"
                # skip-vsync=true: the rockchip BSP's legacy drmModeSetPlane ioctl BLOCKS
                # until the flip latches at vblank (measured 16.67ms/call), so kmssink's
                # own internal vsync wait was a SECOND full vblank -> every frame took
                # 2 vblanks -> display pinned at exactly 30fps on a 60Hz mode. Frame
                # pacing/tearing is still handled by the blocking SetPlane itself.
            )

            logger.debug("[DRMAdBlocker] Creating pipeline with queue element...")
            self.pipeline = Gst.parse_launch(pipeline_str)

            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message::error', self._on_error)
            self.bus.connect('message::eos', self._on_eos)
            self.bus.connect('message::warning', self._on_warning)

            fpsprobe = self.pipeline.get_by_name('fpsprobe')
            if fpsprobe:
                srcpad = fpsprobe.get_static_pad('src')
                srcpad.add_probe(Gst.PadProbeType.BUFFER, self._fps_probe_callback, None)

            logger.info("[DRMAdBlocker] Pipeline created (ustreamer blocking mode)")

        except Exception as e:
            logger.error(f"[DRMAdBlocker] Failed to initialize GStreamer: {e}")
            self.pipeline = None

    def _fps_probe_callback(self, pad, info, user_data):
        current_time = time.time()
        self._last_buffer_time = current_time

        if self._consecutive_failures > 0:
            if current_time - self._last_restart_time > self._success_reset_time:
                self._consecutive_failures = 0

        with self._fps_lock:
            self._frame_count += 1
            elapsed = current_time - self._fps_start_time
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = current_time

        return Gst.PadProbeReturn.OK

    def get_fps(self):
        with self._fps_lock:
            return self._current_fps

    # =========================================================================
    # Color Balance Controls
    # =========================================================================

    def get_color_settings(self):
        """Get current color balance settings.

        Returns:
            dict with saturation, brightness, contrast, hue values
        """
        defaults = {
            'saturation': 1.0,
            'brightness': 0.0,
            'contrast': 1.0,
            'hue': 0.0
        }

        saved = getattr(self, '_saved_color_settings', None)
        if not self.pipeline:
            return dict(saved) if saved else defaults

        colorbalance = self.pipeline.get_by_name('colorbalance')
        if not colorbalance:
            # Element omitted (neutral settings, zero-copy pipeline)
            return dict(saved) if saved else defaults

        return {
            'saturation': colorbalance.get_property('saturation'),
            'brightness': colorbalance.get_property('brightness'),
            'contrast': colorbalance.get_property('contrast'),
            'hue': colorbalance.get_property('hue')
        }

    def set_color_settings(self, saturation=None, brightness=None, contrast=None, hue=None):
        """Set color balance settings dynamically.

        Args:
            saturation: 0.0-2.0 (default 1.0, higher = more saturated)
            brightness: -1.0 to 1.0 (default 0.0)
            contrast: 0.0-2.0 (default 1.0)
            hue: -1.0 to 1.0 (default 0.0)

        Returns:
            dict with success status and current values
        """
        if not self.pipeline:
            return {'success': False, 'error': 'Pipeline not running'}

        colorbalance = self.pipeline.get_by_name('colorbalance')
        if not colorbalance:
            # The pipeline was built without videobalance because settings
            # were neutral (zero-copy fast path). If the new settings are
            # still neutral there is nothing to do; otherwise persist them
            # and rebuild the pipeline so the element gets inserted and
            # _apply_saved_color_settings picks the values up.
            merged = (getattr(self, '_saved_color_settings', None) or {
                'saturation': 1.0, 'brightness': 0.0, 'contrast': 1.0, 'hue': 0.0}).copy()
            if saturation is not None:
                merged['saturation'] = max(0.0, min(2.0, float(saturation)))
            if brightness is not None:
                merged['brightness'] = max(-1.0, min(1.0, float(brightness)))
            if contrast is not None:
                merged['contrast'] = max(0.0, min(2.0, float(contrast)))
            if hue is not None:
                merged['hue'] = max(-1.0, min(1.0, float(hue)))

            self._saved_color_settings = merged.copy()
            self._save_color_settings(merged)

            if self._color_settings_neutral(merged):
                return {'success': True, **merged}

            logger.info("[DRMAdBlocker] Color settings became non-neutral - "
                        "restarting pipeline to insert videobalance")
            self.restart()
            return {'success': True, **merged}

        try:
            if saturation is not None:
                saturation = max(0.0, min(2.0, float(saturation)))
                colorbalance.set_property('saturation', saturation)

            if brightness is not None:
                brightness = max(-1.0, min(1.0, float(brightness)))
                colorbalance.set_property('brightness', brightness)

            if contrast is not None:
                contrast = max(0.0, min(2.0, float(contrast)))
                colorbalance.set_property('contrast', contrast)

            if hue is not None:
                hue = max(-1.0, min(1.0, float(hue)))
                colorbalance.set_property('hue', hue)

            current = self.get_color_settings()
            logger.info(f"[DRMAdBlocker] Color settings updated: sat={current['saturation']:.2f} "
                       f"bright={current['brightness']:.2f} contrast={current['contrast']:.2f} "
                       f"hue={current['hue']:.2f}")

            # Persist settings for next restart
            self._saved_color_settings = current.copy()
            self._save_color_settings(current)

            return {'success': True, **current}

        except Exception as e:
            logger.error(f"[DRMAdBlocker] Error setting color: {e}")
            return {'success': False, 'error': str(e)}

    def start(self):
        # Stop any animations before starting normal pipeline
        self._stop_loading_animation()
        self._stop_no_signal_animation()

        # If we're in loading or no-signal mode, need to reinitialize the normal pipeline
        if self.current_source in ('loading', 'no_hdmi_device'):
            logger.info(f"[DRMAdBlocker] Transitioning from {self.current_source} to normal pipeline")
            # Stop and destroy the standalone pipeline
            if self.pipeline:
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error during pipeline cleanup: {e}")
                self.pipeline = None
            # Reinitialize the normal pipeline
            self._init_pipeline()

        if not self.pipeline:
            logger.error("[DRMAdBlocker] No pipeline to start")
            return False

        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("[DRMAdBlocker] Failed to start pipeline")
                return False

            logger.info("[DRMAdBlocker] Pipeline started")
            self._set_led_state('idle')
            self._start_watchdog()

            # Reset failure counters on fresh start (important after long HDMI outages)
            self._consecutive_failures = 0
            self._last_buffer_time = time.time()

            # Re-detect frame resolution now that ustreamer should be running
            self._update_frame_resolution()

            # Apply saved color settings (user's preferences persist across restarts)
            self._apply_saved_color_settings()

            # Clear loading state
            self.current_source = None
            self.is_visible = False

            return True

        except Exception as e:
            logger.error(f"[DRMAdBlocker] Failed to start pipeline: {e}")
            return False

    def _update_frame_resolution(self):
        """Update frame resolution and recalculate preview dimensions."""
        new_w, new_h = self._detect_frame_resolution()
        if new_w != self._frame_width or new_h != self._frame_height:
            self._frame_width = new_w
            self._frame_height = new_h
            self._preview_w = int(self._frame_width * 0.20)
            self._preview_h = int(self._frame_height * 0.20)
            self._preview_padding = int(self._frame_height * 0.02)
            logger.info(f"[DRMAdBlocker] Updated preview size to {self._preview_w}x{self._preview_h}")

    def _start_watchdog(self):
        self._stop_watchdog.clear()
        self._last_buffer_time = time.time()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="VideoWatchdog")
        self._watchdog_thread.start()

    def _stop_watchdog_thread(self):
        self._stop_watchdog.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

    def _force_hdmi_reinit(self):
        """Force HDMI PHY reinitialization via DPMS cycle.

        After TV restart/hotplug, the HDMI PHY may not reinitialize properly,
        causing the TV to show "No Signal" even though GStreamer reports success.
        A DPMS Off->On cycle forces the HDMI transmitter to reinitialize.

        This must be called AFTER the old pipeline is stopped (so DRM is free)
        but BEFORE the new pipeline is created.
        """
        try:
            # Get current connector ID
            drm_info = probe_drm_output()
            connector_id = drm_info.get('connector_id')
            if not connector_id:
                logger.debug("[DRMAdBlocker] No connector found for DPMS cycle, skipping")
                return

            logger.info(f"[DRMAdBlocker] Forcing HDMI reinit via DPMS cycle on connector {connector_id}")

            # DPMS Off (value 3)
            result = subprocess.run(
                ['modetest', '-M', 'rockchip', '-w', f'{connector_id}:DPMS:3'],
                capture_output=True, timeout=5
            )
            if result.returncode != 0:
                stderr = result.stderr.decode() if result.stderr else ''
                if 'Permission denied' in stderr:
                    logger.debug("[DRMAdBlocker] DPMS cycle skipped - DRM in use (expected on first start)")
                    return
                logger.warning(f"[DRMAdBlocker] DPMS Off failed: {stderr}")
                return

            time.sleep(0.3)

            # DPMS On (value 0)
            result = subprocess.run(
                ['modetest', '-M', 'rockchip', '-w', f'{connector_id}:DPMS:0'],
                capture_output=True, timeout=5
            )
            if result.returncode != 0:
                logger.warning(f"[DRMAdBlocker] DPMS On failed: {result.stderr.decode() if result.stderr else ''}")
                return

            time.sleep(0.3)
            logger.info("[DRMAdBlocker] DPMS cycle complete - HDMI PHY reinitialized")

        except subprocess.TimeoutExpired:
            logger.warning("[DRMAdBlocker] DPMS cycle timed out")
        except Exception as e:
            logger.warning(f"[DRMAdBlocker] DPMS cycle failed: {e}")

    def _watchdog_loop(self):
        while not self._stop_watchdog.is_set():
            self._stop_watchdog.wait(self._watchdog_interval)
            if self._stop_watchdog.is_set():
                break
            if self._pipeline_restarting:
                continue
            # Don't restart if we're intentionally in a special mode
            if self.current_source in ('no_hdmi_device', 'loading'):
                continue

            current_time = time.time()
            time_since_buffer = current_time - self._last_buffer_time if self._last_buffer_time > 0 else 0

            # Check for HDMI bandwidth issues using i2c error detection
            # When HDMI signal fails at high bandwidth, the dwhdmi driver floods
            # dmesg with "i2c read err!" messages. This is the ONLY reliable heuristic
            # because GStreamer FPS stays at 30 even when TV shows "No Signal".
            # We use a sustained error check - errors must persist for multiple checks.
            if (current_time - self._last_i2c_error_check >= self._i2c_error_check_interval and
                not self._bandwidth_fallback_attempted):
                self._last_i2c_error_check = current_time

                has_errors, error_count, errors_per_sec = check_hdmi_i2c_errors(
                    threshold=self._i2c_error_threshold,
                    window_seconds=self._i2c_error_window
                )

                if has_errors and is_connector_connected(self.connector_id):
                    # Track consecutive checks with errors
                    if not hasattr(self, '_i2c_error_consecutive'):
                        self._i2c_error_consecutive = 0
                    self._i2c_error_consecutive += 1

                    # Require 3 consecutive checks with errors (9+ seconds) to avoid
                    # false positives during mode switching/startup
                    if self._i2c_error_consecutive >= 3:
                        logger.warning(f"[DRMAdBlocker] HDMI signal failing: sustained i2c errors "
                                     f"({error_count} errors, {errors_per_sec:.1f}/s) for "
                                     f"{self._i2c_error_consecutive} checks - attempting bandwidth fallback")

                        if self._attempt_bandwidth_fallback():
                            # Fallback was applied, restart pipeline with new settings
                            self._i2c_error_consecutive = 0
                            self._restart_pipeline()
                            continue
                    else:
                        logger.info(f"[DRMAdBlocker] i2c errors detected ({error_count}), "
                                  f"check {self._i2c_error_consecutive}/3")
                else:
                    # Reset consecutive counter when no errors
                    if hasattr(self, '_i2c_error_consecutive'):
                        self._i2c_error_consecutive = 0

            # Original stall detection and restart logic
            if self._last_buffer_time > 0:
                if time_since_buffer > self._stall_threshold:
                    logger.warning(f"[DRMAdBlocker] Pipeline stalled ({time_since_buffer:.1f}s)")
                    self._restart_pipeline()

            if self.pipeline:
                try:
                    state_ret, state, pending = self.pipeline.get_state(0)
                    if state not in (Gst.State.PLAYING, Gst.State.PAUSED):
                        self._restart_pipeline()
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error checking pipeline state: {e}")

    def _restart_pipeline(self, hdmi_reconnect=False):
        with self._restart_lock:
            if self._pipeline_restarting:
                return
            self._pipeline_restarting = True

        try:
            self._restart_count += 1
            self._consecutive_failures += 1
            # Cap exponent at 10 to prevent overflow (2^10 = 1024, way past max_restart_delay anyway)
            exponent = min(self._consecutive_failures - 1, 10)
            delay = min(self._base_restart_delay * (2 ** exponent), self._max_restart_delay)
            logger.warning(f"[DRMAdBlocker] Restarting pipeline (attempt {self._restart_count}, delay {delay:.1f}s, hdmi_reconnect={hdmi_reconnect})")

            if self.pipeline:
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error during pipeline cleanup: {e}")
                self.pipeline = None

            # After 3+ consecutive failures, the MPP decoder may be stuck.
            # Force-restart ustreamer to reset MPP state.
            if self._consecutive_failures >= 3:
                logger.warning(f"[DRMAdBlocker] {self._consecutive_failures} consecutive failures - restarting ustreamer to reset MPP")
                try:
                    import subprocess
                    subprocess.run(['pkill', '-9', 'ustreamer'], capture_output=True, timeout=5)
                    time.sleep(2)
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error killing ustreamer: {e}")

            # For HDMI reconnect (TV power cycle), do DPMS cycle and re-probe DRM
            if hdmi_reconnect:
                time.sleep(0.3)  # Give DRM time to release
                self._force_hdmi_reinit()

                # Re-probe DRM to find currently connected HDMI output
                drm_info = probe_drm_output()
                if drm_info.get('connector_id'):
                    if drm_info['connector_id'] != self.connector_id:
                        logger.info(f"[DRMAdBlocker] Updating DRM output: connector {self.connector_id} -> {drm_info['connector_id']}")
                        self.connector_id = drm_info['connector_id']
                        self.plane_id = drm_info.get('plane_id', self.plane_id)

                    # Always restart audio on HDMI reconnect - TV power cycle disrupts audio
                    # even when the device doesn't change
                    if hasattr(self, 'audio') and self.audio:
                        audio_device = drm_info.get('audio_device', self.audio.playback_device)
                        logger.info(f"[DRMAdBlocker] Restarting audio for HDMI reconnect (device: {audio_device})")
                        self.audio.stop()
                        self.audio.playback_device = audio_device
                        self.audio.start()

            time.sleep(delay)
            self._init_pipeline()

            if self.pipeline:
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret != Gst.StateChangeReturn.FAILURE:
                    logger.info("[DRMAdBlocker] Pipeline restarted successfully")
                    self._last_buffer_time = time.time()
                    self._last_restart_time = time.time()
                    # Don't reset consecutive_failures here - let the buffer probe
                    # reset it after sustained flow. This ensures the ustreamer
                    # force-restart triggers if the pipeline keeps stalling.
        finally:
            self._pipeline_restarting = False

    def restart(self, hdmi_reconnect=False):
        logger.info(f"[DRMAdBlocker] External restart requested (hdmi_reconnect={hdmi_reconnect})")
        threading.Thread(target=self._restart_pipeline, args=(hdmi_reconnect,), daemon=True).start()

    def _set_led_state(self, state):
        """Push a state to the WS2812B status strip if hooked up. Wrapped
        because LED issues must never break ad blocking."""
        if self.minus is None:
            return
        try:
            self.minus._set_led_state(state)
        except Exception:
            pass

    def _attempt_bandwidth_fallback(self) -> bool:
        """
        Attempt to fix display by falling back to lower bandwidth color format.

        This handles the case where the display is connected (EDID readable, TV is on)
        but we're getting 0 FPS due to HDMI bandwidth/signal integrity issues.

        YCbCr 4:2:0 uses half the bandwidth of YCbCr 4:4:4, making 4K@60Hz work
        with cables that can't handle full 18 Gbps bandwidth.

        Returns:
            True if fallback was applied and should retry, False otherwise
        """
        if self._bandwidth_fallback_attempted:
            logger.debug("[DRMAdBlocker] Bandwidth fallback already attempted")
            return False

        # Check if connector is actually connected (TV is on, EDID readable)
        if not is_connector_connected(self.connector_id):
            logger.debug("[DRMAdBlocker] Connector not connected, bandwidth fallback not applicable")
            return False

        # Check current color format
        current_value, current_name = get_color_format(self.connector_id)
        logger.info(f"[DRMAdBlocker] Current color format: {current_name} (value={current_value})")

        # If already at lowest bandwidth, nothing more we can do
        if current_value == COLOR_FORMAT_YCBCR420:
            logger.info("[DRMAdBlocker] Already at YCbCr 4:2:0, no further fallback possible")
            self._bandwidth_fallback_attempted = True
            self._bandwidth_fallback_applied = True  # Mark as applied since we're running at 4:2:0
            return False

        # Apply bandwidth fallback by restarting the service
        # The GStreamer kmssink element holds DRM inside our process, making it
        # impossible to change color_format while running. Instead, we write a
        # marker file and restart the service - on startup, the color_format
        # will be set before any DRM-using processes start.
        logger.warning("[DRMAdBlocker] Display connected but signal failing - triggering service restart for bandwidth fallback")

        # Write marker file indicating fallback is needed
        marker_file = '/tmp/minus_bandwidth_fallback_needed'
        try:
            import subprocess
            with open(marker_file, 'w') as f:
                f.write(f"{self.connector_id}\n")
            logger.info(f"[DRMAdBlocker] Wrote fallback marker to {marker_file}")

            # Mark that we've attempted fallback (will be reset on restart anyway)
            self._bandwidth_fallback_attempted = True

            # Restart the service - this will fully release DRM
            logger.info("[DRMAdBlocker] Restarting minus service to apply bandwidth fallback...")
            subprocess.Popen(['sudo', 'systemctl', 'restart', 'minus'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Give the restart command time to take effect before we continue
            # (the process should be killed shortly after this)
            time.sleep(5)

            # If we're still here, something went wrong with restart
            logger.warning("[DRMAdBlocker] Service restart may have failed")
            return False

        except Exception as e:
            logger.warning(f"[DRMAdBlocker] Error triggering fallback restart: {e}")
            self._bandwidth_fallback_attempted = True
            return False

    def get_bandwidth_status(self) -> dict:
        """Get current bandwidth/color format status for API."""
        current_value, current_name = get_color_format(self.connector_id)
        has_i2c_errors, error_count, errors_per_sec = check_hdmi_i2c_errors(
            threshold=self._i2c_error_threshold,
            window_seconds=self._i2c_error_window
        )
        return {
            'color_format': current_name,
            'color_format_value': current_value,
            'bandwidth_fallback_applied': self._bandwidth_fallback_applied,
            'bandwidth_fallback_attempted': self._bandwidth_fallback_attempted,
            'i2c_errors_detected': has_i2c_errors,
            'i2c_error_count': error_count,
            'i2c_errors_per_second': round(errors_per_sec, 1),
        }

    def start_no_signal_mode(self, skip_dpms=False):
        """Start a standalone display for 'No Signal' message with DVD-style bouncing.

        This creates a simple pipeline using videotestsrc that doesn't depend on ustreamer.
        The text bounces around the screen like the classic DVD screensaver.

        Args:
            skip_dpms: Skip DPMS cycle on initial cold boot (default: False)
                      DPMS cycle is only needed after TV restart/hotplug, not on cold boot.
        """
        try:
            logger.debug("[DRMAdBlocker] Starting no-signal mode...")
            self._set_led_state('no_signal')

            # Stop the watchdog - we don't want it restarting the normal pipeline
            self._stop_watchdog_thread()

            # Stop any existing animations
            self._stop_loading_animation()
            self._stop_no_signal_animation()

            # Stop existing pipeline if any
            if self.pipeline:
                logger.debug("[DRMAdBlocker] Stopping existing pipeline for no-signal mode...")
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception as e:
                    logger.warning(f"[DRMAdBlocker] Error stopping pipeline: {e}")
                self.pipeline = None
                # Give DRM time to release resources
                time.sleep(0.3)

            # Force HDMI PHY reinitialization via DPMS cycle
            # This is needed after TV restart/hotplug to ensure HDMI output works
            # Skip on initial cold boot for faster startup
            if not skip_dpms:
                self._force_hdmi_reinit()

                # Dynamically re-probe DRM to find currently connected HDMI output
                # This handles cases where TV was connected after service started
                # Skip on cold boot since we already have the info from __init__
                drm_info = probe_drm_output()
                if drm_info.get('connector_id'):
                    if drm_info['connector_id'] != self.connector_id:
                        logger.info(f"[DRMAdBlocker] Updating DRM output: connector {self.connector_id} -> {drm_info['connector_id']}")
                        self.connector_id = drm_info['connector_id']
                        self.plane_id = drm_info.get('plane_id', self.plane_id)
                else:
                    # Early-return: attempting to build the pipeline without a
                    # connected DRM output is guaranteed to fail in kmssink, and
                    # each failure leaks ~0-1 FDs from bus signal watch teardown
                    # race. Health monitor retries this every 3-5s when HDMI-TX
                    # is disconnected, so the leaks compound. Bail before
                    # allocating any GStreamer objects.
                    logger.warning("[DRMAdBlocker] No connected HDMI output found for no-signal display — skipping pipeline creation")
                    return False

            # Create a standalone pipeline for no-signal display with positioned text
            # Uses valignment=position and halignment=position to enable xpos/ypos control
            no_signal_pipeline = (
                f"videotestsrc pattern=black ! "
                f"video/x-raw,width=1920,height=1080,framerate=30/1 ! "
                f"textoverlay name=no_signal_text text=\"[ NO SIGNAL ]\" "
                f"valignment=position halignment=position xpos=0.5 ypos=0.5 "
                f"font-desc=\"Sans Bold 24\" ! "
                f"videoconvert ! video/x-raw,format=NV12 ! "
                f"kmssink plane-id={self.plane_id} connector-id={self.connector_id} sync=false skip-vsync=true"
                # skip-vsync=true: the rockchip BSP's legacy drmModeSetPlane ioctl BLOCKS
                # until the flip latches at vblank (measured 16.67ms/call), so kmssink's
                # own internal vsync wait was a SECOND full vblank -> every frame took
                # 2 vblanks -> display pinned at exactly 30fps on a 60Hz mode. Frame
                # pacing/tearing is still handled by the blocking SetPlane itself.
            )

            logger.debug(f"[DRMAdBlocker] Creating no-signal pipeline (plane={self.plane_id}, connector={self.connector_id})...")
            self.pipeline = Gst.parse_launch(no_signal_pipeline)

            if not self.pipeline:
                logger.error("[DRMAdBlocker] Failed to parse no-signal pipeline")
                return False

            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message::error', self._on_error)

            # Get the textoverlay element for animation
            self._no_signal_textoverlay = self.pipeline.get_by_name('no_signal_text')

            logger.debug("[DRMAdBlocker] Setting no-signal pipeline to PLAYING...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                # Try to get the actual error from the bus
                error_msg = "unknown"
                if self.bus:
                    msg = self.bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR)
                    if msg:
                        err, debug = msg.parse_error()
                        error_msg = f"{err.message} (debug: {debug})"
                logger.error(f"[DRMAdBlocker] Failed to start no-signal pipeline: {error_msg}")
                logger.error(f"[DRMAdBlocker] Pipeline was: plane={self.plane_id}, connector={self.connector_id}")
                # Clean up failed pipeline - CRITICAL: must remove bus signal watch to avoid FD leak
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error during no-signal pipeline cleanup: {e}")
                self.pipeline = None
                return False

            self.is_visible = True
            self.current_source = 'no_hdmi_device'

            # Start the bouncing animation
            self._start_no_signal_animation()

            logger.info("[DRMAdBlocker] No-signal display started successfully")
            return True
        except Exception as e:
            logger.error(f"[DRMAdBlocker] Failed to start no-signal mode: {e}")
            import traceback
            traceback.print_exc()
            # Clean up on exception - avoid FD leaks
            try:
                if self.bus:
                    self.bus.remove_signal_watch()
                    self.bus = None
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None
            return False

    def _start_no_signal_animation(self):
        """Start the DVD-style bouncing animation thread."""
        self._stop_no_signal_anim = threading.Event()
        self._no_signal_anim_thread = threading.Thread(
            target=self._no_signal_animation_loop,
            daemon=True,
            name="NoSignalBounce"
        )
        self._no_signal_anim_thread.start()

    def _stop_no_signal_animation(self):
        """Stop the bouncing animation thread."""
        if hasattr(self, '_stop_no_signal_anim'):
            self._stop_no_signal_anim.set()
        if hasattr(self, '_no_signal_anim_thread') and self._no_signal_anim_thread:
            self._no_signal_anim_thread.join(timeout=1.0)
            self._no_signal_anim_thread = None
        self._no_signal_textoverlay = None

    def _no_signal_animation_loop(self):
        """Animate the NO SIGNAL text bouncing around like DVD screensaver."""
        import time

        # Position and velocity (0.0 to 1.0 range)
        x, y = 0.5, 0.5
        vx, vy = 0.008, 0.006  # Velocity per frame

        # Boundaries (leave margin for text size)
        min_x, max_x = 0.1, 0.9
        min_y, max_y = 0.1, 0.9

        # Corner hit celebration
        corner_hit_frames = 0
        spin_angle = 0

        while not self._stop_no_signal_anim.is_set():
            try:
                # Update position
                x += vx
                y += vy

                # Track if we hit edges
                hit_x = False
                hit_y = False

                # Bounce off edges
                if x <= min_x or x >= max_x:
                    vx = -vx
                    x = max(min_x, min(max_x, x))
                    hit_x = True
                if y <= min_y or y >= max_y:
                    vy = -vy
                    y = max(min_y, min(max_y, y))
                    hit_y = True

                # Corner hit! Start celebration spin
                if hit_x and hit_y:
                    corner_hit_frames = 30  # Celebrate for 30 frames (~1 second)
                    logger.info("[DRMAdBlocker] NO SIGNAL hit corner! 🎉")

                # Update textoverlay
                if self._no_signal_textoverlay:
                    self._no_signal_textoverlay.set_property('xpos', x)
                    self._no_signal_textoverlay.set_property('ypos', y)

                    # During corner celebration, cycle through spin text
                    if corner_hit_frames > 0:
                        spin_chars = ['*', '+', 'x', '+']
                        spin_idx = (30 - corner_hit_frames) % 4
                        spin_text = f"[{spin_chars[spin_idx]} NO SIGNAL {spin_chars[spin_idx]}]"
                        self._no_signal_textoverlay.set_property('text', spin_text)
                        corner_hit_frames -= 1
                    else:
                        self._no_signal_textoverlay.set_property('text', '[ NO SIGNAL ]')

                time.sleep(0.033)  # ~30fps animation
            except Exception as e:
                logger.debug(f"[DRMAdBlocker] No-signal animation error, stopping: {e}")
                break

    def start_loading_mode(self):
        """Start a standalone display for 'Loading' with animated ellipses.

        This creates a pipeline using videotestsrc that shows "Loading" with
        animated dots (0-4 dots, increasing then decreasing).
        """
        try:
            # Stop the watchdog - we don't want it restarting the normal pipeline
            self._stop_watchdog_thread()

            # Stop any existing animations
            self._stop_loading_animation()
            self._stop_no_signal_animation()

            # Stop existing pipeline if any
            if self.pipeline:
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error during pipeline cleanup: {e}")
                self.pipeline = None

            # Dynamically re-probe DRM to find currently connected HDMI output
            drm_info = probe_drm_output()
            if drm_info.get('connector_id'):
                if drm_info['connector_id'] != self.connector_id:
                    logger.info(f"[DRMAdBlocker] Updating DRM output: connector {self.connector_id} -> {drm_info['connector_id']}")
                    self.connector_id = drm_info['connector_id']
                    self.plane_id = drm_info.get('plane_id', self.plane_id)

            # Create a standalone pipeline for loading display
            # Uses videotestsrc with named textoverlay for animation
            loading_pipeline = (
                f"videotestsrc pattern=black ! "
                f"video/x-raw,width=1920,height=1080,framerate=30/1 ! "
                f"textoverlay name=loading_text text=\"[ INITIALIZING ]\" "
                f"valignment=center halignment=center font-desc=\"Sans Bold 24\" ! "
                f"videoconvert ! video/x-raw,format=NV12 ! "
                f"kmssink plane-id={self.plane_id} connector-id={self.connector_id} sync=false skip-vsync=true"
                # skip-vsync=true: the rockchip BSP's legacy drmModeSetPlane ioctl BLOCKS
                # until the flip latches at vblank (measured 16.67ms/call), so kmssink's
                # own internal vsync wait was a SECOND full vblank -> every frame took
                # 2 vblanks -> display pinned at exactly 30fps on a 60Hz mode. Frame
                # pacing/tearing is still handled by the blocking SetPlane itself.
            )

            logger.debug("[DRMAdBlocker] Creating loading pipeline...")
            self.pipeline = Gst.parse_launch(loading_pipeline)

            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message::error', self._on_error)

            # Get the textoverlay element for animation
            self._loading_textoverlay = self.pipeline.get_by_name('loading_text')

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("[DRMAdBlocker] Failed to start loading pipeline")
                # Clean up failed pipeline - CRITICAL: must remove bus signal watch to avoid FD leak
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                except Exception as e:
                    logger.debug(f"[DRMAdBlocker] Error during loading pipeline cleanup: {e}")
                self.pipeline = None
                return False

            self.is_visible = True
            self.current_source = 'loading'

            # Start the loading animation thread
            self._start_loading_animation()

            logger.info("[DRMAdBlocker] Loading display started")
            return True
        except Exception as e:
            logger.error(f"[DRMAdBlocker] Failed to start loading mode: {e}")
            # Clean up on exception - avoid FD leaks
            try:
                if self.bus:
                    self.bus.remove_signal_watch()
                    self.bus = None
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None
            return False

    def _start_loading_animation(self):
        """Start the loading dots animation thread."""
        self._stop_loading_anim = threading.Event()
        self._loading_anim_thread = threading.Thread(
            target=self._loading_animation_loop,
            daemon=True,
            name="LoadingAnimation"
        )
        self._loading_anim_thread.start()

    def _stop_loading_animation(self):
        """Stop the loading dots animation thread."""
        if hasattr(self, '_stop_loading_anim'):
            self._stop_loading_anim.set()
        if hasattr(self, '_loading_anim_thread') and self._loading_anim_thread:
            self._loading_anim_thread.join(timeout=1.0)
            self._loading_anim_thread = None
        self._loading_textoverlay = None

    def _loading_animation_loop(self):
        """Animate the loading text with ellipses (0-4 dots, increasing then decreasing)."""
        # Pattern: "", ".", "..", "...", "....", "...", "..", "."
        dot_counts = [0, 1, 2, 3, 4, 3, 2, 1]
        idx = 0
        interval = 0.3  # Update every 300ms

        while not self._stop_loading_anim.is_set():
            if hasattr(self, '_loading_textoverlay') and self._loading_textoverlay:
                dots = "." * dot_counts[idx]
                padding = " " * (4 - dot_counts[idx])  # Keep width consistent
                text = f"[ INITIALIZING{dots}{padding}]"
                try:
                    self._loading_textoverlay.set_property('text', text)
                except Exception as e:
                    # Pipeline may have been destroyed during shutdown
                    logger.debug(f"[DRMAdBlocker] Loading text update failed (pipeline destroyed?): {e}")

            idx = (idx + 1) % len(dot_counts)
            self._stop_loading_anim.wait(interval)

    def _on_error(self, bus, message):
        err, debug = message.parse_error()
        self._pipeline_errors += 1
        self._last_error_time = time.time()
        logger.error(f"[DRMAdBlocker] Pipeline error: {err.message}")
        error_msg = err.message.lower() if err.message else ""
        if any(kw in error_msg for kw in ['connection', 'refused', 'timeout', 'socket', 'http']):
            if not self.is_visible:
                threading.Thread(target=self._restart_pipeline, daemon=True).start()

    def _on_eos(self, bus, message):
        logger.warning("[DRMAdBlocker] Unexpected EOS")
        if not self.is_visible:
            threading.Thread(target=self._restart_pipeline, daemon=True).start()

    def _on_warning(self, bus, message):
        warn, debug = message.parse_warning()
        logger.warning(f"[DRMAdBlocker] Pipeline warning: {warn.message}")

    def get_pipeline_health(self):
        if not self.pipeline:
            return {'healthy': False, 'state': 'stopped', 'errors': self._pipeline_errors}
        state_ret, state, pending = self.pipeline.get_state(0)
        return {
            'healthy': state == Gst.State.PLAYING,
            'state': state.value_nick if state else 'unknown',
            'errors': self._pipeline_errors,
            'last_error': self._last_error_time
        }

    # Content-kind rotation weights. Vocab is the default workhorse but we
    # sprinkle facts in to keep the overlay from feeling monotonous.
    # When _locked_content_kind is set (per-ad-break lock-in), we bypass this.
    _CONTENT_KINDS = ('vocab', 'fact')
    _CONTENT_KIND_WEIGHTS = (0.7, 0.3)
    # When the user has enabled 'photos' replacement mode AND uploaded at
    # least one photo, each ad block has a one-in-N chance of rolling into
    # photo-cycling mode instead of a text rotation. Lock-in applies the
    # same way so we don't flip-flop mid-break.
    _PHOTO_MODE_CHANCE = 0.25

    def _pick_content_kind(self):
        """Choose which kind of content to show next.

        If a per-block lock is active, return it. Otherwise roll a fresh
        lock honouring the user's enabled replacement modes — never pick
        from the unfiltered kind list, or a disabled kind (e.g. facts
        toggled off in Settings) can reappear mid-block.
        """
        kind = getattr(self, '_locked_content_kind', None)
        if not kind:
            kind = self._roll_replacement_mode()
        return kind

    def _roll_replacement_mode(self):
        """Pick a content kind at the start of an ad break.

        Honours the user's preferences from ``minus.replacement_modes``:
          - If only 'vocab' is enabled → always vocab
          - If 'photos' is enabled AND at least one photo is uploaded, there
            is a :attr:`_PHOTO_MODE_CHANCE` chance of photo-cycling
          - Otherwise weighted-random over the currently-enabled text kinds
        """
        modes_enabled = self._get_enabled_replacement_modes()
        # If photos enabled and we have photos on disk, roll the dice.
        if 'photos' in modes_enabled:
            try:
                from photo_library import get_photo_library
                if get_photo_library().random_photo_id() is not None:
                    if random.random() < self._PHOTO_MODE_CHANCE or modes_enabled == {'photos'}:
                        return 'photos'
            except Exception as e:
                logger.debug(f"[DRMAdBlocker] photo pool check failed: {e}")
        # Filter text kinds to what's enabled; fall back to vocab.
        text_kinds = tuple(k for k in self._CONTENT_KINDS if k in modes_enabled)
        if not text_kinds:
            return 'vocab'
        weights = tuple(
            w for k, w in zip(self._CONTENT_KINDS, self._CONTENT_KIND_WEIGHTS) if k in modes_enabled)
        return random.choices(text_kinds, weights=weights, k=1)[0]

    def _get_enabled_replacement_modes(self):
        """Read ``replacement_modes`` preferences from the Minus instance.

        Defaults to {'vocab', 'fact'} (text kinds on, photos off).
        """
        if self.minus and hasattr(self.minus, 'get_replacement_modes'):
            try:
                return set(self.minus.get_replacement_modes())
            except Exception as e:
                logger.debug(f"[DRMAdBlocker] get_replacement_modes failed: {e}")
        return {'vocab', 'fact'}

    def invalidate_replacement_lock(self):
        """Drop a locked content kind that settings no longer allow.

        Called by Minus.set_replacement_modes so a toggle in the web UI
        takes effect at the next rotation / next block, not after the
        30s cooldown lapses. Re-rolls immediately (instead of clearing)
        so a mid-block rotation always has a valid kind to render.
        """
        locked = getattr(self, '_locked_content_kind', None)
        if locked and locked not in self._get_enabled_replacement_modes():
            self._locked_content_kind = self._roll_replacement_mode()
            logger.info(
                f"[DRMAdBlocker] Replacement settings changed — "
                f"re-rolled locked kind {locked} -> {self._locked_content_kind}")

    def _render_vocab(self, header):
        vocab = random.choice(VOCABULARY_COMBINED)
        self._current_vocab = vocab
        # 4-tuple -> single example; 5-tuple -> two examples on own lines
        spanish, pronunciation, english = vocab[0], vocab[1], vocab[2]
        examples = [ex for ex in vocab[3:] if ex]
        example_block = "\n".join(f'"{ex}"' for ex in examples)
        prefix = f"{header}\n\n" if header else ""
        return f"{prefix}{spanish}\n({pronunciation})\n\n= {english}\n\n{example_block}"

    def _render_fact(self, header):
        title, body = random.choice(DID_YOU_KNOW)
        prefix = f"{header}\n\n" if header else ""
        return f"{prefix}DID YOU KNOW?\n{title}\n\n{body}"

    def _format_ocr_trigger(self, raw, source):
        """Build the top-right OCR snippet, ~50 chars max, parens around match.

        ``raw`` is either a (matched_keyword, snippet_text) tuple, an already-
        formatted string, or empty. We wrap the matched substring in parens
        within the snippet, e.g. ``(Ad) 0:30 left``. For VLM-only blocks or
        if no trigger is provided, returns ''.
        """
        if not raw or source in ('vlm', 'vlm+asr'):
            return ""
        try:
            if isinstance(raw, tuple) and len(raw) == 2:
                keyword, snippet = raw
                snippet = (snippet or '').strip()
                keyword = (keyword or '').strip()
                if keyword and snippet:
                    idx = snippet.lower().find(keyword.lower())
                    if idx >= 0:
                        marked = (
                            snippet[:idx]
                            + '(' + snippet[idx:idx + len(keyword)] + ')'
                            + snippet[idx + len(keyword):]
                        )
                    else:
                        marked = f"({keyword}) {snippet}"
                else:
                    marked = snippet or f"({keyword})"
            else:
                marked = str(raw).strip()
        except Exception:
            return ""
        marked = ' '.join(marked.split())  # collapse whitespace + newlines
        if len(marked) > 50:
            marked = marked[:47] + '...'
        return marked

    def _render_ocr_text(self):
        """Return the snippet string ustreamer should render in the top-right.

        Empty when debug overlay is off (so the C side draws nothing).
        """
        if not self._debug_overlay_enabled:
            return ""
        return self._ocr_trigger_text or ""

    def _get_blocking_text(self, source='default'):
        if source == 'hdmi_lost':
            return "[ NO SIGNAL ]\n\nHDMI DISCONNECTED\n\nWaiting for signal..."
        if source == 'no_hdmi_device':
            return "[ NO SIGNAL ]\n\nWAITING FOR HDMI..."
        if not self._debug_overlay_enabled:
            header = ""
        elif source == 'ocr':
            header = "[ BLOCKING // OCR ]"
        elif source == 'ocr+asr':
            header = "[ BLOCKING // OCR+ASR ]"
        elif source == 'vlm':
            header = "[ BLOCKING // VLM ]"
        elif source == 'vlm+asr':
            header = "[ BLOCKING // VLM+ASR ]"
        elif source == 'both':
            header = "[ BLOCKING // OCR+VLM ]"
        elif source == 'both+asr':
            header = "[ BLOCKING // OCR+VLM+ASR ]"
        else:
            header = "[ BLOCKING ]"

        kind = self._pick_content_kind()
        if kind == 'fact':
            return self._render_fact(header)
        return self._render_vocab(header)

    def _get_debug_text(self):
        uptime_str = "N/A"
        if self.minus and hasattr(self.minus, 'start_time'):
            uptime_secs = int(time.time() - self.minus.start_time)
            hours, remainder = divmod(uptime_secs, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"

        current_block_time = 0
        if self._current_block_start:
            current_block_time = time.time() - self._current_block_start

        total_block_secs = int(self._total_blocking_time + current_block_time)
        block_mins, block_secs = divmod(total_block_secs, 60)
        block_hours, block_mins = divmod(block_mins, 60)
        block_time_str = f"{block_hours}h {block_mins}m {block_secs}s" if block_hours > 0 else f"{block_mins}m {block_secs}s"

        # Format time saved
        time_saved_secs = int(self._total_time_saved)
        saved_mins, saved_secs = divmod(time_saved_secs, 60)
        saved_hours, saved_mins = divmod(saved_mins, 60)
        if saved_hours > 0:
            time_saved_str = f"{saved_hours}h {saved_mins}m {saved_secs}s"
        elif saved_mins > 0:
            time_saved_str = f"{saved_mins}m {saved_secs}s"
        else:
            time_saved_str = f"{saved_secs}s"

        debug_text = f"UPTIME    {uptime_str}\nBLOCKED   {self._total_ads_blocked}\nBLK TIME  {block_time_str}\nSAVED     {time_saved_str}"
        # Ad countdown bar (YouTube/Netflix "Ad 0:30" -> drains to 0)
        countdown_bar = self._ad_countdown_bar()
        if countdown_bar:
            debug_text += f"\n{countdown_bar}"
        # Audio-reactive bar (falls silent when we're muted, which is the
        # whole point — the Spanish overlay steals attention from the ad)
        if self.audio and hasattr(self.audio, 'get_level_bars'):
            try:
                bars = self.audio.get_level_bars(width=12)
                if bars:
                    debug_text += f"\nAUDIO     {bars}"
            except Exception as e:
                logger.debug(f"[DRMAdBlocker] audio bars skipped: {e}")
        if self._skip_text:
            debug_text += f"\n> {self._skip_text}"
        return debug_text

    # Palette of Spanish-word colors in YUV (Y=luma, U=blue chroma, V=red
    # chroma). Picked to stay legible on the dark blocking background and to
    # read as distinct hues after the ustreamer MPP encoder maps back to
    # RGB-ish display. Each entry is (name, y, u, v).
    WORD_COLOR_PALETTE = (
        ("purple",    140, 175, 145),   # original accent
        ("magenta",   150, 160, 180),
        ("cyan",      180, 170, 100),
        ("lime",      200, 100, 120),
        ("amber",     200, 100, 160),
        ("pink",      190, 150, 175),
        ("mint",      210, 120, 115),
        ("sky",       175, 180, 115),
        ("coral",     185, 110, 175),
        ("teal",      165, 155, 110),
    )

    def _randomize_word_color(self):
        """Pick a new Spanish-word color and push it to ustreamer.

        Runs before each vocab rotation so the word cycles through the
        palette rather than sitting on a single hue. Failures are harmless —
        the previous color sticks.
        """
        try:
            _name, y, u, v = random.choice(self.WORD_COLOR_PALETTE)
            self._blocking_api_call(
                '/blocking/set',
                {'word_y': str(y), 'word_u': str(u), 'word_v': str(v)},
            )
        except Exception as e:
            logger.debug(f"[DRMAdBlocker] word color randomize skipped: {e}")

    def _rotation_loop(self, source):
        while not self._stop_rotation.is_set():
            kind = self._pick_content_kind()
            # Every rotation re-asserts preview_enabled + preview_grayscale
            # regardless of mode so no path can accidentally drop the corner
            # preview. Cheap (one /blocking/set), and makes the UI promise
            # unconditional: "ad is always visible, desaturated, in the
            # corner — during facts, vocab, photos, whatever."
            preview_state = {
                'preview_enabled': 'true' if self._preview_enabled else 'false',
                'preview_grayscale': 'true' if self._preview_grayscale else 'false',
                'text_ocr': self._render_ocr_text(),
            }
            if kind == 'photos':
                # Photo-cycling replacement mode: swap the background image
                # every ~5s, hide the large text so the photo reads as a
                # screensaver. Stats + countdown bar stay on top. Preview
                # window stays (greyscaled) so the user can still peek at
                # the ad.
                self._push_photo_background()
                self._blocking_api_call('/blocking/set', {'text_vocab': '', **preview_state})
                self._stop_rotation.wait(5.0)
            else:
                self._randomize_word_color()
                text = self._get_blocking_text(source)
                self._blocking_api_call('/blocking/set', {'text_vocab': text, **preview_state})
                self._stop_rotation.wait(random.uniform(11.0, 15.0))

    def _push_photo_background(self):
        """Send a random library photo to ustreamer as the blocking bg.

        ustreamer scales whatever we POST to fill the composite frame, so a
        raw photo of arbitrary aspect gets STRETCHED/distorted. We letterbox
        it to the display aspect first (fit-within + black bars), so the
        encoder's full-frame scale is aspect-preserving. Runs on the ~5s
        photo-swap cadence (not a hot path), so the decode+pad+encode is fine.
        """
        try:
            from photo_library import get_photo_library
            lib = get_photo_library()
            photo_id = lib.random_photo_id()
            if not photo_id:
                return False
            data = lib.get_photo_bytes(photo_id)
            if not data:
                return False
            # Letterbox to display aspect to avoid stretching.
            try:
                import numpy as np
                W, H = int(self.output_width), int(self.output_height)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and W > 0 and H > 0:
                    ih, iw = img.shape[:2]
                    scale = min(W / iw, H / ih)
                    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
                    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                    canvas = np.zeros((H, W, 3), dtype=np.uint8)  # black bars
                    x, y = (W - nw) // 2, (H - nh) // 2
                    canvas[y:y + nh, x:x + nw] = resized
                    ok, enc = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        data = enc.tobytes()
            except Exception as e:
                logger.debug(f"[DRMAdBlocker] photo letterbox skipped (sending raw): {e}")
            result = self._blocking_api_call(
                '/blocking/background', data=data, method='POST', timeout=0.8)
            return bool(result and result.get('ok', False))
        except Exception as e:
            logger.warning(f"[DRMAdBlocker] photo background push failed: {e}")
            return False

    def _start_rotation(self, source):
        self._stop_rotation.clear()
        self._rotation_thread = threading.Thread(target=self._rotation_loop, args=(source,), daemon=True)
        self._rotation_thread.start()

    def _stop_rotation_thread(self):
        self._stop_rotation.set()
        if self._rotation_thread:
            self._rotation_thread.join(timeout=1.0)
            self._rotation_thread = None

    def _debug_loop(self):
        while not self._stop_debug.is_set():
            if self._debug_overlay_enabled:
                self._blocking_api_call('/blocking/set', {'text_stats': self._get_debug_text()})
            self._stop_debug.wait(self._debug_interval)

    def _start_debug(self):
        if not self._debug_overlay_enabled:
            self._blocking_api_call('/blocking/set', {'text_stats': ''})
            return
        self._stop_debug.clear()
        self._debug_thread = threading.Thread(target=self._debug_loop, daemon=True, name="DebugUpdate")
        self._debug_thread.start()

    def _stop_debug_thread(self):
        self._stop_debug.set()
        if self._debug_thread:
            self._debug_thread.join(timeout=2.0)
            self._debug_thread = None

    def _start_snapshot_buffer(self):
        self._stop_snapshot_buffer.clear()
        self._snapshot_buffer_thread = threading.Thread(target=self._snapshot_buffer_loop, daemon=True, name="SnapshotBuffer")
        self._snapshot_buffer_thread.start()

    def _stop_snapshot_buffer_thread(self):
        self._stop_snapshot_buffer.set()
        if self._snapshot_buffer_thread:
            self._snapshot_buffer_thread.join(timeout=2.0)
            self._snapshot_buffer_thread = None

    def _snapshot_buffer_loop(self):
        consecutive_failures = 0
        while not self._stop_snapshot_buffer.is_set():
            try:
                url = f"http://localhost:{self.ustreamer_port}/snapshot"
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    self._snapshot_buffer.append({'data': response.read(), 'time': time.time()})
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                consecutive_failures += 1
                # Log only first failure and every 10th to avoid spam
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    logger.debug(f"[DRMAdBlocker] Snapshot buffer fetch failed ({consecutive_failures}x): {e}")
            self._stop_snapshot_buffer.wait(self._snapshot_interval)

    def _generate_fallback_background(self):
        """Build a dark radial-gradient JPEG for blocks that start before the
        snapshot buffer has content (e.g. ad in the first ~6s after restart).

        Pure-Python/OpenCV, zero network. Returns JPEG bytes or None.
        """
        try:
            import cv2
            import numpy as np
            w, h = 960, 540  # small is fine — ustreamer scales to output
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = w / 2.0, h / 2.0
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            dist /= dist.max()
            # Dark matrix-green radial gradient matching the AESTHETICS palette
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[..., 0] = (25 * (1 - dist)).astype(np.uint8)   # B
            img[..., 1] = (50 * (1 - dist)).astype(np.uint8)   # G
            img[..., 2] = (20 * (1 - dist)).astype(np.uint8)   # R
            _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return enc.tobytes()
        except Exception as e:
            logger.debug(f"[DRMAdBlocker] Fallback gradient generation failed: {e}")
            return None

    def _upload_background(self):
        """Upload pixelated background. Thread-safe for async execution."""
        try:
            # Thread-safe: copy snapshot data atomically to avoid race conditions
            try:
                if not self._snapshot_buffer:
                    # Buffer empty (early boot / post-restart). Upload a cheap
                    # radial gradient as fallback so the overlay isn't flat
                    # black while we wait for the first capture.
                    fallback = self._generate_fallback_background()
                    if fallback:
                        self._blocking_api_call(
                            '/blocking/background', data=fallback,
                            method='POST', timeout=0.5)
                        logger.info(
                            "[DRMAdBlocker] Uploaded fallback gradient "
                            "(snapshot buffer empty)"
                        )
                        return True
                    logger.warning("[DRMAdBlocker] No snapshots in buffer for background")
                    return False
                # Copy data immediately to avoid race with buffer updates
                snapshot_entry = self._snapshot_buffer[0]
                snapshot_data = bytes(snapshot_entry['data'])  # Make a copy
            except (IndexError, KeyError):
                logger.warning("[DRMAdBlocker] Snapshot buffer race condition - skipping background")
                return False

            logger.info(f"[DRMAdBlocker] Uploading background ({len(self._snapshot_buffer)} snapshots in buffer)")

            try:
                import cv2
                import numpy as np
                nparr = np.frombuffer(snapshot_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    factor = 20
                    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    pixelated = (pixelated * 0.6).astype(np.uint8)
                    _, encoded = cv2.imencode('.jpg', pixelated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    snapshot_data = encoded.tobytes()
                    logger.info(f"[DRMAdBlocker] Pixelated background: {w}x{h}, {len(snapshot_data)} bytes")
                else:
                    logger.warning("[DRMAdBlocker] Failed to decode snapshot for pixelation")
            except ImportError:
                logger.warning("[DRMAdBlocker] OpenCV not available for pixelation")
            except Exception as e:
                logger.warning(f"[DRMAdBlocker] Pixelation failed: {e}")

            result = self._blocking_api_call('/blocking/background', data=snapshot_data, method='POST', timeout=0.5)
            success = result is not None and result.get('ok', False)
            if success:
                logger.info(f"[DRMAdBlocker] Background uploaded successfully")
            else:
                logger.warning(f"[DRMAdBlocker] Background upload failed: {result}")
            return success

        except Exception as e:
            # Catch-all for thread safety - don't let exceptions crash the background thread
            logger.exception(f"[DRMAdBlocker] Background upload error: {e}")
            return False

    def _ease_out(self, t):
        return 1 - (1 - t) ** 2

    def _ease_in(self, t):
        return t ** 2

    def _stop_animation_thread(self):
        self._stop_animation.set()
        if self._animation_thread:
            self._animation_thread.join(timeout=2.0)
            self._animation_thread = None
        self._animating = False
        self._animation_direction = None

    def _start_animation(self, direction, source=None):
        self._stop_animation_thread()
        self._stop_animation.clear()
        self._animation_source = source
        self._animating = True
        self._animation_direction = direction
        self._animation_thread = threading.Thread(target=self._animation_loop, args=(direction,), daemon=True, name=f"Animation-{direction}")
        self._animation_thread.start()

    def _animation_loop(self, direction):
        start_time = time.time()
        duration = self._animation_duration_start if direction == 'start' else self._animation_duration_end

        full_x, full_y = 0, 0
        full_w, full_h = self._frame_width, self._frame_height
        corner_x = self._frame_width - self._preview_w - self._preview_padding
        corner_y = self._frame_height - self._preview_h - self._preview_padding
        corner_w, corner_h = self._preview_w, self._preview_h

        while not self._stop_animation.is_set():
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / duration)

            if direction == 'start':
                t = self._ease_out(progress)
                x = int(full_x + (corner_x - full_x) * t)
                y = int(full_y + (corner_y - full_y) * t)
                w = int(full_w + (corner_w - full_w) * t)
                h = int(full_h + (corner_h - full_h) * t)
            else:
                t = self._ease_in(progress)
                x = int(corner_x + (full_x - corner_x) * t)
                y = int(corner_y + (full_y - corner_y) * t)
                w = int(corner_w + (full_w - corner_w) * t)
                h = int(corner_h + (full_h - corner_h) * t)

            self._blocking_api_call('/blocking/set', {'preview_x': str(x), 'preview_y': str(y), 'preview_w': str(w), 'preview_h': str(h)})

            if progress >= 1.0:
                break
            time.sleep(0.1)  # 10fps animation (was 0.016 = 60fps) - reduces HTTP calls from ~19 to ~3

        # Set final position
        if direction == 'start':
            self._blocking_api_call('/blocking/set', {'preview_x': str(corner_x), 'preview_y': str(corner_y), 'preview_w': str(corner_w), 'preview_h': str(corner_h)})
        else:
            self._blocking_api_call('/blocking/set', {'preview_x': '0', 'preview_y': '0', 'preview_w': str(full_w), 'preview_h': str(full_h)})

        self._animating = False
        self._animation_direction = None
        if direction == 'start':
            self._on_start_animation_complete()
        else:
            self._on_end_animation_complete()

    def _on_start_animation_complete(self):
        logger.debug("[DRMAdBlocker] Start animation complete")
        source = self._animation_source or 'default'
        self._blocking_api_call('/blocking/set', {'text_vocab': self._get_blocking_text(source)})
        self._start_rotation(source)
        self._current_block_start = time.time()
        self._total_ads_blocked += 1
        self._start_debug()

    def _on_end_animation_complete(self):
        logger.debug("[DRMAdBlocker] End animation complete")
        self._blocking_api_call('/blocking/set', {'enabled': 'false'}, timeout=0.5)

        # Write blocking state to file for zero-overhead checks (avoids HTTP)
        try:
            with open('/dev/shm/minus_blocking_state', 'w') as f:
                f.write('0')
        except Exception as e:
            logger.debug(f"[DRMAdBlocker] Failed to write blocking state file: {e}")

        if self.audio:
            self.audio.unmute()

    def set_minus(self, minus_instance):
        self.minus = minus_instance

    def set_audio(self, audio):
        self.audio = audio

    def is_preview_enabled(self):
        return self._preview_enabled

    def set_preview_enabled(self, enabled):
        self._preview_enabled = enabled
        logger.info(f"[DRMAdBlocker] Preview {'enabled' if enabled else 'disabled'}")
        if self.is_visible:
            self._blocking_api_call('/blocking/set', {'preview_enabled': 'true' if enabled else 'false'})

    def set_ad_seconds_remaining(self, seconds):
        """Called from the OCR loop when an "Ad N:MM" timer is spotted."""
        if seconds is None:
            return
        try:
            seconds = int(seconds)
        except (TypeError, ValueError):
            return
        if seconds < 0:
            return
        self._ad_seconds_remaining = seconds
        self._ad_seconds_anchor = time.time()
        # Peak resets when value goes UP (new ad started or OCR misread low)
        if self._ad_seconds_peak is None or seconds > self._ad_seconds_peak:
            self._ad_seconds_peak = seconds

    def _clear_ad_countdown(self):
        self._ad_seconds_remaining = None
        self._ad_seconds_anchor = 0.0
        self._ad_seconds_peak = None

    def _ad_countdown_bar(self, width=10):
        """Return a `[▓▓▓░░░] 12s` style bar, or '' if no timer known.

        Uses the peak value seen this ad as 100% so the bar drains instead
        of snapping around. Decays between OCR samples using wall-clock.
        """
        if self._ad_seconds_remaining is None or self._ad_seconds_peak is None:
            return ''
        if self._ad_seconds_peak <= 0:
            return ''
        elapsed = max(0.0, time.time() - self._ad_seconds_anchor)
        current = max(0.0, self._ad_seconds_remaining - elapsed)
        frac = max(0.0, min(1.0, current / self._ad_seconds_peak))
        filled = int(round(frac * width))
        bar = ('#' * filled) + ('.' * (width - filled))
        return f"AD LEFT   [{bar}] {int(current):>2d}s"

    def is_preview_grayscale(self):
        return self._preview_grayscale

    def set_preview_grayscale(self, enabled):
        """Enable/disable greyscale on the ad preview window (live updates)."""
        self._preview_grayscale = enabled
        logger.info(f"[DRMAdBlocker] Preview greyscale {'enabled' if enabled else 'disabled'}")
        if self.is_visible:
            self._blocking_api_call('/blocking/set', {'preview_grayscale': 'true' if enabled else 'false'})

    def is_debug_overlay_enabled(self):
        return self._debug_overlay_enabled

    def set_debug_overlay_enabled(self, enabled):
        self._debug_overlay_enabled = enabled
        logger.info(f"[DRMAdBlocker] Debug overlay {'enabled' if enabled else 'disabled'}")
        if self.is_visible:
            # Re-render the vocab so the [BLOCKING // ...] header appears or
            # disappears immediately, plus push the top-right OCR text in
            # whichever direction (filled when on, empty when off).
            self._blocking_api_call('/blocking/set', {
                'text_vocab': self._get_blocking_text(self.current_source or 'default'),
                'text_ocr': self._render_ocr_text(),
            })
            if enabled:
                if not self._debug_thread or not self._debug_thread.is_alive():
                    self._start_debug()
            else:
                self._stop_debug_thread()
                self._blocking_api_call('/blocking/set', {'text_stats': ''})

    def set_ocr_trigger_text(self, raw, source=None):
        """Update the top-right OCR snippet during an active block.

        Called when OCR re-fires while a block is already up, e.g. countdown
        ticking from "Ad 0:30" -> "Ad 0:15". Cheap (one /blocking/set).
        """
        eff_source = source if source is not None else (self.current_source or 'default')
        self._ocr_trigger_text = self._format_ocr_trigger(raw, eff_source)
        if self.is_visible:
            self._blocking_api_call('/blocking/set', {'text_ocr': self._render_ocr_text()})

    def is_pixelated_background_enabled(self):
        return self._pixelated_background_enabled

    def set_pixelated_background_enabled(self, enabled):
        self._pixelated_background_enabled = enabled
        logger.info(f"[DRMAdBlocker] Pixelated background {'enabled' if enabled else 'disabled'}")

    def set_skip_status(self, available: bool, text: str = None):
        self._skip_available = available
        self._skip_text = text

    def get_skip_status(self) -> tuple:
        return (self._skip_available, self._skip_text)

    def add_time_saved(self, seconds: float):
        """Add to the total time saved by skipping ads."""
        self._total_time_saved += seconds
        logger.info(f"[DRMAdBlocker] Time saved: +{seconds:.0f}s (total: {self._total_time_saved:.0f}s)")

    def get_time_saved(self) -> float:
        """Get total time saved in seconds."""
        return self._total_time_saved

    def get_current_vocabulary(self) -> dict:
        """Get the current vocabulary word being displayed."""
        if self._current_vocab and self.is_visible:
            # VOCABULARY_COMBINED mixes 4-tuples (SPANISH_VOCABULARY) and
            # 5-tuples (SPANISH_VOCABULARY_EXTENDED, which carries a 2nd
            # example). Take the first 4 fields so a 5-tuple doesn't raise
            # "too many values to unpack".
            spanish, pronunciation, english, example = self._current_vocab[:4]
            return {
                'word': spanish,
                'pronunciation': pronunciation,
                'translation': english,
                'example': example,
            }
        return {'word': None, 'pronunciation': None, 'translation': None, 'example': None}

    def set_test_mode(self, duration_seconds: float):
        self._test_blocking_until = time.time() + duration_seconds
        logger.info(f"[DRMAdBlocker] Test mode enabled for {duration_seconds}s")

    def clear_test_mode(self):
        self._test_blocking_until = 0
        logger.info("[DRMAdBlocker] Test mode cleared")

    def is_test_mode_active(self) -> bool:
        return self._test_blocking_until > time.time()

    def show(self, source='default', ocr_trigger_text=''):
        with self._lock:
            # Note: We still enable ustreamer blocking even without display pipeline
            # because blocking overlay works via ustreamer (for web stream) independently
            # of GStreamer display pipeline (for TV output via DRM)

            # Update the snippet only if the caller gave us something fresh, OR if
            # this is now a VLM-only block (which has no OCR snippet). Otherwise
            # keep whatever set the block — OCR text often disappears mid-block as
            # the timer ticks past the model's confidence cutoff and we don't want
            # the top-right slot to flicker on/off.
            new_snippet = self._format_ocr_trigger(ocr_trigger_text, source)
            if new_snippet or source in ('vlm', 'vlm+asr'):
                self._ocr_trigger_text = new_snippet

            if self.is_visible and self._animation_direction != 'end':
                if self.current_source != source:
                    self.current_source = source
                    # Update overlay text to reflect new source (e.g., OCR -> OCR+VLM)
                    self._blocking_api_call('/blocking/set', {
                        'text_vocab': self._get_blocking_text(source),
                        'text_ocr': self._render_ocr_text(),
                    })
                else:
                    # Same source, fresh trigger snippet — push only the OCR text.
                    self._blocking_api_call('/blocking/set', {'text_ocr': self._render_ocr_text()})
                return

            if self._animating and self._animation_direction == 'start':
                if self.current_source != source:
                    self.current_source = source
                    self._blocking_api_call('/blocking/set', {
                        'text_vocab': self._get_blocking_text(source),
                        'text_ocr': self._render_ocr_text(),
                    })
                return

            if self._animating and self._animation_direction == 'end':
                logger.info(f"[DRMAdBlocker] Reversing end animation ({source})")
                self._stop_animation_thread()

            # Lock a replacement-mode choice for this ad break (unless a
            # prior lock is still within the cooldown window, in which case
            # we reuse it — avoids flip-flopping between styles during an
            # ad cluster). If the user has enabled 'photos' mode and uploaded
            # at least one photo, we may roll into photo-cycling instead.
            # The reused kind MUST still be enabled in settings: without that
            # check, disabling e.g. facts mid-ad-pod kept serving facts from
            # the stale lock until the cooldown lapsed (or a restart).
            now = time.time()
            reused = (bool(self._locked_content_kind)
                      and now <= self._content_kind_lock_until
                      and self._locked_content_kind in self._get_enabled_replacement_modes())
            if not reused:
                self._locked_content_kind = self._roll_replacement_mode()
            logger.info(f"[DRMAdBlocker] Starting blocking ({source}) kind={self._locked_content_kind} {'reused' if reused else 'rolled'} lock_until_in={self._content_kind_lock_until - now:.1f}s")

            # Mute audio immediately
            if self.audio:
                self.audio.mute()

            # Calculate text scales based on resolution (scales designed for 4K @ 10/4)
            # At 1080p (1920 width): vocab_scale=5, stats_scale=2
            # At 4K (3840 width): vocab_scale=10, stats_scale=4
            vocab_scale = max(3, min(12, self._frame_width // 384))
            stats_scale = max(2, min(5, self._frame_width // 960))

            # Enable blocking immediately (background will upload async)
            self._blocking_api_call('/blocking/set', {
                'enabled': 'true',
                'preview_x': '0', 'preview_y': '0',
                'preview_w': str(self._frame_width), 'preview_h': str(self._frame_height),
                'preview_enabled': 'true' if self._preview_enabled else 'false',
                'preview_grayscale': 'true' if self._preview_grayscale else 'false',
                'text_vocab': '', 'text_stats': '', 'text_ocr': self._render_ocr_text(),
                'text_vocab_scale': str(vocab_scale),
                'text_stats_scale': str(stats_scale),
                'box_alpha': str(self._box_alpha),
                'text_y': str(self._text_y),
                'text_u': str(self._text_u),
                'text_v': str(self._text_v)
            }, timeout=0.5)

            self.is_visible = True
            self.current_source = source
            self._set_led_state('blocking')

            # Write blocking state to file for zero-overhead checks (avoids HTTP)
            try:
                with open('/dev/shm/minus_blocking_state', 'w') as f:
                    f.write('1')
            except Exception as e:
                logger.debug(f"[DRMAdBlocker] Failed to write blocking state file: {e}")

            if self.minus:
                self.minus.blocking_active = True

            # Upload background asynchronously (don't block animation start)
            if self._pixelated_background_enabled:
                threading.Thread(target=self._upload_background, daemon=True, name="BackgroundUpload").start()

            if self._animation_enabled:
                self._start_animation('start', source)
            else:
                # Skip animation - set final position directly to avoid glitches
                # Animation causes ~19 rapid API calls which creates stream hiccups
                corner_x = self._frame_width - self._preview_w - self._preview_padding
                corner_y = self._frame_height - self._preview_h - self._preview_padding
                self._blocking_api_call('/blocking/set', {
                    'preview_x': str(corner_x),
                    'preview_y': str(corner_y),
                    'preview_w': str(self._preview_w),
                    'preview_h': str(self._preview_h)
                })
                self._animation_source = source
                self._on_start_animation_complete()

    def hide(self, force=False):
        if not force and self._test_blocking_until > time.time():
            return

        with self._lock:
            if self._animating and self._animation_direction == 'end':
                return

            was_visible = self.is_visible
            self.is_visible = False
            self.current_source = None
            if was_visible:
                # Pick the right background state — autonomous mode running,
                # blocking paused, or just plain idle.
                if self.minus and hasattr(self.minus, '_baseline_led_state'):
                    self._set_led_state(self.minus._baseline_led_state())
                else:
                    self._set_led_state('idle')

            if self.minus:
                self.minus.blocking_active = False

            self._stop_rotation_thread()
            self._stop_debug_thread()
            self._clear_ad_countdown()
            # Keep the locked content kind valid for the cooldown window so
            # a fresh ad shortly after this one reuses the same style.
            self._content_kind_lock_until = time.time() + self.CONTENT_KIND_COOLDOWN_SECONDS

            if self._current_block_start:
                self._total_blocking_time += time.time() - self._current_block_start
                self._current_block_start = None

            self._blocking_api_call('/blocking/set', {'text_vocab': '', 'text_stats': ''})

            if not self.pipeline:
                # Still disable ustreamer blocking even without display pipeline
                # (blocking overlay works via ustreamer independently)
                self._blocking_api_call('/blocking/set', {'enabled': 'false'}, timeout=0.5)
                # Write blocking state to file
                try:
                    with open('/dev/shm/minus_blocking_state', 'w') as f:
                        f.write('0')
                except Exception:
                    pass
                if self.audio:
                    self.audio.unmute()
                return

            if not was_visible and self._animation_direction != 'start':
                return

            if self._animating:
                self._stop_animation_thread()

            if self._animation_enabled:
                logger.info("[DRMAdBlocker] Starting end animation")
                self._start_animation('end', None)
            else:
                # Skip animation - disable blocking immediately to avoid glitches
                logger.info("[DRMAdBlocker] Ending blocking (no animation)")
                self._on_end_animation_complete()

    def update(self, ad_detected, is_skippable=False, skip_location=None, ocr_detected=False, vlm_detected=False):
        if ad_detected and not is_skippable:
            if ocr_detected and vlm_detected:
                source = 'both'
            elif ocr_detected:
                source = 'ocr'
            elif vlm_detected:
                source = 'vlm'
            else:
                source = 'default'
            self.show(source)
        else:
            self.hide()

    def destroy(self):
        with self._lock:
            self._stop_watchdog_thread()
            self._stop_rotation_thread()
            self._stop_debug_thread()
            self._stop_animation_thread()
            self._stop_snapshot_buffer_thread()
            self._stop_loading_animation()
            self._stop_no_signal_animation()

            self._blocking_api_call('/blocking/set', {'clear': 'true'}, timeout=0.5)

            if self.pipeline:
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                    logger.info("[DRMAdBlocker] Pipeline stopped")
                except Exception as e:
                    logger.error(f"[DRMAdBlocker] Error stopping pipeline: {e}")
                self.pipeline = None

            self.is_visible = False


AdBlocker = DRMAdBlocker
