#!/usr/bin/env python3
"""
Minus - HDMI passthrough with ML-based ad detection.

Architecture:
- ustreamer captures from HDMI-RX and serves MJPEG stream + HTTP snapshot
- GStreamer with input-selector for instant video/blocking switching
- PaddleOCR on RKNN NPU detects ad-related text (~400ms)
- LFM2.5-VL-450M on Axera NPU provides visual understanding (~0.37s, prefill-only)
- Spanish vocabulary practice during ad blocks!

Key insight: Using GStreamer input-selector allows instant switching between
video and blocking overlay without any process restart or black screen gap.

Performance:
- Display: 30fps via GStreamer kmssink (NV12 → DRM plane 72)
- Snapshot: ~150ms non-blocking HTTP capture
- OCR: ~400-500ms per frame on RKNN NPU
- VLM: ~0.37s per frame on Axera NPU (LFM2.5-VL, prefill-only)
- Ad blocking: INSTANT switching via input-selector
"""

# ============================================================================
# EARLY BANDWIDTH FALLBACK CHECK
# Must run BEFORE any imports that might touch DRM/GStreamer
# ============================================================================
import os as _os
import subprocess as _subprocess
import time as _time

_FALLBACK_MARKER = '/tmp/minus_bandwidth_fallback_needed'
if _os.path.exists(_FALLBACK_MARKER):
    try:
        with open(_FALLBACK_MARKER, 'r') as _f:
            _connector_id = _f.read().strip()
        _os.remove(_FALLBACK_MARKER)
        print(f"[EARLY INIT] Bandwidth fallback marker found for connector {_connector_id}")

        # Aggressively kill ALL modetest and ustreamer processes to ensure DRM is free
        # The previous service instance might have left stuck processes
        print("[EARLY INIT] Killing any stuck DRM processes...")
        for _attempt in range(5):
            _subprocess.run(['pkill', '-9', 'modetest'], capture_output=True, timeout=2)
            _subprocess.run(['pkill', '-9', 'ustreamer'], capture_output=True, timeout=2)
            _time.sleep(0.5)

            # Check if any are still running
            _check = _subprocess.run(['pgrep', 'modetest'], capture_output=True)
            if _check.returncode != 0:  # No modetest found
                print(f"[EARLY INIT] DRM processes cleared after {_attempt + 1} attempts")
                break
        else:
            print("[EARLY INIT] Warning: Could not clear all modetest processes")

        # Wait for DRM to fully release
        _time.sleep(2)

        # Set color_format to YCbCr 4:2:0 (value=3) BEFORE any DRM processes start
        print(f"[EARLY INIT] Setting color_format to YCbCr 4:2:0 on connector {_connector_id}")

        # Try multiple times with short timeout
        for _attempt in range(5):
            # Kill any stuck modetest before each attempt
            _subprocess.run(['pkill', '-9', 'modetest'], capture_output=True, timeout=2)
            _time.sleep(0.3)

            try:
                _result = _subprocess.run(
                    ['sudo', 'modetest', '-M', 'rockchip', '-w', f'{_connector_id}:color_format:3'],
                    capture_output=True, text=True, timeout=5
                )
                if _result.returncode == 0:
                    print(f"[EARLY INIT] Bandwidth fallback applied successfully on attempt {_attempt + 1}")
                    break
                else:
                    print(f"[EARLY INIT] Attempt {_attempt + 1}: modetest returned {_result.returncode}")
            except _subprocess.TimeoutExpired:
                print(f"[EARLY INIT] Attempt {_attempt + 1}: timeout, killing modetest...")
                _subprocess.run(['pkill', '-9', 'modetest'], capture_output=True, timeout=2)
                _time.sleep(1)
        else:
            print("[EARLY INIT] Failed to apply bandwidth fallback after 5 attempts")

    except Exception as _e:
        print(f"[EARLY INIT] Error applying bandwidth fallback: {_e}")
        if _os.path.exists(_FALLBACK_MARKER):
            _os.remove(_FALLBACK_MARKER)

# Clean up early init variables
del _os, _subprocess, _time, _FALLBACK_MARKER
# ============================================================================

import argparse
import gc
import glob
import os
import sys
import signal
import time
import logging
import logging.handlers
import threading
import subprocess
import re
import json
import difflib
from pathlib import Path
from datetime import datetime
# Process-based OCR/VLM workers handle timeouts internally (no ThreadPoolExecutor needed)

import numpy as np
import cv2

# System settings file
SYSTEM_SETTINGS_FILE = Path.home() / '.minus_system_settings.json'


# =============================================================================
# Console Blanking - Hide dmesg/login screen before GStreamer takes over
# =============================================================================
# Add src directory to path early so we can import console module
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from console import blank_console, restore_console


def _norm_alnum(s: str) -> str:
    """Lowercase alphanumeric-only — robust OCR-text identity key
    (drops the spacing/punctuation jitter OCR adds frame-to-frame)."""
    return ''.join(c.lower() for c in s if c.isalnum())


# Blank the console immediately on import (before any output)
blank_console()


# Note: Previously had SuppressLibjpegWarnings context manager here but it caused
# file descriptor leaks over time (~500k calls over 13hrs exhausted FD limit).
# libjpeg warnings are harmless, so we just let them through now.

# Set up logging with rotation (max 5MB, keep 3 backups)
log_format = '%(asctime)s [%(levelname).1s] %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S'

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers to prevent duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add file handler with rotation
# Use /tmp/minus.log - sudoers allows passwordless management
log_file = Path('/tmp/minus.log')
try:
    if log_file.exists():
        try:
            with open(log_file, 'a'):
                pass
        except PermissionError:
            # Use sudo to fix permissions (sudoers.d/minus allows this)
            import subprocess
            subprocess.run(['sudo', 'rm', '-f', str(log_file)], capture_output=True)
    if not log_file.exists():
        log_file.touch(mode=0o666)
except Exception:
    pass

file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3
)
file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
file_handler.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# Add console handler for terminal output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
console_handler.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

logger = logging.getLogger('Minus')

# Suppress OpenCV JPEG warnings (this only affects OpenCV's own logging, not libjpeg)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# Import extracted modules
from drm import probe_drm_output
from v4l2 import probe_v4l2_device
from config import MinusConfig, USTREAMER_PATH, OCR_MODEL_DIR
from capture import UstreamerCapture
from screenshots import ScreenshotManager
from skip_detection import check_skip_opportunity, extract_ad_seconds_remaining

# Import OCR module
try:
    from ocr_worker import OCRProcess
    from ocr import PaddleOCR
    HAS_OCR = True
except ImportError as e:
    logger.warning(f"OCR module not available: {e}")
    HAS_OCR = False
    PaddleOCR = None

# Import AdBlocker module
try:
    from ad_blocker import AdBlocker
    HAS_ADBLOCKER = True
except ImportError as e:
    logger.warning(f"AdBlocker module not available: {e}")
    HAS_ADBLOCKER = False

# Import VLM module (process-based for hard 2s timeout)
try:
    from vlm_worker import VLMProcess
    HAS_VLM = True
except ImportError as e:
    logger.warning(f"VLM module not available: {e}")
    HAS_VLM = False

# Import Audio module
try:
    from audio import AudioPassthrough, AudioASRTap
    HAS_AUDIO = True
except ImportError as e:
    logger.warning(f"Audio module not available: {e}")
    HAS_AUDIO = False

# Import ASR module (faster-whisper-driven, runs in a subprocess worker
# for hard-timeout safety — see src/asr_worker.py). Optional — installs
# without faster-whisper will skip the audio-tap branch and the ASR
# thread, leaving the audio pipeline byte-identical to the pre-ASR shape.
try:
    from asr import ASRManager, is_asr_available
    HAS_ASR = True
except ImportError as e:
    logger.warning(f"ASR module not available: {e}")
    HAS_ASR = False
    def is_asr_available():
        return False

# Import Health Monitor
try:
    from health import HealthMonitor
    HAS_HEALTH = True
except ImportError as e:
    logger.warning(f"Health module not available: {e}")
    HAS_HEALTH = False

# Import Web UI
try:
    from webui import WebUI
    HAS_WEBUI = True
except ImportError as e:
    logger.warning(f"WebUI module not available: {e}")
    HAS_WEBUI = False

# Import Fire TV Setup Manager
try:
    from fire_tv_setup import FireTVSetupManager
    from autonomous_mode import AutonomousMode
    HAS_FIRE_TV = True
except ImportError as e:
    logger.warning(f"Fire TV module not available: {e}")
    HAS_FIRE_TV = False

# Import IR Transmitter (REI 8K HDMI switch control via PWM3)
try:
    from ir_transmitter import IRTransmitter
    HAS_IR = True
except ImportError as e:
    logger.warning(f"IR transmitter module not available: {e}")
    HAS_IR = False

# Import Status LED controller (WS2812B status strip on SPI0 MOSI)
try:
    from status_led_controller import StatusLEDController
    HAS_STATUS_LEDS = True
except ImportError as e:
    logger.warning(f"Status LED module not available: {e}")
    HAS_STATUS_LEDS = False

# Import Notification Overlay
try:
    from overlay import NotificationOverlay, SystemNotification
    HAS_OVERLAY = True
except ImportError as e:
    logger.warning(f"Overlay module not available: {e}")
    HAS_OVERLAY = False

# Import WiFi Manager
try:
    from wifi_manager import get_wifi_manager, WiFiManager
    HAS_WIFI_MANAGER = True
except ImportError as e:
    logger.warning(f"WiFi Manager module not available: {e}")
    HAS_WIFI_MANAGER = False


class Minus:
    """
    Minus - HDMI passthrough with ML-based ad detection.

    Uses a single GStreamer pipeline with input-selector for instant
    switching between video and blocking overlay.
    """

    def __init__(self, config: MinusConfig = None):
        if config is None:
            config = MinusConfig()
        self.config = config
        self.device = config.device
        self.ustreamer_process = None
        self.frame_capture = None
        self.running = False
        self.blocking_active = False
        self._hdmi_recovery_in_progress = False  # Prevent main loop interference during HDMI recovery
        self._hdmi_signal_lost = False  # Pause detection workers when HDMI signal is lost

        # ML processing
        self.ocr = None
        self.vlm = None
        self.ad_blocker = None
        self.audio = None
        self.health_monitor = None
        self.ml_thread = None
        self.vlm_thread = None

        # Display state
        self.display_connected = False
        self.display_error = None
        self._display_retry_thread = None
        self._display_retry_interval = 7  # seconds between retry attempts

        # VLM degradation state
        self.vlm_disabled = False
        self.vlm_consecutive_timeouts = 0
        self.vlm_max_timeouts = 5  # Disable VLM after this many consecutive timeouts

        # OCR detection state (PRIMARY - high trust)
        self.ocr_ad_detected = False
        self.ocr_ad_detection_count = 0
        self.ocr_no_ad_count = 0
        self.last_ocr_ad_time = 0

        # VLM detection state (SECONDARY - contextual trust)
        self.vlm_ad_detected = False
        self.vlm_frame_count = 0
        self.vlm_consecutive_ad_count = 0
        self.vlm_no_ad_count = 0

        # VLM false-positive feedback loop.
        # When the user pauses during a VLM-only block (blocking_source ==
        # "vlm"), that's an explicit signal that VLM misclassified the
        # frame — they're saying "this is not actually an ad". Two
        # responses: (1) save the VLM-triggering frame to screenshots/
        # non_ads/ as training data so a future retrain has the example;
        # (2) put VLM contribution to blocking on a 5-min cooldown so the
        # same misclassified content doesn't immediately re-trigger if
        # the user resumes early. Only applies when blocking_source is
        # exactly "vlm" — if OCR also confirmed (source "both"), the
        # block wasn't a VLM-alone misclassification and we use the
        # existing pause behaviour. OCR-only blocks (source "ocr") are
        # also unaffected.
        self.vlm_paused_until = 0.0
        self.VLM_FALSE_POSITIVE_COOLDOWN = 300.0  # 5 minutes
        # The frame that most-recently caused a VLM AD verdict. Updated
        # in the VLM dispatch loop every time `detect_ad` returns True,
        # cleared when a block ends. This is the frame we save as non_ad
        # when the user signals misclassification.
        self.last_vlm_ad_frame = None
        self.last_vlm_ad_frame_time = 0.0

        # VLM stability system - sliding window approach to prevent waffling
        # Re-retuned for LFM2.5-VL-450M-ft-v2-fused-v2 (May 2026). LFM2 has
        # structurally tighter per-frame separation than FastVLM iter4:
        # holdout non-ad-recall 99.2% vs iter4's 95.25% (≈4× lower per-frame
        # FP rate), with clean-video p_yes ≈ 0.001–0.01 and confident-ad
        # p_yes ≈ 0.97–0.99. The iter4-era hardening to 5 decisions / 0.80
        # agreement was added to absorb iter4's mid-show VLM-only FPs — a
        # failure class LFM2 mostly does not produce. With those iter4
        # params, the LFM2 simulator (tests/test_vlm_decision_sim_lfm2.py,
        # 2560-combo sweep, 64 scenarios × 30 seeds, holdout-bootstrapped)
        # measures V_det mean 9.3s / p95 20s and 7.5% VLM-only miss rate —
        # the start gate cannot accumulate enough votes fast enough on
        # genuine ads.
        # min_decisions 5→3 and start_agreement 0.80→0.70 are the iter4-era
        # SWEEP WINNER's shape, picked here as the middle ground between the
        # LFM2 sweep's most-aggressive winner (min_dec=2 / agree=0.60) and
        # iter4-hardened defaults. Math: at min_dec=3 / 0.70+0.10hyst=0.80
        # effective, P(phantom from 3 consecutive non-ad frames at LFM2's
        # 0.8% per-frame FP) ≈ 5e-7 per window → ~0.1 phantoms/day on
        # holdout-bootstrapped content (vs ~8/hour at the same params on
        # iter4's noisier distribution — that's why iter4 needed 5/0.80).
        # vlm_history_window=8s stays (validated via the iter4 sim sweep
        # — collapses stale-content-vote dilution; LFM2 sweep agrees).
        # Sim metrics on the new params: V_det mean ~4.5s / p95 ~6s,
        # V_miss ~0%, phantom 0. Rollback to 5 / 0.80 if real-world
        # mid-show VLM-only false triggers reappear (the iter4 failure
        # mode); the comment block at the rollback point should document
        # whatever LFM2-era regression motivated it.
        self.vlm_decision_history = []      # List of (timestamp, is_ad) tuples
        self.vlm_history_window = 8.0       # Look at last 8 seconds of decisions (iter4 sweep + LFM2 sweep agree)
        self.vlm_min_decisions = 3          # Need 3 decisions to act solo (iter4 hardened 4→3→5 against iter4-FPs; LFM2 retune 5→3 — see comment block above)
        self.vlm_start_agreement = 0.70     # 70% ad agreement to START blocking solo (+0.10 hysteresis → 0.80 effective; LFM2 retune 0.80→0.70 — see comment block above; OCR-corroborated uses immediate shortcut at ~line 2778)
        self.vlm_stop_agreement = 0.75      # Need 75% no-ad agreement to STOP blocking
        self.vlm_hysteresis_boost = 0.10    # Extra agreement needed to change current state
        self.vlm_start_threshold_cap = 0.95 # Cap on effective start threshold so hysteresis can't push it beyond what real-world noise allows

        # State change rate limiting
        self.vlm_last_state_change = 0      # When VLM state last changed
        self.vlm_min_state_duration = 8.0   # Min seconds before allowing state change
        self.vlm_cooldown_active = False    # Currently in cooldown period

        # Legacy counters (still used for some logic)
        self.vlm_waffle_count = 0           # How many recent flip-flops (used for logging)
        self.vlm_last_state = None          # Last VLM state ('ad' or 'no-ad')
        self.vlm_state_change_time = 0      # When state last changed

        # Home screen detection - suppress ad detection on streaming app interfaces
        # When OCR detects these keywords, both OCR and VLM ad detection is suppressed
        # (e.g., "Sponsored" rows on Fire TV home are promotional but not video ads)
        self.home_screen_keywords = {
            'home', 'disney+', 'netflix', 'youtube', 'hulu', 'prime video',
            'amazon', 'settings', 'search', 'library', 'watchlist', 'my stuff',
            'continue watching', 'recommended', 'trending', 'popular', 'new releases',
            'categories', 'genres', 'apps', 'channels', 'live tv',
            # Fire TV specific
            'surprise me', 'see more', 'for you',
            'recently added', 'top picks', 'movies', 'tv shows'
        }

        # Video player interface keywords - suppress VLM false positives on video UIs
        # VLM often thinks video player interfaces are ads
        self.video_interface_keywords = {
            # Video player controls/info
            'subscribe', 'subscribed', 'description', 'comments',
            'views', 'likes', 'share', 'save', 'download',
            # Time indicators (e.g., "3 years ago", "5 months ago")
            'ago', 'year', 'month', 'week', 'day', 'hour',
            # Music/video platforms
            'colors', 'vevo', 'official', 'music video', 'lyric',
            # Channel indicators
            'channel', 'playlist', 'queue', 'autoplay',
            # YouTube specific
            'show more', 'show less', 'read more',
        }

        self.last_ocr_texts = []            # Last OCR detected texts
        # Periodic non-ad screenshot sampler: capture the current frame as
        # training data every NONAD_SAMPLE_INTERVAL seconds while content
        # is playing normally (no block active, no static-screen freeze, no
        # ad keywords matched). Builds a steady stream of non-ad examples
        # for future VLM retraining beyond the user-pause feedback path.
        self._last_nonad_sample_time = 0.0
        self.NONAD_SAMPLE_INTERVAL = float(
            os.environ.get('MINUS_NONAD_SAMPLE_INTERVAL', '90'))
        # Most recent OCR matched keywords (list of (keyword, snippet) tuples).
        # Picked up by ad_blocker.show() to render a "(Ad) 0:30 left" hint in
        # the top-right of the blocking overlay when debug mode is on.
        self.last_matched_keywords = []
        self.home_screen_detected = False   # True if home screen keywords found
        self.home_screen_detect_time = 0    # When home screen was last detected
        self.video_interface_detected = False  # True if video player interface detected
        self.video_interface_detect_time = 0   # When video interface was last detected

        # HDMI-IN audio_present cache (v4l2-ctl subprocess is expensive)
        self._hdmi_audio_present_cache = None
        self._hdmi_audio_present_cache_time = 0.0

        # Combined ad detection state
        self.ad_detected = False
        self.frame_count = 0
        self.blocking_start_time = 0
        self.blocking_source = None
        # Whether ASR has confirmed (marketing language heard) for the active
        # block. Kept SEPARATE from blocking_source (which stays the base
        # ocr/vlm/both for all stop-logic checks) and only decorates the
        # DISPLAY label → ocr+asr / both+asr / vlm+asr. Set at start from the
        # ASR verdict and upgraded mid-block when ASR later confirms (ASR
        # usually confirms a few seconds after an instant OCR block fires).
        self.blocking_asr_confirmed = False

        # Weighted detection parameters
        self.OCR_TRUST_WINDOW = 5.0
        self.VLM_ALONE_THRESHOLD = self.config.vlm_alone_threshold
        # MIN_BLOCKING_DURATION has a falloff. Each consecutive ad (second ad
        # that starts shortly after the previous one ended) shortens the
        # minimum block duration: 3.0 -> 2.5 -> 2.0 -> 1.5 -> 1.0 s.
        # Floor is 1.0s for OCR-only, 1.5s when OCR+VLM both agree (the extra
        # stagger lets VLM's slower cycle catch up before we unblock). Counter
        # resets after MIN_DURATION_RESET_GAP seconds without any block.
        self.MIN_BLOCKING_DURATION_BASE = 3.0
        self.MIN_BLOCKING_DURATION_STEP = 0.5
        self.MIN_BLOCKING_DURATION_FLOOR_OCR = 1.0
        self.MIN_BLOCKING_DURATION_FLOOR_BOTH = 1.5
        # VLM-only blocks get a very low minimum hold: the hardened
        # VLM-only trigger above makes a false solo block rare, and when
        # one does slip through (e.g. mid-show) it should clear the
        # instant VLM flips to no-ad (gated only by VLM_STOP_THRESHOLD,
        # ~1-2s) rather than being artificially held the 3.0s base.
        # OCR/both keep their anti-flicker floors (real ads with UI text).
        self.MIN_BLOCKING_DURATION_FLOOR_VLM = 0.5
        self.MIN_DURATION_RESET_GAP = 30.0  # seconds
        self.consecutive_ad_count = 0  # 0 = first ad, 1 = second, ...
        # 2 cycles ≈ 1.0s recovery (down from 4 = 2.0s). Tuned via the
        # tests/block_latency_harness.py rig to hit a sub-1.5s recover-to-
        # blocking-off latency. Risk: a single brief OCR miss during a real
        # ad ends blocking; in practice OCR's ~0.5s cycle on stable ad
        # overlays makes consecutive misses extremely rare.
        self.OCR_STOP_THRESHOLD = 2
        self.VLM_STOP_THRESHOLD = 2

        # OCR transience guard. OCR is normally fast-fire (1 frame → block
        # starts). But a single-frame OCR misread — a movie billboard with
        # "SKIP" visible briefly, a sign held by an actor reading
        # "Sponsored", a caption containing an ad-keyword word — should
        # NOT trigger blocking. Real ad UIs keep the keyword visible
        # continuously, so requiring 2 consecutive OCR-matched frames
        # before firing the first block rejects single-frame artifacts
        # at a ~500-1000ms latency cost on legitimate ads (OCR cadence
        # ~500ms/frame). Triangulation override: if VLM-ad-detected OR
        # ASR-confirms in the same window, fast-fire on 1 frame —
        # all three signals agreeing makes the artifact case
        # vanishingly rare. env-overridable.
        self.OCR_TRANSIENCE_MIN_FRAMES = int(
            os.environ.get('MINUS_OCR_TRANSIENCE_MIN_FRAMES', '2'))

        # OCR triangulation veto. OCR is normally authoritative for
        # stopping its own block, but when both VLM and ASR have firmly
        # disagreed for several seconds, OCR is probably misreading a
        # TV-show artifact (sign in a scene, news-ticker text, captions
        # mentioning brands). Force-stop the block in that case. Gated
        # on minimum block duration so the system has time to gather
        # VLM+ASR evidence before vetoing.
        self.OCR_TRIANGULATION_MIN_BLOCK_S = float(
            os.environ.get('MINUS_OCR_TRIANGULATION_MIN_BLOCK_S', '4.0'))
        # VLM "clearly says no-ad" threshold (sliding-window no_ad_ratio).
        # Higher than the start threshold (0.70) — we want HIGH confidence
        # before overriding OCR.
        self.OCR_TRIANGULATION_VLM_NOAD_RATIO = float(
            os.environ.get('MINUS_OCR_TRIANGULATION_VLM_NOAD_RATIO', '0.80'))
        # Sustained OCR overpowers the triangulation veto. If OCR has been
        # matching an ad keyword for ≥OCR_TRUSTED_DWELL_FRAMES consecutive
        # cycles, that's a STRONG indicator it really IS an ad — the ad UI
        # has been visible continuously for ~1.5s+ at OCR's ~500ms cadence.
        # In that case VLM saying no-ad (transient frame-classification
        # noise) and ASR saying veto (a quiet moment in the ad copy) are
        # NOT enough to override OCR. Prevents the veto from killing a
        # legitimate "Skip in 15" ad block when VLM/ASR briefly disagree.
        self.OCR_TRUSTED_DWELL_FRAMES = int(
            os.environ.get('MINUS_OCR_TRUSTED_DWELL_FRAMES', '3'))
        # Transition-frame hold bridges a BRIEF black/solid screen between
        # two ads in a real break so the overlay doesn't flicker. But a
        # dark/low-detail lofi music video reads as "uniform" forever, so
        # an uncapped hold freezes ocr_no_ad_count / vlm_no_ad_count and a
        # block never recovers (observed: a 46.9s VLM block held ~10s on
        # "WYS | Comforting You" after the ad ended). Real inter-ad gaps
        # are ≤~2s; cap the continuous hold so benign uniform content
        # can't defeat the no-ad stop. env-overridable.
        self.TRANSITION_HOLD_MAX_SECONDS = float(
            os.environ.get('MINUS_TRANSITION_HOLD_MAX', '3'))
        self.transition_hold_start = 0.0
        # Hard ceiling on one continuous block (any source). Legit ad
        # breaks — even multi-ad streaming pods — are well under this;
        # anything longer is a static weak-keyword false positive (see the
        # [SAFEGUARD] force-stop in _update_blocking_state). env-overridable.
        self.MAX_BLOCKING_DURATION = float(
            os.environ.get('MINUS_MAX_BLOCKING_DURATION', '150'))
        # Set when the MAX safeguard fires; suppresses re-blocking the
        # SAME frozen frame (stuck upstream stream) until the OCR text
        # meaningfully changes. Prevents the 150s→150s churn on a frozen
        # ad frame. (_safeguard_freeze_text = the frozen frame's OCR text.)
        self._safeguard_freeze_active = False
        self._safeguard_freeze_text = ''
        # EARLY frozen-stream detection. The MAX cap bounds a frozen
        # upstream stream to a single ~150s hold (no churn — the freeze
        # guard handles that), but a 150s hold still violates the
        # zero-multi-minute-holds goal and recurs (~daily). Signature is
        # unambiguous: OCR text byte-identical for tens of seconds WHILE
        # blocking, incl. a stuck "Skip in N" countdown. A real skippable
        # ad's countdown decrements every ≤3s, so its OCR text never
        # stays identical this long; real bumpers end well before 30s.
        # When OCR text is unchanged for FROZEN_EARLY_SECONDS while a
        # block is active, fire the SAME proven force-stop+freeze path as
        # the 150s cap — just earlier. Reuses _norm_alnum/difflib and the
        # validated _safeguard_freeze_* mechanism (only the trigger time
        # is new). env-overridable.
        self.FROZEN_EARLY_SECONDS = float(
            os.environ.get('MINUS_FROZEN_EARLY_SECONDS', '30'))
        self._ocr_text_stable_since = 0.0   # when OCR text last changed
        self._ocr_text_stable_norm = ''     # normalised text at that time
        self._ocr_text_frozen_for = 0.0     # seconds OCR text unchanged
        self.SKIP_DELAY_SECONDS = 4.5  # Wait 4s after ad starts before attempting skip (skip buttons rarely appear sooner)

        self._state_lock = threading.Lock()

        # Scene change detection
        self.prev_frame = None
        self.prev_frame_had_ad = False
        self.scene_skip_count = 0
        self.scene_change_threshold = self.config.scene_change_threshold
        self.max_scene_skip = 30  # Force OCR after this many consecutive skips

        # Static screen suppression - disable blocking for still ads
        # (e.g., paused video with ad, YouTube landing page with sponsored content)
        self.STATIC_TIME_THRESHOLD = 2.5  # Seconds of static screen to trigger suppression
        self.STATIC_OCR_THRESHOLD = 4     # OCR iterations without scene change
        self.DYNAMIC_COOLDOWN = self.config.dynamic_cooldown
        self.static_since_time = 0        # When screen became static (0 = not static)
        self.static_ocr_count = 0         # OCR iterations without scene change
        self.static_blocking_suppressed = False  # Currently suppressing due to static
        self.screen_became_dynamic_time = 0      # When screen went from static to dynamic

        # Strong-ad-signal override for static suppression. Some video ads
        # (Michelob "Skip MI 15", graphic banner ads, etc.) have so little
        # motion that the static detector triggers within 2.5s and the block
        # never gets to run. We can't lower the static threshold without
        # re-introducing pause-on-home-screen false positives. Instead, when
        # OCR matches keywords that ONLY appear in active video-ad UIs
        # (Skip Ad / Skip in / Ad N of M / Ad countdown / Ad with timestamp /
        # Visit advertiser), we treat the screen as "definitely a real ad"
        # and skip the suppressor entirely for STRONG_AD_HOLD_SECONDS.
        # Bare "Sponsored" / "Learn more" / "Shop now" remain weak signals —
        # those CAN appear on home screens / paused-on-ad tiles, so they
        # don't override suppression.
        self.STRONG_AD_KEYWORD_NAMES = frozenset({
            'skip ad', 'skip ads', 'skip in',
            'skip ad (fuzzy)', 'skip ad (fuzzy-spad)', 'skip ad (fuzzy-foad)',
            'video will play after ad',
            'visit advertiser', 'visitadvertiser',
            'ad X of Y', 'ad countdown',
            'ad with timestamp', 'ad with timestamp (cross-element)',
            # 'sponsored' promoted to STRONG per product decision (2026-05):
            # it should decisively trigger ad blocking. NOTE: this overrides
            # static-screen suppression, so a STATIC "Sponsored" promo/tile
            # (not on a detected home screen) can now block and hold until a
            # cap fires — bounded by FROZEN_EARLY_SECONDS (30s, frozen text)
            # and MAX_BLOCKING_DURATION (150s). This re-opens (in a bounded
            # form) the static-promo false-hold class that the weak-keyword
            # layer originally guarded; home-screen detection still suppresses
            # home/browse tiles. Revert by moving 'sponsored' back to WEAK.
            # 'sponsored (fuzzy)' carries the same strength — it is the same
            # signal read through an OCR misread ("Sponoed" on a live Netflix
            # static ad card, Aug 2026), and without STRONG status the static
            # suppressor keeps the card unblocked.
            'sponsored', 'sponsored (fuzzy)',
        })
        self.STRONG_AD_HOLD_SECONDS = 5.0
        self.last_strong_ad_time = 0.0
        # WEAK keywords legitimately appear on static home / promo /
        # masthead screens (YouTube "Sponsored · Learn more / Shop now"
        # tiles), NOT exclusively in active video-ad UIs. If the matched
        # set is ENTIRELY weak and no STRONG keyword was seen within
        # STRONG_AD_HOLD_SECONDS, the frame is suppressed (routed to
        # no-ad accounting so a block decays and cannot start/sustain).
        # Real video ads always also surface a strong keyword (skip in /
        # countdown / visit advertiser) within that window, and VLM
        # independently catches genuine ad video, so OCR can stay strict.
        # Generalised from the original bare-'sponsored'-only check after
        # a 150s VLM+OCR hold on a static "Learn more · Sponsored" promo
        # (the pair evaded the sponsored-only test). Names must match what
        # OCRProcess.check_ad_keywords emits.
        self.WEAK_AD_KEYWORD_NAMES = frozenset({
            # 'sponsored' moved to STRONG_AD_KEYWORD_NAMES (see above).
            'learn more', 'shop now', 'buy now',
            'shop now (fuzzy)', 'shop now (fuzzy-shan)',
        })
        # DEFINITIVE ad-UI keywords: the subset of STRONG that ONLY ever
        # appears inside an active video-ad overlay and is therefore
        # implausible as a single-frame OCR misread of show content. These
        # BYPASS the 2-frame transience guard and fire blocking on the FIRST
        # matched frame (saves ~one OCR cycle, ~0.8-1.5s, off detect latency —
        # the dominant ad-break activation tax). 'sponsored' is deliberately
        # EXCLUDED: it legitimately appears on home/promo tiles and as
        # show-content text, so it keeps the dwell. The transience guard's
        # cited artifact cases ("SKIP" on a billboard, "Sponsored" on a tile,
        # "BUY" in a caption) do NOT match any name below, so single-frame
        # fast-fire here reintroduces none of the FP risk the guard prevents.
        self.DEFINITIVE_AD_KEYWORD_NAMES = (
            self.STRONG_AD_KEYWORD_NAMES
            - frozenset({'sponsored', 'sponsored (fuzzy)'}))

        self.vlm_prev_frame = None
        self.vlm_prev_frame_had_ad = False
        self.vlm_scene_skip_count = 0
        self.vlm_max_scene_skip = 10  # Force VLM after this many consecutive skips

        # Screenshot manager (organizes into ads/, non_ads/, vlm_spastic/, static/ subdirs)
        self.screenshot_manager = ScreenshotManager(
            base_dir=Path(config.screenshot_dir),
            max_screenshots=config.max_screenshots
        )

        # Web UI state
        self.webui = None
        self.start_time = time.time()
        self.blocking_paused_until = 0  # Timestamp when pause expires
        # HDMI reconnect grace period: when the TV reconnects, blocking is
        # suppressed for HDMI_RECONNECT_GRACE_SECONDS so the user can grab the
        # remote and navigate without an overlay jumping in. The health monitor
        # sets hdmi_reconnect_time on reconnect; we compare against it each
        # cycle instead of persisting a "paused_until" to keep the two
        # suppression mechanisms independent.
        self.HDMI_RECONNECT_GRACE_SECONDS = 90.0
        self.hdmi_reconnect_time = 0.0
        # Ad's own countdown (seconds remaining) extracted from OCR, plus the
        # wall-clock time of the last update so the overlay can decay the bar
        # smoothly between OCR samples.
        self.ad_seconds_remaining = None
        self.ad_seconds_remaining_at = 0.0
        from collections import deque
        self.detection_history = deque(maxlen=50)  # Recent detections for web UI

        # System settings (loaded from ~/.minus_system_settings.json)
        self._system_settings = self._load_system_settings()

        # Remote control state
        self.fire_tv_setup = None
        self.fire_tv_controller = None
        self.roku_controller = None
        self._fire_tv_setup_thread = None

        # Night mode - automatic overnight YouTube playback for training data
        self.autonomous_mode = AutonomousMode()

        # IR transmitter (REI 8K HDMI switch). Constructor is hardware-free;
        # initialize() / first send() is what touches the PWM sysfs.
        self.ir_transmitter = IRTransmitter() if HAS_IR else None

        # Status LED strip (7× WS2812B on SPI0 MOSI). Constructor is hardware-
        # free; start() touches /dev/spidev0.0. If the user has the toggle
        # persisted to enabled, the thread is started near the end of __init__
        # so the "initializing" state shows until ad_blocker fires up.
        self.status_leds = (
            StatusLEDController() if HAS_STATUS_LEDS else None)
        if self.status_leds is not None:
            # Gate the strip on the HDMI-TX display being connected when
            # ``leds_require_display`` is set (default). State machine still
            # ticks normally — only the wire output is suppressed, so the
            # animation resumes seamlessly when the display comes back on.
            def _led_drive_predicate():
                if not self.leds_require_display:
                    return True
                return self.is_display_connected_live()
            self.status_leds.set_drive_predicate(_led_drive_predicate)

        # Skip opportunity state - CONSERVATIVE approach to avoid accidental pauses
        # Key principle: Only try to skip ONCE per ad. If it doesn't work, don't retry.
        self.auto_skip_enabled = True  # Enable auto-skip (fixed: now properly detects countdown)
        self.skip_available = False  # True when "Skip" button is ready (no countdown)
        self.skip_countdown = None   # Current countdown value (for tracking transitions)
        self.last_skip_countdown = None  # Previous countdown value (for detecting 1->0 transition)
        self.last_skip_text = None   # The detected skip text
        self.skip_attempted_this_ad = False  # Have we already tried to skip this ad?
        self.last_skip_attempt_time = 0  # When we last attempted a skip
        self.last_skip_success_time = 0  # When we last successfully skipped an ad
        self.SKIP_ATTEMPT_COOLDOWN = 10.0  # Don't try again for 10s after ANY attempt (prevents pause spam)
        # After a successful skip the skipped ad's end-card / transition
        # lingers briefly and still OCR-matches; during this window OCR ad
        # frames are routed to no-ad so the block decays and the
        # _unblock_after_skip reset isn't instantly undone.
        # 8s was TOO LONG: ad pods (skip ad 1 → ad 2 starts ~1-2s later)
        # are extremely common, and an 8s grace suppressed detection of
        # the NEXT ad for several seconds (observed: VLM flags ad 2 at
        # 100%/3 but AD BLOCKING STARTED withheld 2-3s by this grace).
        # 3s covers the typical skipped-ad end-card; the multi-minute
        # backstop the 8s was sized for is now redundant — the universal
        # MAX_BLOCKING_DURATION cap, weak-keyword suppression and the
        # transition-hold cap independently prevent long false holds.
        # env-overridable.
        self.SKIP_UNBLOCK_GRACE_SECONDS = float(
            os.environ.get('MINUS_SKIP_UNBLOCK_GRACE', '3'))

        # Accidental pause detection
        self.blocking_end_time = 0  # When blocking last ended
        self.PAUSE_DETECT_WINDOW = 1.5  # Window after skip to detect accidental pause
        self.accidental_pause_detected = False

        # Probe DRM output to auto-detect connector, plane, resolution, and audio device
        if config.drm_connector_id is None or config.drm_plane_id is None or config.output_width is None or config.audio_playback_device is None:
            logger.info("Probing DRM output for connected HDMI display...")
            drm_info = probe_drm_output()
            if drm_info['connector_id'] is not None:
                if config.drm_connector_id is None:
                    config.drm_connector_id = drm_info['connector_id']
                if config.drm_plane_id is None:
                    config.drm_plane_id = drm_info['plane_id']
                if config.output_width is None:
                    config.output_width = drm_info['width']
                if config.output_height is None:
                    config.output_height = drm_info['height']
                if config.audio_playback_device is None:
                    config.audio_playback_device = drm_info['audio_device']
                logger.info(f"DRM output: connector={config.drm_connector_id}, plane={config.drm_plane_id}, "
                           f"resolution={config.output_width}x{config.output_height}, "
                           f"audio={config.audio_playback_device}")
            else:
                # Fallback to defaults if no display detected
                logger.warning("No HDMI output detected, using defaults")
                config.drm_connector_id = config.drm_connector_id or 215
                config.drm_plane_id = config.drm_plane_id or 72
                config.output_width = config.output_width or 1920
                config.output_height = config.output_height or 1080
                config.audio_playback_device = config.audio_playback_device or 'hw:0,0'

        # Initialize OCR (with retry)
        self.ocr_disabled = False
        if config.no_ocr:
            logger.info("OCR disabled via --no-ocr flag")
            self.ocr_disabled = True
        elif HAS_OCR:
            det_model, rec_model, dict_path = self._find_model_paths()
            if det_model:
                # Use process-based OCR for hard timeout capability
                self.ocr = OCRProcess()
                if self.ocr.start():
                    logger.info("OCR process started (hard 1.2s timeout with keepalive)")
                else:
                    self.ocr = None
                    logger.error("OCR process failed to start - continuing without OCR")
            else:
                logger.warning("OCR model files not found - continuing without OCR")
        else:
            logger.warning("OCR module not available")
            self.ocr_disabled = True

        # Initialize ad blocker (manages display pipeline with input-selector)
        if HAS_ADBLOCKER:
            try:
                self.ad_blocker = AdBlocker(
                    connector_id=config.drm_connector_id,
                    plane_id=config.drm_plane_id,
                    minus_instance=self,
                    ustreamer_port=config.ustreamer_port,
                    output_width=config.output_width,
                    output_height=config.output_height
                )
                # Apply persisted greyscale-preview setting from disk
                if hasattr(self.ad_blocker, 'set_preview_grayscale'):
                    self.ad_blocker.set_preview_grayscale(self.greyscale_preview_enabled)
                # Apply persisted debug-overlay setting from disk
                if hasattr(self.ad_blocker, 'set_debug_overlay_enabled'):
                    self.ad_blocker.set_debug_overlay_enabled(self.debug_overlay_enabled)
                logger.info("AdBlocker initialized (instant input-selector switching)")
            except Exception as e:
                logger.exception(f"AdBlocker init failed: {e}")

        # Initialize VLM
        if config.no_vlm:
            logger.info("VLM disabled via --no-vlm flag")
        elif HAS_VLM:
            try:
                self.vlm = VLMProcess()
                logger.info("VLM process initialized (hard 2s timeout)")
            except Exception as e:
                logger.warning(f"VLM init failed: {e}")
                self.vlm = None

        # Initialize ASR (faster-whisper ad-content confirmation/veto) —
        # must be created BEFORE the audio passthrough so the tap can be
        # wired into the audio pipeline at construction time. The audio
        # tap is only created if faster-whisper is installed (otherwise
        # we leave the audio pipeline shape unchanged, no tee branch).
        self.asr_tap = None
        self.asr = None
        if HAS_ASR and is_asr_available():
            try:
                self.asr_tap = AudioASRTap()
                self.asr = ASRManager(self.asr_tap)
                # Honour the persisted on/off toggle. When off we still
                # construct the manager (so it can be re-enabled at
                # runtime via the web UI) but skip starting the worker.
                self.asr.enabled = bool(self._system_settings.get('asr_enabled', True))
                # ASRManager.get_status() exposes the engine + model name,
                # but it isn't ready yet at construction; read from the
                # manager attrs for the boot log line instead.
                logger.info(f"ASR initialized (faster-whisper "
                            f"{self.asr._model_name} + audio tap, "
                            f"enabled={self.asr.enabled})")
            except Exception as e:
                logger.warning(f"ASR init failed: {e} — running without ASR")
                self.asr_tap = None
                self.asr = None
        elif HAS_ASR:
            logger.info("ASR module loaded but faster-whisper not "
                        "available — running without ASR")

        # Initialize Audio passthrough
        if HAS_AUDIO:
            try:
                self.audio = AudioPassthrough(
                    capture_device=config.audio_capture_device,  # HDMI-RX audio (hw:4,0)
                    playback_device=config.audio_playback_device,  # HDMI-TX audio (auto-detected)
                    asr_tap=self.asr_tap  # Optional; None disables the tap branch entirely
                )
                # Link audio to ad_blocker for mute control
                if self.ad_blocker:
                    self.ad_blocker.set_audio(self.audio)
                logger.info(f"Audio passthrough initialized ({config.audio_capture_device} -> {config.audio_playback_device})")
            except Exception as e:
                logger.warning(f"Audio init failed: {e}")
                self.audio = None

        # Initialize Health Monitor
        if HAS_HEALTH:
            try:
                self.health_monitor = HealthMonitor(self, check_interval=5.0)
                self.health_monitor.on_hdmi_lost(self._on_hdmi_lost)
                self.health_monitor.on_hdmi_restored(self._on_hdmi_restored)
                self.health_monitor.on_ustreamer_stall(self._restart_ustreamer)
                self.health_monitor.on_video_pipeline_stall(self._on_video_pipeline_stall)
                self.health_monitor.on_vlm_failure(self._handle_vlm_failure)
                self.health_monitor.on_memory_critical(self._handle_memory_critical)
                self.health_monitor.on_format_change(self._on_format_change)
                logger.info("Health monitor initialized")
            except Exception as e:
                logger.warning(f"Health monitor init failed: {e}")
                self.health_monitor = None

        # Initialize System Notification overlay (for VLM status, etc.)
        self.system_notification = None
        if HAS_OVERLAY:
            try:
                self.system_notification = SystemNotification(ustreamer_port=config.ustreamer_port)
                logger.info("System notification overlay initialized")
            except Exception as e:
                logger.warning(f"System notification init failed: {e}")
                self.system_notification = None

    def _find_model_paths(self):
        """Find PaddleOCR model paths.
        Search order: local models dir, then configurable OCR_MODEL_DIR (env MINUS_OCR_MODEL_DIR).
        """
        search_paths = [
            Path(__file__).parent / 'models' / 'paddleocr',
            Path(OCR_MODEL_DIR),
        ]

        for base_path in search_paths:
            det_model = list(base_path.glob('ppocrv3_det_*.rknn'))
            rec_model = list(base_path.glob('ppocrv3_rec_*.rknn'))
            dict_path = base_path / 'ppocr_keys_v1.txt'

            if det_model and rec_model and dict_path.exists():
                logger.info(f"Found OCR models at: {base_path}")
                return str(det_model[0]), str(rec_model[0]), str(dict_path)

        return None, None, None

    # ===== Health Recovery Methods =====

    def _set_led_state(self, state):
        """Forward a state change to the status-LED strip. No-op if disabled
        or hardware missing. Errors are swallowed — LED issues must never
        crash core paths."""
        ctrl = getattr(self, 'status_leds', None)
        if ctrl is None or not ctrl.enabled:
            return
        try:
            ctrl.set_state(state)
        except Exception as e:
            logger.debug(f"[StatusLED] set_state({state}) failed: {e}")

    def _baseline_led_state(self):
        """Resolve what the LEDs should show when an acute state (blocking)
        ends. Priority high → low: no_signal, wifi_setup, paused,
        autonomous, idle. The first thing that's currently true wins.

        Used by ad_blocker.hide(), resume_blocking(), and the autonomous
        deactivate hook. error is intentionally NOT in this list — it's a
        manual / subsystem-failure override that the next legitimate state
        change will overwrite anyway.
        """
        # Prefer the health monitor's live `hdmi_signal` over the boolean we
        # toggle in _on_hdmi_lost / _on_hdmi_restored. The flag toggles only
        # on transitions, but the live status is always current.
        try:
            if self.health_monitor is not None:
                hs = self.health_monitor.get_status()
                if hs is not None and not hs.hdmi_signal:
                    return 'no_signal'
        except Exception:
            pass
        if getattr(self, '_hdmi_signal_lost', False):
            return 'no_signal'
        if self.wifi_manager is not None and getattr(
                self.wifi_manager, '_ap_mode_active', False):
            return 'wifi_setup'
        if self.is_blocking_paused():
            return 'paused'
        if getattr(self, '_autonomous_was_active', False):
            return 'autonomous'
        return 'idle'

    def _on_hdmi_lost(self):
        """Handle HDMI signal loss."""
        logger.warning("[Recovery] HDMI signal lost - showing NO SIGNAL display")
        # Pause detection workers to prevent memory leak from repeated snapshot timeouts
        self._hdmi_signal_lost = True
        self._set_led_state('no_signal')
        # Switch to standalone NO SIGNAL display (doesn't depend on ustreamer)
        if self.ad_blocker:
            self.ad_blocker.start_no_signal_mode()
        if self.audio:
            # Pause watchdog to prevent restart loops (source is unavailable)
            self.audio.pause_watchdog()
            self.audio.mute()

    def _on_hdmi_restored(self):
        """Handle HDMI signal restoration."""
        logger.info("[Recovery] HDMI signal restored - showing loading screen")

        # Resume detection workers
        self._hdmi_signal_lost = False

        # Set flag to prevent main loop from interfering with recovery
        self._hdmi_recovery_in_progress = True

        self._set_led_state('initializing')

        try:
            # Switch to loading display while we restart everything
            if self.ad_blocker:
                self.ad_blocker.start_loading_mode()

            # Restart ustreamer first to pick up new signal
            self._restart_ustreamer()

            # Wait for ustreamer to be fully ready before restarting video pipeline
            time.sleep(2)

            # Start the video pipeline (will transition from loading to live).
            # If HDMI-OUT (the TV) is still disconnected, this will fail —
            # kmssink can't open a DRM plane on a disconnected connector.
            # We MUST check the return value: previously we logged "recovery
            # complete" regardless and left self.display_connected stuck at
            # True, which prevented the display retry loop from ever firing
            # — the pipeline then stayed dead until the next service restart.
            pipeline_ok = False
            if self.ad_blocker:
                logger.info("[Recovery] Starting video pipeline...")
                pipeline_ok = bool(self.ad_blocker.start())

            if pipeline_ok:
                self.display_connected = True
                self.display_error = None
                if self.audio:
                    # Resume watchdog and restart pipeline (source is available again)
                    self.audio.resume_watchdog()
                    self.audio.unmute()
                    # The pipeline start above did a modeset; a PCM that kept
                    # streaming (or opened) during link training has a dead
                    # TX audio lane — re-prepare it once things settle.
                    self._schedule_audio_reprepare(reason="hdmi recovery")
                logger.info("[Recovery] HDMI recovery complete")
                self._set_led_state('idle')
            else:
                # Pipeline start failed — almost always because HDMI-OUT is
                # still disconnected. Mark display as down and hand off to
                # the retry loop, which will keep trying every 7s and pick up
                # the moment HDMI-OUT comes back.
                logger.warning(
                    "[Recovery] Display pipeline start failed (HDMI-OUT likely "
                    "disconnected) — marking display down and arming retry loop"
                )
                self.display_connected = False
                self.display_error = (
                    "Display output not available. Check HDMI-TX connection to TV/monitor."
                )
                self._start_display_retry_loop()
                # Audio stays muted/paused; resume_watchdog will fire when the
                # retry loop succeeds (start_display_pipeline → main resume path).
                # LED state stays at no_signal — the display gate keeps the
                # strip dark while the retry loop runs.
        finally:
            self._hdmi_recovery_in_progress = False

    def _on_video_pipeline_stall(self):
        """Handle video pipeline stall detected by health monitor."""
        # Skip if display is disconnected (retry loop handles reconnection)
        if not self.display_connected:
            logger.debug("[Recovery] Skipping pipeline stall recovery - display disconnected")
            return

        logger.warning("[Recovery] Video pipeline stall detected - showing loading and restarting")
        if self.ad_blocker:
            # Show loading while we restart the pipeline
            self.ad_blocker.start_loading_mode()
            # Start will transition from loading to live
            self.ad_blocker.start()

    def _on_format_change(self, new_format: str):
        """Handle HDMI format/resolution change detected by health monitor.

        This enables seamless switching between different streaming devices
        (FireTV, Roku, AppleTV, etc.) without manual intervention.

        Args:
            new_format: New format string like 'NV24@1280x720'
        """
        logger.warning(f"[Recovery] HDMI format changed to {new_format} - restarting ustreamer")

        # Skip if display is disconnected (retry loop handles reconnection)
        if not self.display_connected:
            logger.debug("[Recovery] Skipping format change recovery - display disconnected")
            return

        # Show loading screen while we adapt to new format
        if self.ad_blocker:
            self.ad_blocker.start_loading_mode()

        # Mute audio during transition
        if self.audio:
            self.audio.mute()

        # Restart ustreamer with the new format
        self._restart_ustreamer()

        # Wait for ustreamer to be ready
        time.sleep(2)

        # Restart video pipeline to reconnect
        if self.ad_blocker:
            logger.info("[Recovery] Restarting video pipeline for new format...")
            self.ad_blocker.start()

        # Restore audio
        if self.audio:
            self.audio.unmute()

        logger.info(f"[Recovery] Format change complete - now running at {new_format}")

    def _restart_ustreamer(self):
        """Restart ustreamer process."""
        # Skip if display is disconnected (retry loop handles reconnection)
        if not self.display_connected:
            logger.debug("[Recovery] Skipping ustreamer restart - display disconnected")
            return

        logger.warning("[Recovery] Restarting ustreamer...")
        try:
            # Kill existing ustreamer
            if self.ustreamer_process:
                self.ustreamer_process.terminate()
                try:
                    self.ustreamer_process.wait(timeout=2)
                except:
                    self.ustreamer_process.kill()

            subprocess.run(['pkill', '-9', 'ustreamer'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Also kill anything on the port
            subprocess.run(['fuser', '-k', f'{self.config.ustreamer_port}/tcp'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)

            # Initialize V4L2 device with proper DV timings for the new source
            # This ensures proper format negotiation when switching between devices
            self._init_v4l2_device()

            # Re-probe the device in case format changed
            device_info = probe_v4l2_device(self.device)
            detected_format = device_info.get('ustreamer_format') or getattr(self, '_detected_format', 'NV12')
            width = device_info.get('width') or 3840
            height = device_info.get('height') or 2160
            resolution = f"{width}x{height}"

            video_format = detected_format

            # Update stored format info
            self._detected_format = video_format
            self._detected_resolution = resolution

            logger.info(f"[Recovery] Source outputs {detected_format} at {resolution}")

            # Restart with patched ustreamer using MPP hardware encoder
            # The patched ustreamer handles NV24→NV12 conversion internally
            port = self.config.ustreamer_port
            ustreamer_cmd = [
                USTREAMER_PATH,
                f'--device={self.device}',
                f'--format={video_format}',
                f'--resolution={resolution}',
                '--persistent',
                f'--port={port}',
                '--host=0.0.0.0',          # Bind to all interfaces for remote access
                '--encoder=mpp-jpeg',
                '--encode-scale=passthrough',  # No scaling, use source resolution directly
                '--quality=80',
                '--workers=4',              # 4 parallel MPP encoders
                '--buffers=5',
                '--tcp-nodelay',           # Disable Nagle's algorithm for smoother streaming
            ]

            logger.info(f"[Recovery] Starting ustreamer: {' '.join(ustreamer_cmd)}")

            self.ustreamer_process = subprocess.Popen(
                ustreamer_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            time.sleep(2)

            if self.ustreamer_process.poll() is None:
                logger.info("[Recovery] ustreamer restarted successfully")
                # Also restart video pipeline to reconnect to new ustreamer
                if self.ad_blocker:
                    logger.info("[Recovery] Also restarting video pipeline...")
                    self.ad_blocker.restart()
            else:
                logger.error("[Recovery] ustreamer failed to restart")

        except Exception as e:
            logger.error(f"[Recovery] Error restarting ustreamer: {e}")

    def _handle_vlm_failure(self):
        """Handle VLM consecutive failures - degrade to OCR-only mode."""
        if not self.vlm_disabled:
            self.vlm_disabled = True
            logger.warning("[Recovery] VLM disabled due to failures - running OCR-only mode")
            # Try to restart VLM in background
            threading.Thread(target=self._try_restart_vlm, daemon=True).start()

    def _try_restart_vlm(self):
        """Attempt to restart VLM after a delay."""
        time.sleep(30)  # Wait 30 seconds before retry

        if self.vlm and not self.vlm.is_ready:
            logger.info("[Recovery] Attempting VLM restart...")
            try:
                self.vlm.release()
                time.sleep(2)
                if self.vlm.load_model():
                    self.vlm_disabled = False
                    self.vlm_consecutive_timeouts = 0
                    logger.info("[Recovery] VLM restarted successfully")
                else:
                    logger.warning("[Recovery] VLM restart failed - staying in OCR-only mode")
            except Exception as e:
                logger.error(f"[Recovery] VLM restart error: {e}")

    def _handle_memory_critical(self):
        """Handle critical memory usage."""
        logger.warning("[Recovery] Critical memory usage - cleaning up")
        # Force garbage collection
        import gc
        gc.collect()

        # Clear frame buffers
        self.prev_frame = None
        self.vlm_prev_frame = None

        # NOTE: Screenshot cleanup removed - we want to keep ALL screenshots for training data
        # The memory issue should be addressed by fixing actual memory leaks, not deleting training data

    # ===== VLM Control Methods =====

    def disable_vlm(self) -> dict:
        """
        Disable VLM and unload the model from the Axera NPU.

        Returns:
            dict with success status and message
        """
        if not self.vlm:
            return {'success': False, 'error': 'VLM not initialized'}

        if self.vlm_disabled and not self.vlm.is_ready:
            return {'success': True, 'message': 'VLM already disabled'}

        try:
            # Show unloading notification
            if hasattr(self, 'system_notification') and self.system_notification:
                self.system_notification.show_vlm_unloading()

            logger.info("[VLM] Disabling VLM and unloading model...")
            self.vlm_disabled = True

            # Release the model and free NPU resources
            self.vlm.release()

            # Show disabled notification
            if hasattr(self, 'system_notification') and self.system_notification:
                self.system_notification.show_vlm_disabled()

            logger.info("[VLM] VLM disabled and model unloaded")
            return {'success': True, 'message': 'VLM disabled and model unloaded from NPU'}
        except Exception as e:
            logger.error(f"[VLM] Error disabling VLM: {e}")
            return {'success': False, 'error': str(e)}

    def enable_vlm(self) -> dict:
        """
        Enable VLM and reload the model to the Axera NPU.

        Returns:
            dict with success status and message
        """
        if not self.vlm:
            return {'success': False, 'error': 'VLM not initialized'}

        if not self.vlm_disabled and self.vlm.is_ready:
            return {'success': True, 'message': 'VLM already enabled'}

        try:
            # Show loading notification
            if hasattr(self, 'system_notification') and self.system_notification:
                self.system_notification.show_vlm_loading()

            logger.info("[VLM] Enabling VLM and loading model...")

            # Load the model
            if self.vlm.load_model():
                self.vlm_disabled = False
                self.vlm_consecutive_timeouts = 0

                # Show ready notification
                if hasattr(self, 'system_notification') and self.system_notification:
                    self.system_notification.show_vlm_ready()

                logger.info("[VLM] VLM enabled and model loaded")
                return {'success': True, 'message': 'VLM enabled and model loaded to NPU'}
            else:
                # Show failed notification
                if hasattr(self, 'system_notification') and self.system_notification:
                    self.system_notification.show_vlm_failed()

                return {'success': False, 'error': 'Failed to load VLM model'}
        except Exception as e:
            logger.error(f"[VLM] Error enabling VLM: {e}")
            if hasattr(self, 'system_notification') and self.system_notification:
                self.system_notification.show_vlm_failed()
            return {'success': False, 'error': str(e)}

    def get_vlm_status(self) -> dict:
        """Get detailed VLM status."""
        return {
            'initialized': self.vlm is not None,
            'disabled': self.vlm_disabled,
            'model_loaded': self.vlm.is_ready if self.vlm else False,
            'consecutive_timeouts': self.vlm_consecutive_timeouts,
            'frame_count': self.vlm_frame_count,
        }

    # ===== Device Setup Methods =====

    def _start_device_setup_delayed(self, delay_seconds: float = 15.0):
        """Start device setup after a delay (to let display stabilize first).

        This is device-aware - checks the configured device type and starts
        the appropriate setup flow:
        - Fire TV / Google TV: Start FireTVSetupManager with ADB flow
        - Roku: Auto-connect using saved IP (no setup overlay needed)
        - Other: Skip setup entirely
        """
        def delayed_start():
            logger.info(f"[DeviceSetup] Waiting {delay_seconds}s before starting device setup...")
            time.sleep(delay_seconds)

            if not self.running:
                return

            # Check configured device type
            try:
                from src.device_config import get_device_config_manager
                device_manager = get_device_config_manager()
                config = device_manager.get_config()
                device_type = config.get('device_type', 'none')
                device_ip = config.get('device_ip', '')
                setup_complete = config.get('setup_complete', False)

                logger.info(f"[DeviceSetup] Configured device: {device_type}, IP: {device_ip}, setup_complete: {setup_complete}")
            except Exception as e:
                logger.warning(f"[DeviceSetup] Could not load device config: {e}")
                device_type = 'none'
                device_ip = ''
                setup_complete = False

            # Handle based on device type
            if device_type in ('fire_tv', 'google_tv'):
                self._start_fire_tv_setup(device_ip if setup_complete else None, device_type=device_type)
            elif device_type == 'roku':
                self._start_roku_connection(device_ip if setup_complete else None)
            else:
                logger.info(f"[DeviceSetup] No remote setup needed for device type: {device_type}")

        self._fire_tv_setup_thread = threading.Thread(
            target=delayed_start,
            daemon=True,
            name="DeviceSetupDelay"
        )
        self._fire_tv_setup_thread.start()

    def _start_fire_tv_setup(self, saved_ip: str = None, device_type: str = 'fire_tv'):
        """Start Fire TV / Google TV setup flow."""
        if not HAS_FIRE_TV:
            logger.info("[FireTV] Fire TV module not available")
            return

        device_name = "Google TV" if device_type == 'google_tv' else "Fire TV"
        logger.info(f"[{device_name}] Initializing setup manager...")
        self.fire_tv_setup = FireTVSetupManager(
            ad_blocker=self.ad_blocker,
            ocr_worker=self.ocr,
            ustreamer_port=self.config.ustreamer_port,
            device_type=device_type
        )

        # Set callbacks
        self.fire_tv_setup.set_callbacks(
            on_state_change=self._on_fire_tv_state_change,
            on_connected=self._on_fire_tv_connected
        )

        # If we have a saved IP, try to connect directly first
        if saved_ip:
            logger.info(f"[FireTV] Attempting direct connection to saved IP: {saved_ip}")
            # The setup manager will try this IP first before scanning
            self.fire_tv_setup.set_preferred_ip(saved_ip)

        # Start the setup flow
        self.fire_tv_setup.start_setup()

    def _start_roku_connection(self, saved_ip: str = None):
        """Start Roku connection with appropriate overlay notifications.

        Connection flow:
        1. Try saved IP if available
        2. If that fails or no saved IP, scan for Roku devices
        3. Connect to first discovered Roku
        4. Update config with working IP for next restart
        """
        try:
            from src.roku import RokuController
            from src.overlay import RokuNotification
            from src.device_config import get_device_config_manager

            # Initialize notification overlay
            roku_notification = RokuNotification(ustreamer_port=self.config.ustreamer_port)
            self.roku_controller = RokuController()

            # Persist the Roku's IP whenever auto-reconnect follows it to a
            # new DHCP address, so the next service restart connects directly.
            self.roku_controller.set_ip_change_callback(self._persist_roku_ip)

            # Wire autonomous mode immediately (not only on success):
            # every autonomous action gates on is_connected(), so if the
            # initial connect fails and the background reconnect succeeds
            # later, autonomous mode starts working without user help.
            if self.autonomous_mode:
                self.autonomous_mode.set_device_controller(self.roku_controller, 'roku')
                logger.info("[AutonomousMode] Roku controller wired")

            connected = False
            connected_ip = None

            # Step 1: Try saved IP first
            if saved_ip:
                roku_notification.show_connecting(saved_ip)
                logger.info(f"[Roku] Connecting to saved Roku at {saved_ip}...")

                if self.roku_controller.connect(saved_ip):
                    connected = True
                    connected_ip = saved_ip
                    logger.info(f"[Roku] Connected to saved IP {saved_ip}")
                else:
                    logger.warning(f"[Roku] Failed to connect to saved IP {saved_ip}, will scan...")

            # Step 2: If saved IP failed or not available, scan for Roku devices
            if not connected:
                roku_notification.show_scanning()
                logger.info("[Roku] Scanning for Roku devices...")

                devices = RokuController.discover_devices(timeout=8)
                if devices:
                    # Try to connect to each discovered device
                    for device in devices:
                        device_ip = device.get('ip')
                        if device_ip:
                            logger.info(f"[Roku] Found Roku at {device_ip}, attempting connection...")
                            if self.roku_controller.connect(device_ip):
                                connected = True
                                connected_ip = device_ip
                                logger.info(f"[Roku] Connected via discovery to {device_ip}")
                                break
                            else:
                                logger.warning(f"[Roku] Failed to connect to discovered Roku at {device_ip}")

            # Step 3: Handle connection result
            if connected and connected_ip:
                device_info = self.roku_controller.get_device_info()
                device_name = device_info.get('name', 'Roku') if device_info else 'Roku'
                model = device_info.get('model', '') if device_info else ''
                full_name = f"{device_name} {model}".strip()

                logger.info(f"[Roku] Connected to {full_name} at {connected_ip}")

                # Update saved config with working IP
                try:
                    manager = get_device_config_manager()
                    if manager.config.device_ip != connected_ip:
                        manager.set_device_ip(connected_ip)
                        manager.set_setup_complete(True)
                        logger.info(f"[Roku] Updated saved IP to {connected_ip}")
                except Exception as e:
                    logger.warning(f"[Roku] Could not update config: {e}")

                # Check if Roku is in limited mode
                control_mode = self.roku_controller.check_control_mode()
                if control_mode == 'limited':
                    logger.warning("[Roku] Roku is in Limited mode - commands won't work!")
                    # Show brief connected message, then setup instructions
                    roku_notification.show_connected(full_name)
                    time.sleep(3)
                    roku_notification.show_limited_mode()

                    # Start background thread to auto-hide when mode changes
                    def monitor_control_mode():
                        """Poll for control mode change and hide overlay."""
                        check_interval = 10  # Check every 10 seconds
                        max_checks = 30  # Max 5 minutes (30 x 10s)
                        for _ in range(max_checks):
                            time.sleep(check_interval)
                            if not self.roku_controller or not self.roku_controller.is_connected():
                                break
                            mode = self.roku_controller.check_control_mode()
                            if mode == 'full':
                                logger.info("[Roku] Control mode changed to FULL - hiding setup overlay")
                                roku_notification.hide()
                                roku_notification.show_connected(f"{full_name} - Ready!")
                                break

                    threading.Thread(target=monitor_control_mode, daemon=True, name="Roku-ModeMonitor").start()
                else:
                    roku_notification.show_connected(full_name)
            else:
                logger.warning("[Roku] Could not connect to any Roku device")
                roku_notification.show_failed("No Roku found")
                # Keep trying in the background: the Roku may be powered
                # off or mid-reboot at service start. The reconnect loop
                # retries the saved IP every 10s and periodically rescans
                # the network; a user disconnect via the web UI stops it.
                self.roku_controller.start_monitoring(saved_ip)
                logger.info("[Roku] Background reconnect armed" +
                            (f" (saved IP {saved_ip})" if saved_ip else " (will rescan network)"))

        except Exception as e:
            logger.error(f"[Roku] Error starting Roku connection: {e}")

    def _persist_roku_ip(self, new_ip: str):
        """Persist the Roku's IP after auto-reconnect followed it to a new address."""
        try:
            from src.device_config import get_device_config_manager
            manager = get_device_config_manager()
            if manager.config.device_ip != new_ip:
                manager.set_device_ip(new_ip)
                manager.set_setup_complete(True)
                logger.info(f"[Roku] Saved new Roku IP {new_ip}")
        except Exception as e:
            logger.warning(f"[Roku] Could not persist new IP {new_ip}: {e}")

    def _on_fire_tv_state_change(self, new_state: str):
        """Handle Fire TV setup state changes."""
        logger.info(f"[FireTV] State changed to: {new_state}")

        # If we're waiting for auth, hook into OCR to detect the dialog
        if new_state == FireTVSetupManager.STATE_WAITING_AUTH:
            logger.info("[FireTV] Waiting for ADB authorization - OCR will detect dialog")

    def _on_fire_tv_connected(self, device_info: dict):
        """Handle successful Fire TV connection."""
        self.fire_tv_controller = self.fire_tv_setup.get_controller()

        manufacturer = device_info.get('manufacturer', 'Fire TV')
        model = device_info.get('model', '')
        ip = device_info.get('ip', '')

        logger.info(f"[FireTV] Connected to {manufacturer} {model} at {ip}")
        logger.info("[FireTV] Ad skipping enabled - will send skip commands during ads")

        # Add to detection history
        self.add_detection('FireTV', [f"Connected to {manufacturer} {model}"])

        # Connect autonomous mode to Fire TV controller
        if self.autonomous_mode:
            self.autonomous_mode.set_device_controller(self.fire_tv_controller, 'fire_tv')
            logger.info("[AutonomousMode] Fire TV controller connected")

    def _check_ocr_for_fire_tv_dialog(self, ocr_results: list) -> bool:
        """
        Check OCR results for Fire TV ADB auth dialog.

        This is called from the OCR worker when Fire TV is waiting for authorization.
        If we detect the auth dialog, we can provide better guidance.

        Returns:
            True if auth dialog detected
        """
        if not self.fire_tv_setup:
            return False

        if self.fire_tv_setup.state != FireTVSetupManager.STATE_WAITING_AUTH:
            return False

        return self.fire_tv_setup.check_for_auth_dialog(ocr_results)

    # ===== Display Recovery Methods =====

    def _start_display_retry_loop(self):
        """Start a background thread that retries display connection.

        This only retries the display pipeline (kmssink), not ustreamer.
        ustreamer is kept running for web preview and ML detection.
        """
        if self._display_retry_thread and self._display_retry_thread.is_alive():
            logger.debug("Display retry loop already running")
            return

        def retry_loop():
            logger.info(f"[Display] Starting retry loop (interval: {self._display_retry_interval}s)")

            while self.running and not self.display_connected:
                time.sleep(self._display_retry_interval)

                if not self.running:
                    break

                if self.display_connected:
                    logger.info("[Display] Display connected by another thread, stopping retry loop")
                    break

                logger.info("[Display] Attempting to reconnect display pipeline...")

                try:
                    # Re-probe DRM to detect if display was connected since startup
                    drm_info = probe_drm_output()
                    if drm_info['connector_id'] is not None:
                        # Check if DRM settings changed (display connected after startup)
                        if (drm_info['connector_id'] != self.config.drm_connector_id or
                            drm_info['plane_id'] != self.config.drm_plane_id):
                            logger.info(f"[Display] DRM output changed: connector {self.config.drm_connector_id} -> {drm_info['connector_id']}, "
                                       f"plane {self.config.drm_plane_id} -> {drm_info['plane_id']}")
                            # Update config
                            self.config.drm_connector_id = drm_info['connector_id']
                            self.config.drm_plane_id = drm_info['plane_id']
                            self.config.output_width = drm_info['width']
                            self.config.output_height = drm_info['height']
                            self.config.audio_playback_device = drm_info['audio_device']

                            # Update ad_blocker with new DRM settings
                            if self.ad_blocker:
                                self.ad_blocker.connector_id = drm_info['connector_id']
                                self.ad_blocker.plane_id = drm_info['plane_id']
                                self.ad_blocker.output_width = drm_info['width']
                                self.ad_blocker.output_height = drm_info['height']
                                # Reinitialize pipeline with new settings
                                self.ad_blocker._init_pipeline()
                                logger.info("[Display] Pipeline reinitialized with new DRM settings")

                            # Update audio playback device to match new HDMI output
                            if self.audio and drm_info['audio_device'] != self.audio.playback_device:
                                logger.info(f"[Display] Audio device changed: {self.audio.playback_device} -> {drm_info['audio_device']}")
                                self.audio.stop()
                                self.audio.playback_device = drm_info['audio_device']
                    else:
                        logger.debug("[Display] No HDMI output detected yet")
                        self.display_error = "Display output not available. Check HDMI-TX connection to TV/monitor."
                        continue

                    # Check if ustreamer needs to be restarted
                    if not self._is_ustreamer_running():
                        logger.warning("[Display] ustreamer not running, restarting...")
                        if not self.start_ustreamer():
                            logger.error("[Display] Failed to restart ustreamer")
                            continue

                    # Only retry the display pipeline (not ustreamer)
                    if self.start_display_pipeline():
                        self.display_connected = True
                        self.display_error = None
                        logger.info("[Display] Display pipeline reconnected successfully!")

                        # Show system ready notification if overlay available
                        if self.system_notification:
                            self.system_notification.show_system_ready()
                        break
                    else:
                        logger.warning("[Display] Display pipeline reconnect failed, will retry...")
                        self.display_error = "Display output not available. Check HDMI-TX connection to TV/monitor."
                except Exception as e:
                    logger.error(f"[Display] Reconnect attempt failed with exception: {e}")
                    self.display_error = f"Display error: {e}"

            logger.info("[Display] Retry loop ended")

        self._display_retry_thread = threading.Thread(
            target=retry_loop,
            daemon=True,
            name="DisplayRetryLoop"
        )
        self._display_retry_thread.start()

    def _stop_display_retry_loop(self):
        """Stop the display retry loop if running."""
        # The loop checks self.running and self.display_connected,
        # so setting display_connected = True will stop it
        if self._display_retry_thread and self._display_retry_thread.is_alive():
            logger.info("[Display] Stopping retry loop...")
            # Thread will exit on next iteration

    def try_skip_ad(self):
        """Attempt to skip ad on the connected streaming device.

        Works with Fire TV, Roku, and Google TV — uses whichever is connected.
        For Fire TV/Google TV: uses skip_ad() (ADB select command).
        For Roku: sends 'select' via ECP (same effect as pressing OK on remote).
        """
        device_type = self._get_configured_device_type()

        # Try Fire TV / Google TV (has dedicated skip_ad method)
        if device_type in ('fire_tv', 'google_tv'):
            if self.fire_tv_controller and self.fire_tv_controller.is_connected():
                try:
                    result = self.fire_tv_controller.skip_ad()
                    if result:
                        logger.info(f"[{device_type.upper()}] Sent skip command")
                    return result
                except Exception as e:
                    logger.warning(f"[{device_type.upper()}] Skip command failed: {e}")
            return False

        # Try Roku (send 'select' which presses the skip button)
        if device_type == 'roku':
            if self.roku_controller and self.roku_controller.is_connected():
                try:
                    result = self.roku_controller.send_command('select')
                    if result:
                        logger.info("[ROKU] Sent skip command (select)")
                    return result
                except Exception as e:
                    logger.warning(f"[ROKU] Skip command failed: {e}")
            return False

        # No device connected
        logger.debug("[SKIP] No streaming device connected for skip")
        return False

    # ===== Web UI Methods =====

    def pause_blocking(self, duration_seconds: int = 120):
        """Pause ad blocking for specified duration.

        Special case — user paused during a VLM-only block: routes to
        `_handle_vlm_false_positive_feedback` which (a) saves the
        VLM-triggering frame (not the current frame) as non_ad training
        data and (b) sets a 5-min VLM-specific cooldown. See the
        VLM_FALSE_POSITIVE_COOLDOWN block in __init__.
        """
        # Snapshot whether this is a VLM-only block BEFORE clearing state.
        # Use the active blocking_source — if OCR is also confirming
        # ("both"), the block isn't a VLM-alone misclassification, so the
        # standard pause-and-save-current-frame behaviour applies.
        was_vlm_only_block = (
            self.ad_detected and self.blocking_source == "vlm"
        )

        with self._state_lock:
            self.blocking_paused_until = time.time() + duration_seconds
            logger.info(f"[WebUI] Blocking paused for {duration_seconds}s")

        if was_vlm_only_block:
            # VLM-misclassification path: save the trigger frame + start
            # 5-min VLM cooldown.
            self._handle_vlm_false_positive_feedback()
        elif self.frame_capture:
            # Default path: save the current frame as non_ad training data
            # (this is what the user is looking at and signalling "not an ad").
            frame = self.frame_capture.capture()
            self.screenshot_manager.save_non_ad_screenshot(frame)

        # Immediately hide blocking overlay and unmute
        if self.ad_blocker:
            self.ad_blocker.hide()
        if self.audio:
            self.audio.unmute()

        # ad_blocker.hide() will have set state=idle; override with paused
        # so the LED visually distinguishes "user-paused" from "running clean".
        self._set_led_state('paused')

    def _handle_vlm_false_positive_feedback(self):
        """User paused during a VLM-only block — treat as explicit
        misclassification feedback.

        Two actions:
          1. Save the frame that most-recently caused VLM to verdict AD
             (cached in `last_vlm_ad_frame` by the VLM dispatch loop) to
             screenshots/non_ads/. This becomes training data showing
             content that should NOT be classified as an ad.
          2. Set `vlm_paused_until = now + VLM_FALSE_POSITIVE_COOLDOWN`
             (5 min). The VLM dispatch loop skips inference entirely while
             paused, so the same misclassified content can't immediately
             re-trigger if the user resumes blocking before the cooldown
             expires. OCR keeps running normally.

        Falls back to the current frame if no VLM-AD frame was cached
        (vanishingly rare — would mean the block fired without any
        is_ad=True VLM verdict, which shouldn't happen).
        """
        now = time.time()
        cooldown_seconds = self.VLM_FALSE_POSITIVE_COOLDOWN

        with self._state_lock:
            self.vlm_paused_until = now + cooldown_seconds
            # Clear VLM state so a fresh evaluation happens after cooldown
            self.vlm_ad_detected = False
            self.vlm_decision_history.clear()
            self.vlm_no_ad_count = 0
            # Snapshot the frame we'll save (under the lock so the
            # dispatch loop can't overwrite it mid-read).
            frame_to_save = self.last_vlm_ad_frame
            # frame_age is only meaningful if we actually have a timestamp.
            # Tests can seed `last_vlm_ad_frame` without setting the time,
            # which would produce a bogus 56-year "age" in the log line.
            frame_age = (
                now - self.last_vlm_ad_frame_time
                if frame_to_save is not None and self.last_vlm_ad_frame_time > 0
                else None
            )

        logger.warning(
            f"[Feedback] User paused during VLM-only block — treating "
            f"as misclassification. VLM blocking paused for "
            f"{cooldown_seconds:.0f}s ({cooldown_seconds/60:.0f} min)."
        )

        if frame_to_save is not None:
            self.screenshot_manager.save_non_ad_screenshot(frame_to_save)
            age_str = f"{frame_age:.1f}s" if frame_age is not None else "unknown age"
            logger.info(
                f"[Feedback] Saved VLM-triggering frame as non_ad "
                f"(frame was {age_str} at pause time)"
            )
        elif self.frame_capture:
            # Fallback: no cached frame, use current. Shouldn't normally
            # happen — log it so we notice if it does.
            logger.warning(
                "[Feedback] No cached VLM-trigger frame — falling back "
                "to current frame for non_ad training capture"
            )
            frame = self.frame_capture.capture()
            if frame is not None:
                self.screenshot_manager.save_non_ad_screenshot(frame)

    def is_vlm_user_paused(self) -> bool:
        """True if VLM inference is on a user-feedback cooldown."""
        return time.time() < self.vlm_paused_until

    def get_vlm_pause_remaining(self) -> int:
        """Seconds remaining in the VLM user-feedback cooldown."""
        remaining = self.vlm_paused_until - time.time()
        return max(0, int(remaining))

    def _display_source(self):
        """Source string for the overlay header: the base blocking_source
        plus an '+asr' suffix when ASR confirmed the active block →
        ocr / ocr+asr / vlm / vlm+asr / both / both+asr. blocking_source
        itself stays the base (ocr/vlm/both) so all stop-logic equality
        checks are unaffected."""
        src = self.blocking_source
        if src and self.blocking_asr_confirmed and src in ('ocr', 'vlm', 'both'):
            return src + '+asr'
        return src

    _SOURCE_LABELS = {
        'ocr': 'OCR', 'vlm': 'VLM', 'both': 'OCR+VLM',
        'ocr+asr': 'OCR+ASR', 'vlm+asr': 'VLM+ASR', 'both+asr': 'OCR+VLM+ASR',
    }

    def _display_source_label(self) -> str:
        """Human-readable label for logs / UI, e.g. 'OCR+VLM+ASR'."""
        src = self._display_source() or ''
        return self._SOURCE_LABELS.get(src, src.upper())

    def _asr_verdict(self) -> str:
        """Return ASR's current verdict on whether audio sounds like an ad.

        Returns one of:
          'confirm' — marketing language was heard in the rolling 8s window
          'veto'    — clear show dialog with no markers; suppress VLM-alone
          'unknown' — no ASR signal yet (cold start, music only, ASR disabled)

        Always returns 'unknown' when ASR is unavailable so blocking
        decisions degrade gracefully on installs without faster-whisper.
        """
        if self.asr is None:
            return 'unknown'
        try:
            return self.asr.verdict()
        except Exception as e:
            logger.debug(f"ASR verdict error: {e}")
            return 'unknown'

    def resume_blocking(self):
        """Resume ad blocking immediately."""
        with self._state_lock:
            self.blocking_paused_until = 0
            logger.info("[WebUI] Blocking resumed")

        # Re-evaluate current state
        self._update_blocking_state()
        # Don't override an active block — only return to baseline if we
        # weren't blocking when the user clicked resume.
        if not self.blocking_active:
            self._set_led_state(self._baseline_led_state())

    def is_blocking_paused(self) -> bool:
        """Check if blocking is currently paused."""
        return time.time() < self.blocking_paused_until

    def get_pause_remaining(self) -> int:
        """Get seconds remaining in pause, or 0 if not paused."""
        remaining = self.blocking_paused_until - time.time()
        return max(0, int(remaining))

    def notify_hdmi_reconnect(self):
        """Called by the health monitor when the TV output reconnects."""
        self.hdmi_reconnect_time = time.time()
        logger.info(
            f"[Minus] HDMI reconnect recorded; ad blocking suppressed for "
            f"{self.HDMI_RECONNECT_GRACE_SECONDS:.0f}s"
        )

    def is_in_hdmi_reconnect_grace(self) -> bool:
        """True if we're within the post-reconnect grace window."""
        if not self.hdmi_reconnect_grace_enabled:
            return False
        if self.hdmi_reconnect_time <= 0:
            return False
        return (time.time() - self.hdmi_reconnect_time) < self.HDMI_RECONNECT_GRACE_SECONDS

    def get_hdmi_reconnect_grace_remaining(self) -> int:
        """Seconds left in the HDMI reconnect grace window (0 if inactive)."""
        if not self.is_in_hdmi_reconnect_grace():
            return 0
        remaining = self.HDMI_RECONNECT_GRACE_SECONDS - (time.time() - self.hdmi_reconnect_time)
        return max(0, int(remaining))

    def add_detection(self, source: str, texts: list, matched_keywords: list = None):
        """Add a detection to history for web UI display."""
        from datetime import datetime
        self.detection_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'timestamp': time.time(),
            'source': source,
            'texts': texts[:5] if texts else [],  # Limit to first 5 texts
            'keywords': [kw for kw, _ in matched_keywords] if matched_keywords else [],
        })

    def get_status_dict(self) -> dict:
        """Get current status as dictionary for web API."""
        # Get health status if available
        health_status = None
        if self.health_monitor:
            try:
                health_status = self.health_monitor.get_status()
            except Exception:
                pass

        # Get FPS from ad_blocker (display pipeline) if available
        fps_display = 0
        if self.ad_blocker:
            try:
                fps_display = self.ad_blocker.get_fps() or 0
            except Exception:
                pass

        # Get FPS from ustreamer (capture)
        fps_capture = 0
        try:
            import urllib.request
            import json
            url = "http://localhost:9090/state"
            with urllib.request.urlopen(url, timeout=1.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                fps_capture = data.get('result', {}).get('source', {}).get('captured_fps', 0)
        except Exception:
            pass

        # For backwards compatibility, fps = display fps if available, else capture
        fps = fps_display if fps_display > 0 else fps_capture
        fps_source = 'display' if fps_display > 0 else 'capture'

        uptime = int(time.time() - self.start_time)

        # Check if blocking is active (either via detection or test mode)
        is_blocking = (self.ad_detected and not self.is_blocking_paused() and not self.static_blocking_suppressed)
        # Also check if ad_blocker is directly visible (test mode)
        if self.ad_blocker and self.ad_blocker.is_visible:
            is_blocking = True

        return {
            # Blocking state
            'blocking': is_blocking,
            'blocking_source': self.blocking_source,
            'blocking_source_display': self._display_source(),
            'paused': self.is_blocking_paused(),
            'pause_remaining': self.get_pause_remaining(),
            'vlm_user_paused': self.is_vlm_user_paused(),
            'vlm_user_pause_remaining': self.get_vlm_pause_remaining(),
            'asr_verdict': self._asr_verdict(),
            'asr': (self.asr.get_status() if self.asr is not None else
                    {'available': False, 'enabled': False, 'running': False}),
            'hdmi_reconnect_grace': self.is_in_hdmi_reconnect_grace(),
            'hdmi_reconnect_grace_remaining': self.get_hdmi_reconnect_grace_remaining(),
            'static_suppressed': self.static_blocking_suppressed,

            # Detection counts
            'ocr_detected': self.ocr_ad_detected,
            'vlm_detected': self.vlm_ad_detected,
            'ocr_frame_count': self.frame_count,
            'vlm_frame_count': self.vlm_frame_count,
            'total_detections': self.screenshot_manager.ads_count if self.screenshot_manager else 0,

            # System status
            'fps': fps,
            'fps_capture': fps_capture,
            'fps_display': fps_display,
            'fps_source': fps_source,  # 'display' or 'capture' (for backwards compat)
            'uptime': uptime,
            'uptime_str': f"{uptime // 3600}h {(uptime % 3600) // 60}m",
            'hdmi_signal': health_status.hdmi_signal if health_status else True,
            'vlm_ready': not self.vlm_disabled and (self.vlm is not None and self.vlm.is_ready if self.vlm else False),
            'vlm_disabled': self.vlm_disabled,
            'ocr_ready': self.ocr is not None and self.ocr.is_ready,
            'ocr_disabled': getattr(self, 'ocr_disabled', False),
            # Live OCR block for the webui OCR-Live panel (mirrors 'asr').
            # last_ocr_texts is refreshed every OCR frame, so this is the
            # live on-screen text; matched_keywords lists any ad-keyword
            # hits on the current frame (empty on non-ad content).
            'ocr': {
                'ready': self.ocr is not None and self.ocr.is_ready,
                'disabled': getattr(self, 'ocr_disabled', False),
                'detected': self.ocr_ad_detected,
                'frame_count': self.frame_count,
                'last_text': ' | '.join(self.last_ocr_texts or [])[:300],
                'matched_keywords': [kw for kw, _ in (self.last_matched_keywords or [])],
            },
            'display_connected': self.display_connected,
            'display_error': self.display_error,

            # Bandwidth/color format status
            **self._get_bandwidth_status(),

            # Memory/health
            'memory_percent': health_status.memory_percent if health_status else 0,
            'temperature_c': self._get_soc_temperature(),
            'ustreamer_ok': health_status.ustreamer_responding if health_status else True,
            'video_ok': health_status.video_pipeline_ok if health_status else True,

            # Skip status (Fire TV integration)
            'skip_available': self.skip_available,
            'skip_countdown': self.skip_countdown,
            'skip_text': self.last_skip_text,
            'skip_attempted': self.skip_attempted_this_ad,

            # Fire TV status
            'fire_tv_connected': self.fire_tv_controller is not None and self.fire_tv_controller.is_connected() if self.fire_tv_controller else False,
            'fire_tv_setup_state': self.fire_tv_setup.state if self.fire_tv_setup else None,

            # Roku status
            'roku_connected': self.roku_controller is not None and self.roku_controller.is_connected() if self.roku_controller else False,

            # Device-aware remote status
            'remote_connected': self._is_remote_connected(),
            'remote_device_type': self._get_configured_device_type(),
        }

    def _get_soc_temperature(self):
        """Hottest RK3588 thermal zone in °C (soc/bigcore/littlecore/center/gpu/npu).

        Max across zones: the UI badge answers "how hot is the unit", and the
        throttle trip (~85°C) fires on whichever zone gets there first.
        Returns None if sysfs is unreadable so the UI can show '--'.
        """
        temps = []
        try:
            for zone in glob.glob('/sys/class/thermal/thermal_zone*/temp'):
                try:
                    with open(zone) as f:
                        temps.append(int(f.read().strip()) / 1000.0)
                except (OSError, ValueError):
                    continue
        except Exception:
            return None
        return round(max(temps), 1) if temps else None

    def _get_bandwidth_status(self) -> dict:
        """Get HDMI bandwidth/color format status for API."""
        if self.ad_blocker:
            try:
                return self.ad_blocker.get_bandwidth_status()
            except Exception:
                pass
        return {
            'color_format': None,
            'color_format_value': None,
            'bandwidth_fallback_applied': False,
            'bandwidth_fallback_attempted': False,
        }

    def _is_remote_connected(self) -> bool:
        """Check if the configured remote device is connected."""
        device_type = self._get_configured_device_type()
        if device_type in ('fire_tv', 'google_tv'):
            return self.fire_tv_controller is not None and self.fire_tv_controller.is_connected() if self.fire_tv_controller else False
        elif device_type == 'roku':
            return self.roku_controller is not None and self.roku_controller.is_connected() if self.roku_controller else False
        return False

    def _get_configured_device_type(self) -> str:
        """Get the configured device type."""
        try:
            from src.device_config import get_device_config_manager
            manager = get_device_config_manager()
            return manager.config.device_type
        except:
            return 'none'

    def _compare_frames(self, frame, prev_frame):
        """Compare two frames and return normalized mean difference (0-1)."""
        if frame is None or prev_frame is None:
            return 1.0

        try:
            curr = cv2.resize(frame, (160, 90))
            prev = cv2.resize(prev_frame, (160, 90))
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(curr_gray, prev_gray)
            return diff.mean() / 255.0
        except Exception:
            return 1.0

    def is_scene_changed(self, frame):
        """Check if scene changed (should run OCR)."""
        if self.prev_frame is None:
            return True
        return self._compare_frames(frame, self.prev_frame) > self.scene_change_threshold

    def is_vlm_scene_changed(self, frame):
        """Check if scene changed (should run VLM)."""
        if self.vlm_prev_frame is None:
            return True
        return self._compare_frames(frame, self.vlm_prev_frame) > self.scene_change_threshold

    def _add_vlm_decision(self, is_ad: bool, confidence: float = 0.75):
        """Add a VLM decision to the sliding window history with confidence."""
        now = time.time()
        self.vlm_decision_history.append((now, is_ad, confidence))

        # Prune old decisions outside the window
        cutoff = now - self.vlm_history_window
        self.vlm_decision_history = [
            entry for entry in self.vlm_decision_history
            if entry[0] >= cutoff
        ]

    def _transition_hold_active(self, is_transition: bool) -> bool:
        """Whether to hold a block through a 'transition' frame.

        Only bridges a BRIEF inter-ad gap. `_is_transition_frame` also
        fires on benign uniform content (dark lofi music videos), which
        would otherwise freeze the no-ad counters forever and prevent
        recovery. Cap the continuous hold at TRANSITION_HOLD_MAX_SECONDS;
        past that, treat frames normally so the block can stop.
        Shared by the OCR and VLM loops (a real gap both see resets/extends
        the same window); reset on any non-transition / ad frame.
        """
        if not is_transition:
            self.transition_hold_start = 0.0
            return False
        now = time.time()
        if self.transition_hold_start == 0.0:
            self.transition_hold_start = now
        return (now - self.transition_hold_start) <= self.TRANSITION_HOLD_MAX_SECONDS

    def _is_transition_frame(self, frame, threshold=15, black_threshold=30, uniformity_threshold=0.95) -> tuple:
        """
        Detect if a frame is a transition screen (mostly black or single solid color).

        These frames often appear between ads or between an ad and content.
        When blocking, we should hold through these rather than unblocking.

        Args:
            frame: BGR image (numpy array)
            threshold: Max std dev to consider "uniform" color
            black_threshold: Max brightness to consider "black"
            uniformity_threshold: Min fraction of pixels that must be similar

        Returns:
            (is_transition, reason) - reason is 'black', 'solid_color', or None
        """
        try:
            import numpy as np

            if frame is None or frame.size == 0:
                return False, None

            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame

            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)

            # Check if mostly black (common ad transition)
            if mean_brightness < black_threshold and std_brightness < threshold:
                return True, 'black'

            # Check if solid/uniform color (fade transitions)
            if std_brightness < threshold:
                return True, 'solid_color'

            # Check if most pixels are very similar (near-uniform with minor noise)
            median_val = np.median(gray)
            similar_pixels = np.sum(np.abs(gray - median_val) < 20) / gray.size
            if similar_pixels > uniformity_threshold:
                return True, 'uniform'

            return False, None

        except Exception as e:
            logger.debug(f"Transition detection error: {e}")
            return False, None

    def _current_min_blocking_duration(self) -> float:
        """Compute the dynamic minimum blocking duration for the current ad.

        Consecutive ads shorten the floor so we don't hold the block longer than
        the ad itself. Index 0 is the first ad of a sequence, index N is the
        (N+1)th consecutive ad. Floor depends on whether OCR+VLM both agree
        (slightly longer — 1.5s — because VLM's cycle is slower and we don't
        want to unblock before it confirms) or OCR alone (1.0s).

        When falloff is disabled via the settings toggle, the base 3.0s is held
        regardless of how many ads fired in a row.
        """
        # VLM-only false blocks are the frustrating case — let them clear
        # as soon as VLM says no-ad (VLM_STOP_THRESHOLD), not after the
        # 3.0s base. Applied regardless of the falloff toggle.
        if self.blocking_source == "vlm":
            return self.MIN_BLOCKING_DURATION_FLOOR_VLM
        if not self.block_falloff_enabled:
            return self.MIN_BLOCKING_DURATION_BASE
        floor = (
            self.MIN_BLOCKING_DURATION_FLOOR_BOTH
            if self.blocking_source == "both"
            else self.MIN_BLOCKING_DURATION_FLOOR_OCR
        )
        duration = self.MIN_BLOCKING_DURATION_BASE - self.consecutive_ad_count * self.MIN_BLOCKING_DURATION_STEP
        return max(duration, floor)

    def _get_vlm_agreement(self) -> tuple:
        """
        Calculate VLM agreement percentage from sliding window using confidence-weighted votes.

        High-confidence decisions count more than low-confidence ones.

        Returns:
            (ad_ratio, no_ad_ratio, total_decisions)
            - ad_ratio: confidence-weighted fraction of 'ad' decisions (0.0-1.0)
            - no_ad_ratio: confidence-weighted fraction of 'no-ad' decisions (0.0-1.0)
            - total_decisions: number of decisions in window
        """
        if not self.vlm_decision_history:
            return 0.0, 0.0, 0

        now = time.time()
        cutoff = now - self.vlm_history_window

        # Filter to recent decisions (handle both old and new tuple formats)
        recent = []
        for entry in self.vlm_decision_history:
            if entry[0] >= cutoff:
                if len(entry) == 3:
                    recent.append(entry)  # (time, is_ad, confidence)
                else:
                    # Legacy format without confidence - use default 0.75
                    recent.append((entry[0], entry[1], 0.75))

        if not recent:
            return 0.0, 0.0, 0

        total = len(recent)

        # Confidence-weighted voting
        ad_weight = sum(conf for _, is_ad, conf in recent if is_ad)
        no_ad_weight = sum(conf for _, is_ad, conf in recent if not is_ad)
        total_weight = ad_weight + no_ad_weight

        if total_weight == 0:
            return 0.0, 0.0, total

        return ad_weight / total_weight, no_ad_weight / total_weight, total

    def _should_vlm_start_blocking(self) -> bool:
        """
        Determine if VLM should trigger blocking based on sliding window agreement.

        Uses hysteresis: if we're NOT currently blocking, we need higher agreement to START.
        """
        ad_ratio, _, total = self._get_vlm_agreement()

        if total < self.vlm_min_decisions:
            return False  # Not enough data

        # Check cooldown
        now = time.time()
        if self.vlm_cooldown_active:
            time_since_change = now - self.vlm_last_state_change
            if time_since_change < self.vlm_min_state_duration:
                return False  # Still in cooldown

        # Need strong ad agreement to start. Caller only reaches this function
        # when VLM is acting alone — the OCR-corroborated path at ~line 2778
        # takes an immediate shortcut and bypasses the sliding window entirely.
        threshold = self.vlm_start_agreement
        if not self.vlm_ad_detected:
            # Not currently detecting - need even stronger evidence to start
            threshold += self.vlm_hysteresis_boost
        # Cap so hysteresis + raised base can't push us past what real-world
        # noise allows (a few spurious "no" responses would block triggering forever).
        threshold = min(threshold, self.vlm_start_threshold_cap)

        return ad_ratio >= threshold

    def _should_vlm_stop_blocking(self) -> bool:
        """
        Determine if VLM should stop blocking based on sliding window agreement.

        Uses hysteresis: if we ARE currently blocking, we need higher agreement to STOP.
        """
        _, no_ad_ratio, total = self._get_vlm_agreement()

        if total < self.vlm_min_decisions:
            return False  # Not enough data - keep current state

        # Check cooldown
        now = time.time()
        if self.vlm_cooldown_active:
            time_since_change = now - self.vlm_last_state_change
            if time_since_change < self.vlm_min_state_duration:
                return False  # Still in cooldown

        # Need strong no-ad agreement to stop
        threshold = self.vlm_stop_agreement
        if self.vlm_ad_detected:
            # Currently detecting - need even stronger evidence to stop
            threshold += self.vlm_hysteresis_boost

        return no_ad_ratio >= threshold

    # =========================================================================
    # System Settings
    # =========================================================================

    def _load_system_settings(self) -> dict:
        """Load system settings from disk."""
        defaults = {
            'vlm_preload': True,          # Load VLM at startup (vs wait for HDMI)
            'asr_enabled': True,          # Run faster-whisper ASR confirm/veto for VLM-alone blocks
            'block_falloff': True,        # Shorten min-block duration on consecutive ads
            'hdmi_reconnect_grace': True, # Disable ad blocking for 90s after HDMI reconnect
            'greyscale_preview': True,    # Desaturate the ad preview window in blocking mode
            'debug_overlay': True,        # Show BLOCKING header + bottom-left stats + top-right OCR snippet
            'ir_enabled': False,          # Show REI HDMI-switch IR remote in autonomous mode
            'leds_require_display': True, # Only drive the WS2812B strip when HDMI-TX is connected
                                          # (so a powered-off TV in a dark room stays dark)
            # Which replacement-mode kinds are allowed during ad blocks. A
            # list rather than a dict so the web UI can just toggle checkboxes.
            # Valid kinds: 'vocab', 'fact', 'photos'.
            'replacement_modes': ['vocab', 'fact'],
        }
        try:
            if SYSTEM_SETTINGS_FILE.exists():
                with open(SYSTEM_SETTINGS_FILE) as f:
                    saved = json.load(f)
                    # Merge with defaults
                    for key in defaults:
                        if key in saved:
                            defaults[key] = saved[key]
                    logger.info(
                        f"Loaded system settings: vlm_preload={defaults['vlm_preload']}, "
                        f"block_falloff={defaults['block_falloff']}, "
                        f"hdmi_reconnect_grace={defaults['hdmi_reconnect_grace']}, "
                        f"greyscale_preview={defaults['greyscale_preview']}"
                    )
        except Exception as e:
            logger.warning(f"Could not load system settings: {e}")
        return defaults

    def _save_system_settings(self):
        """Save system settings to disk."""
        try:
            with open(SYSTEM_SETTINGS_FILE, 'w') as f:
                json.dump(self._system_settings, f, indent=2)
            logger.info(f"Saved system settings to {SYSTEM_SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Could not save system settings: {e}")

    def get_system_settings(self) -> dict:
        """Get current system settings."""
        return self._system_settings.copy()

    def set_vlm_preload(self, enabled: bool) -> dict:
        """Set VLM preload preference.

        Args:
            enabled: True to load VLM at startup, False to load when HDMI arrives

        Returns:
            dict with success status and current settings
        """
        self._system_settings['vlm_preload'] = enabled
        self._save_system_settings()
        return {'success': True, 'vlm_preload': enabled}

    @property
    def vlm_preload(self) -> bool:
        """Whether to preload VLM at startup."""
        return self._system_settings.get('vlm_preload', True)

    @property
    def asr_enabled(self) -> bool:
        """Whether ASR (faster-whisper confirm/veto) is enabled."""
        return self._system_settings.get('asr_enabled', True)

    def set_asr_enabled(self, enabled: bool) -> dict:
        """Enable or disable ASR at runtime and persist the choice.

        ASR is a confirm/veto signal for VLM-alone blocks (never blocks
        on its own), so toggling it off only relaxes VLM-alone handling
        back to pre-ASR behaviour — OCR/VLM blocking is unaffected.

        Disabling stops the worker (frees the ~250MB model + the
        inference thread); enabling (re)starts it. Takes effect
        immediately and survives restarts.
        """
        enabled = bool(enabled)
        self._system_settings['asr_enabled'] = enabled
        self._save_system_settings()
        if self.asr is not None:
            self.asr.enabled = enabled
            try:
                if enabled and not self.asr.is_running:
                    self.asr.start()
                elif not enabled and self.asr.is_running:
                    self.asr.stop()
            except Exception as e:
                logger.warning(f"ASR {'start' if enabled else 'stop'} failed: {e}")
                return {'success': False, 'error': str(e), 'asr_enabled': enabled}
        return {'success': True, 'asr_enabled': enabled}

    @property
    def block_falloff_enabled(self) -> bool:
        """Whether consecutive-ad min-duration falloff is active."""
        return self._system_settings.get('block_falloff', True)

    @property
    def hdmi_reconnect_grace_enabled(self) -> bool:
        """Whether ad blocking is suppressed for 90s after HDMI reconnect."""
        return self._system_settings.get('hdmi_reconnect_grace', True)

    @property
    def greyscale_preview_enabled(self) -> bool:
        """Whether the ad preview window is desaturated in blocking mode."""
        return self._system_settings.get('greyscale_preview', True)

    @property
    def debug_overlay_enabled(self) -> bool:
        """Whether the blocking overlay shows debug info (header, stats, OCR snippet)."""
        return self._system_settings.get('debug_overlay', True)

    def set_debug_overlay_enabled(self, enabled: bool) -> dict:
        """Toggle the unified debug-overlay flag and persist it.

        Controls three on-screen elements together: the [BLOCKING // ...]
        header, the bottom-left stats dashboard, and the top-right OCR
        trigger snippet. Off hides all three.
        """
        enabled = bool(enabled)
        self._system_settings['debug_overlay'] = enabled
        self._save_system_settings()
        if self.ad_blocker:
            self.ad_blocker.set_debug_overlay_enabled(enabled)
        return {'success': True, 'debug_overlay': enabled}

    @property
    def ir_enabled(self) -> bool:
        """Whether the REI HDMI-switch IR remote is exposed in the web UI."""
        return self._system_settings.get('ir_enabled', False)

    def set_ir_enabled(self, enabled: bool) -> dict:
        """Toggle IR remote visibility in the UI. When turning off, also
        releases the PWM if it had been initialized."""
        enabled = bool(enabled)
        self._system_settings['ir_enabled'] = enabled
        self._save_system_settings()
        if not enabled and self.ir_transmitter and self.ir_transmitter.initialized:
            try:
                self.ir_transmitter.shutdown()
            except Exception as e:
                logger.warning(f"IR transmitter shutdown failed: {e}")
        return {'success': True, 'ir_enabled': enabled}

    @property
    def leds_require_display(self) -> bool:
        """Whether the status-LED strip should only drive frames when the
        HDMI-TX display is actually connected. Default True — keeps a
        dark room dark when the TV is powered off."""
        return self._system_settings.get('leds_require_display', True)

    def set_leds_require_display(self, enabled: bool) -> dict:
        self._system_settings['leds_require_display'] = bool(enabled)
        self._save_system_settings()
        return {'success': True, 'leds_require_display': bool(enabled)}

    def is_display_connected_live(self) -> bool:
        """Live HDMI-TX presence check (sysfs, no caching).

        ``self.display_connected`` reflects the *display pipeline's* state
        — once the pipeline has come up successfully it stays True until
        a long-running retry loop notices a real disconnect. For the
        LED gate we want a fast yes/no based on the kernel's current
        view of /sys/class/drm, so we route through the health monitor's
        sysfs probe.
        """
        if self.health_monitor is not None:
            try:
                return self.health_monitor._check_hdmi_output_connected()
            except Exception:
                pass
        # Fall back to the cached pipeline-side flag so we never
        # accidentally gate to dark on a flaky probe.
        return self.display_connected

    def set_optimization_setting(self, key: str, enabled: bool) -> dict:
        """Update one of the optimization toggles and persist."""
        allowed = {'block_falloff', 'hdmi_reconnect_grace', 'greyscale_preview'}
        if key not in allowed:
            return {'success': False, 'error': f'unknown setting {key}'}
        self._system_settings[key] = bool(enabled)
        self._save_system_settings()
        return {'success': True, key: bool(enabled)}

    def get_replacement_modes(self) -> list:
        """Which replacement-mode kinds the overlay may roll into."""
        modes = self._system_settings.get(
            'replacement_modes', ['vocab', 'fact'])
        allowed = {'vocab', 'fact', 'photos'}
        # Strip any legacy 'haiku' entries silently (kind was removed)
        return [m for m in modes if m in allowed]

    def set_replacement_modes(self, modes: list) -> dict:
        """Persist the user's replacement-mode selection.

        At least one text kind must remain enabled; if the caller tried to
        disable every text option we force ``vocab`` back on so the overlay
        has *something* to show when photos run out or aren't enabled.
        """
        allowed = {'vocab', 'fact', 'photos'}
        cleaned = [m for m in (modes or []) if m in allowed]
        text_kinds = {'vocab', 'fact'}
        if not any(m in text_kinds for m in cleaned):
            cleaned.append('vocab')
        self._system_settings['replacement_modes'] = sorted(set(cleaned))
        self._save_system_settings()
        # Apply immediately: a locked-in kind from the current/last ad break
        # that is now disabled must not survive into the next rotation or
        # the next block (the 30s style-cooldown would otherwise keep
        # serving e.g. facts after the user turned facts off).
        _ad_blocker = getattr(self, 'ad_blocker', None)
        if _ad_blocker and hasattr(_ad_blocker, 'invalidate_replacement_lock'):
            try:
                _ad_blocker.invalidate_replacement_lock()
            except Exception as e:
                logger.warning(f"Failed to invalidate replacement lock: {e}")
        return {'success': True, 'replacement_modes': self._system_settings['replacement_modes']}

    def _load_vlm_model(self) -> bool:
        """Load VLM model with retry logic.

        Returns:
            True if VLM loaded successfully, False otherwise
        """
        if not self.vlm:
            return False

        # Show loading notification
        if self.system_notification:
            self.system_notification.show_vlm_loading()

        vlm_loaded = False
        for attempt in range(1, 4):
            logger.info(f"Loading VLM model (minus-v0.1)... attempt {attempt}/3")
            try:
                if self.vlm.load_model():
                    logger.info("VLM model loaded successfully")
                    vlm_loaded = True
                    break
                else:
                    logger.warning(f"VLM load_model returned False (attempt {attempt}/3)")
            except Exception as e:
                logger.warning(f"VLM load_model error (attempt {attempt}/3): {e}")
            if attempt < 3:
                logger.info("Retrying VLM load in 5 seconds...")
                time.sleep(5)

        if vlm_loaded:
            # Show success notification (auto-hides after 5s)
            if self.system_notification:
                self.system_notification.show_vlm_ready()
        else:
            logger.error("VLM failed after 3 attempts - continuing without VLM")
            self.vlm = None
            # Show failure notification (auto-hides after 8s)
            if self.system_notification:
                self.system_notification.show_vlm_failed()

        return vlm_loaded

    def _first_match_for_overlay(self):
        """Return ``(keyword, snippet)`` for the most recent OCR match, or ''.

        Picked up by the blocking overlay's top-right "(Ad) 0:30 left" hint.
        Returns '' for VLM-only blocks or if OCR hasn't matched yet — the
        ad_blocker preserves any prior snippet across empty values.
        """
        if not self.last_matched_keywords:
            return ''
        try:
            kw, snippet = self.last_matched_keywords[0]
            return (kw, snippet)
        except (TypeError, ValueError):
            return ''

    def _hdmi_audio_present(self) -> bool:
        """Check if HDMI-IN is currently receiving audio from the source device.

        Uses v4l2-ctl on the capture device, which reports the HDMI audio island
        state independently of our playback pipeline — so it works even when the
        display is disconnected and our alsasink can't open.

        Cached for 2s because the subprocess call adds latency.

        Returns True if audio is present on the HDMI input, False otherwise.
        Defaults to True on error (fail-open: don't suppress real ads).
        """
        now = time.time()
        if (self._hdmi_audio_present_cache is not None and
                now - self._hdmi_audio_present_cache_time < 2.0):
            return self._hdmi_audio_present_cache
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device, '--get-ctrl', 'audio_present'],
                capture_output=True, text=True, timeout=0.5
            )
            present = 'audio_present: 1' in result.stdout
        except Exception:
            present = True
        self._hdmi_audio_present_cache = present
        self._hdmi_audio_present_cache_time = now
        return present

    def check_hdmi_signal(self):
        """Check HDMI signal and return resolution."""
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device, '--query-dv-timings'],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                return None

            width = height = fps = 0
            for line in result.stdout.split('\n'):
                if 'Active width:' in line:
                    width = int(line.split(':')[1].strip())
                elif 'Active height:' in line:
                    height = int(line.split(':')[1].strip())
                elif 'frames per second' in line:
                    match = re.search(r'\((\d+\.?\d*) frames', line)
                    if match:
                        fps = float(match.group(1))

            if width and height:
                return (width, height, fps)
        except Exception as e:
            logger.error(f"Signal check error: {e}")

        return None

    def _init_v4l2_device(self):
        """Initialize V4L2 device with proper DV timings before starting ustreamer.

        The RK3588 HDMI-RX driver requires DV timings to be set before format
        configuration. Without this, some HDMI sources fail with format mismatch errors.
        """
        try:
            # Step 1: Query current DV timings from the source
            logger.info(f"Querying DV timings from {self.device}...")
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device, '--query-dv-timings'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                logger.warning(f"Failed to query DV timings: {result.stderr}")
                return False

            # Parse resolution from output for logging
            width = height = 0
            for line in result.stdout.split('\n'):
                if 'Active width:' in line:
                    width = int(line.split(':')[1].strip())
                elif 'Active height:' in line:
                    height = int(line.split(':')[1].strip())

            logger.info(f"Detected input: {width}x{height}")

            # Step 2: Set the DV timings from query (this is the critical step)
            logger.info("Setting DV timings on device...")
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device, '--set-dv-bt-timings', 'query'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                logger.warning(f"Failed to set DV timings: {result.stderr}")
                return False

            # Give the driver time to stabilize after timing change
            time.sleep(0.3)

            # Step 3: Set pixel format to NV12 for MPP encoder compatibility
            # The RK3588 MPP hardware JPEG encoder only supports NV12 format.
            # Different HDMI sources may output different formats (NV24, NV16, etc.)
            # but we need to request NV12 specifically for the MPP encoder to work.
            logger.info("Setting pixel format to NV12 (required for MPP encoder)...")
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device, '--set-fmt-video', 'pixelformat=NV12'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                # Not fatal - but encoding will likely fail
                logger.warning(f"Failed to set pixel format to NV12: {result.stderr}")
            else:
                logger.info("Pixel format set to NV12 successfully")

            time.sleep(0.2)

            logger.info("V4L2 device initialized successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.warning("Timeout during V4L2 device initialization")
            return False
        except Exception as e:
            logger.warning(f"V4L2 device initialization error: {e}")
            return False

    def start_ustreamer(self):
        """Start ustreamer process for video capture/streaming.

        This keeps the web preview and ML detection working even if
        the display output (HDMI-TX) is disconnected.
        """
        # Kill any existing ustreamer
        subprocess.run(['pkill', '-9', 'ustreamer'], capture_output=True)
        time.sleep(0.5)

        port = self.config.ustreamer_port

        # Initialize V4L2 device with proper DV timings
        # This is critical for adapting to different HDMI sources (FireTV, Roku, AppleTV, etc.)
        self._init_v4l2_device()

        # Probe the device to get current format and resolution
        device_info = probe_v4l2_device(self.device)
        detected_format = device_info.get('ustreamer_format') or 'NV12'
        width = device_info.get('width') or 3840
        height = device_info.get('height') or 2160

        # Use the native format from the source device
        # The patched ustreamer handles NV24→NV12 conversion internally
        # for formats that MPP hardware encoder doesn't support natively
        video_format = detected_format

        # Store for later reference (health monitor, recovery)
        self._detected_format = video_format
        self._detected_resolution = f"{width}x{height}"

        logger.info(f"Source outputs {detected_format} at {width}x{height}")

        # Start ustreamer with detected format and MPP hardware encoder
        # The patched ustreamer will convert NV24→NV12 internally if needed
        ustreamer_cmd = [
            USTREAMER_PATH,
            f'--device={self.device}',
            f'--format={video_format}',
            f'--resolution={width}x{height}',
            '--persistent',
            f'--port={port}',
            '--host=0.0.0.0',          # Bind to all interfaces for remote access
            '--encoder=mpp-jpeg',       # Use RK3588 VPU hardware encoding
            '--encode-scale=passthrough',  # No scaling, use source resolution directly
            '--quality=80',
            '--workers=4',              # 4 parallel MPP encoders
            '--buffers=5',
            '--tcp-nodelay',           # Disable Nagle's algorithm for smoother streaming
        ]

        logger.info(f"Starting ustreamer: {' '.join(ustreamer_cmd)}")

        # Clean up any stale resources from previous runs
        subprocess.run(['fuser', '-k', f'{port}/tcp'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Remove stale frame files that might be owned by root (use glob for PID-based names)
        stale_patterns = ['minus_frame*.jpg', 'minus_vlm_frame*.jpg']
        for pattern in stale_patterns:
            for f in Path('/dev/shm').glob(pattern):
                try:
                    f.unlink(missing_ok=True)
                except PermissionError:
                    # File owned by root, use sudo fallback
                    subprocess.run(['sudo', 'rm', '-f', str(f)], capture_output=True)
        time.sleep(0.5)

        self.ustreamer_process = subprocess.Popen(
            ustreamer_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(2)

        if self.ustreamer_process.poll() is not None:
            logger.error("ustreamer failed to start")
            return False

        logger.info(f"ustreamer started on port {port}")

        # Initialize frame capture
        self.frame_capture = UstreamerCapture(port=port)

        return True

    def _is_ustreamer_running(self):
        """Check if ustreamer process is running."""
        return self.ustreamer_process is not None and self.ustreamer_process.poll() is None

    def start_display_pipeline(self):
        """Start the display pipeline (GStreamer kmssink).

        Requires ustreamer to already be running. Returns True on success,
        False if the display output (HDMI-TX) is not available.
        """
        if not self.ad_blocker:
            logger.error("No ad_blocker available")
            return False

        # Wait if pipeline is currently being restarted (avoid race condition)
        if hasattr(self.ad_blocker, '_pipeline_restarting') and self.ad_blocker._pipeline_restarting:
            logger.info("Pipeline restart in progress, waiting...")
            for _ in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                if not self.ad_blocker._pipeline_restarting:
                    break
            if self.ad_blocker._pipeline_restarting:
                logger.warning("Pipeline still restarting after 10s")
                return False

        logger.info("Starting display pipeline...")
        display_ok = self.ad_blocker.start()
        if display_ok:
            logger.info("Display pipeline started - 30 FPS with instant ad blocking")
        else:
            logger.error("Display pipeline failed to start")

        # Start audio passthrough + ASR REGARDLESS of display state. The audio
        # path is HDMI-RX(capture) -> HDMI-TX(playback); decoupling it from the
        # display (video) pipeline means ASR runs off HDMI-RX even when the TV
        # is off — the audio pipeline falls back to fakesink for playback
        # (see AudioPassthrough._init_pipeline) so capture + the ASR tap stay
        # live. When the TV is present we re-point the sink at the live HDMI
        # output first; if the TV just returned while audio was on fakesink,
        # rebuild with the real alsasink so TV audio resumes.
        if self.audio:
            need_start = not self.audio.is_running
            if display_ok and self.audio.is_running and getattr(self.audio, '_playback_fakesink', False):
                logger.info("HDMI-TX now present — rebuilding audio with alsasink (was fakesink)")
                self.audio.stop()
                need_start = True
            if need_start:
                if display_ok:
                    drm_info = probe_drm_output()
                    if (drm_info['audio_device'] and
                            drm_info['audio_device'] != self.audio.playback_device):
                        logger.info(
                            f"Audio device changed: {self.audio.playback_device} "
                            f"-> {drm_info['audio_device']}"
                        )
                        self.audio.playback_device = drm_info['audio_device']
                        self.config.audio_playback_device = drm_info['audio_device']
                if self.audio.start():
                    logger.info("Audio passthrough started"
                                + (" (fakesink — TV off)" if getattr(self.audio, '_playback_fakesink', False) else ""))
                    # Kick off ASR inference loop now that the audio
                    # pipeline (and tap) is producing buffers. Safe to
                    # call even if asr is None. Skip when disabled via the
                    # persisted toggle (worker not spawned, no model load).
                    if self.asr is not None and self.asr.enabled:
                        try:
                            self.asr.start()
                        except Exception as e:
                            logger.warning(f"ASR start failed: {e}")
                else:
                    logger.warning("Audio passthrough failed to start")

        # Re-prepare HDMI-TX audio after the modeset settles. The display
        # pipeline start above just performed a modeset; if the audio PCM
        # was opened before/while the TX link was training, its audio lane
        # is dead even though buffers flow (see _schedule_audio_reprepare).
        if display_ok:
            self._schedule_audio_reprepare(reason="display pipeline started")

        return display_ok

    def _schedule_audio_reprepare(self, delay=4.0, reason="post-modeset"):
        """One-shot delayed audio pipeline restart after a display modeset.

        Root cause (Aug 2026 "no audio after boot/recovery"): the rockchip
        HDMI-TX driver programs its audio lane (audio infoframe + audio
        clock regen) only at PCM prepare time. When the audio pipeline
        opens the PCM while the TX link is still training — the video
        modeset happens within the same second at signal recovery — the
        stream plays into a dead audio lane and never self-heals. Nothing
        above ALSA can see it: buffers flow, hw_ptr advances, the ALSA
        state is RUNNING, so the buffer-flow watchdog stays green. The
        only reliable recovery is closing and reopening the PCM after the
        link has settled, so we always do a one-shot audio restart a few
        seconds after the display pipeline comes up. Costs ~1-2s of audio
        during an event (boot / HDMI recovery) where audio was down anyway.
        """
        if not self.audio:
            return

        def _reprepare():
            time.sleep(delay)
            try:
                if (self.audio and self.audio.is_running
                        and not getattr(self.audio, '_playback_fakesink', False)):
                    logger.info(f"[Audio] Re-preparing HDMI-TX audio ({reason})")
                    self.audio.restart(reason=f"re-prepare: {reason}")
            except Exception as e:
                logger.warning(f"[Audio] Re-prepare failed: {e}")

        threading.Thread(target=_reprepare, daemon=True,
                         name="audio-reprepare").start()

    def start_display(self):
        """Start ustreamer and display pipeline."""
        # Kill any existing GStreamer processes (ustreamer handled in start_ustreamer)
        subprocess.run(['pkill', '-9', 'gst-launch'], capture_output=True)

        # Start ustreamer first (for web preview and ML detection)
        if not self.start_ustreamer():
            return False

        # Wait for ustreamer to actually be capturing frames (not just responding)
        # This minimizes the black screen gap when transitioning from loading to live
        port = self.config.ustreamer_port
        for attempt in range(20):  # Up to 6 seconds (20 * 0.3s)
            try:
                import urllib.request
                import json
                with urllib.request.urlopen(f'http://localhost:{port}/state', timeout=1) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode())
                        captured_fps = data.get('result', {}).get('source', {}).get('captured_fps', 0)
                        if captured_fps > 0:
                            logger.info(f"ustreamer capturing {captured_fps} fps after {attempt + 1} checks")
                            break
            except Exception:
                pass
            time.sleep(0.3)

        # Then start display pipeline (may fail if HDMI-TX disconnected)
        return self.start_display_pipeline()

    def _update_blocking_state(self):
        """Update combined blocking state using weighted OCR/VLM model."""
        with self._state_lock:
            now = time.time()
            ocr_recent = (now - self.last_ocr_ad_time) < self.OCR_TRUST_WINDOW

            # Starting blocking
            if not self.ad_detected:
                should_start = False
                source = None
                asr_confirmed = False  # decorates the display label only

                if self.ocr_ad_detected:
                    # OCR is primary - always trust it immediately
                    should_start = True
                    source = "both" if self.vlm_ad_detected else "ocr"
                    # Label decoration: if ASR is also hearing marketing copy,
                    # show it (ocr+asr / both+asr). ASR has no say in WHETHER
                    # the OCR block fires — only in the label.
                    asr_confirmed = (self._asr_verdict() == 'confirm')
                elif self.vlm_ad_detected:
                    # VLM alone - vlm_ad_detected is now managed by sliding window
                    # which already requires sustained agreement before setting True
                    # BUT suppress if OCR recently detected home screen or video interface keywords
                    # ALSO suppress if screen is static (prevents blocking on paused video interfaces)
                    if self.home_screen_detected:
                        ad_ratio, _, total = self._get_vlm_agreement()
                        logger.info(f"VLM suppressed - home screen detected (OCR cross-validation). Agreement was {ad_ratio*100:.0f}% of {total}")
                    elif self.video_interface_detected:
                        ad_ratio, _, total = self._get_vlm_agreement()
                        logger.info(f"VLM suppressed - video interface detected (prevents false positive on player UI). Agreement was {ad_ratio*100:.0f}% of {total}")
                    elif self.static_blocking_suppressed:
                        ad_ratio, _, total = self._get_vlm_agreement()
                        logger.info(f"VLM suppressed - static screen detected (prevents false positive on paused content). Agreement was {ad_ratio*100:.0f}% of {total}")
                    else:
                        # ASR confirmation/veto gate. ASR is a SECOND
                        # opinion on the audio channel:
                        #   'confirm' — marketing language heard recently;
                        #               block as "vlm+asr" (highest conf)
                        #   'veto'    — clear show dialog, no markers; do NOT
                        #               start blocking (product-placement case)
                        #   'unknown' — no useful signal yet (cold start, music
                        #               only, ASR disabled); fall through to
                        #               the existing VLM-alone behavior
                        # OCR-driven paths above are untouched: OCR text on
                        # screen is authoritative regardless of audio.
                        # ASR is a CONFIRM-ONLY signal at start now: it can
                        # upgrade a VLM-alone block to "vlm+asr" (higher
                        # confidence label) but NEVER suppresses it. The old
                        # start-veto wrongly killed real ads VLM was sure
                        # about (observed: a Hotels.com ad at 80%, an
                        # insurance ad "What's important to you?" at 100% — ASR
                        # vetoed only because the spoken copy lacked explicit
                        # marketing markers). The visual detector is trusted
                        # at start; a genuine product-placement false positive
                        # is caught by the GATED mid-block ASR rescue below,
                        # which only force-stops once VLM ITSELF has weakened.
                        asr_verdict = self._asr_verdict()
                        should_start = True
                        source = "vlm"  # base stays "vlm" for stop-logic;
                        asr_confirmed = (asr_verdict == 'confirm')  # label only
                        ad_ratio, _, total = self._get_vlm_agreement()
                        if asr_verdict == 'confirm' and self.asr:
                            asr_note = f" + ASR confirmed ({self.asr.last_marker_hits} markers)"
                        elif asr_verdict == 'veto':
                            asr_note = " (ASR veto ignored at start — trusting VLM)"
                        else:
                            asr_note = ''
                        logger.info(f"VLM triggered alone (agreement: "
                                    f"{ad_ratio*100:.0f}% of {total} "
                                    f"decisions){asr_note}")

                if should_start and self.is_in_hdmi_reconnect_grace():
                    remaining = self.get_hdmi_reconnect_grace_remaining()
                    logger.info(
                        f"Blocking suppressed - HDMI reconnect grace period "
                        f"({remaining}s remaining)"
                    )
                    should_start = False

                # Don't re-arm on the dying skipped ad's end-card — BUT a
                # VLM sliding-window-confirmed detection (self.vlm_ad_detected:
                # 3+ decisions ≥80% agreement) right after a skip is a real
                # NEW ad in a pod, NOT the end-card (a transitioning end-card
                # cannot sustain that). Suppressing it delayed pod ad #2 by
                # 2-3s. The OCR-accounting-site grace still neutralises the
                # lingering end-card text, so only block an *uncertain*
                # (non-VLM-confirmed) re-arm here.
                if (should_start and not self.vlm_ad_detected
                        and self.last_skip_success_time > 0):
                    since_skip = time.time() - self.last_skip_success_time
                    if since_skip < self.SKIP_UNBLOCK_GRACE_SECONDS:
                        logger.info(
                            f"Blocking suppressed - post-skip grace "
                            f"({since_skip:.1f}s of "
                            f"{self.SKIP_UNBLOCK_GRACE_SECONDS:.0f}s)")
                        should_start = False

                # Upstream stream is frozen on an ad frame (MAX safeguard
                # already fired once on it). Don't churn 150s→150s on a
                # stuck source — wait for the stream to actually resume
                # (scene change clears this in the OCR loop).
                if should_start and self._safeguard_freeze_active:
                    logger.info(
                        "Blocking suppressed - post-safeguard freeze "
                        "(stream frozen on ad frame; awaiting scene change)")
                    should_start = False

                if should_start:
                    self.ad_detected = True
                    self.blocking_start_time = now
                    self.blocking_source = source
                    self.blocking_asr_confirmed = asr_confirmed
                    # Falloff counter: if the last block ended recently (within the
                    # reset gap), this is a consecutive ad — bump the counter. If
                    # it's been a while, this is a fresh ad sequence — reset.
                    if self.blocking_end_time > 0 and (now - self.blocking_end_time) <= self.MIN_DURATION_RESET_GAP:
                        self.consecutive_ad_count += 1
                    else:
                        self.consecutive_ad_count = 0
                    # Reset skip and pause detection for new ad
                    self.accidental_pause_detected = False
                    self.skip_attempted_this_ad = False
                    self.last_skip_countdown = None
                    logger.warning(f"AD BLOCKING STARTED ({self._display_source_label()})")

                    # NOTE: Ad skipping is handled separately based on skip button detection
                    # We only skip when "Skip" appears without countdown (handled in OCR worker)

            # While blocking
            elif self.ad_detected:
                if self.ocr_ad_detected and self.vlm_ad_detected and self.blocking_source != "both":
                    self.blocking_source = "both"
                # ASR confirms mid-block → upgrade the display label (ocr→ocr+asr,
                # both→both+asr, vlm→vlm+asr). ASR usually confirms a few seconds
                # after an instant OCR block, so this is the common path for the
                # "+ASR" label to actually appear. Logic/stop behaviour unchanged.
                if not self.blocking_asr_confirmed and self._asr_verdict() == 'confirm':
                    self.blocking_asr_confirmed = True
                    logger.info(f"ASR confirmed active block → {self._display_source_label()}")

                blocking_elapsed = now - self.blocking_start_time
                should_stop = False

                min_duration = self._current_min_blocking_duration()
                if blocking_elapsed >= min_duration:
                    ocr_says_stop = (self.ocr_no_ad_count >= self.OCR_STOP_THRESHOLD)
                    # For VLM stopping, use consecutive no-ad count (not sliding window)
                    # This ensures responsive stopping after ad ends
                    vlm_says_stop = (self.vlm_no_ad_count >= self.VLM_STOP_THRESHOLD)

                    if self.blocking_source == "vlm":
                        # VLM triggered alone - VLM must also agree to stop
                        # (OCR never detected the ad, so OCR's opinion is unreliable here)
                        # Use simple consecutive count, not sliding window (for responsiveness)
                        should_stop = vlm_says_stop

                        # ASR force-stop: the "product placement rescue".
                        # A VLM-alone block (source=="vlm") where ASR reports
                        # clear show dialog with zero marketing markers for
                        # ≥4s. GATED on VLM HAVING WEAKENED (ad_ratio < 0.5):
                        # we trust the visual detector, so ASR may only end a
                        # block once VLM itself is no longer confidently
                        # calling it an ad. This keeps real ads — where VLM
                        # stays firmly "ad" — blocked even when their spoken
                        # copy has no markers (the Hotels.com / insurance-ad
                        # case). A genuine brand-in-a-show FP shows VLM voting
                        # no-ad as the scene continues, so its ad_ratio drops
                        # and the rescue fires. OCR/both/vlm+asr unaffected.
                        if (not should_stop and blocking_elapsed >= 4.0 and
                                self.asr is not None and
                                self.asr.verdict() == 'veto'):
                            _vlm_ad_ratio, _, _vlm_total = self._get_vlm_agreement()
                            if _vlm_total >= 3 and _vlm_ad_ratio < 0.5:
                                transcript = self.asr.last_transcript[:60]
                                logger.warning(
                                    f"[VLM] Force-stopping VLM-only blocking "
                                    f"({blocking_elapsed:.1f}s): VLM weakened "
                                    f"(ad {_vlm_ad_ratio*100:.0f}% of {_vlm_total}) "
                                    f"+ ASR show-audio veto. Transcript: '{transcript}'")
                                should_stop = True

                        # SAFEGUARD: Auto-stop VLM-only blocking after 90 seconds
                        # This prevents extended false positives on video interfaces
                        # Real ads rarely last more than 60-90 seconds
                        if blocking_elapsed >= 90.0 and not should_stop:
                            logger.warning(f"[VLM] Auto-stopping VLM-only blocking after {blocking_elapsed:.0f}s (safeguard)")
                            should_stop = True
                    elif self.blocking_source == "both":
                        # BOTH OCR and VLM detected this ad. Either one
                        # clearing (2 consecutive no-ad) is a strong, correct
                        # "ad ended" signal — stop on whichever fires first.
                        # This decouples recovery from slow OCR snapshot
                        # capture (observed ~2.5s/frame when the HDMI-OUT
                        # display is disconnected): VLM clears in ~0.3s + 2
                        # cadence ticks while OCR's 2 no-ad frames can lag
                        # ~3s+. Measured: this cut a real ad→content recovery
                        # from ~3s to ~1.5s. iter4's sharp per-frame
                        # separation (p_yes≈0.99 ad / ≈0.01 content) makes a
                        # spurious 2-in-a-row VLM no-ad mid-ad ~0.3% — and
                        # OCR would still be holding then anyway.
                        should_stop = ocr_says_stop or vlm_says_stop
                    else:
                        # OCR triggered alone (VLM dissented / never saw it).
                        # OCR is authoritative; VLM's opinion is unreliable
                        # here so it must NOT be allowed to stop early.
                        should_stop = ocr_says_stop

                # TRIANGULATION VETO: OCR-source block can be force-stopped
                # when BOTH VLM and ASR firmly disagree. The motivating
                # failure case: OCR picks up an ad-keyword word from a
                # TV-show artifact — a billboard with "SKIP" in a movie
                # scene, a news ticker passing through "BUY", a movie
                # title containing "Sponsored". OCR's start-side
                # transience guard catches the 1-frame version of this,
                # but a multi-frame artifact (e.g. a sign held by an
                # actor for 2-3 seconds) can survive. If during the
                # resulting block VLM clearly sees show content AND ASR
                # clearly hears show dialog, the block is almost
                # certainly an OCR FP and should release.
                #
                # Gated on:
                #   - source in {ocr, both}: VLM agreeing at start
                #     ("both") doesn't preclude later disagreement —
                #     a transient ad-text overlay can match BOTH OCR
                #     and VLM briefly. Keeping both eligible.
                #   - blocking_elapsed >= MIN_BLOCK_S: give VLM/ASR
                #     time to gather sliding-window evidence after the
                #     block starts. Without this, we'd veto a real ad
                #     within the first OCR/VLM cycle before signals settle.
                #   - VLM agreement >= 80% no-ad over ≥ vlm_min_decisions:
                #     a strong, sustained "I don't see an ad" signal.
                #   - ASR verdict == 'veto': clear show dialog, no
                #     marketing markers in the 8s rolling window.
                #   - OCR has NOT been sustained: if OCR has been
                #     matching consecutively for ≥OCR_TRUSTED_DWELL_FRAMES
                #     cycles, OCR has "earned" its authority and the
                #     veto is DISABLED. A persistent on-screen ad UI
                #     (Skip in 15, Ad 2 of 3) cannot be overridden by
                #     transient VLM/ASR noise. The artifact case the
                #     veto targets is by definition NOT sustained —
                #     a sign passes through the scene then leaves.
                #
                # OCR-only "skip in" countdown tail-end: as the ad ends
                # the player hides the countdown but the still-visible
                # "Sponsored" text can keep OCR firing for a few cycles
                # while VLM has already flipped. The existing "both"
                # branch handles that (vlm_says_stop OR ocr_says_stop).
                # This veto adds the "OCR-only but in a TV show"
                # recovery path that the standard logic never reached.
                _ocr_strongly_trusted = (
                    self.ocr_ad_detection_count >= self.OCR_TRUSTED_DWELL_FRAMES
                )
                if (not should_stop
                        and not _ocr_strongly_trusted
                        and self.blocking_source in ('ocr', 'both')
                        and blocking_elapsed >= self.OCR_TRIANGULATION_MIN_BLOCK_S):
                    _ad_ratio, _no_ad_ratio, _total = self._get_vlm_agreement()
                    _vlm_says_clean = (_total >= self.vlm_min_decisions
                                       and _no_ad_ratio
                                       >= self.OCR_TRIANGULATION_VLM_NOAD_RATIO)
                    _asr_says_clean = (self._asr_verdict() == 'veto')
                    if _vlm_says_clean and _asr_says_clean:
                        _transcript = (self.asr.last_transcript[:60]
                                       if self.asr is not None else '')
                        logger.warning(
                            f"[Triangulation] Force-stopping "
                            f"{self.blocking_source} block "
                            f"({blocking_elapsed:.1f}s): VLM "
                            f"no_ad={_no_ad_ratio*100:.0f}% of {_total}, "
                            f"ASR=veto, OCR dwell="
                            f"{self.ocr_ad_detection_count}. Suspect OCR "
                            f"FP on TV-show artifact. "
                            f"Transcript: '{_transcript}'")
                        should_stop = True

                # UNIVERSAL SAFEGUARD: no single continuous block should ever
                # exceed MAX_BLOCKING_DURATION regardless of source. Observed
                # in production: a static "Sponsored · Peel to collect" promo
                # tile (bare 'sponsored', audio present) held an OCR+VLM block
                # for 591s because OCR kept matching the weak keyword so
                # ocr_no_ad_count never reached threshold and there was no cap
                # on the ocr/both path (only vlm-only had one). Real ad breaks
                # — even 2-3-ad streaming pods — are well under this. On cap
                # we clear ALL detection state so a genuinely-ongoing ad
                # re-detects fresh within ~1-2 cycles (brief, rare) rather
                # than the screen staying frozen for minutes (catastrophic).
                _hit_max = blocking_elapsed >= self.MAX_BLOCKING_DURATION
                # Frozen-early needs BOTH timers: the text frozen ≥30s AND
                # the block itself active ≥30s. The original implementation
                # only checked the (global) frozen-text timer, which
                # accumulates while NOT blocking — observed live: a real
                # Netflix static sponsored card ("Sponsored | Skip | 90+",
                # text legitimately static) pre-accumulated 36s of frozen
                # time, so the block it finally triggered was force-stopped
                # 0.5s in and then freeze-suppressed while the ad played on.
                _hit_frozen = (self._ocr_text_frozen_for
                               >= self.FROZEN_EARLY_SECONDS
                               and blocking_elapsed >= self.FROZEN_EARLY_SECONDS)
                if _hit_frozen:
                    # Audio veto: a live static ad card carries real audio
                    # (music/voiceover); a genuinely stuck upstream stream is
                    # silent. With audio content present, trust that this is
                    # a real ad and let the MAX cap bound the worst case.
                    try:
                        if (self.audio and self.audio.get_status()
                                .get('recent_level', 0.0) >= 0.01):
                            _hit_frozen = False
                            logger.debug(
                                "[SAFEGUARD] Frozen-early veto: audio content "
                                "present — treating static ad frame as live ad")
                    except Exception:
                        pass
                if (self.ad_detected and not should_stop
                        and (_hit_max or _hit_frozen)):
                    if _hit_frozen and not _hit_max:
                        logger.warning(
                            f"[SAFEGUARD] Force-stopping {self.blocking_source}"
                            f" block after {blocking_elapsed:.0f}s — OCR text "
                            f"frozen {self._ocr_text_frozen_for:.0f}s "
                            f"(>{self.FROZEN_EARLY_SECONDS:.0f}s); stuck "
                            f"upstream stream, not a live ad; clearing state"
                        )
                    else:
                        logger.warning(
                            f"[SAFEGUARD] Force-stopping {self.blocking_source} "
                            f"block after {blocking_elapsed:.0f}s "
                            f"(>{self.MAX_BLOCKING_DURATION:.0f}s cap) — likely a "
                            f"static weak-keyword false positive; clearing state"
                        )
                    should_stop = True
                    self.ocr_ad_detected = False
                    self.ocr_no_ad_count = 0
                    self.ocr_ad_detection_count = 0
                    self.vlm_no_ad_count = 0
                    # If the safeguard fired because the upstream video
                    # FROZE on an ad frame (observed: stream stuck on
                    # "Sponsored…31 Skip in", countdown frozen, OCR text
                    # byte-identical for 150s), do NOT immediately re-block
                    # the same frozen frame — that produced a 150s→150s
                    # churn. Snapshot the frozen OCR text; the freeze is
                    # cleared in the OCR loop only when the text MEANINGFULLY
                    # changes (stream actually resumed). NOTE: pixel
                    # is_scene_changed() is NOT a reliable "resumed" signal
                    # here — a frozen stream still pixel-jitters (buffering
                    # spinner / compression noise) and tripped it ~1s after
                    # the cap, defeating the guard. A real ad/content shows
                    # different OCR text, so text-change clears it within a
                    # cycle; a stuck source keeps identical text → stays
                    # suppressed (and autonomous mode can recover it).
                    self._safeguard_freeze_active = True
                    self._safeguard_freeze_text = _norm_alnum(
                        ' '.join(self.last_ocr_texts or []))

                if should_stop:
                    self.ad_detected = False
                    source_was = self.blocking_source
                    self.blocking_source = None
                    self.blocking_asr_confirmed = False
                    # Clear the OCR snippet so the next block starts fresh
                    # (otherwise the prior ad's "(Ad) 0:30" would render briefly
                    # on the next OCR trigger before fresh OCR data arrives).
                    self.last_matched_keywords = []
                    # Also clear VLM state so it doesn't immediately re-trigger
                    self.vlm_ad_detected = False
                    self.vlm_decision_history.clear()
                    # Clear cached VLM-trigger frame: a pause AFTER the block
                    # ends is not "during a VLM ad block" — don't act on a
                    # stale frame from a block that already cleared cleanly.
                    self.last_vlm_ad_frame = None
                    self.last_vlm_ad_frame_time = 0.0
                    # Track when blocking ended (for accidental pause detection)
                    self.blocking_end_time = time.time()
                    # Reset skip state for next ad
                    self.skip_available = False
                    self.skip_attempted_this_ad = False
                    self.last_skip_countdown = None
                    self.skip_countdown = None
                    logger.warning(f"AD BLOCKING ENDED after {blocking_elapsed:.1f}s (stopped by {source_was.upper() if source_was else 'unknown'})")

            # Update overlay (respect pause state and static screen suppression)
            # Static screen suppression: don't block still ads (paused video, landing pages)
            # so user can interact with UI
            should_show_blocking = (
                self.ad_detected and
                self.blocking_source and
                not self.is_blocking_paused() and
                not self.static_blocking_suppressed and
                not self.config.no_blocking  # Allow disabling for testing
            )

            if self.ad_blocker:
                if should_show_blocking:
                    self.ad_blocker.show(
                        self._display_source(),
                        ocr_trigger_text=self._first_match_for_overlay(),
                    )
                else:
                    self.ad_blocker.hide()

            # Also control audio based on blocking state (same logic)
            # But respect ad_blocker test mode - don't unmute during tests
            if self.audio:
                if should_show_blocking:
                    self.audio.mute()
                elif not (self.ad_blocker and self.ad_blocker.is_test_mode_active()):
                    self.audio.unmute()

    def ml_worker(self):
        """OCR processing thread."""
        # Lower priority so video passthrough takes precedence
        try:
            os.nice(10)  # Higher nice = lower priority
        except OSError:
            pass  # May fail without permissions
        logger.info("OCR worker thread started")
        time.sleep(2)

        if self.frame_capture is None:
            logger.error("Frame capture not initialized")
            return

        logger.info(f"Using HTTP snapshot at {self.frame_capture.snapshot_url}")

        # OCRProcess handles timeout internally - no ThreadPoolExecutor needed

        while self.running:
            try:
                # Pause when HDMI signal is lost to prevent memory leak from repeated timeouts
                if self._hdmi_signal_lost:
                    time.sleep(1.0)
                    continue

                start_time = time.time()
                frame = self.frame_capture.capture()
                capture_time = (time.time() - start_time) * 1000

                if frame is None:
                    time.sleep(0.5)
                    continue

                self.frame_count += 1

                # Scene change detection (with max skip cap to catch missed ads)
                scene_changed = self.is_scene_changed(frame)
                now = time.time()

                # Clear the post-safeguard freeze ONLY when the OCR text
                # meaningfully changes — i.e. the stream actually resumed.
                # A frozen stream pixel-jitters (so is_scene_changed is
                # unreliable) but its OCR text stays ~identical; a real
                # ad/content shows clearly different text. difflib ratio
                # tolerates OCR's frame-to-frame char jitter on the SAME
                # frozen frame (stays >0.9) while a genuine change is <0.7.
                if self._safeguard_freeze_active:
                    _cur_txt = _norm_alnum(' '.join(self.last_ocr_texts or []))
                    if _cur_txt and self._safeguard_freeze_text:
                        _sim = difflib.SequenceMatcher(
                            None, _cur_txt,
                            self._safeguard_freeze_text).ratio()
                    else:
                        _sim = 1.0  # no text yet → treat as still frozen
                    if _sim < 0.7:
                        logger.info(
                            f"[SAFEGUARD] Stream resumed (OCR text changed, "
                            f"sim={_sim:.2f}) — clearing freeze suppression")
                        self._safeguard_freeze_active = False

                # Track static screen state for suppression of still-ad blocking
                if scene_changed:
                    # Screen became dynamic - reset static tracking
                    if self.static_blocking_suppressed and self.screen_became_dynamic_time == 0:
                        # First scene change after suppression - start cooldown
                        self.screen_became_dynamic_time = now
                        logger.info(f"[Static] Screen became dynamic - cooldown {self.DYNAMIC_COOLDOWN}s before allowing blocking")
                    self.static_since_time = 0
                    self.static_ocr_count = 0
                else:
                    # Screen is static - track duration
                    self.static_ocr_count += 1
                    if self.static_since_time == 0:
                        self.static_since_time = now

                # Check if we should suppress blocking due to static screen
                static_time = (now - self.static_since_time) if self.static_since_time > 0 else 0
                # Strong-ad-signal override: if OCR has matched a video-ad-only
                # keyword (Skip in / Skip Ad / Ad N of M / Ad countdown / Ad with
                # timestamp / Visit advertiser) within the last STRONG_AD_HOLD_SECONDS,
                # the screen is definitely showing an active video ad and we
                # should NOT suppress — even if pixels are static (low-motion
                # graphic ads exist). Also force-clears suppression if it's
                # already on, so a strong signal arriving mid-suppression
                # immediately revives blocking instead of waiting for the
                # next scene change + cooldown.
                strong_ad_recent = (now - self.last_strong_ad_time) < self.STRONG_AD_HOLD_SECONDS

                if self.static_blocking_suppressed and strong_ad_recent:
                    # Strong video-ad signal arrived while suppression was on — clear it
                    # immediately. Don't wait for scene change + cooldown; a low-motion
                    # ad might never trigger a scene change in time.
                    logger.info("[Static] Strong ad signal detected — lifting suppression")
                    self.static_blocking_suppressed = False
                    self.screen_became_dynamic_time = 0
                elif (static_time >= self.STATIC_TIME_THRESHOLD or self.static_ocr_count >= self.STATIC_OCR_THRESHOLD) and not strong_ad_recent:
                    if not self.static_blocking_suppressed:
                        logger.info(f"[Static] Screen static for {static_time:.1f}s / {self.static_ocr_count} OCR cycles - suppressing blocking")
                        self.static_blocking_suppressed = True
                        self.screen_became_dynamic_time = 0  # Reset cooldown timer

                        # Accidental pause detection: if screen went static right after we skipped,
                        # we may have accidentally paused the video. Send PLAY to resume.
                        time_since_blocking_end = now - self.blocking_end_time if self.blocking_end_time > 0 else float('inf')
                        time_since_skip_success = now - self.last_skip_success_time if self.last_skip_success_time > 0 else float('inf')

                        if (time_since_blocking_end < self.PAUSE_DETECT_WINDOW and
                            time_since_skip_success < self.PAUSE_DETECT_WINDOW and
                            not self.accidental_pause_detected):
                            logger.warning(f"[PAUSE] Detected potential accidental pause! Screen static {time_since_blocking_end:.1f}s after blocking ended, {time_since_skip_success:.1f}s after skip. Sending PLAY...")
                            self.accidental_pause_detected = True  # Only try once per ad
                            if self.fire_tv_controller and self.fire_tv_controller.is_connected():
                                if self.fire_tv_controller.send_command("play"):
                                    logger.info("[PAUSE] PLAY command sent - video should resume")
                                else:
                                    logger.warning("[PAUSE] Failed to send PLAY command")

                        # Save screenshot as non-ad training data (still ads shouldn't be blocked)
                        if self.ad_detected:
                            self.screenshot_manager.save_static_ad_screenshot(frame)
                        # If currently blocking, hide the overlay
                        if self.ad_detected:
                            self._update_blocking_state()
                elif self.screen_became_dynamic_time > 0:
                    # In cooldown period after screen became dynamic
                    cooldown_elapsed = now - self.screen_became_dynamic_time
                    if cooldown_elapsed >= self.DYNAMIC_COOLDOWN:
                        logger.info(f"[Static] Cooldown complete - blocking re-enabled")
                        self.static_blocking_suppressed = False
                        self.screen_became_dynamic_time = 0

                        # Clear detection state from static + cooldown periods.
                        # OCR/VLM frames during the static window were on the
                        # paused content and during the cooldown were on the
                        # transitioning frames — neither is fresh evidence for
                        # the post-static screen. We must reset the trigger
                        # COUNTER too, not just the detected flag — otherwise
                        # the very next OCR match (which only needs count >= 1)
                        # immediately re-triggers blocking and the cooldown
                        # had no effect.
                        had_state = (
                            self.ocr_ad_detected or self.vlm_ad_detected or
                            self.ocr_ad_detection_count > 0
                        )
                        if had_state:
                            logger.info(
                                f"[Static] Clearing stale detection state "
                                f"(OCR={self.ocr_ad_detected}, VLM={self.vlm_ad_detected}, "
                                f"ocr_count={self.ocr_ad_detection_count})"
                            )
                            self.ocr_ad_detected = False
                            self.ocr_no_ad_count = 0
                            self.ocr_ad_detection_count = 0
                            self.vlm_ad_detected = False
                            self.vlm_no_ad_count = 0
                            self.vlm_decision_history.clear()  # Clear VLM sliding window
                            self._update_blocking_state()  # Update combined state
                elif not self.static_blocking_suppressed:
                    # Normal state - not suppressed and not in cooldown
                    pass

                # Skip OCR processing if scene unchanged (unless forced or was blocking)
                if not self.ad_detected and not scene_changed and not self.prev_frame_had_ad:
                    self.scene_skip_count += 1
                    # Cap consecutive skips to catch ads that appear without scene change
                    if self.scene_skip_count < self.max_scene_skip:
                        if self.scene_skip_count % 10 == 1:
                            logger.info(f"OCR #{self.frame_count}: SKIPPED - scene unchanged (skipped {self.scene_skip_count} total)")
                        time.sleep(0.1)
                        continue
                    else:
                        logger.debug(f"OCR #{self.frame_count}: Force run after {self.scene_skip_count} skips")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run OCR - OCRProcess has hard 1.2s timeout with process kill
                ocr_results = self.ocr.ocr(frame_rgb)
                ocr_time = (time.time() - start_time) * 1000 - capture_time

                # Empty results could mean timeout (process was killed and restarted)
                if not ocr_results:
                    self.ocr_no_ad_count += 1
                    self.ocr_ad_detection_count = 0
                    if self.ocr_ad_detected and self.ocr_no_ad_count >= self.OCR_STOP_THRESHOLD:
                        self.ocr_ad_detected = False
                        logger.info(f"OCR: ad no longer detected (after {self.OCR_STOP_THRESHOLD} no-results)")
                        self._update_blocking_state()
                    continue

                # Check for Fire TV ADB authorization dialog (if waiting for auth)
                if self.fire_tv_setup and ocr_results:
                    # Convert OCR results to format expected by Fire TV checker
                    ocr_text_list = ocr_results if ocr_results else []
                    if self._check_ocr_for_fire_tv_dialog(ocr_text_list):
                        logger.info("[FireTV] ADB authorization dialog detected on screen!")

                ad_detected, matched_keywords, all_texts, is_terminal = self.ocr.check_ad_keywords(ocr_results)

                # Store OCR texts and check for home screen / video interface keywords
                self.last_ocr_texts = all_texts

                # Periodic non-ad screenshot sampler. Captures the current
                # frame as training data when content is unambiguously not
                # an ad: no ad keywords matched this frame, ad blocker not
                # currently showing, no static-screen suppression active,
                # and the per-sample cooldown has elapsed. Dedup +
                # blank-rejection in ScreenshotManager prevents flooding.
                # Tuned conservatively (90s default) to avoid pressuring
                # the OCR loop / disk IO.
                try:
                    _nonad_now = time.time()
                    if (_nonad_now - self._last_nonad_sample_time
                            >= self.NONAD_SAMPLE_INTERVAL
                            and not matched_keywords
                            and frame is not None
                            and self.ad_blocker is not None
                            and not getattr(self.ad_blocker, 'is_visible',
                                            False)
                            and not getattr(self, 'static_blocking_suppressed',
                                            False)
                            and self.screenshot_manager is not None):
                        self.screenshot_manager.save_non_ad_screenshot(frame)
                        self._last_nonad_sample_time = _nonad_now
                except Exception as _e:
                    logger.debug(
                        f"[NonAdSampler] skipped: {_e}")

                # Track OCR-text stability for EARLY frozen-stream
                # detection. Non-empty text only (empty OCR = content /
                # transition, not a frozen ad frame). difflib ratio
                # tolerates OCR's per-frame char jitter on the SAME frame
                # (>0.93) while a real change (incl. a decrementing
                # countdown) drops it well below.
                _now_fs = time.time()
                _cur_norm = _norm_alnum(' '.join(all_texts or []))
                if not _cur_norm:
                    self._ocr_text_stable_since = 0.0
                    self._ocr_text_stable_norm = ''
                    self._ocr_text_frozen_for = 0.0
                else:
                    if (not self._ocr_text_stable_norm or difflib.SequenceMatcher(
                            None, _cur_norm, self._ocr_text_stable_norm
                            ).ratio() < 0.93):
                        self._ocr_text_stable_norm = _cur_norm
                        self._ocr_text_stable_since = _now_fs
                        self._ocr_text_frozen_for = 0.0
                    else:
                        self._ocr_text_frozen_for = (
                            _now_fs - self._ocr_text_stable_since)
                if matched_keywords:
                    self.last_matched_keywords = matched_keywords
                    # Record timestamp if any "strong" keyword (Skip in / Skip Ad /
                    # Ad with timestamp / etc.) was matched — used by the static
                    # suppressor to keep its hands off active video ads. See
                    # STRONG_AD_KEYWORD_NAMES in __init__.
                    if any(kw in self.STRONG_AD_KEYWORD_NAMES for kw, _ in matched_keywords):
                        self.last_strong_ad_time = time.time()
                if all_texts:
                    combined_text = ' '.join(all_texts).lower()

                    # Home screen detection
                    home_keywords_found = [kw for kw in self.home_screen_keywords if kw in combined_text]
                    if len(home_keywords_found) >= 2:  # Require 2+ keywords to confirm home screen
                        self.home_screen_detected = True
                        self.home_screen_detect_time = time.time()
                    elif time.time() - self.home_screen_detect_time > 5.0:  # Clear after 5s
                        self.home_screen_detected = False

                    # Video player interface detection (suppresses VLM false positives)
                    video_keywords_found = [kw for kw in self.video_interface_keywords if kw in combined_text]
                    if len(video_keywords_found) >= 2:  # Require 2+ keywords to confirm video interface
                        self.video_interface_detected = True
                        self.video_interface_detect_time = time.time()
                    elif time.time() - self.video_interface_detect_time > 5.0:  # Clear after 5s
                        self.video_interface_detected = False

                # Check for skip opportunity (for Fire TV ad skipping)
                # CONSERVATIVE APPROACH: Only try to skip ONCE per ad to avoid accidental pauses
                is_skippable, skip_text, countdown = check_skip_opportunity(all_texts)

                # Extract the ad's *own* countdown (Ad 0:30 etc.) — distinct
                # from the skip-button countdown above. Feeds a progress bar
                # in the blocking overlay so the user sees how long is left.
                ad_seconds_left = extract_ad_seconds_remaining(all_texts)
                if ad_seconds_left is not None:
                    self.ad_seconds_remaining = ad_seconds_left
                    self.ad_seconds_remaining_at = time.time()
                    if self.ad_blocker and hasattr(self.ad_blocker, 'set_ad_seconds_remaining'):
                        self.ad_blocker.set_ad_seconds_remaining(ad_seconds_left)

                # Calculate time since ad blocking started
                time_since_blocking = 0
                if self.ad_detected and self.blocking_start_time > 0:
                    time_since_blocking = time.time() - self.blocking_start_time

                # Track countdown transitions (for pre-emptive skip at 1->0)
                countdown_just_hit_zero = (
                    self.last_skip_countdown is not None and
                    self.last_skip_countdown == 1 and
                    (countdown == 0 or is_skippable)
                )

                # Update countdown tracking
                if countdown is not None:
                    self.last_skip_countdown = countdown
                    self.skip_countdown = countdown
                    self.last_skip_text = skip_text
                    if self.ad_blocker:
                        # 99 = special value meaning "OCR detected 'Skip in' but missed the digit"
                        if countdown == 99:
                            self.ad_blocker.set_skip_status(False, "Skip pending...")
                        else:
                            self.ad_blocker.set_skip_status(False, f"Skip in {countdown}s")
                elif is_skippable:
                    self.skip_countdown = 0
                    self.last_skip_countdown = 0

                # Only allow skip after delay period
                skip_delay_passed = time_since_blocking >= self.SKIP_DELAY_SECONDS

                # Check cooldown since last attempt
                time_since_attempt = time.time() - self.last_skip_attempt_time
                in_cooldown = time_since_attempt < self.SKIP_ATTEMPT_COOLDOWN

                # Determine if we should try to skip
                # Conditions: skippable, delay passed, haven't tried this ad, not in cooldown
                should_skip = (
                    is_skippable and
                    skip_delay_passed and
                    not self.skip_attempted_this_ad and
                    not in_cooldown
                )

                # Also skip on countdown 1->0 transition (pre-emptive)
                if countdown_just_hit_zero and not self.skip_attempted_this_ad and not in_cooldown:
                    should_skip = True
                    logger.info("[SKIP] Countdown hit zero - attempting skip")

                if should_skip:
                    self.skip_available = True
                    self.last_skip_text = skip_text

                    # Check if auto-skip is enabled
                    if not self.auto_skip_enabled:
                        logger.info(f"[SKIP] Skip available but auto-skip DISABLED. Text: '{skip_text}'")
                        if self.ad_blocker:
                            self.ad_blocker.set_skip_status(True, "Manual skip")
                    else:
                        self.skip_attempted_this_ad = True  # Mark as attempted - NO RETRIES
                        self.last_skip_attempt_time = time.time()

                        logger.warning(f"[SKIP] >>> Attempting skip (ONE attempt only). Text: '{skip_text}'")
                        if self.ad_blocker:
                            self.ad_blocker.set_skip_status(True, "Skipping...")

                        if self.try_skip_ad():
                            logger.info(f"[SKIP] Skip command sent! Unblocking after brief delay...")
                            self.last_skip_success_time = time.time()
                            if self.ad_blocker:
                                self.ad_blocker.add_time_saved(30.0)

                            # After successful skip, immediately stop blocking
                            # Wait briefly for skip animation, then force unblock
                            def _unblock_after_skip():
                                time.sleep(1.5)  # Brief delay for skip animation
                                logger.info("[SKIP] Forcing unblock after skip")
                                self.ocr_ad_detected = False
                                self.vlm_ad_detected = False
                                self.ocr_no_ad_count = self.OCR_STOP_THRESHOLD
                                self.blocking_source = None
                                self.blocking_asr_confirmed = False
                                if self.ad_blocker:
                                    self.ad_blocker.hide()
                                if self.audio:
                                    self.audio.unmute()
                            threading.Thread(target=_unblock_after_skip, daemon=True).start()
                        else:
                            logger.warning(f"[SKIP] Skip command failed (device not connected?)")

                elif is_skippable and not skip_delay_passed:
                    wait_remaining = int(self.SKIP_DELAY_SECONDS - time_since_blocking)
                    if wait_remaining > 0 and self.ad_blocker:
                        self.ad_blocker.set_skip_status(False, f"Wait {wait_remaining}s")

                elif is_skippable and self.skip_attempted_this_ad:
                    # Already tried - don't retry (this prevents pause spam)
                    pass

                elif is_skippable and in_cooldown:
                    remaining = int(self.SKIP_ATTEMPT_COOLDOWN - time_since_attempt)
                    logger.debug(f"[SKIP] In cooldown, {remaining}s remaining")

                elif not is_skippable and countdown is None:
                    # No skip button detected
                    if self.skip_available:
                        logger.info("[SKIP] Skip button no longer visible")
                    self.skip_available = False
                    self.skip_countdown = None
                    if self.ad_blocker:
                        self.ad_blocker.set_skip_status(False, None)

                # Bare "Sponsored" with no STRONG video-ad keyword (skip in /
                # visit advertiser / countdown / Ad N of M) seen within
                # STRONG_AD_HOLD_SECONDS is a home/promo tile, not a video ad
                # ("Sponsored · Peel to collect", "Sponsored · Date prisa").
                # The old discriminator was _hdmi_audio_present(), but home
                # and promo screens DO carry audio (autoplay previews, music),
                # so it let a static sponsored tile hold a block for ~10 min
                # in production. strong-ad-recent is the reliable signal:
                # real video ads show skip/countdown alongside "Sponsored"
                # within a few seconds (VLM independently covers any genuine
                # sponsored-only video ad, so OCR can safely stay strict).
                strong_ad_recent = ((time.time() - self.last_strong_ad_time)
                                    < self.STRONG_AD_HOLD_SECONDS)
                weak_only = (len(matched_keywords) > 0 and
                             all(kw in self.WEAK_AD_KEYWORD_NAMES
                                 for kw, _ in matched_keywords))

                since_skip = time.time() - self.last_skip_success_time
                in_skip_grace = (self.last_skip_success_time > 0
                                 and since_skip < self.SKIP_UNBLOCK_GRACE_SECONDS)

                suppress_reason = None
                if ad_detected and not is_terminal:
                    keywords_found = [kw for kw, txt in matched_keywords]
                    if in_skip_grace:
                        # Just skipped — this is the dying ad's end-card /
                        # transition. Route to no-ad so the block decays and
                        # cannot re-arm (don't let stale 'skip in' on the
                        # end-card keep it alive).
                        suppress_reason = (
                            f"post-skip grace ({since_skip:.1f}s of "
                            f"{self.SKIP_UNBLOCK_GRACE_SECONDS:.0f}s)")
                    elif self.home_screen_detected:
                        suppress_reason = (f"home screen detected "
                                           f"(would have been {keywords_found})")
                    elif weak_only and not strong_ad_recent:
                        suppress_reason = (
                            f"weak-only {keywords_found}, no strong ad "
                            f"keyword in {self.STRONG_AD_HOLD_SECONDS:.0f}s "
                            f"(texts {all_texts[:3]})")

                real_ad_frame = (ad_detected and not is_terminal
                                 and suppress_reason is None)

                if real_ad_frame:
                    self.transition_hold_start = 0.0  # ad present → reset gap timer
                    self.ocr_ad_detection_count += 1
                    self.ocr_no_ad_count = 0
                    self.last_ocr_ad_time = time.time()

                    # Transience guard: require ≥OCR_TRANSIENCE_MIN_FRAMES
                    # consecutive OCR-matched frames before firing the
                    # FIRST block. Rejects single-frame OCR misreads
                    # (movie billboard with "SKIP", a sign in a scene
                    # reading "Sponsored", caption text that contains an
                    # ad-keyword word). Real ad UIs keep the keyword
                    # visible continuously so they clear the threshold
                    # within 1-2 OCR cycles (~500-1000ms penalty).
                    #
                    # Fast-fire on 1 frame (skip the dwell) when ANY of:
                    #   - a DEFINITIVE ad-UI keyword matched (skip in / skip
                    #     ad / ad countdown / ad N of M / ad with timestamp /
                    #     visit advertiser / video will play after ad) — these
                    #     never occur as a 1-frame show-content artifact, so
                    #     the transience guard buys nothing but latency here.
                    #   - VLM is also asserting ad, OR ASR confirms marketing
                    #     language — corroboration makes the artifact case
                    #     vanishingly rare.
                    # The 2-frame dwell still protects ambiguous matches
                    # (bare 'sponsored', weak keywords) that CAN be artifacts.
                    definitive_ocr = any(
                        kw in self.DEFINITIVE_AD_KEYWORD_NAMES
                        for kw in keywords_found)
                    fast_fire = (definitive_ocr
                                 or self.vlm_ad_detected
                                 or self._asr_verdict() == 'confirm')
                    required_frames = (1 if fast_fire
                                       else self.OCR_TRANSIENCE_MIN_FRAMES)

                    if (self.ocr_ad_detection_count >= required_frames
                            and not self.ocr_ad_detected):
                        self.ocr_ad_detected = True
                        if required_frames > 1:
                            logger.info(
                                f"OCR ad-detection confirmed after "
                                f"{self.ocr_ad_detection_count}-frame "
                                f"dwell (transience guard)")
                    elif (self.ocr_ad_detection_count < required_frames
                            and not self.ocr_ad_detected):
                        # Logged at INFO so the user can see why blocking
                        # didn't fire — a single-frame OCR hit awaiting
                        # confirmation from the next cycle.
                        logger.info(
                            f"OCR ad-detection pending dwell "
                            f"({self.ocr_ad_detection_count}/"
                            f"{required_frames} frames, keywords="
                            f"{keywords_found})")
                    logger.info(f"OCR detected ad keywords: {keywords_found}")
                    self.screenshot_manager.save_ad_screenshot(frame, matched_keywords, all_texts)
                    self.add_detection('OCR', all_texts, matched_keywords)
                else:
                    # Suppressed (home/weak-sponsored) OR no ad keyword at all.
                    # CRITICAL: route this into the no-ad accounting so an
                    # active block actually DECAYS. The old code only logged
                    # on suppression and fell through without touching
                    # ocr_no_ad_count, so a static suppressed 'sponsored'
                    # screen froze the counters and the block never stopped.
                    if suppress_reason is not None:
                        logger.info(f"OCR suppressed - {suppress_reason}")

                    # Transition frame (black/solid) between ads: hold block.
                    is_transition, transition_type = self._is_transition_frame(frame)
                    if self.ad_detected and self._transition_hold_active(is_transition):
                        logger.info(f"OCR #{self.frame_count}: Transition frame ({transition_type}) - holding block")
                    else:
                        self.ocr_no_ad_count += 1
                        self.ocr_ad_detection_count = 0

                        if self.ocr_ad_detected and self.ocr_no_ad_count >= self.OCR_STOP_THRESHOLD:
                            self.ocr_ad_detected = False
                            logger.info(f"OCR: ad no longer detected (after {self.OCR_STOP_THRESHOLD} no-ads)")

                self._update_blocking_state()

                # Log
                total_time = (time.time() - start_time) * 1000
                blocking_info = ""
                if self.ad_detected:
                    if self.static_blocking_suppressed:
                        blocking_info = " [AD DETECTED - STATIC SUPPRESSED]"
                    elif self.ocr_ad_detected and self.vlm_ad_detected:
                        blocking_info = " [BLOCKING OCR+VLM]"
                    elif self.ocr_ad_detected:
                        blocking_info = " [BLOCKING OCR]"
                    elif self.vlm_ad_detected:
                        blocking_info = " [BLOCKING VLM]"

                if all_texts:
                    text_preview = ' | '.join(all_texts)[:120]
                    logger.info(f"OCR #{self.frame_count}: cap={capture_time:.0f}ms ocr={ocr_time:.0f}ms{blocking_info} - {text_preview}")
                elif self.frame_count % 10 == 0:
                    logger.info(f"OCR #{self.frame_count}: cap={capture_time:.0f}ms ocr={ocr_time:.0f}ms, no text{blocking_info}")

                self.prev_frame = frame.copy()
                self.prev_frame_had_ad = ad_detected and not is_terminal
                self.scene_skip_count = 0  # Reset skip counter after processing

                # Periodic garbage collection to prevent memory leak
                if self.frame_count % 100 == 0:
                    gc.collect()

            except Exception as e:
                logger.exception(f"OCR worker error: {e}")

            time.sleep(0.1)

        logger.info("OCR worker thread stopped")

    def vlm_worker(self):
        """VLM processing thread."""
        # Lower priority so video passthrough takes precedence
        try:
            os.nice(10)  # Higher nice = lower priority
        except OSError:
            pass  # May fail without permissions
        logger.info("VLM worker thread started")
        time.sleep(3)

        if self.frame_capture is None:
            logger.error("Frame capture not initialized for VLM")
            return

        if not self.vlm or not self.vlm.is_ready:
            logger.error("VLM not ready")
            return

        vlm_image_path = f'/dev/shm/minus_vlm_frame_{os.getpid()}.jpg'

        # VLMProcess handles hard 2s timeout internally - no ThreadPoolExecutor needed

        while self.running:
            try:
                # Pause when HDMI signal is lost to prevent memory leak from repeated timeouts
                if self._hdmi_signal_lost:
                    time.sleep(1.0)
                    continue

                # VLM false-positive cooldown: user paused during a
                # VLM-only block recently → don't run VLM inference at
                # all for 5 min. Stops repeat-trigger on the same misclassified
                # content even if it stays on screen. OCR keeps running.
                if self.is_vlm_user_paused():
                    # vlm_frame_count doesn't increment in this branch, so
                    # gate the log on wall-clock time instead.
                    now = time.time()
                    if (now - getattr(self, '_vlm_pause_last_log', 0)) >= 30:
                        self._vlm_pause_last_log = now
                        logger.info(
                            f"VLM: skipping inference — user-feedback "
                            f"cooldown ({self.vlm_paused_until - now:.0f}s "
                            f"remaining)")
                    time.sleep(1.0)
                    continue

                start_time = time.time()
                frame = self.frame_capture.capture()

                if frame is None:
                    time.sleep(0.5)
                    continue

                self.vlm_frame_count += 1

                # Scene change detection (with max skip cap)
                if not self.ad_detected and not self.is_vlm_scene_changed(frame) and not self.vlm_prev_frame_had_ad:
                    self.vlm_scene_skip_count += 1
                    # Cap consecutive skips to catch ads
                    if self.vlm_scene_skip_count < self.vlm_max_scene_skip:
                        if self.vlm_scene_skip_count % 10 == 1:
                            logger.info(f"VLM #{self.vlm_frame_count}: SKIPPED - scene unchanged (skipped {self.vlm_scene_skip_count} total)")
                        time.sleep(0.5)
                        continue
                    else:
                        logger.debug(f"VLM #{self.vlm_frame_count}: Force run after {self.vlm_scene_skip_count} skips")

                cv2.imwrite(vlm_image_path, frame)

                # Run VLM - VLMProcess has hard 2s timeout with process kill
                is_ad, response, elapsed, confidence = self.vlm.detect_ad(vlm_image_path)

                # Check if VLM was killed (response will be "KILLED")
                if response == "KILLED":
                    logger.warning(f"VLM #{self.vlm_frame_count}: KILLED after {elapsed:.1f}s - worker restarted")
                    self.vlm_prev_frame = frame.copy()
                    self.vlm_scene_skip_count = 0
                    continue

                # Discard slow VLM responses - scene likely changed during inference
                VLM_MAX_RELEVANT_TIME = 2.0
                if elapsed > VLM_MAX_RELEVANT_TIME:
                    ad_status = "AD" if is_ad else "NO-AD"
                    response_preview = response[:30] if response else "no response"
                    logger.warning(f"VLM #{self.vlm_frame_count}: {elapsed:.1f}s [{ad_status}] DISCARDED (took >{VLM_MAX_RELEVANT_TIME}s) \"{response_preview}\"")
                    self.vlm_prev_frame = frame.copy()
                    self.vlm_scene_skip_count = 0
                    time.sleep(0.5)
                    continue

                # Add decision to sliding window history with confidence
                now = time.time()
                self._add_vlm_decision(is_ad, confidence)

                # Cache the most-recent VLM-AD-verdict frame for user
                # feedback (see VLM_FALSE_POSITIVE_COOLDOWN block in
                # __init__). If the user later pauses during a VLM-only
                # block, this is the frame we save to non_ads/ as the
                # misclassification example.
                if is_ad:
                    with self._state_lock:
                        self.last_vlm_ad_frame = frame.copy()
                        self.last_vlm_ad_frame_time = now

                # Track state changes for waffle detection and logging
                current_state = 'ad' if is_ad else 'no-ad'
                if self.vlm_last_state is not None and current_state != self.vlm_last_state:
                    time_since_last_change = now - self.vlm_state_change_time
                    if time_since_last_change < 15.0:  # Quick flip-flop
                        self.vlm_waffle_count = min(self.vlm_waffle_count + 1, 10)
                    self.vlm_state_change_time = now
                self.vlm_last_state = current_state

                # Update legacy counters (for logging and spastic detection)
                if is_ad:
                    self.transition_hold_start = 0.0  # ad present → reset gap timer
                    self.vlm_consecutive_ad_count += 1
                    self.vlm_no_ad_count = 0
                else:
                    # Check for transition frame - don't count as "no ad" if blocking
                    is_transition, transition_type = self._is_transition_frame(frame)
                    if self.ad_detected and self._transition_hold_active(is_transition):
                        logger.info(f"VLM #{self.vlm_frame_count}: Transition frame ({transition_type}) - holding block")
                    else:
                        self.vlm_no_ad_count += 1
                        # VLM "spastic" detection: save screenshot for training
                        if 2 <= self.vlm_consecutive_ad_count <= 5:
                            self.screenshot_manager.save_vlm_spastic_screenshot(frame, self.vlm_consecutive_ad_count)
                        self.vlm_consecutive_ad_count = 0

                # Get current agreement stats for logging
                ad_ratio, no_ad_ratio, total_decisions = self._get_vlm_agreement()

                # Use sliding window approach for state changes
                prev_vlm_ad_detected = self.vlm_ad_detected

                if not self.vlm_ad_detected:
                    # Not currently detecting - check if we should START
                    # If OCR already triggered blocking, VLM can confirm immediately
                    # Otherwise, VLM needs sliding window agreement to trigger alone
                    if self.ad_detected and self.blocking_source == "ocr" and is_ad:
                        # VLM confirming OCR detection - upgrade to "both" immediately
                        self.vlm_ad_detected = True
                        logger.info(f"VLM confirming OCR detection: \"{response[:50] if response else ''}\"")
                    elif self._should_vlm_start_blocking():
                        self.vlm_ad_detected = True
                        self.vlm_last_state_change = now
                        self.vlm_cooldown_active = True
                        logger.warning(f"VLM detected ad (agreement: {ad_ratio*100:.0f}% of {total_decisions} decisions): \"{response[:50]}\"")
                        self.add_detection('VLM', [response[:100]] if response else [])
                else:
                    # Currently detecting - check if we should STOP
                    if self._should_vlm_stop_blocking():
                        self.vlm_ad_detected = False
                        self.vlm_last_state_change = now
                        self.vlm_cooldown_active = True
                        self.vlm_waffle_count = max(0, self.vlm_waffle_count - 1)  # Decay on stable stop
                        logger.warning(f"VLM: ad no longer detected (agreement: {no_ad_ratio*100:.0f}% no-ad of {total_decisions} decisions)")

                # Clear cooldown after minimum state duration
                if self.vlm_cooldown_active and (now - self.vlm_last_state_change) >= self.vlm_min_state_duration:
                    self.vlm_cooldown_active = False

                self._update_blocking_state()

                ad_status = "AD" if is_ad else "NO-AD"
                response_preview = response[:40] if response else "no response"
                logger.info(f"VLM #{self.vlm_frame_count}: {elapsed:.1f}s [{ad_status}] conf={confidence:.0%} \"{response_preview}\"")

                # Add VLM detection to history whenever VLM detects an ad
                # (not just when VLM triggers blocking alone - also when confirming OCR)
                if is_ad:
                    self.add_detection('VLM', [f"[AD] {response[:80]}" if response else "[AD]"])

                self.vlm_prev_frame = frame.copy()
                self.vlm_prev_frame_had_ad = is_ad
                self.vlm_scene_skip_count = 0  # Reset skip counter after processing

                # Periodic garbage collection to prevent memory leak
                if self.vlm_frame_count % 50 == 0:
                    gc.collect()

            except Exception as e:
                logger.exception(f"VLM worker error: {e}")

            time.sleep(0.5)

        # Clean up VLM frame file
        try:
            Path(vlm_image_path).unlink(missing_ok=True)
        except Exception:
            pass

        logger.info("VLM worker thread stopped")

    def run(self):
        """Start the stream processing."""
        logger.info("Starting Minus...")

        # Status LEDs: if the user persisted the toggle as enabled, start the
        # animation thread now and show the "initializing" white pulse until
        # the rest of the boot sequence settles into idle / blocking / etc.
        if self.status_leds and self.status_leds.enabled:
            try:
                self.status_leds.start()
                self.status_leds.set_state("initializing")
            except Exception as e:
                logger.warning(f"Status LEDs start failed: {e}")

        # Start web UI early so it's accessible even when waiting for HDMI signal
        if HAS_WEBUI:
            try:
                self.webui = WebUI(
                    minus_instance=self,
                    port=self.config.webui_port,
                    ustreamer_port=self.config.ustreamer_port
                )
                self.webui.start()
                logger.info(f"Web UI available at http://0.0.0.0:{self.config.webui_port}")
            except Exception as e:
                logger.warning(f"Failed to start Web UI: {e}")
                self.webui = None

        # Check HDMI signal IMMEDIATELY - we want to show display ASAP (within 3-5s)
        # This is done BEFORE WiFi manager (which takes ~5s) to minimize startup delay
        signal_info = self.check_hdmi_signal()

        # If HDMI signal present, show loading display IMMEDIATELY
        # The loading mode uses videotestsrc (no ustreamer needed) so it starts instantly
        if signal_info and self.ad_blocker:
            logger.info("HDMI signal detected - showing loading display while initializing...")
            self.ad_blocker.start_loading_mode()

        # Start WiFi manager and monitor thread
        # If no WiFi, it will auto-start AP mode after 30 seconds
        if HAS_WIFI_MANAGER:
            try:
                self.wifi_manager = get_wifi_manager()

                # Define callbacks for AP mode events
                def on_ap_started():
                    logger.info("[WiFi] AP mode started - captive portal available")
                    self._set_led_state('wifi_setup')
                    if HAS_OVERLAY:
                        try:
                            overlay = SystemNotification(ustreamer_port=self.config.ustreamer_port)
                            overlay.show(
                                "Connect to WiFi: Minus\nPassword: minussetup\nOpen browser to configure",
                                duration=0,  # Persistent until AP stops
                                position='center'
                            )
                        except Exception as e:
                            logger.warning(f"Failed to show AP overlay: {e}")

                def on_ap_stopped():
                    logger.info("[WiFi] AP mode stopped - connected to WiFi")
                    self._set_led_state('idle')
                    if HAS_OVERLAY:
                        try:
                            overlay = SystemNotification(ustreamer_port=self.config.ustreamer_port)
                            overlay.hide()
                        except Exception as e:
                            logger.warning(f"Failed to hide AP overlay: {e}")

                self.wifi_manager._on_ap_started = on_ap_started
                self.wifi_manager._on_ap_stopped = on_ap_stopped

                # Start the WiFi monitor thread
                self.wifi_manager.start_monitor()
                logger.info("[WiFi] WiFi manager and monitor started")

                # Log current WiFi status
                status = self.wifi_manager.get_status()
                if status.connected:
                    logger.info(f"[WiFi] Connected to: {status.ssid} ({status.ip_address})")
                else:
                    logger.info("[WiFi] Not connected - AP will start in 30 seconds if no connection")
            except Exception as e:
                logger.warning(f"Failed to start WiFi manager: {e}")
                self.wifi_manager = None
        else:
            self.wifi_manager = None

        # Start health monitor early so status is available
        if self.health_monitor:
            self.health_monitor.start()
            logger.info("Health monitor started")

        # Note: HDMI check and start_loading_mode() already done above (before WiFi manager)
        # signal_info variable is already set from that earlier check

        # Start VLM preload in a background thread (non-blocking)
        # This runs while the loading display is showing and main startup continues
        vlm_preloaded = False
        vlm_preload_thread = None
        if self.vlm_preload and self.vlm:
            def _preload_vlm():
                nonlocal vlm_preloaded
                logger.info("Preloading VLM model in background thread...")
                vlm_preloaded = self._load_vlm_model()
                if vlm_preloaded:
                    logger.info("VLM preload complete - model ready")

            vlm_preload_thread = threading.Thread(target=_preload_vlm, daemon=True)
            vlm_preload_thread.start()
            logger.info("VLM preload started in background")

        # If no HDMI signal, handle the no-signal case
        if not signal_info:
            logger.warning("No HDMI signal detected - starting in no-signal mode")
            # Start display in no-signal mode to show "NO HDMI INPUT"
            no_signal_display_ok = False
            if self.ad_blocker:
                no_signal_display_ok = self.ad_blocker.start_no_signal_mode()
                if no_signal_display_ok:
                    logger.info("Display showing NO SIGNAL message - waiting for HDMI...")
                else:
                    logger.warning("Could not start no-signal display (DRM unavailable?) - waiting for HDMI without display")
            else:
                logger.warning("No ad_blocker available - waiting for HDMI without display")

            # Poll for HDMI signal every 2 seconds (even if no-signal display failed)
            self.running = True
            try:
                poll_count = 0
                while self.running:
                    time.sleep(2)
                    poll_count += 1
                    if poll_count % 15 == 0:  # Log every 30 seconds
                        logger.info("Still waiting for HDMI input...")

                    # Retry no-signal display periodically if it failed initially
                    if not no_signal_display_ok and self.ad_blocker and poll_count % 5 == 0:
                        no_signal_display_ok = self.ad_blocker.start_no_signal_mode()
                        if no_signal_display_ok:
                            logger.info("No-signal display now working")

                    signal_info = self.check_hdmi_signal()
                    if signal_info:
                        width, height, fps = signal_info
                        logger.info(f"HDMI signal detected: {width}x{height} @ {fps}fps - switching to loading mode")
                        # Switch to loading mode while we start the display
                        if self.ad_blocker:
                            self.ad_blocker.start_loading_mode()
                        break
            except KeyboardInterrupt:
                self.stop()
                return True

            if not self.running:
                self.stop()
                return True

        width, height, fps = signal_info
        logger.info(f"HDMI signal: {width}x{height} @ {fps}fps")

        # If ad_blocker doesn't have a loading/no-signal screen showing, start loading now
        # This ensures we always show loading during ustreamer startup
        if self.ad_blocker and self.ad_blocker.current_source not in ('loading', 'no_hdmi_device'):
            logger.info("Starting loading display while initializing...")
            self.ad_blocker.start_loading_mode()

        # Start ML threads flag - set early so threads can start
        self.running = True

        # Start display (will transition from loading to live when ready)
        if not self.start_display():
            logger.warning("Failed to start display - will retry in background")
            self.display_connected = False
            self.display_error = "Display output not available. Check HDMI-TX connection to TV/monitor."
            # Start retry loop in background
            self._start_display_retry_loop()
        else:
            self.display_connected = True
            self.display_error = None
            # Reset signal lost flag - display is running so signal is present
            # This fixes race condition where health monitor may have triggered
            # signal lost callback during startup before ustreamer was ready
            self._hdmi_signal_lost = False
            # Resume audio watchdog if it was paused during startup
            if self.audio:
                self.audio.resume_watchdog()
                self.audio.unmute()
            logger.info("Display running at 30 FPS with instant ad blocking")

        if self.ocr:
            self.ml_thread = threading.Thread(target=self.ml_worker, daemon=True)
            self.ml_thread.start()

            if PaddleOCR:
                all_keywords = PaddleOCR.AD_KEYWORDS_EXACT + PaddleOCR.AD_KEYWORDS_WORD
                logger.info(f"OCR watching for ad keywords: {all_keywords}")

        # Note: Health monitor and Web UI already started at beginning of run()

        # Start Fire TV setup early (runs in parallel with VLM loading)
        # 5 second delay ensures display is stable before scanning
        self._start_device_setup_delayed(delay_seconds=5.0)

        # Load VLM model and start worker thread
        # If vlm_preload=True, wait for background preload to finish. Otherwise load now.
        if self.vlm:
            if vlm_preload_thread and vlm_preload_thread.is_alive():
                # Wait for background preload to complete (should be nearly done by now)
                logger.info("Waiting for VLM preload to complete...")
                vlm_preload_thread.join(timeout=60)  # Max 60s wait
                if vlm_preload_thread.is_alive():
                    logger.warning("VLM preload still running after 60s, continuing anyway")
            elif not vlm_preloaded and not self.vlm_preload:
                # Load VLM now (vlm_preload=False means wait for HDMI)
                logger.info("Loading VLM model after HDMI detected (vlm_preload=False)...")
                vlm_preloaded = self._load_vlm_model()

            # Start VLM worker thread if model is loaded (check vlm.is_ready)
            if self.vlm.is_ready:
                self.vlm_thread = threading.Thread(target=self.vlm_worker, daemon=True)
                self.vlm_thread.start()
                logger.info("VLM worker started (process-based with hard 2s timeout)")

        # Start night mode if it was enabled (persisted setting)
        if self.autonomous_mode:
            # Pass self (MinusAdBlocker) not self.ad_blocker (AdBlocker)
            # Autonomous mode needs: audio module, last_ocr_texts
            self.autonomous_mode.set_ad_blocker(self)
            if hasattr(self, 'vlm') and self.vlm:
                self.autonomous_mode.set_vlm(self.vlm)
            if hasattr(self, 'frame_capture') and self.frame_capture:
                self.autonomous_mode.set_frame_capture(self.frame_capture)

            # Track autonomous active/inactive transitions and reflect on the
            # status LEDs. Don't override blocking — when an ad is blocked
            # while autonomous is running, blocking visuals win and we'll
            # come back to autonomous after hide().
            self._autonomous_was_active = False
            def _on_autonomous_status(status):
                active = bool(status.get('active'))
                if active and not self._autonomous_was_active:
                    self._autonomous_was_active = True
                    if not self.blocking_active:
                        self._set_led_state('autonomous')
                elif not active and self._autonomous_was_active:
                    self._autonomous_was_active = False
                    if not self.blocking_active:
                        # Fall back to whatever baseline applies (paused?
                        # — unlikely while autonomous was running but possible).
                        self._set_led_state(self._baseline_led_state())
            self.autonomous_mode.set_status_callback(_on_autonomous_status)
            self.autonomous_mode.start_if_enabled()

        logger.info("Minus running - press Ctrl+C to stop")

        # Monitor ustreamer
        try:
            restart_failures = 0
            while self.running:
                # Skip main loop restart if HDMI recovery is in progress (health monitor handles it)
                if self._hdmi_recovery_in_progress:
                    time.sleep(1)
                    continue

                # Skip restart if display is already disconnected (retry loop handles it)
                if not self.display_connected:
                    time.sleep(1)
                    continue

                if self.ustreamer_process and self.ustreamer_process.poll() is not None:
                    logger.warning("ustreamer process died, restarting...")
                    if not self.start_display():
                        restart_failures += 1
                        logger.error(f"Failed to restart display (attempt {restart_failures})")
                        # Don't exit - wait and retry (health monitor may also be handling this)
                        if restart_failures >= 5:
                            logger.error("Too many restart failures, exiting")
                            break
                        time.sleep(5)  # Wait before retry
                    else:
                        restart_failures = 0  # Reset on success
                        self.display_connected = True  # Update status
                        self.display_error = None
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.stop()
        return True

    def stop(self):
        """Stop everything."""
        logger.info("Stopping...")
        self.running = False

        # Stop WiFi manager
        if hasattr(self, 'wifi_manager') and self.wifi_manager:
            self.wifi_manager.stop_monitor()
            # Stop AP mode if active
            if self.wifi_manager._ap_mode_active:
                self.wifi_manager.stop_ap_mode()
            self.wifi_manager = None

        # Stop night mode
        if self.autonomous_mode:
            self.autonomous_mode.destroy()
            self.autonomous_mode = None

        # Release IR transmitter PWM
        if self.ir_transmitter:
            try:
                self.ir_transmitter.shutdown()
            except Exception:
                pass
            self.ir_transmitter = None

        # Stop status-LED animation thread and blank the strip
        if self.status_leds:
            try:
                self.status_leds.stop()
            except Exception:
                pass
            self.status_leds = None

        # Stop Fire TV setup first
        if self.fire_tv_setup:
            self.fire_tv_setup.destroy()
            self.fire_tv_setup = None
            self.fire_tv_controller = None

        # Stop health monitor
        if self.health_monitor:
            self.health_monitor.stop()

        # Stop web UI
        if self.webui:
            self.webui.stop()

        # Clean up frame capture temp file
        if self.frame_capture:
            self.frame_capture.cleanup()

        if self.ustreamer_process:
            self.ustreamer_process.terminate()
            try:
                self.ustreamer_process.wait(timeout=5)
            except:
                self.ustreamer_process.kill()

        # Stop ASR BEFORE tearing down audio so any in-flight whisper-cli
        # subprocess can finish/timeout cleanly while the audio pipeline
        # is still alive.
        if self.asr:
            try:
                self.asr.stop()
            except Exception as e:
                logger.debug(f"ASR stop error: {e}")

        if self.audio:
            self.audio.destroy()

        if self.ad_blocker:
            self.ad_blocker.destroy()

        if self.ocr:
            self.ocr.release()

        if self.vlm:
            self.vlm.release()

        # Restore console settings (show cursor, unblank, restore dmesg level)
        restore_console()

        logger.info("Stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Minus - HDMI passthrough with ML-based ad detection'
    )
    parser.add_argument(
        '--device', '-d',
        default='/dev/video0',
        help='Video device path (default: /dev/video0)'
    )
    parser.add_argument(
        '--screenshot-dir', '-s',
        default='screenshots',
        help='Directory to save screenshots (default: screenshots)'
    )
    parser.add_argument(
        '--check-signal',
        action='store_true',
        help='Just check HDMI signal and exit'
    )
    parser.add_argument(
        '--ocr-timeout',
        type=float,
        default=1.5,
        help='Skip OCR frames taking longer than this (seconds, default: 1.5)'
    )
    parser.add_argument(
        '--max-screenshots',
        type=int,
        default=0,
        help='Keep only this many recent screenshots (0=unlimited for training, default: 0)'
    )
    parser.add_argument(
        '--connector-id',
        type=int,
        default=None,
        help='DRM connector ID for HDMI output (auto-detected if not specified)'
    )
    parser.add_argument(
        '--plane-id',
        type=int,
        default=None,
        help='DRM plane ID for video overlay (auto-detected if not specified)'
    )
    parser.add_argument(
        '--webui-port',
        type=int,
        default=80,
        help='Web UI port (default: 80, requires root)'
    )
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR processing (for testing)'
    )
    parser.add_argument(
        '--no-vlm',
        action='store_true',
        help='Disable VLM processing (for testing)'
    )
    parser.add_argument(
        '--no-blocking',
        action='store_true',
        help='Disable all blocking overlays (for testing)'
    )

    args = parser.parse_args()

    # NOTE: Bandwidth fallback check moved to early init at top of file
    # (must run before GStreamer/DRM imports)

    config = MinusConfig(
        device=args.device,
        screenshot_dir=args.screenshot_dir,
        ocr_timeout=args.ocr_timeout,
        max_screenshots=args.max_screenshots,
        drm_connector_id=args.connector_id,
        drm_plane_id=args.plane_id,
        webui_port=args.webui_port,
        no_ocr=args.no_ocr,
        no_vlm=args.no_vlm,
        no_blocking=args.no_blocking,
    )

    minus = Minus(config)

    if args.check_signal:
        signal_info = minus.check_hdmi_signal()
        if signal_info:
            width, height, fps = signal_info
            print(f"HDMI signal detected: {width}x{height} @ {fps}fps")
            sys.exit(0)
        else:
            print("No HDMI signal detected")
            sys.exit(1)

    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        minus.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    minus.run()


if __name__ == '__main__':
    main()
