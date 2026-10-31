"""
Frame capture utilities for Minus.

Provides capture from ustreamer's HTTP snapshot endpoint.

Network resilience:
- Uses persistent HTTP session with connection pooling
- Graceful handling of connection failures
- Rate limiting to prevent HTTP contention
"""

import logging
import os
import random
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def ensure_localhost_available() -> bool:
    """
    Ensure localhost (127.0.0.1) is reachable.

    This should always be true, but can fail in edge cases during boot
    or network reconfiguration.

    Returns:
        True if localhost is reachable, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        result = sock.connect_ex(('127.0.0.1', 9090))
        sock.close()
        # connect_ex returns 0 on success, or error code
        # We don't care about the specific port, just that localhost works
        # Even connection refused (111) means localhost is working
        return result in (0, 111)  # 0 = connected, 111 = connection refused (port not open)
    except Exception:
        return False

# Global rate limiter to prevent HTTP contention when multiple workers
# (OCR + VLM) are requesting snapshots simultaneously
_capture_lock = threading.Lock()
_last_capture_time = 0.0
# NOTE: this limiter is GLOBAL — the OCR loop and the VLM loop share it — so
# each loop's effective cadence is roughly 2x the interval below.
# 0.5 -> 0.2 (2026-10-30): at 0.5 the two loops took turns and each waited
# ~0.4 s per frame for the other (cap=600-900 ms in the OCR log vs ~100 ms
# for the fetch itself), so OCR read the screen ~1x/s and ad blocks started
# a second later than they had to. Snapshots are frames ustreamer already
# encoded, and decode_for_detection made each one ~4x cheaper to decode.
_MIN_CAPTURE_INTERVAL = float(os.environ.get('MINUS_CAPTURE_MIN_INTERVAL', '0.2'))
# During a block the interval used to be RAISED to 1.0s, on the reasoning that
# the MPP encoder is busy compositing the overlay and "glitches during blocking
# are OK". That is true of the picture — the overlay covers it — but it also
# throttled the only signal that can tell us the ad ENDED. Measured over 36h of
# production: capture p50 488ms idle vs 1487ms during a block, so detection
# cadence went 1.0s -> 2.0s and the 3-frame OCR stop took ~6s. Recovery came in
# at p50 5.0s / p90 8.0s, i.e. seconds of real show missed after every break.
# Blocking is when we need frames MOST, so the interval now matches idle.
# OCR_STOP_THRESHOLD stays at 3 (tuned against the measured OCR miss
# distribution); this buys the latency back from cadence instead of by
# weakening the flap resistance.
_MIN_CAPTURE_INTERVAL_BLOCKING = float(
    os.environ.get('MINUS_CAPTURE_MIN_INTERVAL_BLOCKING', '0.2'))

# Blocking state cache to avoid repeated API calls
_blocking_state_cache = {'enabled': False, 'last_check': 0.0}
_BLOCKING_CHECK_INTERVAL = 1.0  # Check blocking state every 1 second (glitches during blocking are OK)

# Transition delay removed - didn't help reduce glitches

# Persistent HTTP session with connection pooling
_http_session = None
_session_lock = threading.Lock()

# Failed snapshots come in streaks: every source sleep, restart or HDMI drop
# times out /snapshot until ustreamer has frames again (44 ERROR lines a night
# on minus, every one of them a sleep or a restart; the health monitor reports
# the signal loss itself). Log a streak once when it starts and once when it
# ends.
_fail_lock = threading.Lock()
_fail_streak = 0
# urllib3 logs each of the session's internal retries as a WARNING ("Retrying
# (Retry(total=1, ...)) after connection broken by ReadTimeoutError"), 2 per
# failed snapshot; the streak above already reports the outage.
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)


def _note_capture_failure(e):
    global _fail_streak
    with _fail_lock:
        _fail_streak += 1
        first = _fail_streak == 1
    if first:
        logger.warning(f"Snapshot capture failing: {e} (repeats are not logged until it recovers)")


def _note_capture_success():
    global _fail_streak
    with _fail_lock:
        n, _fail_streak = _fail_streak, 0
    if n:
        logger.info(f"Snapshot capture recovered after {n} failed attempt{'s' if n != 1 else ''}")


def _get_http_session():
    """Get or create a persistent HTTP session with connection pooling."""
    global _http_session
    with _session_lock:
        if _http_session is None:
            _http_session = requests.Session()
            # Configure retry strategy
            retry = Retry(total=2, backoff_factor=0.1)
            adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=2)
            _http_session.mount('http://', adapter)
        return _http_session


def _is_blocking_active(port=9090):
    """Check if ustreamer blocking mode is active (with caching).

    Returns True if blocking overlay is currently being rendered,
    which means the MPP encoder is busy and we should slow down captures.

    Uses file-based state in /dev/shm for zero-overhead checks (Hypothesis 3).
    Falls back to HTTP if file doesn't exist.
    """
    global _blocking_state_cache

    now = time.time()
    if now - _blocking_state_cache['last_check'] < _BLOCKING_CHECK_INTERVAL:
        return _blocking_state_cache['enabled']

    # Try file-based check first (written by ad_blocker.py)
    try:
        with open('/dev/shm/minus_blocking_state', 'r') as f:
            enabled = f.read().strip() == '1'
            _blocking_state_cache = {'enabled': enabled, 'last_check': now}
            return enabled
    except FileNotFoundError:
        pass  # Fall back to HTTP
    except Exception:
        pass

    # Fall back to HTTP (slower)
    try:
        session = _get_http_session()
        response = session.get(f'http://localhost:{port}/blocking', timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            enabled = data.get('result', {}).get('enabled', False)
            _blocking_state_cache = {'enabled': enabled, 'last_check': now}
            return enabled
    except Exception:
        pass

    return _blocking_state_cache['enabled']


def jpeg_size(data: bytes):
    """(width, height) from a JPEG's SOF marker, or None. Reads headers only."""
    i, n = 2, len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    while i + 9 < n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:          # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = (data[i + 2] << 8) | data[i + 3]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return w, h
        i += 2 + seg
    return None


_REDUCED_FLAGS = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                  4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def decode_for_detection(data: bytes, img_array=None):
    """Decode a snapshot straight to (about) 960x540 for OCR/VLM.

    A 4K frame used to be decoded in full (~63 ms on minus) only to be
    shrunk to 960x540. libjpeg can decode at 1/2, 1/4 or 1/8 scale in the DCT
    domain: a 4K frame comes out at 960x540 directly in ~17 ms, PSNR 57.6 dB
    against decode+INTER_AREA (mean pixel difference 0.11). The largest
    reduction that stays >= 960x540 is used; any remainder is resized as before.
    """
    if img_array is None:
        img_array = np.frombuffer(data, dtype=np.uint8)
    size = jpeg_size(data)
    scale = 1
    if size:
        w, h = size
        while scale < 8 and w // (scale * 2) >= 960 and h // (scale * 2) >= 540:
            scale *= 2
    img = cv2.imdecode(img_array, _REDUCED_FLAGS[scale])
    if img is None and scale != 1:
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# /snapshot/raw older than this (X-Raw-Age, seconds) is not the current picture.
RAW_STALE_S = 1.0
RAW_STALE_RETRY_WAIT_S = 0.3


def raw_frame_is_stale(response) -> bool:
    """True when ustreamer says the raw snapshot it served is stale.

    Older ustreamer builds send no X-Raw-Age header: treated as fresh.
    """
    try:
        age = response.headers.get('X-Raw-Age')
        return response.status_code == 200 and age is not None and float(age) > RAW_STALE_S
    except (TypeError, ValueError, AttributeError):
        return False


class UstreamerCapture:
    """Frame capture using ustreamer's HTTP snapshot endpoint.

    Uses /snapshot/raw which:
    - Returns raw video when blocking is active (for OCR to see ad content)
    - Redirects to /snapshot when not blocking (normal operation)
    """

    def __init__(self, port=9090):
        self.port = port
        # Use /snapshot/raw to always get raw video, even during blocking
        # This is critical for OCR to detect when ads end
        self.snapshot_url = f'http://localhost:{port}/snapshot/raw'
        # Use PID-based filename to avoid conflicts with root-owned stale files
        self.screenshot_path = f'/dev/shm/minus_frame_{os.getpid()}.jpg'

    def cleanup(self):
        """Remove the temporary screenshot file."""
        try:
            Path(self.screenshot_path).unlink(missing_ok=True)
        except Exception:
            pass

    def capture(self):
        """Capture frame via HTTP snapshot and return as numpy array.

        Rate-limited globally (OCR and VLM share the limit) to
        _MIN_CAPTURE_INTERVAL / _MIN_CAPTURE_INTERVAL_BLOCKING.

        Uses persistent HTTP session with connection pooling to avoid
        subprocess overhead from curl.
        """
        global _last_capture_time

        try:
            # Dynamic rate limit based on blocking state
            # When blocking is active, MPP encoder is busy rendering overlays
            # so we slow down capture requests to prevent glitches
            is_blocking = _is_blocking_active(self.port)
            min_interval = _MIN_CAPTURE_INTERVAL_BLOCKING if is_blocking else _MIN_CAPTURE_INTERVAL

            # Rate limit: ensure minimum gap between HTTP requests
            # IMPORTANT: Only hold lock briefly for timing check, NOT during HTTP request
            # Otherwise a slow/stuck HTTP request blocks all workers
            with _capture_lock:
                now = time.time()
                elapsed = now - _last_capture_time
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    time.sleep(wait_time)
                _last_capture_time = time.time()  # Mark time BEFORE request

            # HTTP request OUTSIDE the lock - prevents cascade failure if ustreamer is slow
            session = _get_http_session()
            response = session.get(self.snapshot_url, timeout=2, allow_redirects=True)
            # During a block /snapshot/raw serves the last frame the encoder
            # stored. With no display attached nothing drives the encoder, so
            # that frame could be minutes old and OCR/VLM ended real ad blocks
            # because the stale picture showed no ad. ustreamer now reports
            # the frame's age and wakes the encoder when it is stale: wait for
            # that encode and ask once more.
            if raw_frame_is_stale(response):
                time.sleep(RAW_STALE_RETRY_WAIT_S)
                response = session.get(self.snapshot_url, timeout=2, allow_redirects=True)

            if response.status_code == 200:
                # Decode JPEG directly from memory (no disk I/O)
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = decode_for_detection(response.content, img_array)
                if img is not None:
                    # Scale to 960x540 for OCR - model uses 960x960 anyway
                    # Using INTER_AREA for best quality downscaling, fast on 4K->540p
                    h, w = img.shape[:2]
                    if h > 540 or w > 960:
                        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                    _note_capture_success()
                    return img

            return None
        except Exception as e:
            _note_capture_failure(e)
            return None
