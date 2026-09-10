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
_MIN_CAPTURE_INTERVAL = float(os.environ.get('MINUS_CAPTURE_MIN_INTERVAL', '0.5'))
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
    os.environ.get('MINUS_CAPTURE_MIN_INTERVAL_BLOCKING', '0.5'))

# Blocking state cache to avoid repeated API calls
_blocking_state_cache = {'enabled': False, 'last_check': 0.0}
_BLOCKING_CHECK_INTERVAL = 1.0  # Check blocking state every 1 second (glitches during blocking are OK)

# Transition delay removed - didn't help reduce glitches

# Persistent HTTP session with connection pooling
_http_session = None
_session_lock = threading.Lock()

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

        Uses dynamic rate limiting based on blocking state:
        - During normal operation: 500ms minimum between captures
        - During blocking: 1s minimum (MPP encoder busy with overlays)

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

            if response.status_code == 200:
                # Decode JPEG directly from memory (no disk I/O)
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    # Scale to 960x540 for OCR - model uses 960x960 anyway
                    # Using INTER_AREA for best quality downscaling, fast on 4K->540p
                    h, w = img.shape[:2]
                    if h > 540 or w > 960:
                        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
                    return img

            return None
        except Exception as e:
            logger.error(f"Snapshot capture error: {e}")
            return None
