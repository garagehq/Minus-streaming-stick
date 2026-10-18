"""Keep the audio delay matched to the measured video latency.

Audio passthrough delays the sound with a queue (`syncqueue` in audio.py) so it
leaves the box together with the picture. That delay used to be a constant
300 ms, calibrated in April 2026 when ustreamer took ~110 ms from capture to
output. The July 60 fps ustreamer change cut that to ~17 ms and nothing moved
the constant, so the sound arrived ~300 ms after the picture (measured on
minus, 2026-10-18: video ~105 ms, audio ~405 ms).

So the delay is no longer a number someone has to remember to retune: the
video latency is measured from ustreamer's own per-frame timing headers, and
the audio delay is set to it minus what the audio path adds by itself. A
saved trim (Settings) covers what can't be measured (a TV or projector that
delays its picture but not its sound).

Video latency = one source frame (the HDMI receiver writes a whole frame
before ustreamer can grab it) + ustreamer grab->send (headers) + the display
pipeline (JPEG decode + waiting for the next refresh, DISPLAY_SIDE_MS).
"""

import re
import socket
import statistics
import time

# Display side, measured on minus 2026-10-18 at 3840x2160: demux + mppjpegdec
# p50 10.5 ms / p90 16.2 ms (GStreamer latency tracer), plus kmssink's
# blocking flip, which waits for the next vblank (0-16.7 ms, ~8 average)
# and half a refresh until the middle of the screen is scanned out (~8).
DISPLAY_SIDE_MS = 28.0

# The audio path's own latency outside syncqueue: alsasrc latency-time
# (10 ms) + alsasink's buffer (buffer-time 50 ms, kept nearly full).
# audioqueue's level is measured live and added on top.
AUDIO_PATH_MS = 55.0

MAX_DELAY_MS = 800.0
TRIM_LIMIT_MS = 500.0

_GRAB = re.compile(rb'X-UStreamer-Grab-Time: ([0-9.]+)')
_SEND = re.compile(rb'X-UStreamer-Send-Time: ([0-9.]+)')


def frame_latencies_ms(data: bytes) -> list:
    """grab->send latency in ms per frame, pairing times within each part."""
    out = []
    for part in data.split(b'--boundarydonotcross'):
        head = part[:2048]
        g, s = _GRAB.search(head), _SEND.search(head)
        if g and s:
            lat = (float(s.group(1)) - float(g.group(1))) * 1000.0
            if lat >= 0:
                out.append(lat)
    return out


def measure_ustreamer_ms(port: int = 9090, frames: int = 10, timeout: float = 3.0,
                         host: str = '127.0.0.1'):
    """Median capture->send latency over `frames` frames, or None.

    Reads ustreamer's MJPEG stream with extra headers for a moment. Adds one
    short-lived client; ustreamer encodes once for all clients, so this costs
    only the copy of a few frames.
    """
    deadline = time.monotonic() + timeout
    buf = b''
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(b'GET /stream?extra_headers=1 HTTP/1.0\r\n\r\n')
            while time.monotonic() < deadline and len(buf) < (32 << 20):
                chunk = s.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
                if buf.count(b'X-UStreamer-Send-Time') >= frames + 3:
                    break
    except OSError:
        return None
    lat = frame_latencies_ms(buf)[2:]   # the first frames can be stale buffers
    return statistics.median(lat) if lat else None


def video_latency_ms(ustreamer_ms: float, source_fps: float = 60.0) -> float:
    fps = source_fps if source_fps and source_fps > 1 else 60.0
    return 1000.0 / fps + ustreamer_ms + DISPLAY_SIDE_MS


def target_delay_ms(video_ms: float, audioqueue_ms: float = 0.0, trim_ms: float = 0.0) -> float:
    """syncqueue delay that makes audio leave the box with the picture."""
    d = video_ms - AUDIO_PATH_MS - (audioqueue_ms or 0.0) + clamp_trim(trim_ms)
    return round(max(0.0, min(MAX_DELAY_MS, d)), 1)


def clamp_trim(trim_ms) -> float:
    try:
        t = float(trim_ms)
    except (TypeError, ValueError):
        return 0.0
    return max(-TRIM_LIMIT_MS, min(TRIM_LIMIT_MS, t))
