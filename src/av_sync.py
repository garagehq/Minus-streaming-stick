"""Keep the audio delay matched to the measured video latency.

Audio passthrough delays the sound with a queue (`syncqueue` in audio.py) so it
leaves the box together with the picture. That delay used to be a constant
300 ms, calibrated in April 2026 when ustreamer took ~110 ms from capture to
output. The July 60 fps ustreamer change cut that to ~17 ms and nothing moved
the constant, so the sound arrived ~300 ms after the picture (measured on
minus, 2026-10-18: video ~105 ms, audio ~405 ms).

So the delay is no longer a number someone has to remember to retune. It is

    box video latency      measured live from ustreamer's per-frame headers
  + display video latency  what the TV/projector says in its EDID (HDMI
                           lip-sync fields; the Optoma UHD on minus: 72 ms)
  - display audio latency  same EDID block
  - box audio latency      the audio path's own buffers, measured live
  + learned offset         per display (keyed by its EDID), set by the owner's
                           -/+ in Settings: what neither measurement can see.
                           On minus 10-18 the projector's picture mode added
                           ~200 ms beyond its EDID figure.

so a change on the box (capture rate, resolution, a ustreamer restart) is
followed automatically, a different TV brings its own EDID figures and its own
learned offset, and the owner's correction is remembered per display.

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

# The audio path's latency beyond syncqueue's min-threshold-time. It was
# assumed to be +55 ms (alsasrc period + alsasink buffer), but measured from
# the kernel's ALSA pointers (capture appl_ptr - playback hw_ptr, minus,
# 2026-10-24) the sound reaches the DAC ~20 ms SOONER than the threshold:
# syncqueue's level is in buffer timestamps, which run ahead of the samples
# it really holds (level 360 ms vs 310 ms of samples in the process). With +55
# every delay came out ~75 ms short, i.e. the sound early.
AUDIO_PATH_MS = -20.0

# audioqueue (after syncqueue) absorbs the drift between the HDMI-in and
# HDMI-out audio clocks: it creeps up and, being leaky, drops its oldest audio
# when full. At its old 100 ms cap that was a sawtooth -- the sound slid
# ~100 ms later over minutes, then snapped back -- and a calibration that read
# its level at one moment and subtracted it put the sound 60 ms early as soon
# as it emptied (minus, 2026-10-18 04:02). Capped at 30 ms it drops ~10 ms
# buffers instead (inaudible), and the delay counts its typical level rather
# than chasing the momentary one.
AUDIOQUEUE_MAX_MS = 30.0
AUDIOQUEUE_TYPICAL_MS = 15.0

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


def target_delay_ms(video_ms: float, audioqueue_ms: float = 0.0, trim_ms: float = 0.0,
                    display_audio_ms: float = 0.0) -> float:
    """syncqueue delay that makes the sound and the picture come out together.

    video_ms is the whole picture path (box + display); display_audio_ms is
    what the display adds to the sound, which the delay doesn't need to add.
    """
    d = (video_ms - (display_audio_ms or 0.0) - AUDIO_PATH_MS - (audioqueue_ms or 0.0)
         + clamp_trim(trim_ms))
    # callers pass AUDIOQUEUE_TYPICAL_MS for audioqueue_ms, not a live reading
    return round(max(0.0, min(MAX_DELAY_MS, d)), 1)


# --- the display's own latency (EDID, HDMI 1.4 lip-sync fields) ------------

HDMI_OUI = 0x000C03


def _latency_ms(raw):
    """EDID latency byte -> ms. 0 = unknown, 255 = that path isn't supported."""
    if raw in (None, 0, 255):
        return None
    return 2 * (raw - 1)


def parse_edid(edid: bytes):
    """Name, identity and lip-sync latencies of a display, from its EDID.

    Returns None for an empty/short EDID. Latencies are None when the display
    doesn't advertise them (many don't); callers treat that as 0.
    """
    if not edid or len(edid) < 128:
        return None
    name = ''
    for i in range(54, 126, 18):
        if edid[i:i + 3] == b'\x00\x00\x00' and edid[i + 3] == 0xFC:
            name = edid[i + 5:i + 18].split(b'\n')[0].decode('ascii', 'replace').strip()
    info = {
        'name': name or 'display',
        # manufacturer + product code + serial: the same model with a
        # different serial is a different display (its picture mode, too).
        'id': edid[8:16].hex(),
        'video_ms': None,
        'audio_ms': None,
    }
    for blk in range(1, len(edid) // 128):
        b = edid[blk * 128:(blk + 1) * 128]
        if b[0] != 0x02:          # CTA-861 extension
            continue
        end, i = b[2], 4
        while i < min(end, 127):
            tag, ln = b[i] >> 5, b[i] & 31
            body = b[i + 1:i + 1 + ln]
            if tag == 3 and ln >= 3 and (body[0] | body[1] << 8 | body[2] << 16) == HDMI_OUI:
                if ln >= 10 and body[7] & 0x80:   # Latency_Fields_Present
                    info['video_ms'] = _latency_ms(body[8])
                    info['audio_ms'] = _latency_ms(body[9])
            i += 1 + ln
    return info


def connected_display(drm_root: str = '/sys/class/drm'):
    """parse_edid() of the connected HDMI output, or None."""
    import glob
    import os
    for status in sorted(glob.glob(os.path.join(drm_root, 'card*-HDMI-A-*', 'status'))):
        try:
            with open(status) as f:
                if f.read().strip() != 'connected':
                    continue
            with open(os.path.join(os.path.dirname(status), 'edid'), 'rb') as f:
                info = parse_edid(f.read())
            if info:
                return info
        except OSError:
            continue
    return None


class DelaySmoother:
    """Turn noisy calibrations into rare, deliberate changes.

    ustreamer's capture->send latency moves 30-60 ms between reads, and every
    change of the delay is audible (a gap or a skip). A change is applied only
    when it is bigger than `band` AND two calibrations in a row agree on it
    (their mean is applied); `force` (the owner pressed -/+, or nothing has
    been applied since start) applies the target at once.
    """

    def __init__(self, band: float = 40.0):
        self.band = band
        self.pending = None

    def propose(self, target: float, current: float, force: bool = False):
        if force:
            self.pending = None
            return target
        if abs(target - current) < self.band:
            self.pending = None
            return None
        if self.pending is not None and abs(target - self.pending) <= self.band:
            value = round((target + self.pending) / 2.0, 1)
            self.pending = None
            return value
        self.pending = target
        return None


def clamp_trim(trim_ms) -> float:
    try:
        t = float(trim_ms)
    except (TypeError, ValueError):
        return 0.0
    return max(-TRIM_LIMIT_MS, min(TRIM_LIMIT_MS, t))
