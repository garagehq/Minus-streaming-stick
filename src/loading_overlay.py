"""Loading spinner shown while the A/V delay line refills.

Adaptive delay (src/adaptive_delay.py) runs the box in real time while the
input is quiet and goes back to the full A/V delay when programme audio
starts. Going back means both delay queues stop output until they have
refilled, so the picture freezes for about the delay (1.2-2.4 s). That is
exactly when a viewer who just pressed play, or finished seeking, feels
the box lag; this puts a dimmed layer with a spinner over the frozen
picture for that time.

It cannot be drawn into the video: the ustreamer overlay API draws on the
captured frames, which then wait in the same delay line and nothing comes
out of it while it refills. Instead a second, tiny GStreamer pipeline

    appsrc (pre-rendered BGRA frames) ! kmssink plane-id=<spare plane>

draws on its own DRM plane above the video plane. It shares the video
kmssink's DRM fd (only one fd can be DRM master), scales a 960x540 layer
to full screen, and never touches the video pipeline: any failure just
leaves the spinner off.

MINUS_LOADING_OVERLAY=0 turns it off; MINUS_LOADING_PLANE=<id> picks the
plane instead of the automatic choice; MINUS_LOADING_ZPOS /
MINUS_LOADING_VIDEO_ZPOS set the stacking (see VIDEO_ZPOS).
"""
import logging
import os
import re
import subprocess
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

ENABLED = os.environ.get('MINUS_LOADING_OVERLAY', '1') != '0'
PLANE_OVERRIDE = os.environ.get('MINUS_LOADING_PLANE')
W, H, FPS = 960, 540, 30
FRAMES = 24                  # one turn every 0.8 s
DIM_ALPHA = 0.35             # the semi-transparent layer over the frozen picture
SPOKES = 12


def render_frames(w=W, h=H, n=FRAMES, dim=DIM_ALPHA, spokes=SPOKES):
    """Spinner animation as premultiplied BGRA bytes (DRM ARGB8888, which
    the RK3588 planes blend as pre-multiplied by default), plus one fully
    transparent frame to clear the plane with."""
    from PIL import Image, ImageDraw
    ss = 4                                   # supersample the spinner
    size = int(h * 0.16)                     # spinner box, ~16% of the height
    r_out, r_in, width = 0.46 * size, 0.24 * size, 0.09 * size
    frames = []
    for k in range(n):
        layer = Image.new('L', (size * ss, size * ss), 0)
        d = ImageDraw.Draw(layer)
        c = size * ss / 2
        for i in range(spokes):
            # Spoke i trails the head by i steps; the head turns with k.
            ang = 2 * np.pi * ((k / n) - i / spokes)
            level = int(255 * max(0.15, 1.0 - i / spokes))
            x0, y0 = c + np.sin(ang) * r_in * ss, c - np.cos(ang) * r_in * ss
            x1, y1 = c + np.sin(ang) * r_out * ss, c - np.cos(ang) * r_out * ss
            d.line([(x0, y0), (x1, y1)], fill=level, width=int(width * ss))
            for x, y in ((x0, y0), (x1, y1)):           # round caps
                rr = width * ss / 2
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=level)
        spin = np.asarray(layer.resize((size, size), Image.LANCZOS), np.float32) / 255.0
        a = np.full((h, w), dim, np.float32)
        y0, x0 = (h - size) // 2, (w - size) // 2
        # White spinner over the black dim layer ("over" compositing).
        sa = spin
        a[y0:y0 + size, x0:x0 + size] = sa + dim * (1 - sa)
        white = np.zeros((h, w), np.float32)             # premultiplied colour
        white[y0:y0 + size, x0:x0 + size] = sa
        px = np.empty((h, w, 4), np.uint8)
        v = np.round(white * 255).astype(np.uint8)
        px[..., 0] = v
        px[..., 1] = v
        px[..., 2] = v
        px[..., 3] = np.round(a * 255).astype(np.uint8)
        frames.append(px.tobytes())
    return frames, bytes(w * h * 4)


def parse_planes(text):
    """modetest -p output -> {id: {crtc, possible, formats, type, zpos, zpos_range, zpos_mutable}}."""
    planes, cur, prop = {}, None, None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r'^(\d+)\t(\d+)\t(\d+)\t.*\t(0x[0-9a-fA-F]+)\s*$', line)
        if m:
            cur = {'crtc': int(m.group(2)), 'possible': int(m.group(4), 16),
                   'formats': set(), 'type': None, 'zpos': None,
                   'zpos_range': None, 'zpos_mutable': False}
            planes[int(m.group(1))] = cur
            prop = None
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith('formats:'):
            cur['formats'] = set(s.split()[1:])
            continue
        pm = re.match(r'^\d+ ([A-Za-z_ ]+):$', s)
        if pm and line.startswith('\t') and not line.startswith('\t\t'):
            prop = pm.group(1)
            continue
        if prop == 'type' and s.startswith('value:'):
            cur['type'] = int(s.split()[1])
        elif prop == 'zpos':
            if s.startswith('flags:'):
                cur['zpos_mutable'] = 'immutable' not in s
            elif s.startswith('values:'):
                a, b = s.split()[1:3]
                cur['zpos_range'] = (int(a), int(b))
            elif s.startswith('value:') and cur['zpos'] is None:
                cur['zpos'] = int(s.split()[1])
    return planes


# Fixed stacking: the video plane is pinned at VIDEO_ZPOS (src/ad_blocker.py
# sets it on its kmssink while this feature is on) and the spinner plane at
# OVERLAY_ZPOS. Pinning both is needed because the driver reports the usual
# video plane (192) at zpos 14, above every other plane's settable range
# (0-11), so "a spare plane above the video" does not exist otherwise.
VIDEO_ZPOS = int(os.environ.get('MINUS_LOADING_VIDEO_ZPOS', '10'))
OVERLAY_ZPOS = int(os.environ.get('MINUS_LOADING_ZPOS', '11'))


def pick_plane(planes, video_plane, zpos=OVERLAY_ZPOS):
    """A spare ARGB plane that can share the video plane's CRTC and take
    `zpos`, preferring Overlay-type planes; None if there is none."""
    video = planes.get(video_plane)
    if video is None:
        return None
    cands = []
    for pid, p in planes.items():
        zr = p['zpos_range']
        if (pid == video_plane or p['crtc'] != 0 or 'AR24' not in p['formats']
                or p['type'] == 2 or not (p['possible'] & video['possible'])
                or not p['zpos_mutable'] or not zr or not zr[0] <= zpos <= zr[1]):
            continue
        cands.append((0 if p['type'] == 0 else 1, pid))
    return min(cands)[1] if cands else None


class LoadingOverlay:
    """Spinner on a spare DRM plane while the delay line refills."""

    def __init__(self, video):
        self.video = video                  # DRMAdBlocker
        self._frames = None
        self._clear = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.shown_count = 0
        self.last_error = ''
        self.last_shown_s = None

    def _video_sink(self):
        pipe = getattr(self.video, 'pipeline', None)
        if pipe is None:
            return None
        it = pipe.iterate_elements()
        while True:
            ok, el = it.next()
            if ok != 1:                     # Gst.IteratorResult.OK
                return None
            f = el.get_factory()
            if f is not None and f.get_name() == 'kmssink':
                return el

    def show(self, done_fn, max_s):
        """Show until done_fn() is true or max_s passes. Non-blocking."""
        if not ENABLED:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, args=(done_fn, max_s),
                                            daemon=True, name='loading-overlay')
            self._thread.start()

    def hide(self):
        self._stop.set()

    def _plane(self):
        if PLANE_OVERRIDE:
            return int(PLANE_OVERRIDE)
        out = subprocess.run(['modetest', '-M', 'rockchip', '-p'], capture_output=True,
                             text=True, timeout=5).stdout
        return pick_plane(parse_planes(out), getattr(self.video, 'plane_id', None))

    def _run(self, done_fn, max_s):
        t0 = time.monotonic()
        pipe = None
        try:
            sink = self._video_sink()
            if sink is None:
                return                      # no display pipeline (headless, no signal)
            fd = sink.get_property('fd')
            if fd is None or fd < 0:
                return
            plane = self._plane()
            if plane is None:
                self.last_error = 'no spare ARGB plane'
                return
            if self._frames is None:
                self._frames, self._clear = render_frames()
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            props = f'plane-properties="props,zpos={OVERLAY_ZPOS}" '
            pipe = Gst.parse_launch(
                f"appsrc name=src is-live=true format=time do-timestamp=true "
                f"caps=video/x-raw,format=BGRA,width={W},height={H},framerate={FPS}/1 ! "
                f"kmssink name=sink fd={fd} plane-id={plane} "
                f"connector-id={self.video.connector_id} {props}"
                f"can-scale=true fullscreen=true sync=false async=false skip-vsync=true")
            src = pipe.get_by_name('src')
            pipe.set_state(Gst.State.PLAYING)
            self.shown_count += 1
            k = 0
            next_t = time.monotonic()
            while not self._stop.is_set() and time.monotonic() - t0 < max_s and not done_fn():
                src.emit('push-buffer', Gst.Buffer.new_wrapped(self._frames[k % len(self._frames)]))
                k += 1
                next_t += 1.0 / FPS
                time.sleep(max(0.0, next_t - time.monotonic()))
            # Leave the plane transparent before letting go of it, in case
            # the sink does not disable it on stop.
            src.emit('push-buffer', Gst.Buffer.new_wrapped(self._clear))
            time.sleep(2.0 / FPS)
            self.last_shown_s = round(time.monotonic() - t0, 2)
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"[LoadingOverlay] spinner failed: {e}")
        finally:
            if pipe is not None:
                try:
                    from gi.repository import Gst
                    pipe.set_state(Gst.State.NULL)
                except Exception:
                    pass

    def get_status(self) -> dict:
        return {'enabled': ENABLED, 'shown': self.shown_count,
                'last_shown_s': self.last_shown_s, 'last_error': self.last_error}
