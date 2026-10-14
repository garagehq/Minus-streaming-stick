"""On-screen layer over the video: the refill spinner and the BLOCKED badge.

Refill spinner. Adaptive delay (src/adaptive_delay.py) runs the box in real
time while the input is quiet and goes back to the full A/V delay when
programme audio starts. Going back means both delay queues stop output
until they have refilled, so the picture freezes for about the delay
(1.2-2.4 s). That is exactly when a viewer who just pressed play, or
finished seeking, feels the box lag; a dimmed layer with a spinner covers
the frozen picture for that time.

BLOCKED badge. While the name muter (src/name_mute.py) has the speakers
muted, a red "BLOCKED" badge sits in the top right corner (the same look as
the MUTED badge in the demo videos, tools/demo_compose.py). It follows the
scheduler's actual mute switch, which acts on the delayed output, so it is
on screen with the picture that goes with the muted audio.

Neither can be drawn into the video: the ustreamer overlay API draws on the
captured frames, which then wait in the delay line (and nothing comes out
of it while it refills). Instead a second, tiny GStreamer pipeline

    appsrc (BGRA layer) ! kmssink plane-id=<spare plane>

draws on its own DRM plane above the video plane. It shares the video
kmssink's DRM fd (only one fd can be DRM master), scales the 1920x1080
layer to full screen, and never touches the video pipeline: any failure
just leaves the layer off. Both share the one spare plane, so they are
composited into one layer here.

MINUS_LOADING_OVERLAY=0 turns the layer off (spinner and badge);
MINUS_BLOCKED_BADGE=0 just the badge. MINUS_LOADING_PLANE=<id> picks the
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
BADGE_ENABLED = os.environ.get('MINUS_BLOCKED_BADGE', '1') != '0'
PLANE_OVERRIDE = os.environ.get('MINUS_LOADING_PLANE')
W, H, FPS = 1920, 1080, 30
FRAMES = 24                  # one spinner turn every 0.8 s
DIM_ALPHA = 0.35             # the semi-transparent layer over the frozen picture
SPOKES = 12
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
# The demo videos' MUTED badge (1280x720: 70 px high, 44 px DejaVu Sans
# Bold, radius 12, rgba 210,30,30,235) scaled to 1080p.
BADGE_TEXT = 'BLOCKED'
BADGE_H, BADGE_FONT, BADGE_RADIUS, BADGE_PAD = 105, 66, 18, 57
BADGE_RGBA = (210, 30, 30, 235)
BADGE_MARGIN = 48            # from the top and right edges
BADGE_HOLD_S = 0.25          # mutes this close together keep one badge up
IDLE_CLOSE_S = 5.0           # release the plane after this long with nothing shown


def _premultiply(rgba):
    """uint8 RGBA (straight alpha) -> float32 premultiplied BGRA in 0..1."""
    x = rgba.astype(np.float32) / 255.0
    a = x[..., 3:4]
    return np.concatenate([x[..., 2::-1] * a, a], axis=2)


def render_spinner(n=FRAMES, size=None, spokes=SPOKES, dim=DIM_ALPHA, h=H):
    """Spinner animation: n premultiplied BGRA uint8 patches (size x size),
    already composited over the dim layer."""
    from PIL import Image, ImageDraw
    ss = 4                                   # supersample
    size = size or int(h * 0.16)             # ~16% of the screen height
    r_out, r_in, width = 0.46 * size, 0.24 * size, 0.09 * size
    out = []
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
        sa = np.asarray(layer.resize((size, size), Image.LANCZOS), np.float32) / 255.0
        # White spinner over the black dim layer ("over" compositing).
        px = np.empty((size, size, 4), np.float32)
        px[..., 0] = px[..., 1] = px[..., 2] = sa
        px[..., 3] = sa + dim * (1 - sa)
        out.append(np.round(px * 255).astype(np.uint8))
    return out


def render_badge(text=BADGE_TEXT, h=BADGE_H, font_px=BADGE_FONT, radius=BADGE_RADIUS,
                 pad=BADGE_PAD, rgba=BADGE_RGBA):
    """The red rounded badge with white bold text: premultiplied BGRA float."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(BOLD, font_px)
    probe = ImageDraw.Draw(Image.new('L', (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    w = int(x1 - x0) + 2 * pad
    ss = 4
    im = Image.new('RGBA', (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius * ss, fill=rgba)
    im = im.resize((w, h), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    # Ink box centred in the badge.
    d.text(((w - (x1 - x0)) / 2 - x0, (h - (y1 - y0)) / 2 - y0), text, font=font,
           fill=(255, 255, 255, 255))
    return _premultiply(np.asarray(im))


def spinner_origin(size, w=W, h=H):
    return (h - size) // 2, (w - size) // 2


def compose(spinner=None, badge=None, w=W, h=H, dim=DIM_ALPHA, margin=BADGE_MARGIN):
    """One layer frame (premultiplied BGRA uint8, h x w x 4): the dim layer
    and `spinner` patch if given, `badge` in the top right if given;
    neither = fully transparent."""
    f = np.zeros((h, w, 4), np.uint8)
    if spinner is not None:
        f[..., 3] = round(dim * 255)
        y0, x0 = spinner_origin(spinner.shape[0], w, h)
        f[y0:y0 + spinner.shape[0], x0:x0 + spinner.shape[1]] = spinner
    if badge is not None:
        bh, bw = badge.shape[:2]
        y0, x0 = margin, w - margin - bw
        under = f[y0:y0 + bh, x0:x0 + bw].astype(np.float32) / 255.0
        f[y0:y0 + bh, x0:x0 + bw] = np.round(
            (badge + under * (1 - badge[..., 3:4])) * 255).astype(np.uint8)
    return f


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


def for_video(video):
    """The one overlay for this video output (the spinner and the badge
    share its plane)."""
    ov = getattr(video, '_screen_overlay', None)
    if ov is None:
        ov = ScreenOverlay(video)
        video._screen_overlay = ov
    return ov


class ScreenOverlay:
    """The refill spinner and the BLOCKED badge on a spare DRM plane.

    show()/hide() drive the spinner, set_blocked() the badge; all return at
    once. One worker thread owns the plane: it opens the layer pipeline
    when something is to be shown, redraws on every change (and every
    spinner frame), and lets the plane go once nothing has been shown for
    IDLE_CLOSE_S.
    """

    def __init__(self, video):
        self.video = video                  # DRMAdBlocker
        self._cv = threading.Condition()
        self._spin = None                   # (done_fn, deadline, t0) while wanted
        self._badge_on = False
        self._badge_off_at = None           # pending badge hide (BADGE_HOLD_S)
        self._thread = None
        self._spinner = None                # rendered lazily
        self._badge = None
        self._plane_id = None
        self.shown_count = 0                # spinner shows that reached the screen
        self.badge_count = 0                # badge shows that reached the screen
        self.last_error = ''
        self.last_shown_s = None

    # -- requests (any thread) ------------------------------------------

    def show(self, done_fn, max_s):
        """Spinner until done_fn() is true, max_s passes or hide()."""
        if not ENABLED:
            return
        with self._cv:
            if self._spin is None:
                now = time.monotonic()
                self._spin = (done_fn, now + max_s, now)
            self._kick()

    def hide(self):
        with self._cv:
            self._spin = None
            self._cv.notify_all()

    def set_blocked(self, on: bool):
        """Badge on while the name muter has the audio muted."""
        if not (ENABLED and BADGE_ENABLED):
            return
        with self._cv:
            if on:
                self._badge_on, self._badge_off_at = True, None
            elif self._badge_on and self._badge_off_at is None:
                self._badge_off_at = time.monotonic() + BADGE_HOLD_S
            self._kick()

    def _kick(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True,
                                             name='screen-overlay')
            self._thread.start()
        self._cv.notify_all()

    # -- the plane ------------------------------------------------------

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

    def _plane(self):
        if PLANE_OVERRIDE:
            return int(PLANE_OVERRIDE)
        if self._plane_id is None:
            out = subprocess.run(['modetest', '-M', 'rockchip', '-p'], capture_output=True,
                                 text=True, timeout=5).stdout
            self._plane_id = pick_plane(parse_planes(out), getattr(self.video, 'plane_id', None))
        return self._plane_id

    def _open(self):
        """(pipeline, appsrc) drawing on the spare plane, or None."""
        sink = self._video_sink()
        if sink is None:
            return None                     # no display pipeline (headless, no signal)
        fd = sink.get_property('fd')
        if fd is None or fd < 0:
            return None
        plane = self._plane()
        if plane is None:
            self.last_error = 'no spare ARGB plane'
            return None
        if self._spinner is None:
            self._spinner = render_spinner()
            self._badge = render_badge()
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        pipe = Gst.parse_launch(
            f"appsrc name=src is-live=true format=time do-timestamp=true "
            f"caps=video/x-raw,format=BGRA,width={W},height={H},framerate={FPS}/1 ! "
            f"kmssink name=sink fd={fd} plane-id={plane} "
            f"connector-id={self.video.connector_id} "
            f'plane-properties="props,zpos={OVERLAY_ZPOS}" '
            f"can-scale=true fullscreen=true sync=false async=false skip-vsync=true")
        pipe.set_state(Gst.State.PLAYING)
        return pipe, pipe.get_by_name('src')

    @staticmethod
    def _close(pipe):
        try:
            from gi.repository import Gst
            pipe.set_state(Gst.State.NULL)
        except Exception:
            pass

    def _run(self):
        pipe = src = buf = None
        shown = (False, False)              # (spinner, badge) on the plane now
        k, idle_since = 0, None
        while True:
            with self._cv:
                now = time.monotonic()
                if self._badge_off_at is not None and now >= self._badge_off_at:
                    self._badge_on, self._badge_off_at = False, None
                if self._spin is not None:
                    done_fn, deadline, t0 = self._spin
                    try:
                        done = now >= deadline or done_fn()
                    except Exception:
                        done = True
                    if done:
                        self._spin = None
                        if shown[0]:
                            self.last_shown_s = round(now - t0, 2)
                want = (self._spin is not None, self._badge_on)
                if not any(want) and pipe is None:
                    self._cv.wait()
                    continue
            try:
                if any(want) and pipe is None:
                    opened = self._open()
                    if opened is None:
                        with self._cv:          # nothing to draw on: drop the requests
                            self._spin, self._badge_on, self._badge_off_at = None, False, None
                        continue
                    pipe, src = opened
                    from gi.repository import Gst
                    shown, idle_since = (False, False), None
                if want != shown:
                    if want[0] and not shown[0]:
                        self.shown_count += 1
                        k = 0
                    if want[1] and not shown[1]:
                        self.badge_count += 1
                    buf = compose(self._spinner[k % FRAMES] if want[0] else None,
                                  self._badge if want[1] else None)
                elif want[0]:
                    k += 1
                    sp = self._spinner[k % FRAMES]
                    y0, x0 = spinner_origin(sp.shape[0])
                    buf[y0:y0 + sp.shape[0], x0:x0 + sp.shape[1]] = sp
                if want != shown or want[0]:
                    ret = src.emit('push-buffer', Gst.Buffer.new_wrapped(buf.tobytes()))
                    if ret != Gst.FlowReturn.OK:
                        raise RuntimeError(f'push-buffer: {ret.value_nick}')
                    shown = want
                if any(want):
                    idle_since = None
                elif idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since > IDLE_CLOSE_S:
                    self._close(pipe)
                    pipe = src = buf = None
                    continue
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"[ScreenOverlay] layer failed: {e}")
                if pipe is not None:
                    self._close(pipe)
                pipe = src = buf = None
                with self._cv:
                    self._spin, self._badge_on, self._badge_off_at = None, False, None
                continue
            with self._cv:
                # Spinner frames at FPS; otherwise just watch for changes.
                self._cv.wait(1.0 / FPS if want[0] else 0.05)

    def get_status(self) -> dict:
        return {'enabled': ENABLED, 'badge_enabled': ENABLED and BADGE_ENABLED,
                'shown': self.shown_count, 'badge_shown': self.badge_count,
                'blocked_now': self._badge_on,
                'last_shown_s': self.last_shown_s, 'last_error': self.last_error}
