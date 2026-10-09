"""Adaptive A/V delay: real time while the input is quiet, the full delay once
programme audio starts.

The A/V delay line (config.AV_DELAY_S, src/audio.py syncqueue + the avdelay
queue in src/ad_blocker.py) gives the name muter its head start, but it also
makes menus, the remote and the cursor feel ~2.4 s laggy. Nothing needs
muting while the source is silent, so:

  quiet input for QUIET_S  -> LIVE: both delay queues drop their backlog
                              (which is all silence by then) and pass
                              through. Picture jumps forward by the delay.
  sustained audio, ONSET_S -> DELAYED: both queues are told to hold the full
                              delay again; they stop output until they have
                              refilled, so the picture freezes for ~the delay
                              and then plays on, delayed and in sync.

The onset is caught on the INPUT side of the audio delay queue, and in live
mode that queue still holds its 300 ms video-matching baseline, so with
ONSET_S under 300 ms the first sound is still in the queue when the hold
starts: the opening words play delayed (and mutable) rather than live.
Short UI sounds (remote clicks) are shorter than ONSET_S and do not trigger.

GstQueue does both moves without leaving PLAYING (raising min-threshold-time
pauses output until refilled; leaky=downstream with a small max-size-time
discards the oldest buffers), which tools/ and the tests exercised.

MINUS_ADAPTIVE_DELAY=0 turns it off (fixed delay, as before).
"""
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

ENABLED = os.environ.get('MINUS_ADAPTIVE_DELAY', '1') != '0'
QUIET_DBFS = float(os.environ.get('MINUS_ADAPTIVE_QUIET_DBFS', '-60'))
ONSET_S = float(os.environ.get('MINUS_ADAPTIVE_ONSET_S', '0.2'))
QUIET_S = float(os.environ.get('MINUS_ADAPTIVE_QUIET_S', '5.0'))
GAP_S = 0.1          # a loud run survives dips this short
TICK_S = 0.02

LIVE, DELAYED = 'live', 'delayed'


class DelayModeDecider:
    """Pure decision logic (unit-tested): feed it loud/quiet observations,
    ask it which mode the delay line should be in."""

    def __init__(self, onset_s=ONSET_S, quiet_s=QUIET_S, gap_s=GAP_S):
        self.onset_s, self.quiet_s, self.gap_s = onset_s, quiet_s, gap_s
        self.loud_since = None
        self.last_loud = None

    def observe(self, loud: bool, now: float):
        if not loud:
            return
        if self.last_loud is None or now - self.last_loud > self.gap_s:
            self.loud_since = now
        self.last_loud = now

    def decide(self, mode: str, now: float, started: float) -> str:
        if mode == LIVE:
            if (self.last_loud is not None and now - self.last_loud <= self.gap_s
                    and self.last_loud - self.loud_since >= self.onset_s):
                return DELAYED
            return LIVE
        quiet_from = self.last_loud if self.last_loud is not None else started
        return LIVE if now - quiet_from >= self.quiet_s else DELAYED


class AdaptiveDelay:
    """Drives the audio and video delay queues from the input level."""

    def __init__(self, audio, video, delay_s: float):
        self.audio, self.video, self.delay_s = audio, video, delay_s
        self.decider = DelayModeDecider()
        self.mode = DELAYED            # both pipelines are built delayed
        self.switches = {LIVE: 0, DELAYED: 0}
        self.last_switch = None
        self.input_dbfs = None
        self._started = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.audio.set_input_level_callback(self._on_level)
        self._thread = threading.Thread(target=self._loop, daemon=True, name='adaptive-delay')
        self._thread.start()
        logger.info(f"[AdaptiveDelay] on: live when quiet (<{QUIET_DBFS:.0f} dBFS) for "
                    f"{QUIET_S:.1f}s, {self.delay_s:.1f}s delay after {ONSET_S:.2f}s of audio")

    def stop(self):
        self._stop.set()
        self.audio.set_input_level_callback(None)

    def _on_level(self, dbfs: float, now: float):
        # GStreamer streaming thread: just record.
        self.input_dbfs = dbfs
        self.decider.observe(dbfs >= QUIET_DBFS, now)

    def _loop(self):
        while not self._stop.wait(TICK_S):
            want = self.decider.decide(self.mode, time.monotonic(), self._started)
            if want != self.mode:
                try:
                    self._switch(want)
                except Exception as e:
                    logger.warning(f"[AdaptiveDelay] switch to {want} failed: {e}")

    def _switch(self, mode: str):
        live = mode == LIVE
        t = time.monotonic()
        queues = [q for q in (self.audio.delay_queue_plan(live), self.video.delay_queue_plan(live)) if q]
        with self.audio.queue_lock:
            if live:
                # Drop the (silent) backlog: shrink + leak first so nothing
                # floods out when the threshold falls, then pass through.
                for q, thr, mx in queues:
                    q.set_property('max-size-time', max(thr, 50_000_000))
                    q.set_property('leaky', 2)
                time.sleep(0.1)
                for q, thr, mx in queues:
                    q.set_property('min-threshold-time', thr)
                time.sleep(0.05)
                for q, thr, mx in queues:
                    q.set_property('leaky', 0)
                    q.set_property('max-size-time', mx)
            else:
                # Hold: output pauses until each queue has refilled to the
                # full delay; both fill in real time so they resume together.
                for q, thr, mx in queues:
                    q.set_property('max-size-time', mx)
                    q.set_property('min-threshold-time', thr)
            self.audio.set_delay_live(live)
            self.video.set_delay_live(live)
        self.mode = mode
        self.switches[mode] += 1
        self.last_switch = time.time()
        level = 'n/a' if self.input_dbfs is None else f"{self.input_dbfs:.0f} dBFS"
        logger.info(f"[AdaptiveDelay] -> {mode} ({len(queues)} queues, "
                    f"{(time.monotonic() - t) * 1000:.0f} ms, input {level})")

    def get_status(self) -> dict:
        return {'enabled': True, 'mode': self.mode, 'delay_s': self.delay_s,
                'input_dbfs': None if self.input_dbfs is None else round(self.input_dbfs, 1),
                'quiet_dbfs': QUIET_DBFS, 'onset_s': ONSET_S, 'quiet_s': QUIET_S,
                'switches': dict(self.switches), 'last_switch': self.last_switch}
