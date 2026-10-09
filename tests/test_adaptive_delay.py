#!/usr/bin/env python3
"""Tests for src/adaptive_delay.py: the live/delayed decision, and the queue
switch run against real GStreamer queues (two branches standing in for the
audio syncqueue and the video avdelay queue)."""
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import adaptive_delay
from adaptive_delay import DELAYED, LIVE, AdaptiveDelay, DelayModeDecider


def feed(d, start, end, loud, step=0.01):
    t = start
    while t < end:
        d.observe(loud, t)
        t += step


class TestDecider(unittest.TestCase):
    def setUp(self):
        self.d = DelayModeDecider(onset_s=0.2, quiet_s=5.0, gap_s=0.1)

    def test_quiet_from_start_goes_live_after_quiet_s(self):
        feed(self.d, 0, 4.9, False)
        self.assertEqual(self.d.decide(DELAYED, 4.9, started=0), DELAYED)
        self.assertEqual(self.d.decide(DELAYED, 5.0, started=0), LIVE)

    def test_short_click_stays_live(self):
        feed(self.d, 10, 10.08, True)
        feed(self.d, 10.08, 11, False)
        for t in (10.05, 10.08, 10.5):
            self.assertEqual(self.d.decide(LIVE, t, started=0), LIVE)

    def test_sustained_audio_goes_delayed_within_onset(self):
        feed(self.d, 10, 10.15, True)
        self.assertEqual(self.d.decide(LIVE, 10.15, started=0), LIVE)
        feed(self.d, 10.15, 10.21, True)
        self.assertEqual(self.d.decide(LIVE, 10.21, started=0), DELAYED)

    def test_dips_shorter_than_gap_keep_the_run(self):
        feed(self.d, 10, 10.1, True)
        feed(self.d, 10.1, 10.15, False)    # 50 ms dip
        feed(self.d, 10.15, 10.22, True)
        self.assertEqual(self.d.decide(LIVE, 10.22, started=0), DELAYED)

    def test_pause_in_speech_keeps_delay_until_quiet_s(self):
        feed(self.d, 10, 20, True)
        feed(self.d, 20, 24.9, False)
        self.assertEqual(self.d.decide(DELAYED, 24.9, started=0), DELAYED)
        self.assertEqual(self.d.decide(DELAYED, 25.01, started=0), LIVE)


class FakeSide:
    """Audio/video stand-in exposing one GStreamer queue."""

    def __init__(self, q, live_ns, delayed_ns, max_ns):
        self.q, self.live_ns, self.delayed_ns, self.max_ns = q, live_ns, delayed_ns, max_ns
        self.live = False
        self.queue_lock = threading.Lock()
        self.level_cb = None

    def delay_queue_plan(self, live):
        return self.q, self.live_ns if live else self.delayed_ns, self.max_ns

    def set_delay_live(self, live):
        self.live = live

    def set_input_level_callback(self, cb):
        self.level_cb = cb


class TestQueueSwitch(unittest.TestCase):
    """Two live branches through delay queues, like Minus's audio + video."""

    def setUp(self):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        self.Gst = Gst
        branch = ("videotestsrc is-live=true ! video/x-raw,framerate=50/1,width=32,height=32 ! "
                  "identity name=in{n} ! queue name=q{n} min-threshold-time={thr} "
                  "max-size-time={mx} max-size-buffers=0 max-size-bytes=0 ! "
                  "identity name=out{n} ! fakesink sync=false ")
        # a: "audio" 0.3 s baseline + 1.5 s delay; v: "video" 1.5 s delay.
        self.p = Gst.parse_launch(branch.format(n='a', thr=1_800_000_000, mx=2_000_000_000) +
                                  branch.format(n='v', thr=1_500_000_000, mx=3_500_000_000))
        self.stamps, self.lat, self.lock = {}, {'a': [], 'v': []}, threading.Lock()
        for n in 'av':
            self.p.get_by_name(f'in{n}').get_static_pad('src').add_probe(
                Gst.PadProbeType.BUFFER, self._in, n)
            self.p.get_by_name(f'out{n}').get_static_pad('src').add_probe(
                Gst.PadProbeType.BUFFER, self._out, n)
        self.audio = FakeSide(self.p.get_by_name('qa'), 300_000_000, 1_800_000_000, 2_000_000_000)
        self.video = FakeSide(self.p.get_by_name('qv'), 0, 1_500_000_000, 3_500_000_000)
        self.p.set_state(Gst.State.PLAYING)

    def tearDown(self):
        self.p.set_state(self.Gst.State.NULL)

    def _in(self, pad, info, n):
        with self.lock:
            self.stamps[(n, info.get_buffer().pts)] = time.monotonic()
        return self.Gst.PadProbeReturn.OK

    def _out(self, pad, info, n):
        now = time.monotonic()
        with self.lock:
            t = self.stamps.pop((n, info.get_buffer().pts), None)
            if t is not None:
                self.lat[n].append((now, now - t))
        return self.Gst.PadProbeReturn.OK

    def latency(self, n, since):
        with self.lock:
            xs = [l for t, l in self.lat[n] if t >= since]
        return (min(xs), max(xs)) if xs else None

    def count(self, n, a, b):
        with self.lock:
            return sum(1 for t, _ in self.lat[n] if a <= t < b)

    def test_live_then_delayed(self):
        ad = AdaptiveDelay(self.audio, self.video, 1.5)
        time.sleep(2.5)
        lo, hi = self.latency('v', time.monotonic() - 0.5)
        self.assertAlmostEqual(lo, 1.5, delta=0.05)

        t = time.monotonic()
        ad._switch(LIVE)
        flood = self.count('v', t, t + 0.2)
        self.assertLessEqual(flood, 15, 'backlog must be dropped, not flushed')
        time.sleep(1.0)
        self.assertLess(self.latency('v', time.monotonic() - 0.5)[1], 0.05)
        self.assertAlmostEqual(self.latency('a', time.monotonic() - 0.5)[0], 0.3, delta=0.05)
        self.assertTrue(self.audio.live and self.video.live)

        t = time.monotonic()
        ad._switch(DELAYED)
        time.sleep(1.0)
        self.assertEqual(self.count('v', t + 0.1, time.monotonic()), 0, 'video holds while refilling')
        self.assertEqual(self.count('a', t + 0.4, time.monotonic()), 0, 'audio holds while refilling')
        time.sleep(1.5)
        now = time.monotonic()
        self.assertAlmostEqual(self.latency('v', now - 0.5)[0], 1.5, delta=0.05)
        self.assertAlmostEqual(self.latency('a', now - 0.5)[0], 1.8, delta=0.05)
        self.assertEqual(ad.switches, {LIVE: 1, DELAYED: 1})


if __name__ == '__main__':
    unittest.main()
