"""GStreamer bus messages must not pile up in memory.

Found in the 2026-10-29 overnight soak: Minus never runs a GLib main loop, so the
`bus.add_signal_watch()` watches on every pipeline never dispatch and every posted
message stays queued on its bus. While the HDMI source sleeps, the audio
pipeline's alsasrc posts a "Can't record audio fast enough ... Dropped N samples"
WARNING every few milliseconds: the main process grew ~800 MB per Fire TV
sleep/wake and never gave it back. gst_bus.drop_unconsumed_messages() registers
the bus with a drainer that discards everything except ERROR / EOS (the only
types the code polls for).

The first fix used a Python bus sync handler returning DROP; PyGObject 3.42 leaks
every message passed to it, so memory still grew ~60 MB per sleep on minus-2
(2026-10-29). test_no_memory_growth runs in a subprocess and fails on that leak.
"""
import os
import subprocess
import sys
import time
import unittest

SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC)

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    Gst.init(None)
except Exception:  # dev hosts without GStreamer
    Gst = None


def _err(text):
    return GLib.Error.new_literal(Gst.CoreError.quark(), text, Gst.CoreError.FAILED)


@unittest.skipIf(Gst is None, "needs GStreamer (runs on the box)")
class TestBusBacklog(unittest.TestCase):
    def _bus(self):
        from gst_bus import drop_unconsumed_messages
        pipe = Gst.Pipeline.new("t")
        bus = pipe.get_bus()
        bus.add_signal_watch()          # what the pipelines do (inert without a main loop)
        drop_unconsumed_messages(bus)
        self.addCleanup(bus.remove_signal_watch)
        return pipe, bus

    def _post_noise(self, pipe, bus, n):
        for i in range(n):
            bus.post(Gst.Message.new_warning(pipe, _err(f"Dropped {i} samples"), "test"))
            bus.post(Gst.Message.new_state_changed(pipe, Gst.State.NULL, Gst.State.READY,
                                                   Gst.State.VOID_PENDING))

    def test_warnings_do_not_accumulate(self):
        from gst_bus import drain
        pipe, bus = self._bus()
        self._post_noise(pipe, bus, 2000)
        drain(bus)
        self.assertFalse(bus.have_pending())

    def test_background_drainer_clears_backlog(self):
        from gst_bus import DRAIN_INTERVAL_S
        pipe, bus = self._bus()
        self._post_noise(pipe, bus, 500)
        deadline = time.monotonic() + 10 * DRAIN_INTERVAL_S + 1
        while bus.have_pending() and time.monotonic() < deadline:
            time.sleep(DRAIN_INTERVAL_S / 2)
        self.assertFalse(bus.have_pending())

    def test_errors_and_eos_stay_pollable(self):
        from gst_bus import drain
        pipe, bus = self._bus()
        self._post_noise(pipe, bus, 50)
        bus.post(Gst.Message.new_error(pipe, _err("boom"), "test"))
        self._post_noise(pipe, bus, 50)
        bus.post(Gst.Message.new_eos(pipe))
        drain(bus)
        drain(bus)                      # kept messages survive repeated drains
        err = bus.pop_filtered(Gst.MessageType.ERROR)
        self.assertIsNotNone(err)
        self.assertEqual(err.parse_error()[0].message, "boom")
        self.assertIsNotNone(bus.pop_filtered(Gst.MessageType.EOS))

    def test_kept_errors_are_capped(self):
        from gst_bus import drain, MAX_KEPT
        pipe, bus = self._bus()
        for i in range(MAX_KEPT * 3):
            bus.post(Gst.Message.new_error(pipe, _err(f"e{i}"), "test"))
        self.assertEqual(drain(bus), MAX_KEPT)
        n = 0
        while bus.pop_filtered(Gst.MessageType.ERROR) is not None:
            n += 1
        self.assertEqual(n, MAX_KEPT)

    def test_bus_released_after_pipeline_teardown(self):
        import gc
        import gst_bus
        from gst_bus import DRAIN_INTERVAL_S
        before = gst_bus.registered_count()
        pipe = Gst.parse_launch('fakesrc num-buffers=1 ! fakesink')
        bus = pipe.get_bus()
        bus.add_signal_watch()
        gst_bus.drop_unconsumed_messages(bus)
        self.assertEqual(gst_bus.registered_count(), before + 1)
        # Teardown as audio.py / ad_blocker.py do it.
        bus.remove_signal_watch()
        pipe.set_state(Gst.State.NULL)
        del pipe, bus
        gc.collect()
        deadline = time.monotonic() + 10 * DRAIN_INTERVAL_S + 1
        while gst_bus.registered_count() > before and time.monotonic() < deadline:
            time.sleep(DRAIN_INTERVAL_S / 2)
        self.assertLessEqual(gst_bus.registered_count(), before)


LEAK_SCRIPT = r'''
import sys, gc
sys.path.insert(0, sys.argv[1])
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)
from gst_bus import drop_unconsumed_messages, drain
pipe = Gst.parse_launch('fakesrc num-buffers=1 ! fakesink name=s')
el = pipe.get_by_name('s')
bus = pipe.get_bus()
bus.add_signal_watch()
drop_unconsumed_messages(bus)
def rss():
    return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
err = GLib.Error.new_literal(Gst.CoreError.quark(), "Can't record audio fast enough", Gst.CoreError.FAILED)
dbg = "gstaudiobasesrc.c(851): Dropped 89760 samples. " * 3
for i in range(20000):          # warm up allocator pools
    el.post_message(Gst.Message.new_warning(el, err, dbg))
drain(bus); gc.collect()
r0 = rss()
for i in range(200000):
    el.post_message(Gst.Message.new_warning(el, err, dbg))
    if i % 2000 == 0:
        drain(bus)
drain(bus); gc.collect()
print(rss() - r0)
'''


@unittest.skipIf(Gst is None, "needs GStreamer (runs on the box)")
class TestNoLeak(unittest.TestCase):
    def test_no_memory_growth(self):
        """200k warnings through the drainer: memory must stay flat.

        The old sync-handler version grew ~160 MB here.
        """
        out = subprocess.run([sys.executable, '-c', LEAK_SCRIPT, SRC],
                             capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr)
        growth_mb = int(out.stdout.strip().splitlines()[-1])
        self.assertLess(growth_mb, 10, f"RSS grew {growth_mb} MB over 200k messages")


class TestEveryPipelineUsesIt(unittest.TestCase):
    def test_each_signal_watch_is_paired(self):
        """Every bus that gets add_signal_watch() must also be registered."""
        for name in ('audio.py', 'ad_blocker.py'):
            src = open(os.path.join(SRC, name)).read()
            self.assertEqual(src.count('add_signal_watch()'), src.count('drop_unconsumed_messages(self.bus)'), name)


if __name__ == '__main__':
    unittest.main()
