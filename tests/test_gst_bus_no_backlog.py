"""GStreamer bus messages must not pile up in memory.

Found in the 2026-10-29 overnight soak: Minus never runs a GLib main loop, so the
`bus.add_signal_watch()` watches on every pipeline never dispatch and every posted
message stays queued on its bus. While the HDMI source sleeps, the audio
pipeline's alsasrc posts a "Can't record audio fast enough ... Dropped N samples"
WARNING every few milliseconds: the main process grew ~800 MB per Fire TV
sleep/wake and never gave it back. gst_bus.drop_unconsumed_messages() drops
everything except ERROR / EOS (the only types the code polls for) at post time.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
        return pipe, bus

    def test_warnings_do_not_accumulate(self):
        pipe, bus = self._bus()
        for i in range(2000):
            bus.post(Gst.Message.new_warning(pipe, _err(f"Dropped {i} samples"), "test"))
            bus.post(Gst.Message.new_state_changed(pipe, Gst.State.NULL, Gst.State.READY, Gst.State.VOID_PENDING))
        self.assertFalse(bus.have_pending())

    def test_errors_and_eos_stay_pollable(self):
        pipe, bus = self._bus()
        bus.post(Gst.Message.new_error(pipe, _err("boom"), "test"))
        bus.post(Gst.Message.new_eos(pipe))
        self.assertIsNotNone(bus.pop_filtered(Gst.MessageType.ERROR))
        self.assertIsNotNone(bus.pop_filtered(Gst.MessageType.EOS))


class TestEveryPipelineUsesIt(unittest.TestCase):
    def test_each_signal_watch_is_paired(self):
        """Every bus that gets add_signal_watch() must also get the drop handler."""
        root = os.path.join(os.path.dirname(__file__), '..', 'src')
        for name in ('audio.py', 'ad_blocker.py'):
            src = open(os.path.join(root, name)).read()
            self.assertEqual(src.count('add_signal_watch()'), src.count('drop_unconsumed_messages(self.bus)'), name)


if __name__ == '__main__':
    unittest.main()
