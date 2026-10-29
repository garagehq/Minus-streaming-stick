"""Keep GStreamer bus messages from piling up in memory.

Minus never runs a GLib main loop, so `bus.add_signal_watch()` watches never
dispatch, and every message a pipeline posts stays queued on its bus for the
pipeline's lifetime. That is harmless while little is posted, but a starved
alsasrc (HDMI source asleep) posts a "Can't record audio fast enough ...
Dropped N samples" WARNING every few milliseconds: ~800 MB per Fire TV
sleep/wake in the main process (found 2026-10-29).

drop_unconsumed_messages() installs a sync handler that drops, at post time,
every message type nothing reads. ERROR and EOS are kept: the code polls the
bus for errors (`pop_filtered` / `timed_pop_filtered`).
"""
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402

_KEEP = Gst.MessageType.ERROR | Gst.MessageType.EOS


def _sync_handler(bus, message, *_):
    if message.type & _KEEP:
        return Gst.BusSyncReply.PASS
    return Gst.BusSyncReply.DROP


def drop_unconsumed_messages(bus):
    """Drop every bus message except ERROR / EOS as it is posted."""
    if bus is not None:
        bus.set_sync_handler(_sync_handler)
    return bus
