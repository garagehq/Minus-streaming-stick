"""Keep GStreamer bus messages from piling up in memory.

Minus never runs a GLib main loop, so `bus.add_signal_watch()` watches never
dispatch, and every message a pipeline posts stays queued on its bus for the
pipeline's lifetime. That is harmless while little is posted, but a starved
alsasrc (HDMI source asleep) posts a "Can't record audio fast enough ...
Dropped N samples" WARNING every few milliseconds: ~800 MB per Fire TV
sleep/wake in the main process (found 2026-10-29).

drop_unconsumed_messages() registers the bus with a background drainer that,
every DRAIN_INTERVAL_S, discards every queued message except ERROR and EOS.
ERROR and EOS are kept: the code polls the bus for errors (`pop_filtered` /
`timed_pop_filtered`).

Why not a bus sync handler that returns DROP (the first version of this fix):
with PyGObject 3.42 every message handed to a Python sync handler leaks
(~0.8 KB each, measured: 200k messages took a test process from 19 to 179 MB
with the handler returning DROP), so the handler only slowed the leak
(~60 MB per source sleep instead of ~490 MB). Popping from Python does not
leak, and gst_bus_pop_filtered() unrefs the non-matching messages it skips in
C, so one call per bus per tick clears the backlog without a Python object
per message. The ERROR/EOS it returns is posted again so its owner still sees
it.
"""
import logging
import threading
import time

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst  # noqa: E402

logger = logging.getLogger(__name__)

_KEEP = Gst.MessageType.ERROR | Gst.MessageType.EOS
DRAIN_INTERVAL_S = 0.25
# ERROR/EOS messages kept per bus per tick. A pipeline that keeps posting
# errors nobody pops would otherwise grow the kept set without bound.
MAX_KEPT = 16

_lock = threading.Lock()
_buses = []          # strong refs; pruned once the pipeline lets go of a bus
_thread = None


def drain(bus):
    """Discard queued messages except ERROR/EOS (re-posted, newest MAX_KEPT).

    Returns how many messages were kept.
    """
    kept = []
    # Bounded so a bus that keeps returning messages can't spin this thread.
    for _ in range(MAX_KEPT * 8):
        # Discards (unrefs, in C) every message before the first ERROR/EOS.
        msg = bus.pop_filtered(_KEEP)
        if msg is None:
            break
        kept.append(msg)
    kept = kept[-MAX_KEPT:]
    for msg in kept:
        bus.post(msg)
    return len(kept)


def _orphaned(bus):
    # Our list holds the only reference: the pipeline that owned the bus is gone.
    try:
        return bus.__grefcount__ <= 1
    except Exception:
        return False


def _drain_loop():
    while True:
        time.sleep(DRAIN_INTERVAL_S)
        with _lock:
            _buses[:] = [b for b in _buses if not _orphaned(b)]
            buses = list(_buses)
        for bus in buses:
            try:
                drain(bus)
            except Exception as e:  # never let one bus stop the drainer
                logger.debug(f"[gst_bus] drain failed: {e}")


def drop_unconsumed_messages(bus):
    """Drop every bus message except ERROR / EOS (drained in the background)."""
    global _thread
    try:
        real = isinstance(bus, Gst.Bus)
    except TypeError:                   # Gst itself mocked in unit tests
        real = False
    if not real:                        # None, or a mock bus in unit tests
        return bus
    with _lock:
        if not any(b is bus for b in _buses):
            _buses.append(bus)
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_drain_loop, name='gst-bus-drain',
                                       daemon=True)
            _thread.start()
    drain(bus)
    return bus


def registered_count():
    with _lock:
        return len(_buses)
