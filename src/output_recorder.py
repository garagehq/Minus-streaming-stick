"""Record the audio Minus actually outputs (after the A/V delay and the mute).

Opt-in for demos and A/B recordings: set MINUS_RECORD_OUTPUT_DIR and the
audio pipeline tees the playback branch, after the `vol` element, into an
appsink. Samples are appended to <dir>/output_<start>.s16 (48 kHz stereo
S16LE) and every buffer's arrival time on the box's monotonic clock goes to
<dir>/output_<start>.idx as "mono_time total_frames" lines, so a recording
can be lined up with anything else stamped with time.monotonic() (video
frames, logs). Arrival time is output time: the branch runs after the delay
queue, and with no TV attached the sink is a non-syncing fakesink.

A muted stretch is exact digital silence (the volume element writes zeros).

The appsink must not preroll (async=false): its first buffer only arrives
after the delay line fills, and AudioPassthrough gives a new pipeline 2 s to
reach PLAYING.
"""
import logging
import os
import time

logger = logging.getLogger(__name__)

RECORD_DIR = os.environ.get('MINUS_RECORD_OUTPUT_DIR', '')


def tee_chain(sink):
    """Playback tail: `vol ! sink`, plus the recording branch when enabled."""
    if not RECORD_DIR:
        return sink
    return (f"tee name=outtee allow-not-linked=true "
            f"outtee. ! queue ! {sink} "
            f"outtee. ! queue max-size-time=2000000000 ! "
            f"appsink name=rec_sink emit-signals=true sync=false async=false drop=false max-buffers=200")


class OutputRecorder:
    def __init__(self):
        os.makedirs(RECORD_DIR, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.raw = open(os.path.join(RECORD_DIR, f'output_{stamp}.s16'), 'ab')
        self.idx = open(os.path.join(RECORD_DIR, f'output_{stamp}.idx'), 'a')
        self.frames = 0
        logger.info(f"[OutputRecorder] recording output audio to {self.raw.name}")

    def attach(self, pipeline):
        sink = pipeline.get_by_name('rec_sink')
        if sink is not None:
            sink.connect('new-sample', self._on_sample)

    def _on_sample(self, sink):
        from gi.repository import Gst
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                self.raw.write(info.data)
                self.frames += len(info.data) // 4
                self.idx.write(f"{time.monotonic():.4f} {self.frames}\n")
            finally:
                buf.unmap(info)
            if self.frames % 48000 < 2048:
                self.raw.flush()
                self.idx.flush()
        return Gst.FlowReturn.OK


_recorder = None


def recorder():
    """The process-wide recorder, or None when recording is off."""
    global _recorder
    if RECORD_DIR and _recorder is None:
        _recorder = OutputRecorder()
    return _recorder
