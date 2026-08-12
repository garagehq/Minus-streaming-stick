"""
Audio Passthrough for Stream Sentry.

Captures audio from HDMI-RX and outputs to HDMI-TX with mute control for ad blocking.

Features:
- Automatic error detection via GStreamer bus messages
- Watchdog thread to detect pipeline stalls
- Auto-restart on failure
- Auto-detection of HDMI capture device

Architecture:

    PLAYBACK BRANCH (always present, latency-critical):
        alsasrc (hw:X,0) -> audioconvert -> volume -> alsasink (hw:Y,0)
                                              ^
                                              | mute=true during ads

    ASR TAP BRANCH (optional, attached when AudioASRTap is passed in):
        alsasrc -> tee ─┬─► playback branch (above)
                        │
                        └─► leaky queue -> audioresample 48kHz/2ch -> 16kHz/1ch
                                          -> appsink -> AudioASRTap ring buffer
                                          -> snapshot_to_wav() for the ASR worker

    Tap branch is non-blocking: its queue is `leaky=downstream` so a slow
    consumer (the ASR worker falling behind) drops the oldest buffers rather
    than backpressuring the tee. Playback latency is identical with or
    without the tap branch present. See `_init_pipeline` for the exact
    pipeline string + the safety properties asserted by
    tests/test_audio_tap.py.
"""

import gc
import logging
import os
import re
import subprocess
import threading
import time
import wave

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

logger = logging.getLogger(__name__)

# Timeout for GStreamer state changes (in nanoseconds)
GST_STATE_CHANGE_TIMEOUT = 5 * Gst.SECOND  # 5 seconds


def detect_hdmi_capture_device() -> str:
    """
    Auto-detect the HDMI capture device by scanning ALSA cards.

    Looks for 'hdmiin' in the card name which indicates the HDMI-RX audio input.

    Returns:
        ALSA device string (e.g., 'hw:4,0') or 'hw:4,0' as fallback
    """
    try:
        # Read /proc/asound/cards to find hdmiin device
        with open('/proc/asound/cards', 'r') as f:
            cards_output = f.read()

        # Parse lines like: " 4 [rockchiphdmiin ]: rockchip_hdmiin - rockchip,hdmiin"
        for line in cards_output.split('\n'):
            if 'hdmiin' in line.lower():
                # Extract card number from start of line
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    card_num = parts[0]
                    device = f"hw:{card_num},0"
                    logger.info(f"[AudioPassthrough] Auto-detected HDMI capture device: {device}")
                    return device

        logger.warning("[AudioPassthrough] Could not find hdmiin in /proc/asound/cards, using fallback hw:4,0")
        return "hw:4,0"

    except Exception as e:
        logger.warning(f"[AudioPassthrough] Error detecting HDMI capture device: {e}, using fallback hw:4,0")
        return "hw:4,0"


class AudioPassthrough:
    """
    Audio passthrough from HDMI-RX to HDMI-TX.

    Uses GStreamer pipeline with volume element for instant mute control.
    Runs as a separate pipeline from video for simplicity and robustness.
    Includes automatic error recovery and watchdog monitoring.
    """

    # HDMI capture device candidates - we try these in order until one works
    # The card number can change depending on boot order
    HDMI_CAPTURE_DEVICES = ["hw:4,0", "hw:2,0", "hw:3,0", "hw:5,0"]

    def __init__(self, capture_device=None, playback_device="hw:0,0",
                 asr_tap=None):
        """
        Initialize audio passthrough.

        Args:
            capture_device: ALSA capture device (HDMI-RX) - auto-detected if None or "auto"
            playback_device: ALSA playback device (HDMI-TX)
            asr_tap: optional AudioASRTap instance. If passed, the pipeline
                will include a parallel `tee` branch that feeds 16kHz mono
                S16LE audio into the tap's ring buffer for ASR.
                The playback branch is unchanged whether asr_tap is set or
                not — same elements, same parameters, same latency budget.
                Passing None keeps the pre-ASR pipeline shape byte-identical
                (used by installs without faster-whisper present).
        """
        # Auto-detect capture device if not specified or set to "auto"
        if capture_device is None or capture_device == "auto":
            capture_device = detect_hdmi_capture_device()

        self.capture_device = capture_device
        self.playback_device = playback_device
        self.asr_tap = asr_tap
        self._capture_device_candidates = self.HDMI_CAPTURE_DEVICES.copy()
        # Move the auto-detected/specified device to front of candidates list
        if capture_device in self._capture_device_candidates:
            self._capture_device_candidates.remove(capture_device)
        self._capture_device_candidates.insert(0, capture_device)
        self.pipeline = None
        self.volume = None
        self.bus = None
        self.is_muted = False
        self.is_running = False
        self._lock = threading.Lock()

        # Watchdog state
        self._watchdog_thread = None
        self._stop_watchdog = threading.Event()
        self._watchdog_paused = False  # Pause watchdog when HDMI is lost
        self._last_buffer_time = 0
        self._restart_count = 0
        self._watchdog_interval = 3.0  # Check every 3 seconds
        self._stall_threshold = 6.0  # Consider stalled if no buffer for 6s

        # Exponential backoff for restarts (no max limit - always try to recover)
        self._base_restart_delay = 1.0  # Start with 1 second delay
        self._max_restart_delay = 60.0  # Cap at 60 seconds
        self._current_restart_delay = self._base_restart_delay
        self._last_restart_time = 0
        self._consecutive_failures = 0

        # Track our PID for ALSA device ownership check
        self._our_pid = os.getpid()

        # Restart coordination - prevent multiple concurrent restarts
        self._restart_in_progress = False
        self._restart_lock = threading.Lock()  # Separate lock for restart coordination
        # True when the playback branch fell back to fakesink because HDMI-TX
        # is disconnected (capture + ASR tap still run; no TV audio out).
        self._playback_fakesink = False

        # A/V sync watchdog - periodic queue flush to prevent drift
        # The sync queue adds fixed delay but clock drift over time could cause issues
        # Flushing every 45 minutes resets the sync queue with minimal audio dropout (~300ms)
        self._sync_interval = 45 * 60  # 45 minutes between sync resets (currently disabled — see _sync_reset_enabled)
        self._last_sync_reset = 0

        # Rolling audio level history for the blocking-overlay visualizer.
        # Stores normalized RMS (0.0-1.0) samples of the mixer output. Kept
        # as a bounded deque so memory is O(1) over a 24h run.
        from collections import deque
        self._level_history = deque(maxlen=16)
        self._last_level_sample_time = 0.0
        # Only sample RMS every N seconds — full-rate would thrash the CPU
        # and we only need ~10 Hz updates for the bar visualizer.
        self._level_sample_interval = 0.1
        # Disabled: the sync-queue flush cannot cleanly recover the pipeline —
        # alsasink closes the PCM device when the pipeline goes to PAUSED mid-
        # flush, and with min-threshold-time on the upstream queue, nothing
        # short of a full pipeline restart wakes it back up. The flush has
        # therefore always triggered a full restart every fire (~ once per
        # 45min of uptime), which is the exact "spurious restart" we set out
        # to eliminate. Drift isn't a real concern here (provide-clock=false,
        # sync=false) and 48h+ runs without a working flush showed no A/V
        # sync issues. Set to True only if a proper drift-recovery mechanism
        # is added.
        self._sync_reset_enabled = False

        # Initialize GStreamer (may already be initialized by video pipeline)
        Gst.init(None)

    def _is_alsa_device_running(self) -> bool:
        """Check if our ALSA playback device is actually producing audio.

        Uses /proc/asound/.../status as the source of truth: ALSA state must
        be RUNNING AND hw_ptr must advance between two samples. If hw_ptr moves,
        audio is definitively flowing — regardless of what GStreamer's state
        machine or owner_pid reports.

        Historical note: the old implementation compared ALSA's owner_pid to
        self._our_pid (main PID), but owner_pid is actually a thread TID — and
        often a stale one from the thread that originally opened the device.
        That made this check return False in healthy states and the watchdog
        fired a spurious full restart every time GStreamer transiently reported
        PAUSED (e.g. post-flush). Sampling hw_ptr sidesteps all that.
        """
        try:
            if not self.playback_device.startswith("hw:"):
                return False

            parts = self.playback_device[3:].split(",")
            if len(parts) != 2:
                return False
            card = parts[0]
            device = parts[1]

            status_path = f"/proc/asound/card{card}/pcm{device}p/sub0/status"
            if not os.path.exists(status_path):
                return False

            def _read():
                with open(status_path, 'r') as f:
                    content = f.read()
                state_running = False
                hw_ptr = None
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('state:'):
                        state_running = 'RUNNING' in line
                    elif line.startswith('hw_ptr'):
                        try:
                            hw_ptr = int(line.split(':', 1)[1].strip())
                        except (IndexError, ValueError):
                            pass
                return state_running, hw_ptr

            state1, ptr1 = _read()
            if not state1 or ptr1 is None:
                return False

            # 50ms at 48kHz ≈ 2400 frames — any real playback will advance
            # well beyond any noise. Costs ~50ms on the watchdog thread,
            # which runs every 3s.
            time.sleep(0.05)
            state2, ptr2 = _read()
            if not state2 or ptr2 is None:
                return False

            return ptr2 > ptr1

        except Exception as e:
            logger.debug(f"[AudioPassthrough] Error checking ALSA status: {e}")
            return False

    def _hdmi_tx_connected(self) -> bool:
        """Whether an HDMI-TX output (the TV) is connected, via sysfs.

        Fast, no-subprocess read of /sys/class/drm/card0-HDMI-A-*/status —
        mirrors HealthMonitor._check_hdmi_output_connected(). Used to decide
        whether the playback branch can use alsasink (TV present) or must fall
        back to fakesink (TV off) so HDMI-RX capture + the ASR tap stay alive.
        Fails OPEN (returns True) on any error so a probe glitch never
        needlessly downgrades a working playback path to fakesink.
        """
        try:
            from pathlib import Path
            for connector in Path('/sys/class/drm').glob('card0-HDMI-A-*'):
                status_file = connector / 'status'
                if status_file.exists() and status_file.read_text().strip() == 'connected':
                    return True
            return False
        except Exception as e:
            logger.debug(f"[AudioPassthrough] HDMI-TX check error: {e}")
            return True

    def _init_pipeline(self):
        """Initialize GStreamer audio pipeline.

        SAFETY INVARIANT: the playback branch (alsasrc → syncqueue →
        audioqueue → audioconvert → volume → alsasink) MUST be
        byte-identical regardless of whether the ASR tap is attached.
        Same elements, same names, same parameters. Latency budget
        (~500ms to match video) is unchanged. The tap is added as a
        parallel `tee` branch with its own leaky queue so a slow
        ASR consumer cannot backpressure the playback side.
        """
        try:
            # Audio passthrough pipeline with A/V sync delay
            # Video pipeline has ~350-500ms latency (HTTP streaming + decode + queue)
            # Audio needs matching delay to stay in sync
            # Using provide-clock=false to prevent alsasrc from being clock master
            #
            # Latency breakdown (PLAYBACK branch — applies whether or not the
            # ASR tap branch is attached):
            #   - alsasrc: 50ms buffer
            #   - syncqueue: 300ms delay (min-threshold-time forces buffering before output)
            #   - audioqueue: up to 100ms for jitter absorption
            #   - alsasink: 50ms buffer
            #   Total: ~500ms to match video latency
            #
            # When asr_tap is not None, we additionally insert a `tee` right
            # after the source caps, with the playback chain on one branch
            # and the 16kHz mono downsampling + appsink on the other. tee
            # buffer fanout is zero-copy reference-counting, so it adds no
            # measurable latency to the playback branch.
            # Pick the playback sink based on HDMI-TX availability. When the
            # TV (HDMI-TX) is off/disconnected, alsasink can't open its device
            # and would error the WHOLE pipeline — which also kills the HDMI-RX
            # capture feeding the ASR tap. Fall back to fakesink so capture +
            # the ASR tap keep running (ASR works off HDMI-RX even with the TV
            # off — useful for autonomous-mode data collection). The display
            # recovery loop rebuilds with alsasink when HDMI-TX returns.
            self._playback_fakesink = not self._hdmi_tx_connected()
            if self._playback_fakesink:
                sink = "fakesink sync=false"
                logger.warning("[AudioPassthrough] HDMI-TX not connected — "
                               "playback via fakesink; HDMI-RX capture + ASR tap stay live")
            else:
                sink = (f"alsasink device={self.playback_device} "
                        f"buffer-time=50000 latency-time=10000 sync=false")
            playback_chain = (
                f"queue name=syncqueue min-threshold-time=300000000 max-size-time=500000000 max-size-buffers=0 max-size-bytes=0 ! "
                f"queue max-size-buffers=10 max-size-time=100000000 leaky=downstream name=audioqueue ! "
                f"audioconvert ! "
                f"volume name=vol volume=1.0 mute=false ! "
                f"{sink}"
            )
            if self.asr_tap is not None:
                # ASR tap branch is leaky: if the ASR worker falls behind, the
                # tap queue drops the oldest buffers rather than blocking
                # the tee. max-size-buffers=40 at 48kHz with ~1024-sample
                # buffers ≈ 850ms of headroom; more than enough for the
                # 2s ASR cadence even under transient CPU pressure.
                asr_chain = (
                    f"audiotee. ! "
                    f"queue name=asrqueue max-size-buffers=40 max-size-time=2000000000 leaky=downstream ! "
                    f"audioconvert ! audioresample quality=4 ! "
                    f"audio/x-raw,rate=16000,channels=1,format=S16LE ! "
                    f"appsink name=asr_sink emit-signals=true drop=true sync=false max-buffers=20"
                )
                pipeline_str = (
                    f"alsasrc device={self.capture_device} buffer-time=50000 latency-time=10000 provide-clock=false ! "
                    f"audio/x-raw,rate=48000,channels=2,format=S16LE ! "
                    f"tee name=audiotee allow-not-linked=true "
                    # Playback branch (this comes off the tee and runs the
                    # same elements as the no-tap pipeline)
                    f"audiotee. ! {playback_chain} "
                    # ASR tap branch
                    f"{asr_chain}"
                )
            else:
                # No tap configured — keep the original linear pipeline so
                # behaviour on installs without faster-whisper is identical
                # to pre-ASR Minus. No tee, no risk of pipeline-shape
                # regressions on those installs.
                pipeline_str = (
                    f"alsasrc device={self.capture_device} buffer-time=50000 latency-time=10000 provide-clock=false ! "
                    f"audio/x-raw,rate=48000,channels=2,format=S16LE ! "
                    f"{playback_chain}"
                )

            logger.debug(f"[AudioPassthrough] Creating pipeline: {pipeline_str}")
            self.pipeline = Gst.parse_launch(pipeline_str)

            # Get volume element for mute control
            self.volume = self.pipeline.get_by_name('vol')

            # Set up bus message handling for error detection
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message::error', self._on_error)
            self.bus.connect('message::eos', self._on_eos)
            self.bus.connect('message::state-changed', self._on_state_changed)

            # Add probe to track buffer flow (for stall detection)
            queue = self.pipeline.get_by_name('audioqueue')
            if queue:
                pad = queue.get_static_pad('src')
                if pad:
                    pad.add_probe(Gst.PadProbeType.BUFFER, self._buffer_probe, None)

            # Attach the ASR tap to the appsink (if a tap is configured).
            # Done here, after parse_launch, so the tap gets re-attached
            # automatically across pipeline restarts (the appsink element
            # is recreated each time the pipeline is rebuilt).
            if self.asr_tap is not None:
                appsink = self.pipeline.get_by_name('asr_sink')
                if appsink is not None:
                    self.asr_tap.attach_to(appsink)
                    logger.info("[AudioPassthrough] ASR tap attached to pipeline")
                else:
                    logger.error("[AudioPassthrough] asr_tap configured but appsink "
                                 "'asr_sink' not found in pipeline — tap inactive")

            if self.volume:
                tap_status = " + ASR tap" if self.asr_tap is not None else ""
                logger.info(f"[AudioPassthrough] Pipeline created: "
                            f"{self.capture_device} -> {self.playback_device}{tap_status}")
            else:
                logger.error("[AudioPassthrough] Failed to get volume element")

        except Exception as e:
            logger.error(f"[AudioPassthrough] Failed to create pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None

    def _buffer_probe(self, pad, info, user_data):
        """Probe callback to track buffer flow for stall detection.

        Also samples the buffer's RMS at `_level_sample_interval` so the
        blocking overlay can render an audio-reactive bar visualization.
        Skips the RMS math on most buffers to keep CPU overhead negligible.
        """
        now = time.time()
        self._last_buffer_time = now

        # Sample audio level for the visualizer (throttled)
        if now - self._last_level_sample_time >= self._level_sample_interval:
            self._last_level_sample_time = now
            try:
                buf = info.get_buffer()
                if buf is not None:
                    self._sample_rms(buf)
            except Exception as e:
                logger.debug(f"[AudioPassthrough] RMS sample skipped: {e}")

        # Reset backoff counter after sustained buffer flow (5+ seconds)
        if self._consecutive_failures > 0:
            time_since_restart = now - self._last_restart_time
            if time_since_restart > 5.0:
                self._consecutive_failures = 0
                self._current_restart_delay = self._base_restart_delay
                logger.debug("[AudioPassthrough] Backoff reset - sustained buffer flow")

        return Gst.PadProbeReturn.OK

    def _sample_rms(self, buf):
        """Compute RMS from an S16LE audio buffer, append to history.

        Format is locked to S16LE stereo at 48 kHz elsewhere in the pipeline
        so we can treat the buffer as signed 16-bit little-endian samples.
        """
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return
        try:
            import struct
            data = bytes(mapinfo.data)
            # Down-sample — we don't need every one of ~1000 samples per 50ms
            # buffer. Stride picks one sample every ~0.5 ms, plenty for bar
            # heights.
            n = len(data) // 2  # S16 = 2 bytes
            if n == 0:
                return
            stride = max(1, n // 64)
            samples = struct.unpack_from(f'<{n}h', data)
            total = 0
            count = 0
            peak = 0
            for i in range(0, n, stride):
                s = samples[i]
                total += s * s
                if abs(s) > peak:
                    peak = abs(s)
                count += 1
            if count == 0:
                return
            import math
            rms = math.sqrt(total / count) / 32767.0
            # Nudge the perceived range — quiet speech is ~0.02 RMS, peaks
            # are ~0.3. A sqrt curve makes the bars feel more responsive.
            visual = min(1.0, math.sqrt(rms))
            self._level_history.append(visual)
        finally:
            buf.unmap(mapinfo)

    def get_level_bars(self, width=16):
        """Render the current audio history as a unicode block bar string.

        Returns a `width`-character string like `.,;ozIMI;,.` that rises and
        falls with audio energy. Designed for the blocking overlay stats
        area (monospace). Returns empty string if no history yet.
        """
        if not self._level_history:
            return ''
        # ASCII-only bar ramp so it renders reliably through the MPP text pass
        # no matter what font the encoder picked.
        ramp = ' .,-;+ox*#@'
        levels = list(self._level_history)[-width:]
        # Pad with zeros on the left if history is shorter than width
        if len(levels) < width:
            levels = [0.0] * (width - len(levels)) + levels
        chars = []
        for lv in levels:
            idx = int(round(lv * (len(ramp) - 1)))
            idx = max(0, min(len(ramp) - 1, idx))
            chars.append(ramp[idx])
        return ''.join(chars)

    def _on_error(self, bus, message):
        """Handle GStreamer error messages."""
        err, debug = message.parse_error()
        logger.error(f"[AudioPassthrough] Pipeline error: {err.message}")
        logger.debug(f"[AudioPassthrough] Debug info: {debug}")

        # Schedule restart on error
        threading.Thread(target=self._restart_pipeline, daemon=True).start()

    def _on_eos(self, bus, message):
        """Handle end-of-stream (shouldn't happen for live source)."""
        logger.warning("[AudioPassthrough] Unexpected EOS received")

        # Restart on EOS
        threading.Thread(target=self._restart_pipeline, daemon=True).start()

    def _on_state_changed(self, bus, message):
        """Handle state changes."""
        if message.src != self.pipeline:
            return

        old, new, pending = message.parse_state_changed()
        if new == Gst.State.PLAYING:
            self._last_buffer_time = time.time()
            logger.debug("[AudioPassthrough] Pipeline now PLAYING")

    def _flush_sync_queue(self):
        """Flush the sync queue to reset A/V sync without full pipeline restart.

        This causes a brief audio dropout (~300ms) while the queue refills,
        but is much faster than a full pipeline restart (~1s).

        Returns:
            True if flush succeeded, False otherwise
        """
        if not self.pipeline:
            return False

        try:
            syncqueue = self.pipeline.get_by_name('syncqueue')
            if not syncqueue:
                logger.warning("[AudioPassthrough] Could not find syncqueue element")
                return False

            # Get the sink pad to send flush events
            sink_pad = syncqueue.get_static_pad('sink')
            if not sink_pad:
                logger.warning("[AudioPassthrough] Could not get syncqueue sink pad")
                return False

            # Send flush-start event (clears queue, puts downstream in flushing mode)
            flush_start = Gst.Event.new_flush_start()
            if not sink_pad.send_event(flush_start):
                logger.warning("[AudioPassthrough] Failed to send flush-start event")
                return False

            # Brief pause to let flush propagate
            time.sleep(0.05)

            # Send flush-stop event (ends flushing, allows data flow to resume)
            # reset_time=True resets the running time
            flush_stop = Gst.Event.new_flush_stop(True)
            if not sink_pad.send_event(flush_stop):
                logger.warning("[AudioPassthrough] Failed to send flush-stop event")
                return False

            self._last_sync_reset = time.time()
            logger.info("[AudioPassthrough] Sync queue flushed - A/V sync reset (~300ms dropout)")
            return True

        except Exception as e:
            logger.error(f"[AudioPassthrough] Error flushing sync queue: {e}")
            return False

    def _restart_pipeline(self):
        """Restart the audio pipeline after an error with exponential backoff.

        This method is resilient to memory pressure and includes:
        - Protection against concurrent restart attempts
        - Timeouts on all GStreamer state changes
        - Explicit garbage collection to help during memory pressure
        - Comprehensive error logging
        """
        # Prevent multiple concurrent restarts using a separate lock
        # Use trylock to avoid blocking the watchdog if restart is in progress
        if not self._restart_lock.acquire(blocking=False):
            logger.debug("[AudioPassthrough] Restart already in progress, skipping")
            return

        try:
            self._restart_in_progress = True

            # Use timeout on main lock to prevent deadlock during memory pressure
            if not self._lock.acquire(timeout=10.0):
                logger.error("[AudioPassthrough] Could not acquire lock for restart (timeout)")
                return

            try:
                self._restart_count += 1
                self._consecutive_failures += 1

                # After 3 consecutive failures, try the next capture device
                if self._consecutive_failures >= 3 and len(self._capture_device_candidates) > 1:
                    old_device = self.capture_device
                    # Rotate to next device
                    self._capture_device_candidates.append(self._capture_device_candidates.pop(0))
                    self.capture_device = self._capture_device_candidates[0]
                    logger.warning(f"[AudioPassthrough] Trying alternate capture device: {old_device} -> {self.capture_device}")
                    self._consecutive_failures = 0  # Reset for new device

                # Calculate backoff delay (cap exponent at 10 to prevent overflow)
                exponent = min(self._consecutive_failures - 1, 10)
                delay = min(
                    self._base_restart_delay * (2 ** exponent),
                    self._max_restart_delay
                )
                self._current_restart_delay = delay

                logger.warning(
                    f"[AudioPassthrough] Restarting pipeline (attempt {self._restart_count}, "
                    f"delay {delay:.1f}s, {self._consecutive_failures} consecutive failures)"
                )

                # Stop current pipeline and clean up resources
                if self.pipeline:
                    try:
                        # CRITICAL: Remove bus signal watch to prevent file descriptor leak
                        if hasattr(self, 'bus') and self.bus:
                            try:
                                self.bus.remove_signal_watch()
                            except Exception as e:
                                logger.debug(f"[AudioPassthrough] Error removing bus watch: {e}")
                            self.bus = None
                        self.pipeline.set_state(Gst.State.NULL)
                        # Wait for NULL state with TIMEOUT - prevents indefinite blocking
                        ret, state, pending = self.pipeline.get_state(GST_STATE_CHANGE_TIMEOUT)
                        if ret == Gst.StateChangeReturn.FAILURE:
                            logger.warning("[AudioPassthrough] Failed to set pipeline to NULL state")
                        elif ret != Gst.StateChangeReturn.SUCCESS:
                            logger.warning(f"[AudioPassthrough] Timeout waiting for NULL state (ret={ret})")
                    except Exception as e:
                        logger.warning(f"[AudioPassthrough] Error stopping pipeline: {e}")
                    self.pipeline = None
                    self.volume = None

                    # Force garbage collection to help during memory pressure
                    gc.collect()

                    # Extra delay to ensure ALSA device is fully released
                    time.sleep(0.5)

            finally:
                self._lock.release()

            # Wait with exponential backoff OUTSIDE the lock
            # This allows other operations to proceed during the delay
            time.sleep(delay)

            # Check if we should still be running
            if not self.is_running:
                logger.info("[AudioPassthrough] Restart cancelled - not running")
                return

            # Reacquire lock for pipeline creation
            if not self._lock.acquire(timeout=10.0):
                logger.error("[AudioPassthrough] Could not acquire lock for pipeline creation (timeout)")
                return

            try:
                # Force GC before creating new pipeline
                gc.collect()

                # Recreate and start
                self._init_pipeline()
                if not self.pipeline:
                    logger.error("[AudioPassthrough] Failed to create pipeline during restart")
                    return

                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    # Get more details about the failure
                    logger.error("[AudioPassthrough] Failed to restart pipeline (set_state returned FAILURE)")
                    # Try to get error from bus
                    if self.bus:
                        msg = self.bus.pop_filtered(Gst.MessageType.ERROR)
                        if msg:
                            err, debug = msg.parse_error()
                            logger.error(f"[AudioPassthrough] GStreamer error: {err.message}")
                            logger.debug(f"[AudioPassthrough] Debug: {debug}")
                elif ret == Gst.StateChangeReturn.ASYNC:
                    # Wait for state change with timeout
                    ret2, state, pending = self.pipeline.get_state(GST_STATE_CHANGE_TIMEOUT)
                    if ret2 == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING:
                        logger.info("[AudioPassthrough] Pipeline restarted successfully (async)")
                        self._last_buffer_time = time.time()
                        self._last_restart_time = time.time()
                        self._last_sync_reset = time.time()  # Reset A/V sync timer
                        if self.is_muted and self.volume:
                            self.volume.set_property('mute', True)
                    else:
                        logger.error(f"[AudioPassthrough] Failed to reach PLAYING state: ret={ret2}, state={state.value_nick if state else 'None'}")
                else:
                    logger.info("[AudioPassthrough] Pipeline restarted successfully")
                    self._last_buffer_time = time.time()
                    self._last_restart_time = time.time()
                    self._last_sync_reset = time.time()  # Reset A/V sync timer
                    # Restore mute state
                    if self.is_muted and self.volume:
                        self.volume.set_property('mute', True)
            finally:
                self._lock.release()

        except Exception as e:
            logger.error(f"[AudioPassthrough] Unexpected error during restart: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._restart_in_progress = False
            self._restart_lock.release()

    def _watchdog_loop(self):
        """Watchdog thread to detect pipeline stalls.

        Enhanced with:
        - Protection against triggering restart if one is already in progress
        - Better logging of pipeline state
        - Detection of completely dead pipelines (no pipeline object)
        """
        logger.debug("[AudioPassthrough] Watchdog started")

        while not self._stop_watchdog.is_set():
            self._stop_watchdog.wait(self._watchdog_interval)

            if self._stop_watchdog.is_set():
                break

            if not self.is_running:
                continue

            # Skip restart attempts if watchdog is paused (e.g., HDMI lost)
            if self._watchdog_paused:
                continue

            # Skip if a restart is already in progress
            if self._restart_in_progress:
                logger.debug("[AudioPassthrough] Watchdog: restart in progress, skipping check")
                continue

            needs_restart = False
            restart_reason = ""

            # A/V Sync reset - periodically flush sync queue to prevent clock drift
            # Uses queue flush instead of full restart for minimal dropout (~300ms vs ~1s)
            if self._sync_reset_enabled and self._last_sync_reset > 0:
                time_since_sync = time.time() - self._last_sync_reset
                if time_since_sync >= self._sync_interval:
                    # Try queue flush first (faster, less disruptive)
                    if self._flush_sync_queue():
                        # Flush succeeded. The flush-start/flush-stop event pair
                        # transiently moves the pipeline out of PLAYING; running the
                        # state check below in the same iteration mis-reads that as
                        # a stall and triggers a spurious full restart. Skip to next
                        # watchdog tick — by then the pipeline is PLAYING again.
                        self._last_buffer_time = time.time()
                        continue
                    else:
                        # Flush failed, fall back to full restart
                        needs_restart = True
                        restart_reason = f"A/V sync reset failed flush (every {self._sync_interval // 60}min)"

            # Check if pipeline exists at all
            if not self.pipeline:
                needs_restart = True
                restart_reason = "pipeline is None"
            # Check if buffers are flowing
            elif self._last_buffer_time > 0:
                time_since_buffer = time.time() - self._last_buffer_time
                if time_since_buffer > self._stall_threshold:
                    # GStreamer says stalled, but check ALSA status first
                    # PipeWire can interfere with GStreamer buffer probes
                    if self._is_alsa_device_running():
                        logger.debug(
                            f"[AudioPassthrough] Buffer probe stale ({time_since_buffer:.1f}s) "
                            f"but ALSA device is RUNNING - updating buffer time"
                        )
                        self._last_buffer_time = time.time()
                    else:
                        needs_restart = True
                        restart_reason = f"stalled ({time_since_buffer:.1f}s since last buffer)"

            # Check pipeline state (only if we have a pipeline and haven't already decided to restart)
            if not needs_restart and self.pipeline:
                try:
                    state_ret, state, pending = self.pipeline.get_state(0)
                    if state != Gst.State.PLAYING and self.is_running:
                        # GStreamer says not PLAYING, but check if audio is ACTUALLY flowing
                        # via ALSA /proc status. This handles PipeWire/WirePlumber interference
                        # where GStreamer state may be incorrect but audio works fine.
                        if self._is_alsa_device_running():
                            logger.debug(
                                f"[AudioPassthrough] GStreamer reports {state.value_nick if state else 'None'} "
                                f"but ALSA device is RUNNING with our PID - skipping restart"
                            )
                            # Update last buffer time since audio is actually flowing
                            self._last_buffer_time = time.time()
                        else:
                            needs_restart = True
                            restart_reason = f"not in PLAYING state ({state.value_nick if state else 'None'})"
                except Exception as e:
                    needs_restart = True
                    restart_reason = f"error checking state: {e}"

            if needs_restart:
                logger.warning(f"[AudioPassthrough] Pipeline issue detected: {restart_reason}")
                # Start restart in a separate thread to not block watchdog
                threading.Thread(
                    target=self._restart_pipeline,
                    daemon=True,
                    name="AudioRestart"
                ).start()

        logger.debug("[AudioPassthrough] Watchdog stopped")

    def start(self):
        """Start audio passthrough.

        Tries all candidate capture devices until one works.
        """
        with self._lock:
            # Try each candidate device until one works
            devices_tried = []
            for candidate in self._capture_device_candidates:
                if candidate in devices_tried:
                    continue
                devices_tried.append(candidate)

                # Update capture device and reinitialize pipeline
                self.capture_device = candidate

                # Clean up any existing pipeline
                if self.pipeline:
                    try:
                        if hasattr(self, 'bus') and self.bus:
                            try:
                                self.bus.remove_signal_watch()
                            except:
                                pass
                            self.bus = None
                        self.pipeline.set_state(Gst.State.NULL)
                        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
                    except:
                        pass
                    self.pipeline = None
                    self.volume = None
                    time.sleep(0.3)

                # Initialize pipeline with this device
                self._init_pipeline()

                if not self.pipeline:
                    logger.warning(f"[AudioPassthrough] Failed to create pipeline with {candidate}")
                    continue

                try:
                    ret = self.pipeline.set_state(Gst.State.PLAYING)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.warning(f"[AudioPassthrough] Failed to start pipeline with {candidate}")
                        continue

                    # For async state changes, wait for completion
                    if ret == Gst.StateChangeReturn.ASYNC:
                        ret2, state, pending = self.pipeline.get_state(2 * Gst.SECOND)
                        if ret2 != Gst.StateChangeReturn.SUCCESS or state != Gst.State.PLAYING:
                            logger.warning(f"[AudioPassthrough] Pipeline {candidate} failed to reach PLAYING state")
                            continue

                    # Success!
                    self.is_running = True
                    self._last_buffer_time = time.time()
                    self._last_sync_reset = time.time()  # Start A/V sync timer
                    self._restart_count = 0

                    # Reorder candidates so working device is first
                    if candidate in self._capture_device_candidates:
                        self._capture_device_candidates.remove(candidate)
                    self._capture_device_candidates.insert(0, candidate)

                    # Start watchdog thread
                    self._stop_watchdog.clear()
                    self._watchdog_thread = threading.Thread(
                        target=self._watchdog_loop,
                        daemon=True,
                        name="AudioWatchdog"
                    )
                    self._watchdog_thread.start()

                    logger.info(f"[AudioPassthrough] Audio passthrough started with {candidate}")
                    return True

                except Exception as e:
                    logger.warning(f"[AudioPassthrough] Failed to start with {candidate}: {e}")
                    continue

            # All devices failed
            logger.error(f"[AudioPassthrough] All capture devices failed: {devices_tried}")
            return False

    def mute(self):
        """Mute audio (for ad blocking)."""
        with self._lock:
            if self.volume and not self.is_muted:
                self.volume.set_property('mute', True)
                self.is_muted = True
                logger.info("[AudioPassthrough] Audio MUTED")

    def unmute(self):
        """Unmute audio (after ad ends)."""
        with self._lock:
            if self.volume and self.is_muted:
                self.volume.set_property('mute', False)
                self.is_muted = False
                logger.info("[AudioPassthrough] Audio UNMUTED")

    def set_volume(self, level):
        """
        Set volume level.

        Args:
            level: Volume level (0.0 = silent, 1.0 = 100%, 10.0 = 1000%)
        """
        with self._lock:
            if self.volume:
                self.volume.set_property('volume', level)
                logger.info(f"[AudioPassthrough] Volume set to {level}")

    def get_status(self):
        """Get current audio pipeline status."""
        # Use timeout to prevent hanging if lock is held by restart
        if not self._lock.acquire(timeout=2.0):
            return {
                "state": "unknown",
                "muted": self.is_muted,
                "restart_count": self._restart_count,
                "restart_in_progress": self._restart_in_progress,
                "last_buffer_age": time.time() - self._last_buffer_time if self._last_buffer_time > 0 else -1
            }

        try:
            if not self.pipeline:
                return {
                    "state": "stopped",
                    "muted": self.is_muted,
                    "restart_count": self._restart_count,
                    "restart_in_progress": self._restart_in_progress,
                    "last_buffer_age": -1
                }

            try:
                state_ret, state, pending = self.pipeline.get_state(0)
                state_name = state.value_nick if state else "unknown"
            except Exception:
                state_name = "error"

            # Peak RMS over recent buffer history. Used by autonomous mode
            # to distinguish "audio buffer flowing with silence" (HDMI source
            # paused → still emits silent buffers) from "audio has real
            # content". Empty history → 0.0 (no buffers seen yet).
            try:
                recent_level = max(self._level_history) if self._level_history else 0.0
            except Exception:
                recent_level = 0.0
            return {
                "state": state_name,
                "muted": self.is_muted,
                "restart_count": self._restart_count,
                "restart_in_progress": self._restart_in_progress,
                "last_buffer_age": time.time() - self._last_buffer_time if self._last_buffer_time > 0 else -1,
                "recent_level": recent_level
            }
        finally:
            self._lock.release()

    def pause_watchdog(self):
        """Pause the watchdog to prevent restart loops (e.g., when HDMI is lost).

        The pipeline will be stopped but the module remains ready to resume.
        """
        # Use timeout on lock to prevent hanging
        if not self._lock.acquire(timeout=10.0):
            logger.error("[AudioPassthrough] Could not acquire lock for pause (timeout)")
            # Still set paused flag even if we can't acquire lock
            self._watchdog_paused = True
            return

        try:
            self._watchdog_paused = True
            logger.info("[AudioPassthrough] Watchdog paused - no auto-restart")

            # Stop current pipeline to save resources
            if self.pipeline:
                try:
                    self.pipeline.set_state(Gst.State.NULL)
                    # Wait with timeout
                    self.pipeline.get_state(GST_STATE_CHANGE_TIMEOUT)
                except Exception as e:
                    logger.debug(f"[AudioPassthrough] Error pausing pipeline: {e}")
        finally:
            self._lock.release()

    def resume_watchdog(self):
        """Resume the watchdog and restart the pipeline.

        Call this when HDMI signal is restored.
        """
        # Use timeout on lock to prevent hanging
        if not self._lock.acquire(timeout=10.0):
            logger.error("[AudioPassthrough] Could not acquire lock for resume (timeout)")
            return

        try:
            self._watchdog_paused = False

            # Reset failure counters
            self._consecutive_failures = 0
            self._current_restart_delay = self._base_restart_delay

            # Check if pipeline is already working - don't restart unnecessarily
            if self.pipeline:
                try:
                    state_ret, state, pending = self.pipeline.get_state(0)
                    if state == Gst.State.PLAYING:
                        logger.info("[AudioPassthrough] Watchdog resumed - pipeline already PLAYING, no restart needed")
                        return
                    # Also check ALSA status - if device is running with our PID, audio is working
                    if self._is_alsa_device_running():
                        logger.info("[AudioPassthrough] Watchdog resumed - ALSA device already running, no restart needed")
                        self._last_buffer_time = time.time()
                        return
                except Exception as e:
                    logger.debug(f"[AudioPassthrough] Error checking pipeline state during resume: {e}")

            logger.info("[AudioPassthrough] Watchdog resumed - restarting pipeline")

            # Force GC before restart
            gc.collect()

            # Clean up existing pipeline before creating new one
            if self.pipeline:
                try:
                    if hasattr(self, 'bus') and self.bus:
                        try:
                            self.bus.remove_signal_watch()
                        except:
                            pass
                        self.bus = None
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline.get_state(GST_STATE_CHANGE_TIMEOUT)
                except Exception as e:
                    logger.debug(f"[AudioPassthrough] Error cleaning up old pipeline: {e}")
                self.pipeline = None
                self.volume = None
                time.sleep(0.5)  # Give ALSA time to release device

            # Restart pipeline
            if self.is_running:
                self._init_pipeline()
                if self.pipeline:
                    ret = self.pipeline.set_state(Gst.State.PLAYING)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        logger.error("[AudioPassthrough] Failed to resume pipeline (set_state returned FAILURE)")
                    elif ret == Gst.StateChangeReturn.ASYNC:
                        # Wait with timeout
                        ret2, state, pending = self.pipeline.get_state(GST_STATE_CHANGE_TIMEOUT)
                        if ret2 == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING:
                            logger.info("[AudioPassthrough] Pipeline resumed successfully (async)")
                            self._last_buffer_time = time.time()
                            self._last_sync_reset = time.time()  # Reset A/V sync timer
                            if self.is_muted and self.volume:
                                self.volume.set_property('mute', True)
                        else:
                            logger.error(f"[AudioPassthrough] Failed to resume: ret={ret2}, state={state.value_nick if state else 'None'}")
                    else:
                        logger.info("[AudioPassthrough] Pipeline resumed successfully")
                        self._last_buffer_time = time.time()
                        self._last_sync_reset = time.time()  # Reset A/V sync timer
                        if self.is_muted and self.volume:
                            self.volume.set_property('mute', True)
                else:
                    logger.error("[AudioPassthrough] Failed to create pipeline during resume")
        except Exception as e:
            logger.error(f"[AudioPassthrough] Error during resume: {e}")
        finally:
            self._lock.release()

    def stop(self):
        """Stop audio passthrough."""
        # Mark as not running first to signal restart threads to stop
        self.is_running = False

        # Stop watchdog
        self._stop_watchdog.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None

        # Wait for any in-progress restart to complete
        if self._restart_in_progress:
            logger.debug("[AudioPassthrough] Waiting for restart to complete before stopping...")
            for _ in range(50):  # Wait up to 5 seconds
                if not self._restart_in_progress:
                    break
                time.sleep(0.1)

        # Use timeout on lock in case restart is stuck
        if not self._lock.acquire(timeout=5.0):
            logger.warning("[AudioPassthrough] Could not acquire lock for stop, forcing cleanup")
            # Force cleanup even without lock
            try:
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
            except:
                pass
            self.pipeline = None
            self.volume = None
            self.bus = None
            return

        try:
            # Stop pipeline
            if self.pipeline:
                try:
                    if self.bus:
                        self.bus.remove_signal_watch()
                    self.pipeline.set_state(Gst.State.NULL)
                    # Wait with timeout
                    self.pipeline.get_state(GST_STATE_CHANGE_TIMEOUT)
                    logger.info("[AudioPassthrough] Audio passthrough stopped")
                except Exception as e:
                    logger.error(f"[AudioPassthrough] Error stopping: {e}")

                self.pipeline = None
                self.volume = None
                self.bus = None
        finally:
            self._lock.release()

    def restart(self, reason="manual"):
        """Force a full pipeline rebuild (web UI button / post-modeset re-prepare).

        Closing and reopening the ALSA playback device is what makes the
        rockchip HDMI-TX driver re-program its audio lane (audio infoframe +
        clock regen). That re-programming only happens at PCM prepare time,
        so a stream that was opened while the TX link was still training
        (TV power-on / video modeset race) plays into a dead audio lane
        forever — buffers flow, hw_ptr advances, but the TV hears nothing.
        The buffer-flow watchdog cannot see that state; a rebuild fixes it.

        Returns a dict for the web UI. The actual restart runs in a
        background thread (~1-2s audio dropout).
        """
        if not self.is_running:
            return {'success': False, 'error': 'Audio not running'}
        if self._restart_in_progress:
            return {'success': True, 'message': 'Restart already in progress'}
        logger.info(f"[AudioPassthrough] Pipeline restart requested ({reason})")
        # A requested restart is not a failure — skip the backoff penalty
        # so the rebuild happens after the base delay (~1s), not 2^n.
        self._consecutive_failures = 0
        threading.Thread(target=self._restart_pipeline, daemon=True,
                         name="audio-manual-restart").start()
        return {'success': True, 'message': 'Audio pipeline restarting (~1-2s dropout)'}

    def get_diagnostics(self):
        """End-to-end audio health check for the web UI Check Audio button.

        Walks the chain capture -> pipeline -> mute -> ALSA playback device
        and returns a verdict plus the raw signals, so silent-audio problems
        can be triaged from the Home tab without shell access.
        """
        status = self.get_status()
        diag = dict(status)
        diag['is_running'] = self.is_running
        diag['capture_device'] = self.capture_device
        diag['playback_device'] = self.playback_device
        diag['playback_fakesink'] = bool(getattr(self, '_playback_fakesink', False))

        # Kernel-side view of the playback PCM (authoritative — GStreamer
        # state can lie when PipeWire or a dead thread is involved).
        alsa_state = 'unknown'
        try:
            m = re.match(r'hw:(\d+),(\d+)', self.playback_device or '')
            if m:
                path = f"/proc/asound/card{m.group(1)}/pcm{m.group(2)}p/sub0/status"
                with open(path) as f:
                    content = f.read()
            else:
                content = ''
            sm = re.search(r'state:\s*(\w+)', content)
            if sm:
                alsa_state = sm.group(1)
        except Exception:
            alsa_state = 'unavailable'
        diag['alsa_playback_state'] = alsa_state

        problems = []
        if not self.is_running:
            problems.append('audio passthrough is not running')
        elif status.get('state') != 'playing':
            problems.append(f"pipeline not playing (state: {status.get('state')})")
        if diag['playback_fakesink']:
            problems.append('TV (HDMI-TX) disconnected — playback routed to fakesink')
        age = status.get('last_buffer_age', -1)
        if age is not None and (age < 0 or age > 3.0):
            problems.append('no audio buffers flowing from HDMI input')
        if status.get('muted'):
            problems.append('audio is muted (ad block active or stuck mute)')
        if alsa_state != 'RUNNING' and not diag['playback_fakesink'] and self.is_running:
            problems.append(f'ALSA playback device not running (state: {alsa_state})')

        # Everything green but the source itself is quiet — not a fault,
        # but worth surfacing (paused video, menu without autoplay, etc.).
        source_silent = (not problems
                         and status.get('recent_level', 0.0) < 0.01)
        diag['source_silent'] = source_silent

        diag['problems'] = problems
        diag['ok'] = not problems
        if problems:
            diag['verdict'] = problems[0]
        elif source_silent:
            diag['verdict'] = 'audio path healthy — source is currently silent'
        else:
            diag['verdict'] = 'audio path healthy — sound is flowing to the TV'
        return diag

    def reset_av_sync(self):
        """Manually reset A/V sync by flushing the sync queue.

        This can be called via the web UI when audio/video are out of sync.
        Causes a brief audio dropout (~300ms) while the queue refills.

        Returns:
            dict with success status and message
        """
        if not self.is_running:
            return {'success': False, 'error': 'Audio not running'}

        if not self.pipeline:
            return {'success': False, 'error': 'No audio pipeline'}

        if self._flush_sync_queue():
            return {
                'success': True,
                'message': 'A/V sync reset - audio will resume in ~300ms'
            }
        else:
            # Fall back to full pipeline restart
            logger.info("[AudioPassthrough] Flush failed, doing full restart for A/V sync")
            threading.Thread(target=self._restart_pipeline, daemon=True).start()
            return {
                'success': True,
                'message': 'A/V sync reset via pipeline restart - audio will resume in ~1s'
            }

    def destroy(self):
        """Clean up resources."""
        self.stop()


# ===========================================================================
# Audio ASR Tap — ring buffer for ASR worker consumption
# ===========================================================================


class AudioASRTap:
    """Ring-buffer audio tap fed by the GStreamer appsink on the ASR
    pipeline branch.

    Receives 16kHz mono S16LE buffers from the GStreamer appsink
    callback (called from a GStreamer streaming thread, NOT the main
    Python thread). Writes them into a fixed-size numpy ring buffer.
    Exposes `snapshot_to_wav(seconds)` for the ASR worker to atomically
    grab the most recent N seconds as a WAV file on disk for
    the ASR worker (faster-whisper).

    Threading model:
      - GStreamer streaming thread: holds _lock briefly to append samples
      - ASR thread:                 holds _lock briefly to copy the
                                    snapshot region out of the ring
      - Both lock holds are O(samples) without I/O; the lock is fine to
        be a regular Lock, no contention concern.

    Survives pipeline restarts because `attach_to` is called fresh in
    `_init_pipeline` each time the GStreamer pipeline is rebuilt. The
    ring buffer state persists across restarts (older audio stays until
    overwritten) — desired behaviour, since whisper inference doesn't
    care about discontinuities and a brief audio gap mid-buffer just
    looks like silence in the transcript.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    SAMPLE_WIDTH_BYTES = 2  # S16LE
    BUFFER_SECONDS = 8  # Hold a bit more than the 5s window so we never run short

    def __init__(self, wav_path: str = '/dev/shm/minus_asr_window.wav'):
        # numpy import is local so AudioPassthrough's import chain stays
        # numpy-free for installs that never enable the tap.
        import numpy as np
        self._np = np
        self.wav_path = wav_path
        self._buffer_samples = self.SAMPLE_RATE * self.BUFFER_SECONDS
        self._ring = np.zeros(self._buffer_samples, dtype=np.int16)
        self._write_pos = 0      # Next index to write (0..buffer_samples-1)
        self._samples_written = 0  # Total samples ever received
        self._lock = threading.Lock()
        self._last_buffer_time = 0.0
        self._attach_count = 0   # Bumped on each attach_to (track restarts)

    def attach_to(self, appsink):
        """Connect this tap to a GStreamer appsink. Idempotent; called
        once per `_init_pipeline` invocation (including across pipeline
        restarts)."""
        appsink.connect('new-sample', self._on_new_sample)
        self._attach_count += 1

    def _on_new_sample(self, appsink):
        """GStreamer appsink callback. Pull the sample and append to the
        ring buffer. Runs on a GStreamer streaming thread."""
        sample = appsink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        if buf is None:
            return Gst.FlowReturn.OK
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            samples = self._np.frombuffer(mapinfo.data, dtype=self._np.int16)
            n = len(samples)
            if n == 0:
                return Gst.FlowReturn.OK
            with self._lock:
                end = self._write_pos + n
                if end <= self._buffer_samples:
                    self._ring[self._write_pos:end] = samples
                else:
                    split = self._buffer_samples - self._write_pos
                    self._ring[self._write_pos:] = samples[:split]
                    self._ring[:n - split] = samples[split:]
                self._write_pos = end % self._buffer_samples
                self._samples_written += n
                self._last_buffer_time = time.time()
        finally:
            buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def snapshot_to_wav(self, seconds: float = 5.0) -> bool:
        """Write the most recent `seconds` of audio to self.wav_path as
        a 16kHz mono 16-bit WAV file. Atomic via tmp + rename.

        Returns False if the ring buffer doesn't yet have `seconds` of
        audio (cold start) — the ASR worker waits and retries.
        """
        n_samples = min(int(seconds * self.SAMPLE_RATE), self._buffer_samples)
        with self._lock:
            if self._samples_written < n_samples:
                return False
            start = (self._write_pos - n_samples) % self._buffer_samples
            if start + n_samples <= self._buffer_samples:
                data = self._ring[start:start + n_samples].copy()
            else:
                first_part = self._buffer_samples - start
                data = self._np.concatenate([
                    self._ring[start:],
                    self._ring[:n_samples - first_part],
                ])

        # Atomic write — whisper-cli may open the file while we're
        # writing if we don't go through tmp+rename.
        tmp = self.wav_path + '.tmp'
        with wave.open(tmp, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.SAMPLE_WIDTH_BYTES)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(data.tobytes())
        os.replace(tmp, self.wav_path)
        return True

    @property
    def last_buffer_age(self) -> float:
        """Seconds since the last appsink callback. -1 if no buffer ever
        received (no audio source connected, or pipeline not running)."""
        if self._last_buffer_time == 0.0:
            return -1.0
        return time.time() - self._last_buffer_time

    @property
    def is_active(self) -> bool:
        age = self.last_buffer_age
        return 0.0 <= age < 5.0

    def get_status(self) -> dict:
        return {
            'attached_count': self._attach_count,
            'samples_written': self._samples_written,
            'last_buffer_age_s': round(self.last_buffer_age, 2),
            'is_active': self.is_active,
            'buffer_seconds': self.BUFFER_SECONDS,
            'wav_path': self.wav_path,
        }
