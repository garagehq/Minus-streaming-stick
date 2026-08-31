"""
Health Monitor for Minus.

Unified watchdog that monitors all subsystems and triggers recovery actions.

Features:
- HDMI signal monitoring (detects unplug/replug)
- ustreamer health (frame freshness, not just PID)
- VLM/OCR health monitoring
- Memory/disk usage monitoring
- Automatic recovery actions
"""

import logging
import threading
import time
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import MinusConfig
from v4l2 import probe_v4l2_device

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Current health status of all subsystems."""
    hdmi_signal: bool = False
    hdmi_resolution: str = ""
    hdmi_format: str = ""  # Pixel format (NV12, NV24, etc.)
    ustreamer_alive: bool = False
    ustreamer_responding: bool = False
    last_frame_age: float = -1
    video_pipeline_ok: bool = False
    audio_pipeline_ok: bool = False
    hdmi_tx_audio_link_ok: bool = True  # Sink-side HDMI audio link (Jack + ELD)
    hdmi_tx_audio_link_details: dict = field(default_factory=dict)  # Raw jack/ELD readings for observability
    vlm_ready: bool = False
    vlm_consecutive_timeouts: int = 0
    ocr_ready: bool = False
    memory_percent: float = 0
    disk_free_mb: float = 0
    uptime_seconds: float = 0
    output_fps: float = 0.0


class HealthMonitor:
    """
    Unified health monitor for Minus.

    Runs a background thread that periodically checks all subsystems
    and triggers recovery actions when issues are detected.
    """

    def __init__(self, minus, check_interval: float = 5.0, config: MinusConfig = None):
        """
        Initialize health monitor.

        Args:
            minus: Reference to main Minus instance
            check_interval: How often to check health (seconds)
            config: MinusConfig instance for threshold values
        """
        self.minus = minus
        self.check_interval = check_interval
        self._config = config or MinusConfig()

        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._start_time = time.time()
        self._last_hdmi_signal = None  # None = first check not done yet (avoids false "signal lost" on startup)
        self._hdmi_lost_time = 0
        self._hdmi_fps_zero_since = 0  # When captured_fps first dropped to 0
        self._hdmi_signal_loss_threshold = 5.0  # Seconds of 0 FPS before signal is considered lost
        self._last_no_signal_trigger = 0  # When we last triggered NO SIGNAL mode
        self._no_signal_retry_interval = 3.0  # Retry interval for NO SIGNAL mode if it failed

        # HDMI-TX output tracking (for TV disconnect/reconnect detection)
        self._last_hdmi_output_connected = None  # Track output connector status
        self._hdmi_output_reconnect_time = 0  # When output was last reconnected
        # Two-strike confirmation for the audio unmute watchdog (see below)
        self._muted_not_blocking_seen = False

        # Format/resolution tracking for adaptive restart
        self._last_hdmi_format = None  # Last detected V4L2 format (NV12, NV24, etc.)
        self._last_hdmi_width = 0
        self._last_hdmi_height = 0
        self._format_stable_since = 0  # Time when format became stable (debounce)
        self._format_change_debounce = 2.0  # Wait 2s before triggering restart on format change

        # V4L2 format cache (to avoid probing on every status call)
        self._v4l2_format_cache = ""
        self._v4l2_format_cache_time = 0
        self._v4l2_format_cache_ttl = 10.0  # Cache for 10 seconds

        # HDMI-TX audio link state (populated by monitor loop, read by get_status())
        # Values the monitor thread writes; get_status() may be called from any
        # HTTP request thread, so we never run amixer in that path.
        self._hdmi_tx_audio_link_ok = True
        self._hdmi_tx_audio_link_details: dict = {}
        self._hdmi_tx_audio_link_last_logged = None  # last ok value we logged a transition for

        # Recovery callbacks
        self._on_hdmi_lost: Optional[Callable] = None
        self._on_hdmi_restored: Optional[Callable] = None
        self._on_ustreamer_stall: Optional[Callable] = None
        self._on_video_pipeline_stall: Optional[Callable] = None
        self._on_vlm_failure: Optional[Callable] = None
        self._on_memory_critical: Optional[Callable] = None
        self._on_format_change: Optional[Callable] = None  # Called when HDMI format/resolution changes

        # Thresholds (from config)
        self.frame_stale_threshold = self._config.frame_stale_threshold
        self.memory_warning_percent = 80
        self.memory_critical_percent = 90
        self.disk_warning_mb = 500
        self.vlm_timeout_threshold = 3  # consecutive timeouts before action
        self.startup_grace_period = 30.0  # Don't check ustreamer for first 30s

    def start(self):
        """Start the health monitor thread."""
        self._stop_event.clear()
        self._start_time = time.time()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._monitor_thread.start()
        logger.info("[HealthMonitor] Started")

    def stop(self):
        """Stop the health monitor thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("[HealthMonitor] Stopped")

    def get_status(self) -> HealthStatus:
        """Get current health status."""
        status = HealthStatus()
        status.uptime_seconds = time.time() - self._start_time

        # HDMI signal
        status.hdmi_signal, status.hdmi_resolution = self._check_hdmi_signal()

        # Get current V4L2 format (for format change detection)
        status.hdmi_format = self._get_v4l2_format()

        # ustreamer
        status.ustreamer_alive = self._check_ustreamer_alive()
        status.ustreamer_responding, status.last_frame_age = self._check_ustreamer_responding()

        # Pipelines
        if self.minus.ad_blocker:
            status.video_pipeline_ok = self._check_video_pipeline()
            status.output_fps = self.minus.ad_blocker.get_fps()
        if self.minus.audio:
            status.audio_pipeline_ok = self._check_audio_pipeline()
            status.hdmi_tx_audio_link_ok = self._hdmi_tx_audio_link_ok
            status.hdmi_tx_audio_link_details = dict(self._hdmi_tx_audio_link_details)

        # ML workers
        if self.minus.vlm:
            status.vlm_ready = self.minus.vlm.is_ready
            status.vlm_consecutive_timeouts = getattr(
                self.minus.vlm, 'consecutive_timeouts', 0
            )
        if self.minus.ocr:
            status.ocr_ready = True  # OCR doesn't have ready state

        # Resources
        status.memory_percent = self._get_memory_percent()
        status.disk_free_mb = self._get_disk_free_mb()

        return status

    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.debug("[HealthMonitor] Monitor loop started")

        while not self._stop_event.is_set():
            try:
                self._check_and_recover()
            except Exception as e:
                logger.error(f"[HealthMonitor] Error in check loop: {e}")

            self._stop_event.wait(self.check_interval)

        logger.debug("[HealthMonitor] Monitor loop stopped")

    def _check_and_recover(self):
        """Check health and trigger recovery if needed."""
        # Refresh cached HDMI-TX audio link state before building status.
        # Runs amixer subprocesses; only done here (5s loop), never in get_status().
        self._update_hdmi_tx_audio_link_cache()

        status = self.get_status()

        # HDMI signal monitoring
        # Skip transition callbacks until we've completed startup grace period
        # This prevents false "signal lost/restored" during startup before ustreamer is ready
        uptime = time.time() - self._start_time
        if self._last_hdmi_signal is not None and uptime > self.startup_grace_period:
            if not status.hdmi_signal and self._last_hdmi_signal:
                # Signal just lost
                self._hdmi_lost_time = time.time()
                self._last_no_signal_trigger = time.time()  # Track when we triggered
                logger.warning("[HealthMonitor] HDMI signal LOST")
                if self._on_hdmi_lost:
                    self._on_hdmi_lost()

            elif status.hdmi_signal and not self._last_hdmi_signal:
                # Signal just restored (only if it was actually lost, not just starting up)
                if self._hdmi_lost_time > 0:
                    lost_duration = time.time() - self._hdmi_lost_time
                    logger.info(f"[HealthMonitor] HDMI signal RESTORED (was lost {lost_duration:.1f}s)")
                    if self._on_hdmi_restored:
                        self._on_hdmi_restored()
                else:
                    logger.debug(f"[HealthMonitor] HDMI signal present (initial detection)")
        else:
            # Still in startup grace period - just track state changes without callbacks
            if self._last_hdmi_signal is None:
                logger.debug(f"[HealthMonitor] Initial HDMI signal state: {status.hdmi_signal}")

        self._last_hdmi_signal = status.hdmi_signal

        # Continuous NO SIGNAL mode enforcement
        # Even if we already triggered _on_hdmi_lost, the display might have failed or crashed
        # Check if signal is lost but NO SIGNAL mode isn't active, and re-trigger if needed
        if not status.hdmi_signal and uptime > self.startup_grace_period:
            now = time.time()
            no_signal_active = self._is_no_signal_mode_active()

            if not no_signal_active:
                # NO SIGNAL mode is not active but should be
                time_since_last_trigger = now - self._last_no_signal_trigger
                if time_since_last_trigger >= self._no_signal_retry_interval:
                    logger.warning(f"[HealthMonitor] Signal lost but NO SIGNAL mode not active - re-triggering")
                    self._last_no_signal_trigger = now
                    # Also stamp lost-time so a subsequent restoration fires
                    # _on_hdmi_restored. Without this, when the signal was
                    # already absent at startup (no lost→present transition
                    # ever recorded), `_hdmi_lost_time` stays 0 and the
                    # restored callback is gated out — leaving the lost flag
                    # set on consumers (e.g. detect loops) forever.
                    if self._hdmi_lost_time == 0:
                        self._hdmi_lost_time = now
                    if self._on_hdmi_lost:
                        self._on_hdmi_lost()

        # HDMI-TX output monitoring (detect TV disconnect/reconnect)
        # When TV restarts, the kmssink pipeline loses its DRM connection but keeps "running"
        # We need to detect output reconnection and restart the no-signal pipeline
        if uptime > self.startup_grace_period:
            output_connected = self._check_hdmi_output_connected()

            if self._last_hdmi_output_connected is not None:
                if output_connected and not self._last_hdmi_output_connected:
                    # Output just reconnected (TV turned on/restarted)
                    self._hdmi_output_reconnect_time = time.time()
                    logger.info("[HealthMonitor] HDMI output reconnected (TV turned on)")
                    # Tell the detection loop so it can apply the grace window
                    if self.minus and hasattr(self.minus, 'notify_hdmi_reconnect'):
                        try:
                            self.minus.notify_hdmi_reconnect()
                        except Exception as e:
                            logger.warning(f"[HealthMonitor] notify_hdmi_reconnect failed: {e}")

                    # Give HDMI link time to fully establish before restarting
                    # The TV needs time to complete HDCP handshake and EDID negotiation
                    logger.info("[HealthMonitor] Waiting 2s for HDMI link to stabilize...")
                    time.sleep(2.0)

                    # Restart the appropriate display mode with HDMI reconnect flag
                    # This triggers DPMS cycle and DRM re-probe in the restart method
                    if self._is_no_signal_mode_active():
                        logger.info("[HealthMonitor] Restarting NO SIGNAL display for reconnected output")
                        # For no-signal mode, DPMS cycle is done in start_no_signal_mode()
                        self._force_hdmi_reinit()
                        self._last_no_signal_trigger = time.time()
                        if self._on_hdmi_lost:
                            self._on_hdmi_lost()
                    else:
                        # We have HDMI input but output reconnected - restart display pipeline
                        # with hdmi_reconnect=True to trigger DPMS cycle and DRM re-probe
                        logger.info("[HealthMonitor] Restarting display pipeline for reconnected output (with HDMI reinit)")
                        if self.minus and self.minus.ad_blocker:
                            try:
                                self.minus.ad_blocker.restart(hdmi_reconnect=True)
                                logger.info("[HealthMonitor] Display pipeline restart with HDMI reinit requested")
                            except Exception as e:
                                logger.error(f"[HealthMonitor] Failed to restart display pipeline: {e}")

                elif not output_connected and self._last_hdmi_output_connected:
                    # Output just disconnected (TV turned off)
                    logger.info("[HealthMonitor] HDMI output disconnected (TV turned off)")

            self._last_hdmi_output_connected = output_connected

        # Format/resolution change detection (for adaptive device switching)
        # This allows seamless switching between FireTV, Roku, AppleTV, etc.
        if status.hdmi_signal and status.hdmi_format and uptime > self.startup_grace_period:
            current_format = status.hdmi_format
            if self._last_hdmi_format is None:
                # First detection - just record it
                self._last_hdmi_format = current_format
                self._format_stable_since = time.time()
                logger.info(f"[HealthMonitor] Initial format detected: {current_format}")
            elif current_format != self._last_hdmi_format:
                # Format changed - debounce to avoid rapid restarts during negotiation
                now = time.time()
                if self._format_stable_since == 0:
                    # Start debounce timer
                    self._format_stable_since = now
                    logger.info(f"[HealthMonitor] Format change detected: {self._last_hdmi_format} -> {current_format} (waiting {self._format_change_debounce}s to stabilize)")
                elif (now - self._format_stable_since) >= self._format_change_debounce:
                    # Format has been stable for debounce period - trigger restart
                    old_format = self._last_hdmi_format
                    self._last_hdmi_format = current_format
                    self._format_stable_since = 0
                    logger.warning(f"[HealthMonitor] HDMI format changed: {old_format} -> {current_format}")
                    if self._on_format_change:
                        self._on_format_change(current_format)
            else:
                # Format matches - reset debounce
                self._format_stable_since = 0

        # ustreamer health (skip during startup grace period)
        uptime = time.time() - self._start_time
        if uptime > self.startup_grace_period:
            if status.ustreamer_alive and not status.ustreamer_responding:
                logger.warning("[HealthMonitor] ustreamer not responding to HTTP requests")
                if self._on_ustreamer_stall:
                    self._on_ustreamer_stall()
                # Also trigger video pipeline restart after ustreamer restart
                if self._on_video_pipeline_stall:
                    # Give ustreamer time to restart before triggering video restart
                    import threading
                    def delayed_video_restart():
                        import time
                        time.sleep(3)
                        if self._on_video_pipeline_stall:
                            self._on_video_pipeline_stall()
                    threading.Thread(target=delayed_video_restart, daemon=True).start()

        # Video pipeline health check (FPS-based)
        if uptime > self.startup_grace_period and status.hdmi_signal:
            # If we have HDMI signal but FPS is 0 for a while, pipeline may be stuck
            if status.output_fps == 0 and status.video_pipeline_ok:
                # Pipeline thinks it's OK but no frames flowing
                logger.warning("[HealthMonitor] Video pipeline has 0 FPS - may be stalled")
                if self._on_video_pipeline_stall:
                    self._on_video_pipeline_stall()

        # Audio unmute watchdog - ensure audio is not muted when not blocking
        # This prevents audio from getting stuck muted due to race conditions or bugs.
        #
        # ad_blocker.hide() now unmutes synchronously in the same locked
        # section that clears is_visible, so this should never fire. The two
        # reads below are still not atomic with respect to each other, so
        # require the condition to hold across two consecutive checks before
        # acting — that way a sub-millisecond straddle can't produce a
        # spurious WARN (13 such warnings in 48h before the hide() fix).
        # A genuine stuck mute is still recovered, just one cycle later.
        if self.minus.audio and self.minus.ad_blocker:
            is_blocking = self.minus.ad_blocker.is_visible
            is_muted = self.minus.audio.is_muted

            if not is_blocking and is_muted:
                if self._muted_not_blocking_seen:
                    logger.warning("[HealthMonitor] Audio stuck muted while not blocking - forcing unmute")
                    self.minus.audio.unmute()
                    self._muted_not_blocking_seen = False
                else:
                    self._muted_not_blocking_seen = True
                    logger.debug("[HealthMonitor] Muted while not blocking - "
                                 "confirming on next check")
            else:
                self._muted_not_blocking_seen = False

        # VLM health
        if status.vlm_consecutive_timeouts >= self.vlm_timeout_threshold:
            logger.warning(f"[HealthMonitor] VLM failing ({status.vlm_consecutive_timeouts} consecutive timeouts)")
            if self._on_vlm_failure:
                self._on_vlm_failure()

        # Memory check
        if status.memory_percent >= self.memory_critical_percent:
            logger.error(f"[HealthMonitor] CRITICAL memory usage: {status.memory_percent:.1f}%")
            if self._on_memory_critical:
                self._on_memory_critical()
        elif status.memory_percent >= self.memory_warning_percent:
            logger.warning(f"[HealthMonitor] High memory usage: {status.memory_percent:.1f}%")

        # Disk check
        if status.disk_free_mb < self.disk_warning_mb:
            logger.warning(f"[HealthMonitor] Low disk space: {status.disk_free_mb:.0f}MB free")

        # FPS log (every 60 seconds)
        if int(status.uptime_seconds) % 60 < self.check_interval:
            if status.output_fps > 0:
                logger.info(f"[HealthMonitor] FPS: {status.output_fps:.1f}")
                if status.output_fps < 25:
                    logger.warning(f"[HealthMonitor] Low FPS detected: {status.output_fps:.1f}")

        # Periodic status log (every 5 minutes)
        if int(status.uptime_seconds) % 300 < self.check_interval:
            self._log_status(status)

    def _log_status(self, status: HealthStatus):
        """Log periodic health status."""
        uptime_min = status.uptime_seconds / 60
        logger.info(
            f"[HealthMonitor] Status: uptime={uptime_min:.0f}m "
            f"fps={status.output_fps:.1f} "
            f"hdmi={'OK' if status.hdmi_signal else 'LOST'} "
            f"video={'OK' if status.video_pipeline_ok else 'ERR'} "
            f"audio={'OK' if status.audio_pipeline_ok else 'ERR'} "
            f"vlm={'OK' if status.vlm_ready else 'ERR'} "
            f"mem={status.memory_percent:.0f}% "
            f"disk={status.disk_free_mb:.0f}MB"
        )

    def _is_no_signal_mode_active(self) -> bool:
        """Check if the NO SIGNAL display mode is currently active.

        Returns True if ad_blocker is in 'no_hdmi_device' mode.
        """
        try:
            if not self.minus.ad_blocker:
                return False

            current_source = getattr(self.minus.ad_blocker, 'current_source', None)
            return current_source == 'no_hdmi_device'
        except Exception:
            return False

    def _check_hdmi_output_connected(self) -> bool:
        """Check if any HDMI-TX output is connected via sysfs.

        This is a fast check (no subprocess) that reads /sys/class/drm/card0-HDMI-A-*/status.
        Returns True if at least one HDMI output shows 'connected'.
        """
        try:
            from pathlib import Path
            drm_path = Path('/sys/class/drm')

            for connector in drm_path.glob('card0-HDMI-A-*'):
                status_file = connector / 'status'
                if status_file.exists():
                    status = status_file.read_text().strip()
                    if status == 'connected':
                        return True

            return False
        except Exception:
            return False

    def _force_hdmi_reinit(self):
        """Force HDMI PHY reinitialization via DPMS cycle.

        The kernel hotplug detection doesn't always fully reinitialize the HDMI
        link after TV restart. Cycling DPMS (power management) forces a full
        reinit of the HDMI PHY, similar to physically unplugging/replugging.
        """
        try:
            from drm import probe_drm_output

            # Get current connector ID
            drm_info = probe_drm_output()
            connector_id = drm_info.get('connector_id')
            if not connector_id:
                logger.warning("[HealthMonitor] No connector found for DPMS cycle")
                return

            logger.info(f"[HealthMonitor] Forcing HDMI reinit via DPMS cycle on connector {connector_id}")

            # DPMS Off (value 3) then On (value 0)
            subprocess.run(
                ['modetest', '-M', 'rockchip', '-w', f'{connector_id}:DPMS:3'],
                capture_output=True, timeout=5
            )
            time.sleep(0.5)
            subprocess.run(
                ['modetest', '-M', 'rockchip', '-w', f'{connector_id}:DPMS:0'],
                capture_output=True, timeout=5
            )

            logger.info("[HealthMonitor] DPMS cycle complete")
        except Exception as e:
            logger.warning(f"[HealthMonitor] DPMS cycle failed: {e}")

    def _get_v4l2_format(self) -> str:
        """Get current V4L2 device format and resolution.

        Returns format string like 'NV24@1280x720' for change detection.
        Uses caching to avoid excessive v4l2-ctl calls from frequent API polls.
        """
        # Return cached value if still valid
        now = time.time()
        if self._v4l2_format_cache and (now - self._v4l2_format_cache_time) < self._v4l2_format_cache_ttl:
            return self._v4l2_format_cache

        try:
            device = getattr(self.minus, 'device', '/dev/video0')
            info = probe_v4l2_device(device)
            fmt = info.get('ustreamer_format') or info.get('format') or 'unknown'
            width = info.get('width', 0)
            height = info.get('height', 0)
            result = f"{fmt}@{width}x{height}"

            # Update cache
            self._v4l2_format_cache = result
            self._v4l2_format_cache_time = now
            return result
        except Exception as e:
            logger.debug(f"[HealthMonitor] Error getting V4L2 format: {e}")
            return self._v4l2_format_cache or ""

    def _check_hdmi_signal(self) -> tuple[bool, str]:
        """Check if HDMI signal is present using ustreamer's HTTP API.

        Uses ustreamer's /state endpoint instead of v4l2-ctl --query-dv-timings
        because the V4L2 ioctl can disrupt the HDMI-RX stream causing brief skips.

        Also checks captured_fps to detect when source goes to sleep (still
        connected but not sending frames). If FPS is 0 for more than 5 seconds,
        signal is considered lost.
        """
        try:
            import urllib.request
            import json

            # Use config port if available, default to 9090
            port = getattr(self.minus.config, 'ustreamer_port', 9090) if hasattr(self.minus, 'config') else 9090
            url = f"http://localhost:{port}/state"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                source = data.get('result', {}).get('source', {})

                online = source.get('online', False)
                resolution = source.get('resolution', {})
                width = resolution.get('width', 0)
                height = resolution.get('height', 0)
                captured_fps = source.get('captured_fps', 0)

                # Device must be online with valid resolution
                if not (online and width and height):
                    self._hdmi_fps_zero_since = 0  # Reset FPS tracking
                    return False, ""

                # Check if we're receiving frames (FPS > 0)
                if captured_fps > 0:
                    # Signal is good - reset FPS zero tracking
                    self._hdmi_fps_zero_since = 0
                    return True, f"{width}x{height}"
                else:
                    # FPS is 0 - source might be sleeping
                    now = time.time()
                    if self._hdmi_fps_zero_since == 0:
                        # Just started being 0
                        self._hdmi_fps_zero_since = now
                        # Give it a grace period - still report signal OK
                        return True, f"{width}x{height}"
                    elif (now - self._hdmi_fps_zero_since) < self._hdmi_signal_loss_threshold:
                        # Within grace period - still report signal OK
                        return True, f"{width}x{height}"
                    else:
                        # FPS has been 0 for too long - signal lost
                        logger.debug(f"[HealthMonitor] FPS 0 for {now - self._hdmi_fps_zero_since:.1f}s - signal lost")
                        return False, ""

        except Exception:
            return False, ""

    def _check_ustreamer_alive(self) -> bool:
        """Check if ustreamer process is running."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'ustreamer'],
                capture_output=True, timeout=1
            )
            return result.returncode == 0
        except Exception:
            return False

    def _check_ustreamer_responding(self) -> tuple[bool, float]:
        """Check if ustreamer is serving fresh frames via HTTP."""
        try:
            import urllib.request
            import urllib.error

            # Try to fetch snapshot from ustreamer
            req = urllib.request.Request(
                'http://localhost:9090/snapshot',
                method='HEAD'
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True, 0

            return False, -1

        except urllib.error.URLError:
            return False, -1
        except Exception:
            return False, -1

    def _check_video_pipeline(self) -> bool:
        """Check if video GStreamer pipeline is healthy."""
        try:
            if not self.minus.ad_blocker:
                return False

            pipeline = self.minus.ad_blocker.pipeline
            if not pipeline:
                return False

            # Check pipeline state
            from gi.repository import Gst
            state_ret, state, pending = pipeline.get_state(0)
            return state == Gst.State.PLAYING

        except Exception:
            return False

    def _check_audio_pipeline(self) -> bool:
        """Check if audio GStreamer pipeline is healthy.

        Also verifies the ALSA device owner PID is alive to detect zombie states
        where GStreamer thinks it's playing but the actual audio thread crashed.
        """
        try:
            if not self.minus.audio:
                return False

            status = self.minus.audio.get_status()
            if not isinstance(status, dict):
                return False

            # Skip zombie checks if restart is already in progress
            if status.get('restart_in_progress', False):
                logger.debug("[HealthMonitor] Audio restart in progress, skipping zombie check")
                return False  # Report unhealthy but don't interfere

            gst_playing = status.get('state') == 'playing'

            if gst_playing:
                # Check cooldown - don't run zombie detection right after a restart
                # The audio module tracks last_restart_time internally
                last_restart = getattr(self.minus.audio, '_last_restart_time', 0)
                if last_restart > 0 and (time.time() - last_restart) < 10.0:
                    # Just restarted within 10 seconds, give it time to stabilize
                    logger.debug("[HealthMonitor] Audio recently restarted, skipping zombie check")
                    return True  # Assume healthy during stabilization

                # GStreamer thinks it's playing - verify ALSA device owner is alive
                # BUT only trigger restart if audio buffers are ALSO stale
                # (ALSA owner_pid can be stale even when audio is working fine)
                zombie_detected = self._check_alsa_zombie_state()
                if zombie_detected:
                    # Check if audio buffers are actually flowing
                    last_buffer = getattr(self.minus.audio, '_last_buffer_time', 0)
                    buffer_age = time.time() - last_buffer if last_buffer > 0 else -1

                    # Only restart if buffers are stale (> 5 seconds old)
                    # If buffers are flowing, ALSA owner_pid is just stale metadata
                    if buffer_age < 0 or buffer_age > 5.0:
                        logger.warning(f"[HealthMonitor] Audio zombie: ALSA owner dead AND buffer stale ({buffer_age:.1f}s)")
                        # Trigger a FULL audio restart (not just queue flush)
                        # Use threading to avoid blocking the health monitor
                        import threading
                        def restart_audio():
                            try:
                                # Use _restart_pipeline directly for full restart
                                self.minus.audio._restart_pipeline()
                            except Exception as e:
                                logger.error(f"[HealthMonitor] Failed to restart zombie audio: {e}")
                        threading.Thread(target=restart_audio, daemon=True, name="ZombieAudioRestart").start()
                        logger.info("[HealthMonitor] Triggered full audio pipeline restart for zombie recovery")
                        return False  # Report unhealthy
                    else:
                        logger.debug(f"[HealthMonitor] ALSA owner stale but audio flowing ({buffer_age:.1f}s), ignoring")

            return gst_playing

        except Exception:
            return False

    def _update_hdmi_tx_audio_link_cache(self):
        """Refresh the cached HDMI-TX audio-link state and log transitions.

        Runs in the monitor thread at the normal check interval. Writes
        self._hdmi_tx_audio_link_ok / _details so get_status() can read them
        cheaply from any thread (HTTP handlers call get_status at request rate).

        Only checked when audio pipeline is supposed to be running -
        avoids noisy log transitions during HDMI-signal-loss or startup.
        """
        try:
            if not self.minus or not self.minus.audio or not self.minus.audio.is_running:
                return

            ok, details = self._check_hdmi_tx_audio_link()
            self._hdmi_tx_audio_link_ok = ok
            self._hdmi_tx_audio_link_details = details

            # Log only on state transition to avoid flooding at 5s cadence.
            if self._hdmi_tx_audio_link_last_logged is None:
                self._hdmi_tx_audio_link_last_logged = ok
            elif ok != self._hdmi_tx_audio_link_last_logged:
                if not ok:
                    logger.warning(
                        f"[HealthMonitor] HDMI-TX audio link DESYNC "
                        f"(sink silently stopped honoring audio; projector/TV "
                        f"power-cycle likely required): {details}"
                    )
                else:
                    logger.info(
                        f"[HealthMonitor] HDMI-TX audio link RESTORED: {details}"
                    )
                self._hdmi_tx_audio_link_last_logged = ok
        except Exception as e:
            logger.debug(f"[HealthMonitor] HDMI-TX audio link cache update failed: {e}")

    def _check_hdmi_tx_audio_link(self) -> tuple:
        """Probe sink-side HDMI audio link via ALSA controls.

        ALSA state=RUNNING and hw_ptr advancing do not prove audio is
        actually reaching the sink; we've seen a projector's HDMI audio
        receiver silently wedge while every upstream indicator stayed green.
        This reads two sink-reported signals:

        - Jack control ('rockchip-hdmi* Jack'): boolean, 'on' while the sink
          advertises audio presence. Independent from the DRM connector's
          display status.
        - ELD (EDID-Like Data, 128 bytes): the sink's audio capability block.
          Bytes at offset 16-19 hold the manufacturer/product ID; when the
          sink is in a bad state this region goes all-zero.

        Diagnostic only. Restarting Minus has been observed NOT to recover
        this state; recovery requires the sink to be power-cycled. We log
        so the user knows which way to go.

        Returns:
            (ok, details) — ok False if desync detected. details has 'jack'
            and 'eld_mfr' keys when readable.
        """
        try:
            playback_device = getattr(self.minus.audio, 'playback_device', '')
            if not playback_device.startswith('hw:'):
                return (True, {})
            card_num = playback_device[3:].split(',')[0]

            # Enumerate controls once so we don't hard-code numids that can
            # shift across kernel versions or card types.
            try:
                listing = subprocess.run(
                    ['amixer', '-c', str(card_num), 'controls'],
                    capture_output=True, text=True, timeout=2
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return (True, {})
            if listing.returncode != 0:
                return (True, {})

            jack_numid = None
            eld_numid = None
            for line in listing.stdout.split('\n'):
                if 'Jack' in line and 'iface=CARD' in line:
                    jack_numid = line.split('numid=', 1)[1].split(',', 1)[0]
                elif "name='ELD'" in line:
                    eld_numid = line.split('numid=', 1)[1].split(',', 1)[0]

            details = {}
            jack_ok = True
            eld_ok = True

            if jack_numid:
                try:
                    r = subprocess.run(
                        ['amixer', '-c', str(card_num), 'cget', f'numid={jack_numid}'],
                        capture_output=True, text=True, timeout=2
                    )
                    for line in r.stdout.split('\n'):
                        line = line.strip()
                        if line.startswith(': values='):
                            val = line.split('=', 1)[1].strip()
                            details['jack'] = val
                            if val not in ('on', '1'):
                                jack_ok = False
                            break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

            if eld_numid:
                try:
                    r = subprocess.run(
                        ['amixer', '-c', str(card_num), 'cget', f'numid={eld_numid}'],
                        capture_output=True, text=True, timeout=2
                    )
                    for line in r.stdout.split('\n'):
                        line = line.strip()
                        if line.startswith(': values='):
                            eld_bytes = []
                            for tok in line.split('=', 1)[1].split(','):
                                tok = tok.strip()
                                if tok.startswith('0x'):
                                    try:
                                        eld_bytes.append(int(tok, 16))
                                    except ValueError:
                                        pass
                            if len(eld_bytes) >= 20:
                                # Manufacturer/product ID bytes. All-zero here
                                # means the sink hasn't negotiated audio caps.
                                if all(b == 0 for b in eld_bytes[16:20]):
                                    eld_ok = False
                                    details['eld_mfr'] = 'zeroed'
                                else:
                                    details['eld_mfr'] = 'ok'
                            break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

            return (jack_ok and eld_ok, details)
        except Exception as e:
            logger.debug(f"[HealthMonitor] HDMI-TX audio link probe error: {e}")
            return (True, {})

    def _check_alsa_zombie_state(self) -> bool:
        """Check if ALSA playback device is in zombie state (owned by dead process).

        Returns:
            True if zombie state detected (device RUNNING but owner PID dead)
            False if healthy or unable to determine
        """
        try:
            if not self.minus.audio:
                return False

            # Get playback device from audio module (e.g., "hw:1,0")
            playback_device = getattr(self.minus.audio, 'playback_device', 'hw:0,0')
            if not playback_device.startswith('hw:'):
                return False

            # Parse card number from "hw:X,Y"
            parts = playback_device[3:].split(',')
            if not parts:
                return False
            card_num = parts[0]

            # Read ALSA status
            status_path = f'/proc/asound/card{card_num}/pcm0p/sub0/status'
            try:
                with open(status_path, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                return False  # Device doesn't exist, not a zombie

            # Parse state and owner_pid
            state = None
            owner_pid = None
            for line in content.split('\n'):
                if line.startswith('state:'):
                    state = line.split(':')[1].strip()
                elif line.startswith('owner_pid'):
                    try:
                        owner_pid = int(line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass

            # If state is RUNNING but owner PID doesn't exist, it's a zombie
            if state == 'RUNNING' and owner_pid:
                # Check if owner PID/TID exists
                # Note: ALSA owner_pid can be a thread ID (TID), not just a process ID (PID)
                # Threads appear under /proc/{main_pid}/task/{tid}, not /proc/{tid}
                # So we need to check both locations
                pid_exists = os.path.exists(f'/proc/{owner_pid}')

                # Also check if it's a thread of the main minus process
                if not pid_exists and self.minus:
                    main_pid = os.getpid()
                    tid_exists = os.path.exists(f'/proc/{main_pid}/task/{owner_pid}')
                    if tid_exists:
                        pid_exists = True  # It's a valid thread of our process

                if not pid_exists:
                    logger.debug(f"[HealthMonitor] ALSA zombie: card{card_num} state=RUNNING but owner_pid={owner_pid} is dead")
                    return True

            return False

        except Exception as e:
            logger.debug(f"[HealthMonitor] ALSA zombie check error: {e}")
            return False

    def _get_memory_percent(self) -> float:
        """Get current memory usage percentage."""
        try:
            with open('/proc/meminfo') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(':')] = int(parts[1])

            total = meminfo.get('MemTotal', 1)
            available = meminfo.get('MemAvailable', 0)
            used = total - available
            return (used / total) * 100

        except Exception:
            return 0

    def _get_disk_free_mb(self) -> float:
        """Get free disk space in MB for current directory."""
        try:
            stat = os.statvfs('.')
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes / (1024 * 1024)
        except Exception:
            return 0

    # Recovery action setters
    def on_hdmi_lost(self, callback: Callable):
        """Set callback for HDMI signal loss."""
        self._on_hdmi_lost = callback

    def on_hdmi_restored(self, callback: Callable):
        """Set callback for HDMI signal restoration."""
        self._on_hdmi_restored = callback

    def on_ustreamer_stall(self, callback: Callable):
        """Set callback for ustreamer stall."""
        self._on_ustreamer_stall = callback

    def on_video_pipeline_stall(self, callback: Callable):
        """Set callback for video pipeline stall."""
        self._on_video_pipeline_stall = callback

    def on_vlm_failure(self, callback: Callable):
        """Set callback for VLM failure."""
        self._on_vlm_failure = callback

    def on_memory_critical(self, callback: Callable):
        """Set callback for critical memory usage."""
        self._on_memory_critical = callback

    def on_format_change(self, callback: Callable):
        """Set callback for HDMI format/resolution change.

        The callback receives the new format string (e.g., 'NV24@1280x720').
        This allows automatic adaptation when switching between devices
        like FireTV, Roku, AppleTV, etc.
        """
        self._on_format_change = callback
