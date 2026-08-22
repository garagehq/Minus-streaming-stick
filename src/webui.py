"""
Minus Web UI

Lightweight Flask-based web interface for monitoring and controlling Minus.
Accessible via Tailscale for remote debugging and control.

Features:
- Live video feed (proxied from ustreamer)
- Status display (blocking state, FPS, HDMI, etc.)
- Pause/resume blocking (custom duration support)
- Recent detection history
- Log viewer
- Fire TV remote control
- WiFi network management (nmcli)
- ADB RSA key management
- Screenshot gallery
- Configuration management
"""

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, Response, send_from_directory, redirect
import requests


AXCL_SMI_PATH = '/usr/bin/axcl/axcl-smi'
_axera_metrics_cache = {'ts': 0.0, 'data': None}
_AXERA_METRICS_TTL = 5.0


def get_axera_metrics():
    """Query axcl-smi for Axera NPU telemetry.

    Returns dict with temperature_c, npu_usage_pct, cmm_used_kib, cmm_total_kib,
    or None if axcl-smi is unavailable or returns an error.

    Cached for 5s — subprocess latency is ~100ms per call and these metrics
    don't change meaningfully faster than that.
    """
    now = time.time()
    if (_axera_metrics_cache['data'] is not None and
            now - _axera_metrics_cache['ts'] < _AXERA_METRICS_TTL):
        return _axera_metrics_cache['data']
    if not os.path.exists(AXCL_SMI_PATH):
        return None
    metrics = {}
    try:
        r = subprocess.run([AXCL_SMI_PATH, 'info', '--temp'],
                           capture_output=True, text=True, timeout=1.0)
        m = re.search(r'temperature\s*:\s*(\d+)', r.stdout)
        if m:
            # axcl-smi returns temperature in milli-degrees Celsius
            metrics['temperature_c'] = int(m.group(1)) / 1000.0
        r = subprocess.run([AXCL_SMI_PATH, 'info', '--npu'],
                           capture_output=True, text=True, timeout=1.0)
        m = re.search(r':\s*(\d+)%', r.stdout)
        if m:
            metrics['npu_usage_pct'] = int(m.group(1))
        r = subprocess.run([AXCL_SMI_PATH, 'info', '--cmm'],
                           capture_output=True, text=True, timeout=1.0)
        mt = re.search(r'CMM Total\s*:\s*(\d+)', r.stdout)
        mu = re.search(r'CMM Used\s*:\s*(\d+)', r.stdout)
        if mt:
            metrics['cmm_total_kib'] = int(mt.group(1))
        if mu:
            metrics['cmm_used_kib'] = int(mu.group(1))
    except Exception:
        return None
    if not metrics:
        return None
    _axera_metrics_cache['data'] = metrics
    _axera_metrics_cache['ts'] = now
    return metrics

from src.wifi_manager import get_wifi_manager

# Reviewed screenshots tracking
REVIEWED_FILE = Path("/home/radxa/.minus_reviewed_screenshots.json")

logger = logging.getLogger('Minus.WebUI')


class WebUI:
    """Web UI server for Minus."""

    def __init__(self, minus_instance, port: int = 80, ustreamer_port: int = 9090):
        """
        Initialize web UI.

        Args:
            minus_instance: Minus instance to control
            port: Port to run web server on
            ustreamer_port: Port where ustreamer is running (for stream proxy)
        """
        self.minus = minus_instance
        self.port = port
        self.ustreamer_port = ustreamer_port
        self.server_thread = None
        self.running = False

        # Screenshot review state
        self._reviewed = set()
        self._reviewed_lock = threading.Lock()
        self._undo_stack = []  # [{action, filename, source, target}]
        self._load_reviewed()

        # Create Flask app
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'),
        )

        # Disable Flask's default logging (we use our own)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register all Flask routes."""

        @self.app.route('/')
        def index():
            """Serve the main UI page."""
            return send_from_directory(
                self.app.template_folder,
                'index.html'
            )

        @self.app.route('/api/status')
        def api_status():
            """Get current status."""
            try:
                status = self.minus.get_status_dict()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/pause/<int:minutes>', methods=['POST'])
        def api_pause(minutes):
            """Pause blocking for specified minutes (1-600)."""
            if minutes < 1 or minutes > 600:
                return jsonify({'error': 'Invalid duration. Use 1-600 minutes.'}), 400

            try:
                self.minus.pause_blocking(minutes * 60)
                return jsonify({
                    'success': True,
                    'paused_until': self.minus.blocking_paused_until,
                    'duration_minutes': minutes,
                })
            except Exception as e:
                logger.error(f"Error pausing: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/resume', methods=['POST'])
        def api_resume():
            """Resume blocking immediately."""
            try:
                self.minus.resume_blocking()
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Error resuming: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/detections')
        def api_detections():
            """Get recent detection history."""
            try:
                detections = list(self.minus.detection_history)
                # Return in reverse order (newest first)
                return jsonify({'detections': detections[::-1]})
            except Exception as e:
                logger.error(f"Error getting detections: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/logs')
        def api_logs():
            """Get recent log lines."""
            try:
                log_file = Path('/tmp/minus.log')
                if log_file.exists():
                    # Read last 100 lines
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]
                    return jsonify({'lines': [line.rstrip() for line in lines]})
                return jsonify({'lines': []})
            except Exception as e:
                logger.error(f"Error reading logs: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/preview')
        def api_preview_status():
            """Get preview window status."""
            try:
                enabled = False
                if self.minus.ad_blocker:
                    enabled = self.minus.ad_blocker.is_preview_enabled()
                return jsonify({'preview_enabled': enabled})
            except Exception as e:
                logger.error(f"Error getting preview status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/preview/enable', methods=['POST'])
        def api_preview_enable():
            """Enable the preview window."""
            try:
                if self.minus.ad_blocker:
                    self.minus.ad_blocker.set_preview_enabled(True)
                return jsonify({'success': True, 'preview_enabled': True})
            except Exception as e:
                logger.error(f"Error enabling preview: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/preview/disable', methods=['POST'])
        def api_preview_disable():
            """Disable the preview window."""
            try:
                if self.minus.ad_blocker:
                    self.minus.ad_blocker.set_preview_enabled(False)
                return jsonify({'success': True, 'preview_enabled': False})
            except Exception as e:
                logger.error(f"Error disabling preview: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug-overlay')
        def api_debug_overlay_status():
            """Get debug overlay status."""
            try:
                enabled = False
                if self.minus.ad_blocker:
                    enabled = self.minus.ad_blocker.is_debug_overlay_enabled()
                return jsonify({'debug_overlay_enabled': enabled})
            except Exception as e:
                logger.error(f"Error getting debug overlay status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug-overlay/enable', methods=['POST'])
        def api_debug_overlay_enable():
            """Enable the debug overlay (persisted)."""
            try:
                self.minus.set_debug_overlay_enabled(True)
                return jsonify({'success': True, 'debug_overlay_enabled': True})
            except Exception as e:
                logger.error(f"Error enabling debug overlay: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug-overlay/disable', methods=['POST'])
        def api_debug_overlay_disable():
            """Disable the debug overlay (persisted)."""
            try:
                self.minus.set_debug_overlay_enabled(False)
                return jsonify({'success': True, 'debug_overlay_enabled': False})
            except Exception as e:
                logger.error(f"Error disabling debug overlay: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/pixelated-background')
        def api_pixelated_background_status():
            """Get pixelated background status."""
            try:
                enabled = False
                if self.minus.ad_blocker:
                    enabled = self.minus.ad_blocker.is_pixelated_background_enabled()
                return jsonify({'pixelated_background_enabled': enabled})
            except Exception as e:
                logger.error(f"Error getting pixelated background status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/pixelated-background/enable', methods=['POST'])
        def api_pixelated_background_enable():
            """Enable pixelated background during ad blocking."""
            try:
                if self.minus.ad_blocker:
                    self.minus.ad_blocker.set_pixelated_background_enabled(True)
                return jsonify({'success': True, 'pixelated_background_enabled': True})
            except Exception as e:
                logger.error(f"Error enabling pixelated background: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/pixelated-background/disable', methods=['POST'])
        def api_pixelated_background_disable():
            """Disable pixelated background during ad blocking."""
            try:
                if self.minus.ad_blocker:
                    self.minus.ad_blocker.set_pixelated_background_enabled(False)
                return jsonify({'success': True, 'pixelated_background_enabled': False})
            except Exception as e:
                logger.error(f"Error disabling pixelated background: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/firetv-keepalive')
        def api_firetv_keepalive_status():
            """Get Fire TV keep-alive status."""
            try:
                enabled = False
                if self.minus.fire_tv_controller:
                    enabled = self.minus.fire_tv_controller.is_keepalive_enabled()
                return jsonify({'keepalive_enabled': enabled})
            except Exception as e:
                logger.error(f"Error getting Fire TV keep-alive status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/firetv-keepalive/enable', methods=['POST'])
        def api_firetv_keepalive_enable():
            """Enable Fire TV keep-alive pings."""
            try:
                if self.minus.fire_tv_controller:
                    self.minus.fire_tv_controller.set_keepalive_enabled(True)
                return jsonify({'success': True, 'keepalive_enabled': True})
            except Exception as e:
                logger.error(f"Error enabling Fire TV keep-alive: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/firetv-keepalive/disable', methods=['POST'])
        def api_firetv_keepalive_disable():
            """Disable Fire TV keep-alive pings."""
            try:
                if self.minus.fire_tv_controller:
                    self.minus.fire_tv_controller.set_keepalive_enabled(False)
                return jsonify({'success': True, 'keepalive_enabled': False})
            except Exception as e:
                logger.error(f"Error disabling Fire TV keep-alive: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/test/trigger-block', methods=['POST'])
        def api_test_trigger_block():
            """Trigger ad blocking for testing.

            Optional JSON body:
            - duration: seconds to block (default: 10, max: 60)
            - source: detection source ('ocr', 'vlm', 'both', 'default')
            - kind: force a specific replacement kind for this block instead
              of letting the roll decide. One of 'vocab' / 'fact' / 'photos'.
              Bypasses the 30s cooldown and lock-in RNG so test scripts can
              exercise each rendering path deterministically.
            """
            try:
                data = request.get_json() or {}

                # Validate duration
                duration = data.get('duration', 10)
                if not isinstance(duration, (int, float)) or duration < 1 or duration > 60:
                    return jsonify({
                        'success': False,
                        'error': 'duration must be a number between 1 and 60 seconds'
                    }), 400
                duration = int(duration)

                # Validate source
                source = data.get('source', 'ocr')
                valid_sources = ('ocr', 'vlm', 'both', 'default')
                if source not in valid_sources:
                    return jsonify({
                        'success': False,
                        'error': f'source must be one of: {", ".join(valid_sources)}'
                    }), 400

                # Validate (optional) forced replacement kind
                kind = data.get('kind')
                valid_kinds = ('vocab', 'fact', 'photos')
                if kind is not None and kind not in valid_kinds:
                    return jsonify({
                        'success': False,
                        'error': f'kind must be one of: {", ".join(valid_kinds)}'
                    }), 400

                if self.minus.ad_blocker:
                    # Enable test mode to prevent detection loop from hiding
                    self.minus.ad_blocker.set_test_mode(duration)

                    # If caller asked for a specific replacement kind, set
                    # the lock BEFORE show() so the block-start roll uses
                    # the forced value. Extend the cooldown past the test
                    # duration so rotations stick with it.
                    if kind is not None:
                        self.minus.ad_blocker._locked_content_kind = kind
                        self.minus.ad_blocker._content_kind_lock_until = (
                            time.time() + duration + 60
                        )

                    # Show blocking overlay. For OCR-shaped sources, inject a
                    # synthetic trigger snippet so the top-right "(Ad) 0:30 left"
                    # hint actually renders during a test-fired block (real
                    # detection would supply this from check_ad_keywords).
                    fake_trigger = ''
                    if source in ('ocr', 'both'):
                        fake_trigger = ('Ad', 'Ad 0:30 left')
                    self.minus.ad_blocker.show(source, ocr_trigger_text=fake_trigger)

                    # Schedule auto-hide after duration
                    def auto_hide():
                        time.sleep(duration)
                        if self.minus.ad_blocker:
                            self.minus.ad_blocker.clear_test_mode()
                            self.minus.ad_blocker.hide(force=True)
                            logger.info(f"[WebUI] Test blocking ended after {duration}s")

                    threading.Thread(target=auto_hide, daemon=True).start()

                    logger.info(
                        f"[WebUI] Test blocking triggered: source={source}, "
                        f"duration={duration}s, kind={kind or 'auto'}"
                    )
                    return jsonify({
                        'success': True,
                        'source': source,
                        'duration': duration,
                        'kind': kind,
                        'message': f'Blocking for {duration} seconds',
                    })
                else:
                    return jsonify({'error': 'Ad blocker not initialized'}), 500
            except Exception as e:
                logger.error(f"Error triggering test block: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/test/replacement-mode/<kind>', methods=['POST'])
        def api_test_replacement_mode(kind):
            """Convenience: trigger a block with `kind` forced.

            Shorthand for ``/api/test/trigger-block`` with a pinned kind.
            ``POST /api/test/replacement-mode/photos`` → 30s block locked to
            photo-cycling regardless of cooldown or weights.
            """
            valid_kinds = ('vocab', 'fact', 'photos')
            if kind not in valid_kinds:
                return jsonify({
                    'success': False,
                    'error': f'kind must be one of: {", ".join(valid_kinds)}'
                }), 400
            try:
                # silent=True so a POST with no body is allowed (common when
                # exercising from curl without Content-Type: application/json).
                data = request.get_json(silent=True) or {}
                duration = int(data.get('duration', 30))
                if duration < 1 or duration > 60:
                    duration = 30

                if not self.minus.ad_blocker:
                    return jsonify({'success': False, 'error': 'ad_blocker missing'}), 500

                # Gate photos mode on having at least one photo; otherwise
                # the rotation loop silently falls back to text, which is
                # confusing while testing.
                if kind == 'photos':
                    from photo_library import get_photo_library
                    if get_photo_library().random_photo_id() is None:
                        return jsonify({
                            'success': False,
                            'error': 'no photos uploaded — upload at least one in /api/media/photos first'
                        }), 400

                ab = self.minus.ad_blocker
                ab.set_test_mode(duration)
                ab._locked_content_kind = kind
                ab._content_kind_lock_until = time.time() + duration + 60
                ab.show('ocr')

                def auto_hide():
                    time.sleep(duration)
                    if self.minus.ad_blocker:
                        self.minus.ad_blocker.clear_test_mode()
                        self.minus.ad_blocker.hide(force=True)
                threading.Thread(target=auto_hide, daemon=True).start()

                return jsonify({
                    'success': True,
                    'kind': kind,
                    'duration': duration,
                    'message': f'Blocking in {kind} mode for {duration}s',
                })
            except Exception as e:
                logger.error(f"Error in replacement-mode test: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/test/countdown-bar', methods=['POST'])
        def api_test_countdown_bar():
            """Simulate an OCR ad-timer reading so the countdown bar renders.

            POST body: ``{"seconds": 15}``. Triggers a 20s test block with
            vocab mode and injects the given seconds as the ad-time-remaining
            so the `AD LEFT [###...] 12s` bar appears in stats. Good for
            visual regression checks of the progress bar.
            """
            try:
                data = request.get_json() or {}
                seconds = int(data.get('seconds', 15))
                if seconds < 0 or seconds > 300:
                    return jsonify({
                        'success': False,
                        'error': 'seconds must be 0..300'
                    }), 400
                duration = int(data.get('duration', 20))

                if not self.minus.ad_blocker:
                    return jsonify({'success': False, 'error': 'ad_blocker missing'}), 500

                ab = self.minus.ad_blocker
                ab.set_test_mode(duration)
                ab._locked_content_kind = 'vocab'
                ab._content_kind_lock_until = time.time() + duration + 60
                ab.show('ocr')
                # Inject the countdown right after show() — the bar picks
                # this up on the next /blocking/set tick.
                ab.set_ad_seconds_remaining(seconds)

                def auto_hide():
                    time.sleep(duration)
                    if self.minus.ad_blocker:
                        self.minus.ad_blocker.clear_test_mode()
                        self.minus.ad_blocker.hide(force=True)
                threading.Thread(target=auto_hide, daemon=True).start()

                return jsonify({'success': True, 'seconds': seconds, 'duration': duration})
            except Exception as e:
                logger.error(f"Error in countdown-bar test: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/test/audio-bars', methods=['GET'])
        def api_test_audio_bars():
            """Return the current audio RMS history + the rendered bar string.

            Handy for verifying the audio-reactive visualizer without having
            to wait for a real ad block. ``width`` query param controls the
            bar width (default 16).
            """
            try:
                width = int(request.args.get('width', 16))
                width = max(1, min(64, width))
                if not getattr(self.minus, 'audio', None):
                    return jsonify({'bars': '', 'history': [], 'error': 'no audio'}), 200
                audio = self.minus.audio
                history = list(getattr(audio, '_level_history', []))
                bars = audio.get_level_bars(width=width) if hasattr(audio, 'get_level_bars') else ''
                return jsonify({
                    'bars': bars,
                    'width': width,
                    'history': [round(v, 3) for v in history],
                    'samples': len(history),
                })
            except Exception as e:
                logger.error(f"Error in audio-bars test: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/test/stop-block', methods=['POST'])
        def api_test_stop_block():
            """Stop ad blocking (for testing)."""
            try:
                if self.minus.ad_blocker:
                    self.minus.ad_blocker.clear_test_mode()
                    self.minus.ad_blocker.hide(force=True)
                    logger.info("[WebUI] Test blocking stopped")
                    return jsonify({'success': True})
                else:
                    return jsonify({'error': 'Ad blocker not initialized'}), 500
            except Exception as e:
                logger.error(f"Error stopping test block: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/stream')
        def stream_proxy():
            """Proxy the MJPEG stream from ustreamer (for CORS bypass)."""
            try:
                # Stream from ustreamer
                url = f'http://localhost:{self.ustreamer_port}/stream'
                req = requests.get(url, stream=True, timeout=10)

                def generate():
                    try:
                        for chunk in req.iter_content(chunk_size=8192):
                            if chunk:
                                yield chunk
                    finally:
                        # Ensure the upstream socket is closed when the client
                        # disconnects; otherwise we leak sockets on every page
                        # refresh / stream reconnect.
                        req.close()

                # Pass through the Content-Type from ustreamer (includes correct boundary)
                content_type = req.headers.get('Content-Type', 'multipart/x-mixed-replace;boundary=boundarydonotcross')

                return Response(
                    generate(),
                    mimetype=content_type,
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                    }
                )
            except Exception as e:
                logger.error(f"Stream proxy error: {e}")
                return Response(status=503)

        @self.app.route('/snapshot')
        def snapshot_proxy():
            """Proxy a single snapshot from ustreamer."""
            try:
                url = f'http://localhost:{self.ustreamer_port}/snapshot'
                req = requests.get(url, timeout=5)
                return Response(
                    req.content,
                    mimetype='image/jpeg',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                    }
                )
            except Exception as e:
                logger.error(f"Snapshot proxy error: {e}")
                return Response(status=503)

        # =========================================================================
        # Device Configuration
        # =========================================================================

        @self.app.route('/api/device/config')
        def api_device_config():
            """Get current device configuration."""
            try:
                from src.device_config import get_device_config_manager
                manager = get_device_config_manager()
                return jsonify(manager.get_config())
            except Exception as e:
                logger.error(f"Error getting device config: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/device/types')
        def api_device_types():
            """Get available device types."""
            try:
                from src.device_config import get_device_config_manager
                manager = get_device_config_manager()
                return jsonify({'devices': manager.get_available_devices()})
            except Exception as e:
                logger.error(f"Error getting device types: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/device/select', methods=['POST'])
        def api_device_select():
            """Select a device type and switch to appropriate setup."""
            try:
                from src.device_config import get_device_config_manager
                data = request.get_json() or {}
                device_type = data.get('device_type')

                if not device_type:
                    return jsonify({'success': False, 'error': 'device_type required'}), 400

                # Disconnect any existing device controllers
                if hasattr(self.minus, 'roku_controller') and self.minus.roku_controller:
                    try:
                        self.minus.roku_controller.disconnect()
                        self.minus.roku_controller = None
                        logger.info("[WebUI] Disconnected Roku controller")
                    except Exception as e:
                        logger.warning(f"[WebUI] Error disconnecting Roku: {e}")

                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    try:
                        self.minus.fire_tv_setup.stop_setup()
                        logger.info("[WebUI] Stopped Fire TV setup")
                    except Exception as e:
                        logger.warning(f"[WebUI] Error stopping Fire TV setup: {e}")

                if hasattr(self.minus, 'fire_tv_controller') and self.minus.fire_tv_controller:
                    try:
                        self.minus.fire_tv_controller.disconnect()
                        self.minus.fire_tv_controller = None
                        logger.info("[WebUI] Disconnected Fire TV controller")
                    except Exception as e:
                        logger.warning(f"[WebUI] Error disconnecting Fire TV: {e}")

                # Save the new device type
                manager = get_device_config_manager()
                result = manager.set_device_type(device_type)
                logger.info(f"[WebUI] Device type set to: {device_type}")

                # Show setup instructions overlay for the new device
                try:
                    from src.overlay import FireTVNotification, GoogleTVNotification, RokuNotification
                    if device_type == 'fire_tv':
                        notification = FireTVNotification(ustreamer_port=self.minus.config.ustreamer_port)
                        notification.show_adb_enable_instructions(timeout_remaining=300)
                    elif device_type == 'google_tv':
                        notification = GoogleTVNotification(ustreamer_port=self.minus.config.ustreamer_port)
                        notification.show_adb_enable_instructions(timeout_remaining=300)
                    elif device_type == 'roku':
                        notification = RokuNotification(ustreamer_port=self.minus.config.ustreamer_port)
                        notification.show_setup_instructions()
                except Exception as e:
                    logger.warning(f"[WebUI] Could not show setup overlay: {e}")

                return jsonify(result)
            except Exception as e:
                logger.error(f"Error selecting device: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/device/ip', methods=['POST'])
        def api_device_ip():
            """Set device IP address."""
            try:
                from src.device_config import get_device_config_manager
                data = request.get_json() or {}
                ip = data.get('ip', '')

                manager = get_device_config_manager()
                result = manager.set_device_ip(ip)
                logger.info(f"[WebUI] Device IP set to: {ip}")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error setting device IP: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/device/setup-complete', methods=['POST'])
        def api_device_setup_complete():
            """Mark device setup as complete."""
            try:
                from src.device_config import get_device_config_manager
                data = request.get_json() or {}
                complete = data.get('complete', True)

                manager = get_device_config_manager()
                result = manager.set_setup_complete(complete)
                logger.info(f"[WebUI] Device setup complete: {complete}")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error setting setup complete: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/device/reset', methods=['POST'])
        def api_device_reset():
            """Reset device configuration."""
            try:
                from src.device_config import get_device_config_manager
                manager = get_device_config_manager()
                result = manager.reset()
                logger.info("[WebUI] Device configuration reset")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error resetting device config: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Fire TV Remote Control
        # =========================================================================

        @self.app.route('/api/firetv/status')
        def api_firetv_status():
            """Get Fire TV connection status."""
            try:
                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    setup = self.minus.fire_tv_setup
                    return jsonify({
                        'connected': setup.is_connected(),
                        'state': setup.state,
                        'device_ip': setup.device_ip if hasattr(setup, 'device_ip') else None,
                    })
                return jsonify({'connected': False, 'state': 'not_initialized'})
            except Exception as e:
                logger.error(f"Error getting Fire TV status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/firetv/scan')
        def api_firetv_scan():
            """Scan for Fire TV devices on the network."""
            try:
                from src.fire_tv import FireTVController
                devices = FireTVController.discover_devices(timeout=10.0)
                return jsonify({
                    'devices': devices,
                    'count': len(devices),
                })
            except Exception as e:
                logger.error(f"Error scanning for Fire TV: {e}")
                return jsonify({'devices': [], 'error': str(e)}), 500

        @self.app.route('/api/firetv/connect', methods=['POST'])
        def api_firetv_connect():
            """Connect to a Fire TV device by IP."""
            try:
                data = request.get_json() or {}
                ip_address = data.get('ip')
                if not ip_address:
                    return jsonify({'error': 'IP address required'}), 400

                # Start Fire TV setup with the provided IP
                if hasattr(self.minus, '_start_fire_tv_setup'):
                    self.minus._start_fire_tv_setup(saved_ip=ip_address, device_type='fire_tv')
                    return jsonify({'success': True, 'message': f'Connecting to {ip_address}...'})
                else:
                    return jsonify({'error': 'Fire TV setup not available'}), 500
            except Exception as e:
                logger.error(f"Error connecting to Fire TV: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/firetv/command', methods=['POST'])
        def api_firetv_command():
            """Send a command to Fire TV."""
            try:
                data = request.get_json() or {}
                command = data.get('command')

                valid_commands = [
                    'up', 'down', 'left', 'right', 'select', 'back', 'home', 'menu',
                    'play', 'pause', 'play_pause', 'fast_forward', 'rewind',
                    'volume_up', 'volume_down', 'mute'
                ]

                if command not in valid_commands:
                    return jsonify({'error': f'Invalid command. Valid: {valid_commands}'}), 400

                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    controller = self.minus.fire_tv_setup.get_controller()
                    if controller and controller.is_connected:
                        controller.send_command(command)
                        return jsonify({'success': True, 'command': command})
                    return jsonify({'error': 'Fire TV not connected'}), 503

                return jsonify({'error': 'Fire TV not initialized'}), 500
            except Exception as e:
                logger.error(f"Error sending Fire TV command: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Google TV / Android TV Remote Control (uses ADB like Fire TV)
        # =========================================================================

        @self.app.route('/api/googletv/status')
        def api_googletv_status():
            """Get Google TV connection status."""
            try:
                # Google TV uses the same ADB controller as Fire TV
                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    controller = self.minus.fire_tv_setup.get_controller()
                    if controller:
                        return jsonify({
                            'connected': controller.is_connected,
                            'device_info': {
                                'ip': controller._ip_address,
                                'device_type': 'google_tv',
                            } if controller.is_connected else None
                        })
                return jsonify({'connected': False, 'device_info': None})
            except Exception as e:
                logger.error(f"Error getting Google TV status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/googletv/scan')
        def api_googletv_scan():
            """Scan for Google TV / Android TV devices on the network."""
            try:
                from src.fire_tv import FireTVController
                # Use the same discovery as Fire TV - both use ADB
                devices = FireTVController.discover_devices(timeout=10.0)
                # Filter or label as Google TV
                for device in devices:
                    device['device_type'] = 'google_tv'
                return jsonify({
                    'devices': devices,
                    'count': len(devices),
                })
            except Exception as e:
                logger.error(f"Error scanning for Google TV: {e}")
                return jsonify({'devices': [], 'error': str(e)}), 500

        @self.app.route('/api/googletv/connect', methods=['POST'])
        def api_googletv_connect():
            """Connect to a Google TV device by IP."""
            try:
                data = request.get_json() or {}
                ip_address = data.get('ip')
                if not ip_address:
                    return jsonify({'error': 'IP address required'}), 400

                # Start setup with the provided IP (uses same ADB setup as Fire TV)
                if hasattr(self.minus, '_start_fire_tv_setup'):
                    self.minus._start_fire_tv_setup(saved_ip=ip_address, device_type='google_tv')
                    return jsonify({'success': True, 'message': f'Connecting to {ip_address}...'})
                else:
                    return jsonify({'error': 'Google TV setup not available'}), 500
            except Exception as e:
                logger.error(f"Error connecting to Google TV: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/googletv/command', methods=['POST'])
        def api_googletv_command():
            """Send a command to Google TV."""
            try:
                data = request.get_json() or {}
                command = data.get('command')

                valid_commands = [
                    'up', 'down', 'left', 'right', 'select', 'back', 'home', 'menu',
                    'play', 'pause', 'play_pause', 'fast_forward', 'rewind',
                    'volume_up', 'volume_down', 'mute', 'assistant'
                ]

                if command not in valid_commands:
                    return jsonify({'error': f'Invalid command. Valid: {valid_commands}'}), 400

                # Google TV uses the same ADB controller as Fire TV
                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    controller = self.minus.fire_tv_setup.get_controller()
                    if controller and controller.is_connected:
                        controller.send_command(command)
                        return jsonify({'success': True, 'command': command})
                    return jsonify({'error': 'Google TV not connected'}), 503

                return jsonify({'error': 'Google TV not initialized'}), 500
            except Exception as e:
                logger.error(f"Error sending Google TV command: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Roku Remote Control
        # =========================================================================

        @self.app.route('/api/roku/status')
        def api_roku_status():
            """Get Roku connection status."""
            try:
                if hasattr(self.minus, 'roku_controller') and self.minus.roku_controller:
                    controller = self.minus.roku_controller
                    return jsonify({
                        'connected': controller.is_connected(),
                        'device_info': controller.get_device_info(),
                    })
                return jsonify({'connected': False, 'device_info': None})
            except Exception as e:
                logger.error(f"Error getting Roku status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/roku/discover')
        def api_roku_discover():
            """Discover Roku devices on the network."""
            try:
                from src.roku import RokuController
                devices = RokuController.discover_devices(timeout=5.0)
                return jsonify({'devices': devices})
            except Exception as e:
                logger.error(f"Error discovering Roku devices: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/roku/connect', methods=['POST'])
        def api_roku_connect():
            """Connect to a Roku device."""
            try:
                data = request.get_json() or {}
                ip = data.get('ip')

                if not ip:
                    return jsonify({'success': False, 'error': 'IP address required'}), 400

                # Create or get Roku controller
                from src.roku import RokuController
                if not hasattr(self.minus, 'roku_controller') or not self.minus.roku_controller:
                    self.minus.roku_controller = RokuController()

                if self.minus.roku_controller.connect(ip):
                    # Update device config
                    from src.device_config import get_device_config_manager
                    manager = get_device_config_manager()
                    manager.set_device_ip(ip)
                    manager.set_setup_complete(True)

                    # Follow the Roku across DHCP changes and persist the
                    # new IP (mirrors the startup wiring in minus.py)
                    if hasattr(self.minus, '_persist_roku_ip'):
                        self.minus.roku_controller.set_ip_change_callback(self.minus._persist_roku_ip)

                    # Connect autonomous mode to Roku controller
                    if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                        self.minus.autonomous_mode.set_device_controller(self.minus.roku_controller, 'roku')
                        logger.info("[API] Roku controller connected to autonomous mode")

                    # Show connected notification and clear setup overlay
                    try:
                        from src.overlay import RokuNotification
                        device_info = self.minus.roku_controller.get_device_info()
                        device_name = device_info.get('name', 'Roku') if device_info else 'Roku'
                        model = device_info.get('model', '') if device_info else ''
                        full_name = f"{device_name} {model}".strip()
                        notification = RokuNotification(ustreamer_port=self.minus.config.ustreamer_port)
                        notification.show_connected(full_name)
                    except Exception as e:
                        logger.warning(f"Could not show connected notification: {e}")

                    return jsonify({
                        'success': True,
                        'connected': True,
                        'device_info': self.minus.roku_controller.get_device_info()
                    })
                return jsonify({'success': False, 'error': 'Connection failed'}), 503
            except Exception as e:
                logger.error(f"Error connecting to Roku: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/roku/command', methods=['POST'])
        def api_roku_command():
            """Send a command to Roku."""
            try:
                data = request.get_json() or {}
                command = data.get('command')

                valid_commands = [
                    'up', 'down', 'left', 'right', 'select', 'back', 'home', 'info',
                    'play', 'pause', 'play_pause', 'fast_forward', 'rewind',
                    'volume_up', 'volume_down', 'mute', 'instant_replay', 'search'
                ]

                if command not in valid_commands:
                    return jsonify({'error': f'Invalid command. Valid: {valid_commands}'}), 400

                if hasattr(self.minus, 'roku_controller') and self.minus.roku_controller:
                    if self.minus.roku_controller.is_connected():
                        self.minus.roku_controller.send_command(command)
                        return jsonify({'success': True, 'command': command})
                    return jsonify({'error': 'Roku not connected'}), 503

                return jsonify({'error': 'Roku not initialized'}), 500
            except Exception as e:
                logger.error(f"Error sending Roku command: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/roku/launch/<app>', methods=['POST'])
        def api_roku_launch(app):
            """Launch an app on Roku."""
            try:
                if hasattr(self.minus, 'roku_controller') and self.minus.roku_controller:
                    if self.minus.roku_controller.is_connected():
                        if self.minus.roku_controller.launch_app(app):
                            return jsonify({'success': True, 'app': app})
                        return jsonify({'error': f'Failed to launch {app}'}), 500
                    return jsonify({'error': 'Roku not connected'}), 503
                return jsonify({'error': 'Roku not initialized'}), 500
            except Exception as e:
                logger.error(f"Error launching Roku app: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Current Vocabulary Word
        # =========================================================================

        @self.app.route('/api/vocabulary')
        def api_vocabulary():
            """Get current vocabulary word being displayed."""
            try:
                if self.minus.ad_blocker:
                    word_info = self.minus.ad_blocker.get_current_vocabulary()
                    return jsonify(word_info)
                return jsonify({'word': None, 'translation': None, 'example': None})
            except Exception as e:
                logger.error(f"Error getting vocabulary: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Screenshot Gallery
        # =========================================================================

        @self.app.route('/api/screenshots')
        def api_screenshots():
            """Get list of screenshots with pagination.

            Query params:
            - type: 'ads', 'non_ads', 'vlm_spastic', 'static' (default: 'ads')
            - page: page number starting from 1 (default: 1)
            - limit: items per page (default: 5, max: 20)
            """
            try:
                screenshot_type = request.args.get('type', 'ads')
                page = max(1, int(request.args.get('page', 1)))
                limit = min(20, max(1, int(request.args.get('limit', 5))))

                valid_types = ['ads', 'non_ads', 'vlm_spastic', 'static']
                if screenshot_type not in valid_types:
                    screenshot_type = 'ads'

                screenshots_dir = Path(__file__).parent.parent / 'screenshots' / screenshot_type

                if not screenshots_dir.exists():
                    return jsonify({
                        'screenshots': [],
                        'total': 0,
                        'page': page,
                        'pages': 0,
                        'has_more': False
                    })

                # Get all files sorted by modification time
                all_files = sorted(screenshots_dir.glob('*.png'), key=lambda x: x.stat().st_mtime, reverse=True)
                total = len(all_files)
                pages = (total + limit - 1) // limit  # Ceiling division

                # Paginate
                start = (page - 1) * limit
                end = start + limit
                files = all_files[start:end]

                screenshots = [{'name': f.name, 'path': f'/api/screenshots/{screenshot_type}/{f.name}'} for f in files]

                return jsonify({
                    'screenshots': screenshots,
                    'total': total,
                    'page': page,
                    'pages': pages,
                    'has_more': page < pages
                })
            except Exception as e:
                logger.error(f"Error listing screenshots: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/screenshots/<subdir>/<filename>')
        def api_screenshot_file(subdir, filename):
            """Serve a screenshot file."""
            try:
                valid_subdirs = ['ads', 'non_ads', 'vlm_spastic', 'static', 'debug']
                if subdir not in valid_subdirs:
                    return Response(status=404)
                # Sanitize filename
                if '..' in filename or '/' in filename:
                    return Response(status=400)
                screenshots_dir = Path(__file__).parent.parent / 'screenshots' / subdir
                return send_from_directory(screenshots_dir, filename)
            except Exception as e:
                logger.error(f"Error serving screenshot: {e}")
                return Response(status=404)

        # =========================================================================
        # Screenshot Review (Tinder-style classification)
        # =========================================================================

        @self.app.route('/api/screenshots/review/<category>')
        def api_screenshots_review(category):
            """Get unreviewed screenshots for swipe classification.

            Returns oldest-first, skipping already-reviewed items.
            """
            try:
                valid = {'ads', 'non_ads', 'vlm_spastic', 'static'}
                if category not in valid:
                    return jsonify({'error': 'Invalid category'}), 400

                screenshots_dir = Path(__file__).parent.parent / 'screenshots' / category
                if not screenshots_dir.exists():
                    return jsonify({'items': [], 'total': 0, 'unreviewed': 0, 'reviewed_count': 0})

                all_files = sorted(screenshots_dir.glob('*.png'), key=lambda x: x.stat().st_mtime)
                total = len(all_files)
                unreviewed = [f for f in all_files if f.name not in self._reviewed]

                # Return up to 200 items (lazy-load images on frontend)
                items = [{
                    'name': f.name,
                    'path': f'/api/screenshots/{category}/{f.name}',
                } for f in unreviewed[:200]]

                return jsonify({
                    'items': items,
                    'total': total,
                    'unreviewed': len(unreviewed),
                    'reviewed_count': total - len(unreviewed),
                })
            except Exception as e:
                logger.error(f"Error listing review screenshots: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/screenshots/classify', methods=['POST'])
        def api_screenshots_classify():
            """Reclassify a screenshot (move between categories)."""
            try:
                data = request.get_json() or {}
                filename = data.get('filename')
                source = data.get('source')
                target = data.get('target')

                valid_sources = {'ads', 'non_ads', 'vlm_spastic', 'static'}
                valid_targets = {'ads', 'non_ads'}
                if source not in valid_sources or target not in valid_targets:
                    return jsonify({'error': 'Invalid source or target'}), 400
                if not filename or '..' in filename or '/' in filename:
                    return jsonify({'error': 'Invalid filename'}), 400

                base = Path(__file__).parent.parent / 'screenshots'
                source_path = base / source / filename
                if not source_path.exists():
                    return jsonify({'error': 'File not found'}), 404

                target_path = base / target / filename
                # Handle filename collision
                if target_path.exists():
                    stem = target_path.stem
                    target_path = base / target / f"{stem}_reclassified.png"

                shutil.move(str(source_path), str(target_path))

                self._reviewed.add(target_path.name)
                self._save_reviewed()

                self._undo_stack.append({
                    'action': 'classify',
                    'original_name': filename,
                    'new_name': target_path.name,
                    'source': source,
                    'target': target,
                })

                return jsonify({'success': True, 'moved_to': target, 'new_name': target_path.name})
            except Exception as e:
                logger.error(f"Error classifying screenshot: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/screenshots/approve', methods=['POST'])
        def api_screenshots_approve():
            """Mark a screenshot as correctly classified (reviewed)."""
            try:
                data = request.get_json() or {}
                filename = data.get('filename')
                category = data.get('category')

                if not filename or not category:
                    return jsonify({'error': 'Missing filename or category'}), 400

                base = Path(__file__).parent.parent / 'screenshots'
                path = base / category / filename
                if not path.exists():
                    return jsonify({'error': 'File not found'}), 404

                self._reviewed.add(filename)
                self._save_reviewed()

                self._undo_stack.append({
                    'action': 'approve',
                    'original_name': filename,
                    'category': category,
                })

                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Error approving screenshot: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/screenshots/undo', methods=['POST'])
        def api_screenshots_undo():
            """Undo the last review action."""
            try:
                if not self._undo_stack:
                    return jsonify({'error': 'Nothing to undo'}), 400

                last = self._undo_stack.pop()
                base = Path(__file__).parent.parent / 'screenshots'

                if last['action'] == 'classify':
                    current = base / last['target'] / last['new_name']
                    original = base / last['source'] / last['original_name']
                    if current.exists():
                        shutil.move(str(current), str(original))
                    self._reviewed.discard(last['new_name'])
                    self._save_reviewed()
                elif last['action'] == 'approve':
                    self._reviewed.discard(last['original_name'])
                    self._save_reviewed()

                return jsonify({'success': True, 'undone': last})
            except Exception as e:
                logger.error(f"Error undoing action: {e}")
                return jsonify({'error': str(e)}), 500

        # =========================================================================
        # Debug Snapshot (Screenshot + Logs)
        # =========================================================================

        @self.app.route('/api/debug/snapshot', methods=['POST'])
        def api_debug_snapshot():
            """Take a debug snapshot (screenshot + last 100 log lines).

            Saves a screenshot and a companion log file with the same timestamp
            to the screenshots/debug/ directory.

            Returns the paths and log content for immediate display.
            """
            try:
                from datetime import datetime

                # Create debug screenshots directory
                debug_dir = Path(__file__).parent.parent / 'screenshots' / 'debug'
                debug_dir.mkdir(parents=True, exist_ok=True)

                # Generate timestamp for filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

                # Capture snapshot from ustreamer
                try:
                    url = f'http://localhost:{self.ustreamer_port}/snapshot'
                    req = requests.get(url, timeout=5)
                    if req.status_code == 200:
                        screenshot_filename = f'debug_{timestamp}.jpg'
                        screenshot_path = debug_dir / screenshot_filename
                        screenshot_path.write_bytes(req.content)
                        logger.info(f"[WebUI] Debug snapshot saved: {screenshot_filename}")
                    else:
                        screenshot_filename = None
                        logger.warning(f"[WebUI] Failed to capture snapshot: HTTP {req.status_code}")
                except Exception as e:
                    screenshot_filename = None
                    logger.warning(f"[WebUI] Failed to capture snapshot: {e}")

                # Read last 100 log lines
                log_lines = []
                log_file = Path('/tmp/minus.log')
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        log_lines = [line.rstrip() for line in f.readlines()[-100:]]

                # Save log lines to companion file
                log_filename = f'debug_{timestamp}.log'
                log_path = debug_dir / log_filename
                log_path.write_text('\n'.join(log_lines))

                # Get current status for context
                status = {}
                try:
                    status = self.minus.get_status_dict()
                except Exception as e:
                    logger.debug(f"[WebUI] Failed to get status for debug snapshot: {e}")

                # Save status to companion JSON file
                import json
                status_filename = f'debug_{timestamp}.json'
                status_path = debug_dir / status_filename
                status_path.write_text(json.dumps(status, indent=2, default=str))

                return jsonify({
                    'success': True,
                    'timestamp': timestamp,
                    'screenshot': f'/api/screenshots/debug/{screenshot_filename}' if screenshot_filename else None,
                    'log_file': f'/api/screenshots/debug/{log_filename}',
                    'status_file': f'/api/screenshots/debug/{status_filename}',
                    'log_lines': log_lines,
                    'status': status,
                    'message': f'Debug snapshot saved: {timestamp}'
                })
            except Exception as e:
                logger.error(f"Error taking debug snapshot: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug/snapshots')
        def api_debug_snapshots():
            """List saved debug snapshots."""
            try:
                debug_dir = Path(__file__).parent.parent / 'screenshots' / 'debug'

                if not debug_dir.exists():
                    return jsonify({'snapshots': []})

                # Find all debug snapshot files (by unique timestamps)
                timestamps = set()
                for f in debug_dir.glob('debug_*.jpg'):
                    # Extract timestamp from filename
                    ts = f.stem.replace('debug_', '')
                    timestamps.add(ts)
                for f in debug_dir.glob('debug_*.log'):
                    ts = f.stem.replace('debug_', '')
                    timestamps.add(ts)

                snapshots = []
                for ts in sorted(timestamps, reverse=True):
                    snapshot = {'timestamp': ts}
                    jpg_file = debug_dir / f'debug_{ts}.jpg'
                    log_file = debug_dir / f'debug_{ts}.log'
                    json_file = debug_dir / f'debug_{ts}.json'

                    if jpg_file.exists():
                        snapshot['screenshot'] = f'/api/screenshots/debug/debug_{ts}.jpg'
                    if log_file.exists():
                        snapshot['log_file'] = f'/api/screenshots/debug/debug_{ts}.log'
                    if json_file.exists():
                        snapshot['status_file'] = f'/api/screenshots/debug/debug_{ts}.json'

                    snapshots.append(snapshot)

                return jsonify({'snapshots': snapshots[:50]})  # Limit to 50 most recent
            except Exception as e:
                logger.error(f"Error listing debug snapshots: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug/snapshot/<timestamp>', methods=['DELETE'])
        def api_debug_snapshot_delete(timestamp):
            """Delete a debug snapshot and its associated files."""
            try:
                # Validate timestamp format (prevent path traversal)
                if not re.match(r'^[\d_]+$', timestamp):
                    return jsonify({'error': 'Invalid timestamp format'}), 400

                debug_dir = Path(__file__).parent.parent / 'screenshots' / 'debug'
                deleted = []

                # Delete all files with this timestamp
                for ext in ['jpg', 'log', 'json']:
                    file_path = debug_dir / f'debug_{timestamp}.{ext}'
                    if file_path.exists():
                        file_path.unlink()
                        deleted.append(f'debug_{timestamp}.{ext}')

                if deleted:
                    logger.info(f"[WebUI] Deleted debug snapshot: {timestamp} ({len(deleted)} files)")
                    return jsonify({'success': True, 'deleted': deleted})
                else:
                    return jsonify({'error': 'No files found for this timestamp'}), 404
            except Exception as e:
                logger.error(f"Error deleting debug snapshot: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # WiFi Management (nmcli)
        # =========================================================================

        @self.app.route('/api/wifi/connections')
        def api_wifi_connections():
            """Get saved WiFi connections with priorities."""
            try:
                wifi_mgr = get_wifi_manager()
                ap_name = wifi_mgr._ap_connection_name  # e.g. "Hotspot"
                result = subprocess.run(
                    ['nmcli', '-t', '-f', 'NAME,UUID,TYPE,DEVICE,AUTOCONNECT,AUTOCONNECT-PRIORITY', 'connection', 'show'],
                    capture_output=True, text=True, timeout=10
                )
                connections = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':')
                        # Filter for wifi connections (802-11-wireless)
                        if len(parts) >= 4 and parts[2] == '802-11-wireless':
                            name = parts[0]
                            # Hide the internal AP/hotspot profile — it's surfaced
                            # separately via the WiFi status card, and isn't a
                            # user-facing "saved network" they'd want to click.
                            if name == ap_name:
                                continue
                            connections.append({
                                'name': name,
                                'uuid': parts[1],
                                'type': 'wifi',
                                'device': parts[3] if parts[3] else None,
                                'autoconnect': parts[4] == 'yes' if len(parts) > 4 else True,
                                'priority': int(parts[5]) if len(parts) > 5 and parts[5] else 0,
                            })
                return jsonify({'connections': connections})
            except Exception as e:
                logger.error(f"Error getting WiFi connections: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/scan')
        def api_wifi_scan():
            """Scan for available WiFi networks.

            Returns ``ap_mode`` so clients can explain why live scan is
            degraded on a single-radio device, and ``saved`` on each network
            so the UI can mark known ones and skip the password prompt.
            """
            try:
                wifi_mgr = get_wifi_manager()
                # Refresh the cached _ap_mode_active flag so the response
                # reflects the true state, not a stale value left over from
                # the monitor thread or a prior API call.
                wifi_mgr.get_status()
                # If AP is active, single-radio hardware can only scan by
                # briefly dropping the AP. The user explicitly accepted this
                # tradeoff — a few seconds of hotspot dropout in exchange for
                # real scan results. If ?skip_bounce=1 is passed, fall back
                # to the cheap cached-scan behavior.
                skip_bounce = request.args.get('skip_bounce', '').lower() in ('1', 'true', 'yes')
                networks = wifi_mgr.scan_networks(
                    bounce_ap=wifi_mgr._ap_mode_active and not skip_bounce
                )
                return jsonify({
                    'ap_mode': wifi_mgr._ap_mode_active,
                    'networks': [
                        {
                            'ssid': n.ssid,
                            'signal': n.signal,
                            'security': n.security,
                            'in_use': n.in_use,
                            'saved': n.saved,
                        }
                        for n in networks
                    ]
                })
            except Exception as e:
                logger.error(f"Error scanning WiFi: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/connect-saved', methods=['POST'])
        def api_wifi_connect_saved():
            """Activate a previously saved WiFi connection by name."""
            try:
                data = request.get_json() or {}
                name = data.get('name')
                if not name:
                    return jsonify({'error': 'Connection name is required'}), 400

                wifi_mgr = get_wifi_manager()
                result = wifi_mgr.connect_saved(name)
                if result.get('success'):
                    logger.info(f"[WebUI] Activated saved WiFi: {name}")
                    return jsonify({'success': True, 'message': f'Connected to {name}', 'ssid': name})
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Activation failed'),
                    'error_code': result.get('error_code', 'unknown'),
                }), 400
            except Exception as e:
                logger.error(f"Error activating saved WiFi: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/connect', methods=['POST'])
        def api_wifi_connect():
            """Connect to a WiFi network."""
            try:
                data = request.get_json() or {}
                ssid = data.get('ssid')
                password = data.get('password', '')

                if not ssid:
                    return jsonify({'error': 'SSID is required'}), 400

                # Use WiFiManager for better error handling and AP mode management
                wifi_mgr = get_wifi_manager()
                result = wifi_mgr.connect_to_network(ssid, password)

                if result.get('success'):
                    logger.info(f"[WebUI] Connected to WiFi: {ssid}")
                    return jsonify({'success': True, 'message': f'Connected to {ssid}', 'ssid': ssid})
                else:
                    return jsonify({'success': False, 'error': result.get('error', 'Connection failed')}), 400
            except Exception as e:
                logger.error(f"Error connecting to WiFi: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/delete', methods=['POST'])
        def api_wifi_delete():
            """Delete a saved WiFi connection."""
            try:
                data = request.get_json() or {}
                name = data.get('name')

                if not name:
                    return jsonify({'error': 'Connection name is required'}), 400

                wifi_mgr = get_wifi_manager()
                result = wifi_mgr.forget_network(name)

                if result.get('success'):
                    logger.info(f"[WebUI] Deleted WiFi connection: {name}")
                    return jsonify({'success': True, 'message': f'Deleted {name}'})
                else:
                    return jsonify({'success': False, 'error': result.get('error', 'Delete failed')}), 500
            except Exception as e:
                logger.error(f"Error deleting WiFi connection: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/priority', methods=['POST'])
        def api_wifi_priority():
            """Update WiFi connection priority."""
            try:
                data = request.get_json() or {}
                name = data.get('name')
                priority = data.get('priority', 0)

                if not name:
                    return jsonify({'error': 'Connection name is required'}), 400

                result = subprocess.run(
                    ['nmcli', 'connection', 'modify', name, 'connection.autoconnect-priority', str(priority)],
                    capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    return jsonify({'success': True, 'message': f'Updated priority for {name}'})
                else:
                    return jsonify({'error': result.stderr or 'Update failed'}), 500
            except Exception as e:
                logger.error(f"Error updating WiFi priority: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/status')
        def api_wifi_status():
            """Get current WiFi status including AP mode."""
            try:
                wifi_mgr = get_wifi_manager()
                status = wifi_mgr.get_status()
                return jsonify({
                    'connected': status.connected,
                    'ssid': status.ssid,
                    'ip_address': status.ip_address,
                    'signal': status.signal,
                    'ap_mode_active': status.ap_mode_active,
                    'ap_clients': status.ap_clients
                })
            except Exception as e:
                logger.error(f"Error getting WiFi status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/disconnect', methods=['POST'])
        def api_wifi_disconnect():
            """Disconnect from current WiFi network.

            A manual disconnect would otherwise leave the box with no way to
            reach it until the monitor thread boots the hotspot after 30s.
            Immediately bring the Minus AP up so the user can always rejoin.
            """
            try:
                wifi_mgr = get_wifi_manager()
                result = wifi_mgr.disconnect_network()
                if result.get('success'):
                    # Give the radio a moment to finish disassociating before
                    # we try to reconfigure it as an AP.
                    time.sleep(1.5)
                    ap_result = wifi_mgr.start_ap_mode()
                    result['ap_started'] = ap_result.get('success', False)
                    if ap_result.get('success'):
                        result['ap_ssid'] = ap_result.get('ssid')
                        result['ap_password'] = ap_result.get('password')
                        result['ap_ip'] = ap_result.get('ip')
                    else:
                        result['ap_error'] = ap_result.get('error')
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error disconnecting WiFi: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/ap/start', methods=['POST'])
        def api_wifi_ap_start():
            """Start the Minus WiFi access point."""
            try:
                wifi_mgr = get_wifi_manager()
                result = wifi_mgr.start_ap_mode()
                if result.get('success'):
                    logger.info("[WebUI] AP mode started via API")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error starting AP mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/ap/stop', methods=['POST'])
        def api_wifi_ap_stop():
            """Stop the Minus WiFi access point."""
            try:
                wifi_mgr = get_wifi_manager()
                result = wifi_mgr.stop_ap_mode()
                if result.get('success'):
                    logger.info("[WebUI] AP mode stopped via API")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error stopping AP mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/wifi/ap/status')
        def api_wifi_ap_status():
            """Get AP mode status."""
            try:
                wifi_mgr = get_wifi_manager()
                status = wifi_mgr.get_ap_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting AP status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Captive Portal Detection Routes
        # These endpoints trigger the captive portal popup on mobile devices
        # =========================================================================

        @self.app.route('/wifi-setup')
        def wifi_setup():
            """Serve the WiFi setup/captive portal page."""
            return send_from_directory(
                self.app.template_folder,
                'wifi_setup.html'
            )

        @self.app.route('/generate_204')
        def captive_portal_android():
            """Android captive portal detection.
            Returns 204 if connected to internet, 302 redirect if in AP mode.
            """
            try:
                wifi_mgr = get_wifi_manager()
                if wifi_mgr._ap_mode_active:
                    return redirect('/wifi-setup')
                return '', 204
            except Exception:
                return '', 204

        @self.app.route('/hotspot-detect.html')
        def captive_portal_apple():
            """Apple captive portal detection.
            Returns 'Success' if connected, redirects if in AP mode.
            """
            try:
                wifi_mgr = get_wifi_manager()
                if wifi_mgr._ap_mode_active:
                    return redirect('/wifi-setup')
                return '<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>'
            except Exception:
                return '<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>'

        @self.app.route('/connecttest.txt')
        def captive_portal_windows():
            """Windows captive portal detection."""
            try:
                wifi_mgr = get_wifi_manager()
                if wifi_mgr._ap_mode_active:
                    return redirect('/wifi-setup')
                return 'Microsoft Connect Test'
            except Exception:
                return 'Microsoft Connect Test'

        @self.app.route('/ncsi.txt')
        def captive_portal_windows_ncsi():
            """Windows NCSI captive portal detection."""
            try:
                wifi_mgr = get_wifi_manager()
                if wifi_mgr._ap_mode_active:
                    return redirect('/wifi-setup')
                return 'Microsoft NCSI'
            except Exception:
                return 'Microsoft NCSI'

        @self.app.route('/success.txt')
        def captive_portal_firefox():
            """Firefox captive portal detection."""
            try:
                wifi_mgr = get_wifi_manager()
                if wifi_mgr._ap_mode_active:
                    return redirect('/wifi-setup')
                return 'success\n'
            except Exception:
                return 'success\n'

        # =========================================================================
        # ADB RSA Key Management
        # =========================================================================

        @self.app.route('/api/adb/keys')
        def api_adb_keys():
            """Get ADB RSA key info."""
            try:
                adbkey_path = Path.home() / '.android' / 'adbkey'
                adbkey_pub_path = Path.home() / '.android' / 'adbkey.pub'

                result = {'exists': False, 'public_key': None, 'fingerprint': None}

                if adbkey_pub_path.exists():
                    result['exists'] = True
                    pub_key = adbkey_pub_path.read_text().strip()
                    result['public_key'] = pub_key[:50] + '...' if len(pub_key) > 50 else pub_key

                    # Generate fingerprint (MD5 of public key)
                    import hashlib
                    fingerprint = hashlib.md5(pub_key.encode()).hexdigest()
                    result['fingerprint'] = ':'.join(fingerprint[i:i+2] for i in range(0, 32, 2))

                return jsonify(result)
            except Exception as e:
                logger.error(f"Error getting ADB keys: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/adb/keys/revoke', methods=['POST'])
        def api_adb_keys_revoke():
            """Revoke (delete) ADB RSA keys."""
            try:
                adbkey_path = Path.home() / '.android' / 'adbkey'
                adbkey_pub_path = Path.home() / '.android' / 'adbkey.pub'

                deleted = []
                if adbkey_path.exists():
                    adbkey_path.unlink()
                    deleted.append('adbkey')
                if adbkey_pub_path.exists():
                    adbkey_pub_path.unlink()
                    deleted.append('adbkey.pub')

                # Disconnect Fire TV if connected
                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    controller = self.minus.fire_tv_setup.get_controller()
                    if controller:
                        controller.disconnect()

                if deleted:
                    logger.info(f"[WebUI] Revoked ADB keys: {deleted}")
                    return jsonify({'success': True, 'deleted': deleted, 'message': 'ADB keys revoked. You will need to re-authorize on the TV.'})
                else:
                    return jsonify({'success': True, 'deleted': [], 'message': 'No keys found to revoke.'})
            except Exception as e:
                logger.error(f"Error revoking ADB keys: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Stats (Ads Blocked, Time Saved)
        # =========================================================================

        @self.app.route('/api/stats')
        def api_stats():
            """Get blocking statistics."""
            try:
                stats = {
                    'ads_blocked_today': 0,
                    'total_blocking_time': 0,
                    'time_saved': 0,
                    'blocking_start_time': None,
                    'current_blocking_duration': 0,
                }

                if self.minus.ad_blocker:
                    stats['ads_blocked_today'] = getattr(self.minus.ad_blocker, '_total_ads_blocked', 0)
                    stats['total_blocking_time'] = getattr(self.minus.ad_blocker, '_total_blocking_time', 0)
                    stats['time_saved'] = getattr(self.minus.ad_blocker, '_total_time_saved', 0)

                    if self.minus.ad_blocker.is_visible:
                        start_time = getattr(self.minus.ad_blocker, '_current_block_start', 0)
                        if start_time:
                            stats['blocking_start_time'] = start_time
                            stats['current_blocking_duration'] = time.time() - start_time

                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Audio Mute Status
        # =========================================================================

        @self.app.route('/api/audio/status')
        def api_audio_status():
            """Get audio mute status."""
            try:
                muted = False
                if hasattr(self.minus, 'audio') and self.minus.audio:
                    muted = getattr(self.minus.audio, 'is_muted', False)
                return jsonify({'muted': muted})
            except Exception as e:
                logger.error(f"Error getting audio status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/audio/check')
        def api_audio_check():
            """End-to-end audio health check (Home tab Check Audio button).

            Walks capture -> pipeline -> mute -> ALSA playback and returns
            a human-readable verdict plus the raw signals.
            """
            try:
                if hasattr(self.minus, 'audio') and self.minus.audio:
                    diag = self.minus.audio.get_diagnostics()
                    diag['success'] = True
                    return jsonify(diag)
                return jsonify({'success': False, 'error': 'Audio not initialized'}), 500
            except Exception as e:
                logger.error(f"Error checking audio: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/audio/restart', methods=['POST'])
        def api_audio_restart():
            """Force a full audio pipeline rebuild (Home tab Restart Audio button).

            Recovers the silent-but-healthy-looking state where the HDMI-TX
            audio lane died during a modeset (~1-2s audio dropout).
            """
            try:
                if hasattr(self.minus, 'audio') and self.minus.audio:
                    result = self.minus.audio.restart(reason="web UI")
                    logger.info(f"[WebUI] Audio restart: {result.get('message', result.get('error', 'unknown'))}")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Audio not initialized'}), 500
            except Exception as e:
                logger.error(f"Error restarting audio: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/audio/sync-reset', methods=['POST'])
        def api_audio_sync_reset():
            """Reset A/V sync by flushing the audio sync queue.

            Use this when audio and video are out of sync.
            Causes a brief audio dropout (~300ms) while the queue refills.
            """
            try:
                if hasattr(self.minus, 'audio') and self.minus.audio:
                    result = self.minus.audio.reset_av_sync()
                    logger.info(f"[WebUI] A/V sync reset: {result.get('message', 'unknown')}")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Audio not initialized'}), 500
            except Exception as e:
                logger.error(f"Error resetting A/V sync: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Webhooks
        # =========================================================================

        @self.app.route('/api/webhooks')
        def api_webhooks_get():
            """Get webhook configuration."""
            try:
                from src.webhooks import get_webhook_manager
                manager = get_webhook_manager()
                return jsonify({
                    'enabled': manager.enabled,
                    'urls': manager.get_urls()
                })
            except Exception as e:
                logger.error(f"Error getting webhooks: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/webhooks', methods=['POST'])
        def api_webhooks_set():
            """Configure webhooks.

            JSON body:
            - enabled: true/false to enable/disable webhooks
            - urls: list of webhook URLs
            - add_url: single URL to add
            - remove_url: single URL to remove
            """
            try:
                from src.webhooks import get_webhook_manager
                manager = get_webhook_manager()
                data = request.get_json() or {}

                if 'enabled' in data:
                    manager.set_enabled(bool(data['enabled']))

                if 'urls' in data:
                    with manager._lock:
                        manager.urls = list(data['urls'])

                if 'add_url' in data:
                    manager.add_url(data['add_url'])

                if 'remove_url' in data:
                    manager.remove_url(data['remove_url'])

                return jsonify({
                    'success': True,
                    'enabled': manager.enabled,
                    'urls': manager.get_urls()
                })
            except Exception as e:
                logger.error(f"Error setting webhooks: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/webhooks/test', methods=['POST'])
        def api_webhooks_test():
            """Send a test notification to all webhook URLs."""
            try:
                from src.webhooks import get_webhook_manager
                manager = get_webhook_manager()

                if not manager.urls:
                    return jsonify({'success': False, 'error': 'No webhook URLs configured'}), 400

                manager.notify('test', {'message': 'Test notification from Minus'})
                return jsonify({
                    'success': True,
                    'message': f'Test notification sent to {len(manager.urls)} URL(s)'
                })
            except Exception as e:
                logger.error(f"Error testing webhooks: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Health Check
        # =========================================================================

        @self.app.route('/api/health')
        def api_health():
            """Detailed health check endpoint for monitoring.

            Returns 200 OK if service is running, with detailed subsystem status.
            Use ?simple=1 for basic ok/error response suitable for uptime monitors.
            """
            try:
                simple_mode = request.args.get('simple', '0') == '1'

                health = {
                    'status': 'ok',
                    'service': 'minus',
                    'timestamp': time.time(),
                    'subsystems': {}
                }

                issues = []

                # Video subsystem
                video_status = {'status': 'not_initialized'}
                if hasattr(self.minus, 'ad_blocker') and self.minus.ad_blocker:
                    if self.minus.ad_blocker.pipeline:
                        fps = self.minus.ad_blocker.get_fps()
                        restart_count = getattr(self.minus.ad_blocker, '_restart_count', 0)
                        video_status = {
                            'status': 'ok',
                            'fps': float(fps) if fps is not None else 0.0,
                            'blocking': bool(self.minus.ad_blocker.is_visible),
                            'restart_count': int(restart_count) if isinstance(restart_count, (int, float)) else 0
                        }
                    else:
                        video_status = {'status': 'error', 'reason': 'no_pipeline'}
                        issues.append('video_pipeline_down')
                health['subsystems']['video'] = video_status

                # Audio subsystem
                audio_status = {'status': 'not_initialized'}
                if hasattr(self.minus, 'audio') and self.minus.audio:
                    if self.minus.audio.is_running:
                        restart_count = getattr(self.minus.audio, '_restart_count', 0)
                        audio_status = {
                            'status': 'ok',
                            'muted': bool(self.minus.audio.is_muted),
                            'restart_count': int(restart_count) if isinstance(restart_count, (int, float)) else 0
                        }
                    else:
                        audio_status = {'status': 'stopped'}
                        issues.append('audio_stopped')
                # Sink-side HDMI audio link: catches the "ALSA/GStreamer green
                # but projector silently wedged" state that a Minus restart
                # does not fix. Surfaces it so the user can see when to power-
                # cycle the TV.
                if hasattr(self.minus, 'health_monitor') and self.minus.health_monitor:
                    hm = self.minus.health_monitor
                    audio_status['hdmi_tx_link'] = {
                        'ok': bool(getattr(hm, '_hdmi_tx_audio_link_ok', True)),
                        'details': dict(getattr(hm, '_hdmi_tx_audio_link_details', {})),
                    }
                    if not audio_status['hdmi_tx_link']['ok']:
                        issues.append('hdmi_tx_audio_link_desync')
                health['subsystems']['audio'] = audio_status

                # VLM subsystem
                vlm_status = {'status': 'disabled'}
                if hasattr(self.minus, 'vlm') and self.minus.vlm:
                    # Note: is_ready is a property, not a method
                    if self.minus.vlm.is_ready:
                        vlm_status = {'status': 'ok'}
                    else:
                        vlm_status = {'status': 'loading'}
                    # Inference latency stats (for detecting NPU degraded state)
                    if hasattr(self.minus.vlm, 'get_latency_stats'):
                        try:
                            vlm_status['latency'] = self.minus.vlm.get_latency_stats()
                        except Exception:
                            pass
                    if hasattr(self.minus.vlm, '_restart_count'):
                        vlm_status['restart_count'] = self.minus.vlm._restart_count
                # Axera NPU telemetry (same card the VLM runs on)
                axera = get_axera_metrics()
                if axera:
                    vlm_status['axera'] = axera
                health['subsystems']['vlm'] = vlm_status

                # OCR subsystem
                ocr_status = {'status': 'disabled'}
                if hasattr(self.minus, 'ocr') and self.minus.ocr:
                    ocr_status = {'status': 'ok'}
                health['subsystems']['ocr'] = ocr_status

                # Fire TV subsystem
                firetv_status = {'status': 'disabled'}
                if hasattr(self.minus, 'fire_tv_controller') and self.minus.fire_tv_controller:
                    if self.minus.fire_tv_controller.is_connected():
                        firetv_status = {'status': 'connected'}
                    else:
                        firetv_status = {'status': 'disconnected'}
                health['subsystems']['fire_tv'] = firetv_status

                # Health monitor status
                if hasattr(self.minus, 'health_monitor') and self.minus.health_monitor:
                    hm_status = self.minus.health_monitor.get_status()
                    health['hdmi_signal'] = hm_status.hdmi_signal
                    health['hdmi_resolution'] = hm_status.hdmi_resolution
                    health['uptime_seconds'] = hm_status.uptime_seconds

                # File descriptor usage — we've had FD leaks cause 500 errors
                # once the soft limit was hit. Surface fd count + limit for
                # monitoring.
                try:
                    import resource
                    pid = os.getpid()
                    fd_count = len(os.listdir(f'/proc/{pid}/fd'))
                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    health['fds'] = {
                        'open': fd_count,
                        'soft_limit': soft,
                        'hard_limit': hard,
                        'usage_pct': round(100.0 * fd_count / soft, 1) if soft > 0 else 0.0,
                    }
                    if soft > 0 and fd_count > soft * 0.8:
                        issues.append('fd_limit_near_exhaustion')
                except Exception:
                    pass

                # Overall status
                if issues:
                    health['status'] = 'degraded'
                    health['issues'] = issues

                # Simple mode returns just status for uptime monitors
                if simple_mode:
                    return jsonify({
                        'status': health['status'],
                        'timestamp': health['timestamp']
                    })

                return jsonify(health)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({'status': 'error', 'error': str(e)}), 500

        @self.app.route('/api/metrics')
        def api_metrics():
            """Prometheus-compatible metrics endpoint.

            Returns metrics in Prometheus text format for monitoring systems.
            """
            try:
                lines = []

                # Helper to add metric
                def add_metric(name, value, help_text, metric_type='gauge', labels=None):
                    lines.append(f'# HELP {name} {help_text}')
                    lines.append(f'# TYPE {name} {metric_type}')
                    if labels:
                        label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
                        lines.append(f'{name}{{{label_str}}} {value}')
                    else:
                        lines.append(f'{name} {value}')

                # Uptime
                uptime = 0
                if hasattr(self.minus, 'health_monitor') and self.minus.health_monitor:
                    status = self.minus.health_monitor.get_status()
                    uptime = status.uptime_seconds
                add_metric('minus_uptime_seconds', uptime, 'Time since service start')

                # Video stats
                if hasattr(self.minus, 'ad_blocker') and self.minus.ad_blocker:
                    fps = self.minus.ad_blocker.get_fps() or 0
                    add_metric('minus_video_fps', fps, 'Current video FPS')

                    blocking = 1 if self.minus.ad_blocker.is_visible else 0
                    add_metric('minus_blocking_active', blocking, 'Whether blocking is active')

                    restart_count = getattr(self.minus.ad_blocker, '_restart_count', 0)
                    if isinstance(restart_count, (int, float)):
                        add_metric('minus_video_restarts_total', restart_count, 'Total video pipeline restarts', 'counter')

                # Audio stats
                if hasattr(self.minus, 'audio') and self.minus.audio:
                    running = 1 if self.minus.audio.is_running else 0
                    add_metric('minus_audio_running', running, 'Whether audio is running')

                    muted = 1 if self.minus.audio.is_muted else 0
                    add_metric('minus_audio_muted', muted, 'Whether audio is muted')

                    restart_count = getattr(self.minus.audio, '_restart_count', 0)
                    if isinstance(restart_count, (int, float)):
                        add_metric('minus_audio_restarts_total', restart_count, 'Total audio pipeline restarts', 'counter')

                # Detection stats
                if hasattr(self.minus, 'detection_counts'):
                    counts = self.minus.detection_counts
                    add_metric('minus_ocr_detections_total', counts.get('ocr', 0), 'Total OCR ad detections', 'counter')
                    add_metric('minus_vlm_detections_total', counts.get('vlm', 0), 'Total VLM ad detections', 'counter')

                # HDMI signal
                if hasattr(self.minus, 'health_monitor') and self.minus.health_monitor:
                    status = self.minus.health_monitor.get_status()
                    signal = 1 if status.hdmi_signal else 0
                    add_metric('minus_hdmi_signal', signal, 'Whether HDMI signal is present')

                # Time saved
                if hasattr(self.minus, 'ad_blocker') and self.minus.ad_blocker:
                    time_saved = getattr(self.minus.ad_blocker, '_total_time_saved', 0)
                    if isinstance(time_saved, (int, float)):
                        add_metric('minus_time_saved_seconds', time_saved, 'Total time saved by blocking ads', 'counter')

                # File descriptor usage
                try:
                    import resource
                    pid = os.getpid()
                    fd_count = len(os.listdir(f'/proc/{pid}/fd'))
                    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
                    add_metric('minus_open_fds', fd_count, 'Open file descriptors')
                    add_metric('minus_fd_soft_limit', soft, 'Soft limit on open FDs')
                except Exception:
                    pass

                # Axera NPU telemetry
                axera = get_axera_metrics()
                if axera:
                    if 'temperature_c' in axera:
                        add_metric('minus_axera_temperature_celsius', axera['temperature_c'],
                                   'Axera AX650N SoC temperature')
                    if 'npu_usage_pct' in axera:
                        add_metric('minus_axera_npu_usage_percent', axera['npu_usage_pct'],
                                   'Axera NPU utilization')
                    if 'cmm_used_kib' in axera:
                        add_metric('minus_axera_cmm_used_kib', axera['cmm_used_kib'],
                                   'Axera CMM memory in use')
                    if 'cmm_total_kib' in axera:
                        add_metric('minus_axera_cmm_total_kib', axera['cmm_total_kib'],
                                   'Axera CMM memory total')

                response = '\n'.join(lines) + '\n'
                return response, 200, {'Content-Type': 'text/plain; charset=utf-8'}
            except Exception as e:
                logger.error(f"Metrics error: {e}")
                return f'# Error generating metrics: {e}\n', 500, {'Content-Type': 'text/plain; charset=utf-8'}

        # =========================================================================
        # Video Pipeline Control
        # =========================================================================

        @self.app.route('/api/video/restart', methods=['POST'])
        def api_video_restart():
            """Force restart the video pipeline.

            Use this when video is glitching or frozen without restarting the whole service.
            """
            try:
                if hasattr(self.minus, 'ad_blocker') and self.minus.ad_blocker:
                    logger.info("[WebUI] Video pipeline restart requested")
                    # Restart the pipeline (runs in background thread)
                    self.minus.ad_blocker.restart()
                    return jsonify({'success': True, 'message': 'Video pipeline restart initiated'})
                return jsonify({'success': False, 'error': 'Ad blocker not initialized'}), 500
            except Exception as e:
                logger.error(f"Error restarting video pipeline: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/service/restart', methods=['POST'])
        def api_service_restart():
            """Restart the entire Minus service via systemd.

            This will cause ~15 seconds of downtime while the service restarts.
            The web UI will become unavailable during this time.
            """
            import subprocess
            import threading

            def do_restart():
                import time
                time.sleep(0.5)  # Give response time to send
                subprocess.run(['sudo', 'systemctl', 'restart', 'minus'],
                             capture_output=True, timeout=30)

            try:
                logger.info("[WebUI] Service restart requested")
                # Run restart in background thread so we can send response first
                threading.Thread(target=do_restart, daemon=True).start()
                return jsonify({'success': True, 'message': 'Service restart initiated'})
            except Exception as e:
                logger.error(f"Error initiating service restart: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Video Color Settings
        # =========================================================================

        @self.app.route('/api/video/color')
        def api_video_color_get():
            """Get current video color balance settings.

            Returns saturation, brightness, contrast, hue values.
            """
            try:
                if hasattr(self.minus, 'ad_blocker') and self.minus.ad_blocker:
                    settings = self.minus.ad_blocker.get_color_settings()
                    return jsonify(settings)
                return jsonify({'error': 'Ad blocker not initialized'}), 500
            except Exception as e:
                logger.error(f"Error getting color settings: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/video/color', methods=['POST'])
        def api_video_color_set():
            """Set video color balance settings.

            JSON body can include any of:
            - saturation: 0.0-2.0 (default 1.0, higher = more saturated)
            - brightness: -1.0 to 1.0 (default 0.0)
            - contrast: 0.0-2.0 (default 1.0)
            - hue: -1.0 to 1.0 (default 0.0)
            """
            try:
                if not hasattr(self.minus, 'ad_blocker') or not self.minus.ad_blocker:
                    return jsonify({'success': False, 'error': 'Ad blocker not initialized'}), 500

                data = request.get_json() or {}

                # Validate input ranges
                errors = []
                if 'saturation' in data:
                    if not isinstance(data['saturation'], (int, float)) or not (0.0 <= data['saturation'] <= 2.0):
                        errors.append('saturation must be a number between 0.0 and 2.0')
                if 'brightness' in data:
                    if not isinstance(data['brightness'], (int, float)) or not (-1.0 <= data['brightness'] <= 1.0):
                        errors.append('brightness must be a number between -1.0 and 1.0')
                if 'contrast' in data:
                    if not isinstance(data['contrast'], (int, float)) or not (0.0 <= data['contrast'] <= 2.0):
                        errors.append('contrast must be a number between 0.0 and 2.0')
                if 'hue' in data:
                    if not isinstance(data['hue'], (int, float)) or not (-1.0 <= data['hue'] <= 1.0):
                        errors.append('hue must be a number between -1.0 and 1.0')

                if errors:
                    return jsonify({'success': False, 'errors': errors}), 400

                result = self.minus.ad_blocker.set_color_settings(
                    saturation=data.get('saturation'),
                    brightness=data.get('brightness'),
                    contrast=data.get('contrast'),
                    hue=data.get('hue')
                )

                if result.get('success'):
                    return jsonify(result)
                else:
                    return jsonify(result), 500
            except Exception as e:
                logger.error(f"Error setting color: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # OCR/VLM Testing
        # =========================================================================

        @self.app.route('/api/ocr/test', methods=['POST'])
        def api_ocr_test():
            """Run OCR on current frame and return detected text.

            Does NOT save the screenshot (for testing only).
            Returns detected text and any ad keywords found.
            """
            try:
                if not hasattr(self.minus, 'ocr') or not self.minus.ocr:
                    return jsonify({'success': False, 'error': 'OCR not initialized'}), 500

                if not hasattr(self.minus, 'frame_capture') or not self.minus.frame_capture:
                    return jsonify({'success': False, 'error': 'Capture not initialized'}), 500

                # Capture snapshot
                import cv2
                start_time = time.time()
                frame = self.minus.frame_capture.capture()
                capture_time = time.time() - start_time

                if frame is None:
                    return jsonify({'success': False, 'error': 'Failed to capture frame'}), 500

                # Convert to RGB for OCR
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run OCR
                ocr_start = time.time()
                ocr_results = self.minus.ocr.ocr(frame_rgb)
                ocr_time = time.time() - ocr_start

                # Check for ad keywords
                ad_detected, matched_keywords, all_texts, is_terminal = self.minus.ocr.check_ad_keywords(ocr_results)

                return jsonify({
                    'success': True,
                    'is_ad': ad_detected,
                    'is_terminal': is_terminal,
                    'texts': all_texts[:20] if all_texts else [],  # Limit to 20 text items
                    'keywords': matched_keywords,
                    'capture_time_ms': round(capture_time * 1000),
                    'ocr_time_ms': round(ocr_time * 1000),
                })
            except Exception as e:
                logger.error(f"OCR test error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/vlm/test', methods=['POST'])
        def api_vlm_test():
            """Run VLM on current frame and return ad detection verdict.

            Does NOT save the screenshot (for testing only).
            Returns the VLM's verdict and confidence.
            """
            try:
                if not hasattr(self.minus, 'vlm') or not self.minus.vlm:
                    return jsonify({'success': False, 'error': 'VLM not initialized'}), 500

                if not self.minus.vlm.is_ready:
                    return jsonify({'success': False, 'error': 'VLM not ready (still loading)'}), 503

                if not hasattr(self.minus, 'frame_capture') or not self.minus.frame_capture:
                    return jsonify({'success': False, 'error': 'Capture not initialized'}), 500

                # Capture snapshot
                import cv2
                import tempfile
                start_time = time.time()
                frame = self.minus.frame_capture.capture()
                capture_time = time.time() - start_time

                if frame is None:
                    return jsonify({'success': False, 'error': 'Failed to capture frame'}), 500

                # Save to temp file for VLM (VLM requires file path)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, frame)

                try:
                    # Run VLM
                    vlm_start = time.time()
                    is_ad, raw_response, elapsed, confidence = self.minus.vlm.detect_ad(tmp_path)
                    vlm_time = time.time() - vlm_start

                    return jsonify({
                        'success': True,
                        'is_ad': is_ad,
                        'confidence': confidence,
                        'raw_response': raw_response[:200] if raw_response else None,  # Truncate
                        'capture_time_ms': round(capture_time * 1000),
                        'vlm_time_ms': round(vlm_time * 1000),
                    })
                finally:
                    # Clean up temp file
                    import os
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception as e:
                logger.error(f"VLM test error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # VLM Control (Enable/Disable)
        # =========================================================================

        @self.app.route('/api/vlm/status')
        def api_vlm_status():
            """Get detailed VLM status including model load state."""
            try:
                if hasattr(self.minus, 'get_vlm_status'):
                    status = self.minus.get_vlm_status()
                    return jsonify(status)
                return jsonify({
                    'initialized': False,
                    'disabled': True,
                    'model_loaded': False
                })
            except Exception as e:
                logger.error(f"Error getting VLM status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/vlm/disable', methods=['POST'])
        def api_vlm_disable():
            """Disable VLM and unload the model from the Axera NPU.

            This completely frees the NPU resources used by the VLM model.
            Detection will continue in OCR-only mode.
            """
            try:
                if hasattr(self.minus, 'disable_vlm'):
                    result = self.minus.disable_vlm()
                    if result.get('success'):
                        return jsonify(result)
                    return jsonify(result), 500
                return jsonify({'success': False, 'error': 'VLM control not available'}), 500
            except Exception as e:
                logger.error(f"Error disabling VLM: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/vlm/enable', methods=['POST'])
        def api_vlm_enable():
            """Enable VLM and load the model to the Axera NPU.

            This loads the minus-v0.1 model which takes ~9-11 seconds.
            An overlay notification will show loading progress.
            """
            try:
                if hasattr(self.minus, 'enable_vlm'):
                    result = self.minus.enable_vlm()
                    if result.get('success'):
                        return jsonify(result)
                    return jsonify(result), 500
                return jsonify({'success': False, 'error': 'VLM control not available'}), 500
            except Exception as e:
                logger.error(f"Error enabling VLM: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/asr/status', methods=['GET'])
        def api_asr_status():
            """Get ASR (faster-whisper) status block.

            Mirrors the `asr` sub-object in /api/status; provided as a
            dedicated route so the Settings toggle can load its state.
            """
            try:
                if getattr(self.minus, 'asr', None) is not None:
                    return jsonify(self.minus.asr.get_status())
                return jsonify({'available': False, 'enabled': False,
                                'running': False})
            except Exception as e:
                logger.error(f"Error getting ASR status: {e}")
                return jsonify({'available': False, 'enabled': False,
                                'running': False, 'error': str(e)}), 500

        @self.app.route('/api/asr/enable', methods=['POST'])
        def api_asr_enable():
            """Enable ASR (faster-whisper confirm/veto) and start the worker."""
            try:
                result = self.minus.set_asr_enabled(True)
                return jsonify(result), (200 if result.get('success') else 500)
            except Exception as e:
                logger.error(f"Error enabling ASR: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/asr/disable', methods=['POST'])
        def api_asr_disable():
            """Disable ASR and stop the worker (frees the whisper model)."""
            try:
                result = self.minus.set_asr_enabled(False)
                return jsonify(result), (200 if result.get('success') else 500)
            except Exception as e:
                logger.error(f"Error disabling ASR: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/asr/test', methods=['POST'])
        def api_asr_test():
            """Run a WAV through the live ASR worker end-to-end (transcribe →
            keyword markers → verdict) and fold it into the rolling state so
            the ASR-Live panel reflects it. Exercises ASR without live HDMI-RX
            audio (e.g. TV off → source sends digital silence).

            Body (all optional): {"wav": "<path>"} to transcribe a specific
            file, or {"sample": "ad"|"show"} to use a bundled corpus clip.
            Defaults to a bundled ad clip so the panel shows a real
            transcript + 'confirm' verdict.
            """
            try:
                if getattr(self.minus, 'asr', None) is None:
                    return jsonify({'success': False, 'error': 'ASR not available'}), 500
                data = request.get_json(silent=True) or {}
                wav = data.get('wav')
                if not wav:
                    sample = data.get('sample', 'ad')
                    base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        'tests', 'asr_corpus')
                    wav = os.path.join(base, 'ad_cta_strong.wav' if sample == 'ad'
                                       else 'show_dialog.wav')
                if not os.path.exists(wav):
                    return jsonify({'success': False, 'error': f'wav not found: {wav}'}), 400
                # The bundled corpus clips are ~13s; Moonshine's per-call time
                # scales with audio length, so a full clip exceeds the ASR
                # worker's soft timeout (tuned for the ~2s production window)
                # and returns 'timeout'. Trim to the first ~4s (where ad CTAs
                # sit) so the test reflects production latency and still hits
                # markers. Uploaded wavs are used as-is unless oversized.
                try:
                    import wave as _wave
                    max_s = float(data.get('max_seconds', 4.0))
                    w = _wave.open(wav, 'rb')
                    nframes = w.getnframes()
                    keep = min(nframes, int(max_s * w.getframerate()))
                    if keep < nframes:
                        frames = w.readframes(keep)
                        params = w.getparams()
                        w.close()
                        trimmed = '/tmp/minus_asr_test_clip.wav'
                        ow = _wave.open(trimmed, 'wb')
                        ow.setnchannels(params.nchannels)
                        ow.setsampwidth(params.sampwidth)
                        ow.setframerate(params.framerate)
                        ow.writeframes(frames)
                        ow.close()
                        wav = trimmed
                    else:
                        w.close()
                except Exception as e:
                    logger.debug(f"ASR test trim skipped: {e}")
                result = self.minus.asr.test_transcribe(wav)
                return jsonify(result), (200 if result.get('success') else 500)
            except Exception as e:
                logger.error(f"Error in ASR test: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # System Settings
        # =========================================================================

        @self.app.route('/api/settings', methods=['GET'])
        def api_settings_get():
            """Get system settings."""
            try:
                settings = self.minus.get_system_settings()
                return jsonify(settings)
            except Exception as e:
                logger.error(f"Error getting system settings: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/settings/vlm-preload', methods=['GET', 'POST'])
        def api_vlm_preload():
            """Get or set VLM preload preference.

            GET: Returns current vlm_preload setting
            POST: Set vlm_preload (body: {"enabled": true/false})

            When enabled (default), VLM loads at startup even without HDMI signal.
            When disabled, VLM waits for HDMI signal before loading.
            """
            try:
                if request.method == 'GET':
                    return jsonify({'vlm_preload': self.minus.vlm_preload})

                # POST
                data = request.get_json() or {}
                enabled = data.get('enabled', True)
                result = self.minus.set_vlm_preload(bool(enabled))
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error with VLM preload setting: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/settings/replacement-modes', methods=['GET', 'POST'])
        def api_replacement_modes():
            """Get or set which replacement-mode kinds are enabled.

            GET returns the list; POST takes ``{"modes": ["vocab","photos"]}``
            and persists it via Minus.set_replacement_modes. Photos-only is
            allowed; only a fully-empty selection forces vocab back on.
            """
            try:
                if request.method == 'GET':
                    return jsonify({
                        'replacement_modes': self.minus.get_replacement_modes()
                    })
                data = request.get_json() or {}
                modes = data.get('modes', [])
                if not isinstance(modes, list):
                    return jsonify({'success': False, 'error': 'modes must be a list'}), 400
                return jsonify(self.minus.set_replacement_modes(modes))
            except Exception as e:
                logger.error(f"Error with replacement-modes setting: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Photo library (for 'photos' replacement mode)
        # =========================================================================

        @self.app.route('/api/media/photos', methods=['GET', 'POST'])
        def api_media_photos():
            """List photos (GET) or upload a new one (POST multipart).

            POST accepts ``multipart/form-data`` with a single ``file`` field.
            Server re-encodes to JPEG, caps dimensions/size, and stores under
            ``~/.minus_media/photos/``. See src/photo_library.py.
            """
            from photo_library import get_photo_library
            lib = get_photo_library()
            try:
                if request.method == 'GET':
                    return jsonify({
                        'photos': lib.list_photos(),
                        'total_bytes': lib.total_bytes(),
                        'count': lib.count(),
                    })
                # POST upload
                if 'file' not in request.files:
                    return jsonify({'success': False, 'error': 'missing file field'}), 400
                f = request.files['file']
                data = f.read()
                if not data:
                    return jsonify({'success': False, 'error': 'empty upload'}), 400
                try:
                    meta = lib.add_photo(data, original_name=f.filename or 'photo')
                    return jsonify({'success': True, 'photo': meta})
                except ValueError as ve:
                    return jsonify({'success': False, 'error': str(ve)}), 400
            except Exception as e:
                logger.error(f"Error in photo library: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/media/photos/<photo_id>', methods=['GET', 'DELETE'])
        def api_media_photo_detail(photo_id):
            """Download (GET) or delete (DELETE) one photo by id."""
            from photo_library import get_photo_library
            lib = get_photo_library()
            try:
                if request.method == 'DELETE':
                    removed = lib.remove_photo(photo_id)
                    return jsonify({'success': removed})
                # GET — return JPEG bytes inline
                data = lib.get_photo_bytes(photo_id)
                if data is None:
                    return jsonify({'error': 'not found'}), 404
                from flask import Response
                return Response(data, mimetype='image/jpeg')
            except Exception as e:
                logger.error(f"Error in photo detail {photo_id}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/settings/optimization', methods=['GET', 'POST'])
        def api_optimization_settings():
            """Get or set the three optimization toggles.

            Keys: ``block_falloff``, ``hdmi_reconnect_grace``, ``greyscale_preview``.
            POST body: ``{"key": "block_falloff", "enabled": true}``.
            """
            try:
                if request.method == 'GET':
                    return jsonify({
                        'block_falloff': self.minus.block_falloff_enabled,
                        'hdmi_reconnect_grace': self.minus.hdmi_reconnect_grace_enabled,
                        'greyscale_preview': self.minus.greyscale_preview_enabled,
                    })

                data = request.get_json() or {}
                key = data.get('key')
                enabled = data.get('enabled')
                if key is None or enabled is None:
                    return jsonify({'success': False, 'error': 'key and enabled required'}), 400
                result = self.minus.set_optimization_setting(key, bool(enabled))
                if not result.get('success'):
                    return jsonify(result), 400

                # Greyscale needs to flow to ad_blocker immediately so the
                # current (and future) blocking uses the new setting.
                if key == 'greyscale_preview' and self.minus.ad_blocker:
                    try:
                        self.minus.ad_blocker.set_preview_grayscale(bool(enabled))
                    except Exception as e:
                        logger.warning(f"[WebUI] set_preview_grayscale failed: {e}")
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error with optimization setting: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # IR Transmitter (REI 8K HDMI switch)
        # =========================================================================

        @self.app.route('/api/ir/status')
        def api_ir_status():
            """Status of the IR transmitter feature.

            Returns:
                enabled: persisted user toggle (UI gate).
                available: pwm3 hardware visible (overlay loaded).
                initialized: PWM has been set up and is ready to send.
                codes: list of valid button names for /api/ir/command.
            """
            try:
                from ir_transmitter import CODES, IRTransmitter
                tx = self.minus.ir_transmitter
                return jsonify({
                    'enabled': bool(self.minus.ir_enabled),
                    'available': IRTransmitter.hardware_available(),
                    'initialized': bool(tx and tx.initialized),
                    'codes': list(CODES.keys()),
                })
            except Exception as e:
                logger.error(f"[WebUI] /api/ir/status failed: {e}")
                return jsonify({'enabled': False, 'available': False,
                                'initialized': False, 'codes': [],
                                'error': str(e)}), 500

        @self.app.route('/api/ir/enable', methods=['POST'])
        def api_ir_enable():
            try:
                from ir_transmitter import IRTransmitter
                if not IRTransmitter.hardware_available():
                    return jsonify({
                        'success': False,
                        'error': ('IR hardware not available — enable the '
                                  'rk3588-pwm3-m1 overlay and reboot.'),
                    }), 503
                result = self.minus.set_ir_enabled(True)
                # Initialize eagerly so the first button press doesn't pay
                # the export+polarity setup latency.
                if self.minus.ir_transmitter is not None:
                    try:
                        self.minus.ir_transmitter.initialize()
                    except Exception as e:
                        logger.warning(f"[WebUI] IR initialize failed: {e}")
                return jsonify(result)
            except Exception as e:
                logger.error(f"[WebUI] /api/ir/enable failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/ir/disable', methods=['POST'])
        def api_ir_disable():
            try:
                return jsonify(self.minus.set_ir_enabled(False))
            except Exception as e:
                logger.error(f"[WebUI] /api/ir/disable failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/ir/command', methods=['POST'])
        def api_ir_command():
            """Send an IR button. Body: {"button": "<name>"}.

            Refuses unless ir_enabled is true so the toggle is the single
            on/off gate the user controls.
            """
            try:
                from ir_transmitter import (
                    CODES, IRCooldownError, IRTransmitterError,
                )
                if not self.minus.ir_enabled:
                    return jsonify({
                        'success': False,
                        'error': 'IR transmitter disabled in settings',
                    }), 403
                tx = self.minus.ir_transmitter
                if tx is None:
                    return jsonify({
                        'success': False,
                        'error': 'IR transmitter not loaded',
                    }), 503
                data = request.get_json(silent=True) or {}
                button = data.get('button')
                if button not in CODES:
                    return jsonify({
                        'success': False,
                        'error': f"unknown button '{button}'",
                        'codes': list(CODES.keys()),
                    }), 400
                try:
                    tx.send(button)
                except IRCooldownError as e:
                    return jsonify({
                        'success': False,
                        'error': str(e),
                        'retry_after': round(e.remaining_s, 2),
                    }), 429
                except IRTransmitterError as e:
                    return jsonify({'success': False, 'error': str(e)}), 503
                return jsonify({'success': True, 'button': button})
            except Exception as e:
                logger.error(f"[WebUI] /api/ir/command failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Status LEDs (WS2812B strip on SPI0 MOSI)
        # =========================================================================

        @self.app.route('/api/leds/status')
        def api_leds_status():
            """Status of the status-LED strip.

            Returns:
                available: /dev/spidev0.0 visible (SPI overlay loaded).
                enabled:   persisted user toggle (UI gate).
                running:   animation thread alive.
                state:     current animation state name.
                states:    list of valid state names.
            """
            try:
                ctrl = self.minus.status_leds
                if ctrl is None:
                    return jsonify({
                        'available': False,
                        'enabled': False,
                        'running': False,
                        'state': 'off',
                        'states': [],
                        'error': 'status LED module not loaded',
                    }), 503
                return jsonify(ctrl.status())
            except Exception as e:
                logger.error(f"[WebUI] /api/leds/status failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/leds/enable', methods=['POST'])
        def api_leds_enable():
            try:
                from status_led_controller import StatusLEDController
                ctrl = self.minus.status_leds
                if ctrl is None:
                    return jsonify({
                        'success': False,
                        'error': 'status LED module not loaded',
                    }), 503
                if not StatusLEDController.hardware_available():
                    return jsonify({
                        'success': False,
                        'error': ('SPI hardware not available — enable the '
                                  'rk3588-spi0-m2-cs0-spidev overlay and reboot.'),
                    }), 503
                result = ctrl.set_enabled(True)
                # Pick up the right baseline (no_signal / paused / autonomous
                # / idle) instead of always starting on idle — matches what
                # the system would have shown if it hadn't been disabled.
                if hasattr(self.minus, '_baseline_led_state'):
                    ctrl.set_state(self.minus._baseline_led_state())
                else:
                    ctrl.set_state('idle')
                return jsonify(result)
            except Exception as e:
                logger.error(f"[WebUI] /api/leds/enable failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/leds/disable', methods=['POST'])
        def api_leds_disable():
            try:
                ctrl = self.minus.status_leds
                if ctrl is None:
                    return jsonify({
                        'success': False,
                        'error': 'status LED module not loaded',
                    }), 503
                return jsonify(ctrl.set_enabled(False))
            except Exception as e:
                logger.error(f"[WebUI] /api/leds/disable failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/leds/state', methods=['POST'])
        def api_leds_state():
            """Set the animation state. Body: {"state": "<name>"}.

            Refuses unless the strip is enabled so the toggle is the single
            on/off gate the user controls.
            """
            try:
                from status_led_controller import VALID_STATES
                ctrl = self.minus.status_leds
                if ctrl is None:
                    return jsonify({
                        'success': False,
                        'error': 'status LED module not loaded',
                    }), 503
                if not ctrl.enabled:
                    return jsonify({
                        'success': False,
                        'error': 'status LEDs disabled in settings',
                    }), 403
                data = request.get_json(silent=True) or {}
                state = data.get('state')
                if not ctrl.set_state(state):
                    return jsonify({
                        'success': False,
                        'error': f"unknown state {state!r}",
                        'states': list(VALID_STATES),
                    }), 400
                return jsonify({'success': True, 'state': state})
            except Exception as e:
                logger.error(f"[WebUI] /api/leds/state failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/leds/require_display', methods=['GET'])
        def api_leds_require_display_get():
            """Read the 'only drive LEDs while display is connected' toggle
            plus a live snapshot of the HDMI-TX state, so the UI can
            explain why the strip is currently dark."""
            try:
                return jsonify({
                    'leds_require_display':
                        bool(self.minus.leds_require_display),
                    'display_connected':
                        bool(self.minus.is_display_connected_live()),
                })
            except Exception as e:
                logger.error(
                    f"[WebUI] /api/leds/require_display GET failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/leds/require_display', methods=['POST'])
        def api_leds_require_display_post():
            try:
                data = request.get_json(silent=True) or {}
                enabled = bool(data.get('enabled', True))
                return jsonify(self.minus.set_leds_require_display(enabled))
            except Exception as e:
                logger.error(
                    f"[WebUI] /api/leds/require_display POST failed: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Blocking Control
        # =========================================================================

        @self.app.route('/api/blocking/skip', methods=['POST'])
        def api_blocking_skip():
            """Trigger skip button press on the connected streaming device.

            Uses the device-agnostic try_skip_ad() which works with
            Fire TV, Roku, and Google TV.
            """
            try:
                if not self.minus._is_remote_connected():
                    device_type = self.minus._get_configured_device_type()
                    return jsonify({'success': False, 'error': f'{device_type} not connected'}), 503

                result = self.minus.try_skip_ad()
                if not result:
                    return jsonify({'success': False, 'error': 'Skip command failed'}), 500

                device_type = self.minus._get_configured_device_type()
                logger.info(f"[WebUI] Skip ad command sent to {device_type}")

                # Force unblock after brief delay (don't wait for OCR to detect)
                def _unblock_after_skip():
                    import time as _time
                    _time.sleep(1.5)
                    logger.info("[WebUI] Forcing unblock after manual skip")
                    if self.minus.ad_blocker:
                        self.minus.ad_blocker.hide()
                    if self.minus.audio:
                        self.minus.audio.unmute()
                    self.minus.ocr_ad_detected = False
                    self.minus.vlm_ad_detected = False
                    self.minus.blocking_source = None
                import threading
                threading.Thread(target=_unblock_after_skip, daemon=True).start()

                return jsonify({'success': True, 'message': 'Skip command sent'})
            except Exception as e:
                logger.error(f"Error sending skip command: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Network Info
        # =========================================================================

        @self.app.route('/api/network')
        def api_network():
            """Get network information (IP addresses)."""
            try:
                result = subprocess.run(
                    ['ip', '-4', '-o', 'addr', 'show'],
                    capture_output=True, text=True, timeout=5
                )
                interfaces = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        if len(parts) >= 4:
                            iface = parts[1]
                            # Extract IP from "inet x.x.x.x/xx" format
                            ip_part = parts[3].split('/')[0]
                            if iface != 'lo':  # Skip loopback
                                interfaces.append({'interface': iface, 'ip': ip_part})

                # Get hostname
                hostname = subprocess.run(['hostname'], capture_output=True, text=True, timeout=5).stdout.strip()

                return jsonify({
                    'hostname': hostname,
                    'interfaces': interfaces
                })
            except Exception as e:
                logger.error(f"Error getting network info: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Clear Detections
        # =========================================================================

        @self.app.route('/api/detections/clear', methods=['POST'])
        def api_detections_clear():
            """Clear detection history."""
            try:
                if hasattr(self.minus, 'detection_history'):
                    self.minus.detection_history.clear()
                    logger.info("[WebUI] Detection history cleared")
                    return jsonify({'success': True, 'message': 'Detection history cleared'})
                return jsonify({'error': 'Detection history not available'}), 500
            except Exception as e:
                logger.error(f"Error clearing detections: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Fire TV App Launch
        # =========================================================================

        @self.app.route('/api/firetv/launch/<app>', methods=['POST'])
        def api_firetv_launch(app):
            """Launch an app on Fire TV."""
            # App package mappings
            apps = {
                'youtube': 'com.amazon.firetv.youtube',
                'netflix': 'com.netflix.ninja',
                'prime': 'com.amazon.avod',
                'hulu': 'com.hulu.plus',
                'disney': 'com.disney.disneyplus',
                'hbomax': 'com.hbo.hbonow',
                'peacock': 'com.peacocktv.peacockandroid',
                'plex': 'com.plexapp.android',
                'kodi': 'org.xbmc.kodi',
                'spotify': 'com.spotify.tv.android',
                'twitch': 'tv.twitch.android.app',
                'home': 'com.amazon.tv.launcher',
            }

            try:
                if app.lower() not in apps:
                    return jsonify({'error': f'Unknown app: {app}. Available: {list(apps.keys())}'}), 400

                package = apps[app.lower()]

                if hasattr(self.minus, 'fire_tv_setup') and self.minus.fire_tv_setup:
                    controller = self.minus.fire_tv_setup.get_controller()
                    if controller and controller.is_connected and hasattr(controller, '_device') and controller._device:
                        # Use monkey to launch the app
                        controller._device.adb_shell(f'monkey -p {package} -c android.intent.category.LAUNCHER 1')
                        logger.info(f"[WebUI] Launched {app} on Fire TV")
                        return jsonify({'success': True, 'app': app, 'package': package})
                    return jsonify({'error': 'Fire TV not connected'}), 503

                return jsonify({'error': 'Fire TV not initialized'}), 500
            except Exception as e:
                logger.error(f"Error launching app: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        # =========================================================================
        # Night Mode
        # =========================================================================

        @self.app.route('/api/autonomous')
        def api_autonomous_status():
            """Get night mode status."""
            try:
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    return jsonify(self.minus.autonomous_mode.get_status())
                return jsonify({
                    'enabled': False,
                    'active': False,
                    'error': 'Night mode not initialized'
                })
            except Exception as e:
                logger.error(f"Error getting night mode status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/enable', methods=['POST'])
        def api_autonomous_enable():
            """Enable night mode (scheduled 12am-8am ET)."""
            try:
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    result = self.minus.autonomous_mode.enable()
                    logger.info("[WebUI] Night mode enabled")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Night mode not initialized'}), 500
            except Exception as e:
                logger.error(f"Error enabling night mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/disable', methods=['POST'])
        def api_autonomous_disable():
            """Disable night mode."""
            try:
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    result = self.minus.autonomous_mode.disable()
                    logger.info("[WebUI] Night mode disabled")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Night mode not initialized'}), 500
            except Exception as e:
                logger.error(f"Error disabling night mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/toggle', methods=['POST'])
        def api_autonomous_toggle():
            """Toggle night mode on/off."""
            try:
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    result = self.minus.autonomous_mode.toggle()
                    logger.info(f"[WebUI] Night mode toggled: enabled={result.get('enabled')}")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Night mode not initialized'}), 500
            except Exception as e:
                logger.error(f"Error toggling night mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/music-mode', methods=['POST'])
        def api_autonomous_music_mode():
            """Toggle music mode - steer autonomous playback toward music
            videos (higher ad density = more training data per hour)."""
            try:
                data = request.get_json() or {}
                enabled = bool(data.get('enabled', False))
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    result = self.minus.autonomous_mode.set_music_mode(enabled)
                    logger.info(f"[WebUI] Autonomous music mode: {enabled}")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Autonomous mode not initialized'}), 500
            except Exception as e:
                logger.error(f"Error setting music mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/start', methods=['POST'])
        def api_autonomous_start():
            """Start night mode immediately (manual override)."""
            try:
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    result = self.minus.autonomous_mode.start_now()
                    logger.info("[WebUI] Night mode started immediately (manual)")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Night mode not initialized'}), 500
            except Exception as e:
                logger.error(f"Error starting night mode: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/logs')
        def api_autonomous_logs():
            """Get autonomous mode logs."""
            try:
                lines = int(request.args.get('lines', 50))
                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    log_content = self.minus.autonomous_mode.get_log_tail(lines)
                    return jsonify({'logs': log_content})
                return jsonify({'logs': 'Autonomous mode not initialized'})
            except Exception as e:
                logger.error(f"Error getting autonomous mode logs: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/autonomous/schedule', methods=['POST'])
        def api_autonomous_schedule():
            """Set autonomous mode schedule."""
            try:
                data = request.get_json() or {}
                start_hour = int(data.get('start_hour', 0))
                end_hour = int(data.get('end_hour', 8))
                always_on = bool(data.get('always_on', False))

                if hasattr(self.minus, 'autonomous_mode') and self.minus.autonomous_mode:
                    result = self.minus.autonomous_mode.set_schedule(start_hour, end_hour, always_on)
                    logger.info(f"[WebUI] Autonomous mode schedule set: {start_hour}:00-{end_hour}:00, always_on={always_on}")
                    return jsonify(result)
                return jsonify({'success': False, 'error': 'Autonomous mode not initialized'}), 500
            except Exception as e:
                logger.error(f"Error setting autonomous mode schedule: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

    def _load_reviewed(self):
        """Load reviewed screenshots set from disk."""
        try:
            if REVIEWED_FILE.exists():
                with open(REVIEWED_FILE) as f:
                    self._reviewed = set(json.load(f))
                logger.info(f"[WebUI] Loaded {len(self._reviewed)} reviewed screenshots")
        except Exception as e:
            logger.warning(f"[WebUI] Could not load reviewed screenshots: {e}")
            self._reviewed = set()

    def _save_reviewed(self):
        """Save reviewed screenshots set to disk (atomic, thread-safe).

        Holds a lock across the full snapshot+write+rename so concurrent
        review requests can't race on the JSON file. Atomic rename via a
        tmp file prevents readers from ever seeing a partial write.
        """
        try:
            with self._reviewed_lock:
                tmp = REVIEWED_FILE.with_suffix('.tmp')
                with open(tmp, 'w') as f:
                    json.dump(sorted(self._reviewed), f)
                tmp.replace(REVIEWED_FILE)
        except Exception as e:
            logger.warning(f"[WebUI] Could not save reviewed screenshots: {e}")

    def start(self):
        """Start the web server in a background thread."""
        if self.running:
            return

        self.running = True

        def run_server():
            logger.info(f"[WebUI] Starting on http://0.0.0.0:{self.port}")
            try:
                # Use threaded=True for concurrent requests
                self.app.run(
                    host='0.0.0.0',
                    port=self.port,
                    threaded=True,
                    use_reloader=False,
                    debug=False,
                )
            except Exception as e:
                logger.error(f"[WebUI] Server error: {e}")
            finally:
                self.running = False

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Give it a moment to start
        time.sleep(0.5)
        logger.info(f"[WebUI] Server started on port {self.port}")

    def stop(self):
        """Stop the web server."""
        self.running = False
        logger.info("[WebUI] Server stopping...")
        # Flask doesn't have a clean shutdown in this mode,
        # but since it's a daemon thread, it will stop when the process exits
