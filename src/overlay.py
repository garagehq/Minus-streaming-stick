"""
Notification Overlay Module for Minus.

Provides small corner overlays for notifications and guidance
without blocking the main video content.

Uses ustreamer's overlay API to render text directly on the
video stream via the MPP hardware encoder - no GStreamer
pipeline modifications needed.

Overlay Priority System:
- Long-duration overlays (setup instructions) are "persistent"
- Short overlays (status updates) can temporarily interrupt
- After short overlay finishes, persistent overlay is restored
- State changes (connection success) clear pending setup overlays
"""

import logging
import threading
import time
import urllib.request
import urllib.parse
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class OverlayManager:
    """
    Singleton manager for overlay priority and restoration.

    Tracks persistent overlays and restores them after interruptions.
    Uses module-level state to ensure true singleton behavior across imports.
    """
    pass  # State is stored at module level below


# Module-level state for true singleton behavior
_overlay_state = {
    'persistent_overlay': None,  # Dict with text, params, api_base
    'persistent_expiry': 0.0,    # Unix timestamp
    'restore_timer': None,       # threading.Timer
    'lock': threading.Lock(),
    'initialized': False,
}


def _init_overlay_manager():
    """Initialize the overlay manager state (called once at module load)."""
    global _overlay_state
    if not _overlay_state['initialized']:
        _overlay_state['initialized'] = True
        logger.info("[OverlayManager] Initialized (module-level singleton)")


# Initialize at module load time
_init_overlay_manager()

def _set_persistent(text: str, params: dict, duration: float, api_base: str):
    """Register a persistent overlay that should be restored if interrupted."""
    global _overlay_state
    _init_overlay_manager()
    with _overlay_state['lock']:
        # Cancel any existing monitor
        if _overlay_state.get('monitor_thread') and _overlay_state['monitor_thread'].is_alive():
            _overlay_state['stop_monitor'] = True

        _overlay_state['persistent_overlay'] = {
            'text': text,
            'params': params.copy(),
            'api_base': api_base,
        }
        _overlay_state['persistent_expiry'] = time.time() + duration
        _overlay_state['stop_monitor'] = False
        logger.info(f"[OverlayManager] Registered persistent overlay: {text[:40]}... (expires in {duration}s)")

    # Start a monitoring thread that will restore the overlay if overwritten
    def monitor_and_restore():
        check_interval = 5.0  # Check every 5 seconds
        while True:
            time.sleep(check_interval)

            with _overlay_state['lock']:
                if _overlay_state.get('stop_monitor'):
                    logger.info("[OverlayManager] Monitor stopped")
                    return

                if not _overlay_state['persistent_overlay']:
                    return

                if time.time() >= _overlay_state['persistent_expiry']:
                    logger.info("[OverlayManager] Persistent overlay expired")
                    _overlay_state['persistent_overlay'] = None
                    return

                # Check if overlay is still showing our text
                try:
                    params = _overlay_state['persistent_overlay']['params']
                    api_base = _overlay_state['persistent_overlay']['api_base']

                    # Query current overlay state
                    base_url = api_base.replace('/overlay/set', '/overlay')
                    req = urllib.request.Request(base_url)
                    with urllib.request.urlopen(req, timeout=2.0) as response:
                        import json
                        data = json.loads(response.read().decode())
                        result = data.get('result', {})
                        current_text = result.get('text', '')
                        enabled = result.get('enabled', False)

                        # Check if our overlay is still showing
                        expected_text = params.get('text', '')
                        if not enabled or current_text != expected_text:
                            # Our overlay was overwritten, restore it
                            remaining = int(_overlay_state['persistent_expiry'] - time.time())
                            logger.info(f"[OverlayManager] Restoring overwritten overlay ({remaining}s remaining)")
                            query = urllib.parse.urlencode(params)
                            url = f"{api_base}?{query}"
                            req2 = urllib.request.Request(url)
                            with urllib.request.urlopen(req2, timeout=2.0):
                                pass

                except Exception as e:
                    logger.debug(f"[OverlayManager] Monitor check error: {e}")

    thread = threading.Thread(target=monitor_and_restore, daemon=True, name="OverlayMonitor")
    thread.start()
    _overlay_state['monitor_thread'] = thread


def _clear_persistent():
    """Clear persistent overlay (e.g., on state change like successful connection)."""
    global _overlay_state
    with _overlay_state['lock']:
        if _overlay_state['persistent_overlay']:
            logger.info("[OverlayManager] Cleared persistent overlay")
        _overlay_state['persistent_overlay'] = None
        _overlay_state['persistent_expiry'] = 0
        _overlay_state['stop_monitor'] = True  # Stop the monitor thread
        if _overlay_state.get('restore_timer'):
            _overlay_state['restore_timer'].cancel()
            _overlay_state['restore_timer'] = None




def _is_persistent_active() -> bool:
    """Check if a persistent overlay is currently registered and not expired."""
    global _overlay_state
    with _overlay_state['lock']:
        active = _overlay_state['persistent_overlay'] is not None and time.time() < _overlay_state['persistent_expiry']
        return active


class NotificationOverlay:
    """
    Manages small corner notification overlays via ustreamer API.

    Shows guidance text in a corner of the screen without blocking
    the main video content. Uses ustreamer's /overlay/set HTTP API
    to render text directly in the MPP encoder.

    Position: Top-right corner by default
    """

    # Overlay positions (matching ustreamer API)
    POSITION_TOP_LEFT = 0
    POSITION_TOP_RIGHT = 1
    POSITION_BOTTOM_LEFT = 2
    POSITION_BOTTOM_RIGHT = 3
    POSITION_CENTER = 4

    # String position names for compatibility
    POS_TOP_RIGHT = 'top-right'
    POS_TOP_LEFT = 'top-left'
    POS_BOTTOM_RIGHT = 'bottom-right'
    POS_BOTTOM_LEFT = 'bottom-left'
    POS_CENTER = 'center'

    # Map string positions to API values
    _POS_MAP = {
        'top-left': 0,
        'top-right': 1,
        'bottom-left': 2,
        'bottom-right': 3,
        'center': 4,
    }

    def __init__(self, ustreamer_port: int = 9090, position: str = POS_TOP_RIGHT):
        """
        Initialize notification overlay.

        Args:
            ustreamer_port: Port where ustreamer is running (default: 9090)
            position: Corner position for overlay
        """
        self.ustreamer_port = ustreamer_port
        self.position = position
        self._api_position = self._POS_MAP.get(position, 1)  # Default top-right

        # Overlay state
        self._visible = False
        self._current_text = None
        self._lock = threading.Lock()

        # Auto-hide timer
        self._auto_hide_timer: Optional[threading.Timer] = None
        self._default_duration = 10.0  # Default auto-hide after 10s

        # Default styling (scale increased 20% from 3 to 4)
        self._scale = 4  # Text scale factor
        self._bg_alpha = 200  # Background transparency

        # API endpoint
        self._api_base = f"http://localhost:{ustreamer_port}/overlay/set"

        logger.info(f"[Overlay] Initialized with ustreamer API at port {ustreamer_port}, position={position}")

    def _call_api(self, params: dict) -> bool:
        """
        Call the ustreamer overlay API.

        Args:
            params: Query parameters for the API

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build query string
            query = urllib.parse.urlencode(params)
            url = f"{self._api_base}?{query}"

            # Make request with short timeout
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2.0) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(f"[Overlay] API returned status {response.status}")
                    return False

        except urllib.error.URLError as e:
            logger.error(f"[Overlay] API connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"[Overlay] API error: {e}")
            return False

    def show(self, text: str, duration: float = None, background: bool = True, persistent: bool = None):
        """
        Show notification overlay with text.

        Args:
            text: Text to display (supports newlines)
            duration: Auto-hide after this many seconds (None = stay visible)
            background: Show semi-transparent background behind text
            persistent: If True, this overlay will be restored after short interruptions.
                        If None, auto-detect based on duration (>60s = persistent)
        """
        with self._lock:
            # Cancel any pending auto-hide
            if self._auto_hide_timer:
                self._auto_hide_timer.cancel()
                self._auto_hide_timer = None

            self._current_text = text
            self._visible = True

            # Call ustreamer API
            params = {
                'text': text,
                'position': self._api_position,
                'scale': self._scale,
                'enabled': 'true',
            }

            if background:
                params['bg_enabled'] = 'true'
                params['bg_alpha'] = self._bg_alpha
            else:
                params['bg_enabled'] = 'false'

            # Determine if this is a persistent overlay (duration > 60 seconds)
            is_persistent = persistent if persistent is not None else (duration is not None and duration > 60)

            # Check if we're interrupting a persistent overlay with a short one
            # (The persistent overlay will be restored by its monitor thread)

            success = self._call_api(params)

            # Register as persistent if it's a long-duration overlay
            if is_persistent and duration is not None:
                _set_persistent(text, params, duration, self._api_base)

            # Set auto-hide timer if duration specified
            if duration is not None and duration > 0:
                self._auto_hide_timer = threading.Timer(duration, self.hide)
                self._auto_hide_timer.daemon = True
                self._auto_hide_timer.start()

            logger.info(f"[Overlay] Show {'OK' if success else 'FAILED'}: {text[:60]}...")

    def hide(self):
        """Hide the notification overlay."""
        with self._lock:
            # Cancel any pending auto-hide
            if self._auto_hide_timer:
                self._auto_hide_timer.cancel()
                self._auto_hide_timer = None

            self._visible = False
            self._current_text = None

            # Clear the overlay via API
            self._call_api({'clear': 'true', 'enabled': 'false'})

            logger.debug("[Overlay] Hidden")

    def update(self, text: str):
        """Update overlay text without resetting auto-hide timer."""
        with self._lock:
            if not self._visible:
                return

            self._current_text = text

            # Update via API
            params = {
                'text': text,
                'position': self._api_position,
                'scale': self._scale,
                'enabled': 'true',
                'bg_enabled': 'true',
                'bg_alpha': self._bg_alpha,
            }
            self._call_api(params)

    def set_position(self, position: str):
        """
        Change overlay position.

        Args:
            position: One of 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'
        """
        self.position = position
        self._api_position = self._POS_MAP.get(position, 1)

        # Update position if currently visible
        if self._visible and self._current_text:
            self.show(self._current_text)

    def set_scale(self, scale: int):
        """Set text scale factor (1-10)."""
        self._scale = max(1, min(10, scale))

    def set_background_alpha(self, alpha: int):
        """Set background transparency (0-255, 255=opaque)."""
        self._bg_alpha = max(0, min(255, alpha))

    @property
    def is_visible(self) -> bool:
        """Check if overlay is currently visible."""
        return self._visible

    @property
    def current_text(self) -> Optional[str]:
        """Get current overlay text."""
        return self._current_text

    def destroy(self):
        """Clean up resources."""
        with self._lock:
            if self._auto_hide_timer:
                self._auto_hide_timer.cancel()
                self._auto_hide_timer = None
            self._visible = False
            self._current_text = None

        # Clear overlay on destroy
        try:
            self._call_api({'clear': 'true', 'enabled': 'false'})
        except:
            pass


class FireTVNotification(NotificationOverlay):
    """
    Specialized notification overlay for Fire TV setup guidance.

    Shows compact setup instructions in the top-right corner
    while the user navigates their Fire TV to enable ADB or
    authorize the connection.
    """

    def __init__(self, ustreamer_port: int = 9090):
        super().__init__(ustreamer_port=ustreamer_port, position=NotificationOverlay.POS_TOP_RIGHT)
        # Scale increased 20% from 2 to 3
        self._scale = 3

    def show_scanning(self):
        """Show 'Scanning for Fire TV...' notification."""
        text = "Scanning for Fire TV..."
        self.show(text, duration=None)

    def show_adb_enable_instructions(self, timeout_remaining: int = None):
        """Show instructions for enabling ADB debugging."""
        lines = [
            "Fire TV Setup",
            "",
            "Enable ADB Debugging:",
            "1. Settings > My Fire TV",
            "2. Developer Options",
            "3. Turn ON ADB Debugging",
        ]

        if timeout_remaining is not None:
            lines.append("")
            lines.append(f"Scanning... ({timeout_remaining}s)")

        text = "\n".join(lines)
        self.show(text, duration=None)

    def show_auth_instructions(self, ip_address: str = None, attempt: int = None, timeout_remaining: int = None):
        """Show instructions for authorizing ADB connection."""
        lines = [
            "Fire TV Found!",
            "",
            "On your TV, press Allow",
            "on the USB Debugging dialog.",
            "",
            "Check: Always allow",
        ]

        if ip_address:
            lines.insert(1, f"IP: {ip_address}")

        if attempt is not None and timeout_remaining is not None:
            lines.append("")
            lines.append(f"Attempt {attempt}... ({timeout_remaining}s)")

        text = "\n".join(lines)
        self.show(text, duration=None)

    def show_connected(self, device_name: str = "Fire TV"):
        """Show connection success notification. Clears any pending setup overlays."""
        # Clear persistent setup overlays - connection success is a state change
        _clear_persistent()
        text = f"{device_name} Connected!\n\nAd skipping enabled."
        self.show(text, duration=5.0)  # Auto-hide after 5s

    def show_failed(self, reason: str = "Connection failed"):
        """Show connection failure notification."""
        text = f"{reason}\n\nFire TV skipping disabled."
        self.show(text, duration=10.0)  # Auto-hide after 10s

    def show_skipped(self):
        """Show setup skipped notification."""
        text = "Fire TV setup skipped.\n\nManual skip unavailable."
        self.show(text, duration=5.0)


class GoogleTVNotification(NotificationOverlay):
    """
    Specialized notification overlay for Google TV / Android TV setup guidance.

    Shows compact setup instructions in the top-right corner
    while the user navigates their Google TV to enable ADB or
    authorize the connection.
    """

    def __init__(self, ustreamer_port: int = 9090):
        super().__init__(ustreamer_port=ustreamer_port, position=NotificationOverlay.POS_TOP_RIGHT)
        self._scale = 3

    def show_scanning(self):
        """Show 'Scanning for Google TV...' notification."""
        text = "Scanning for Google TV..."
        self.show(text, duration=None)

    def show_adb_enable_instructions(self, timeout_remaining: int = None):
        """Show instructions for enabling Wireless debugging."""
        lines = [
            "Google TV Setup",
            "",
            "Enable Wireless Debugging:",
            "1. Settings > System > About",
            "2. Click Build 7x for Dev Mode",
            "3. System > Developer options",
            "4. Turn ON Wireless debugging",
            "5. Enter IP:Port shown on TV",
            "   in the web UI Remote tab",
        ]

        if timeout_remaining is not None:
            lines.append("")
            lines.append(f"Waiting... ({timeout_remaining}s)")

        text = "\n".join(lines)
        self.show(text, duration=None)

    def show_auth_instructions(self, ip_address: str = None, attempt: int = None, timeout_remaining: int = None):
        """Show instructions for authorizing ADB connection."""
        lines = [
            "Connecting to Google TV...",
            "",
            "On your TV, press Allow",
            "on the debugging dialog.",
            "",
            "Check: Always allow",
        ]

        if ip_address:
            lines.insert(1, f"Address: {ip_address}")

        if attempt is not None and timeout_remaining is not None:
            lines.append("")
            lines.append(f"Attempt {attempt}... ({timeout_remaining}s)")

        text = "\n".join(lines)
        self.show(text, duration=None)

    def show_connected(self, device_name: str = "Google TV"):
        """Show connection success notification. Clears any pending setup overlays."""
        _clear_persistent()
        text = f"{device_name} Connected!\n\nRemote control enabled."
        self.show(text, duration=5.0)

    def show_failed(self, reason: str = "Connection failed"):
        """Show connection failure notification."""
        text = f"{reason}\n\nGoogle TV control disabled."
        self.show(text, duration=10.0)

    def show_skipped(self):
        """Show setup skipped notification."""
        text = "Google TV setup skipped.\n\nRemote control unavailable."
        self.show(text, duration=5.0)


class RokuNotification(NotificationOverlay):
    """
    Specialized notification overlay for Roku setup guidance.

    Shows compact setup instructions in the top-right corner
    for Roku configuration and connection.
    """

    def __init__(self, ustreamer_port: int = 9090):
        super().__init__(ustreamer_port=ustreamer_port, position=NotificationOverlay.POS_TOP_RIGHT)
        self._scale = 3

    def show_scanning(self):
        """Show 'Scanning for Roku...' notification."""
        text = "Scanning for Roku..."
        self.show(text, duration=None)

    def show_setup_instructions(self):
        """Show instructions for enabling Roku ECP control."""
        lines = [
            "Roku Setup",
            "",
            "Enable Device Connect:",
            "1. Settings > System",
            "2. Advanced system settings",
            "3. Control by mobile apps",
            "4. Set to Default or Permissive",
        ]
        text = "\n".join(lines)
        self.show(text, duration=None)

    def show_connecting(self, ip_address: str = None):
        """Show connecting notification."""
        lines = ["Connecting to Roku..."]
        if ip_address:
            lines.append(f"IP: {ip_address}")
        text = "\n".join(lines)
        self.show(text, duration=None)

    def show_connected(self, device_name: str = "Roku"):
        """Show connection success notification. Clears any pending setup overlays."""
        # Clear persistent setup overlays - connection success is a state change
        _clear_persistent()
        text = f"{device_name} Connected!\n\nRemote control enabled."
        self.show(text, duration=5.0)

    def show_limited_mode(self):
        """Show notification when Roku is in Limited mode."""
        logger.info("[RokuNotification] Showing limited mode setup instructions")
        lines = [
            "ROKU SETUP REQUIRED",
            "",
            "Remote is in Limited mode.",
            "To enable full control:",
            "",
            "1. Press Home on Roku remote",
            "2. Settings",
            "3. System",
            "4. Advanced system settings",
            "5. Control by mobile apps",
            "6. Set to Default or Permissive",
        ]
        text = "\n".join(lines)
        # Keep visible for 300s (5 min) - user needs time to navigate Roku menus
        self.show(text, duration=300.0)

    def show_failed(self, reason: str = "Connection failed"):
        """Show connection failure notification."""
        text = f"Roku: {reason}\n\nCheck network settings."
        self.show(text, duration=10.0)

    def show_not_configured(self):
        """Show notification when no Roku is configured."""
        lines = [
            "Roku Not Configured",
            "",
            "Use the web UI Remote tab",
            "to set up your Roku.",
        ]
        text = "\n".join(lines)
        self.show(text, duration=10.0)


class SystemNotification(NotificationOverlay):
    """
    System notification overlay for status updates.

    Shows system status like VLM initialization progress
    in the top-right corner without blocking video content.
    """

    def __init__(self, ustreamer_port: int = 9090):
        super().__init__(ustreamer_port=ustreamer_port, position=NotificationOverlay.POS_TOP_RIGHT)
        # Scale increased 20% from 2 to 3
        self._scale = 3

    def show_vlm_loading(self):
        """Show VLM loading notification."""
        text = "[ LOADING VLM ]\n\nminus-v0.1"
        self.show(text, duration=None)

    def show_vlm_ready(self):
        """Show VLM ready notification (auto-hides)."""
        text = "[ VLM READY ]\n\nDETECTION ACTIVE"
        self.show(text, duration=5.0)

    def show_vlm_failed(self):
        """Show VLM load failure notification (auto-hides)."""
        text = "[ VLM FAILED ]\n\nOCR-ONLY MODE"
        self.show(text, duration=8.0)

    def show_vlm_unloading(self):
        """Show VLM unloading notification."""
        text = "[ UNLOADING VLM ]\n\nReleasing NPU..."
        self.show(text, duration=None)

    def show_vlm_disabled(self):
        """Show VLM disabled notification (auto-hides)."""
        text = "[ VLM DISABLED ]\n\nOCR-ONLY MODE"
        self.show(text, duration=5.0)

    def show_system_ready(self):
        """Show system ready notification (auto-hides)."""
        text = "[ MINUS READY ]\n\nBLOCKING ACTIVE"
        self.show(text, duration=5.0)
