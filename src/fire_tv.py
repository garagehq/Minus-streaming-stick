"""
Fire TV Controller for Minus/Stream Sentry.

Provides ADB-based remote control of Fire TV devices over WiFi for ad skipping
and playback control during ad blocking.

Features:
- Auto-discovery of Fire TV devices on local network
- Verification that discovered device is actually a Fire TV
- ADB key generation and persistent storage for pairing
- Auto-reconnect on connection drops
- Full remote control: play, pause, select, back, d-pad, etc.
- Async-compatible interface
- Clear instructions when ADB debugging needs to be enabled

Requirements:
- Fire TV must have ADB debugging enabled (Settings > My Fire TV > Developer Options)
- First connection requires approving RSA key on TV screen
"""

import asyncio
import logging
import os
import random
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Known Fire TV device identifiers (manufacturer and model patterns)
FIRE_TV_MANUFACTURERS = ["Amazon", "amazon", "AMAZON"]
FIRE_TV_MODEL_PATTERNS = [
    r"AFT.*",      # Fire TV Stick (AFTMM, AFTT, etc.)
    r"KFMUWI.*",   # Fire HD tablets
    r"Fire.*",    # Fire TV Cube and others
    r".*Fire.*TV.*",
]

# Default ADB key location
DEFAULT_ADBKEY_PATH = os.path.expanduser("~/.android/adbkey")

# ADB port for Fire TV
ADB_PORT = 5555

# Connection timeout
CONNECT_TIMEOUT = 10.0
AUTH_TIMEOUT = 30.0  # Longer timeout for first-time authorization

# Reconnect settings
RECONNECT_DELAY_BASE = 1.0
RECONNECT_DELAY_MAX = 30.0
RECONNECT_CHECK_INTERVAL = 5.0

# Key codes for Fire TV remote (Android key events)
KEY_CODES = {
    # Navigation
    "up": "KEYCODE_DPAD_UP",
    "down": "KEYCODE_DPAD_DOWN",
    "left": "KEYCODE_DPAD_LEFT",
    "right": "KEYCODE_DPAD_RIGHT",
    "select": "KEYCODE_DPAD_CENTER",
    "enter": "KEYCODE_ENTER",
    "ok": "KEYCODE_DPAD_CENTER",

    # Media controls
    "play": "KEYCODE_MEDIA_PLAY",
    "pause": "KEYCODE_MEDIA_PAUSE",
    "play_pause": "KEYCODE_MEDIA_PLAY_PAUSE",
    "stop": "KEYCODE_MEDIA_STOP",
    "fast_forward": "KEYCODE_MEDIA_FAST_FORWARD",
    "rewind": "KEYCODE_MEDIA_REWIND",
    "next": "KEYCODE_MEDIA_NEXT",
    "previous": "KEYCODE_MEDIA_PREVIOUS",

    # Fire TV / Android TV specific
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "menu": "KEYCODE_MENU",
    "search": "KEYCODE_SEARCH",
    "assistant": "KEYCODE_ASSIST",  # Google Assistant on Google TV

    # Volume
    "volume_up": "KEYCODE_VOLUME_UP",
    "volume_down": "KEYCODE_VOLUME_DOWN",
    "mute": "KEYCODE_VOLUME_MUTE",

    # Power
    "power": "KEYCODE_POWER",
    "sleep": "KEYCODE_SLEEP",
    "wakeup": "KEYCODE_WAKEUP",
}


class FireTVController:
    """
    Controller for Fire TV devices via ADB over WiFi.

    Handles connection management, key pairing, and remote control commands.
    Designed to integrate with Minus ad blocking system.
    """

    def __init__(self, adbkey_path: Optional[str] = None):
        """
        Initialize Fire TV controller.

        Args:
            adbkey_path: Path to ADB private key file. If None, uses default ~/.android/adbkey
        """
        self.adbkey_path = adbkey_path or DEFAULT_ADBKEY_PATH
        self._device = None
        self._ip_address: Optional[str] = None
        self._port: int = ADB_PORT
        self._connected = False
        self._lock = threading.Lock()

        # Auto-reconnect state
        self._auto_reconnect = True
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_reconnect = threading.Event()
        self._consecutive_failures = 0

        # Keep-alive state (optional feature to prevent disconnects)
        self._keepalive_enabled = False
        self._keepalive_thread: Optional[threading.Thread] = None
        self._stop_keepalive = threading.Event()
        self._keepalive_base_interval = 300.0  # Base: 5 minutes
        self._keepalive_noise = 60.0  # Random noise: +/- 60 seconds

        # Connection callback
        self._on_connection_change: Optional[Callable[[bool], None]] = None

        # Ensure ADB keys exist
        self._ensure_adb_keys()

    def _ensure_adb_keys(self):
        """Generate ADB keys if they don't exist."""
        key_path = Path(self.adbkey_path)
        pub_key_path = Path(f"{self.adbkey_path}.pub")

        if key_path.exists() and pub_key_path.exists():
            logger.debug(f"[FireTV] Using existing ADB keys at {self.adbkey_path}")
            return

        # Create directory if needed
        key_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[FireTV] Generating new ADB keys at {self.adbkey_path}")

        try:
            # Try using adb keygen if available
            result = subprocess.run(
                ["adb", "keygen", str(key_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("[FireTV] ADB keys generated successfully")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fall back to manual key generation using Python
        try:
            from adb_shell.auth.keygen import keygen
            keygen(str(key_path))
            logger.info("[FireTV] ADB keys generated using adb_shell.keygen")
        except Exception as e:
            logger.error(f"[FireTV] Failed to generate ADB keys: {e}")
            logger.info("[FireTV] You can manually generate keys with: adb keygen ~/.android/adbkey")

    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Set callback for connection state changes."""
        self._on_connection_change = callback

    def _notify_connection_change(self, connected: bool):
        """Notify callback of connection state change."""
        if self._on_connection_change:
            try:
                self._on_connection_change(connected)
            except Exception as e:
                logger.error(f"[FireTV] Connection callback error: {e}")

    @staticmethod
    def _is_fire_tv_device(manufacturer: str, model: str) -> bool:
        """Check if device info indicates a Fire TV."""
        # Check manufacturer
        if manufacturer in FIRE_TV_MANUFACTURERS:
            return True

        # Check model patterns
        for pattern in FIRE_TV_MODEL_PATTERNS:
            if re.match(pattern, model or "", re.IGNORECASE):
                return True

        return False

    @staticmethod
    def discover_devices(timeout: float = 5.0, verify_fire_tv: bool = False) -> list[dict]:
        """
        Discover devices on the local network with ADB port open.

        Args:
            timeout: Scan timeout in seconds
            verify_fire_tv: If True, attempts to connect and verify each device is a Fire TV

        Returns:
            List of dicts with 'ip', 'hostname', and optionally 'verified' keys
        """
        devices = []

        # Get local IP to determine network range
        # Try multiple methods for resilience
        local_ip = None
        try:
            # Method 1: Connect to a LAN address (works without internet)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1.0)
            s.connect(("10.255.255.255", 1))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        if not local_ip or local_ip.startswith("127."):
            try:
                # Method 2: Try Google DNS (requires internet)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(2.0)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                pass

        if not local_ip or local_ip.startswith("127."):
            try:
                # Method 3: Get from hostname
                local_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                pass

        if not local_ip or local_ip.startswith("127."):
            logger.warning("[FireTV] No network connectivity - cannot discover devices")
            return devices

        # Get network prefix (assume /24)
        network_prefix = ".".join(local_ip.split(".")[:3])
        logger.info(f"[FireTV] Scanning network {network_prefix}.0/24 for ADB devices...")

        # Scan for open ADB ports
        def check_host(ip: str, results: list):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, ADB_PORT))
                sock.close()
                if result == 0:
                    # Try to get hostname
                    try:
                        hostname = socket.gethostbyaddr(ip)[0]
                    except:
                        hostname = None
                    results.append({"ip": ip, "hostname": hostname, "port": ADB_PORT})
                    logger.info(f"[FireTV] Found ADB device: {ip} ({hostname or 'unknown'})")
            except:
                pass

        # Scan in parallel
        threads = []
        results = []
        for i in range(1, 255):
            ip = f"{network_prefix}.{i}"
            t = threading.Thread(target=check_host, args=(ip, results))
            t.start()
            threads.append(t)

        # Wait for all threads with timeout
        start_time = time.time()
        for t in threads:
            remaining = timeout - (time.time() - start_time)
            if remaining > 0:
                t.join(timeout=remaining)

        if not results:
            logger.info("[FireTV] No devices with ADB port open found on network")
            logger.info("[FireTV] This could mean:")
            logger.info("[FireTV]   1. No Fire TV on this network")
            logger.info("[FireTV]   2. Fire TV has ADB debugging disabled")
            logger.info("[FireTV]   3. Fire TV is asleep or powered off")
            return devices

        logger.info(f"[FireTV] Found {len(results)} device(s) with ADB port open")
        return results

    def detect_fire_tv(self, timeout: float = 10.0) -> Optional[dict]:
        """
        Scan network and find a Fire TV device.

        This method scans the network, connects to devices with ADB open,
        and verifies which one is actually a Fire TV.

        Args:
            timeout: Scan timeout in seconds

        Returns:
            Dict with Fire TV info if found, None otherwise
        """
        logger.info("[FireTV] Searching for Fire TV devices on network...")

        # First, find devices with ADB port open
        devices = self.discover_devices(timeout=timeout)

        if not devices:
            logger.warning("[FireTV] No devices found with ADB debugging enabled")
            self._print_no_devices_found()
            return None

        # Try to connect to each and verify if it's a Fire TV
        for device in devices:
            ip = device["ip"]
            logger.info(f"[FireTV] Checking if {ip} is a Fire TV...")

            # Use wait_for_auth=True for first-time connections
            if self.connect(ip, wait_for_auth=True):
                info = self.get_device_info()
                manufacturer = info.get("manufacturer", "") if info else ""
                model = info.get("model", "") if info else ""

                # Check if it's a Fire TV
                if self._is_fire_tv_device(manufacturer, model):
                    logger.info(f"[FireTV] Confirmed Fire TV: {manufacturer} {model} at {ip}")
                    device["verified"] = True
                    device["manufacturer"] = manufacturer
                    device["model"] = model
                    return device
                elif manufacturer or model:
                    # Got device info but not a Fire TV
                    logger.info(f"[FireTV] Device at {ip} is not a Fire TV: {manufacturer} {model}")
                    self.disconnect()
                else:
                    # Couldn't get device info - try alternative method
                    logger.info(f"[FireTV] Checking device identity via shell...")
                    try:
                        # Try to get Fire TV specific properties
                        result = self._device.adb_shell("getprop ro.product.brand")
                        brand = result.strip() if result else ""
                        result = self._device.adb_shell("getprop ro.product.device")
                        device_name = result.strip() if result else ""
                        result = self._device.adb_shell("getprop ro.product.model")
                        model = result.strip() if result else ""
                        result = self._device.adb_shell("getprop ro.product.manufacturer")
                        manufacturer = result.strip() if result else ""

                        logger.info(f"[FireTV] Device info: brand={brand}, device={device_name}, model={model}, manufacturer={manufacturer}")

                        # Check if any indicator suggests Fire TV
                        is_fire_tv = (
                            self._is_fire_tv_device(manufacturer, model) or
                            "amazon" in brand.lower() or
                            "fire" in device_name.lower() or
                            "aft" in device_name.lower()
                        )

                        if is_fire_tv:
                            logger.info(f"[FireTV] Confirmed Fire TV via shell: {manufacturer or brand} {model or device_name} at {ip}")
                            device["verified"] = True
                            device["manufacturer"] = manufacturer or brand
                            device["model"] = model or device_name
                            return device
                        else:
                            logger.info(f"[FireTV] Device at {ip} does not appear to be a Fire TV")
                            self.disconnect()
                    except Exception as e:
                        logger.warning(f"[FireTV] Shell query failed: {e}")
                        # If we connected successfully, assume it might be a Fire TV
                        logger.info(f"[FireTV] Connected to {ip} but couldn't verify device type - assuming Fire TV")
                        device["verified"] = False
                        device["manufacturer"] = "Unknown"
                        device["model"] = "ADB Device"
                        return device

        logger.warning("[FireTV] No Fire TV devices found")
        self._print_no_fire_tv_help()
        return None

    def _print_no_devices_found(self):
        """Print help when no ADB devices are found."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     No ADB Devices Found on Network                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This could mean:                                                            ║
║                                                                              ║
║  1. You don't have a Fire TV on this network                                 ║
║     → Only Fire TV devices are supported at this time                        ║
║                                                                              ║
║  2. ADB Debugging is NOT enabled on your Fire TV                             ║
║     → See instructions below to enable it                                    ║
║                                                                              ║
║  3. Your Fire TV is asleep or powered off                                    ║
║     → Wake it up and try again                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        self._print_adb_enable_instructions()

    def _print_no_fire_tv_help(self):
        """Print help when ADB devices found but none are Fire TV."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ADB Devices Found, But No Fire TV Detected                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  We found devices with ADB enabled, but none appear to be Fire TV devices.   ║
║                                                                              ║
║  Currently supported devices:                                                ║
║  • Fire TV Stick (all generations)                                           ║
║  • Fire TV Cube                                                              ║
║  • Fire TV (box)                                                             ║
║                                                                              ║
║  If you have a Fire TV that wasn't detected:                                 ║
║  1. Make sure it's awake (not in screensaver)                                ║
║  2. Verify ADB debugging is enabled (see below)                              ║
║  3. Try running the scan again                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    def connect(self, ip_address: str, timeout: float = CONNECT_TIMEOUT, wait_for_auth: bool = False) -> bool:
        """
        Connect to a Fire TV device.

        Args:
            ip_address: IP address of the Fire TV (can include port as IP:PORT)
            timeout: Connection timeout in seconds
            wait_for_auth: If True, uses longer timeout and shows auth instructions

        Returns:
            True if connected successfully

        Note:
            First connection requires approving the RSA key on the TV screen.
            The TV will show a dialog asking to allow USB debugging.
            For Google TV Wireless debugging, use IP:PORT format (e.g., 192.168.1.100:37421).
        """
        with self._lock:
            if self._connected and self._ip_address == ip_address:
                logger.debug(f"[FireTV] Already connected to {ip_address}")
                return True

            # Disconnect existing connection
            if self._connected:
                self._disconnect_internal()

            # Parse IP:PORT format (for Google TV Wireless debugging)
            if ':' in ip_address:
                host, port_str = ip_address.rsplit(':', 1)
                try:
                    port = int(port_str)
                    self._ip_address = host
                    self._port = port
                    logger.info(f"[FireTV] Using custom port {port} (Wireless debugging)")
                except ValueError:
                    # Not a valid port, treat as IP
                    self._ip_address = ip_address
                    self._port = ADB_PORT
            else:
                self._ip_address = ip_address
                self._port = ADB_PORT

            # Use longer timeout for first-time auth
            actual_timeout = AUTH_TIMEOUT if wait_for_auth else timeout

            if wait_for_auth:
                addr_display = f"{self._ip_address}:{self._port}"
                print(f"\n[FireTV] Connecting to {addr_display}...")
                print("[FireTV] If this is your first connection, please look at your TV screen")
                print("[FireTV] and press 'Allow' when the authorization dialog appears.")
                print(f"[FireTV] Waiting up to {int(actual_timeout)} seconds for authorization...\n")

            logger.info(f"[FireTV] Connecting to device at {self._ip_address}:{self._port}...")

            try:
                from androidtv import FireTVSync
                from adb_shell.auth.sign_pythonrsa import PythonRSASigner

                # Load or generate ADB keys
                key_path = Path(self.adbkey_path)
                if not key_path.exists():
                    self._ensure_adb_keys()

                # Load the signer
                with open(self.adbkey_path, 'r') as f:
                    priv_key = f.read()
                with open(f"{self.adbkey_path}.pub", 'r') as f:
                    pub_key = f.read()

                signer = PythonRSASigner(pub_key, priv_key)

                # Create FireTVSync instance with signer
                self._device = FireTVSync(self._ip_address, port=self._port, adbkey=self.adbkey_path, signer=signer)

                # Connect with authentication
                success = self._device.adb_connect(auth_timeout_s=actual_timeout)

                if success:
                    self._connected = True
                    self._consecutive_failures = 0
                    logger.info(f"[FireTV] Connected to {ip_address}")
                    self._notify_connection_change(True)

                    # Start auto-reconnect thread
                    self._start_reconnect_thread()

                    return True
                else:
                    logger.error(f"[FireTV] Failed to connect to {ip_address}")
                    self._print_connection_help()
                    return False

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[FireTV] Connection error: {error_msg}")

                if "refused" in error_msg.lower():
                    logger.error("[FireTV] Connection refused - ADB debugging may not be enabled")
                    self._print_adb_enable_instructions()
                elif "timeout" in error_msg.lower():
                    logger.error("[FireTV] Connection timed out - check if device is awake and on network")
                elif "auth" in error_msg.lower():
                    logger.error("[FireTV] Authentication failed - check TV screen for authorization dialog")
                    self._print_auth_instructions()

                self._device = None
                self._connected = False
                return False

    def _print_adb_enable_instructions(self):
        """Print instructions for enabling ADB debugging on Fire TV."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              How to Enable ADB Debugging on Fire TV                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Step 1: Go to Settings (gear icon on Fire TV home screen)                   ║
║                                                                              ║
║  Step 2: Select "My Fire TV" (or "Device & Software")                        ║
║                                                                              ║
║  Step 3: Select "Developer Options"                                          ║
║          ┌──────────────────────────────────────────────────────────────┐    ║
║          │ Don't see "Developer Options"?                               │    ║
║          │ Go to "About" → click "Fire TV Stick" (or your device)      │    ║
║          │ 7 times rapidly. Then go back - it will appear!             │    ║
║          └──────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  Step 4: Turn ON "ADB Debugging"                                             ║
║          → Select "OK" on the warning dialog                                 ║
║                                                                              ║
║  Step 5: Find your Fire TV's IP address                                      ║
║          Settings → My Fire TV → About → Network                             ║
║          Look for the IP address (e.g., 192.168.1.xxx)                       ║
║                                                                              ║
║  After enabling, run this script again!                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    def _print_auth_instructions(self):
        """Print instructions for authorizing ADB connection."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Authorize ADB Connection                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  When connecting for the first time, your Fire TV will show a dialog:        ║
║                                                                              ║
║       ┌────────────────────────────────────────┐                             ║
║       │      Allow USB debugging?              │                             ║
║       │                                        │                             ║
║       │  ☑ Always allow from this computer    │                             ║
║       │                                        │                             ║
║       │        [ Cancel ]    [ OK ]           │                             ║
║       └────────────────────────────────────────┘                             ║
║                                                                              ║
║  1. Look at your TV screen NOW                                               ║
║  2. Check "Always allow from this computer"                                  ║
║  3. Select "OK" to authorize                                                 ║
║                                                                              ║
║  If you don't see the dialog:                                                ║
║  • Wake up your Fire TV (press any button on remote)                         ║
║  • Make sure the Fire TV is not in screensaver mode                          ║
║  • Run the connection again                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    def _print_connection_help(self):
        """Print general connection help."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Fire TV Connection Troubleshooting                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Checklist:                                                                  ║
║                                                                              ║
║  ☐ Fire TV is ON and awake (not in sleep/screensaver)                        ║
║  ☐ ADB debugging is enabled                                                  ║
║  ☐ IP address is correct (Settings → My Fire TV → About → Network)           ║
║  ☐ Both devices are on the same WiFi network                                 ║
║  ☐ First connection: Watch TV screen for authorization dialog                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    def _disconnect_internal(self):
        """Internal disconnect without lock."""
        if self._device:
            try:
                self._device.adb_close()
            except:
                pass
            self._device = None
        self._connected = False

    def disconnect(self):
        """Disconnect from Fire TV."""
        with self._lock:
            # Stop auto-reconnect
            self._stop_reconnect.set()
            if self._reconnect_thread:
                self._reconnect_thread.join(timeout=2.0)
                self._reconnect_thread = None

            if not self._connected:
                return

            logger.info(f"[FireTV] Disconnecting from {self._ip_address}")
            self._disconnect_internal()
            self._notify_connection_change(False)

    def _start_reconnect_thread(self):
        """Start auto-reconnect monitoring thread."""
        if not self._auto_reconnect:
            return

        # Don't start if already running - prevents thread explosion bug
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            logger.debug("[FireTV] Reconnect thread already running, not starting another")
            return

        self._stop_reconnect.clear()
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="FireTV-Reconnect"
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        """Monitor connection and auto-reconnect if dropped."""
        logger.debug("[FireTV] Auto-reconnect monitor started")

        while not self._stop_reconnect.is_set():
            self._stop_reconnect.wait(RECONNECT_CHECK_INTERVAL)

            if self._stop_reconnect.is_set():
                break

            # Check connection health
            if not self._check_connection():
                logger.warning("[FireTV] Connection lost, attempting reconnect...")
                self._connected = False
                self._notify_connection_change(False)

                # Exponential backoff
                delay = min(
                    RECONNECT_DELAY_BASE * (2 ** self._consecutive_failures),
                    RECONNECT_DELAY_MAX
                )
                self._consecutive_failures += 1

                logger.info(f"[FireTV] Reconnecting in {delay:.1f}s (attempt {self._consecutive_failures})")
                self._stop_reconnect.wait(delay)

                if self._stop_reconnect.is_set():
                    break

                # Try to reconnect
                if self._ip_address:
                    try:
                        # Clear old device (don't need lock - connect() will take it)
                        self._device = None

                        # connect() takes its own lock internally
                        if self.connect(self._ip_address):
                            self._consecutive_failures = 0
                    except Exception as e:
                        logger.error(f"[FireTV] Reconnect failed: {e}")

        logger.debug("[FireTV] Auto-reconnect monitor stopped")

    def _check_connection(self) -> bool:
        """Check if connection is still alive."""
        if not self._device or not self._connected:
            return False

        try:
            # Simple check - try to run a command
            result = self._device.adb_shell("echo ok")
            return result is not None and "ok" in str(result)
        except Exception as e:
            logger.debug(f"[FireTV] Connection check failed: {e}")
            return False

    # Keep-alive methods
    def set_keepalive_enabled(self, enabled: bool):
        """Enable or disable keep-alive pings."""
        if enabled and not self._keepalive_enabled:
            self._keepalive_enabled = True
            self._start_keepalive_thread()
            logger.info("[FireTV] Keep-alive enabled (pings every ~5m with random jitter)")
        elif not enabled and self._keepalive_enabled:
            self._keepalive_enabled = False
            self._stop_keepalive_thread()
            logger.info("[FireTV] Keep-alive disabled")

    def is_keepalive_enabled(self) -> bool:
        """Check if keep-alive is enabled."""
        return self._keepalive_enabled

    def _start_keepalive_thread(self):
        """Start the keep-alive thread."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return

        self._stop_keepalive.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="FireTV-KeepAlive",
            daemon=True
        )
        self._keepalive_thread.start()

    def _stop_keepalive_thread(self):
        """Stop the keep-alive thread."""
        self._stop_keepalive.set()
        if self._keepalive_thread:
            self._keepalive_thread.join(timeout=2.0)
            self._keepalive_thread = None

    def _keepalive_loop(self):
        """Keep-alive loop - pings Fire TV periodically with randomized intervals."""
        logger.debug("[FireTV] Keep-alive thread started")
        while not self._stop_keepalive.is_set():
            # Randomized interval: base +/- noise (4-6 minutes by default)
            noise = random.uniform(-self._keepalive_noise, self._keepalive_noise)
            interval = self._keepalive_base_interval + noise

            # Wait for randomized interval
            if self._stop_keepalive.wait(timeout=interval):
                break  # Stop requested

            # Send ping if connected
            if self._connected and self._device:
                try:
                    with self._lock:
                        if self._connected and self._device:
                            self._device.adb_shell("echo keepalive")
                            logger.debug(f"[FireTV] Keep-alive ping sent (next in ~{interval/60:.1f}m)")
                except Exception as e:
                    logger.warning(f"[FireTV] Keep-alive ping failed: {e}")

        logger.debug("[FireTV] Keep-alive thread stopped")

    def is_connected(self) -> bool:
        """Check if currently connected to Fire TV."""
        with self._lock:
            return self._connected and self._check_connection()

    def send_command(self, command: str) -> bool:
        """
        Send a remote control command to Fire TV.

        Args:
            command: Command name (e.g., "select", "back", "play_pause")
                    See KEY_CODES dict for all available commands.

        Returns:
            True if command sent successfully
        """
        if command not in KEY_CODES:
            logger.error(f"[FireTV] Unknown command: {command}. Available: {list(KEY_CODES.keys())}")
            return False

        return self._send_key(KEY_CODES[command])

    def _send_key(self, keycode: str) -> bool:
        """Send a key event to Fire TV."""
        with self._lock:
            if not self._connected or not self._device:
                logger.warning(f"[FireTV] Not connected, cannot send {keycode}")
                return False

            try:
                # Use adb_shell to send key event
                self._device.adb_shell(f"input keyevent {keycode}")
                logger.debug(f"[FireTV] Sent key: {keycode}")
                return True
            except Exception as e:
                logger.error(f"[FireTV] Failed to send key {keycode}: {e}")
                self._connected = False
                return False

    def _call_device_method(self, method_name: str) -> bool:
        """Call a built-in device method."""
        with self._lock:
            if not self._connected or not self._device:
                logger.warning(f"[FireTV] Not connected, cannot call {method_name}")
                return False

            try:
                method = getattr(self._device, method_name, None)
                if method and callable(method):
                    method()
                    logger.debug(f"[FireTV] Called method: {method_name}")
                    return True
                else:
                    logger.error(f"[FireTV] Method not found: {method_name}")
                    return False
            except Exception as e:
                logger.error(f"[FireTV] Failed to call {method_name}: {e}")
                return False

    def send_keys(self, *commands: str, delay: float = 0.1) -> bool:
        """
        Send multiple commands in sequence.

        Args:
            *commands: Command names to send
            delay: Delay between commands in seconds

        Returns:
            True if all commands sent successfully
        """
        success = True
        for i, cmd in enumerate(commands):
            if i > 0 and delay > 0:
                time.sleep(delay)
            if not self.send_command(cmd):
                success = False
        return success

    def skip_ad(self, method: str = "select") -> bool:
        """
        Attempt to skip an ad.

        Different streaming services have different skip mechanisms:
        - Netflix/Prime: Usually "select" when skip button appears
        - YouTube: "right" to seek, then "select"
        - Hulu: "select" for skip button

        Args:
            method: Skip method - "select" (default), "fast_forward", or "seek_right"

        Returns:
            True if command sent successfully
        """
        logger.info(f"[FireTV] Attempting to skip ad (method={method})")

        if method == "select":
            return self.send_command("select")
        elif method == "fast_forward":
            return self.send_command("fast_forward")
        elif method == "seek_right":
            # Send multiple right presses to seek forward
            return self.send_keys("right", "right", "right", delay=0.05)
        else:
            logger.warning(f"[FireTV] Unknown skip method: {method}")
            return self.send_command("select")

    def is_awake(self) -> Optional[bool]:
        """Ask Android's power manager whether the device is awake.

        Returns True/False, or None when it cannot be determined (not
        connected, ADB error, unexpected output).
        """
        with self._lock:
            if not self._connected or not self._device:
                return None
            try:
                out = self._device.adb_shell("dumpsys power | grep -m1 mWakefulness=") or ''
            except Exception as e:
                logger.debug(f"[FireTV] wakefulness query failed: {e}")
                return None
        if 'mWakefulness=Awake' in out:
            return True
        if 'mWakefulness=' in out:
            return False   # Asleep / Dozing / Dreaming
        return None

    def get_current_app(self) -> Optional[str]:
        """
        Get the currently active app package name.

        Returns:
            Package name (e.g., "com.netflix.ninja") or None if not connected
        """
        with self._lock:
            if not self._connected or not self._device:
                return None

            try:
                # current_app is a property that returns current app info
                # Need to call update() first to refresh state, then access the property
                self._device.update()
                current_app = self._device.current_app
                if current_app:
                    # current_app returns a dict with 'package' key or just the package string
                    if isinstance(current_app, dict):
                        return current_app.get("package")
                    elif callable(current_app):
                        # It's a method, call it
                        result = current_app()
                        if isinstance(result, dict):
                            return result.get("package")
                        return str(result) if result else None
                    return str(current_app)
                return None
            except Exception as e:
                logger.error(f"[FireTV] Failed to get current app: {e}")
                return None

    def get_device_info(self) -> Optional[dict]:
        """
        Get Fire TV device information.

        Returns:
            Dict with device info or None if not connected
        """
        with self._lock:
            if not self._connected or not self._device:
                return None

            try:
                # Get device properties
                props = self._device.get_device_properties()
                if props:
                    manufacturer = props.get("ro.product.manufacturer", "")
                    model = props.get("ro.product.model", "")
                    android_version = props.get("ro.build.version.release", "")
                else:
                    # Fall back to shell commands
                    manufacturer = self._device.adb_shell("getprop ro.product.manufacturer").strip()
                    model = self._device.adb_shell("getprop ro.product.model").strip()
                    android_version = self._device.adb_shell("getprop ro.build.version.release").strip()

                return {
                    "ip": self._ip_address,
                    "manufacturer": manufacturer,
                    "model": model,
                    "android_version": android_version,
                }
            except Exception as e:
                logger.error(f"[FireTV] Failed to get device info: {e}")
                return None

    def get_status(self) -> dict:
        """Get controller status."""
        return {
            "connected": self._connected,
            "ip_address": self._ip_address,
            "auto_reconnect": self._auto_reconnect,
            "reconnect_failures": self._consecutive_failures,
        }

    def wake_up(self) -> bool:
        """Wake up Fire TV from sleep."""
        logger.info("[FireTV] Waking up device")
        return self.send_command("wakeup")

    def go_home(self) -> bool:
        """Go to Fire TV home screen."""
        return self.send_command("home")

    def go_back(self) -> bool:
        """Press back button."""
        return self.send_command("back")

    # Async wrappers for integration with async code

    async def async_connect(self, ip_address: str, timeout: float = CONNECT_TIMEOUT) -> bool:
        """Async version of connect()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.connect, ip_address, timeout)

    async def async_send_command(self, command: str) -> bool:
        """Async version of send_command()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.send_command, command)

    async def async_skip_ad(self, method: str = "select") -> bool:
        """Async version of skip_ad()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.skip_ad, method)

    async def async_get_current_app(self) -> Optional[str]:
        """Async version of get_current_app()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_current_app)

    @staticmethod
    async def async_discover_devices(timeout: float = 5.0) -> list[dict]:
        """Async version of discover_devices()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, FireTVController.discover_devices, timeout)

    def destroy(self):
        """Clean up resources."""
        self.disconnect()


# Convenience function for quick testing
def quick_connect(ip_address: Optional[str] = None) -> Optional[FireTVController]:
    """
    Quick connect to a Fire TV device.

    If no IP provided, scans network and auto-discovers Fire TV.
    Verifies the device is actually a Fire TV before returning.

    Args:
        ip_address: Optional IP address. If None, auto-discovers.

    Returns:
        Connected FireTVController or None if failed
    """
    controller = FireTVController()

    if not ip_address:
        # Auto-detect Fire TV on network
        logger.info("[FireTV] No IP provided, searching for Fire TV on network...")
        fire_tv = controller.detect_fire_tv(timeout=10.0)

        if not fire_tv:
            # detect_fire_tv already printed helpful messages
            return None

        logger.info(f"[FireTV] Found and connected to Fire TV at {fire_tv['ip']}")
        return controller
    else:
        # Connect to specified IP
        if controller.connect(ip_address):
            # Verify it's a Fire TV
            info = controller.get_device_info()
            if info:
                manufacturer = info.get("manufacturer", "")
                model = info.get("model", "")
                if controller._is_fire_tv_device(manufacturer, model):
                    logger.info(f"[FireTV] Connected to Fire TV: {manufacturer} {model}")
                    return controller
                else:
                    logger.warning(f"[FireTV] Device at {ip_address} is not a Fire TV: {manufacturer} {model}")
                    print(f"\nNote: The device at {ip_address} appears to be a {manufacturer} {model}")
                    print("This module is designed for Fire TV devices.")
                    controller.disconnect()
                    return None
            else:
                # Couldn't get info but connected - might still be Fire TV
                logger.warning("[FireTV] Connected but couldn't verify device type")
                return controller

    return None


def auto_setup() -> Optional[FireTVController]:
    """
    Fully automatic Fire TV setup.

    Scans network, finds Fire TV, connects, and handles first-time setup.
    Provides user-friendly messages throughout the process.

    Returns:
        Connected FireTVController or None if setup failed
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Minus Fire TV Auto-Setup                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    controller = FireTVController()

    print("Scanning network for Fire TV devices...")
    print("(This may take up to 10 seconds)\n")

    fire_tv = controller.detect_fire_tv(timeout=10.0)

    if fire_tv:
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         Fire TV Found!                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Device: {fire_tv.get('manufacturer', 'Amazon')} {fire_tv.get('model', 'Fire TV'):<48} ║
║  IP Address: {fire_tv['ip']:<63} ║
║  Status: Connected and ready                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        return controller

    return None
