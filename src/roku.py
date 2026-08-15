"""
Roku Controller for Minus.

Provides ECP (External Control Protocol) remote control of Roku devices over WiFi.

Features:
- Auto-discovery of Roku devices via SSDP
- Full remote control: navigation, media, volume
- App launching
- Device info retrieval

Requirements:
- Roku must have Device Connect enabled (Settings > System > Advanced > Control by mobile apps)
"""

import logging
import socket
import threading
import time
from typing import Optional, Callable, Dict, Any
import requests

logger = logging.getLogger(__name__)

# Roku ECP port
ROKU_PORT = 8060

# Discovery timeout
DISCOVERY_TIMEOUT = 5.0

# Connection check interval
CHECK_INTERVAL = 10.0

# Roku ECP key mappings
ECP_KEYS = {
    'up': 'Up',
    'down': 'Down',
    'left': 'Left',
    'right': 'Right',
    'select': 'Select',
    'back': 'Back',
    'home': 'Home',
    'info': 'Info',
    'play': 'Play',
    'pause': 'Pause',
    'play_pause': 'Play',  # Roku toggles with Play
    'fast_forward': 'Fwd',
    'rewind': 'Rev',
    'volume_up': 'VolumeUp',
    'volume_down': 'VolumeDown',
    'mute': 'VolumeMute',
    'power': 'Power',
    'power_off': 'PowerOff',
    'instant_replay': 'InstantReplay',
    'search': 'Search',
    'enter': 'Enter',
}

# Roku app IDs
ROKU_APPS = {
    'netflix': '12',
    'youtube': '837',
    'prime': '13',
    'hulu': '2285',
    'disney': '291097',
    'hbo': '61322',
    'peacock': '593099',
    'paramount': '31440',
    'apple_tv': '551012',
    'plex': '13535',
    'spotify': '22297',
    'tubi': '41468',
    'pluto': '74519',
}


class RokuController:
    """
    Controller for Roku devices via ECP (External Control Protocol).
    """

    def __init__(self):
        self._ip_address: Optional[str] = None
        self._device_info: Optional[Dict[str, Any]] = None
        self._connected = False
        self._lock = threading.Lock()

        # Auto-reconnect
        self._auto_reconnect = True
        self._reconnect_thread: Optional[threading.Thread] = None
        self._stop_reconnect = threading.Event()
        self._consecutive_failures = 0

        # Serial number of the device we last connected to, so rediscovery
        # after a DHCP change follows the SAME physical Roku (and can't
        # hijack a second Roku on the network).
        self._last_serial: Optional[str] = None

        # Connection callback
        self._on_connection_change: Optional[Callable[[bool], None]] = None

        # Fired when auto-reconnect finds the Roku at a NEW IP address
        # (DHCP lease change) so the owner can persist it.
        self._on_ip_change: Optional[Callable[[str], None]] = None

    def set_connection_callback(self, callback: Callable[[bool], None]):
        """Set callback for connection state changes."""
        self._on_connection_change = callback

    def set_ip_change_callback(self, callback: Callable[[str], None]):
        """Set callback fired when auto-reconnect follows the Roku to a new IP."""
        self._on_ip_change = callback

    def _notify_connection_change(self, connected: bool):
        """Notify callback of connection state change."""
        if self._on_connection_change:
            try:
                self._on_connection_change(connected)
            except Exception as e:
                logger.error(f"[Roku] Connection callback error: {e}")

    @staticmethod
    def discover_devices(timeout: float = DISCOVERY_TIMEOUT) -> list:
        """
        Discover Roku devices on the local network using SSDP.

        Returns:
            List of dicts with 'ip', 'name', 'model' keys
        """
        devices = []

        # SSDP M-SEARCH message for Roku devices
        ssdp_request = (
            'M-SEARCH * HTTP/1.1\r\n'
            'HOST: 239.255.255.250:1900\r\n'
            'MAN: "ssdp:discover"\r\n'
            'MX: 3\r\n'
            'ST: roku:ecp\r\n'
            '\r\n'
        )

        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.settimeout(timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Send M-SEARCH to multicast address
            sock.sendto(ssdp_request.encode(), ('239.255.255.250', 1900))

            logger.info(f"[Roku] Scanning for Roku devices (timeout={timeout}s)...")

            start_time = time.time()
            seen_ips = set()

            while time.time() - start_time < timeout:
                try:
                    data, addr = sock.recvfrom(1024)
                    ip = addr[0]

                    if ip in seen_ips:
                        continue
                    seen_ips.add(ip)

                    response = data.decode('utf-8', errors='ignore')

                    # Check if it's a Roku response
                    if 'roku' in response.lower() or 'LOCATION' in response:
                        # Try to get device info
                        try:
                            info_url = f'http://{ip}:{ROKU_PORT}/query/device-info'
                            r = requests.get(info_url, timeout=2)
                            if r.status_code == 200:
                                # Parse basic info from XML
                                content = r.text
                                name = _extract_xml_value(content, 'user-device-name') or \
                                       _extract_xml_value(content, 'friendly-device-name') or \
                                       'Roku'
                                model = _extract_xml_value(content, 'model-name') or 'Unknown'
                                serial = _extract_xml_value(content, 'serial-number') or ''

                                device = {
                                    'ip': ip,
                                    'name': name,
                                    'model': model,
                                    'serial': serial,
                                }
                                devices.append(device)
                                logger.info(f"[Roku] Found: {name} ({model}) at {ip}")
                        except Exception as e:
                            logger.debug(f"[Roku] Could not get device info for {ip}: {e}")
                            devices.append({'ip': ip, 'name': 'Roku', 'model': 'Unknown'})

                except socket.timeout:
                    break
                except Exception as e:
                    logger.debug(f"[Roku] Discovery recv error: {e}")

            sock.close()

        except Exception as e:
            logger.error(f"[Roku] Discovery error: {e}")

        if not devices:
            logger.info("[Roku] No Roku devices found on network")
            logger.info("[Roku] Make sure Device Connect is enabled on your Roku:")
            logger.info("[Roku]   Settings > System > Advanced system settings > Control by mobile apps")

        return devices

    def connect(self, ip_address: str, timeout: float = 5.0) -> bool:
        """
        Connect to a Roku device.

        Args:
            ip_address: IP address of the Roku
            timeout: Connection timeout

        Returns:
            True if connected successfully
        """
        with self._lock:
            if self._connected and self._ip_address == ip_address:
                return True

            self._ip_address = ip_address
            logger.info(f"[Roku] Connecting to {ip_address}...")

            try:
                # Try to get device info to verify connection
                url = f'http://{ip_address}:{ROKU_PORT}/query/device-info'
                response = requests.get(url, timeout=timeout)

                if response.status_code == 200:
                    content = response.text
                    self._device_info = {
                        'ip': ip_address,
                        'name': _extract_xml_value(content, 'user-device-name') or
                                _extract_xml_value(content, 'friendly-device-name') or 'Roku',
                        'model': _extract_xml_value(content, 'model-name') or 'Unknown',
                        'serial': _extract_xml_value(content, 'serial-number') or '',
                        'software_version': _extract_xml_value(content, 'software-version') or '',
                    }
                    self._connected = True
                    self._consecutive_failures = 0
                    self._last_serial = self._device_info.get('serial') or self._last_serial
                    logger.info(f"[Roku] Connected to {self._device_info['name']} ({self._device_info['model']})")
                    self._notify_connection_change(True)
                    self._start_reconnect_thread()
                    return True
                else:
                    logger.error(f"[Roku] Connection failed: HTTP {response.status_code}")
                    return False

            except requests.exceptions.Timeout:
                logger.error(f"[Roku] Connection timeout to {ip_address}")
                return False
            except requests.exceptions.ConnectionError as e:
                logger.error(f"[Roku] Connection error: {e}")
                logger.info("[Roku] Make sure Device Connect is enabled on your Roku")
                return False
            except Exception as e:
                logger.error(f"[Roku] Connection error: {e}")
                return False

    def disconnect(self):
        """Disconnect from Roku (user-initiated — permanently stops auto-reconnect)."""
        # Stop the reconnect thread OUTSIDE the lock: the loop calls
        # connect(), which takes the lock, so joining while holding it can
        # time out and leave an orphan thread that reconnects after the
        # user asked to disconnect.
        self._stop_reconnect.set()
        thread = self._reconnect_thread
        self._reconnect_thread = None
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3.0)

        with self._lock:
            if self._connected:
                logger.info(f"[Roku] Disconnected from {self._ip_address}")
            self._connected = False
            self._device_info = None
        self._notify_connection_change(False)

    def start_monitoring(self, ip_address: Optional[str] = None):
        """Arm the auto-reconnect loop without requiring a successful
        initial connection.

        Used at startup when the saved Roku is unreachable (powered off,
        rebooting, mid-DHCP-renew): the loop keeps retrying the saved IP
        and periodically rescans the network, so the controller comes up
        on its own once the device returns. A user-initiated disconnect()
        still stops the loop for good.
        """
        if ip_address:
            self._ip_address = ip_address
        self._start_reconnect_thread()

    def _start_reconnect_thread(self):
        """Start connection monitoring thread."""
        if not self._auto_reconnect:
            return

        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return

        self._stop_reconnect.clear()
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop,
            daemon=True,
            name="Roku-Reconnect"
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self):
        """Monitor connection and auto-reconnect if dropped."""
        while not self._stop_reconnect.is_set():
            self._stop_reconnect.wait(CHECK_INTERVAL)

            if self._stop_reconnect.is_set():
                break

            self._reconnect_tick()

    def _reconnect_tick(self):
        """One health-check / reconnect cycle (separated for testability).

        Reconnects when EITHER the device stops answering ECP OR the
        `_connected` flag was dropped by a failed keypress. Checking only
        the live HTTP endpoint was a sticky trap: a transient outage
        (Roku's nightly update reboot, a WiFi blip) cleared `_connected`,
        the single immediate reconnect attempt failed while the device was
        still booting, and once the Roku came back the health check passed
        again — so connect() was never called and the controller stayed
        "disconnected" until the user reconnected via the web UI.
        """
        if self._connected and self._check_connection():
            self._consecutive_failures = 0
            return

        if self._connected:
            logger.warning("[Roku] Connection lost, attempting reconnect...")
            self._connected = False
            self._notify_connection_change(False)

        self._consecutive_failures += 1

        if self._stop_reconnect.is_set():
            return

        # Try the last known IP first
        if self._ip_address and self.connect(self._ip_address):
            return

        # The known IP keeps failing — the Roku may have picked up a new
        # DHCP lease. Every 3rd consecutive failure (~30s), rescan the
        # network and follow the device to its new address.
        if self._consecutive_failures % 3 == 0 and not self._stop_reconnect.is_set():
            self._rediscover_and_connect()

    def _rediscover_and_connect(self) -> bool:
        """Scan the network for the Roku after the saved IP went stale.

        Prefers the same physical device (serial match); a device whose
        serial definitely differs from the one we were paired with is
        skipped. On success at a new address, fires the ip-change callback
        so the owner can persist it for the next restart.
        """
        old_ip = self._ip_address
        logger.info(f"[Roku] {old_ip or 'Saved IP'} unreachable — rescanning network "
                    "(the Roku's IP may have changed)")
        try:
            devices = self.discover_devices(timeout=DISCOVERY_TIMEOUT)
        except Exception as e:
            logger.debug(f"[Roku] Rediscovery error: {e}")
            return False

        candidates = []
        for device in devices:
            serial = device.get('serial', '') or ''
            if self._last_serial and serial and serial != self._last_serial:
                continue  # a DIFFERENT Roku on the network — not ours
            candidates.append(device)
        # Exact serial matches first, unknown-serial entries as fallback
        candidates.sort(key=lambda d: 0 if (self._last_serial and d.get('serial') == self._last_serial) else 1)

        for device in candidates:
            if self._stop_reconnect.is_set():
                return False
            ip = device.get('ip')
            if not ip:
                continue
            if self.connect(ip):
                if ip != old_ip:
                    logger.info(f"[Roku] Roku moved {old_ip} -> {ip}")
                    if self._on_ip_change:
                        try:
                            self._on_ip_change(ip)
                        except Exception as e:
                            logger.error(f"[Roku] IP-change callback error: {e}")
                return True
        return False

    def _check_connection(self) -> bool:
        """Check if connection is still alive."""
        if not self._ip_address:
            return False

        try:
            url = f'http://{self._ip_address}:{ROKU_PORT}/'
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except:
            return False

    def is_connected(self) -> bool:
        """Check if connected to Roku."""
        with self._lock:
            return self._connected

    def send_command(self, command: str) -> bool:
        """
        Send a remote control command to Roku.

        Args:
            command: Command name (e.g., 'select', 'back', 'play')

        Returns:
            True if command sent successfully
        """
        if command not in ECP_KEYS:
            logger.error(f"[Roku] Unknown command: {command}")
            return False

        return self._send_keypress(ECP_KEYS[command])

    def _send_keypress(self, key: str) -> bool:
        """Send a keypress via ECP."""
        if not self._connected or not self._ip_address:
            logger.warning("[Roku] Not connected")
            return False

        url = f'http://{self._ip_address}:{ROKU_PORT}/keypress/{key}'

        # Retry up to 2 times on timeout
        for attempt in range(2):
            try:
                response = requests.post(url, timeout=5)
                # Roku returns 200/204 for successful key presses; under
                # load it returns 202 Accepted (key queued) — also success
                # (observed live 2026-07-02 during a Back sequence).
                if response.status_code in (200, 202, 204):
                    logger.debug(f"[Roku] Sent key: {key}")
                    return True
                else:
                    logger.error(f"[Roku] Key press failed: HTTP {response.status_code}")
                    return False
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(f"[Roku] Key press timeout, retrying...")
                    continue
                logger.error(f"[Roku] Key press timeout after retry")
                return False
            except Exception as e:
                logger.error(f"[Roku] Key press error: {e}")
                self._connected = False
                return False
        return False

    def send_keys(self, *commands: str, delay: float = 0.1) -> bool:
        """Send multiple commands in sequence."""
        success = True
        for i, cmd in enumerate(commands):
            if i > 0 and delay > 0:
                time.sleep(delay)
            if not self.send_command(cmd):
                success = False
        return success

    def launch_app(self, app_name: str) -> bool:
        """
        Launch an app on Roku.

        Args:
            app_name: App name (e.g., 'netflix', 'youtube')

        Returns:
            True if launch command sent
        """
        app_id = ROKU_APPS.get(app_name.lower())
        if not app_id:
            logger.error(f"[Roku] Unknown app: {app_name}. Available: {list(ROKU_APPS.keys())}")
            return False

        if not self._connected or not self._ip_address:
            logger.warning("[Roku] Not connected")
            return False

        try:
            url = f'http://{self._ip_address}:{ROKU_PORT}/launch/{app_id}'
            response = requests.post(url, timeout=5)
            # Roku returns 200/204 for successful launches; 202 = queued
            if response.status_code in (200, 202, 204):
                logger.info(f"[Roku] Launched app: {app_name}")
                return True
            else:
                logger.error(f"[Roku] App launch failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"[Roku] App launch error: {e}")
            return False

    def launch_app_with_content(self, app_name: str, content_id: str) -> bool:
        """Launch an app deep-linked to a specific content item.

        For YouTube (app 837), content_id is a plain video id — ECP
        `/launch/837?contentId=<id>` starts that video playing directly.

        Args:
            app_name: App name (e.g., 'youtube')
            content_id: App-specific content identifier

        Returns:
            True if launch command was accepted
        """
        app_id = ROKU_APPS.get(app_name.lower())
        if not app_id:
            logger.error(f"[Roku] Unknown app: {app_name}. Available: {list(ROKU_APPS.keys())}")
            return False

        if not self._connected or not self._ip_address:
            logger.warning("[Roku] Not connected")
            return False

        try:
            url = f'http://{self._ip_address}:{ROKU_PORT}/launch/{app_id}?contentId={content_id}'
            response = requests.post(url, timeout=5)
            if response.status_code in (200, 202, 204):
                logger.info(f"[Roku] Launched {app_name} with contentId={content_id}")
                return True
            logger.error(f"[Roku] Deep-link launch failed: HTTP {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"[Roku] Deep-link launch error: {e}")
            return False

    def get_active_app(self) -> Optional[str]:
        """Get the currently active app on Roku via ECP.

        Returns the app name (e.g., 'YouTube', 'Home'), or None if unavailable.
        """
        if not self._connected or not self._ip_address:
            return None

        try:
            url = f'http://{self._ip_address}:{ROKU_PORT}/query/active-app'
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                import re
                # Extract app name from XML: <app ...>AppName</app>
                match = re.search(r'<app[^>]*>([^<]+)</app>', response.text)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            logger.debug(f"[Roku] Active app query error: {e}")
            return None

    def get_active_app_id(self) -> Optional[str]:
        """Get the currently active app ID on Roku.

        Returns the app ID (e.g., '837' for YouTube, '562859' for Home).
        """
        if not self._connected or not self._ip_address:
            return None

        try:
            url = f'http://{self._ip_address}:{ROKU_PORT}/query/active-app'
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                import re
                # Accept non-numeric ids too: Roku OS 15.x reports the home screen
                # as <app id="native-ui">; a digits-only match returned None which
                # callers treat as "query failed, don't interfere" (observed live
                # 2026-07-12: 50-minute stuck-on-home loop).
                match = re.search(r'<app\s+id="([^"]+)"', response.text)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            logger.debug(f"[Roku] Active app ID query error: {e}")
            return None

    def is_screensaver_active(self) -> bool:
        """Check if the Roku screensaver is currently active.

        When the screensaver activates, the /query/active-app response includes
        a <screensaver> element alongside the <app> element. The app may still
        be YouTube, but the screensaver overlays it.
        """
        if not self._connected or not self._ip_address:
            return False

        try:
            url = f'http://{self._ip_address}:{ROKU_PORT}/query/active-app'
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return '<screensaver' in response.text.lower()
            return False
        except Exception as e:
            logger.debug(f"[Roku] Screensaver check error: {e}")
            return False

    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """Get device information."""
        return self._device_info

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            'connected': self._connected,
            'ip_address': self._ip_address,
            'device_info': self._device_info,
        }

    def check_control_mode(self) -> str:
        """
        Check if Roku is in full control mode or limited mode.

        In Limited mode, ECP commands return 200 but don't actually work.
        We detect this by trying to query the apps list - Limited mode
        returns 403 or a restricted response.

        Returns:
            'full' - Full control available
            'limited' - Limited mode (commands won't work)
            'unknown' - Could not determine
        """
        if not self._connected or not self._ip_address:
            return 'unknown'

        try:
            # Try to query apps list - this is restricted in Limited mode
            url = f'http://{self._ip_address}:{ROKU_PORT}/query/apps'
            response = requests.get(url, timeout=3)

            if response.status_code == 200:
                # Check if we got actual app data
                content = response.text
                if '<app' in content.lower():
                    logger.info("[Roku] Control mode: FULL (apps accessible)")
                    return 'full'
                else:
                    logger.warning("[Roku] Control mode: LIMITED (apps query returned empty)")
                    return 'limited'
            elif response.status_code == 403:
                logger.warning("[Roku] Control mode: LIMITED (403 Forbidden)")
                return 'limited'
            else:
                logger.warning(f"[Roku] Control mode check returned HTTP {response.status_code}")
                return 'limited'

        except Exception as e:
            logger.error(f"[Roku] Control mode check error: {e}")
            return 'unknown'

    def is_limited_mode(self) -> bool:
        """Check if Roku is in limited control mode."""
        return self.check_control_mode() == 'limited'

    def destroy(self):
        """Clean up resources."""
        self.disconnect()


def _extract_xml_value(xml_content: str, tag: str) -> Optional[str]:
    """Extract a value from simple XML content."""
    import re
    match = re.search(f'<{tag}>([^<]*)</{tag}>', xml_content)
    return match.group(1) if match else None


# Convenience function for quick setup
def quick_connect(ip_address: Optional[str] = None) -> Optional[RokuController]:
    """
    Quick connect to a Roku device.

    If no IP provided, scans network and connects to first found device.

    Returns:
        Connected RokuController or None if failed
    """
    controller = RokuController()

    if not ip_address:
        logger.info("[Roku] Scanning for Roku devices...")
        devices = controller.discover_devices()

        if not devices:
            return None

        ip_address = devices[0]['ip']
        logger.info(f"[Roku] Auto-detected Roku at {ip_address}")

    if controller.connect(ip_address):
        return controller

    return None
