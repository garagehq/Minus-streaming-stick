"""
WiFi Manager Module for Minus.

Handles WiFi connectivity and captive portal AP mode:
- Detect WiFi connection status
- Scan for available networks
- Connect to networks via NetworkManager
- Create "Minus" hotspot AP when no WiFi is connected
- Auto-restart AP if WiFi drops for 30+ seconds
- Auto-recover from monitor-started AP mode back to a saved network when it
  reappears (a router reboot must not strand the box on its own hotspot)
"""

import os
import subprocess
import logging
import threading
import time
import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

# Config file path
WIFI_CONFIG_FILE = Path.home() / '.minus_wifi_config.json'

# AP Configuration
AP_SSID = 'Minus'
AP_IP = '10.42.0.1'
AP_INTERFACE = 'wlP2p33s0'  # Will be auto-detected if this fails

# Timing
WIFI_CHECK_INTERVAL = 5  # seconds
WIFI_DISCONNECT_THRESHOLD = 30  # seconds before starting AP

# While a monitor-started hotspot is up, periodically try to get back onto a
# saved WiFi network. Without this, AP mode is a terminal state: a transient
# router outage (30s is enough) flips the box to the setup hotspot and it
# never returns — observed live Aug 2026: a 33s "ssid-not-found" blip left the
# box stranded in AP mode (no LAN, no Tailscale) for ~30h until a power cycle.
AP_RECONNECT_RETRY_INTERVAL = int(os.environ.get('MINUS_WIFI_AP_RECONNECT_INTERVAL', '120'))
AP_RECONNECT_MAX_BACKOFF = 900  # cap for exponential backoff on failed attempts


@dataclass
class WiFiNetwork:
    """Represents a WiFi network."""
    ssid: str
    signal: int  # 0-100
    security: str  # 'Open', 'WPA', 'WPA2', etc.
    in_use: bool = False
    bssid: str = ''
    saved: bool = False  # True if already in NetworkManager's saved connections


@dataclass
class WiFiStatus:
    """Current WiFi status."""
    connected: bool = False
    ssid: str = ''
    ip_address: str = ''
    signal: int = 0
    ap_mode_active: bool = False
    ap_clients: int = 0


class WiFiManager:
    """Manager for WiFi connectivity and captive portal AP mode."""

    def __init__(self, on_ap_started: Callable = None, on_ap_stopped: Callable = None):
        self._interface = self._detect_wifi_interface()
        self._ap_mode_active = False
        self._ap_connection_name = 'Hotspot'  # nmcli hotspot always uses this name
        self._last_wifi_connected_time = time.time()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False
        self._on_ap_started = on_ap_started
        self._on_ap_stopped = on_ap_stopped
        self._connecting = False
        self._last_connection_error = ''
        # True when the AP was started by the monitor (eligible for automatic
        # recovery back to a saved network). User-initiated hotspots set this
        # False and are left alone. Defaults True so an AP that appears via
        # external state sync (get_status) is still recoverable.
        self._ap_auto_started = True
        self._ap_last_reconnect_attempt = 0.0
        self._ap_reconnect_failures = 0

        logger.info(f"[WiFi] Initialized with interface: {self._interface}")

    def _detect_wifi_interface(self) -> str:
        """Detect the WiFi interface name."""
        try:
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'DEVICE,TYPE', 'device'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                if ':wifi' in line:
                    interface = line.split(':')[0]
                    logger.info(f"[WiFi] Detected interface: {interface}")
                    return interface
        except Exception as e:
            logger.warning(f"[WiFi] Failed to detect interface: {e}")
        return AP_INTERFACE  # Fallback

    def _run_nmcli(self, args: List[str], timeout: int = 30) -> tuple:
        """Run nmcli command and return (success, output)."""
        try:
            cmd = ['nmcli'] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"[WiFi] nmcli timeout: {args}")
            return False, 'Command timed out'
        except Exception as e:
            logger.error(f"[WiFi] nmcli error: {e}")
            return False, str(e)

    def is_wifi_connected(self) -> bool:
        """Check if connected to any WiFi network (not AP mode)."""
        if self._ap_mode_active:
            return False

        # Check the actual connection name - if it's our AP, we're not "connected"
        success, output = self._run_nmcli([
            '-t', '-f', 'NAME,TYPE,DEVICE',
            'connection', 'show', '--active'
        ])

        if success:
            for line in output.split('\n'):
                parts = line.split(':')
                if len(parts) >= 3:
                    name, conn_type, device = parts[0], parts[1], parts[2]
                    # Skip our AP connection
                    if name == self._ap_connection_name:
                        continue
                    # Found a WiFi connection that's not our AP
                    if conn_type == '802-11-wireless' and device == self._interface:
                        return True

        return False

    def get_status(self) -> WiFiStatus:
        """Get current WiFi status."""
        status = WiFiStatus()

        # Check if we're in AP mode by looking at active connections
        success, output = self._run_nmcli([
            '-t', '-f', 'NAME,TYPE,DEVICE',
            'connection', 'show', '--active'
        ])

        if success:
            for line in output.split('\n'):
                parts = line.split(':')
                if len(parts) >= 3:
                    name, conn_type, device = parts[0], parts[1], parts[2]
                    if device == self._interface and conn_type == '802-11-wireless':
                        if name == self._ap_connection_name:
                            # We're in AP mode
                            status.ap_mode_active = True
                            self._ap_mode_active = True  # Sync the flag
                            status.ap_clients = self._count_ap_clients()
                            return status
                        else:
                            # We're connected to a WiFi network
                            status.connected = True
                            status.ssid = name
                            self._ap_mode_active = False  # Sync the flag
                            break

        # Sync AP mode flag based on actual state
        if not status.connected and not status.ap_mode_active:
            self._ap_mode_active = False

        if status.connected:
            # Get IP address
            status.ip_address = self._get_ip_address()
            # Get signal strength
            status.signal = self._get_signal_strength(status.ssid)

        return status

    def _get_ip_address(self) -> str:
        """Get the current WiFi IP address."""
        try:
            result = subprocess.run(
                ['ip', '-4', '-o', 'addr', 'show', self._interface],
                capture_output=True, text=True, timeout=5
            )
            # Parse: "2: wlP2p33s0 inet 192.168.1.15/24 ..."
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        except Exception as e:
            logger.warning(f"[WiFi] Failed to get IP: {e}")
        return ''

    def _get_signal_strength(self, ssid: str) -> int:
        """Get signal strength for connected network."""
        try:
            result = subprocess.run(
                ['nmcli', '-t', '-f', 'SIGNAL,SSID', 'device', 'wifi', 'list'],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split('\n'):
                parts = line.split(':')
                if len(parts) >= 2 and parts[1] == ssid:
                    return int(parts[0])
        except Exception:
            pass
        return 0

    def _count_ap_clients(self) -> int:
        """Count devices connected to our AP."""
        try:
            # Check ARP table for devices on AP subnet
            result = subprocess.run(
                ['ip', 'neigh', 'show', 'dev', self._interface],
                capture_output=True, text=True, timeout=5
            )
            # Count reachable neighbors
            count = len([l for l in result.stdout.strip().split('\n') if l and 'REACHABLE' in l])
            return count
        except Exception:
            pass
        return 0

    def scan_networks(self, bounce_ap: bool = False) -> List[WiFiNetwork]:
        """Scan for available WiFi networks.

        On single-radio WiFi chips (like the RK3588's) the radio can't scan
        while hosting an AP, so a scan taken while the Minus hotspot is active
        only sees the hotspot itself. If ``bounce_ap`` is True and the AP is
        currently active, we briefly bring the Hotspot connection down, scan,
        then bring it back up. Clients on the hotspot will see Minus drop for
        ~5 seconds.

        Saved networks discovered during the live scan are marked with
        ``saved=True`` and can be connected without re-entering a password.
        Saved networks that weren't seen in the live scan are still returned
        (signal=0) so the user can click them directly; the caller sorts so
        strong live results rank above out-of-range saved entries.
        """
        networks: List[WiFiNetwork] = []
        saved_names = {s['name'] for s in self.get_saved_networks()}

        ap_bounced = False
        if bounce_ap and self._ap_mode_active:
            logger.info("[WiFi] Bouncing AP to run a live scan...")
            # Pause auto-restart behavior by leaving _ap_mode_active True so
            # the monitor thread treats us as still in AP mode. We only take
            # the Hotspot connection down; the profile stays so bringing it
            # back up is a single nmcli call.
            self._run_nmcli(['connection', 'down', self._ap_connection_name])
            time.sleep(2.0)  # Radio switches from AP to station mode
            ap_bounced = True

        # Trigger rescan (harmless if it fails — we still read the cached list)
        self._run_nmcli(['device', 'wifi', 'rescan'])
        # Give the driver enough time to actually complete a fresh scan when
        # we just dropped AP. Otherwise 1s is enough to read cached results.
        time.sleep(4.0 if ap_bounced else 1.0)

        # Get network list - use different format to avoid BSSID colon escaping issues
        success, output = self._run_nmcli([
            '-t', '-f', 'SSID,SIGNAL,SECURITY,IN-USE',
            'device', 'wifi', 'list'
        ])

        seen_ssids = set()
        if success:
            for line in output.split('\n'):
                if not line:
                    continue

                # Format: SSID:SIGNAL:SECURITY:IN-USE
                # Split from the right to handle SSIDs that might contain colons
                parts = line.rsplit(':', 3)
                if len(parts) >= 4:
                    ssid = parts[0].replace('\\:', ':')  # Unescape any colons in SSID
                    try:
                        signal = int(parts[1])
                    except ValueError:
                        signal = 0
                    security = parts[2] if parts[2] else 'Open'
                    in_use = parts[3] == '*'

                    # Skip empty SSIDs, duplicates, and our own AP
                    if not ssid or ssid in seen_ssids:
                        continue
                    if ssid == AP_SSID and in_use:
                        continue
                    seen_ssids.add(ssid)

                    networks.append(WiFiNetwork(
                        ssid=ssid,
                        signal=signal,
                        security=security,
                        in_use=in_use,
                        bssid='',
                        saved=ssid in saved_names,
                    ))
        else:
            logger.warning("[WiFi] Live scan failed, returning saved networks only")

        # If the live scan returned nothing useful (e.g. in AP mode the radio
        # is busy hosting the hotspot and can't scan), surface saved networks
        # so the user can still reconnect to a known one.
        for name in saved_names:
            if name in seen_ssids:
                continue
            networks.append(WiFiNetwork(
                ssid=name,
                signal=0,
                security='WPA2',  # Assume protected; password already stored
                in_use=False,
                bssid='',
                saved=True,
            ))

        # Sort: in-use first, then by signal strength. Saved is a visual flag,
        # not a sort priority — a saved network with signal=0 (not in range)
        # shouldn't float above strong live results the user can actually join.
        # Tiebreaker favors saved within the same signal bucket.
        networks.sort(key=lambda n: (not n.in_use, -n.signal, not n.saved))

        if ap_bounced:
            logger.info("[WiFi] Restoring AP after live scan")
            self._run_nmcli(['connection', 'up', self._ap_connection_name], timeout=30)
            time.sleep(1.0)
            # _ap_mode_active was never cleared, so status stays consistent.

        return networks

    def get_saved_networks(self) -> List[Dict[str, Any]]:
        """Get list of saved WiFi connections."""
        networks = []

        success, output = self._run_nmcli([
            '-t', '-f', 'NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY',
            'connection', 'show'
        ])

        if not success:
            return networks

        for line in output.split('\n'):
            parts = line.split(':')
            if len(parts) >= 4 and parts[1] == '802-11-wireless':
                name = parts[0]
                # Skip our AP connection
                if name == self._ap_connection_name:
                    continue
                networks.append({
                    'name': name,
                    'autoconnect': parts[2] == 'yes',
                    'priority': int(parts[3]) if parts[3].lstrip('-').isdigit() else 0
                })

        return networks

    def connect_to_network(self, ssid: str, password: str = '') -> Dict[str, Any]:
        """Connect to a WiFi network."""
        self._connecting = True
        self._last_connection_error = ''

        try:
            # Stop AP mode if active
            if self._ap_mode_active:
                self.stop_ap_mode()
                time.sleep(2)  # Wait for interface to be ready

            # Check if connection already exists
            existing = self._get_connection_by_ssid(ssid)

            if existing:
                # Modify password if provided
                if password:
                    self._run_nmcli([
                        'connection', 'modify', existing,
                        '802-11-wireless-security.psk', password
                    ])
                # Activate existing connection
                success, output = self._run_nmcli([
                    'connection', 'up', existing
                ], timeout=45)
            else:
                # Create new connection
                if password:
                    success, output = self._run_nmcli([
                        'device', 'wifi', 'connect', ssid,
                        'password', password,
                        'ifname', self._interface
                    ], timeout=45)
                else:
                    # Open network
                    success, output = self._run_nmcli([
                        'device', 'wifi', 'connect', ssid,
                        'ifname', self._interface
                    ], timeout=45)

            if success:
                # Verify connection
                time.sleep(2)
                if self.is_wifi_connected():
                    self._last_wifi_connected_time = time.time()
                    logger.info(f"[WiFi] Connected to {ssid}")
                    return {'success': True, 'ssid': ssid}
                else:
                    self._last_connection_error = 'Connection established but verification failed'
            else:
                # Parse error message
                if 'Secrets were required' in output or 'No secrets' in output:
                    self._last_connection_error = 'Incorrect password'
                elif 'not found' in output.lower():
                    self._last_connection_error = 'Network not found'
                elif 'timeout' in output.lower():
                    self._last_connection_error = 'Connection timed out'
                else:
                    self._last_connection_error = output or 'Connection failed'

            logger.warning(f"[WiFi] Failed to connect to {ssid}: {self._last_connection_error}")
            return {'success': False, 'error': self._last_connection_error}

        except Exception as e:
            self._last_connection_error = str(e)
            logger.error(f"[WiFi] Connection error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            self._connecting = False

    def connect_saved(self, name: str) -> Dict[str, Any]:
        """Activate an existing saved NetworkManager connection by name.

        Unlike ``connect_to_network``, this does not need a password — it just
        runs ``nmcli connection up <name>`` on the saved profile. If the AP
        hotspot is currently active, we bring it down first so the radio is
        available for client-mode association.
        """
        self._connecting = True
        self._last_connection_error = ''
        try:
            # Confirm the connection actually exists before we tear down AP
            saved = {s['name'] for s in self.get_saved_networks()}
            if name not in saved:
                self._last_connection_error = f'No saved connection named "{name}"'
                return {'success': False, 'error': self._last_connection_error}

            if self._ap_mode_active:
                self.stop_ap_mode()
                time.sleep(2)  # Let the radio come back as a client

            success, output = self._run_nmcli([
                'connection', 'up', name
            ], timeout=45)

            if success:
                time.sleep(2)
                if self.is_wifi_connected():
                    self._last_wifi_connected_time = time.time()
                    logger.info(f"[WiFi] Activated saved connection: {name}")
                    return {'success': True, 'ssid': name}
                self._last_connection_error = 'Activation reported success but verification failed'
                error_code = 'verification_failed'
            else:
                low = output.lower()
                if ('secrets were required' in low or 'no secrets' in low
                        or 'passwords or encryption keys are required' in low):
                    self._last_connection_error = 'Saved password is missing or wrong — please re-enter it'
                    error_code = 'password_required'
                elif 'not found' in low:
                    self._last_connection_error = 'Network is not in range'
                    error_code = 'out_of_range'
                elif 'timeout' in low:
                    self._last_connection_error = 'Connection timed out'
                    error_code = 'timeout'
                else:
                    self._last_connection_error = output or 'Activation failed'
                    error_code = 'unknown'

            logger.warning(f"[WiFi] Failed to activate saved {name}: {self._last_connection_error}")
            return {'success': False, 'error': self._last_connection_error, 'error_code': error_code}
        except Exception as e:
            self._last_connection_error = str(e)
            logger.error(f"[WiFi] connect_saved error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            self._connecting = False

    def _get_connection_by_ssid(self, ssid: str) -> Optional[str]:
        """Get connection name for an SSID if it exists."""
        success, output = self._run_nmcli([
            '-t', '-f', 'NAME,TYPE,802-11-wireless.ssid',
            'connection', 'show'
        ])

        if success:
            for line in output.split('\n'):
                parts = line.split(':')
                if len(parts) >= 3 and parts[1] == '802-11-wireless':
                    # SSID might be in parts[2] or later
                    if ssid in line:
                        return parts[0]
        return None

    def disconnect_network(self) -> Dict[str, Any]:
        """Disconnect from current WiFi network."""
        success, output = self._run_nmcli([
            'device', 'disconnect', self._interface
        ])

        if success:
            logger.info("[WiFi] Disconnected from network")
            return {'success': True}
        else:
            return {'success': False, 'error': output}

    def forget_network(self, name: str) -> Dict[str, Any]:
        """Delete a saved network connection."""
        success, output = self._run_nmcli([
            'connection', 'delete', name
        ])

        if success:
            logger.info(f"[WiFi] Deleted connection: {name}")
            return {'success': True}
        else:
            return {'success': False, 'error': output}

    def start_ap_mode(self, auto: bool = False) -> Dict[str, Any]:
        """Start the Minus WiFi access point.

        ``auto=True`` marks the AP as monitor-started, which makes it eligible
        for automatic recovery back to a saved network (see
        ``_maybe_reconnect_from_ap``). User-initiated starts (web UI) keep the
        hotspot up until the user acts.
        """
        if self._ap_mode_active:
            return {'success': True, 'message': 'AP already active'}

        logger.info("[WiFi] Starting AP mode...")

        try:
            # Stop any existing WiFi connection first
            self._run_nmcli(['device', 'disconnect', self._interface])
            time.sleep(1)

            # Delete any existing AP connection to ensure clean state
            self._run_nmcli(['connection', 'delete', self._ap_connection_name])
            time.sleep(0.5)

            # Create hotspot using NetworkManager
            # Note: This creates a WPA2-secured hotspot by default
            # We use a simple password since open networks are harder with NM
            success, output = self._run_nmcli([
                'device', 'wifi', 'hotspot',
                'ifname', self._interface,
                'con-name', self._ap_connection_name,
                'ssid', AP_SSID,
                'band', 'bg',
                'password', 'minus123'  # Simple 8-char password for setup
            ], timeout=30)

            if success:
                self._ap_mode_active = True
                self._ap_auto_started = auto
                self._ap_last_reconnect_attempt = time.time()
                logger.info(f"[WiFi] AP mode started: SSID={AP_SSID} (auto={auto})")

                # Get AP IP for captive portal
                time.sleep(2)
                ap_ip = self._get_ip_address()

                if self._on_ap_started:
                    self._on_ap_started()

                return {
                    'success': True,
                    'ssid': AP_SSID,
                    'password': 'minus123',
                    'ip': ap_ip or AP_IP
                }
            else:
                logger.error(f"[WiFi] Failed to start AP: {output}")
                return {'success': False, 'error': output}

        except Exception as e:
            logger.error(f"[WiFi] AP start error: {e}")
            return {'success': False, 'error': str(e)}

    def stop_ap_mode(self) -> Dict[str, Any]:
        """Stop the Minus WiFi access point."""
        if not self._ap_mode_active:
            return {'success': True, 'message': 'AP not active'}

        logger.info("[WiFi] Stopping AP mode...")

        try:
            # Set flag first to prevent monitor from restarting
            self._ap_mode_active = False

            # Bring down the hotspot connection
            self._run_nmcli(['connection', 'down', self._ap_connection_name])
            time.sleep(1)

            # Delete the hotspot connection
            self._run_nmcli(['connection', 'delete', self._ap_connection_name])
            time.sleep(1)

            # Verify interface is available for client mode
            for _ in range(5):
                result = subprocess.run(
                    ['nmcli', '-t', '-f', 'DEVICE,STATE', 'device'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if self._interface in line and 'disconnected' in line:
                        logger.info("[WiFi] Interface ready for client mode")
                        break
                else:
                    time.sleep(1)
                    continue
                break

            if self._on_ap_stopped:
                self._on_ap_stopped()

            logger.info("[WiFi] AP mode stopped")
            return {'success': True}

        except Exception as e:
            logger.error(f"[WiFi] AP stop error: {e}")
            self._ap_mode_active = False  # Ensure flag is reset even on error
            return {'success': False, 'error': str(e)}

    def get_ap_status(self) -> Dict[str, Any]:
        """Get AP mode status."""
        return {
            'active': self._ap_mode_active,
            'ssid': AP_SSID if self._ap_mode_active else None,
            'password': 'minus123' if self._ap_mode_active else None,
            'ip': self._get_ip_address() if self._ap_mode_active else None,
            'clients': self._count_ap_clients() if self._ap_mode_active else 0
        }

    def start_monitor(self):
        """Start the background WiFi monitor thread."""
        if self._monitor_running:
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[WiFi] Monitor thread started")

    def stop_monitor(self):
        """Stop the background WiFi monitor thread."""
        self._monitor_running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("[WiFi] Monitor thread stopped")

    def _monitor_loop(self):
        """Background thread that monitors WiFi and auto-starts AP if needed."""
        disconnect_time = None

        while self._monitor_running:
            try:
                if self._connecting:
                    # Don't interfere while connecting
                    time.sleep(WIFI_CHECK_INTERVAL)
                    continue

                is_connected = self.is_wifi_connected()

                if is_connected:
                    # Connected - reset disconnect timer
                    disconnect_time = None
                    self._last_wifi_connected_time = time.time()

                    # If AP was active, stop it
                    if self._ap_mode_active:
                        logger.info("[WiFi] WiFi connected, stopping AP")
                        self.stop_ap_mode()
                else:
                    # Not connected
                    if not self._ap_mode_active:
                        if disconnect_time is None:
                            disconnect_time = time.time()
                            logger.info("[WiFi] WiFi disconnected, starting timer")
                        elif time.time() - disconnect_time >= WIFI_DISCONNECT_THRESHOLD:
                            logger.info(f"[WiFi] No WiFi for {WIFI_DISCONNECT_THRESHOLD}s, starting AP")
                            self._ap_reconnect_failures = 0
                            self.start_ap_mode(auto=True)
                            disconnect_time = None
                    else:
                        # In AP mode. If the monitor put us here, periodically
                        # try to get back onto a saved network — AP mode must
                        # not be a terminal state after a transient router
                        # outage.
                        self._maybe_reconnect_from_ap()

            except Exception as e:
                logger.error(f"[WiFi] Monitor error: {e}")

            time.sleep(WIFI_CHECK_INTERVAL)

    def _maybe_reconnect_from_ap(self) -> bool:
        """While a monitor-started hotspot is up, periodically try to rejoin a
        saved WiFi network.

        Root-cause fix for the Aug 2026 stranding: a 33s router blip
        ("ssid-not-found") started the hotspot, and nothing ever retried the
        saved network — the box sat off-network for ~30h until a power cycle.

        The retry is deliberately conservative:
        - only for an AP the monitor started itself; a user-initiated hotspot
          or manual disconnect is respected and left alone
        - skipped while any client is connected to the hotspot (someone is in
          the middle of captive-portal setup — don't yank it)
        - a bounce-scan first: the AP only stays down for a full connect
          attempt when a saved network is actually visible
        - exponential backoff (up to AP_RECONNECT_MAX_BACKOFF) when a saved
          network is visible but activation keeps failing (e.g. the router's
          password changed), so the AP doesn't flap every interval

        Returns True if a reconnect attempt (successful or not) was made.
        """
        if not (self._ap_mode_active and self._ap_auto_started):
            return False
        if self._connecting:
            return False

        interval = min(
            AP_RECONNECT_RETRY_INTERVAL * (2 ** min(self._ap_reconnect_failures, 3)),
            AP_RECONNECT_MAX_BACKOFF
        )
        if time.time() - self._ap_last_reconnect_attempt < interval:
            return False

        if self._count_ap_clients() > 0:
            # Someone is using the portal; re-check a full interval from now.
            self._ap_last_reconnect_attempt = time.time()
            logger.info("[WiFi] AP-mode recovery deferred: client connected to hotspot")
            return False

        self._ap_last_reconnect_attempt = time.time()
        logger.info("[WiFi] AP-mode recovery: scanning for saved networks...")
        try:
            networks = self.scan_networks(bounce_ap=True)
        except Exception as e:
            logger.warning(f"[WiFi] AP-mode recovery scan failed: {e}")
            return False

        # scan_networks sorts by signal; entries with signal=0 are saved-but-
        # not-in-range placeholders, so require a live sighting.
        visible_saved = [n for n in networks if n.saved and n.signal > 0]
        if not visible_saved:
            logger.info("[WiFi] AP-mode recovery: no saved network in range yet")
            return False

        target = visible_saved[0]
        logger.info(f"[WiFi] AP-mode recovery: '{target.ssid}' in range "
                    f"(signal {target.signal}) — attempting reconnect")
        result = self.connect_saved(target.ssid)
        if result.get('success'):
            logger.info(f"[WiFi] AP-mode recovery: reconnected to '{target.ssid}'")
            self._ap_reconnect_failures = 0
            return True

        # connect_saved tore the AP down for the attempt; restore it so the
        # box stays reachable for manual setup.
        self._ap_reconnect_failures += 1
        logger.warning(f"[WiFi] AP-mode recovery: reconnect to '{target.ssid}' failed "
                       f"({result.get('error')}); restoring AP "
                       f"(failure #{self._ap_reconnect_failures})")
        self.start_ap_mode(auto=True)
        return True

    def get_last_error(self) -> str:
        """Get the last connection error message."""
        return self._last_connection_error


# Singleton instance
_wifi_manager: Optional[WiFiManager] = None


def get_wifi_manager() -> WiFiManager:
    """Get the singleton WiFiManager instance."""
    global _wifi_manager
    if _wifi_manager is None:
        _wifi_manager = WiFiManager()
    return _wifi_manager
