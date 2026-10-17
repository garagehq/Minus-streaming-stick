"""A throwaway Minus web UI for the Wi-Fi portal tests — it never touches Wi-Fi.

Why this exists: tests/test_wifi_portal.py used to drive http://localhost:80.
On a dev machine nothing answers there; on a Minus box it is the LIVE web UI,
running as root with the real radio. On 2026-10-15 a full test run on a box
made it save and join a neighbour's open smart-plug network (no internet, set
to auto-connect) plus a junk WPA profile, and the box stayed off the network
until someone went to it with an ethernet cable.

This server runs the real `WebUI` routes and the real `wifi_setup.html` on a
random 127.0.0.1 port, with:
  - a fake Wi-Fi manager (fixed scan results, connect attempts recorded, no
    nmcli, no NetworkManager, no radio), installed as the singleton that every
    `/api/wifi/*` route asks for;
  - an allow-list in front of the app: only the portal page, its assets, the
    Wi-Fi endpoints the page uses and the captive-portal probes are served;
    everything else is 403, so nothing else in the web UI can run from a test.
"""
import threading
import time

from werkzeug.serving import make_server

from src import wifi_manager as wm

FAKE_NETWORKS = [
    # A saved network the box is on, two secured strangers and an open one —
    # the shapes the portal has to render (signal bars, lock, "saved").
    wm.WiFiNetwork(ssid='TestHome', signal=82, security='WPA2', in_use=True, saved=True),
    wm.WiFiNetwork(ssid='Neighbour 5G', signal=64, security='WPA2'),
    wm.WiFiNetwork(ssid='CoffeeShop', signal=41, security='WPA2'),
    wm.WiFiNetwork(ssid='Open Setup AP', signal=33, security='Open'),
]

ALLOWED_PATHS = {
    '/wifi-setup', '/favicon.ico',
    '/api/wifi/status', '/api/wifi/scan', '/api/wifi/connect', '/api/wifi/connect-saved',
    '/generate_204', '/hotspot-detect.html', '/connecttest.txt', '/ncsi.txt', '/success.txt',
}


class FakeWiFiManager:
    """Stands in for WiFiManager. Records what the UI asked for; changes nothing."""

    _ap_connection_name = 'Hotspot'

    def __init__(self, scan_delay_s: float = 1.0):
        self._ap_mode_active = False
        self.scan_delay_s = scan_delay_s
        self.connect_attempts = []   # (ssid, password)

    def get_status(self):
        return wm.WiFiStatus(connected=True, ssid='TestHome', ip_address='192.0.2.10', signal=82)

    def scan_networks(self, bounce_ap: bool = False):
        time.sleep(self.scan_delay_s)   # the page shows a "Scanning" state meanwhile
        return list(FAKE_NETWORKS)

    def connect_to_network(self, ssid: str, password: str = ''):
        self.connect_attempts.append((ssid, password))
        return {'success': False, 'error': 'Incorrect password'}

    def connect_saved(self, name: str):
        self.connect_attempts.append((name, None))
        return {'success': True, 'ssid': name}

    def __getattr__(self, name):
        # Any other WiFiManager method (disconnect, start_ap_mode, forget, ...)
        # must never be reached from the portal tests.
        raise AssertionError(f'portal test server: WiFiManager.{name} is not allowed in tests')


class _NoMinus:
    """The web UI only needs a Minus instance for routes this server never serves."""

    def __getattr__(self, name):
        raise AttributeError(name)


class PortalTestServer:
    """Start with `.start()`, use `.base_url`, stop with `.stop()`."""

    def __init__(self, scan_delay_s: float = 1.0):
        self.wifi = FakeWiFiManager(scan_delay_s=scan_delay_s)
        self._saved_singleton = None
        self._server = None
        self._thread = None
        self.base_url = None

    def start(self):
        from flask import abort, request
        from src import webui

        self._saved_singleton = wm._wifi_manager
        wm._wifi_manager = self.wifi            # what get_wifi_manager() returns
        app = webui.WebUI(_NoMinus()).app

        @app.before_request
        def _allow_list():
            if request.path not in ALLOWED_PATHS and not request.path.startswith('/static/'):
                abort(403)

        self._server = make_server('127.0.0.1', 0, app, threaded=True)
        self.base_url = f'http://127.0.0.1:{self._server.server_port}'
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        wm._wifi_manager = self._saved_singleton
