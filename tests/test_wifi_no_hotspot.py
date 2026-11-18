"""
The KingBlocker9000 unit never raises its own Wi-Fi hotspot (client request,
2026-11-18): with no Wi-Fi it stays off the air and NetworkManager keeps
trying the saved networks. MINUS_WIFI_AP=1 (wifi_manager.AP_ENABLED) brings
the hotspot back; tests/test_wifi_ap_recovery.py covers that mode.

No test reaches a real nmcli call.

Run: python3 -m unittest tests.test_wifi_no_hotspot
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.wifi_manager as wifi_manager  # noqa: E402
from src.wifi_manager import WiFiManager  # noqa: E402


def make_manager():
    with patch.object(WiFiManager, '_detect_wifi_interface', return_value='wlan-test'):
        mgr = WiFiManager()
    mgr._run_nmcli = MagicMock(return_value=(True, ''))
    return mgr


def run_monitor(mgr, iterations, clock_step=40.0):
    """Run the monitor loop for `iterations` passes, advancing the clock
    by clock_step seconds per pass (past WIFI_DISCONNECT_THRESHOLD)."""
    clock = [1000.0]
    passes = [0]

    def fake_sleep(_):
        passes[0] += 1
        clock[0] += clock_step
        if passes[0] >= iterations:
            mgr._monitor_running = False

    mgr._monitor_running = True
    with patch('src.wifi_manager.time.sleep', side_effect=fake_sleep), \
            patch('src.wifi_manager.time.time', side_effect=lambda: clock[0]):
        mgr._monitor_loop()


class TestNoHotspot(unittest.TestCase):

    def setUp(self):
        self._saved = wifi_manager.AP_ENABLED
        wifi_manager.AP_ENABLED = False

    def tearDown(self):
        wifi_manager.AP_ENABLED = self._saved

    def test_off_by_default(self):
        import importlib
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('MINUS_WIFI_AP', None)
            mod = importlib.reload(wifi_manager)
            try:
                self.assertFalse(mod.AP_ENABLED)
            finally:
                importlib.reload(wifi_manager)
                wifi_manager.AP_ENABLED = False

    def test_start_ap_mode_refuses(self):
        mgr = make_manager()
        result = mgr.start_ap_mode()
        self.assertFalse(result['success'])
        self.assertFalse(mgr._ap_mode_active)
        mgr._run_nmcli.assert_not_called()

    def test_monitor_never_starts_hotspot_without_wifi(self):
        mgr = make_manager()
        mgr.is_wifi_connected = MagicMock(return_value=False)
        mgr.start_ap_mode = MagicMock()
        run_monitor(mgr, iterations=6)   # 240 s without Wi-Fi
        mgr.start_ap_mode.assert_not_called()

    def test_leftover_hotspot_is_stopped(self):
        mgr = make_manager()
        mgr._ap_mode_active = True
        mgr.is_wifi_connected = MagicMock(return_value=False)
        mgr.stop_ap_mode = MagicMock(side_effect=lambda: setattr(mgr, '_ap_mode_active', False))
        mgr._maybe_reconnect_from_ap = MagicMock()
        run_monitor(mgr, iterations=2)
        mgr.stop_ap_mode.assert_called_once()
        mgr._maybe_reconnect_from_ap.assert_not_called()

    def test_enabled_still_starts_hotspot(self):
        wifi_manager.AP_ENABLED = True
        mgr = make_manager()
        mgr.is_wifi_connected = MagicMock(return_value=False)
        mgr.start_ap_mode = MagicMock()
        run_monitor(mgr, iterations=3)
        mgr.start_ap_mode.assert_called_once_with(auto=True)


class TestProfileDefaults(unittest.TestCase):
    """Every saved Wi-Fi profile retries forever and has backup DNS."""

    def _mgr(self, profiles):
        mgr = make_manager()
        mgr._ap_connection_name = 'KingBlocker9000'
        calls = []

        def nmcli(args, timeout=30):
            calls.append(args)
            if args[:3] == ['-t', '-f', 'UUID,TYPE,NAME,DEVICE']:
                return True, ''.join(f"{u}:{t}:{n}:{d}\n" for u, t, n, d, _, _ in profiles)
            if args[:1] == ['-g']:
                p = next(p for p in profiles if p[0] == args[-1])
                return True, f"{p[4]}\n{p[5]}"
            return True, ''
        mgr._run_nmcli = nmcli
        return mgr, calls

    def test_sets_retries_and_dns_where_missing(self):
        mgr, calls = self._mgr([
            ('u-home', '802-11-wireless', 'Home', 'wlan0', '-1', ''),        # needs both
            ('u-test', '802-11-wireless', 'TestNetwork1337', '', '0', '1.1.1.1,9.9.9.9'),  # done
            ('u-ap', '802-11-wireless', 'KingBlocker9000', '', '-1', ''),    # AP: skipped
            ('u-ts', 'tun', 'tailscale0', 'tailscale0', '-1', ''),           # not Wi-Fi
        ])
        self.assertEqual(mgr.ensure_profile_defaults(), 1)
        mods = [a for a in calls if a[:2] == ['connection', 'modify']]
        self.assertEqual(len(mods), 1)
        self.assertEqual(mods[0][2], 'u-home')
        self.assertIn('connection.autoconnect-retries', mods[0])
        self.assertIn('ipv4.dns', mods[0])
        self.assertIn(['device', 'reapply', 'wlan0'], calls)   # active: applied now

    def test_existing_dns_is_kept(self):
        mgr, calls = self._mgr([('u', '802-11-wireless', 'Office', '', '-1', '10.0.0.53')])
        mgr.ensure_profile_defaults()
        mod = next(a for a in calls if a[:2] == ['connection', 'modify'])
        self.assertNotIn('ipv4.dns', mod)                       # their own DNS untouched
        self.assertIn('connection.autoconnect-retries', mod)

    def test_old_name_still_works(self):
        mgr, _ = self._mgr([])
        self.assertEqual(mgr.ensure_retry_forever(), 0)

    def test_runs_when_monitor_starts(self):
        mgr = make_manager()
        mgr.ensure_profile_defaults = MagicMock(return_value=0)
        mgr.is_wifi_connected = MagicMock(return_value=True)
        run_monitor(mgr, iterations=1)
        mgr.ensure_profile_defaults.assert_called_once()


class TestInterfaceDetection(unittest.TestCase):

    def test_p2p_device_is_not_the_radio(self):
        # With the radio off, nmcli lists the P2P device first; it used to be
        # picked because ':wifi' is a substring of ':wifi-p2p'.
        out = 'p2p-dev-wlP2p33s0:wifi-p2p\nwlP2p33s0:wifi\nlo:loopback\n'
        with patch('src.wifi_manager.subprocess.run',
                   return_value=MagicMock(stdout=out, returncode=0)):
            mgr = WiFiManager()
        self.assertEqual(mgr._interface, 'wlP2p33s0')


if __name__ == '__main__':
    unittest.main()
