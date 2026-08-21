#!/usr/bin/env python3
"""Unit tests for WiFi AP-mode auto-recovery (_maybe_reconnect_from_ap).

Root-cause regression tests for the Aug 2026 stranding: a 33s router blip
started the "Minus" hotspot and nothing ever retried the saved network, so
the box sat off-network (no LAN, no Tailscale) for ~30h until a power cycle.

All hardware/nmcli interaction is mocked — these tests never touch the real
radio (live AP tests are forbidden: they sever remote access to the box).
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wifi_manager import WiFiManager, WiFiNetwork


def make_manager():
    with patch.object(WiFiManager, '_detect_wifi_interface', return_value='wlan-test'):
        mgr = WiFiManager()
    # Never let a test reach a real nmcli call
    mgr._run_nmcli = MagicMock(return_value=(True, ''))
    return mgr


def saved_net(ssid='HomeNet', signal=70):
    return WiFiNetwork(ssid=ssid, signal=signal, security='WPA2', saved=True)


class TestAPModeRecovery(unittest.TestCase):

    def setUp(self):
        self.mgr = make_manager()
        # Simulate: monitor started the AP a while ago, retry interval elapsed
        self.mgr._ap_mode_active = True
        self.mgr._ap_auto_started = True
        self.mgr._ap_last_reconnect_attempt = 0.0
        self.mgr._count_ap_clients = MagicMock(return_value=0)
        self.mgr.scan_networks = MagicMock(return_value=[saved_net()])
        self.mgr.connect_saved = MagicMock(return_value={'success': True, 'ssid': 'HomeNet'})
        self.mgr.start_ap_mode = MagicMock(return_value={'success': True})

    def test_reconnects_when_saved_network_visible(self):
        attempted = self.mgr._maybe_reconnect_from_ap()
        self.assertTrue(attempted)
        self.mgr.scan_networks.assert_called_once_with(bounce_ap=True)
        self.mgr.connect_saved.assert_called_once_with('HomeNet')
        self.assertEqual(self.mgr._ap_reconnect_failures, 0)
        # AP must NOT be restarted after a successful reconnect
        self.mgr.start_ap_mode.assert_not_called()

    def test_manual_ap_never_auto_recovers(self):
        self.mgr._ap_auto_started = False
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.scan_networks.assert_not_called()
        self.mgr.connect_saved.assert_not_called()

    def test_no_attempt_before_interval_elapsed(self):
        self.mgr._ap_last_reconnect_attempt = time.time()
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.scan_networks.assert_not_called()

    def test_deferred_while_portal_client_connected(self):
        self.mgr._count_ap_clients = MagicMock(return_value=1)
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.scan_networks.assert_not_called()
        # Timer bumped so the next check waits a full interval
        self.assertGreater(self.mgr._ap_last_reconnect_attempt, time.time() - 5)

    def test_no_attempt_while_connecting(self):
        self.mgr._connecting = True
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.scan_networks.assert_not_called()

    def test_saved_but_out_of_range_placeholder_ignored(self):
        # signal=0 entries are saved-but-not-seen placeholders from
        # scan_networks — they must not trigger a connect attempt.
        self.mgr.scan_networks = MagicMock(return_value=[saved_net(signal=0)])
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.connect_saved.assert_not_called()
        self.assertEqual(self.mgr._ap_reconnect_failures, 0)

    def test_unsaved_networks_ignored(self):
        self.mgr.scan_networks = MagicMock(return_value=[
            WiFiNetwork(ssid='NeighborNet', signal=90, security='WPA2', saved=False)
        ])
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.connect_saved.assert_not_called()

    def test_failed_attempt_restores_ap_and_backs_off(self):
        self.mgr.connect_saved = MagicMock(
            return_value={'success': False, 'error': 'Incorrect password'})
        attempted = self.mgr._maybe_reconnect_from_ap()
        self.assertTrue(attempted)
        # AP restored so the box stays reachable
        self.mgr.start_ap_mode.assert_called_once_with(auto=True)
        self.assertEqual(self.mgr._ap_reconnect_failures, 1)
        # Backoff: with 1 failure the next attempt needs 2x the base interval,
        # so an immediate re-check must not attempt again.
        self.mgr._ap_last_reconnect_attempt = time.time() - 150  # > base 120s
        self.mgr.scan_networks.reset_mock()
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.scan_networks.assert_not_called()

    def test_scan_exception_keeps_ap_and_returns_false(self):
        self.mgr.scan_networks = MagicMock(side_effect=RuntimeError('radio busy'))
        self.assertFalse(self.mgr._maybe_reconnect_from_ap())
        self.mgr.connect_saved.assert_not_called()

    def test_strongest_saved_network_preferred(self):
        self.mgr.scan_networks = MagicMock(return_value=[
            saved_net('StrongNet', 80), saved_net('WeakNet', 30)
        ])
        self.mgr._maybe_reconnect_from_ap()
        self.mgr.connect_saved.assert_called_once_with('StrongNet')


class TestStartAPModeAutoFlag(unittest.TestCase):

    def setUp(self):
        self.mgr = make_manager()
        self.mgr._get_ip_address = MagicMock(return_value='10.42.0.1')
        self.mgr._count_ap_clients = MagicMock(return_value=0)

    def test_monitor_start_marks_auto(self):
        with patch('src.wifi_manager.time.sleep'):
            result = self.mgr.start_ap_mode(auto=True)
        self.assertTrue(result['success'])
        self.assertTrue(self.mgr._ap_auto_started)

    def test_manual_start_marks_not_auto(self):
        with patch('src.wifi_manager.time.sleep'):
            result = self.mgr.start_ap_mode()
        self.assertTrue(result['success'])
        self.assertFalse(self.mgr._ap_auto_started)

    def test_start_sets_retry_clock(self):
        before = time.time()
        with patch('src.wifi_manager.time.sleep'):
            self.mgr.start_ap_mode(auto=True)
        self.assertGreaterEqual(self.mgr._ap_last_reconnect_attempt, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
