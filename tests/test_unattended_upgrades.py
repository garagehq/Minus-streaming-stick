#!/usr/bin/env python3
"""Tests for src/unattended_upgrades.py — all filesystem/subprocess mocked."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import unattended_upgrades as uu


class TestConfigFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.auto = Path(self.tmp) / '20auto-upgrades'
        self.policy = Path(self.tmp) / '52minus'
        self._p1 = patch.object(uu, 'AUTO_UPGRADES_FILE', self.auto)
        self._p2 = patch.object(uu, 'POLICY_FILE', self.policy)
        self._p1.start(); self._p2.start()

    def tearDown(self):
        self._p1.stop(); self._p2.stop()

    def test_is_enabled_parses_switch(self):
        self.auto.write_text('APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n')
        self.assertTrue(uu.is_enabled())
        self.auto.write_text('APT::Periodic::Unattended-Upgrade "0";\n')
        self.assertFalse(uu.is_enabled())

    def test_is_enabled_missing_file_is_false(self):
        self.assertFalse(uu.is_enabled())

    def test_set_enabled_requires_root(self):
        with patch.object(uu, 'is_root', return_value=False):
            r = uu.set_enabled(True)
        self.assertFalse(r['success'])
        self.assertIn('manual_command', r)
        self.assertFalse(self.auto.exists())

    def test_set_enabled_requires_package(self):
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=False):
            r = uu.set_enabled(True)
        self.assertFalse(r['success'])
        self.assertEqual(r['manual_command'], uu.INSTALL_COMMAND)

    def test_enable_writes_switch_and_policy(self):
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=True):
            r = uu.set_enabled(True)
        self.assertTrue(r['success'])
        self.assertTrue(uu.is_enabled())
        self.assertTrue(uu.policy_present())
        text = self.policy.read_text()
        self.assertIn('Automatic-Reboot "false"', text)
        for must in ('linux-image-.*', 'u-boot-.*', 'librockchip-.*', 'gstreamer1.0-.*',
                     'mesa-.*', 'rknpu.*', 'tailscale'):
            self.assertIn(f'"{must}";', text)

    def test_disable_flips_switch_off(self):
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=True):
            uu.set_enabled(True)
            r = uu.set_enabled(False)
        self.assertTrue(r['success'])
        self.assertFalse(uu.is_enabled())
        self.assertIn('"0"', self.auto.read_text())

    def test_disable_does_not_need_package(self):
        """Turning it off must work even if the package was removed."""
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=False):
            self.assertTrue(uu.set_enabled(False)['success'])

    def test_policy_header_marks_ownership(self):
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=True):
            uu.set_enabled(True)
        self.assertTrue(self.policy.read_text().startswith(uu.POLICY_HEADER))
        self.policy.write_text('// someone else\n')
        self.assertFalse(uu.policy_present())


class TestLogParsing(unittest.TestCase):
    def _log(self, text):
        f = Path(tempfile.mkdtemp()) / 'u.log'
        f.write_text(text)
        return f

    def test_last_run_with_result(self):
        f = self._log(
            "2026-09-05 06:55:01,000 INFO Starting unattended upgrades script\n"
            "2026-09-05 06:55:01,100 INFO Allowed origins are: o=Debian,a=bookworm-security\n"
            "2026-09-05 06:55:03,000 INFO No packages found that can be upgraded unattended and no pending auto-removals\n"
            "2026-09-06 06:54:00,000 INFO Starting unattended upgrades script\n"
            "2026-09-06 06:54:02,000 INFO Packages that will be upgraded: libssl3 openssl\n"
            "2026-09-06 06:54:40,000 INFO All upgrades installed\n")
        ts, res = uu.last_run(f)
        self.assertEqual(ts, '2026-09-06 06:54:00,000')
        self.assertEqual(res, 'All upgrades installed')

    def test_last_run_missing_log(self):
        self.assertEqual(uu.last_run(Path('/nonexistent/x.log')), (None, None))

    def test_last_run_in_progress_has_no_result(self):
        f = self._log("2026-09-06 06:54:00,000 INFO Starting unattended upgrades script\n")
        ts, res = uu.last_run(f)
        self.assertEqual(ts, '2026-09-06 06:54:00,000')
        self.assertIsNone(res)


class TestStatus(unittest.TestCase):
    def test_status_keys_when_not_installed(self):
        with patch.object(uu, 'is_installed', return_value=False), \
             patch.object(uu, 'reboot_required', return_value=False):
            st = uu.status()
        for k in ('installed', 'enabled', 'policy_present', 'last_run_time',
                  'last_run_result', 'next_run', 'reboot_required', 'is_root',
                  'install_command', 'blacklist'):
            self.assertIn(k, st)
        self.assertFalse(st['installed'])
        self.assertFalse(st['enabled'])
        self.assertIsNone(st['next_run'])

    def test_next_run_na_is_none(self):
        m = MagicMock(); m.stdout = 'n/a\n'
        with patch.object(uu.subprocess, 'run', return_value=m):
            self.assertIsNone(uu.next_run())

    def test_is_installed_parses_dpkg(self):
        m = MagicMock(); m.returncode = 0; m.stdout = 'install ok installed'
        with patch.object(uu.subprocess, 'run', return_value=m):
            self.assertTrue(uu.is_installed())
        m.stdout = 'deinstall ok config-files'
        with patch.object(uu.subprocess, 'run', return_value=m):
            self.assertFalse(uu.is_installed())


class TestMinusWiring(unittest.TestCase):
    def _minus(self, setting=True):
        from minus import Minus
        m = Minus.__new__(Minus)
        m._system_settings = {'unattended_upgrades': setting}
        m._save_system_settings = MagicMock()
        return m

    def test_reconcile_enables_when_setting_on(self):
        m = self._minus(True)
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=True), \
             patch.object(uu, 'is_enabled', return_value=False), \
             patch.object(uu, 'policy_present', return_value=False), \
             patch.object(uu, 'set_enabled', return_value={'success': True, 'enabled': True}) as se, \
             patch.object(uu, 'status', return_value={}):
            r = m._apply_unattended_upgrades_setting()
        se.assert_called_once_with(True)
        self.assertTrue(r['success'])

    def test_reconcile_noop_when_already_matching(self):
        m = self._minus(True)
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=True), \
             patch.object(uu, 'is_enabled', return_value=True), \
             patch.object(uu, 'policy_present', return_value=True), \
             patch.object(uu, 'set_enabled') as se, \
             patch.object(uu, 'status', return_value={}):
            m._apply_unattended_upgrades_setting()
        se.assert_not_called()

    def test_reconcile_never_raises_when_not_root(self):
        m = self._minus(True)
        with patch.object(uu, 'is_root', return_value=False), \
             patch.object(uu, 'status', return_value={}):
            r = m._apply_unattended_upgrades_setting()
        self.assertFalse(r['success'])

    def test_set_persists_then_applies(self):
        m = self._minus(True)
        with patch.object(uu, 'is_root', return_value=True), \
             patch.object(uu, 'is_installed', return_value=True), \
             patch.object(uu, 'is_enabled', return_value=True), \
             patch.object(uu, 'set_enabled', return_value={'success': True, 'enabled': False}) as se, \
             patch.object(uu, 'status', return_value={}):
            r = m.set_unattended_upgrades(False)
        self.assertFalse(m._system_settings['unattended_upgrades'])
        m._save_system_settings.assert_called_once()
        se.assert_called_once_with(False)
        self.assertTrue(r['success'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
