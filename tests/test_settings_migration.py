"""
Settings v2 migration: every bare "James" is muted (2026-11-18). A settings
file saved while name_mute_surname defaulted to off holds an explicit False,
which would win over the new default; v2 turns it on once and drops the
obsolete name_mute_james_context key. A deliberate "off" saved as v2 stays.

Imports minus.py, so it runs on a unit (GStreamer etc. installed).
Run: python3 -m unittest tests.test_settings_migration
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import minus  # noqa: E402


class TestSettingsMigration(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'settings.json'
        self._orig = minus.SYSTEM_SETTINGS_FILE
        minus.SYSTEM_SETTINGS_FILE = self.path

    def tearDown(self):
        minus.SYSTEM_SETTINGS_FILE = self._orig
        self.dir.cleanup()

    def load(self):
        return minus.Minus._load_system_settings(object())

    def test_fresh_unit_has_every_james_on(self):
        s = self.load()
        self.assertTrue(s['name_mute_surname'])
        self.assertEqual(s['settings_version'], 2)

    def test_old_saved_false_is_migrated_on_and_saved(self):
        self.path.write_text(json.dumps({'name_mute': True, 'name_mute_surname': False,
                                         'name_mute_james_context': True,
                                         'debug_overlay': False}))
        s = self.load()
        self.assertTrue(s['name_mute_surname'])
        self.assertNotIn('name_mute_james_context', s)
        self.assertFalse(s['debug_overlay'])          # other saved values kept
        on_disk = json.loads(self.path.read_text())
        self.assertTrue(on_disk['name_mute_surname'])
        self.assertEqual(on_disk['settings_version'], 2)

    def test_deliberate_off_after_migration_is_kept(self):
        self.path.write_text(json.dumps({'name_mute_surname': False, 'settings_version': 2}))
        self.assertFalse(self.load()['name_mute_surname'])


if __name__ == '__main__':
    unittest.main()
