"""Loading system settings keeps every saved key (port of main f97b0d5).

_load_system_settings used to copy only keys it had defaults for, so anything
else in the settings file was dropped at startup and then overwritten by the
next save.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import minus  # noqa: E402


class TestLoadKeepsKeys(unittest.TestCase):
    def _load(self, saved):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'settings.json'
            p.write_text(json.dumps(saved))
            with patch.object(minus, 'SYSTEM_SETTINGS_FILE', p):
                return minus.Minus._load_system_settings(object.__new__(minus.Minus))

    def test_unknown_saved_key_survives(self):
        s = self._load({'name_mute': False, 'some_newer_key': {'a': 1}})
        self.assertEqual(s['some_newer_key'], {'a': 1})
        self.assertFalse(s['name_mute'])

    def test_missing_keys_get_defaults(self):
        s = self._load({})
        self.assertTrue(s['name_mute'])
        self.assertFalse(s['ad_blocking'])


if __name__ == '__main__':
    unittest.main()
