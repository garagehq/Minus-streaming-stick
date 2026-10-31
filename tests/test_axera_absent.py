"""
VLM with no AX-M1 card on PCIe.

The ax_pcie_dev_host driver loads at boot with or without the card, and the
axcl runtime then fails every model load. Minus used to retry the load three
times after each start (~25 s) with the webui showing the VLM as loading.
It now checks for a bound card first and reports the VLM as off.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


class TestAxeraCardPresent(unittest.TestCase):

    def _driver_dir(self, entries):
        d = tempfile.mkdtemp()
        for name in entries:
            (Path(d) / name).touch()
        return d

    def test_driver_without_card(self):
        from minus import axera_card_present
        d = self._driver_dir(['bind', 'module', 'new_id', 'remove_id', 'uevent', 'unbind'])
        self.assertFalse(axera_card_present(d))

    def test_driver_with_bound_card(self):
        from minus import axera_card_present
        d = self._driver_dir(['bind', 'module', 'uevent', 'unbind', '0001:11:00.0'])
        self.assertTrue(axera_card_present(d))

    def test_no_driver(self):
        from minus import axera_card_present
        self.assertFalse(axera_card_present('/nonexistent/ax_pcie_dev_host'))


class TestVlmOffInWebui(unittest.TestCase):

    def test_badge_and_settings_say_off(self):
        html = (ROOT / 'src' / 'templates' / 'index.html').read_text()
        self.assertIn("vlmStatus.textContent = 'VLM: Off'", html)
        self.assertNotIn("'Not initialized'", html)


if __name__ == '__main__':
    unittest.main()
