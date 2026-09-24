#!/usr/bin/env python3
"""Disk usage on the dashboard, alongside Mem and Temp.

The health monitor already tracked free space and warned below 500 MB, but the
number never reached the UI -- the first sign of a full disk would have been
screenshot writes failing. Screenshots accumulate on this box continuously, so
it is a real failure mode rather than a curiosity.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from health import HealthMonitor, HealthStatus  # noqa: E402


class TestDiskUsedPercent(unittest.TestCase):

    def _stat(self, blocks, bfree, bavail, frsize=4096):
        st = MagicMock()
        st.f_blocks, st.f_bfree, st.f_bavail, st.f_frsize = blocks, bfree, bavail, frsize
        return st

    def _pct(self, blocks, bfree, bavail):
        h = object.__new__(HealthMonitor)
        with patch('health.os.statvfs', return_value=self._stat(blocks, bfree, bavail)):
            return HealthMonitor._get_disk_used_percent(h)

    def test_half_full(self):
        self.assertAlmostEqual(self._pct(1000, 500, 500), 50.0, places=1)

    def test_empty_filesystem(self):
        self.assertAlmostEqual(self._pct(1000, 1000, 1000), 0.0, places=1)

    def test_full_filesystem(self):
        self.assertAlmostEqual(self._pct(1000, 0, 0), 100.0, places=1)

    def test_root_reserve_is_not_counted_as_headroom(self):
        """bfree > bavail: the gap is root-only reserve we cannot spend.

        Measuring against the raw total would report comfortable free space
        while ordinary writes were already failing.
        """
        pct = self._pct(1000, 100, 50)   # 900 used, only 50 usable
        self.assertAlmostEqual(pct, 900 / 950 * 100, places=1)
        self.assertGreater(pct, 90.0)

    def test_unreadable_filesystem_is_zero_not_a_crash(self):
        h = object.__new__(HealthMonitor)
        with patch('health.os.statvfs', side_effect=OSError('gone')):
            self.assertEqual(HealthMonitor._get_disk_used_percent(h), 0.0)

    def test_matches_the_real_filesystem(self):
        """Sanity-check the arithmetic against the live mount."""
        h = object.__new__(HealthMonitor)
        pct = HealthMonitor._get_disk_used_percent(h)
        s = os.statvfs('.')
        avail = s.f_bavail * s.f_frsize
        used = (s.f_blocks - s.f_bfree) * s.f_frsize
        self.assertAlmostEqual(pct, used / (used + avail) * 100, places=1)
        self.assertTrue(0.0 <= pct <= 100.0)


class TestStatusFieldPlumbing(unittest.TestCase):

    def test_health_status_carries_the_field(self):
        self.assertTrue(hasattr(HealthStatus(), 'disk_used_percent'))
        self.assertTrue(hasattr(HealthStatus(), 'disk_free_mb'))

    def test_status_dict_exposes_both(self):
        src = (ROOT / 'minus.py').read_text()
        self.assertIn("'disk_used_percent'", src)
        self.assertIn("'disk_free_mb'", src)

    def test_check_populates_the_field(self):
        src = (ROOT / 'src' / 'health.py').read_text()
        self.assertIn('status.disk_used_percent = self._get_disk_used_percent()', src)


class TestDashboardBadge(unittest.TestCase):

    def setUp(self):
        self.html = (ROOT / 'src' / 'templates' / 'index.html').read_text()

    def test_badge_exists_beside_the_others(self):
        self.assertIn('id="disk-status"', self.html)
        info = self.html[self.html.index('id="memory-status"'):]
        info = info[:info.index('</div>')]
        self.assertIn('id="disk-status"', info,
                      'disk badge must sit in the same system-info row')

    def test_badge_is_updated_from_status(self):
        self.assertIn("getElementById('disk-status')", self.html)
        self.assertIn('status.disk_used_percent', self.html)

    def test_badge_uses_styles_that_exist(self):
        css = (ROOT / 'src' / 'static' / 'style.css').read_text()
        for cls in ('.system-badge.ok', '.system-badge.warning', '.system-badge.error'):
            self.assertIn(cls, css, f'{cls} is referenced by the badge but undefined')


if __name__ == "__main__":
    unittest.main(verbosity=2)
