#!/usr/bin/env python3
"""Tests for /api/debug/memory and the health memory subsystem."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


class TestDebugMemoryEndpoint(unittest.TestCase):
    def _client(self):
        from webui import WebUI
        m = MagicMock()
        m.health_monitor = MagicMock()
        m.health_monitor.get_process_memory.return_value = {
            'rss_mb': 400.0, 'pss_anon_mb': 300.0, 'system_percent': 12.0}
        ui = WebUI(m)
        ui.app.config['TESTING'] = True
        return ui.app.test_client()

    def test_basic_shape_without_census(self):
        c = self._client()
        r = c.get('/api/debug/memory')
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        for k in ('process', 'threads', 'fds', 'gc', 'tracemalloc'):
            self.assertIn(k, d)
        self.assertEqual(d['process']['rss_mb'], 400.0)
        self.assertNotIn('top_types', d)
        self.assertIn('enabled', d['tracemalloc'])

    def test_census_returns_top_types(self):
        c = self._client()
        d = c.get('/api/debug/memory?objects=1&top=5').get_json()
        self.assertIn('top_types', d)
        self.assertLessEqual(len(d['top_types']), 5)
        self.assertGreater(d['object_total'], 0)

    def test_tracemalloc_sections_when_tracing(self):
        import tracemalloc
        c = self._client()
        tracemalloc.start(3)
        try:
            d0 = c.get('/api/debug/memory?objects=0&top=3').get_json()
            self.assertTrue(d0['tracemalloc']['enabled'])
            self.assertNotIn('top', d0['tracemalloc'])  # snapshots are opt-in
            d = c.get('/api/debug/memory?objects=0&top=3&snapshot=1').get_json()
            self.assertIn('top', d['tracemalloc'])
            self.assertTrue(d['tracemalloc'].get('baseline_stored'))
            d2 = c.get('/api/debug/memory?objects=0&top=3&snapshot=1&diff=1').get_json()
            self.assertIn('diff_since_baseline', d2['tracemalloc'])
        finally:
            tracemalloc.stop()


if __name__ == '__main__':
    unittest.main(verbosity=2)
