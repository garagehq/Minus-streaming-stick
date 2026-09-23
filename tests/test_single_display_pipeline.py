#!/usr/bin/env python3
"""There must only ever be one display pipeline.

When the TV came on, the health monitor's reconnect restart and the display
retry loop rebuilt the pipeline in the same second. The loser was overwritten
in self.pipeline while still PLAYING, and nothing referenced it again, so
nothing ever stopped it. ustreamer showed two clients each pulling 61fps.

Two costs: both decoded 4K at 60fps (double VPU/CPU on a SoC at its thermal
trip), and the thermal frame gate keeps its token bucket on the blocker rather
than per pipeline, so in degraded mode they split the 30fps budget and the
screen showed ~15fps.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import ad_blocker as ab  # noqa: E402
from ad_blocker import DRMAdBlocker  # noqa: E402


def blocker():
    b = object.__new__(DRMAdBlocker)
    b.pipeline = None
    b.bus = None
    b._build_lock = threading.RLock()
    b._framegate_credit = 1.7
    b._framegate_last_ts = 123.0
    built = []

    def fake_build():
        p = MagicMock(name=f'pipeline{len(built)}')
        built.append(p)
        b.pipeline = p
        b.bus = MagicMock()
        return True
    b._build_pipeline = fake_build
    b._built = built
    return b


class TestSinglePipelineInvariant(unittest.TestCase):

    def test_first_build_has_nothing_to_tear_down(self):
        b = blocker()
        b._init_pipeline()
        self.assertEqual(len(b._built), 1)
        b._built[0].set_state.assert_not_called()

    def test_rebuild_stops_the_live_pipeline_first(self):
        b = blocker()
        b._init_pipeline()
        old = b.pipeline
        old_bus = b.bus
        b._init_pipeline()
        old.set_state.assert_called_with(ab.Gst.State.NULL)
        old_bus.remove_signal_watch.assert_called_once()
        self.assertIsNot(b.pipeline, old)

    def test_concurrent_builders_leave_exactly_one_live(self):
        """The production race: two recovery paths building at once."""
        b = blocker()
        barrier = threading.Barrier(8)

        def go():
            barrier.wait()
            b._init_pipeline()
        ts = [threading.Thread(target=go) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        stopped = [p for p in b._built
                   if any(c.args == (ab.Gst.State.NULL,) for c in p.set_state.call_args_list)]
        live = [p for p in b._built if p not in stopped]
        self.assertEqual(len(b._built), 8)
        self.assertEqual(live, [b.pipeline],
                         'every pipeline except the current one must be stopped')

    def test_new_pipeline_gets_a_fresh_frame_budget(self):
        b = blocker()
        b._init_pipeline()
        self.assertEqual(b._framegate_credit, 0.0)
        self.assertIsNone(b._framegate_last_ts)

    def test_teardown_survives_a_failing_pipeline(self):
        b = blocker()
        b._init_pipeline()
        b.pipeline.set_state.side_effect = RuntimeError('gst')
        b._init_pipeline()
        self.assertEqual(len(b._built), 2)


class TestCpuPercent(unittest.TestCase):

    def _hm(self):
        from health import HealthMonitor
        return object.__new__(HealthMonitor)

    def _read(self, hm, line):
        from unittest.mock import mock_open, patch
        with patch('builtins.open', mock_open(read_data=line + '\n')):
            return hm._get_cpu_percent()

    def test_first_read_is_baseline_only(self):
        hm = self._hm()
        self.assertEqual(self._read(hm, 'cpu  100 0 100 800 0 0 0 0 0 0'), 0.0)

    def test_delta_between_reads(self):
        hm = self._hm()
        self._read(hm, 'cpu  100 0 100 800 0 0 0 0 0 0')
        # +300 busy, +100 idle -> 75% busy
        self.assertAlmostEqual(self._read(hm, 'cpu  300 0 200 900 0 0 0 0 0 0'), 75.0)

    def test_iowait_counts_as_idle(self):
        hm = self._hm()
        self._read(hm, 'cpu  0 0 0 0 0 0 0 0 0 0')
        self.assertAlmostEqual(self._read(hm, 'cpu  50 0 0 0 50 0 0 0 0 0'), 50.0)

    def test_live_value_is_a_percentage(self):
        hm = self._hm()
        hm._get_cpu_percent()
        import time; time.sleep(0.2)
        v = hm._get_cpu_percent()
        self.assertTrue(0.0 <= v <= 100.0)

    def test_badge_wired(self):
        html = (ROOT / 'src' / 'templates' / 'index.html').read_text()
        self.assertIn('id="cpu-status"', html)
        self.assertIn('status.cpu_percent', html)
        self.assertIn("'cpu_percent'", (ROOT / 'minus.py').read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
