"""
ASR must come back on its own when its worker fails to load or dies.

Found live on kingblocker9000 (2026-11-18): the unit has no RTC battery,
so it boots with a stale wall clock and NTP steps it forward ~30 s after
boot, right while the SenseVoice worker loads. ASRProcess.start() waited
with one ready_event.wait(timeout=30); multiprocessing waits on a
wall-clock deadline, so the step (2 min 47 s, and 2 h 11 min the boot
before) expired it at once. The worker was killed ~1 s before it was
ready, ASRManager.start() logged "ASR disabled", and nothing ever retried:
two boots in a row ran with captions as the only name detector.

Covers:
  - ASRProcess.start() / OCRProcess.start() keep waiting when a wait
    returns early (what a forward clock step looks like from Python),
    and give up fast when the worker process exits.
  - ASRManager._ensure_worker(): restarts a dead worker right away,
    backs off 5 s, 10 s, 20 s ... (capped) on repeated failures, and
    stops a worker that came up after stop().
  - ASRManager.start() starts the loops even when every worker failed,
    so the loops can bring them back.

Run: python3 -m pytest tests/test_asr_worker_recovery.py -v
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


class _JumpyEvent:
    """Event whose wait() returns False immediately for the first `early`
    calls (a wall-clock deadline expired by a clock step), then reports
    set."""

    def __init__(self, early=3, never=False):
        self.early = early
        self.never = never
        self.calls = 0

    def wait(self, timeout=None):
        self.calls += 1
        if self.never or self.calls <= self.early:
            return False
        return True

    def is_set(self):
        return not self.never and self.calls > self.early

    def set(self):
        pass


class _FakeTap:
    wav_path = '/tmp/minus_asr_test.wav'


class TestASRProcessLoadWait(unittest.TestCase):

    def _start_with(self, ready, alive=True, failed=False):
        import asr_worker
        events = iter([ready, MagicMock(), MagicMock()])

        def make_event():
            ev = next(events)
            if isinstance(ev, MagicMock):
                ev.is_set.return_value = failed
            return ev

        proc_obj = MagicMock()
        proc_obj.is_alive.return_value = alive
        proc_obj.pid = 4242
        proc_obj.exitcode = 1
        with patch.object(asr_worker, 'Process', return_value=proc_obj), \
                patch.object(asr_worker, 'Queue', MagicMock), \
                patch.object(asr_worker, 'Event', side_effect=make_event):
            p = asr_worker.ASRProcess(model_name='tiny.en', cpu_threads=3, npu_core=1)
            p.WORKER_LOAD_TIMEOUT = 3.0
            t = time.monotonic()
            ok = p.start()
            return p, ok, time.monotonic() - t, proc_obj

    def test_early_wait_returns_do_not_fail_the_load(self):
        # Three waits "expire" early, as after an NTP step; the old code
        # gave up on the first one.
        p, ok, _, _ = self._start_with(_JumpyEvent(early=3))
        self.assertTrue(ok)
        self.assertTrue(p.is_ready)

    def test_times_out_on_the_monotonic_clock(self):
        p, ok, elapsed, proc_obj = self._start_with(_JumpyEvent(never=True))
        self.assertFalse(ok)
        self.assertFalse(p.is_ready)
        # Wait returns immediately here, so the loop is bounded only by
        # the monotonic budget (3 s), not by the number of waits.
        self.assertGreaterEqual(elapsed, 2.9)
        proc_obj.kill.assert_called()

    def test_dead_worker_fails_fast(self):
        p, ok, elapsed, _ = self._start_with(_JumpyEvent(never=True), alive=False)
        self.assertFalse(ok)
        self.assertLess(elapsed, 1.0)


class TestOCRProcessLoadWait(unittest.TestCase):

    def test_early_wait_returns_do_not_fail_the_load(self):
        import ocr_worker
        ready = _JumpyEvent(early=3)
        events = iter([ready, MagicMock()])
        proc_obj = MagicMock()
        proc_obj.is_alive.return_value = True
        with patch.object(ocr_worker, 'Process', return_value=proc_obj), \
                patch.object(ocr_worker, 'Queue', MagicMock), \
                patch.object(ocr_worker, 'Event', side_effect=lambda: next(events)):
            p = ocr_worker.OCRProcess()
            self.assertTrue(p.start())
            self.assertTrue(p.is_ready)


class TestASRManagerRecovery(unittest.TestCase):

    def _make(self):
        from asr import ASRManager
        m = ASRManager(_FakeTap())
        return m

    def _proc(self, alive):
        p = MagicMock()
        p.is_ready = alive
        p.process = MagicMock()
        p.process.is_alive.return_value = alive
        return p

    def test_healthy_worker_is_left_alone(self):
        m = self._make()
        p = self._proc(True)
        retry = {'fails': 0, 'next': 0.0}
        self.assertTrue(m._ensure_worker(p, retry))
        p.start.assert_not_called()

    def test_worker_that_died_is_restarted_at_once(self):
        m = self._make()
        p = self._proc(False)
        p.start.return_value = True
        retry = {'fails': 0, 'next': 0.0}
        self.assertTrue(m._ensure_worker(p, retry))
        p.start.assert_called_once()
        self.assertEqual(m.worker_restarts, 1)
        self.assertEqual(retry['fails'], 0)

    def test_failed_restarts_back_off(self):
        from asr import ASRManager
        m = self._make()
        p = self._proc(False)
        p.start.return_value = False
        retry = {'fails': 0, 'next': 0.0}
        delays = []
        for _ in range(9):
            retry['next'] = 0.0  # pretend the backoff has elapsed
            before = time.monotonic()
            self.assertFalse(m._ensure_worker(p, retry))
            delays.append(round(retry['next'] - before))
        self.assertEqual(delays[:4], [5, 10, 20, 40])
        self.assertEqual(max(delays), ASRManager.WORKER_RETRY_MAX_S)
        self.assertEqual(m.worker_start_failures, 9)

    def test_no_attempt_before_backoff_elapses(self):
        m = self._make()
        p = self._proc(False)
        retry = {'fails': 2, 'next': time.monotonic() + 60}
        self.assertFalse(m._ensure_worker(p, retry))
        p.start.assert_not_called()

    def test_recovers_after_failures(self):
        m = self._make()
        p = self._proc(False)
        p.start.side_effect = [False, False, True]
        retry = {'fails': 0, 'next': 0.0}
        results = []
        for _ in range(3):
            retry['next'] = 0.0
            results.append(m._ensure_worker(p, retry))
        self.assertEqual(results, [False, False, True])
        self.assertEqual(retry['fails'], 0)

    def test_worker_started_during_stop_is_stopped(self):
        m = self._make()
        p = self._proc(False)

        def start():
            m._stop_event.set()
            return True
        p.start.side_effect = start
        self.assertFalse(m._ensure_worker(p, {'fails': 0, 'next': 0.0}))
        p.stop.assert_called_once()

    def test_start_runs_loops_even_if_every_worker_failed(self):
        from asr import ASRManager
        m = self._make()
        p = self._proc(False)
        p.start.return_value = False
        m._processes = [p]
        m._process = p
        with patch('asr.is_asr_available', return_value=True), \
                patch.object(ASRManager, '_loop', lambda self, proc, i, n: None):
            m.start()
        self.assertTrue(m.is_running)
        status = m.get_status()
        self.assertEqual(status['workers_ready'], 0)
        self.assertEqual(status['workers_total'], 1)


if __name__ == '__main__':
    unittest.main()
