#!/usr/bin/env python3
"""The health monitor trims the main process's heap periodically.

minus, 2026-10-30: the main process grew ~80 MB/h with the display pipeline
running; malloc_trim(0) on the live process returned 112 MB at once.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import health  # noqa: E402


def _hm():
    return object.__new__(health.HealthMonitor)


def test_trims_once_per_interval():
    hm = _hm()
    assert hm._trim_process_heap(now=1000.0) is not None
    assert hm._trim_process_heap(now=1000.0 + 10) is None
    assert hm._trim_process_heap(now=1000.0 + hm.HEAP_TRIM_INTERVAL_S + 1) is not None


def test_never_raises(monkeypatch):
    hm = _hm()
    import ctypes
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: (_ for _ in ()).throw(OSError('no libc')))
    assert hm._trim_process_heap(now=5000.0) is None


def test_called_from_the_health_loop():
    src = (ROOT / 'src' / 'health.py').read_text()
    body = src[src.index('def _check_and_recover'):]
    assert 'self._trim_process_heap()' in body[:400]
