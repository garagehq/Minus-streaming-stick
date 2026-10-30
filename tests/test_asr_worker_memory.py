#!/usr/bin/env python3
"""The ASR worker's memory must stay bounded (ported from asr-first).

SenseVoice worker soaks, 2026-10-26/27: anonymous memory grew ~80-110 MB/h.
The worker is spawned with malloc tunables (large buffers on mmap, two
arenas), runs gc.collect(0) after each reply (rknnlite leaves its buffers in
reference cycles) and malloc_trim(0)s every TRIM_EVERY calls.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import asr_worker  # noqa: E402


class _FakeProcess:
    """Records the environment at start() instead of spawning a worker."""
    seen = None

    def __init__(self, *a, **k):
        self.pid = 0

    def start(self):
        _FakeProcess.seen = {k: os.environ.get(k) for k in asr_worker.ASRProcess.WORKER_MALLOC_ENV}

    def is_alive(self):
        return False


def _start_with_fake(monkeypatch):
    monkeypatch.setattr(asr_worker, 'Process', _FakeProcess)
    monkeypatch.setattr(asr_worker.ASRProcess, 'WORKER_LOAD_TIMEOUT', 0.01)
    p = asr_worker.ASRProcess()
    monkeypatch.setattr(p, '_kill_process_locked', lambda: None)
    p.start()


def test_worker_malloc_env_set_only_for_spawn(monkeypatch):
    for k in asr_worker.ASRProcess.WORKER_MALLOC_ENV:
        monkeypatch.delenv(k, raising=False)
    _start_with_fake(monkeypatch)
    assert _FakeProcess.seen == asr_worker.ASRProcess.WORKER_MALLOC_ENV
    assert all(k not in os.environ for k in asr_worker.ASRProcess.WORKER_MALLOC_ENV)


def test_worker_malloc_env_respects_explicit_value(monkeypatch):
    monkeypatch.setenv('MALLOC_ARENA_MAX', '4')
    _start_with_fake(monkeypatch)
    assert _FakeProcess.seen['MALLOC_ARENA_MAX'] == '4'
    assert os.environ['MALLOC_ARENA_MAX'] == '4'


def test_malloc_trimmer_is_glibc_trim():
    trim = asr_worker._malloc_trimmer()
    assert trim is not None
    assert trim(0) in (0, 1)


def test_malloc_trimmer_disabled_with_zero(monkeypatch):
    monkeypatch.setattr(asr_worker, 'TRIM_EVERY', 0)
    assert asr_worker._malloc_trimmer() is None


def test_worker_loop_collects_after_each_reply():
    src = (ROOT / 'src' / 'asr_worker.py').read_text()
    reply = src.index("response_queue.put(('ok', text, time.time() - start))")
    assert 'gc.collect(0)' in src[reply:reply + 400]
