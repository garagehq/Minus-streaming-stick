"""resolve_engine() runs after every transcription; it must not grow sys.path."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asr_worker


def test_resolve_engine_does_not_grow_sys_path(monkeypatch):
    monkeypatch.setenv('MINUS_ASR_ENGINE', 'sensevoice')
    asr_worker.resolve_engine()
    before = len(sys.path)
    for _ in range(50):
        asr_worker.resolve_engine()
    assert len(sys.path) == before


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
