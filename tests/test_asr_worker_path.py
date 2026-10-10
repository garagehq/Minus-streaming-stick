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
