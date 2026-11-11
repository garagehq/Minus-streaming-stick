"""is_asr_available() follows the selected engine, not faster-whisper.

A box with the SenseVoice model + rknnlite but no faster_whisper package
(the kingblocker9000 unit) used to run with ASR off, so the name muter
never heard anything.
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asr  # noqa: E402


def _blocked_import(*names):
    real = __import__

    def fake(name, *a, **k):
        if name.split('.')[0] in names:
            raise ImportError(name)
        return real(name, *a, **k)
    return fake


def test_sensevoice_needs_no_faster_whisper():
    with mock.patch.object(asr, 'resolve_engine', return_value='sensevoice'), \
         mock.patch('builtins.__import__', _blocked_import('faster_whisper')):
        assert asr.is_asr_available() is True


def test_moonshine_needs_its_package():
    with mock.patch.object(asr, 'resolve_engine', return_value='moonshine'), \
         mock.patch('builtins.__import__', _blocked_import('moonshine_voice')):
        assert asr.is_asr_available() is False


def test_parakeet_needs_sherpa():
    with mock.patch.object(asr, 'resolve_engine', return_value='parakeet'), \
         mock.patch('builtins.__import__', _blocked_import('sherpa_onnx')):
        assert asr.is_asr_available() is False


def test_whisper_engine_needs_faster_whisper():
    with mock.patch.object(asr, 'resolve_engine', return_value='whisper'), \
         mock.patch('builtins.__import__', _blocked_import('faster_whisper')):
        assert asr.is_asr_available() is False


def test_resolve_failure_means_unavailable():
    with mock.patch.object(asr, 'resolve_engine', side_effect=RuntimeError('x')):
        assert asr.is_asr_available() is False
