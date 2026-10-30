#!/usr/bin/env python3
"""Fixes ported from the asr-first branch (Oct 2026)."""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


# --- Autonomous mode: no Home for an awake source (cfffa1d) -----------------

def _auto(awake):
    import autonomous_mode as am
    a = object.__new__(am.AutonomousMode)
    a._last_source_wake = 0.0
    a._log_event = MagicMock()
    dev = MagicMock()
    dev.is_connected.return_value = True
    if awake == 'missing':
        del dev.is_awake
    else:
        dev.is_awake.return_value = awake
    a._device_controller = dev
    return a, dev


def test_awake_source_gets_no_home():
    a, dev = _auto(True)
    assert a._wake_sleeping_source() is False
    dev.send_command.assert_not_called()


def test_asleep_or_unknown_source_gets_home():
    for awake in (False, None, 'missing'):
        a, dev = _auto(awake)
        assert a._wake_sleeping_source() is True, awake
        dev.send_command.assert_called_once_with('home')


def test_fire_tv_is_awake_parses_dumpsys():
    import fire_tv
    c = object.__new__(fire_tv.FireTVController)
    c._lock = threading.Lock()
    c._connected = True
    c._device = MagicMock()
    for out, want in (('  mWakefulness=Awake\n', True), ('  mWakefulness=Asleep\n', False),
                      ('  mWakefulness=Dreaming\n', False), ('', None)):
        c._device.adb_shell.return_value = out
        assert c.is_awake() is want, out
    c._device.adb_shell.side_effect = RuntimeError('adb gone')
    assert c.is_awake() is None
    c._connected = False
    assert c.is_awake() is None


# --- VLM worker: fail fast on an early exit (f668e32) -----------------------

def test_vlm_start_fails_fast_when_worker_dies():
    import vlm_worker
    src = Path(vlm_worker.__file__).read_text()
    body = src[src.index('Waiting for model to load'):src.index('def kill(self)')]
    assert 'self.process.is_alive()' in body and 'time.monotonic()' in body


# --- Startup: self.vlm can be None after a failed preload (c187bcf) ---------

def test_startup_rechecks_vlm_after_preload_join():
    src = (ROOT / 'minus.py').read_text()
    join = src.index('vlm_preload_thread.join(timeout=60)')
    after = src[join:join + 1500]
    assert 'if self.vlm and self.vlm.is_ready:' in after


# --- Stream fps fallback (31becd2, adapted) ----------------------------------

def test_stream_fps_fallback_uses_the_thermal_cap():
    src = (ROOT / 'minus.py').read_text()
    assert 'STREAM_FPS' not in src
    assert "cap = getattr(self, '_capture_fps_cap', 0) or 0" in src
