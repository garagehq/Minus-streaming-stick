#!/usr/bin/env python3
"""The stuck-mute watchdog leaves the deliberate signal-loss mute alone.

_on_hdmi_lost mutes on purpose; the watchdog forced an unmute ~10 s later
and logged "Audio stuck muted while not blocking" on every source sleep.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import health  # noqa: E402


def _hm(signal_lost, muted=True, blocking=False):
    hm = object.__new__(health.HealthMonitor)
    hm._muted_not_blocking_seen = False
    m = MagicMock()
    m._hdmi_signal_lost = signal_lost
    m.audio.is_muted = muted
    m.ad_blocker.is_visible = blocking
    hm.minus = m
    return hm, m


def test_signal_lost_mute_is_left_alone():
    hm, m = _hm(signal_lost=True)
    for _ in range(5):
        hm._check_audio_unmute()
    m.audio.unmute.assert_not_called()


def test_stuck_mute_with_signal_is_recovered_on_the_second_check():
    hm, m = _hm(signal_lost=False)
    hm._check_audio_unmute()
    m.audio.unmute.assert_not_called()
    hm._check_audio_unmute()
    m.audio.unmute.assert_called_once()


def test_mute_while_blocking_is_left_alone():
    hm, m = _hm(signal_lost=False, blocking=True)
    for _ in range(3):
        hm._check_audio_unmute()
    m.audio.unmute.assert_not_called()


def test_no_stale_confirmation_carried_over_a_signal_loss():
    hm, m = _hm(signal_lost=False)
    hm._check_audio_unmute()                 # seen once
    m._hdmi_signal_lost = True
    hm._check_audio_unmute()                 # signal lost: resets
    m._hdmi_signal_lost = False
    hm._check_audio_unmute()                 # first check after restore
    m.audio.unmute.assert_not_called()
