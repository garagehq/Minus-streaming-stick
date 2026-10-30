#!/usr/bin/env python3
"""The destructive-action audio guard reads the audio module's continuous
audibility, not only its own sparse samples.

minus, 2026-10-30 01:33: after an 18 s block (no autonomous sampling during
it) the 20 s "recently flowing" window had lapsed; the post-skip frame read
as a keyboard and Back+navigate walked YouTube into the account picker and
the Amazon Appstore.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import autonomous_mode as am  # noqa: E402


def _mode(quiet, sampled_flowing=False, last_sample_age=None):
    m = object.__new__(am.AutonomousMode)
    m._AUDIO_RECENT_WINDOW = 20.0
    m._last_audio_flowing_time = None if last_sample_age is None else time.time() - last_sample_age
    m._ad_blocker = MagicMock()
    if quiet == 'missing':
        del m._ad_blocker.audio.seconds_since_audible
    else:
        m._ad_blocker.audio.seconds_since_audible.return_value = quiet
    m._is_audio_flowing = lambda: sampled_flowing
    return m


def test_recently_audible_input_vetoes_even_with_a_stale_sample():
    assert _mode(quiet=3.0, last_sample_age=40)._audio_recently_flowing() is True


def test_long_silence_does_not_veto():
    assert _mode(quiet=120.0, last_sample_age=200)._audio_recently_flowing() is False


def test_unknown_falls_back_to_the_sampled_window():
    assert _mode(quiet=None, last_sample_age=5)._audio_recently_flowing() is True
    assert _mode(quiet=None, last_sample_age=60)._audio_recently_flowing() is False
    assert _mode(quiet='missing', sampled_flowing=True)._audio_recently_flowing() is True


def test_flowing_now_still_vetoes():
    assert _mode(quiet=None, sampled_flowing=True)._audio_recently_flowing() is True
