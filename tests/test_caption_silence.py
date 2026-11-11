"""A caption hit with no audio under it is a title card: no mute, no badge.

Seen live on the kingblocker9000 unit (2026-11-11): highlight videos open
on title cards ("LeBron James' Top 40 Career", "lebron james highlight
reel") that OCR read as captions and muted, with nobody saying the name.
"""
import math
import os
import sys
import time
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from name_mute import NameMuteScheduler, NameMuteController  # noqa: E402


class FakeAudio:
    def set_name_mute(self, muted):
        pass


def _ctl(level):
    sch = NameMuteScheduler(FakeAudio(), lambda: 2.0)
    ctl = NameMuteController(sch)
    ctl.audio_level_fn = level
    return ctl, sch


def test_silent_caption_not_muted():
    seen = []
    ctl, sch = _ctl(lambda t0, t1: seen.append((t0, t1)) or -90.0)
    t = time.monotonic()
    ctl.on_caption_texts(["LeBron James' Top 40 Career"], t)
    assert sch._windows == []
    assert ctl.caption_silent_skips == 1
    assert ctl.get_status()['caption_silent_skips'] == 1
    # Measured over the stretch before the frame, where the word was said.
    assert seen == [(t - ctl.CAPTION_AUDIO_S, t)]


def test_caption_over_audio_is_muted():
    ctl, sch = _ctl(lambda t0, t1: -20.0)
    ctl.on_caption_texts(["here's LeBron James and"], time.monotonic())
    assert len(sch._windows) == 1
    assert ctl.caption_silent_skips == 0


def test_unknown_level_still_mutes():
    for fn in (lambda t0, t1: None, None, mock.Mock(side_effect=RuntimeError('x'))):
        ctl, sch = _ctl(fn)
        ctl.on_caption_texts(["here's LeBron James and"], time.monotonic())
        assert len(sch._windows) == 1


def test_env_turns_check_off():
    with mock.patch.dict(os.environ, {'MINUS_NAME_MUTE_CAPTION_QUIET_DBFS': ''}):
        ctl, sch = _ctl(lambda t0, t1: -150.0)
    ctl.on_caption_texts(["here's LeBron James and"], time.monotonic())
    assert len(sch._windows) == 1


def test_asr_hits_ignore_the_level():
    ctl, sch = _ctl(lambda t0, t1: -150.0)
    ctl.on_asr_words("pass to LeBron", [("pass", 0.1, 0.4), ("to", .4, .6),
                                        ("LeBron", 1.0, 1.4)], time.monotonic())
    assert len(sch._windows) == 1


def test_tap_level_dbfs():
    gi = pytest_importorskip_gi()
    if gi is None:
        return
    from audio import AudioASRTap
    tap = AudioASRTap()
    sr = tap.SAMPLE_RATE
    loud = (0.1 * 32767 * np.sin(np.arange(sr) * 2 * math.pi * 440 / sr)).astype(np.int16)
    quiet = np.zeros(sr, dtype=np.int16)
    data = np.concatenate([loud, quiet])          # 1 s tone, then 1 s silence
    tap._ring[:len(data)] = data
    tap._write_pos = len(data)
    tap._samples_written = len(data)
    tap._last_write_mono = 100.0                  # newest sample at t=100
    tone = tap.level_dbfs(99.0 - 0.9, 99.0 - 0.1)
    assert abs(tone - (20 * math.log10(0.1 / math.sqrt(2)))) < 0.5   # ~-23 dBFS
    assert tap.level_dbfs(99.1, 99.9) < -100
    assert tap.level_dbfs(50.0, 60.0) is None    # older than the ring holds
    assert tap.level_dbfs(99.5, 99.5) is None


def pytest_importorskip_gi():
    try:
        import gi  # noqa: F401
        return gi
    except ImportError:
        return None
