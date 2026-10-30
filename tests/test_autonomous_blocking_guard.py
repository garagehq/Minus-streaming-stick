#!/usr/bin/env python3
"""Autonomous mode's "mid ad-block" guards see the real overlay state.

set_ad_blocker() is given the Minus instance, which has no is_visible; the
overlay state is on minus.ad_blocker. All seven guards read the wrong object
and never fired: a music-mode steer deep-linked a new video mid-block on
minus (2026-10-30 02:26:51, block 02:26:38-58).
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import autonomous_mode as am  # noqa: E402


def _minus_like(visible):
    """Shaped like minus.Minus: no is_visible, overlay on .ad_blocker."""
    return SimpleNamespace(ad_blocker=SimpleNamespace(is_visible=visible),
                           last_ocr_texts=['Lous and the Yakuza', '18:22', 'Tiny Desk'],
                           audio=None)


def _mode(blocker):
    m = object.__new__(am.AutonomousMode)
    m._ad_blocker = blocker
    return m


def test_sees_the_overlay_through_the_minus_instance():
    assert _mode(_minus_like(True))._blocking_active() is True
    assert _mode(_minus_like(False))._blocking_active() is False


def test_a_bare_drm_blocker_still_works():
    assert _mode(SimpleNamespace(is_visible=True))._blocking_active() is True
    assert _mode(SimpleNamespace(is_visible=False))._blocking_active() is False
    assert _mode(None)._blocking_active() is False


def test_mock_auto_attributes_do_not_read_as_blocking():
    assert _mode(MagicMock(is_visible=False))._blocking_active() is False


def test_no_longform_sightings_recorded_mid_block():
    m = _mode(_minus_like(True))
    m._music_mode = True
    m._longform_sightings = []
    m._looks_long_form = lambda text: True
    m.observe_ocr_text(['18:22'])
    assert m._longform_sightings == []
    m._ad_blocker.ad_blocker.is_visible = False
    m.observe_ocr_text(['18:22'])
    assert len(m._longform_sightings) == 1


def test_no_raw_is_visible_reads_left():
    src = (ROOT / 'src' / 'autonomous_mode.py').read_text()
    assert "getattr(self._ad_blocker, 'is_visible'" not in src.replace(
        "if getattr(b, 'is_visible', False)", '')
