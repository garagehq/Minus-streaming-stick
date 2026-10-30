#!/usr/bin/env python3
"""A streak of failed snapshots is logged once when it starts and once when
it ends (every source sleep used to log dozens of ERROR lines)."""

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

capture = pytest.importorskip('capture')


def test_streak_logged_once_then_recovery(caplog):
    capture._fail_streak = 0
    with caplog.at_level(logging.INFO, logger=capture.logger.name):
        for _ in range(40):
            capture._note_capture_failure(TimeoutError('read timed out'))
        capture._note_capture_success()
        capture._note_capture_success()
    msgs = [r for r in caplog.records if r.name == capture.logger.name]
    assert [r.levelno for r in msgs] == [logging.WARNING, logging.INFO]
    assert 'read timed out' in msgs[0].getMessage()
    assert '40 failed attempts' in msgs[1].getMessage()
    assert not any(r.levelno >= logging.ERROR for r in msgs)


def test_a_new_streak_is_logged_again(caplog):
    capture._fail_streak = 0
    with caplog.at_level(logging.INFO, logger=capture.logger.name):
        capture._note_capture_failure(OSError('a'))
        capture._note_capture_success()
        capture._note_capture_failure(OSError('b'))
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == capture.logger.name]
    assert len(warns) == 2
