#!/usr/bin/env python3
"""ustreamer must be killed by executable path and port, never by name alone.

ustreamer names its main thread "main", which on Linux is the process name,
so `pkill -9 ustreamer` matched nothing (confirmed on minus, 2026-10-30:
comm=main, `pgrep -x ustreamer` empty). On the format-change restart path
the old instance kept port 9090 serving stale frames (asr-first 182080f).
"""

import re
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import ustreamer_proc as up  # noqa: E402


def test_pattern_matches_the_binary_only():
    pat = re.compile(up._pattern())
    path = up.USTREAMER_PATH
    assert pat.search(f'{path} --device /dev/video0 --port 9090')
    assert pat.search(path)
    assert not pat.search(f'vim {path}.c')
    assert not pat.search(f'{path}-old --port 9090')
    assert not pat.search(f'pkill -9 -f ^{path}')


def test_kill_uses_path_and_port_and_waits():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type('R', (), {'returncode': 1})()   # nothing left running
    with patch.object(up.subprocess, 'run', side_effect=fake_run):
        assert up.kill_ustreamer(9090, timeout=0.5) is True
    assert ['pkill', '-9', '-f', up._pattern()] in calls
    assert ['fuser', '-k', '-9', '9090/tcp'] in calls


def test_kill_reports_a_survivor():
    with patch.object(up.subprocess, 'run',
                      return_value=type('R', (), {'returncode': 0})()), \
            patch.object(up.time, 'sleep'):
        assert up.kill_ustreamer(9090, timeout=0.05) is False


def test_no_name_only_kills_left_in_python():
    for f in [ROOT / 'minus.py', *(ROOT / 'src').glob('*.py')]:
        for line in f.read_text().splitlines():
            if "'pkill', '-9', 'ustreamer'" in line:
                # The early-init fallback keeps it only next to a path kill.
                assert f.name == 'minus.py', f
