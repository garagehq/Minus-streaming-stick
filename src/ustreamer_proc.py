"""Reliably stop the ustreamer process.

`pkill ustreamer` matches the process NAME, and ustreamer renames its main
thread (which is the process name on Linux) to "main". Every
`pkill -9 ustreamer` in Minus was therefore a no-op. Where a port kill
followed, it saved the day; on the format-change restart path it did not:
the old instance kept port 9090 serving its last frames while the new one
failed to bind (observed: a Google TV switch to 4K NV12 left the stream
frozen for over an hour, and autonomous mode read the missing capture as a
sleeping source and kept pressing Home).

Match the executable path instead, also kill whatever holds the port, and
wait until it is really gone.
"""

import logging
import subprocess
import time

from config import USTREAMER_PATH

logger = logging.getLogger('Minus.Ustreamer')


def _pattern() -> str:
    # pkill -f matches the full command line as an extended regex. Anchored
    # on the path so it cannot match unrelated processes (or pkill itself).
    return '^' + USTREAMER_PATH.replace('.', r'\.') + '( |$)'


def _running(port: int) -> bool:
    by_path = subprocess.run(['pgrep', '-f', _pattern()],
                             capture_output=True).returncode == 0
    by_port = subprocess.run(['fuser', f'{port}/tcp'],
                             capture_output=True).returncode == 0
    return by_path or by_port


def kill_ustreamer(port: int = 9090, timeout: float = 3.0) -> bool:
    """SIGKILL every ustreamer instance and anything holding `port`.

    Returns True once none is left (or False after `timeout`).
    """
    for cmd in (['pkill', '-9', '-f', _pattern()],
                ['pkill', '-9', '-x', 'ustreamer'],
                ['fuser', '-k', '-9', f'{port}/tcp']):
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _running(port):
            return True
        time.sleep(0.1)
    logger.warning(f"ustreamer still running {timeout:.0f}s after SIGKILL")
    return False
