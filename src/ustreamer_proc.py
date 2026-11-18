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


# ustreamer's own output. It used to go to /dev/null, so a ustreamer that
# exited at startup left only "ustreamer failed to start" in the log.
USTREAMER_LOG = '/tmp/ustreamer.log'
USTREAMER_LOG_MAX = 2 * 1024 * 1024


def spawn(cmd) -> subprocess.Popen:
    """Start ustreamer with stdout+stderr appended to USTREAMER_LOG."""
    try:
        import os
        if os.path.getsize(USTREAMER_LOG) > USTREAMER_LOG_MAX:
            os.replace(USTREAMER_LOG, USTREAMER_LOG + '.1')
    except OSError:
        pass
    try:
        out = open(USTREAMER_LOG, 'ab')
        out.write(f"==== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        out.flush()
    except OSError:
        out = subprocess.DEVNULL
    try:
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    finally:
        if out is not subprocess.DEVNULL:
            out.close()   # the child keeps its own copy of the descriptor


def exit_report(proc, lines: int = 8) -> str:
    """Exit code and the last lines ustreamer wrote, for the error log."""
    code = proc.poll()
    try:
        with open(USTREAMER_LOG, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode(errors='replace').splitlines()[-lines:]
    except OSError:
        tail = []
    return f"exit code {code}; last output: " + ' | '.join(t.strip() for t in tail)


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
