"""
ustreamer crash recovery without a display.

On a headless board (nothing on HDMI-out) _restart_ustreamer used to bail
out early, so a crashed ustreamer stayed dead: no frames, no OCR, and the
health monitor reported the signal lost while the source was playing. Seen
live on 2026-10-30, when ustreamer died at 11:00:30 and stayed down for
the rest of the round. ustreamer is the capture side and must be restarted
regardless; only the display pipeline restart needs a display.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


class TestRestartUstreamerHeadless(unittest.TestCase):

    def _minus(self, display_connected):
        from minus import Minus
        m = Minus.__new__(Minus)
        m.display_connected = display_connected
        m.ustreamer_process = MagicMock()
        m.device = '/dev/video0'
        m.config = MagicMock(ustreamer_port=9090)
        m.ad_blocker = MagicMock()
        m._init_v4l2_device = MagicMock()
        return m

    def _restart(self, m):
        new_proc = MagicMock()
        new_proc.poll.return_value = None  # still running after start
        with patch('minus.kill_ustreamer') as kill, \
                patch('minus.probe_v4l2_device', return_value={
                    'ustreamer_format': 'BGR24', 'width': 3840, 'height': 2160}), \
                patch('minus.subprocess.Popen', return_value=new_proc) as popen, \
                patch('minus.time.sleep'):
            m._restart_ustreamer()
        return kill, popen

    def test_restarts_without_display(self):
        old = MagicMock()
        m = self._minus(display_connected=False)
        m.ustreamer_process = old
        kill, popen = self._restart(m)
        old.terminate.assert_called_once()
        kill.assert_called_once_with(9090)
        popen.assert_called_once()
        cmd = popen.call_args[0][0]
        self.assertIn('--format=BGR24', cmd)
        self.assertIn('--resolution=3840x2160', cmd)

    def test_no_display_pipeline_restart_without_display(self):
        m = self._minus(display_connected=False)
        self._restart(m)
        m.ad_blocker.restart.assert_not_called()

    def test_display_pipeline_restarted_with_display(self):
        m = self._minus(display_connected=True)
        _, popen = self._restart(m)
        popen.assert_called_once()
        m.ad_blocker.restart.assert_called_once()


if __name__ == '__main__':
    unittest.main()
