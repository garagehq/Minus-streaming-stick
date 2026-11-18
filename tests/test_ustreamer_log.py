"""
/tmp/ustreamer.log lives in RAM (tmpfs). ustreamer_proc.trim_log() must keep
it bounded while ustreamer is still writing to it (O_APPEND), without leaving
a sparse hole or losing the writer.

Run: python3 -m unittest tests.test_ustreamer_log
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import ustreamer_proc  # noqa: E402


class TestUstreamerLogTrim(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self._orig = ustreamer_proc.USTREAMER_LOG
        ustreamer_proc.USTREAMER_LOG = os.path.join(self.dir.name, 'ustreamer.log')

    def tearDown(self):
        ustreamer_proc.USTREAMER_LOG = self._orig
        self.dir.cleanup()

    def test_small_log_left_alone(self):
        with open(ustreamer_proc.USTREAMER_LOG, 'ab') as f:
            f.write(b'x' * 1000)
        self.assertFalse(ustreamer_proc.trim_log(max_bytes=4096))
        self.assertEqual(os.path.getsize(ustreamer_proc.USTREAMER_LOG), 1000)

    def test_trim_while_writer_keeps_appending(self):
        log = ustreamer_proc.USTREAMER_LOG
        writer = open(log, 'ab', buffering=0)      # as ustreamer holds it
        try:
            writer.write(b'a' * 10000)
            self.assertTrue(ustreamer_proc.trim_log(max_bytes=4096))
            self.assertEqual(os.path.getsize(log), 0)
            self.assertEqual(os.path.getsize(log + '.1'), 10000)
            writer.write(b'b' * 100)               # writer did not reopen
        finally:
            writer.close()
        # O_APPEND: the write landed at the new end, no hole before it.
        with open(log, 'rb') as f:
            self.assertEqual(f.read(), b'b' * 100)

    def test_bounded_over_many_cycles(self):
        log = ustreamer_proc.USTREAMER_LOG
        with open(log, 'ab', buffering=0) as w:
            for _ in range(50):
                w.write(b'z' * 3000)
                ustreamer_proc.trim_log(max_bytes=4096)
                self.assertLessEqual(os.path.getsize(log), 4096 + 3000)
                if os.path.exists(log + '.1'):
                    self.assertLessEqual(os.path.getsize(log + '.1'), 4096 + 3000)

    def test_missing_log_is_fine(self):
        self.assertFalse(ustreamer_proc.trim_log())


if __name__ == '__main__':
    unittest.main()
