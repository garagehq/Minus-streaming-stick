#!/usr/bin/env python3
"""A block must survive a ustreamer restart.

minus, 2026-10-18 03:38: an ad was being blocked when a thermal capture
switch restarted ustreamer. The /blocking/set that started the block hit the
dying process ("Connection refused"), the new ustreamer came up with blocking
off, and Minus -- still believing it was blocking -- only ever sent text
updates. OCR and VLM both said AD at 97-100% while the ad played on screen.
"""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from ad_blocker import DRMAdBlocker  # noqa: E402


def blocker(ustreamer_says=None):
    b = object.__new__(DRMAdBlocker)
    b._lock = threading.Lock()
    b.is_visible = True
    b.current_source = 'ocr'
    b._animation_direction = None
    b._animating = False
    b._block_verified_at = 0.0
    b.calls = []

    def api(endpoint, params=None, **kw):
        b.calls.append((endpoint, params))
        if endpoint == '/blocking':
            return ustreamer_says
        return {'ok': True}
    b._blocking_api_call = api
    return b


class TestLostBlockDetection(unittest.TestCase):

    def test_new_ustreamer_with_blocking_off_is_a_lost_block(self):
        b = blocker({'ok': True, 'result': {'enabled': False}})
        self.assertTrue(b._ustreamer_lost_block(now=100.0))

    def test_blocking_on_is_fine(self):
        b = blocker({'ok': True, 'result': {'enabled': True}})
        self.assertFalse(b._ustreamer_lost_block(now=100.0))

    def test_unreachable_ustreamer_is_unknown_not_lost(self):
        b = blocker(None)
        self.assertFalse(b._ustreamer_lost_block(now=100.0))

    def test_checks_at_most_every_two_seconds(self):
        b = blocker({'ok': True, 'result': {'enabled': True}})
        b._ustreamer_lost_block(now=100.0)
        b._ustreamer_lost_block(now=101.0)
        b._ustreamer_lost_block(now=102.5)
        self.assertEqual([c for c in b.calls if c[0] == '/blocking'].__len__(), 2)


class TestShowReappliesALostBlock(unittest.TestCase):

    def test_show_restarts_the_block_when_ustreamer_lost_it(self):
        b = blocker({'ok': True, 'result': {'enabled': False}})
        started = []
        b._stop_animation_thread = MagicMock()

        # Stop right where a fresh block begins: the code after the early
        # returns rolls the replacement mode first.
        def roll():
            started.append(True)
            raise StopIteration
        b._locked_content_kind = None
        b._content_kind_lock_until = 0.0
        b._get_enabled_replacement_modes = lambda: []
        b._roll_replacement_mode = roll
        b._format_ocr_trigger = lambda text, source: ''
        b._ocr_trigger_text = ''
        with self.assertRaises(StopIteration):
            b.show('ocr')
        self.assertTrue(started, 'a lost block must go through the full start path again')
        self.assertFalse(b.is_visible)

    def test_show_keeps_a_healthy_block(self):
        b = blocker({'ok': True, 'result': {'enabled': True}})
        b._format_ocr_trigger = lambda text, source: ''
        b._ocr_trigger_text = ''
        b._render_ocr_text = lambda: ''
        b._roll_replacement_mode = MagicMock(side_effect=AssertionError('restarted a healthy block'))
        b.show('ocr')
        self.assertTrue(b.is_visible)

    def test_no_signal_screen_is_not_mistaken_for_a_block(self):
        b = blocker({'ok': True, 'result': {'enabled': False}})
        b.current_source = 'no_hdmi_device'
        self.assertFalse(b.reapply_block_if_lost())

    def test_reapply_after_restart_bypasses_the_rate_limit(self):
        b = blocker({'ok': True, 'result': {'enabled': True}})
        b._block_verified_at = 10**12   # just checked
        seen = []
        b.show = lambda source: seen.append((source, b._block_verified_at))
        self.assertTrue(b.reapply_block_if_lost())
        self.assertEqual(seen, [('ocr', 0.0)])


class TestRestartPathsReapply(unittest.TestCase):

    def test_both_ustreamer_start_paths_schedule_a_reapply(self):
        src = (ROOT / 'minus.py').read_text()
        self.assertEqual(src.count('self._schedule_block_reapply()'), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
