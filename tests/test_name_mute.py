"""Tests for src/name_mute.py (matcher, scheduler, controller)."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from name_mute import NameMatcher, NameMuteScheduler, NameMuteController  # noqa: E402


class FakeAudio:
    def __init__(self):
        self.calls = []  # (monotonic, muted)

    def set_name_mute(self, muted):
        self.calls.append((time.monotonic(), muted))


class TestNameMatcher(unittest.TestCase):
    def setUp(self):
        self.m = NameMatcher()

    def test_text_variants(self):
        for t in ["Here's LeBron James and easy finish", "LeBron's drive",
                  "Le Bron with the dunk", "by the Bron James", "Lebrun scores",
                  "KING JAMES!", "pass to LEBRON"]:
            self.assertTrue(self.m.find(t), t)

    def test_text_non_matches(self):
        for t in ["The Bronx team", "James Harden for three", "lebanon",
                  "brontosaurus", ""]:
            self.assertFalse(self.m.find(t), t)

    def test_surname_toggle(self):
        self.assertFalse(self.m.find("James with the block"))
        self.assertTrue(NameMatcher(include_surname=True).find("James with the block"))

    def test_word_spans_full_name_merges(self):
        words = [("Here's", 0.0, 0.3), ("LeBron", 0.3, 0.7), ("James", 0.7, 1.0),
                 ("and", 1.0, 1.1)]
        self.assertEqual(self.m.find_word_spans(words), [(0.3, 1.0)])

    def test_word_spans_first_name(self):
        words = [("LeBron.", 1.0, 1.4), ("Nice", 1.5, 1.7)]
        self.assertEqual(self.m.find_word_spans(words), [(1.0, 1.4)])

    def test_word_spans_bronx_needs_james(self):
        self.assertEqual(self.m.find_word_spans(
            [("the", 0, .1), ("Bronx", .1, .5), ("team", .5, .8)]), [])
        self.assertEqual(self.m.find_word_spans(
            [("the", 0, .1), ("Bron", .1, .4), ("James", .4, .8)]), [(.1, .8)])


class TestScheduler(unittest.TestCase):
    def test_on_time_window(self):
        audio = FakeAudio()
        sch = NameMuteScheduler(audio, lambda: 1.0)
        sch.start()
        try:
            now = time.monotonic()
            # Captured 0.1s from now, played 1.0s later.
            sch.schedule(now + 0.1, now + 0.2, 'asr', 'LeBron')
            time.sleep(1.9)
        finally:
            sch.stop()
        on = [t for t, m in audio.calls if m]
        off = [t for t, m in audio.calls if not m]
        self.assertTrue(on and off)
        self.assertAlmostEqual(on[0] - now, 0.1 + 1.0 - sch.PAD_BEFORE_S, delta=0.06)
        self.assertAlmostEqual(off[0] - now, 0.2 + 1.0 + sch.PAD_AFTER_S, delta=0.06)
        self.assertEqual(sch.late_count, 0)

    def test_covered_detection_is_not_rescheduled(self):
        sch = NameMuteScheduler(FakeAudio(), lambda: 5.0)
        now = time.monotonic()
        sch.schedule(now, now + 1.0, 'caption', 'LeBron')       # wide window
        sch.schedule(now + 0.2, now + 0.5, 'asr', 'LeBron')     # inside it
        self.assertEqual(sch.mute_count, 1)
        self.assertEqual(sch.duplicate_count, 1)
        sch.schedule(now + 3.0, now + 3.4, 'asr', 'LeBron')     # new mention
        self.assertEqual(sch.mute_count, 2)

    def test_late_detection_still_mutes(self):
        audio = FakeAudio()
        sch = NameMuteScheduler(audio, lambda: 0.0)
        sch.start()
        try:
            now = time.monotonic()
            sch.schedule(now - 2.0, now - 1.5, 'asr', 'LeBron')
            time.sleep(0.2)
            self.assertTrue(sch.get_status()['muted'])
        finally:
            sch.stop()
        self.assertEqual(sch.late_count, 1)


class TestController(unittest.TestCase):
    def _ctl(self):
        sch = NameMuteScheduler(FakeAudio(), lambda: 4.0)
        return NameMuteController(sch), sch

    def test_asr_overlapping_windows_dedup(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        # Same mention seen by two overlapping windows (1s apart).
        ctl.on_asr_words("Here's LeBron", [("Here's", 1.0, 1.3), ("LeBron", 1.3, 1.8)], t0)
        ctl.on_asr_words("LeBron James", [("LeBron", 0.3, 0.8), ("James", 0.8, 1.1)], t0 + 1.0)
        self.assertEqual(ctl.asr_hits, 1)

    def test_truncated_word_extended(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("here's LeBron", [("here's", 1.5, 2.0), ("LeBron", 2.1, 2.5)], t0)
        start, end = sch._windows[0][:2]
        self.assertAlmostEqual(end - start,
                               (2.5 + ctl.TRUNCATED_WORD_EXTRA_S - 2.1)
                               + sch.PAD_BEFORE_S + sch.PAD_AFTER_S, delta=0.01)

    def test_asr_two_mentions(self):
        ctl, _ = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("LeBron passes to LeBron", [("LeBron", 0.1, 0.4), ("passes", .4, .7),
                                                    ("to", .7, .8), ("LeBron", 1.9, 2.3)], t0)
        self.assertEqual(ctl.asr_hits, 2)

    def test_caption_growing_line_dedup(self):
        ctl, _ = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["here's LeBron"], t)
        ctl.on_caption_texts(["here's LeBron James and"], t + 0.5)
        ctl.on_caption_texts(["here's LeBron James and easy finish"], t + 1.0)
        self.assertEqual(ctl.caption_hits, 1)
        ctl.on_caption_texts(["LeBron again on the break"], t + 2.0)
        self.assertEqual(ctl.caption_hits, 2)

    def test_disabled(self):
        ctl, _ = self._ctl()
        ctl.enabled = False
        ctl.on_caption_texts(["LeBron"], time.monotonic())
        ctl.on_asr_words("LeBron", [("LeBron", 0, .4)], time.monotonic())
        self.assertEqual(ctl.caption_hits + ctl.asr_hits, 0)


if __name__ == '__main__':
    unittest.main()
