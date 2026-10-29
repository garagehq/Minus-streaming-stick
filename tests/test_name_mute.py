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

    def test_king(self):
        for t in ["the King with the dunk", "KING JAMES!", "long live the king"]:
            self.assertTrue(self.m.find(t), t)
        for t in ["Sacramento Kings", "the Kings win", "kingdom"]:
            self.assertFalse(self.m.find(t), t)
        self.assertEqual(self.m.find_word_spans(
            [("the", 0, .1), ("King", .1, .4), ("scores", .4, .8)]), [(.1, .4)])
        self.assertEqual(self.m.find_word_spans(
            [("King", 0, .3), ("James", .3, .7)]), [(0, .7)])
        self.assertEqual(self.m.find_word_spans([("Kings", 0, .3)]), [])

    def test_text_non_matches(self):
        for t in ["The Bronx team", "James Harden for three", "lebanon",
                  "brontosaurus", "", "James Jones"]:
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

    def test_mute_change_hook_follows_the_switch(self):
        # Drives the on-screen BLOCKED badge.
        audio, seen = FakeAudio(), []
        sch = NameMuteScheduler(audio, lambda: 0.3)
        sch.on_mute_change = seen.append
        sch.start()
        try:
            now = time.monotonic()
            sch.schedule(now, now + 0.1, 'asr', 'LeBron', pad_before=0.0, pad_after=0.0)
            time.sleep(0.7)
        finally:
            sch.stop()
        self.assertEqual(seen[:2], [True, False])
        self.assertEqual([m for _, m in audio.calls][:2], [True, False])


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

    def test_garbled_first_name_before_james(self):
        m = NameMatcher()
        w = lambda t: [(x, i * 0.3, i * 0.3 + 0.3) for i, x in enumerate(t.split())]
        for t in ("amron james has", "thebron james with", "lebra james"):
            self.assertEqual(len(m.find_word_spans(w(t))), 1, t)
        for t in ("aaron james", "cameron james", "the bronx", "ron james"):
            self.assertEqual(m.find_word_spans(w(t)), [], t)

    def test_word_cut_at_window_start_extended_back(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("thebron james with", [("thebron", 0.0, 0.4), ("james", 0.4, 0.8),
                                                ("with", 0.8, 1.0)], t0)
        start, _ = sch._windows[0][:2]
        self.assertAlmostEqual(start, t0 + 4.0 - ctl.TRUNCATED_WORD_EXTRA_S - sch.PAD_BEFORE_S,
                               delta=0.01)

    def test_caption_line_ending_on_first_name_runs_longer(self):
        ctl, sch = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["left-handed in and LeBron"], t)
        ctl.on_caption_texts(["here's LeBron James and"], t + 5)
        (s1, e1), (s2, e2) = [w[:2] for w in sch._windows[:2]]
        self.assertAlmostEqual((e1 - s1) - (e2 - s2), ctl.CAPTION_LINE_END_EXTRA_S, delta=0.01)

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

    def test_caption_ocr_misread_is_same_mention(self):
        ctl, _ = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["they looking for LeBron down inside"], t)
        ctl.on_caption_texts(["they oking for LeBron dow inside"], t + 0.5)
        self.assertEqual(ctl.caption_hits, 1)

    def test_caption_line_starting_with_name(self):
        # No text before the name: a growing line ("LeBron" -> "LeBron
        # drives ...") must stay one mention, so it is keyed on the name alone
        # for CAPTION_MEMORY_S; a new one after that counts again.
        ctl, _ = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["LeBron"], t)
        ctl.on_caption_texts(["LeBron drives to the rim"], t + 1.0)
        self.assertEqual(ctl.caption_hits, 1)
        ctl._recent_captions = {k: (v[0] - ctl.CAPTION_MEMORY_S - 1, v[1])
                                for k, v in ctl._recent_captions.items()}
        ctl.on_caption_texts(["LeBron for three"], t + 12.0)
        self.assertEqual(ctl.caption_hits, 2)

    def test_allcaps_graphic_ignored(self):
        ctl, _ = self._ctl()
        ctl.on_caption_texts(["LEBRON JAMES"], time.monotonic())
        self.assertEqual(ctl.caption_hits, 0)

    def test_graphic_name_ignored(self):
        ctl, _ = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["LEBRON 2011 Finals"], t)
        ctl.on_caption_texts(["LeBRON JAMES"], t)
        self.assertEqual(ctl.caption_hits, 0)
        ctl.on_caption_texts(["and lebron drives"], t)
        ctl.on_caption_texts(["here comes LeBron"], t)
        self.assertEqual(ctl.caption_hits, 2)

    def test_zero_length_asr_word_ignored(self):
        ctl, _ = self._ctl()
        ctl.on_asr_words("LeBron. LeBron.", [("LeBron.", 0.07, 0.07)], time.monotonic())
        self.assertEqual(ctl.asr_hits, 0)

    def test_asr_replaces_pending_caption_window(self):
        ctl, sch = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["and here comes LeBron"], t)
        self.assertEqual([w[2] for w in sch._windows], ['caption'])
        ctl.on_asr_words("here comes LeBron", [("here", 0.5, 0.7), ("comes", 0.7, 1.0),
                                               ("LeBron", 1.0, 1.4)], t - 1.5)
        self.assertEqual([w[2] for w in sch._windows], ['asr'])

    def test_edge_king_ignored_only_at_window_edges(self):
        ctl, sch = self._ctl()
        ctl.window_s = 3.0
        ctl.edge_king = True
        t0 = time.monotonic()
        # "tal|king": window starts inside the word.
        ctl.on_asr_words("king about this", [("king", 0.0, 0.2), ("about", 0.3, 0.6), ("this", 0.6, 0.9)], t0)
        # "sacramento king|s": window ends inside the word.
        ctl.on_asr_words("sacramento king", [("sacramento", 2.0, 2.6), ("king", 2.65, 2.95)], t0 + 5)
        self.assertEqual(ctl.asr_hits, 0)
        self.assertEqual(ctl.edge_king_skips, 2)
        # A whole "king" inside the window and "King James" at an edge still mute.
        ctl.on_asr_words("long live the king now", [("long", 0.5, 0.8), ("live", 0.8, 1.0), ("the", 1.0, 1.1),
                                                   ("king", 1.1, 1.5), ("now", 1.6, 1.9)], t0 + 10)
        ctl.on_asr_words("king james", [("king", 0.0, 0.3), ("james", 0.3, 0.7)], t0 + 20)
        ctl.on_asr_words("lebron", [("lebron", 2.6, 2.95)], t0 + 30)
        self.assertEqual(ctl.asr_hits, 3)

    def test_edge_king_on_by_default(self):
        ctl, sch = self._ctl()
        ctl.window_s = 3.0
        ctl.on_asr_words("sacramento king", [("sacramento", 2.0, 2.6), ("king", 2.65, 2.95)], time.monotonic())
        self.assertEqual(ctl.asr_hits, 0)
        self.assertEqual(ctl.edge_king_skips, 1)

    def test_edge_king_off_with_env(self):
        os.environ['MINUS_NAME_MUTE_EDGE_KING'] = '0'
        try:
            ctl, sch = self._ctl()
        finally:
            del os.environ['MINUS_NAME_MUTE_EDGE_KING']
        ctl.window_s = 3.0
        ctl.on_asr_words("sacramento king", [("sacramento", 2.0, 2.6), ("king", 2.65, 2.95)], time.monotonic())
        self.assertEqual(ctl.asr_hits, 1)

    def test_edge_king_skip_does_not_trigger_whole_window_fallback(self):
        ctl, sch = self._ctl()
        ctl.window_s = 3.0
        ctl.edge_king = True
        ctl.on_asr_words("the king", [("the", 2.3, 2.5), ("king", 2.6, 2.95)], time.monotonic())
        self.assertEqual(sch._windows, [])

    def test_words_log(self):
        import json
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), 'w.jsonl')
        os.environ['MINUS_ASR_WORDS_LOG'] = path
        try:
            ctl, _ = self._ctl()
        finally:
            del os.environ['MINUS_ASR_WORDS_LOG']
        ctl.on_asr_words("hi there", [("hi", 0.1, 0.3), ("there", 0.3, 0.6)], 100.0)
        row = json.loads(open(path).read())
        self.assertEqual(row['ws'], 100.0)
        self.assertEqual(row['words'][1], ["there", 0.3, 0.6])

    def test_disabled(self):
        ctl, _ = self._ctl()
        ctl.enabled = False
        ctl.on_caption_texts(["LeBron"], time.monotonic())
        ctl.on_asr_words("LeBron", [("LeBron", 0, .4)], time.monotonic())
        self.assertEqual(ctl.caption_hits + ctl.asr_hits, 0)


class TestPartialNames(unittest.TestCase):
    """A cut-off "le…"/"lebr…" at the end of a window mutes provisionally."""

    def _ctl(self, delay=2.0, partial=True):
        sch = NameMuteScheduler(FakeAudio(), lambda: delay)
        ctl = NameMuteController(sch, partial_names=partial)
        ctl.window_s = 3.0
        return ctl, sch

    def test_fragment_at_window_end_mutes(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("what you know about leb",
                         [("what", 1.0, 1.2), ("you", 1.2, 1.4), ("know", 1.4, 1.7),
                          ("about", 1.7, 2.6), ("leb", 2.7, 3.0)], t0)
        self.assertEqual(ctl.partial_hits, 1)
        self.assertEqual(sch._windows[0][2], 'asr_partial')
        start, end = sch._windows[0][:2]
        self.assertAlmostEqual(start, t0 + 2.7 + 2.0 - sch.PAD_BEFORE_S, delta=0.01)
        self.assertAlmostEqual(end - start, ctl.PARTIAL_MUTE_S + sch.PAD_BEFORE_S + sch.PAD_AFTER_S,
                               delta=0.01)

    def test_fragment_not_at_edge_ignored(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("le", [("le", 1.0, 1.2)], t0)
        ctl.on_asr_words("le monde said", [("le", 0.5, 0.7), ("monde", 0.7, 1.0), ("said", 1.0, 1.3)], t0 + 1)
        self.assertEqual(ctl.partial_hits, 0)
        self.assertEqual(sch._windows, [])

    def test_other_words_ending_window_ignored(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        for frag in ("left", "let", "lead", "lebanon's", "l"):
            ctl.on_asr_words(f"he went {frag}", [("he", 1.0, 1.5), ("went", 1.5, 2.6), (frag, 2.7, 3.0)], t0)
        self.assertEqual(ctl.partial_hits, 0)

    def test_next_window_cancels_wrong_guess(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("he went le", [("he", 1.0, 1.5), ("went", 1.5, 2.6), ("le", 2.7, 3.0)], t0)
        self.assertEqual(len(sch._windows), 1)
        # Half a second later the same stretch is heard whole: "left".
        ctl.on_asr_words("went left the court today",
                         [("went", 1.0, 2.1), ("left", 2.2, 2.6), ("the", 2.6, 2.7),
                          ("court", 2.7, 2.95), ("today", 2.95, 3.0)], t0 + 0.5)
        self.assertEqual(sch._windows, [])
        self.assertEqual(ctl.partial_cancels, 1)

    def test_next_window_with_name_replaces_guess(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("know about leb", [("know", 1.0, 1.5), ("about", 1.5, 2.6), ("leb", 2.7, 3.0)], t0)
        ctl.on_asr_words("about lebron james", [("about", 1.0, 2.1), ("lebron", 2.2, 2.6),
                                                ("james", 2.6, 2.9)], t0 + 0.5)
        self.assertEqual(ctl.partial_cancels, 0)
        self.assertEqual(ctl.asr_hits, 1)
        # The provisional mute gives way to the heard name's own span.
        self.assertEqual([w[2] for w in sch._windows], ['asr'])
        start, end = sch._windows[0][:2]
        self.assertAlmostEqual(start, t0 + 0.5 + 2.2 + 2.0 - sch.PAD_BEFORE_S, delta=0.01)
        self.assertAlmostEqual(end, t0 + 0.5 + 2.9 + 2.0 + sch.PAD_AFTER_S, delta=0.01)


    def test_started_mute_is_cut_short_not_dropped(self):
        ctl, sch = self._ctl(delay=0.0)
        t0 = time.monotonic() - 3.0
        ctl.on_asr_words("he went le", [("he", 1.0, 1.5), ("went", 1.5, 2.6), ("le", 2.7, 3.0)], t0)
        ctl.on_asr_words("went left the court",
                         [("went", 1.0, 2.1), ("left", 2.2, 2.6), ("the", 2.6, 2.7), ("court", 2.7, 3.0)],
                         t0 + 0.5)
        self.assertEqual(ctl.partial_cancels, 1)
        for w in sch._windows:
            self.assertLessEqual(w[1], time.monotonic() + 0.01)

    def test_off_by_default_and_env_override(self):
        sch = NameMuteScheduler(FakeAudio(), lambda: 2.0)
        self.assertFalse(NameMuteController(sch).partial_names)
        old = os.environ.get('MINUS_NAME_MUTE_PARTIAL')
        try:
            os.environ['MINUS_NAME_MUTE_PARTIAL'] = '0'
            self.assertFalse(NameMuteController(sch, partial_names=True).partial_names)
            os.environ['MINUS_NAME_MUTE_PARTIAL'] = '1'
            self.assertTrue(NameMuteController(sch).partial_names)
        finally:
            if old is None:
                os.environ.pop('MINUS_NAME_MUTE_PARTIAL', None)
            else:
                os.environ['MINUS_NAME_MUTE_PARTIAL'] = old

    def test_disabled_does_nothing(self):
        ctl, sch = self._ctl(partial=False)
        t0 = time.monotonic()
        ctl.on_asr_words("know about leb", [("know", 1.0, 1.5), ("about", 1.5, 2.6), ("leb", 2.7, 3.0)], t0)
        self.assertEqual(sch._windows, [])


class TestNarrowing(unittest.TestCase):
    """A precise ASR word span narrows coarser mutes for the same mention."""

    def _ctl(self, delay=2.0):
        sch = NameMuteScheduler(FakeAudio(), lambda: delay)
        ctl = NameMuteController(sch)
        ctl.window_s = 3.0
        return ctl, sch

    def test_whole_sighting_narrows_cut_off_word(self):
        ctl, sch = self._ctl()
        t0 = time.monotonic()
        ctl.on_asr_words("said lebron", [("said", 2.0, 2.4), ("lebron", 2.5, 3.0)], t0)
        self.assertEqual([w[2] for w in sch._windows], ['asr_edge'])
        ctl.on_asr_words("said lebron today", [("said", 1.5, 1.9), ("lebron", 2.0, 2.4),
                                                ("today", 2.4, 2.8)], t0 + 0.5)
        self.assertEqual([w[2] for w in sch._windows], ['asr'])
        self.assertAlmostEqual(sch._windows[0][1], t0 + 0.5 + 2.4 + 2.0 + sch.PAD_AFTER_S, delta=0.01)
        self.assertEqual(ctl.asr_hits, 1)
        # Later repeats of the whole word change nothing.
        ctl.on_asr_words("lebron today", [("lebron", 1.5, 1.9), ("today", 1.9, 2.3)], t0 + 1.0)
        self.assertEqual(len(sch._windows), 1)

    def test_playing_caption_mute_is_cut_to_the_word(self):
        ctl, sch = self._ctl(delay=0.0)
        now = time.monotonic()
        ctl.on_caption_texts(["here comes LeBron again"], now)   # mutes now-0.7 .. now+0.7
        self.assertEqual([w[2] for w in sch._windows], ['caption'])
        ctl.on_asr_words("comes lebron", [("comes", 0.5, 0.9), ("lebron", 1.0, 1.3)], now - 1.5)
        end = sch._windows[0][1]
        self.assertLess(end, now + 0.7 - 0.2)
        log = [d for d in sch.detection_log if d['source'] == 'caption'][0]
        self.assertEqual(log['play_end'], end)
        self.assertTrue(log['narrowed'])

    def test_dropped_window_logged_as_zero_length(self):
        ctl, sch = self._ctl()
        now = time.monotonic()
        ctl.on_caption_texts(["here comes LeBron again"], now)
        ctl.on_asr_words("comes lebron", [("comes", 0.5, 0.9), ("lebron", 1.0, 1.3)], now - 1.5)
        self.assertEqual([w[2] for w in sch._windows], ['asr'])
        log = [d for d in sch.detection_log if d['source'] == 'caption'][0]
        self.assertEqual(log['play_end'], log['play_start'])


class TestBanners(unittest.TestCase):
    """On-screen banners naming him are not spoken mentions."""

    def _ctl(self, delay=2.0):
        sch = NameMuteScheduler(FakeAudio(), lambda: delay)
        ctl = NameMuteController(sch)
        ctl.window_s = 3.0
        return ctl, sch

    def test_banner_misreads_mute_once(self):
        ctl, sch = self._ctl()
        t = time.monotonic()
        reads = ["Skip's Twitter explodes after questioning LeBron's GOAT status",
                 "stioning LeBron's GOAT status", "ig LeBron's GOAT status",
                 "ing LeBron's GOAT status", "g LeBron's GOAT status",
                 "questiofing LeBron's GOAT status"]
        for i, r in enumerate(reads * 3):           # 18 frames over 27 s
            ctl.on_caption_texts([r], t + 1.5 * i)
        self.assertEqual(ctl.caption_hits, 1)

    def test_mid_line_hit_cancelled_when_not_spoken(self):
        ctl, sch = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["Is LeBron James really better than Michael Jordan?"], t)
        self.assertEqual([w[2] for w in sch._windows], ['caption_mid'])
        # The ASR heard that moment clearly, and no name in it.
        ctl.on_asr_words("the thing is he was dominant in that",
                         [("the", 0.3, 0.5), ("thing", 0.5, 0.8), ("is", 0.8, 0.9),
                          ("he", 0.9, 1.0), ("was", 1.0, 1.2), ("dominant", 1.2, 1.7),
                          ("in", 1.7, 1.8), ("that", 1.8, 2.5)], t - 2.0)
        self.assertEqual(sch._windows, [])
        self.assertEqual(ctl.caption_mid_cancels, 1)

    def test_rolling_caption_at_line_end_not_provisional(self):
        ctl, sch = self._ctl()
        t = time.monotonic()
        ctl.on_caption_texts(["what they did for LeBron"], t)
        ctl.on_asr_words("the thing is he was dominant in that",
                         [("the", 0.3, 0.5), ("thing", 0.5, 0.8), ("is", 0.8, 0.9),
                          ("he", 0.9, 1.0), ("was", 1.0, 1.2), ("dominant", 1.2, 1.7),
                          ("in", 1.7, 1.8), ("that", 1.8, 2.5)], t - 2.0)
        self.assertEqual([w[2] for w in sch._windows], ['caption'])


if __name__ == '__main__':
    unittest.main()