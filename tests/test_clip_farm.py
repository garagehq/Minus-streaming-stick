#!/usr/bin/env python3
"""Unit tests for tools/clip_farm.py helpers (no TV needed)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import clip_farm as cf  # noqa: E402

CAPTIONS = ("that when you look at LeBron James and his greatness compared to "
            "Michael Jordan the physicality the defense Pippen embraces").split()


class TestCaptionMatch(unittest.TestCase):
    def test_matching_audio_scores_high(self):
        heard = ["look at lebron james and his greatness", "compared to michael jordan",
                 "pippen embraces the defense"]
        self.assertGreater(cf.caption_match(heard, CAPTIONS), 0.8)

    def test_ad_audio_scores_low(self):
        heard = ["parts and labor included in my first year", "your rate stays the same",
                 "nearly two decades endurance just gives me"]
        self.assertLess(cf.caption_match(heard, CAPTIONS), cf.MISMATCH_MAX)

    def test_wrong_video_sharing_common_words_scores_low(self):
        # A golf lesson played instead of the queued video: only common words
        # like 'that' overlap (scored ~0.15-0.2, above the old 0.15 limit).
        heard = ["that wedge setup", "thinking about contact"]
        m = cf.caption_match(heard, CAPTIONS)
        self.assertGreater(m, 0.15)
        self.assertLess(m, cf.MISMATCH_MAX)

    def test_too_little_heard_is_none(self):
        self.assertIsNone(cf.caption_match(["and the", "lebron"], CAPTIONS))

    def test_short_words_ignored(self):
        # Only words of 4+ letters count, so 'the and is' can't make an ad look like a match.
        heard = ["the and is of to the and", "scotland lighthouse keepers peninsula",
                 "highlands weather tidal island"]
        self.assertEqual(cf.caption_match(heard, CAPTIONS), 0.0)


class TestResync(unittest.TestCase):
    # caption words (word, start_s, end_s) around 600s
    CAP = [(w, 600 + i, 600.5 + i) for i, w in enumerate(CAPTIONS)]

    def test_same_video_at_new_position_is_resync(self):
        window = [(602, "look at lebron james and his greatness"),
                  (606, "compared to michael jordan"), (610, "pippen embraces the defense")]
        self.assertTrue(cf.is_resync(window, self.CAP))

    def test_ad_during_drop_is_not_resync(self):
        window = [(5, "parts and labor included in my first year"),
                  (9, "your rate stays the same"), (13, "nearly two decades endurance just gives me")]
        self.assertFalse(cf.is_resync(window, self.CAP))

    def test_nothing_heard_is_not_resync(self):
        self.assertFalse(cf.is_resync([(602, "")], self.CAP))
        self.assertFalse(cf.is_resync([], self.CAP))


class TestNicknames(unittest.TestCase):
    def test_find_nicknames(self):
        lc = cf.label_clips
        self.assertEqual(lc.find_nicknames("L.B.J. -- the Chosen One, Bron-Bron!"),
                         ['bron bron', 'chosen one', 'lbj'])
        self.assertEqual(lc.find_nicknames("the Kings and the king, brain train"), [])

    def test_nickname_hits_give_caption_time(self):
        words = [('and', 4.0, 4.2), ('Captain', 5.0, 5.4), ('LeMerica', 5.4, 6.0), ('L-Train', 9.0, 9.5)]
        self.assertEqual(cf.label_clips.nickname_hits(words),
                         [('captain lemerica', 5.0), ('l train', 9.0)])

    def test_request_window(self):
        sent = []
        orig = cf.api
        cf.api = lambda path, body=None, timeout=5: sent.append(body)
        try:
            hits = [('lbj', 10.0), ('chosen one', 30.0), ('king james', 60.0)]
            nxt = cf.request_nickname_clips('vid', hits, 5.0, 35.0, 1000.0)
        finally:
            cf.api = orig
        self.assertEqual(nxt, 35.0)
        # 10.0 is 25 s back (outside the tap buffer); 60.0 hasn't played yet.
        self.assertEqual([round(b['clip_at'], 1) for b in sent], [995.0])
        self.assertIn('chosen one', sent[0]['label'])


if __name__ == '__main__':
    unittest.main()
