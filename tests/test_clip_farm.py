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

    def test_too_little_heard_is_none(self):
        self.assertIsNone(cf.caption_match(["and the", "lebron"], CAPTIONS))

    def test_short_words_ignored(self):
        # Only words of 4+ letters count, so 'the and is' can't make an ad look like a match.
        heard = ["the and is of to the and", "scotland lighthouse keepers peninsula",
                 "highlands weather tidal island"]
        self.assertEqual(cf.caption_match(heard, CAPTIONS), 0.0)


if __name__ == '__main__':
    unittest.main()
