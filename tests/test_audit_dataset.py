"""Tests for tools/audit_dataset.py's review logic (no ASR model needed)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import audit_dataset as A


def tr(words, step=0.4, t0=0.0):
    return {'text': ' '.join(words),
            'words': [(w, t0 + k * step, t0 + k * step + step * 0.8) for k, w in enumerate(words)]}


def row(text, duration=6.0, overlap=0.9):
    return {'text': text, 'duration': duration, 'asr_overlap': overlap}


class TestDistances(unittest.TestCase):
    def test_lev(self):
        self.assertEqual(A.lev('kitten', 'sitting'), 3)
        self.assertEqual(A.lev('', 'abc'), 3)
        self.assertEqual(A.lev('lillard', 'willard'), 1)

    def test_align(self):
        ops = A.align('a b c d'.split(), 'x a c d'.split()[1:] + ['e'])
        self.assertEqual([o[0] for o in ops], ['=', 'd', '=', '=', 'i'])


class TestDecide(unittest.TestCase):
    def test_matching_label_kept(self):
        d = A.decide(row('and lebron drives to the rim'), tr('and lebron drives to the rim'.split()))
        self.assertEqual(d['action'], 'keep')

    def test_edge_words_trimmed(self):
        words = 'the and lebron drives to the rim he'.split()
        d = A.decide(row('and lebron drives to the rim', duration=3.4), tr(words))
        self.assertEqual(d['action'], 'fix')
        a, b = d['trim']
        self.assertTrue(0.32 <= a < 0.4)          # after "the", before "and"
        self.assertTrue(2.4 < b <= 2.8)           # after "rim", before "he"

    def test_garbled_label_dropped(self):
        d = A.decide(row("we ain't the money baby come on goals tonight"),
                     tr('we hit the buddy baby come on go ain\'t no telling what i might do'.split()))
        self.assertEqual(d['action'], 'drop')

    def test_parakeet_missing_the_name_is_not_held_against_the_label(self):
        d = A.decide(row('and lebron drives to the rim'), tr('and la bron drives to the rim'.split()))
        self.assertNotEqual(d['action'], 'drop')

    def test_name_misspelling_fixed_when_anchored(self):
        d = A.decide(row('a lot of bran fans love this'), tr('a lot of lebron fans love this'.split()))
        self.assertEqual(d['text'], 'a lot of lebron fans love this')
        self.assertEqual(d['name_fixes'], [['bran', 'lebron']])

    def test_possessive_and_nickname_not_changed(self):
        d = A.decide(row("the series lebron's stats were great"), tr('the series lebron stats were great'.split()))
        self.assertFalse(d.get('name_fixes'))
        d = A.decide(row('cavs sneak one out bron had twelve'), tr('cavs sneak one out lebron had twelve'.split()))
        self.assertFalse(d.get('name_fixes'))

    def test_empty_parakeet_transcript_falls_back_to_sensevoice(self):
        empty = {'text': '', 'words': []}
        self.assertEqual(A.decide(row('in a world of trouble', overlap=1.0), empty)['action'], 'keep')
        self.assertEqual(A.decide(row('in a world of trouble', overlap=0.6), empty)['action'], 'drop')

    def test_unanchored_name_substitution_left_alone(self):
        d = A.decide(row('clock two times jab step put up three'),
                     tr('class two times james jabstep put up three'.split()))
        self.assertFalse(d.get('name_fixes'))

    def test_relabelled_edge_words_parakeet_did_not_hear_dropped(self):
        d = A.decide(row('at the end of games lebron shies away from driving because he is afraid of being sent'),
                     tr('at the end of games lebron shies away from driving because he is afraid of'.split()),
                     relabelled=True)
        self.assertEqual(d['action'], 'fix')
        self.assertEqual(d['text'], 'at the end of games lebron shies away from driving because he is afraid of')


if __name__ == '__main__':
    unittest.main()
