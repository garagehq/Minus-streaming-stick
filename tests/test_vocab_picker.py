#!/usr/bin/env python3
"""Spanish words during blocks: variety, persistence, and characters that draw.

Owner, 2026-10-18: "I'm seeing a lot of the same words and phrases". The
picker was random.choice() over the list (repeats within ~33 blocks for ~800
words) and each word always showed the same sentences.
"""

import json
import random
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import vocab_picker as vp  # noqa: E402
from vocabulary import SPANISH_VOCABULARY, VOCABULARY_COMBINED  # noqa: E402


def entries(n):
    return [(f'w{i}', 'p', 'e', (f'a{i}', f'b{i}', f'c{i}')) for i in range(n)]


class TestShuffleBag(unittest.TestCase):

    def test_no_repeat_until_every_word_was_shown(self):
        bag = vp.VocabBag(entries(50), path=None, rng=random.Random(1))
        first = [bag.next()[0] for _ in range(50)]
        self.assertEqual(len(set(first)), 50)
        second = [bag.next()[0] for _ in range(50)]
        self.assertEqual(sorted(second), sorted(first))
        self.assertNotEqual(first[-1], second[0], 'a round must not open with the last word')

    def test_restart_keeps_the_place(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d, 'bag.json'))
            bag = vp.VocabBag(entries(30), path=path, rng=random.Random(2))
            shown = {bag.next()[0] for _ in range(20)}
            bag2 = vp.VocabBag(entries(30), path=path, rng=random.Random(3))
            rest = {bag2.next()[0] for _ in range(10)}
            self.assertEqual(shown & rest, set(), 'a restart showed a word again too soon')
            self.assertEqual(len(shown | rest), 30)

    def test_new_words_join_the_unseen_pile(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d, 'bag.json'))
            bag = vp.VocabBag(entries(10), path=path, rng=random.Random(4))
            shown = {bag.next()[0] for _ in range(8)}
            bigger = entries(15)
            bag2 = vp.VocabBag(bigger, path=path, rng=random.Random(5))
            rest = {bag2.next()[0] for _ in range(7)}
            self.assertEqual(shown | rest, {e[0] for e in bigger})

    def test_corrupt_or_unwritable_state_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, 'bag.json')
            p.write_text('{not json')
            bag = vp.VocabBag(entries(5), path=str(p))
            self.assertEqual(len({bag.next()[0] for _ in range(5)}), 5)
        bag = vp.VocabBag(entries(5), path='/proc/nope/bag.json')
        self.assertEqual(len({bag.next()[0] for _ in range(5)}), 5)


class TestExamples(unittest.TestCase):

    def test_two_random_examples(self):
        rng = random.Random(7)
        seen = {tuple(vp.pick_examples(('a', 'b', 'c', 'd'), rng=rng)) for _ in range(60)}
        self.assertTrue(all(len(s) == 2 for s in seen))
        self.assertGreater(len(seen), 6, 'the pairs should vary')

    def test_short_lists(self):
        self.assertEqual(vp.pick_examples(('only',)), ['only'])


class TestLatin1(unittest.TestCase):
    """ustreamer draws one byte per glyph; Latin-1 bytes == code points."""

    def test_spanish_survives(self):
        s = '¿Mañana? ¡Sí, señor! Ünico ó á'
        self.assertEqual(vp.latin1_safe(s), s)

    def test_other_characters_degrade_gracefully(self):
        self.assertEqual(vp.latin1_safe('Hola — “sí”…'), 'Hola - "sí"...')
        self.assertEqual(vp.latin1_safe('能加'), '??')
        self.assertEqual(vp.latin1_safe(5), 5)

    def test_api_call_sends_latin1(self):
        import urllib.parse
        from ad_blocker import DRMAdBlocker
        from unittest.mock import patch, MagicMock
        b = object.__new__(DRMAdBlocker)
        b.ustreamer_port = 9090
        seen = []
        resp = MagicMock()
        resp.__enter__.return_value.read.return_value = b'{"ok": true}'
        with patch('urllib.request.urlopen', side_effect=lambda req, timeout: seen.append(req.full_url) or resp):
            b._blocking_api_call('/blocking/set', {'text_vocab': 'mañana'})
        self.assertIn('ma%F1ana', seen[0])     # Latin-1 0xF1, not UTF-8 %C3%B1


class TestVocabularyData(unittest.TestCase):

    def test_shape_and_limits(self):
        self.assertIs(VOCABULARY_COMBINED, SPANISH_VOCABULARY)
        self.assertGreater(len(SPANISH_VOCABULARY), 1200)
        for sp, pron, en, examples in SPANISH_VOCABULARY:
            self.assertTrue(sp and pron and en, sp)
            self.assertIsInstance(examples, tuple)
            self.assertGreaterEqual(len(examples), 1, sp)
            for ex in examples:
                self.assertLessEqual(len(ex), 48, ex)
                ex.encode('latin-1')
            (sp + pron + en).encode('latin-1')

    def test_no_duplicate_headwords_or_sentences(self):
        def norm(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                           if unicodedata.category(c) != 'Mn')
        heads = [norm(e[0]) for e in SPANISH_VOCABULARY]
        self.assertEqual(len(heads), len(set(heads)))
        sentences = [norm(x) for e in SPANISH_VOCABULARY for x in e[3]]
        self.assertEqual(len(sentences), len(set(sentences)))

    def test_most_words_have_several_examples(self):
        many = sum(1 for e in SPANISH_VOCABULARY if len(e[3]) >= 3)
        self.assertGreater(many / len(SPANISH_VOCABULARY), 0.9)

    def test_spelling_has_accents_now(self):
        text = ' '.join(x for e in SPANISH_VOCABULARY for x in e[3])
        for w in ('mañana', 'está', 'qué'):
            self.assertIn(w, text.lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
