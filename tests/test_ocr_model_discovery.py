#!/usr/bin/env python3
"""OCR model discovery must work for whichever generation is present.

Two checks used to hardcode PP-OCRv3 filenames:

  * config.OCR_MODEL_DIR only chose the in-repo folder when it contained a
    ppocrv3_det_*.rknn, otherwise falling back to the legacy install path.
  * Minus._find_model_paths() -- which decides whether OCR starts AT ALL --
    looked only for ppocrv3_* files, in its own folder list.

So a checkout carrying just the default v6 models logged "OCR model files not
found" and ran with OCR off, while the OCR worker (which uses
resolve_ocr_models) would have loaded them fine. Not hit in practice only
because both generations ship. Both checks now defer to the same logic.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

V6 = ('ppocrv6_det_rk3588_x.rknn', 'ppocrv6_rec_rk3588_x.rknn', 'ppocrv6_keys.txt')
V3 = ('ppocrv3_det_rk3588_x.rknn', 'ppocrv3_rec_rk3588_x.rknn', 'ppocr_keys_v1.txt')


def make_dir(*file_sets, skip=()):
    d = Path(tempfile.mkdtemp())
    for files in file_sets:
        for f in files:
            if f not in skip:
                (d / f).write_bytes(b'x')
    return d


class TestHasOcrModels(unittest.TestCase):

    def test_v6_only(self):
        self.assertTrue(config._has_ocr_models(make_dir(V6)))

    def test_v3_only(self):
        self.assertTrue(config._has_ocr_models(make_dir(V3)))

    def test_both(self):
        self.assertTrue(config._has_ocr_models(make_dir(V6, V3)))

    def test_empty(self):
        self.assertFalse(config._has_ocr_models(make_dir()))

    def test_incomplete_set_is_not_a_model_set(self):
        """A detector without its dictionary cannot be used."""
        self.assertFalse(config._has_ocr_models(make_dir(V6, skip=('ppocrv6_keys.txt',))))

    def test_mismatched_halves_are_not_a_model_set(self):
        """v6 det + v3 rec/dict is not a usable generation."""
        d = make_dir((V6[0], V3[1], V3[2]))
        self.assertFalse(config._has_ocr_models(d))

    def test_missing_directory(self):
        self.assertFalse(config._has_ocr_models('/nonexistent/ocr/models'))


class TestDefaultModelDir(unittest.TestCase):

    def test_v6_only_repo_is_used(self):
        """The regression: this used to fall back to the legacy path."""
        repo = make_dir(V6)
        self.assertEqual(config._default_ocr_model_dir(repo, '/legacy'), str(repo))

    def test_v3_only_repo_is_used(self):
        repo = make_dir(V3)
        self.assertEqual(config._default_ocr_model_dir(repo, '/legacy'), str(repo))

    def test_empty_repo_falls_back_to_legacy(self):
        self.assertEqual(config._default_ocr_model_dir(make_dir(), '/legacy'), '/legacy')

    def test_shipped_repo_folder_is_chosen(self):
        self.assertEqual(config._default_ocr_model_dir(),
                         str(ROOT / 'models' / 'paddleocr'))


class TestResolveMatchesWhatShips(unittest.TestCase):

    def test_v6_only_resolves_v6(self):
        m = config.resolve_ocr_models(make_dir(V6), 'v6')
        self.assertEqual(m['version'], 'v6')

    def test_auto_prefers_v6(self):
        self.assertEqual(config.resolve_ocr_models(make_dir(V6, V3), 'auto')['version'], 'v6')

    def test_auto_falls_back_to_v3(self):
        self.assertEqual(config.resolve_ocr_models(make_dir(V3), 'auto')['version'], 'v3')

    def test_requesting_an_absent_generation_finds_nothing(self):
        """An explicit v3 request in a v6-only folder is not silently swapped."""
        self.assertIsNone(config.resolve_ocr_models(make_dir(V6), 'v3'))

    def test_repo_default_is_v6(self):
        self.assertEqual(config.resolve_ocr_models()['version'], 'v6')


class TestStartupGateAgreesWithTheWorker(unittest.TestCase):
    """Minus._find_model_paths decides whether OCR starts at all."""

    def _find(self, resolved):
        import minus as m
        o = object.__new__(m.Minus)
        with patch('config.resolve_ocr_models', return_value=resolved):
            return m.Minus._find_model_paths(o)

    def test_v6_only_starts_ocr(self):
        d = make_dir(V6)
        det, rec, dct = self._find(config.resolve_ocr_models(d, 'v6'))
        self.assertTrue(det and det.endswith(V6[0]))
        self.assertTrue(dct.endswith('ppocrv6_keys.txt'))

    def test_no_models_disables_ocr(self):
        self.assertEqual(self._find(None), (None, None, None))

    def test_gate_uses_the_workers_resolver(self):
        """The two must not be able to disagree again."""
        src = (ROOT / 'minus.py').read_text()
        body = src[src.index('    def _find_model_paths(self):'):]
        body = body[:body.index('\n    def ', 10)]
        self.assertIn('resolve_ocr_models', body)
        self.assertNotIn('ppocrv3_det_', body)
        worker = (ROOT / 'src' / 'ocr_worker.py').read_text()
        self.assertIn('resolve_ocr_models()', worker)

    def test_live_repo_starts_ocr_with_v6(self):
        import minus as m
        o = object.__new__(m.Minus)
        det, rec, dct = m.Minus._find_model_paths(o)
        self.assertIn('ppocrv6_det_', det)


if __name__ == "__main__":
    unittest.main(verbosity=2)
