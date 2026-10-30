#!/usr/bin/env python3
"""'Ad N' countdowns match in any real casing, but not OCR's 'aD' misreads.

minus, 2026-10-29: a gas station's LED price sign (3.05 / 3.08) in a music
video read as "aD88" and matched the countdown rule: a 5 s false block while
the VLM said no at 98-100%.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

ocr = pytest.importorskip('ocr')


def countdown(text):
    _found, matched, _texts, _term = ocr.PaddleOCR.check_ad_keywords([{'text': text}])
    return any(k == 'ad countdown' for k, _ in matched)


@pytest.mark.parametrize('text', ['Ad 10', 'AD 30', 'ad 5', 'Ad5', 'AD15'])
def test_real_countdowns_match(text):
    assert countdown(text)


@pytest.mark.parametrize('text', ['aD88', 'aD 30'])
def test_misread_casing_does_not_match(text):
    assert not countdown(text)
