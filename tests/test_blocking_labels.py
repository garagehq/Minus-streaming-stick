"""The per-frame log tag names every signal behind the active block, ASR included.

Reported 2026-10-29: the `OCR #N: ... [BLOCKING ...]` tag (the log line people
watch) only ever said OCR or OCR+VLM. It was built from the OCR and VLM flags
alone, while ASR confirmed blocks dozens of times a day (the overlay header and
"ASR confirmed active block -> OCR+VLM+ASR" did show it). It now uses the same
OCR / VLM / ASR parts as the overlay label.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _m(ad=True, ocr=False, vlm=False, asr=False, static=False, source='ocr'):
    import minus as minus_mod
    m = minus_mod.Minus.__new__(minus_mod.Minus)
    m.ad_detected = ad
    m.ocr_ad_detected = ocr
    m.vlm_ad_detected = vlm
    m.blocking_asr_confirmed = asr
    m.static_blocking_suppressed = static
    m.blocking_source = source
    return m


class TestOcrLogBlockingTag(unittest.TestCase):
    def test_not_blocking(self):
        self.assertEqual(_m(ad=False)._ocr_log_blocking_tag(), "")

    def test_static_suppressed(self):
        self.assertEqual(_m(ocr=True, static=True)._ocr_log_blocking_tag(),
                         " [AD DETECTED - STATIC SUPPRESSED]")

    def test_without_asr_unchanged(self):
        self.assertEqual(_m(ocr=True)._ocr_log_blocking_tag(), " [BLOCKING OCR]")
        self.assertEqual(_m(vlm=True)._ocr_log_blocking_tag(), " [BLOCKING VLM]")
        self.assertEqual(_m(ocr=True, vlm=True)._ocr_log_blocking_tag(), " [BLOCKING OCR+VLM]")

    def test_asr_confirmed_is_named(self):
        self.assertEqual(_m(ocr=True, asr=True)._ocr_log_blocking_tag(), " [BLOCKING OCR+ASR]")
        self.assertEqual(_m(vlm=True, asr=True)._ocr_log_blocking_tag(), " [BLOCKING VLM+ASR]")
        self.assertEqual(_m(ocr=True, vlm=True, asr=True)._ocr_log_blocking_tag(),
                         " [BLOCKING OCR+VLM+ASR]")

    def test_held_block_with_only_asr(self):
        # OCR/VLM can drop out mid-block while the block is held; ASR still backs it.
        self.assertEqual(_m(asr=True)._ocr_log_blocking_tag(), " [BLOCKING ASR]")


if __name__ == '__main__':
    unittest.main()
