# PaddleOCR PP-OCRv3 models for the RK3588 NPU

These are the text-detection and text-recognition models Minus runs on the
RK3588's built-in NPU (via `rknnlite`) to read on-screen ad UI such as
"Skip in 5", "Ad 2 of 3", "Sponsored" and countdown timers. They are
PP-OCRv3 (English) models exported from PaddleOCR to ONNX and converted to
RKNN with `rknn-toolkit2` for the RK3588 (`rknpu2` 2.3.0 runtime). They
are shipped in the repository so a fresh clone has everything the OCR path
needs; no download step is required.

| File | Size | MD5 | Role |
|---|---|---|---|
| `ppocrv3_det_rk3588_2025-12-20.rknn` | 3.4 MB | `b76c87db6aac30415391865e53b328a4` | **Detection** (required). Finds text regions in the 960x540 frame. Input 960x960. |
| `ppocrv3_rec_rk3588_2025-12-20.rknn` | 6.5 MB | `d5671fffd4b94f6ab8f76719d67c992d` | **Recognition** (required). Reads each detected box into a string. Input 48x320 per box. |
| `ppocr_keys_v1.txt` | 26 KB | `0eecb02b7413c25597dfc06c56b7242a` | **Character dictionary** (required). The recognition output is CTC-decoded through this list. |
| `ppocrv3_cls_rk3588_2025-12-20.rknn` | 1.0 MB | `5e4017182f4e23b8f0adb02117177a7a` | Direction classifier (optional, **not loaded by Minus**). Detects 180-degree rotated text. TV overlay text is never rotated, so it is skipped to save one NPU pass per box. |

## How Minus finds them

`src/config.py` sets `OCR_MODEL_DIR` to this directory whenever it contains a
`ppocrv3_det_*.rknn`, and only falls back to the legacy
`rknn-llm/.../models/paddleocr` path when it does not. Override either with
`MINUS_OCR_MODEL_DIR=/some/dir`. The loader globs `ppocrv3_det_*.rknn`,
`ppocrv3_rec_*.rknn` and `ppocr_keys_v1.txt`, so a re-converted model with a
new date suffix drops in without a code change (keep exactly one of each in
the directory). Both are loaded in the OCR worker subprocess
(`src/ocr_worker.py`), each with its own `RKNNLite` runtime.

## Pipeline

```
frame 960x540 ──▶ det ──▶ N text boxes ──▶ crop each ──▶ rec ──▶ strings ──▶ keyword matcher
                 ~100ms                                  ~30ms/box            (src/ocr.py)
```

Latency scales with the number of boxes, which is why text-dense screens
(scoreboards, credits, menu grids) run slowest; the p99.9 round-trip is
~1.0 s and the worker's hard timeout is 1.5 s (`--ocr-timeout`).

## Re-converting

The conversion is the standard rknn-toolkit2 flow on the official
PP-OCRv3 ONNX exports (`en_PP-OCRv3_det` / `en_PP-OCRv3_rec`):
`rknn.config(target_platform='rk3588')` → `load_onnx` → `build` →
`export_rknn`. Keep the filename prefixes above so the glob still matches.
Verify a new build with:

```bash
curl -s -X POST http://localhost/api/ocr/test | python3 -m json.tool
```
