# PaddleOCR models for the RK3588 NPU

These are the text-detection and text-recognition models Minus runs on the
RK3588's built-in NPU (via `rknnlite`) to read on-screen ad UI such as
"Skip in 5", "Ad 2 of 3", "Sponsored" and countdown timers. They are shipped
in the repository so a fresh clone has everything the OCR path needs; no
download step is required.

Two generations ship side by side:

- **PP-OCRv6 tiny — the default.** Chosen on latency: its tighter tail keeps
  text-dense ad frames under the hard timeout. See
  [*PP-OCRv6 is the default*](#pp-ocrv6-is-the-default-sep-2026) below.
- **PP-OCRv3 — kept as a rollback**, selected with `MINUS_OCR_MODEL_VERSION=v3`.

Both were exported from PaddleOCR to ONNX and converted to RKNN with
`rknn-toolkit2` for the RK3588 (`rknpu2` 2.3.0 runtime).

| File | Size | MD5 | Role |
|---|---|---|---|
| `ppocrv6_det_rk3588_2026-09-09.rknn` | 3.4 MB | `5d6232e19deac5cf7b6eb762e3c2f0c4` | **v6 detection** (default). Finds text regions in the 960x540 frame. Input 960x960. |
| `ppocrv6_rec_rk3588_2026-09-09.rknn` | 2.9 MB | `5006c455f2503a04f11a553e3a19de42` | **v6 recognition** (default). Reads each detected box into a string. Input 48x320 per box. |
| `ppocrv6_keys.txt` | 27 KB | `c98660b3c39317cab66a66ff91fd777f` | **v6 character dictionary** (default). 6904 entries. |
| `ppocrv3_det_rk3588_2025-12-20.rknn` | 3.4 MB | `b76c87db6aac30415391865e53b328a4` | v3 detection (rollback). Input 960x960. |
| `ppocrv3_rec_rk3588_2025-12-20.rknn` | 6.5 MB | `d5671fffd4b94f6ab8f76719d67c992d` | v3 recognition (rollback). Input 48x320 per box. |
| `ppocr_keys_v1.txt` | 26 KB | `0eecb02b7413c25597dfc06c56b7242a` | v3 character dictionary (rollback). |
| `ppocrv3_cls_rk3588_2025-12-20.rknn` | 1.0 MB | `5e4017182f4e23b8f0adb02117177a7a` | Direction classifier (optional, **not loaded by Minus**). Detects 180-degree rotated text. TV overlay text is never rotated, so it is skipped to save one NPU pass per box. |

A generation's dictionary is **not** interchangeable with the other's: the v3
dictionary against v6 weights decodes to the wrong characters entirely.

## How Minus finds them

`src/config.py` sets `OCR_MODEL_DIR` to this directory whenever it holds a
complete model set for either generation, and otherwise falls back to the
legacy `rknn-llm/.../models/paddleocr` path. Override with
`MINUS_OCR_MODEL_DIR=/some/dir`. The startup check that decides whether OCR
runs at all uses the same resolver as the OCR worker, so the two cannot
disagree about whether models are present.

`MINUS_OCR_MODEL_VERSION` picks the generation: `v6` (default), `v3`, or
`auto` (v6 when present, else v3). `resolve_ocr_models()` binds each
generation's detection model, recognition model, dictionary **and** its own DB
post-process thresholds together, so they can never be mixed:

| generation | det glob | rec glob | dictionary | DB thresh / box_thresh / unclip |
|---|---|---|---|---|
| v6 | `ppocrv6_det_*.rknn` | `ppocrv6_rec_*.rknn` | `ppocrv6_keys.txt` | 0.2 / 0.4 / 1.4 |
| v3 | `ppocrv3_det_*.rknn` | `ppocrv3_rec_*.rknn` | `ppocr_keys_v1.txt` | 0.3 / 0.5 / 1.5 |

Because it globs, a re-converted model with a new date suffix drops in without
a code change (keep exactly one of each per generation). Both models load in
the OCR worker subprocess (`src/ocr_worker.py`), each with its own `RKNNLite`
runtime. The startup log names the generation: `Loading models (v6) from ...`.

## Pipeline

```
frame 960x540 ──▶ det ──▶ N text boxes ──▶ crop each ──▶ rec ──▶ strings ──▶ keyword matcher
                                                                              (src/ocr.py)
```

Measured in production with v6: **183 ms p50, 274 ms p90** per frame across
31k frames. Latency scales with the number of boxes, which is why text-dense
screens (scoreboards, credits, pharma fine print, menu grids) run slowest; the
worker's hard timeout is 1.5 s (`--ocr-timeout`).

## Re-converting

The conversion is the standard rknn-toolkit2 flow on the official PaddleOCR
ONNX exports: `rknn.config(target_platform='rk3588')` → `load_onnx` → `build`
→ `export_rknn`. Detection and recognition need **different** normalization
baked in at conversion time (see the audit below). Keep the filename prefixes
above so the globs still match. Verify a new build with:

```bash
curl -s -X POST http://localhost/api/ocr/test | python3 -m json.tool
```

## Input normalization audit (Sep 2026)

PaddleOCR uses **different** input normalization for detection and recognition,
and RKNN freezes that choice into the model at conversion time (`rknn.config`
mean/std). Reference:

| stage | source | mean | std |
|---|---|---|---|
| detection | `NormalizeImage`, ImageNet stats | 123.675, 116.28, 103.53 | 58.395, 57.12, 57.375 |
| recognition | `RecResizeImg`: `img/255; -=0.5; /=0.5` | 127.5, 127.5, 127.5 | 127.5, 127.5, 127.5 |

What the shipped models actually contain (read from model metadata; the audit
runs as `tests/test_ocr_model_normalization.py`):

| model | baked mean/std | correct |
|---|---|---|
| `ppocrv3_det` | ImageNet | yes |
| `ppocrv6_det` | ImageNet | yes |
| `ppocrv6_rec` | 127.5 | yes |
| `ppocrv3_rec` | **ImageNet** | **no — detection stats on a recognition model** |

### Why it is not hot-patched

There is no lossless runtime correction. `inputs_pass_through=1` **segfaults**
the RK3588 runtime (it wants NPU-native layout, not float32 NHWC), and
pre-compensating the uint8 input so the model's own normalization lands on the
right values spans only ~117 of 256 levels, trading correct scaling for a 2.2x
precision loss.

Measured on 40 real ad frames from `screenshots/ads/`, scored with the
production keyword matcher (frames flagged as an ad):

| variant | frames detected | mean conf |
|---|---|---|
| v3 as shipped | **39/40** | 0.630 |
| v3 grey-padded | 37/40 | 0.609 |
| v3 pre-compensated to 127.5 | 38/40 | 0.606 |
| v6 as shipped (correct) | 38/40 | 0.604 |

Correcting it measured no better. **Caveat that keeps this open rather than
closed:** `screenshots/ads/` only contains frames the *current* pipeline already
detected, so the corpus is biased toward the shipped settings. Read this as "no
evidence of harm", not "the deviation is fine".

### Fixing it properly

Re-convert the v3 rec ONNX with `mean_values=[[127.5]*3]`,
`std_values=[[127.5]*3]`, then A/B at full precision on a corpus that was not
collected with the current settings. The converter does not install on the
device: `rknn-toolkit2` 2.3.2 pins `torch<=2.2.0`, `onnx==1.16.1`,
`protobuf==3.20.3` and needs `onnxoptimizer`, which has no aarch64 wheel and
fails to build. Do it on the machine that built the v6 models. Source ONNX:
`PaddlePaddle/PP-OCRv3_mobile_rec` (Paddle inference format, needs paddle2onnx)
or the community `cycloneboy/ch_PP-OCRv3_rec_infer` (`model.onnx`, 6625-wide
output matching `ppocr_keys_v1.txt`). Verify provenance first by converting the
same ONNX with the ImageNet stats and checking it reproduces the current
`ppocrv3_rec` outputs; only then is the comparison a clean normalization A/B.

### Second deviation: recognition padding

PaddleOCR pads **after** normalizing, with 0.0 in normalized space, i.e. pixel
127.5 (mid grey). `src/ocr.py` pads **before** normalizing with pixel 0 (black),
which normalizes to -1.0. This affects **both** generations. It matters most for
exactly the text Minus cares about: a 60x22 "Skip" crop resized to 48x320 is
**59% padding**.

Measured neutral on the product metric (v6: 38/40 ad frames either way), so it
is documented rather than changed mid-flight. Same corpus-bias caveat. Worth
re-testing alongside a re-converted v3 rec.

## PP-OCRv6 is the default (Sep 2026)

Chosen on latency, not accuracy. A 5.67h production soak measured:

| | v3 | v6 |
|---|---|---|
| OCR inference p50 | 231 ms idle / 330 ms in-block | **195 ms** |
| OCR inference p90 | 550 ms | **257 ms** |
| hard timeouts | 3 in 36h | **0 in 5.67h** |

The tighter tail is what keeps text-dense ad frames (fine print, disclaimer
cards) under the 1.5 s hard timeout — the failure that used to unblock a live
ad mid-break. Accuracy is a wash: 40 identical ad frames through the production
keyword matcher gave v3 39/40 and v6 38/40, with v6 reading more text overall.

Roll back with `MINUS_OCR_MODEL_VERSION=v3`; both model sets ship side by side.
