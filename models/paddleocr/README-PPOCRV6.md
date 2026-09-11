# PP-OCRv6 (tiny) RKNN models — staged for evaluation

These are **not loaded by Minus**. `src/config.py` globs `ppocrv3_det_*.rknn`, so
these `ppocrv6_*` files sit inert next to the v3 models and production keeps
running PP-OCRv3 until you deliberately switch (see *How to try them* below).

Converted 2026-09-09 from the official PaddlePaddle ONNX exports with
`rknn-toolkit2` 2.3.2 for `rk3588` (device runtime is `rknpu2` 2.3.0).

| File | Size | MD5 | Role |
|---|---|---|---|
| `ppocrv6_det_rk3588_2026-09-09.rknn` | 3.4 MB | `5d6232e19deac5cf7b6eb762e3c2f0c4` | Detection. Input 960x960. |
| `ppocrv6_rec_rk3588_2026-09-09.rknn` | 2.9 MB | `5006c455f2503a04f11a553e3a19de42` | Recognition. Input 48x320. |
| `ppocrv6_keys.txt` | 6904 lines | `c98660b3c39317cab66a66ff91fd777f` | **New dictionary — required.** Not interchangeable with `ppocr_keys_v1.txt`. |

## Why the *tiny* tier

PP-OCRv6 ships tiny (1.5M params) / small (7.7M) / medium (34.5M). Only tiny is
in the same size class as the v3 models in use today:

| | v3 ONNX | **v6 tiny** | v6 small | v6 medium |
|---|---|---|---|---|
| det | 2.43 MB | **1.78 MB** | 9.88 MB | 62.0 MB |
| rec | 10.66 MB | **4.46 MB** | 21.16 MB | 76.6 MB |

As RKNN, det lands at 3.4 MB (v3 det: 3.4 MB — identical) and rec at 2.9 MB
(v3 rec: 6.5 MB — less than half). If you later want accuracy over footprint,
`small` is the next step up and converts the same way (`--tier small`-equivalent:
just point the script at the small ONNX; input shapes are unchanged).

## Drop-in compatibility

Everything the runtime cares about is unchanged:

* Input shapes match `src/ocr.py` exactly (`det_input_h/w = 960`, `rec_input_h/w = 48/320`).
* The host still passes **uint8 RGB NHWC**; normalization stays baked into the
  model via `rknn.config()`.
* Decoding is still `CTCLabelDecode`. Minus builds `['blank'] + dict + [' ']`,
  which for this dictionary is 1 + 6904 + 1 = **6906**, exactly the model's
  output width.

Three things *do* differ and matter:

1. **Dictionary.** 6904 entries vs v3's 6624. You must pass `ppocrv6_keys.txt`;
   using `ppocr_keys_v1.txt` will decode to wrong characters.
2. **DB post-process params.** PP-OCRv6's `inference.yml` specifies
   `thresh=0.2, box_thresh=0.4, unclip_ratio=1.4`. `src/ocr.py` constructs
   `DBPostProcessor()` with its defaults (`0.3 / 0.5 / 1.5`). The v6 values were
   used in the verification below and found more text regions.
3. **Rec normalization.** This build uses PaddleOCR's rec convention
   `(x/255 - 0.5)/0.5` (mean=std=127.5). The v3 build applied ImageNet mean/std
   to the rec model as well, which does not match PaddleOCR's `RecResizeImg` —
   likely a latent bug in the current deployment. No runtime change is needed
   either way, since normalization lives inside the model.

## Verification (before deployment)

Full det -> crop -> rec pipeline run on three real captured TV frames, comparing
ONNX Runtime fp32 against the RKNN simulator, reusing Minus's own
`DBPostProcessor` / `CTCLabelDecode`:

* det probability-map cosine ONNX vs RKNN: **1.0000** on all three frames.
* rec transcripts: **33 of 34 strings byte-identical** (the one difference was
  `1,2M` vs `1.2M`).

So the conversion itself is faithful. Quality vs the deployed v3 models on the
same frames (both imperfect at 960x540, v6 modestly ahead):

| frame | v3 regions | v6 regions | example |
|---|---|---|---|
| ad_detected_...0001 | 10 | **12** | v6 `Rendimemo de comoustbie cesde` vs v3 `Redmiemodecomubleced` |
| ad_...0005 | 9 | **11** | both got `We Got Evicted` |
| ad_...0013 | 9 | **11** | both got `We Got Evicted`, `glarses company` |

v6 tiny consistently finds more text regions and separates words better; v3 won
on at least one string (`$307,400` vs v6's `8307,400`). Neither is clean on
low-resolution overlay text — this is a real A/B worth running against the
keyword matcher, not a slam dunk.

## How to try them

The loader hardcodes the `ppocrv3_*` prefix, so pick one:

```bash
# A) Separate dir + env override (no repo edits, fully reversible)
mkdir -p ~/ocr-v6 && cd ~/ocr-v6
cp ~/Minus/models/paddleocr/ppocrv6_det_*.rknn ppocrv3_det_v6.rknn
cp ~/Minus/models/paddleocr/ppocrv6_rec_*.rknn ppocrv3_rec_v6.rknn
cp ~/Minus/models/paddleocr/ppocrv6_keys.txt ppocr_keys_v1.txt
sudo systemctl stop minus
MINUS_OCR_MODEL_DIR=~/ocr-v6 python3 ~/Minus/minus.py    # or export it for the service
```

(The filenames are a lie in that directory, but it needs zero code changes and
you can delete the whole folder to revert.)

```bash
# B) Point src/config.py + the dict path at the v6 files, then:
curl -s -X POST http://localhost/api/ocr/test | python3 -m json.tool
```

Either way, also try the v6 DB thresholds (`DBPostProcessor(thresh=0.2,
box_thresh=0.4, unclip_ratio=1.4)`) — a chunk of the extra regions above came
from those, not from the model alone.

## Re-converting / other tiers

`convert_ppocrv6_rknn.py` (in the dev repo at
`models/ppocrv6-rknn/`) downloads nothing; point it at ONNX pulled from
`PaddlePaddle/PP-OCRv6_{tiny,small,medium}_{det,rec}_onnx` on HuggingFace
(`inference.onnx` + `inference.yml`; the yml carries the char dict under
`PostProcess.character_dict`). `verify_ppocrv6.py` runs the ONNX-vs-RKNN
comparison above.
