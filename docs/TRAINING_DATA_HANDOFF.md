# Training Data Handoff — pulling Minus screenshots and training the next VLM

A brief for the agent that takes over: pull the ad / non-ad training data that
the `minus` box has collected, clean the labels, fine-tune the next version of
the ad classifier, and get it back onto the box. Written 2026-10-26, updated 2026-10-30.

## TL;DR

- **The current model is `minus-v0.2`** (shipped 2026-10-29, `TheGarageDev/Minus-v0.2`). It was **already trained on the box's frames from 2026-10-07 → 10-26** (19,133 frames; labels checked by two models plus a visual review). Don't re-pull or re-label that set: the cleaned copy is on the training machine (see *Where the training happens*).
- **New, untrained data = frames dated after 2026-10-26.** On 2026-10-30 the box held **~4,530 frames (~2.4 GB)**, almost all from 10-27 on. The folder was pruned after the v0.2 pull, so it's starting from small again.
- **The collection rolls.** It's capped at 10 GB (`MINUS_SCREENSHOTS_MAX_GB`) and the **oldest files are deleted** once full (~1.5 GB/day, so roughly a week of history). Pull at least weekly; `rsync` only copies what's new.
- There is a second data source: the **opt-in contribution dataset** on Hugging Face, fed by the ingest server (see *The ingest-server dataset*).
- The labels come from heuristics (OCR keywords, periodic sampling), not from people. Expect noise, and clean it before training (see *Label semantics*).

## The box

| | |
|---|---|
| Host | `minus` (Tailscale `minus.tailf22a5.ts.net`, `100.119.248.124`) |
| User | `radxa` (ask the owner for the sudo password; it is not written here) |
| Other boxes | `minus-2` (was the reference box) is no longer on this tailnet; `informal-minus-1` was last seen ~2026-09-28. Only `minus` is reachable from here |
| Service | `minus.service` runs **as root**, so files under `/root/` need sudo |
| Disk | 59 GB root, ~11 GB free. Don't stage large copies on the box |

## What's there (2026-10-30)

| Path on `minus` | Files | Size | What it is |
|---|---|---|---|
| `/home/radxa/Minus/screenshots/ads/` | 2,287 | 1.1 GB | Frames where **OCR matched an ad keyword** |
| `/home/radxa/Minus/screenshots/non_ads/` | 2,062 | 1.2 GB | Mostly **periodic content samples** (see below), plus frames saved when the user paused a block |
| `/home/radxa/Minus/screenshots/vlm_spastic/` | 176 | 87 MB | VLM said AD N times in a row (`_2x`…`_5x` in the filename), then flipped to NO-AD. Unlabeled; good hard cases (for v0.2, 342 of 440 turned out to be show content, not ads) |
| `/home/radxa/Minus/screenshots/static/` | 5 | 2 MB | Ad keywords on a **static** screen (paused / promo tile). Ambiguous |
| `/root/.minus_contrib_queue/` | ≤200 png + json | ~90 MB | Opt-in contribution queue, waiting to upload (10-min cool-down). Frames from blocked ads plus a JSON sidecar (`verdict`, `engine` e.g. `minus-tv/ocr`, `_block` id). It overlaps with `ads/` and ends up in the ingest dataset anyway, so it's optional to pull |
| `/home/radxa/.minus_reviewed_screenshots.json` | — | — | Filenames a human reviewed in the web UI swipe tool. Almost all are April-era files that no longer exist; not useful |

Only frames dated after `20261026` are new to the model. A few `non_ads/` frames from 10-16 → 10-26 are still on the box; they were already in the v0.2 set.

All frames are **960×540 PNG**, OCR's downscaled snapshot (`/snapshot/raw`, captured *before* the blocking overlay is drawn, so the overlay never appears in the frames).

Filenames are `<category>_<YYYYMMDD>_<HHMMSS>_<ms>_<counter>.png`, in **UTC**. The counter resets on every service restart, so it is **not** unique: always use the full filename as the key.

Recount at pull time, because the numbers change continuously:

```bash
ssh radxa@minus 'cd ~/Minus/screenshots && for d in ads non_ads vlm_spastic static; do echo "$d $(ls $d | wc -l) $(du -sh $d | cut -f1)"; done'
```

## Pull it

From the training machine (which has the GPU and the disk space):

```bash
DEST=~/training/minus-data-2026-11   # new data; the 10-07..10-26 set is already in v0.2
mkdir -p $DEST
# Screenshots (radxa-owned, no sudo needed). rsync is resumable; re-run to pick up new frames.
rsync -avh --partial --info=progress2 \
  radxa@minus:/home/radxa/Minus/screenshots/{ads,non_ads,vlm_spastic,static} $DEST/

# Contribution queue (root-owned): stage through a readable copy
ssh radxa@minus 'sudo tar -C /root -czf /tmp/contrib.tgz .minus_contrib_queue && sudo chown radxa /tmp/contrib.tgz'
scp radxa@minus:/tmp/contrib.tgz $DEST/ && ssh radxa@minus 'rm /tmp/contrib.tgz'
```

**Don't delete anything on the box.** The disk budget does its own eviction. Copying is read-only and safe to run while the service is up.

### Also pull the per-frame metadata from the journal (important for cleaning)

The screenshots have no sidecar files, but the systemd journal (kept back to 2026-08-26, so it covers every frame on the box) logs, for every saved `ads/` frame, the OCR keywords that triggered it, and right next to that, the VLM verdict and `p_yes` for that moment:

```
17:35:19 VLM #167158: 0.4s [NO-AD] conf=100% "No (p=0.0009)"
17:35:19 OCR detected ad keywords: ['sponsored', 'skip in', 'skip']
17:35:19   Screenshot saved: ad_20261026_173519_723_1421.png
```

Export it once and join on filename:

```bash
ssh radxa@minus 'sudo journalctl -u minus --since 2026-10-26 -o short-iso | grep -E "Screenshot saved|Non-ad screenshot saved|spastic screenshot|static screenshot|Blocking paused|\[Feedback\]|OCR detected ad keywords|VLM #[0-9]+:|OCR #[0-9]+:" | gzip' > $DEST/journal_labels.log.gz
```

For each saved file, take the closest preceding `OCR detected ad keywords` line and the closest `VLM #` line (within ~2 s). The `OCR #` line holds the full OCR text of the frame, which is useful for keyword-based filtering. Journal timestamps are box-local (UTC on this box), the same timezone as the filenames.

## Label semantics (read before training)

| Folder | How a frame gets there | Noise to expect |
|---|---|---|
| `ads/` | OCR matched any ad keyword (`minus.py` → `save_ad_screenshot`). This includes **weak** keywords (`learn more`, `shop now`, `buy now`) and lone `ad`/`ads` tokens, which can fire on non-ad content. | **OCR false positives**: home-screen "Sponsored" tiles, show content containing "Skip"/"Ad", YouTube promo shelves. Frames where VLM said NO-AD with high confidence are the first ones to review. Some are real ads VLM missed (keep those, they're the most valuable), and some are OCR FPs (relabel them). Strong keywords (`skip in`, `skip ad`, `ad N of M`, `ad countdown`, `ad with timestamp`, `visit advertiser`, `video will play after ad`) are near-certain ads. |
| `non_ads/` | (a) **Every 90 s** (`MINUS_NONAD_SAMPLE_INTERVAL`), a frame with **no OCR keyword**, no block showing and no static suppression. (b) The user paused a block (VLM-only block: the frame that triggered VLM; otherwise the current frame). | (a) can contain **ads that had no readable keyword** and that VLM also missed, i.e. bumpers and ads without a Skip UI. Frames with high VLM `p_yes` should be reviewed. (b) is a human-corrected false positive and **high value** (these are mixed in with (a); the journal line `[Feedback]` / `Blocking paused` identifies them). Many frames are near-duplicate music-video / YouTube-UI content from autonomous mode. Dedup by dHash before training. |
| `vlm_spastic/` | VLM flipped AD→NO-AD after 2–5 consecutive AD votes. | **Unlabeled.** Hard examples. Label them by hand (or with a stronger model, then spot-check) before using. |
| `static/` | Ad keywords on a static screen (paused-on-ad, promo tile). | Ambiguous. Product policy treats a paused real ad as an ad and a home-screen promo tile as not an ad. Label each by hand or leave them out. |

Content bias: most frames come from **autonomous mode on YouTube** (Roku / Fire TV, often music videos). Netflix, Hulu and other streaming UIs are under-represented. Check the class and source balance before training.

Every frame passed `_should_save`: rate-limited to one per 5 s per category, black/solid frames rejected, and near-duplicates (dHash < 10 bits against the last 200) dropped. Consecutive frames of the same ad are still correlated, so **split train/val by time block (e.g. by hour), not at random**, or the validation score will be inflated.

## Holdout: do not train on it

The 800-image benchmark used for every model decision so far lives at
`minus:/home/radxa/axera_models/nontrained_test_data/` (`ads/` 400, `non_ads/` 400, `labels.json` with `original_path` back to the April–May source files). Never add holdout frames to training. Report new models on this same 800, **measured on the box's NPU**. The bar is minus-v0.2: **99.1% accuracy, 99.8% ad-recall, 98.5% non-ad-recall** (v0.1 measured the same way: 95.6%). The older "97.0%" figure in `axera_models/BENCHMARKS.md` is v0.1 measured another way; don't compare against it.

The v0.2 release also used two more test sets, both on the box. Report on both, with `~/axera_models/minus-v0.2/eval_v02.py` (`ad <dir> <out.json>` / `screen <manifest.csv> <img_root> <out.json>`); v0.2's own results are `eval_800.json`, `eval_hard.json` and `eval_screen.json` next to it:
- **`~/axera_models/hard_test_328/`** (`ads/`, `non_ads/`): 328 frames v0.1 got wrong or flip-flopped on (v0.1 83.2% → v0.2 95.1%). They are from 10-07 → 10-26, so they're also out of any future training set.
- **`~/axera_models/screen_test_193/`**: 193 hand-labeled screen-state frames for `query_image` (PLAYING / PAUSED / DIALOG / MENU / SCREENSAVER; v0.2 95.9%, and it never read a playing frame as an action-taking class). v0.2 is the first model with a working screen-state head, so the next model must not lose it. Labels: `screen_test_193/manifest.csv`.

It would also be worth carving a **new holdout from the post-10-26 data** (e.g. ~400 + 400, hand-verified, split off by time block) so the model is judged on today's content mix too.

## The ingest-server dataset (second data source)

Boxes with Settings → *Help improve Minus* turned on, and the Chrome extension, upload frames to `https://minus-ingest-garage.fly.dev/ingest`, which writes them to a **private Hugging Face dataset owned by the Minus project**. `minus` has it turned on and uploads ~2–3 frames per ad block (`[Contribute] Uploaded N anonymous ad snapshot(s)` in the journal).
- What's in a sample (`src/contribute.py`, `docs/PRIVACY.md`): a ≤960 px PNG of a frame Minus **blocked as an ad** plus `p_ad` (VLM probability, null if only OCR fired), `verdict` (always `ad`), `host` (`tv`, or the site for the extension), `engine` (`minus-tv/vlm`, `minus-tv/ocr`, or the extension's), and a random `key`. **Ads only**: there are no non-ad frames, and the labels are the detector's own verdicts, so they carry the same false positives. Frames from blocks the user paused are never uploaded.
- The dataset's name and the Hugging Face write token aren't on the box. The reference ingest server is `server/ingest-server.mjs` in `github.com/garagehq/Minus-chrome-extension`. Ask the owner for read access to the dataset.

## Where the training happens

- Fine-tuning runs on a separate GPU machine, `ubuntu@marisol` (`axera_models/docs/LFM2_VL_AX650_OPTIMIZATION.md`), not on the box. April–May data lived at `/home/ubuntu/training/minus-data` there. The cleaned 10-07 → 10-26 set and the v0.2 training scripts and settings should be there as well, but none of that is on the box: **ask the owner, or the Marisol agent that trained v0.2, for the path and the recipe**.
- The conversion from the trained weights to Axera axmodels (Pulsar2) is in the `conversion/` folder of `TheGarageDev/Minus-v0.2` on Hugging Face.
- Two lessons from v0.2 that must carry over (Minus `CLAUDE.md` → *minus-v0.2*):
  - **Train on the box's input.** `_encode_image` squashes the frame to 512×512, ignoring the aspect ratio. Earlier models trained on aspect-correct resizes lost ~2.5 points to that mismatch.
  - **Keep the residual stream in FP32 in the Axera build** (both residual adds plus the attention score/mask add, on top of the RMSNorm chains). Without it the screen-state check lost 17 points on the NPU.

## Training and deploying the model — constraints that are easy to break

Background reading, all on `minus`:
- `~/axera_models/CLAUDE.md`, `~/axera_models/docs/LFM25_NPU_DEPLOYMENT.md`, `docs/LFM2_VL_AX650_OPTIMIZATION.md`: how the AX650 build works (Pulsar2, 16 fused layers, NPU3, FP32 RMSNorm chain) and what was tried and failed.
- `~/axera_models/minus-v0.2/`: current production model. `~/axera_models/minus-v0.1/` (previous) has `eval_fused.py` / `infer_vlm_fused.py` for batch/single eval on the NPU; point them at the new model's directory.
- Minus repo `CLAUDE.md` → *VLM Model* and *FastVLM iter4 → LFM2.5-VL Migration*.

Hard requirements for a drop-in replacement (`src/vlm.py` depends on all of them):
1. **Same base architecture and layout**: LFM2.5-VL-450M → `vision_encoder_512.axmodel`, `fused_models/l{N}_{conv,attn}_fused.axmodel` (10 conv + 6 attn), `decode_models/post_d.axmodel`, `embed.npy`, `tokenizer.json`.
2. **Prompt must stay byte-exact**: system `"You are a helpful multimodal assistant by Liquid AI."`, user `"Is this an advertisement? Answer Yes or No."`, LFM chat template. The answer is read from the Yes/No first-token logits. Train on exactly this prompt.
3. **Preprocessing**: direct bilinear resize to 512×512, `(x/255-0.5)/0.5`, patchify to `(1,1024,768)`. Not expand2square / CLIP.
4. **Prefill window is 320 tokens** (compiled into the axmodels). The ad prompt is 293 tokens. The autonomous-mode screen prompt (`PLAYING/PAUSED/DIALOG/MENU/SCREENSAVER`, 312 tokens) runs on the **same model**, so check it still classifies sensibly after the fine-tune, not only the Yes/No head.
5. Target latency stays ~0.37–0.39 s/frame on the AX650. If a rebuild loses NPU3, latency rises ~40%.

Shipping: models are published to Hugging Face (current: `TheGarageDev/Minus-v0.2`, subfolder `axera`; next would be `Minus-v0.3`, see `docs/RELEASING.md`). The box reads `Minus/models/vlm_model.json` (`name`, `hf_repo`, `subfolder`, `revision`, `local_dir`), and `tools/minus_update.py --install-model` / Settings → *Minus Updates* installs it. A new model means a new repo/revision plus an updated `vlm_model.json`. For a quick local test, use `MINUS_VLM_MODEL_DIR=/path/to/new/model` instead.

Before swapping it in on the box, after the NPU holdout eval:
- `python3 tests/test_vlm_decision_sim_lfm2.py` (re-bootstrap from the new model's holdout scores; the sliding-window params in `minus.py` were tuned to the previous model's error rates and may need a `--sweep`; `MINUS_LFM2_HOLDOUT` points it at a results file).
- `tests/run_box_e2e.sh` on the box.

## Rules on the box

- **Don't restart `minus.service`, swap the model or change settings without the owner's OK.** Someone is watching TV through it.
- **Never run the Wi-Fi/AP tests** (`tests/test_wifi_portal.py` etc.). They take the box off the network.
- No AI attribution lines in commits or PRs.
