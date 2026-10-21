# Plan: 95% captions-off name muting with tight mutes

## Goal

Mute "LeBron" / "LeBron James" with **no help from captions**, and mute
**only the name**:

| Target | Today | Goal |
|---|---|---|
| Names fully muted, captions off, 2.4 s delay | 89% (24/27, Oct 13) | **≥95%** |
| Names fully muted, captions off, 1.2 s delay | 41% (11/27) | **≥95%** (stretch goal) |
| Silence per name | ~1.0–1.3 s at 2.4 s (demo clip) | **≤ ~1 s**, with no audible name |
| False mutes on non-LeBron content | 0–2 per 10 min (except "Kings") | no worse |

The delay settings are fixed: 2.4 s (`MINUS_AV_DELAY_S=2.0`) and 1.2 s
(`0.85`). The work is accuracy, not timing.

## Where the misses come from

* **At 2.4 s, the misses are recognition failures.** Detection is fast
  enough (p90 ~1.7 s, under the 2.36 s budget). The names that get through
  are ones SenseVoice never recognises, mostly in fast game commentary. In
  the clip-farm data, SenseVoice hears the name in about 3 of 4 clips whose
  captions contain it, and in about 1 of 3 of the clips only the captions
  caught. Fine-tuning targets exactly this.
* **At 1.2 s, the misses are mostly latency.** Saying the name takes
  0.4–0.5 s, windows hop every 0.5 s and inference adds ~0.4 s. The median
  detection comes 1.1–1.3 s after the name starts, so even perfect
  recognition is often too late. Fine-tuning alone will not reach 95%.
  It needs an earlier detector (phase 4).
* **Tight mutes trade against coverage.** Smaller pads mean less silence
  but more partial names. A model that is more certain of the word
  also times it better, which allows smaller pads.

## Confidence

* 2.4 s captions-off ≥95% with ~1 s mutes, after fine-tuning:
  **moderate (~60–70%)**.
* 1.2 s captions-off ≥95% with fine-tuning alone: **low (~15–25%)**.
  With a first-syllable keyword spotter: plausible, but it is a research
  item.

## Phase 0: keep collecting data (running now)

* The clip farm runs continuously (`tools/clip_farm.py`), and
  `tools/label_clips.py` labels clips with caption text that agrees with
  the ASR. The dataset cap is 24 GB.
* Improve diversity:
  * skip duplicate re-uploads of the same event (same caption text);
  * favour videos with ≥0.5 mentions per minute;
  * keep a news / talk / game-commentary mix.
* Keep the evaluation videos out of the farm (`CLIP_FARM_EXCLUDE`, and the
  held-out list in `label_clips.py`).
* **Done when** game-commentary mentions are at least a third of the name
  clips and growth has flattened. Check weekly.

## Phase 1: a trustworthy evaluation (first, before any training)

27 mentions can't show 95%: one miss is 96%, two is 93%.

1. **Held-out set of 150–200 mentions** from 12–20 videos the farm never
   plays: about half game commentary, plus news, talk shows and
   interviews. Add a negative set: 60+ min with no LeBron, including a
   Kings game and other "King" content.
2. **Offline replay harness**, so each model or setting is scored in
   minutes rather than live hours:
   * source audio → production-style windows (`tests/asr_window_dump.py`);
   * → the real `NameMuteController` and scheduler at a given delay;
   * → muted audio.

   The partial-name simulator is the starting point. It needs caption
   narrowing, the banner rules and the late-detection rules from
   `src/name_mute.py`, and it must use the real classes, not copies.
3. **Score by audibility, not caption timestamps.** Caption word times
   are only accurate to ±0.2 s, which makes tight pads look worse than
   they are.
   * Treat a mention as fully muted if no more than ~30 ms of the name's
     onset is audible.
   * Confirm with a separate ASR (not the one under test) that the name
     can't be recognised in the muted clip, and with spot listening.
   * Get word boundaries by forced alignment on the clean source audio.
4. **Report** for each delay: fully muted, partial (audible ms), missed,
   silence per mention, and false mutes per hour.
5. **Calibrate** offline against 2–3 live runs (`tools/asr_live_bench.sh`)
   so the two agree within ~1 mention.

## Phase 2: fine-tune the ASR

Train on a GPU machine, not the board. Split train/validation by video,
never by clip.

**Data**
* Label quality: drop clips whose label and ASR overlap is borderline.
  Re-check `caption_only` clips by forced alignment.
* Mix in about 20–30% clips without the name (the `random` kind), so the
  model doesn't learn to say "LeBron" everywhere. Track false mutes from
  the first experiment.
* Augment: crowd noise, music beds, level changes, and bandwidth matching
  the HDMI audio path.

**Models**, in this order:
1. **Parakeet-110M** (NeMo).
   * Fine-tune the CTC/TDT head and the top encoder layers.
   * It deploys on one CPU core exactly as now, with no quantization
     step, so it is the fastest path to a measurable win.
2. **SenseVoice-Small** (FunASR fine-tune).
   * Then re-export to ONNX, convert with rknn-toolkit2 for the NPU and
     compare against the unquantized model.
   * Risk: 8-bit quantization can erase the gain. Try mixed precision, or
     keep sensitive layers in fp16.

**Gate:** each candidate must beat the current model on held-out recall and
on fully muted at 2.4 s captions off, with no more false mutes. It must
also keep NPU/CPU time per window within today's budget (≤ ~0.4 s for
SenseVoice).

## Phase 3: tighter mutes

* Use the fine-tuned model's word timings, checked against forced
  alignment on the held-out set, to re-fit:
  * the ASR pads (now 0.35 s before, 0.25 s after);
  * the partial-name window (0.8 s);
  * the cut-off-word extension (0.4 s).
* Sweep the pads in the replay harness and pick the smallest that keeps
  ≥95% fully muted (audibility-scored). Expected result: about 0.8–1.0 s of
  silence per name.
* Fix the remaining caption false mutes: 8 in the first live run, likely
  graphics where the name ends a banner line. Captions-on must not get
  worse, even though captions-off is the target.

## Phase 4: 1.2 s, an earlier detector (research)

* Train a **small keyword spotter for the onset "LeB…"**.
  * It looks at the last ~0.5 s every 50–100 ms and fires about 0.25–0.3 s
    after the name starts. That leaves time to mute from the start at
    1.2 s.
  * Model: a small CNN/CRNN on log-mel features, which can run on the NPU
    or one CPU core.
* Training data comes from this dataset:
  * positives: name onsets located by forced alignment;
  * hard negatives: "Lakers", "level", "let's", "Le-" words and "Bron"-like
    sounds.
* It mutes provisionally, and the ASR confirms or cancels, as partial
  names do today. The false-mute budget is the main constraint.
* Also try:
  * a 0.25 s hop with the fine-tuned model (more windows, more NPU time);
  * a fine-tuned model that emits the name from a cut-off "LeBr" fragment.
* **Gate:** ≥95% at 1.2 s captions off, with false mutes no worse than
  today.

## Rough timeline

| Phase | Effort | Depends on |
|---|---|---|
| 1. Evaluation set + replay harness | 3–5 days | — |
| 2a. Parakeet fine-tune + eval | ~1 week | 1, a GPU machine |
| 2b. SenseVoice fine-tune + NPU conversion | 1–2 weeks | 1, GPU |
| 3. Pad re-fit, caption false mutes | 2–3 days | 2 |
| 4. Keyword spotter for 1.2 s | 2–4 weeks, uncertain | 1, alignments from 2 |

## Risks

* **Labels:** caption-derived labels are noisy (auto-captions mishear
  too). Mitigation: the ASR-overlap filter, forced-alignment checks, and a
  hand-checked evaluation set.
* **Domain overfit:** a farm of LeBron highlight videos may make the model
  hear "LeBron" too eagerly. Mitigation: negatives in training, and the
  false-mute benchmark as a hard gate.
* **NPU quantization** of a fine-tuned SenseVoice. Mitigation: Parakeet
  on CPU as the fallback deployment.
* **Overfitting to a small test set:** report confidence intervals, and
  don't tune on the held-out set.
