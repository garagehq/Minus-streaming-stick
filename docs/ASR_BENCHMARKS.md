# ASR benchmarks for the name muter (RK3588, Oct 2026)

Every speech-recognition option tried for the `asr-first` name muter
("LeBron James" / "LeBron" / "King"), measured on minus-2 (Radxa ROCK 5B+,
RK3588, 16 GB). The goal is the smallest A/V delay that still mutes each
name, without loading the CPU, because a hot, throttled CPU degrades the
video.

**TL;DR**

* The **delay floor is about 0.8 s end to end**, set by how long it takes to
  say the name plus detection time, not by model speed. At 0.6 s, half the
  names leak.
* **Captions do most of the work at low delay.** Caption OCR still covered
  nearly every on-screen mention at 0.8 s. At 1.2 s and below the ASR alone
  mostly arrives late. With captions off, the ASR sets the floor.
* **SenseVoice on the NPU** (the default) is the coolest ASR: ~2% CPU and
  ~3–5 °C over ASR off, which is NPU heat. On its own it needs about 1.6 s
  or more to mute names on time.
* **Parakeet-110M on one A76 core** detects about 0.1–0.3 s sooner and
  mutes on time more often below 1.6 s. It costs ~7% more total CPU and
  ~3 °C more than SenseVoice. It is the best choice for ASR-only operation
  at 1.2 s.
* Larger or slower models (Parakeet 0.6B, Moonshine, whisper) do not help
  latency, and their heat cost is what hurts the video. The old Moonshine
  default drove this headless box to 80 °C at 86% CPU, versus 48 °C at
  10% for SenseVoice.

## Decision table

Each row is one ASR engine at one effective delay. Caption OCR runs in
every row (NPU core 0). Coverage adds both videos where both were run.

* **Fully muted** is the overall result with captions on.
* **ASR alone** is what you would get with captions off.
* Temperature and CPU are the live means during those runs, on a box with
  no TV attached (ASR off: 43–46 °C, 8% CPU). All runs held 30 fps with no
  throttling.

| Engine | Where it runs | Effective delay (`MINUS_AV_DELAY_S`) | Fully muted (captions on) | ASR alone (captions off) | SoC temp | CPU busy (whole system) |
|---|---|---|---|---|---|---|
| SenseVoice | NPU core 1 | 2.35 s (2.0, today's default) | 15/15 (100%) | 11/15 (73%) | ~48 °C | ~10% |
| SenseVoice | NPU core 1 | 2.0 s (1.65) | 11/12 (92%) | 8/12 (67%) | 48.7 °C | 10% |
| SenseVoice | NPU core 1 | 1.6 s (1.25) | 13/14 (93%) | 11/14 (79%) | 47.9 °C | 10% |
| SenseVoice | NPU core 1 | 1.2 s (0.85) | 31/32 (97%) | 6/32 (19%) | 48.3–48.5 °C | 10% |
| SenseVoice | NPU core 1 | 0.8 s (0.45) | 26/30 (87%) | 0/30 (0%) | 48.0–48.5 °C | 10% |
| Parakeet-110M | 1 A76 core | 1.6 s (1.25) | 8/9 (89%) | 7/9 (78%) | 50.8 °C | 16% |
| Parakeet-110M | 1 A76 core | 1.2 s (0.85) | 29/30 (97%) | 14/30 (47%) | 50.9–51.1 °C | 16% |
| Parakeet-110M | 1 A76 core | 0.8 s (0.45) | 30/32 (94%) | 9/32 (28%) | 50.9–51.1 °C | 16% |
| Parakeet-110M | 1 A76 core | 0.6 s (0.25) | 8/17 (47%) | 0/17 (0%) | 50.8 °C | 17% |
| Parakeet 0.6B | 3 CPU cores | 2.35 s (2.0) | 14/15 (93%) | 12/15 (80%) | 67 °C | 46% |
| Moonshine medium (old) | 3 CPU cores | 5.35 s (5.0) | 108/110 (98%, earlier runs) | most | 80 °C | 86% |

Notes on the rows:
* No name was missed outright in any row; the rest were partial.
* The Parakeet 0.6B and Moonshine temperature and CPU come from the
  7-minute heat runs below.
* Rows with one video have 9–15 mentions, so ±1 mention is noise.

## Definitions

* **Effective delay**: how far the TV output lags the source.
  `MINUS_AV_DELAY_S` plus the audio queue's fixed 0.35 s floor (the
  service reports it as `delay_s` in `GET /api/name-mute`). "0.8 s" below
  means `MINUS_AV_DELAY_S=0.45`.
* **Fully muted**: the mute covers the whole spoken name, from the YouTube
  caption word time to the next word (`tests/name_mute_live_analyze.py`).
  **Partial**: some of the name was audible.
* **ASR alone**: names the ASR's own mute windows covered fully, ignoring
  captions. This is what you get with captions off.
* **Detection lag**: from the start of the name (capture time) to the
  detector reporting it. A mute starts on time only if lag + pad (0.4 s
  ASR, 0.7 s caption) is under the delay. Late mutes start the moment the
  name is detected, so they still catch the rest of it.
* **Recall** (offline): share of caption-track mentions the model found,
  on 3 s windows every 0.5 s over a 5.6 min NBA commentary clip
  (`nbc16k.wav`, 26 mentions).

## Offline: recall, speed and word timing

Same clip and scorer for all. Cost is per window on the live box with
Minus running.

| Engine (runtime) | Where | Cost / window | Name recall | Word start vs caption |
|---|---|---|---|---|
| **SenseVoice-small** (RKNN) | NPU core 1 | **0.35 s** fixed (NPU 0.30 + fbank 0.01) | 88% | +0.12 / +0.16 / +0.30 s |
| SenseVoice, 2 s windows | NPU | 0.37 s | 77% | similar |
| SenseVoice, one inference on cores 0+1 / 0+1+2 | NPU | 0.51 / 0.75 s | same | same |
| **Parakeet TDT-CTC 110M** int8 (sherpa-onnx) | 1 A76 core | **0.16 s** | 81% | −0.08 / +0.06 / +0.19 s |
| Parakeet 110M | 2 A76 cores | 0.12 s | 81% | same |
| Parakeet 110M | 1 A55 core | 0.65 s | 81% | same |
| Parakeet 110M, 2 s / 1.5 s windows | 1 A76 core | 0.11 / 0.09 s | 65% / 54% | same |
| Parakeet TDT 0.6B v2 int8 | 3 cores | 0.66 s | 96% | −0.08 / +0.02 / +0.12 s |
| Parakeet unified 0.6B streaming (240 ms) | 1 core | RTF ~16 | 0% (no output) | — |
| Moonshine medium-streaming (old default) | 3 cores | 1.10 s p50 / 1.50 p95 (2.5 s windows) | 88% | +0.22 to +0.40 s |
| Moonshine small / base / tiny | 3 cores | 0.79 / 0.53 / 0.45 s | 81% / 58% / 42% | — |
| Moonshine medium, streaming API | 3 cores | falls behind real time (name seen 5–10 s late) | — | — |
| whisper.cpp tiny.en v1.9.5 | 3 cores | ~0.9 s per 5 s clip (encode 0.16 s) | — | — |
| whisper.cpp tiny / base, Vulkan on Mali-G610 | GPU | encode 0.62 / 0.85 s (2–4x slower than CPU) | — | — |
| faster-whisper tiny.en | 3 cores | 3.3–5 s (fixed 30 s encoder) | — | — |
| sherpa-onnx keyword spotter (zipformer 3.3M) | CPU | RTF 0.07 | 27% | fires ~1.0 s after word |

Word start columns are p10 / p50 / p90 where available.

Notes:
* Recall counts a single NBA clip. One "false alarm" both Parakeets
  reported was a real "block by LeBron James" that the auto-captions wrote
  as "[Applause]". SenseVoice's 88% includes the garbled-first-name rule
  ("amron james"); it was 85% without it.
* The Parakeet streaming export re-runs the 0.6B encoder over ~5.6 s of
  context every 80 ms ("buffered streaming") and returned no text on
  sherpa-onnx 1.13.8. It cannot run in real time here.
* Little cores are 4x slower; keep CPU ASR on an A76 (cores 4–7).
  `MINUS_ASR_CPU_AFFINITY` defaults to {3,4,5}, which includes A55 core 3.
* GPU details and the bring-up guide: [GPU_SETUP.md](GPU_SETUP.md).

## Live: coverage vs delay

Google TV playing YouTube through Minus, no TV attached. Clock sync with
the TV over ADB; scored against the video's caption track. Only stretches
where the video clock was steady are scored (YouTube mid-roll ads stop the
clock). Captions on, with caption OCR on NPU core 0 in every run.

* Video A: `2nC9z57MuaI`, about 12–17 scored minutes per run.
* Video B: `MIWYB7qtq6c`, about 15–16 scored minutes per run.
* Parakeet-110M settings: 1 thread on core 4, a window every 0.25 s
  (`MINUS_ASR_MIN_CYCLE`).
* SenseVoice settings: a window every 0.5 s.

| Effective delay | Engine | Video | Mentions | Fully muted | Partial | ASR alone | Mute lead (median) | ASR lag p50 / p90 | Muted vs spoken |
|---|---|---|---|---|---|---|---|---|---|
| 2.35 s | SenseVoice | A | 15 | 15 | 0 | 11 | +0.49 s | 0.93 / 1.29 s | 2.6x |
| 2.35 s | SenseVoice, 2 NPU cores | A | 15 | 14 | 1 | 12 | +0.60 s | 1.00 / 1.20 s | 2.5x |
| 2.35 s | Parakeet 0.6B, 3 cores | A | 15 | 14 | 1 | 12 | +0.60 s | 1.28 / 1.66 s | 2.8x |
| 2.35 s | SenseVoice (before matcher fixes) | B | 21 | 18 | 3 | 16 | +0.58 s | 1.16 / 3.20 s | 2.4x |
| 2.0 s | SenseVoice | A | 12 | 11 | 1 | 8 | +0.61 s | 0.96 / 1.75 s | 2.6x |
| 1.6 s | SenseVoice | A | 14 | 13 | 1 | 11 | +0.54 s | 0.86 / 1.84 s | 2.8x |
| 1.6 s | Parakeet 110M | A | 9 | 8 | 1 | 7 | +0.70 s | 0.85 / 2.10 s | 2.5x |
| 1.2 s | SenseVoice | A | 14 | 13 | 1 | 5 | +0.54 s | 1.01 / 2.28 s | 2.6x |
| 1.2 s | SenseVoice | B | 18 | 18 | 0 | 1 | +0.46 s | 1.43 / 2.90 s | 2.6x |
| 1.2 s | Parakeet 110M | A | 12 | 11 | 1 | 8 | +0.56 s | 0.90 / 2.75 s | 2.4x |
| 1.2 s | Parakeet 110M | B | 18 | 18 | 0 | 6 | +0.52 s | 1.10 / 2.81 s | 2.7x |
| 0.8 s | SenseVoice | A | 14 | 11 | 3 | 0 | +0.26 s | 0.96 / 1.33 s | 2.1x |
| 0.8 s | SenseVoice | B | 16 | 15 | 1 | 0 | +0.16 s | 1.32 / 1.80 s | 2.5x |
| 0.8 s | Parakeet 110M | A | 14 | 13 | 1 | 7 | +0.20 s | 0.85 / 1.59 s | 2.3x |
| 0.8 s | Parakeet 110M | B | 18 | 17 | 1 | 2 | +0.23 s | 0.97 / 1.29 s | 2.2x |
| 0.6 s | Parakeet 110M | B | 17 | **8** | **9** | 0 | +0.05 s | 1.01 / 2.10 s | 2.3x |

Column notes:
* **Mute lead (median)** is how long before the name the mute starts.
* **Muted vs spoken** is total muted time divided by total time spent
  saying the names.

Reading it:
* **Combined coverage holds down to 0.8 s and collapses at 0.6 s.** Caption
  OCR is seen ~0.5 s after the name starts (p90 0.6 s). With its 0.7 s
  lead pad it needs ~1.2 s to be fully on time. Below that, a caption mute
  still covers most names because it begins as soon as the caption is
  read, but at 0.6 s the start of most names is audible.
* **The ASR-alone column is the no-captions case.** SenseVoice falls from
  11/14 at 1.6 s to 1–5 at 1.2 s and 0 at 0.8 s. Parakeet-110M keeps 6–8 at
  1.2 s and 2–7 at 0.8 s. It wins because it is 0.2 s faster per window,
  starts a window twice as often, and its word starts are ~0.2 s earlier.
* **ASR lag is dominated by the name itself.** Detection is 0.85–1.0 s after
  the name starts at p50 for every engine. "LeBron James" takes ~0.6–0.8 s
  to say, and the window must contain enough of it. The p90 tail (2–3 s)
  is mostly windows where the name was found in the text but without word
  timings, which mute the whole window.
* Sample sizes are small (9–21 mentions per run), and the two videos
  differ: B's commentary gave both ASRs fewer on-time hits. Treat
  differences of one or two mentions as noise.
* Missed (no mute at all): 0 in every scored run.

## Live: heat and CPU cost

Each engine ran for 7 minutes while the same video played, and
`tests/thermal_sample.py` sampled every 5 s; the table averages the last
5 minutes. The delay was 2.0 s, the stream ran at 2K30, no TV was attached
and every engine config ran real inferences. "ASR off" was measured before
the engines and again right after Moonshine (still cooling).

| Engine | SoC temp | Whole-system CPU busy | A76 clock | ASR time / window |
|---|---|---|---|---|
| ASR off | 42.9 °C (45.6 °C after) | 8% | 2.0–2.2 GHz | — |
| **SenseVoice, NPU** | **48.1 °C** | **9.7%** | 2.2 GHz | 0.32 s |
| **Parakeet-110M, 1 A76 core** | **51.5 °C** | **16.6%** | 2.35 GHz | 0.17 s |
| Parakeet TDT 0.6B, 3 cores | 67.0 °C | 46.5% | 2.35 GHz | 0.62 s |
| Moonshine medium, 3 cores (old default) | **79.9 °C** | **86%** | 2.35 GHz | 1.40 s |

* No throttling in any run, and the stream held 30 fps throughout, but this
  box is **not** doing 4K passthrough to a TV. In production the SoC already
  sits near its 85 °C trip, so read these as the extra heat each engine adds.
  Moonshine's +35 °C is what pushed production into throttling. SenseVoice
  adds ~+3–5 °C and Parakeet-110M ~+6–8 °C over ASR off.
* The live matrix runs agree: SenseVoice 47.9–48.7 °C at 10% CPU busy and
  Parakeet-110M 50.8–51.1 °C at 16–17%, across both videos and every delay.
* "CPU busy" is the whole 8-core system, so Moonshine's 86% means nearly
  every core was saturated.

## Recommendations

| Goal | Setting |
|---|---|
| Lowest delay with captions available | `MINUS_AV_DELAY_S=0.45` (0.8 s effective). Either ASR; SenseVoice adds the least heat. |
| Lowest delay without relying on captions | Parakeet-110M on one A76 core, `MINUS_AV_DELAY_S=0.85` (1.2 s). |
| Under 2 s with good ASR-only coverage | SenseVoice, `MINUS_AV_DELAY_S=1.25` (1.6 s), ~2% CPU. |
| Coolest | SenseVoice on the NPU (any delay). |

Parakeet-110M settings:

```
MINUS_ASR_ENGINE=parakeet
MINUS_PARAKEET_DIR=/home/radxa/asr_models/parakeet/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8
MINUS_ASR_THREADS=1 MINUS_ASR_CPU_AFFINITY=4 MINUS_ASR_MIN_CYCLE=0.25
```

Ideas to get under the ~0.8 s floor, not yet tried:
* **Mute on a partial name.** Act on "le b…"/"lebr" or a CTC prefix as
  soon as it appears, instead of waiting for the full word. This trades
  false mutes for 0.2–0.4 s.
* **Fine-tune the 110M CTC model on NBA commentary.** Its recall is the
  weak point (81%), and its speed leaves headroom for a 0.15 s cycle.
* **Smaller caption pad.** Caption mutes use 0.7 s before the frame where
  the name first appears, which makes them late below 1.2 s. Captions
  appear only ~0.1–0.2 s after the word, so ~0.4 s may be enough.
* **Run Parakeet-110M on the NPU.** It would need an RKNN conversion of the
  CTC model; that removes its CPU cost.

## Reproducing

```bash
# offline
python3 tests/sensevoice_npu_eval.py ~/asr_models/sensevoice-rknn clip16k.wav captions.json3 3.0 0.5 2
taskset -c 4 python3 tests/parakeet_eval.py ctc ~/asr_models/parakeet/<model> clip16k.wav captions.json3 3.0 0.5 1
# live matrix (root): one config per "label|ENV=..." argument
sudo tools/asr_live_bench.sh <tv-ip> <video-id> captions.json3 1000 OUTDIR "sv-1.2|MINUS_AV_DELAY_S=0.85"
```

Models:
* SenseVoice: `~/asr_models/sensevoice-rknn` (HF
  `happyme531/SenseVoiceSmall-RKNN2`).
* Parakeet: `~/asr_models/parakeet/*` (sherpa-onnx `asr-models` release).

Raw results are in `~/lebron_test/bench`, `~/lebron_test/bench2`,
`~/lebron_test/heat2` and `~/lebron_test/live5-8.json` on minus-2.
