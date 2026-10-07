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
* **Most viewers won't have captions on, so captions-off is the number
  that matters.** With partial-name muting (below), SenseVoice at 1.2 s
  fully muted **18/33 (55%)** with captions off, up from 6/32, which ties
  Parakeet-110M while staying at SenseVoice's 49 °C and 10% CPU.
  At 1.6 s, clear news speech is solved (Parakeet 15/15), but fast game
  commentary is not (Parakeet 9/17, SenseVoice 8/17, 5 names never
  recognised). Those misses are recognition failures, which fine-tuning
  would address and more delay would not; see
  [Captions off at 1.6 s and partial names](#captions-off-at-16-s-and-partial-names).
* **Captions-off runs confirm it** (Oct 7). At 1.2 s with no captions,
  SenseVoice fully muted 6/32 and Parakeet-110M 17/31, close to the
  "ASR alone" column below. At that delay the name was still recognisable
  in 10 of 25 SenseVoice clips but only 2 of 23 Parakeet clips.
* **Partial mutes don't give the name away when captions are on.** At
  0.8 s, mutes that clipped the name left 2–52 ms audible (one 345 ms case
  is likely caption timing). SenseVoice could not hear the name in any
  muted clip.
* **Bare "King" is the main false-mute source.** Ordinary news and a
  non-Kings game: 0 false mutes with SenseVoice, 1–2 per 10 min with
  Parakeet (clipped "making" → "king"). A Kings game: 7–9 per 10 min,
  because the "s" is lost. Content about another King (MLK, Stephen King)
  is muted 4–6 times a minute. See
  [False mutes](#false-mutes-on-non-lebron-content).
* **Streaming models don't beat SenseVoice.** The sherpa-onnx RK3588
  streaming Zipformer can't recognise "LeBron" (4/38) and still lags more
  (0.72 s after the name ends, versus ~0.41 s). Kroko recognises it
  (74%) but is slower (~1.3 s).
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

## Captions off (real runs)

The "ASR alone" column above is reconstructed from runs with captions on.
These runs turn caption OCR off (`MINUS_NAME_MUTE_CAPTIONS=0`) at 1.2 s
effective (`MINUS_AV_DELAY_S=0.85`), on the same two videos.

| Engine | Video | Fully muted | Partial | Missed | Name still recognisable after muting* | Detection lag p50 / p90 | Temp / CPU |
|---|---|---|---|---|---|---|---|
| SenseVoice | B (game) | 1/16 | 4 | 11 | 8 of 12 | 1.21 / 2.69 s | 48.6 °C / 10% |
| SenseVoice | A (news) | 5/16 | 8 | 3 | 2 of 13 | 1.11 / 1.63 s | 48.8 °C / 10% |
| Parakeet-110M | B (game) | 6/17 | 5 | 6 | 2 of 12 | 1.22 / 2.71 s | 50.8 °C / 16% |
| Parakeet-110M | A (news) | 11/14 | 3 | 0 | 0 of 11 | 0.78 / 2.99 s | 52.5 °C / 17% |

\* Counted with SenseVoice over clips where it could hear the name in the
original audio (`tests/name_mute_render.py`).

* The totals (SenseVoice 6/32, Parakeet 17/31) agree with the
  reconstruction (6/32 and 14/30), so that column can be trusted.
* Fast game commentary (video B) is the hard case. Without captions,
  SenseVoice lets the name through most of the time at 1.2 s.
* For ASR-only operation below 1.6 s, Parakeet-110M is clearly better.
  SenseVoice needs about 1.6 s or more.

## Captions off at 1.6 s and partial names

### 1.6 s, captions off (Oct 7)

| Engine | Video | Fully muted | Partial | Missed | Detection lag p50 / p90 | Temp / CPU |
|---|---|---|---|---|---|---|
| Parakeet-110M | B (game) | 9/17 (53%) | 3 | 5 | 1.11 / 2.82 s | (71 °C / 36%, inflated by an offline job running alongside) |
| SenseVoice | B (game) | 8/17 (47%) | 4 | 5 | 1.35 / 2.68 s | (63 °C / 21%, same) |
| Parakeet-110M | A (news) | **15/15 (100%)** | 0 | 0 | 0.77 / 1.66 s | 52.2 °C / 16% |

On clear speech, 1.6 s is enough. On fast game commentary, both engines
miss about a third of the names outright, so the model never recognised
them. That is a recognition problem, not a latency one.

### Partial names

When a 3 s window ends partway through "LeBron", the engine writes the
start of it: SenseVoice gives "le", "leb", "lebra" or "leron", and
Parakeet gives "L" or "Lebr". The full word only arrives in the next
window. Muting on that fragment gains a whole cycle (0.5 s for SenseVoice).

Rule (`NameMuteController._check_partial`):
* The last word of a window matches `le|leb|lebr…|lebo…|ler|lero|leron|lbr…`
  and starts in the window's last 0.5 s. It mutes provisionally for
  0.8 s from the fragment (`asr_partial`).
* If a later window hears that stretch whole with no name, the mute is
  dropped if it hasn't started, or cut short if it has.

**Offline simulation** (`tests/asr_window_dump.py` dumps every window
over 20 min of each video, and `tests/partial_name_sim.py` replays them
through each rule):

| Engine | Delay | Fully muted, current rule | With partial names | False triggers / h |
|---|---|---|---|---|
| SenseVoice | 1.0 s | 7/44 (16%) | **20/44 (45%)** | 9 → 12 |
| SenseVoice | 1.2 s | 19/44 (43%) | **30/44 (68%)** | 7.5 → 12 |
| SenseVoice | 1.6 s | 33/44 (75%) | 34/44 (77%) | 3 → 7.5 |
| SenseVoice | 2.35 s | 38/44 (86%) | 38/44 | 3 → 3 |
| Parakeet-110M | 1.2 s | 32/44 (73%) | 32/44 (no gain) | 9 → 23 |

* SenseVoice fragments over the 40 minutes:
  * near a real name: "leb" ×9, "le" ×4, "leron", "lebra";
  * another word: "le" ×2, "leb" ×1.
* Parakeet only ever ended a window on a bare "le", and all 9 of those
  were false. Its 0.25 s cycle already sees the whole word a quarter
  second later, so the rule is on by default for SenseVoice only
  (`MINUS_NAME_MUTE_PARTIAL=0/1` overrides).
* The simulation is optimistic in absolute terms: it has no ads and no
  live jitter, so offline base SenseVoice at 1.2 s is 43% versus 19%
  live. The relative gain is what carried over.

**Live, captions off, 1.2 s (SenseVoice + partial names):**

| Video | Before | With partial names | False mutes |
|---|---|---|---|
| B (game) | 1/16 full, 11 missed | **5/17 full, 4 missed** | 1 partial + 2 ASR in 15.6 min |
| A (news) | 5/16 full, 3 missed | **13/16 full, 1 missed** | 1 ASR in 15.8 min |
| Total | 6/32 (19%) | **18/33 (55%)** | |

That ties Parakeet-110M at 1.2 s (17/31) at SenseVoice's heat: 49.2–49.6 °C,
9–10% CPU.

## Training clips for fine-tuning

`src/asr_clips.py` (on by default, `MINUS_ASR_CLIPS=0` to disable) saves
10 s of 16 kHz mono audio around moments that matter for fine-tuning, to
`screenshots/asr_clips/`:
* every name detection, kind `asr`, `both`, or **`caption_only`**. A
  `caption_only` clip is a mention the captions saw and the ASR missed,
  which makes it the most valuable training example;
* a random clip of non-silent audio every 5 min (`MINUS_ASR_CLIPS_RANDOM_S`).

Each WAV has a JSON sidecar with the mention time and every detection.
It also holds every ASR window transcript with word times and every
distinct OCR screen during the clip, all relative to the clip start.
That's enough to label clips later from captions where present, or with
a larger teacher model (Parakeet 0.6B, 96% offline recall).

* Clips are ~330 KB each. The budget is 2 GB (`MINUS_ASR_CLIPS_BUDGET_MB`),
  about 6,000 clips, oldest evicted first.
* The audio tap's ring buffer grew from 8 s to 30 s to hold the context.
* `GET /api/name-mute` reports the clip count under `training_clips`.

## Does a partial mute give the name away?

`tests/name_mute_render.py` applies a run's mute windows to the source
audio and cuts one clip per mention, from 1.5 s before the name to 1.5 s
after. Audible time is measured against caption word timing, which is
only accurate to about ±0.1–0.2 s. SenseVoice then listens to each muted
clip. Clips and a `listen_all.wav` per run are in `~/lebron_test/render/`.

| Run (captions on) | Mentions | Full | Partial | Audible in partials | Name heard after muting |
|---|---|---|---|---|---|
| Parakeet-110M 0.8 s, video B | 14 | 13 | 1 | 52 ms | 0 |
| SenseVoice 0.8 s, video B | 14 | 13 | 1 | 2 ms | 0 |
| Parakeet-110M 0.8 s, video A | 14 | 13 | 1 | 34 ms | 0 |
| SenseVoice 0.8 s, video A | 14 | 11 | 3 | 16, 31, 345 ms | 0 |
| SenseVoice 1.2 s, video B | — | all | 0 | — | 0 |
| Parakeet-110M 0.6 s, video B | 17 | 8 | 9 | 36–357 ms (median ~144) | 0 |

* With captions on, a partial mute usually clips only the first few tens
  of milliseconds: the start of the "L". The 345 ms case (video A, 299.5 s)
  was partial in every run, including at 2.35 s, so the caption timing is
  probably off there rather than the mute.
* Even at 0.6 s, SenseVoice did not recognise the name in any muted clip.
  A person may still catch a "…Bron" after 150–350 ms, so listen to
  `render/B-pk-0.6/listen_all.wav` before going that low.
* Captions off is different: partials leave 40–640 ms audible, and the name
  stays recognisable in some clips (table above).

## False mutes on non-LeBron content

`tools/false_mute_bench.sh` plays videos and logs every mute with the
text that triggered it. Captions on, 1.2 s effective, 8–10 minutes per
video.

| Video | SenseVoice | Parakeet-110M | What triggered it |
|---|---|---|---|
| PBS NewsHour, Oct 5 2026 (218gMy1KfyQ), 10 min | 0 | 2 | "king very hard", "king about": the "-king" of "making"/"talking" clipped at the start of a window |
| Warriors–Celtics Finals G6 (AOYACk7m7Fk), 10 min | 0 | 1 | "king down shots": same clipped "-king" |
| Warriors at Kings (Rmjz1iGXPeU, no YouTube captions), 10 min | 7 (6 ASR, 1 OCR) | 9 (6 ASR, 3 OCR) | "Sacramento King", "the King": the plural "s" is lost; OCR read "King" in on-screen graphics |
| PBS: MLK biography (m0nr13xnRzU), 8 min | 51 (32 ASR, 19 OCR) | 54 (35 ASR, 19 OCR) | Every "Dr. King" / "Martin Luther King" |
| PBS: Stephen King interview (CQF37Z4CEWg), 8 min | 29 (10 ASR, 19 OCR) | 32 (11 ASR, 21 OCR) | Every "Stephen King", plus OCR on the "STEPHEN KING:" speaker labels |

Counts are mute events, about 1 s each; one spoken name often triggers
both an ASR and a caption mute.

* **Bare "King" works as specified, and the cost is real.** Content
  about any other King gets muted 4–6 times a minute: MLK, Stephen King,
  King Charles, *The Lion King*, Burger King. With captions on, OCR also
  mutes whenever a speaker label says "KING:", even if nobody says the
  word.
* **Kings games: ~40–55 mutes an hour.** The ASR often drops the final
  "s" of "Kings". The pattern excludes plural "Kings", but it can't see an
  "s" the ASR never wrote.
* **Parakeet invents "king" from clipped words.** When a window starts
  mid-word, the tail of "making"/"talking"/"taking" comes out as "king".
  That gave 3 false mutes in 20 minutes of ordinary speech. SenseVoice
  had none.
* **Ordinary speech is otherwise clean.** Bare "James" is not targeted,
  so the dozens of "James" in the test videos never muted.
* Possible fixes, none built yet:
  * Ignore "king" when it's the first word and starts at the window edge;
    the next window re-hears it whole. This fixes the clipped-word case.
  * Ignore "king" right after Sacramento/Lion/Stephen/Burger/Dr./Luther/
    Charles, and ignore a "KING:" speaker label in captions.
  * Make bare "King" opt-in: only "LeBron" / "King James" by default.
* Test-setup trap: the TV has YouTube **Restricted Mode** on, which
  blocks some videos (NBC Nightly News, a *Lion King* review) behind a
  "Something went wrong" screen. Those runs showed 0 mutes because there
  was no audio. Check a frame (`curl localhost:9090/snapshot`) or
  `asr.last_transcript` before trusting a 0.

## Parakeet on the NPU?

As of Oct 2026, nobody has published an RKNN conversion of Parakeet
(110M or 0.6B). Hugging Face, GitHub and the sherpa-onnx release assets
were all checked. The pip sherpa-onnx wheel is also built without RKNN.
The RK3588 options that do exist:

* sherpa-onnx's own RKNN builds: `sherpa-onnx-rk3588-*-sense-voice-*`
  (the same SenseVoice model, with 5 s/10 s/… fixed inputs) and
  `sherpa-onnx-rk3588-streaming-zipformer-en-2023-06-26`, an English
  streaming transducer. The Zipformer is the one untested option that
  could beat the window-based floor, because it emits words as they
  stream in. **Tested since: it doesn't**; see
  [Streaming models](#streaming-models-sherpa-onnx-oct-7).
* Convert Parakeet-110M ourselves. `rknn-toolkit2` 2.3.2 installs with
  pip on aarch64 and matches the board's `librknnrt` 2.3.2. The CTC model
  is a single ONNX file. Its FastConformer encoder needs a fixed input
  length (e.g. 3 s), and attention and layer-norm ops often fall back to
  the CPU or lose accuracy in int8 on RKNN, so expect fp16 and some
  debugging. The gain would be CPU heat (Parakeet-110M costs ~7% more CPU
  and ~3 °C over SenseVoice), not latency, which is already small.

## Streaming models (sherpa-onnx), Oct 7

Streaming models emit words as audio arrives instead of re-reading a 3 s
window. The question was whether that beats SenseVoice's sliding window,
which mutes about 0.41 s after the word ends (live median).

Method: feed the audio in 20 ms steps and decode whenever a chunk is
ready. For each word, record when it first appeared plus that chunk's
decode time, then compare with the caption word timing. Recall is
measured on the first 15 min of video B plus the first 15 min of video A
(38 name mentions).

| Model | Runs on | Name recall | "James" emitted after the name ends (p50 / p90) | Compute |
|---|---|---|---|---|
| `sherpa-onnx-rk3588-streaming-zipformer-en-2023-06-26` | NPU (sherpa-onnx 1.13.8 RKNN build) | 4/38 (11%) | 0.72 / 1.06 s | 142 ms of NPU per 0.64 s chunk (22% of a core) |
| `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | 1 A76 core | 28/38 (74%) | 1.33 / 2.45 s | RTF 0.075 (7.5% of one core) |
| NeMo streaming FastConformer CTC 80 ms / 480 ms int8 | 1 A76 core | ~0 | — | RTF 0.39 / 0.15 |
| SenseVoice sliding window (reference) | NPU | — | ~0.41 s (live median, time to mute) | 0.30 s of NPU per 0.5 s cycle |

* **The NPU Zipformer is fast to emit, but it can't recognise the name,
  and it is still slower than SenseVoice.**
  * It was trained on LibriSpeech, so "LeBron" comes out as "the bron",
    "abroad", "lebran", "philipron", "apron" and so on.
  * The keyword spotter on the same model, with `▁LE B RO N` targeted,
    never triggered, even at threshold 0.01 and boost 5. As a check, "JAMES"
    did trigger.
  * Hotword biasing didn't help either.
  * A fuzzy `l?b?r[aeiou]n` matcher would catch about half of the names,
    but real words like "brown" and "abroad" would trigger it too.
* **Kroko recognises the name** ("an easy lay up there for Lebron James")
  and is very cheap, but its large chunks (~1.28 s) make it slower than
  SenseVoice: p50 1.5–1.9 s after the name starts.
* **NeMo streaming CTC** works on its bundled LibriSpeech test file but
  produces almost nothing on TV audio, even level-matched. Not pursued.
* Streaming doesn't help here. Chunk length plus look-ahead costs about as
  much as SenseVoice's 0.5 s cycle and 0.3 s inference. The ~0.8 s floor
  comes from how long the name takes to say, not from model architecture.
  Converting Kroko to RKNN would cut its CPU cost but not its chunk delay.

Harnesses are in `tests/streaming_asr/`:
* `zf_stream.cc` (C++, links the sherpa-onnx RKNN build);
* `stream_cpu.py` (pip sherpa-onnx);
* `zf_analyze.py` (scoring).

Build and run:

```bash
# sherpa-onnx-v1.13.8-rknn-linux-aarch64-shared + headers from the v1.13.8 tag
g++ -O2 -std=c++17 -Iinc tests/streaming_asr/zf_stream.cc -L$SHERPA/lib \
    -lsherpa-onnx-cxx-api -lsherpa-onnx-c-api -Wl,-rpath,$SHERPA/lib -o zf_stream
./zf_stream asr ~/asr_models/zipformer-rknn/sherpa-onnx-rk3588-streaming-zipformer-en-2023-06-26 clip16k.wav > out.tsv
python3 tests/streaming_asr/zf_analyze.py out.tsv captions.json3
```

## Recommendations

| Goal | Setting |
|---|---|
| Lowest delay with captions available | `MINUS_AV_DELAY_S=0.45` (0.8 s effective). Either ASR; SenseVoice adds the least heat. |
| Lowest delay without relying on captions | SenseVoice with partial names (default), `MINUS_AV_DELAY_S=0.85` (1.2 s): 55% fully muted, same as Parakeet-110M but cooler. |
| Under 2 s, captions off, best coverage | `MINUS_AV_DELAY_S=1.25` (1.6 s). News 100%, game commentary ~50%. Game commentary is limited by recognition, not delay, so the next step there is fine-tuning. |
| Coolest | SenseVoice on the NPU (any delay). |

Parakeet-110M settings:

```
MINUS_ASR_ENGINE=parakeet
MINUS_PARAKEET_DIR=/home/radxa/asr_models/parakeet/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8
MINUS_ASR_THREADS=1 MINUS_ASR_CPU_AFFINITY=4 MINUS_ASR_MIN_CYCLE=0.25
```

Ideas to get under the ~0.8 s floor, not yet tried:
* ~~Mute on a partial name~~: done, see
  [partial names](#partial-names).
* **Fine-tune on NBA commentary.** This is now the biggest lever with
  captions off. Clips are being collected
  ([Training clips](#training-clips-for-fine-tuning)).
* **Smaller caption pad.** Caption mutes use 0.7 s before the frame where
  the name first appears, which makes them late below 1.2 s. Captions
  appear only ~0.1–0.2 s after the word, so ~0.4 s may be enough.
* **Run Parakeet-110M on the NPU.** No conversion exists yet; see
  [Parakeet on the NPU?](#parakeet-on-the-npu).
* **Fix "Kings" false mutes** (see
  [False mutes](#false-mutes-on-non-lebron-content)).

## Reproducing

```bash
# offline
python3 tests/sensevoice_npu_eval.py ~/asr_models/sensevoice-rknn clip16k.wav captions.json3 3.0 0.5 2
taskset -c 4 python3 tests/parakeet_eval.py ctc ~/asr_models/parakeet/<model> clip16k.wav captions.json3 3.0 0.5 1
# live matrix (root): one config per "label|ENV=..." argument
sudo tools/asr_live_bench.sh <tv-ip> <video-id> captions.json3 1000 OUTDIR "sv-1.2|MINUS_AV_DELAY_S=0.85"
# per-window ASR dump + offline rule simulation (partial names)
python3 tests/asr_window_dump.py sensevoice source16k.wav sv.jsonl 0 1200
python3 tests/partial_name_sim.py sv.jsonl captions.json3 [more.jsonl more.json3 ...]
# what the viewer heard: per-mention clips + audibility check
python3 tests/name_mute_render.py OUTDIR/sv-1.2.json captions.json3 source16k.wav RENDERDIR 2
# false mutes on videos that should never mute
sudo tools/false_mute_bench.sh <tv-ip> OUTDIR 600 "sv-1.2|MINUS_AV_DELAY_S=0.85" -- <video-id> ...
```

Models:
* SenseVoice: `~/asr_models/sensevoice-rknn` (HF
  `happyme531/SenseVoiceSmall-RKNN2`).
* Parakeet: `~/asr_models/parakeet/*` (sherpa-onnx `asr-models` release).

Raw results are in `~/lebron_test/bench`, `~/lebron_test/bench2`,
`~/lebron_test/heat2`, `~/lebron_test/nocap-{A,B}`, `~/lebron_test/render`,
`~/lebron_test/fp` and `~/lebron_test/live5-8.json` on minus-2.
