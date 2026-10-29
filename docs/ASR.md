# ASR (audio-based ad confirmation)

## 2026-10-18: SenseVoice on the NPU, the veto back (calibrated), audio as an input

From the `asr-first` experiments (the name-muter branch; its
`docs/ASR_BENCHMARKS.md` has every measurement):

- **Engine: SenseVoice-small on RK3588 NPU core 1** (`src/sensevoice_npu.py`,
  model `happyme531/SenseVoiceSmall-RKNN2` in
  `/home/radxa/asr_models/sensevoice-rknn`, installed by `install.sh` step
  14). ~0.38 s per window whatever its length, ~25% of one CPU core, +3-5 C
  over ASR off (Moonshine medium: +35 C at 86% CPU; that cost is why ASR
  was left switched off). PaddleOCR keeps core 0. 3 s windows every 1 s.
  `MINUS_ASR_ENGINE=sensevoice` is the default and falls back to moonshine
  when the model or `kaldi_native_fbank`/`sentencepiece`/`soundfile` is
  missing; `MINUS_ASR_NPU_CORE` picks the core.
- **The start veto counts again, but only against an unsure VLM.** A
  VLM-alone block doesn't start when ASR hears show dialog with no ad
  language and VLM agreement is below `ASR_VETO_MAX_VLM_RATIO` (0.95,
  `MINUS_ASR_VETO_MAX_VLM`). The cases that got the old veto removed (below)
  were at 80% and 100%: the 100% one still blocks. OCR blocks never consult
  ASR.
- **Audio as an input, for VLM-alone detections** (setting
  `block_requires_audio`, default on; Settings → "Only Block When There's
  Sound"): menus, home-screen banners and paused screens are silent and fool
  the VLM. A VLM-alone block doesn't start after 3 s of input below -55 dBFS
  (`MINUS_AUDIO_GATE_START_S`, `MINUS_AUDIBLE_DBFS`) and ends after 10 s of
  it (`MINUS_AUDIO_GATE_STOP_S`) unless the ad's own countdown says it's
  running. OCR-detected ads block regardless of sound: gating them too left
  silent ad cards (frozen end cards, stalled ads, static image ads) visible,
  32% of all ad frames in a day's test (2026-10-19).
- **Other false-positive guards (2026-10-19):** a VLM-alone start needs VLM's
  agreement to still be >= 60% when it starts (the detection flag is sticky
  while other gates hold it back); and overruling an ASR veto needs VLM to be
  unanimous (>= 95%) AND confident (mean AD confidence >= 0.85). Measured before the mute, so a
  block doesn't read as silence. Unknown audio (no pipeline, no samples)
  never holds a block back.
- **Audio pipeline robustness:** 200 ms alsasrc ring (was 50 ms: a busy box
  overran it and lost 10-20% of the audio), a `srcqueue` after capture, and
  the Python buffer probe on the ASR branch instead of the playback path (a
  GIL stall there paused the output up to 0.7 s).

Not taken from `asr-first`: the name muter, clip farm and demo tooling (that
experiment), Parakeet/Moonshine-medium engines, and its "adaptive delay"
(drops the A/V delay while the input is quiet), which conflicts with the
lip-sync delay on main.

## Before 2026-10-18

**Moonshine tiny-en** (`moonshine-voice`, ONNX) running on 3 pinned CPU
cores in a dedicated multiprocessing worker subprocess
(`src/asr_worker.py`). Fed from a parallel `tee` branch of the existing
GStreamer audio pipeline, used as a **CONFIRM-ONLY** signal that
decorates the block label and gates nothing at start. Does not trigger
blocking alone; does not suppress OCR/VLM blocking.

**Engine: Moonshine (default), faster-whisper (fallback).** Selectable via
`MINUS_ASR_ENGINE=moonshine|faster-whisper`. On the target hardware
(RK3588 doing 4K passthrough) the SoC thermally throttles cores to
408MHz–1.6GHz, where faster-whisper's **fixed 30s-padded encoder** costs
~3.3–5s per window regardless of audio length — a floor that streaming /
smaller chunks does NOT reduce (benchmarked). Moonshine processes audio
**proportionally** (no 30s pad), so a 2s window is ~1.6s (p50, max <2s)
even on wall-to-wall continuous speech. It trades a little accuracy
(~8/10 vs faster-whisper's 10/10 on the clean corpus, and it misses some
quiet/sung speech) but never false-positives on show audio, which is what
matters for the confirm role. Set `MINUS_ASR_ENGINE=faster-whisper` on a
cool/idle host to trade latency for the extra accuracy.

**Window = 2s, env `MINUS_ASR_WINDOW`.** Sized so Moonshine stays under 2s
even on dense speech (5s window was ~3.3s — too slow). Interval 1.5s
(`MINUS_ASR_INTERVAL`), overlapping windows. Since ASR is confirm-only,
a CTA split across two short windows still lands in the 8s rolling
history and a missed confirm only changes the LABEL, never whether a
block fires.

**3-core affinity** (`MINUS_ASR_CPU_AFFINITY`, default `{3,4,5}` on the
8-core RK3588 — the least thermally-throttled cores). Honours the "≤3
CPUs" budget; onnxruntime/CTranslate2 would otherwise grab every core.

**Engine history:** whisper.cpp tiny.en (binary subprocess) →
faster-whisper (corpus benchmark: 25% faster at 10/10) → Moonshine
(2026-05, for sub-2s latency on the thermally-throttled production box;
see *Moonshine migration + decision-engine retune* below). The keyword
module (`src/asr_keywords.py`) is engine-agnostic.

> **2026-05 decision-engine change — the ASR VETO was REMOVED.** ASR used
> to *suppress* a VLM-alone block when it heard speech with no marketing
> markers ("product placement" veto). In practice it suppressed REAL ads
> VLM was 100% sure about (a Hotels.com ad at 80%, an insurance ad at
> 100% — the spoken copy just lacked explicit markers). ASR now NEVER
> suppresses at start; it only (a) decorates the label (`+asr`) when it
> confirms, and (b) does a GATED mid-block rescue that force-stops a
> VLM-only block only once **VLM itself has weakened** (`ad_ratio < 0.5`),
> so a confident visual detection is always trusted. See the table below.

## Why it exists

VLM sometimes calls a frame an ad when the visual is ambiguous — a
character holding a Coke can during a Netflix show, a brand storefront
in the background, a product placement scene. OCR sees nothing (no
"Sponsored" / "Skip" text on screen) and VLM-alone fires the block on
genuine show content.

ASR provides a **second opinion on the audio channel**. Real ads have a
marketing-language acoustic signature (announcer voice, CTAs like
"available now / call 1-800 / save 50%", price strings with cents,
URLs spoken as "dot com"). Product placements in shows have *actor
dialogue*, no marketing copy. Whisper transcribes the audio, the
keyword module counts marketing markers, the verdict gates VLM-alone.

## Architecture

```
HDMI-RX (hw:4,0)
   │
   ▼
alsasrc (existing) ──► audio/x-raw,48kHz,2ch,S16LE
                                        │
                                        ▼
                              tee (allow-not-linked=true)
                              │                        │
       ┌──────────────────────┘                        └──────────────────────┐
       │ PLAYBACK BRANCH (unchanged, latency-critical)                        │ ASR TAP BRANCH (new)
       ▼                                                                     ▼
syncqueue (300ms threshold)                                       queue leaky=downstream
       │                                                                     │
       ▼                                                                     ▼
audioqueue (jitter buffer)                                        audioconvert + audioresample
       │                                                                     │
       ▼                                                                     ▼
audioconvert                                                      audio/x-raw,16kHz,1ch,S16LE
       │                                                                     │
       ▼                                                                     ▼
volume (mute=true during ads) ◄── ad_blocker mute control          appsink "asr_sink"
       │                                                                     │
       ▼                                                                     ▼ (Python callback)
alsasink (hw:0,0 → HDMI-TX)                                       AudioASRTap ring buffer
       │                                                          (8s ring, 16kHz S16LE)
       ▼                                                                     │
       TV speakers                                                           ▼
                                                                  snapshot_to_wav() every 2s
                                                                             │
                                                                             ▼
                                                                  whisper.cpp tiny.en
                                                                  (subprocess, 3 threads,
                                                                   2.5s hard timeout)
                                                                             │
                                                                             ▼
                                                                  transcript → count_marker_hits()
                                                                             │
                                                                             ▼
                                                                  ASRManager.verdict()
                                                                  ──► confirm / veto / unknown
                                                                             │
                                                                             ▼
                                                                  Minus._update_blocking_state()
                                                                  gates VLM-alone block start
                                                                  + force-stops mid-block on veto
```

**Safety invariant:** the playback branch (`syncqueue → audioqueue →
audioconvert → volume → alsasink`) is byte-identical whether the tap
is attached or not. Same elements, same names, same parameters, same
latency budget (~500ms to match video). Tee fanout is zero-copy ref-
counting; latency added to playback = 0. The tap branch is leaky
(`queue leaky=downstream`) so a slow whisper consumer drops buffers
rather than backpressuring the playback side. Asserted by
`tests/test_asr.py::TestAudioPipelineShape`.

## Decision-engine integration (2026-05 — veto removed)

ASR never suppresses a block at start. It decorates the label when it
confirms, and rescues mid-block ONLY when VLM itself weakens.

| OCR | VLM | ASR verdict | Action / `blocking_source` |
|---|---|---|---|
| ad | * | confirm | **block** `ocr+asr` (or `ocr+vlm+asr` if VLM too) — OCR authoritative; ASR decorates label |
| ad | * | veto/unknown | **block** `ocr` (or `both`) — OCR authoritative; ASR ignored |
| no | ad (3+, 80%+) | confirm | **block** `vlm+asr` |
| no | ad (3+, 80%+) | veto/unknown | **block** `vlm` — VLM trusted; **veto IGNORED at start** |
| (mid-block) | source=`vlm`, **ad_ratio <0.5** | veto ≥4s | **force-stop** — VLM has weakened AND ASR hears show audio |
| (mid-block) | source=`vlm`, ad_ratio ≥0.5 | veto | **keep blocking** — VLM still confident, ASR cannot override |

Notes:

- `+asr` is a label only (`blocking_asr_confirmed` flag); `blocking_source`
  stays the base `ocr`/`vlm`/`both` for all stop-logic. Set at start from
  the ASR verdict and **upgraded mid-block** when ASR later confirms
  (the common path — OCR blocks instantly, ASR confirms a few seconds in).
  Overlay headers: `OCR+ASR`, `VLM+ASR`, `OCR+VLM+ASR`. The per-frame log tag
  (`OCR #N: ... [BLOCKING OCR+VLM+ASR]`) names ASR the same way once it has
  confirmed the block (`Minus._ocr_log_blocking_tag`; before Oct 2026 it only
  ever said OCR / VLM / OCR+VLM). A mid-block upgrade only applies to an active
  block, so it is logged as `ASR confirmed active block → OCR+VLM+ASR`.
- `verdict='confirm'`: ≥1 marker hit in the rolling 8-second window.
- `verdict='veto'`: 0 marker hits AND ≥10 alpha-words. NO LONGER suppresses
  at start; only feeds the gated mid-block rescue.
- `verdict='unknown'`: cold start, music/silence, ASR disabled, or worker
  hung. Always degrades to plain VLM/OCR behaviour.
- The mid-block rescue is gated on `ad_ratio < 0.5` precisely so it can't
  kill a real ad whose spoken copy lacks markers (a Hotels.com / insurance
  ad VLM is sure about). A genuine brand-in-a-show FP shows VLM drifting to
  no-ad as the scene continues, so its ad_ratio drops and the rescue fires.

## Install

```bash
pip3 install --break-system-packages --user moonshine-voice   # default engine
pip3 install --break-system-packages --user faster-whisper     # fallback engine
```

Moonshine ONNX model files auto-download on first use (tiny-en, ~40 MB);
faster-whisper's tiny.en (~75 MB) downloads to `~/.cache/huggingface/`.
Subsequent loads are local.

Env knobs: `MINUS_ASR_ENGINE` (`moonshine`/`faster-whisper`),
`MINUS_ASR_WINDOW` (2.0s), `MINUS_ASR_INTERVAL` (1.5s),
`MINUS_ASR_CPU_AFFINITY` (`3,4,5`), `MINUS_ASR_SOFT_TIMEOUT` (4.0s),
`MINUS_ASR_HARD_TIMEOUT` (6.0s), `MINUS_ASR_MODEL` (faster-whisper size).

**Self-deadlock fix (2026-05):** `ASRProcess._call_lock` is a
`threading.RLock` (was a plain `Lock`). `transcribe()` holds the lock and,
on the hard-timeout escalation, calls `restart()` → `stop()`/`start()`,
which re-acquire it — a plain Lock self-deadlocked there and froze the
worker mid-restart under sustained load (observed as `inference_count`
stuck with no `killed` count).

**Manual test path:** `POST /api/asr/test` (and the Home → ASR-Live
"Test (ad)" / "Test (show)" buttons) pipe a bundled corpus clip through the
live worker and update the panel — lets you exercise ASR end-to-end with
no live audio (e.g. TV off → source sends digital silence). The endpoint
trims the clip to ~4s so it fits the 2s-window-tuned worker timeout.

CTranslate2 (the inference engine under faster-whisper) auto-detects
ARM NEON kernels on RK3588 — no manual flags needed.

**whisper.cpp (legacy):** the original implementation lives at
`/home/radxa/whisper.cpp`. Kept on disk for reference + the
`tests/asr_corpus/bench.py` baseline comparison. No longer
loaded at runtime.

## Cost on RK3588

Bench: `tests/asr_corpus/bench.py` against 10-sample synthesized corpus.
**Measured on RK3588 with `minus` service stopped** so each engine got
all 8 cores to itself (no contention with OCR/VLM workers). 3 threads
each for direct comparison. Re-run from the repo root:

```
python3 tests/asr_corpus/bench.py
```

| Engine | Pass | Total | Avg / 5 s window | Notes |
|---|---|---|---|---|
| **faster-whisper tiny.en** (CTranslate2 int8) | **10/10** | 11.45 s | **1.14 s** | Fastest at 10/10. Best transcripts. |
| moonshine tiny (ONNX) | 8/10 | 11.25 s | 1.13 s | Tied on speed; missed 2 subtle ads (`narcissism`-style hallucinations dropped one marker each). |
| **whisper.cpp tiny.en** *(current)* | 10/10 | 15.15 s | 1.52 s | Production baseline. |
| moonshine small-streaming (ONNX) | 9/10 | 19.55 s | 1.95 s | Cleaner transcripts than tiny; one ad-pharma miss. |
| moonshine medium-streaming (ONNX) | 10/10 | 26.31 s | 2.63 s | Best transcription quality of any tested model. |

**Findings:**

1. **faster-whisper tiny.en is the winner**: 25% faster than whisper.cpp
   at identical pass rate (10/10), and slightly better transcript quality
   (correct punctuation, retains `$` in phone numbers like `$15`,
   captures `1-800-555-1234` shape). CTranslate2 int8 quantization
   does heavy lifting.
2. **Moonshine tiny (ONNX) is competitive with faster-whisper on speed**,
   but loses 2 subtle-ad cases in this corpus. The streaming variants
   exist primarily for *streaming* use (live audio chunks); we're using
   them as non-streaming transcribers which probably understates their
   real strength.
3. **Moonshine medium-streaming** has the cleanest transcripts of any
   model tested — "Now only 999. Free shipping on orders over $25.
   Order yours today" is basically perfect. 10/10 pass, but 2.3×
   slower than faster-whisper.

**The first install attempt did NOT use the right Moonshine package.**
We initially installed `useful-moonshine` from PyPI which is the
deprecated Keras+Torch implementation — that gave us ~9 s/sample,
orders of magnitude slower than the published Pi 5 benchmarks. The
correct package is `moonshine-voice` which ships ONNX models and the
optimized C++/ONNX runtime. The numbers above are from `moonshine-voice
0.1.0`.

**Trade-off vs whisper.cpp (the architectural angle):**
- whisper.cpp invokes a separate binary subprocess per inference.
  Hard timeout is straightforward (subprocess.run with `timeout=`).
  If whisper.cpp hangs, the OS reaps the child. Zero risk of Python-
  memory leak through repeated runs.
- faster-whisper runs in-process. Hard timeout requires either
  threading (with kill via `_thread.interrupt_main()` — messy) or
  subprocess-wrapping the Python invocation. We'd need to match the
  OCR/VLM worker pattern (multiprocessing.Process + Queue) for
  parity with the existing safety story.
- Moonshine `Transcriber` is a Python-wrapped C++ library; in-process
  with native code under the hood. Same Python-side timeout issue as
  faster-whisper, plus a less battle-tested project (younger than
  whisper.cpp).

Sample transcripts (ad_cta_strong.wav, real text: "Call now! 1-800-555-1234. Save up to 50 percent..."):
- whisper.cpp tiny: `"Call now 85551234. Same up to 50%. Available now on our website..."`
- faster-whisper tiny: `"Call now, 1-800-555-1234  Same up to 50%  Available now at our website..."` ← clean toll-free
- moonshine tiny: `"Cold power. 1800 555 1234 Same up to 50%..."` ← "Cold power" hallucination from "Call now"
- moonshine small-streaming: `"Call now. 1800-555-1234. Save up to 50%  Limited time only."`
- moonshine medium-streaming: `"Call now. 18551234 Save up to 50 percent Available now at our website..."` ← best

Memory: tiny.en model + ring buffer + ONNX/CT2 runtime ≈ 200-300 MB
depending on engine. Plenty of headroom (~12 GB free).

3× real-time on 3 threads (current whisper.cpp) → 2-second inference
cadence has 50%+ idle margin. faster-whisper at 1.14s/5s-window would
push that to ~75% idle. Either way, no audio backpressure concern.

**Recommendation**: faster-whisper tiny.en is a credible upgrade path
(25% faster, slightly better quality, same accuracy on this corpus)
but the swap requires migrating to a Python-subprocess worker pattern
to preserve the hard-timeout safety guarantee. Defer the engine swap
until we've validated real-world TV audio for ~24h on the current
whisper.cpp build. The keyword module is engine-agnostic, so the
future swap is a localized change in `src/asr.py` only.

## Keyword set design (`src/asr_keywords.py`)

Whisper-tiny mistranscribes much like OCR mis-reads — drops syllables,
mangles uncommon words, hallucinates short phrases on silence. The
keyword set accommodates this:

- **Multiple variants per phrase**: `available now` also matches
  `vailable now`, `a vailable now`. Tiny.en regularly drops unstressed
  syllables.
- **Phrase-level matching** (not substring): `now` does not match
  everywhere; `available now` is a full phrase.
- **Regex shape matchers** for prices, URLs, phone numbers: structurally
  stable across transcription noise.
- **Exclusion list**: show-content phrasings that incidentally use
  keywords (`available on netflix this friday`, `subscribe to my
  channel`, `previously on`). These take priority and produce 0 hits
  regardless of any markers also present.
- **Tightened price regex**: `$X` alone is NOT a marker (show characters
  mention bare dollar amounts; whisper renders "fifty dollars" as
  "$15"). Only `$X.XX` with cents, `only $X`, `save $X`, `starting at
  $X`, "X dollars and Y cents" — contextual prices.
- **Minimum-content gate**: transcripts with <3 alpha words score 0
  (filters whisper hallucinations on silence like "you" / "thank you").

If tiny.en quality proves insufficient, swap the model via
`MINUS_WHISPER_MODEL=ggml-base.en.bin` — the keyword module is
backend-agnostic, no code change needed. See task #22 for the
faster-whisper / Moonshine evaluation.

## Control corpus

`tests/asr_corpus/` — 10 espeak-ng synthesized samples spanning ad copy
(pharma, strong CTA, subtle, price-focused, streaming promo) and
negatives (show dialog, money-mention, Netflix promo, YouTube creator,
silence). Each sample has an `expected_min_hits` ground-truth value;
`bench.py` validates the keyword module + transcription engine against
those expectations and writes `BENCH_RESULTS.json` for the architecture
log.

Run:
```
python3 tests/asr_corpus/bench.py
```

## Testing

`tests/test_asr.py` — 44 tests covering:

- **Keyword module**: ad-copy scores, mistranscription resilience,
  exclusion list overrides, whisper-hallucination filtering.
- **ASRManager state machine**: verdict() three-state output, rolling
  window aging, graceful degradation when ASR is disabled/missing.
- **AudioASRTap ring buffer**: write correctness, wraparound, snapshot
  atomicity, concurrent write+snapshot.
- **Pipeline shape**: playback branch byte-identical with/without tap;
  tee + appsink only present with tap; tap branch is leaky; tap
  resamples to 16kHz mono.
- **Audio recovery survival**: `_init_pipeline` re-attaches the tap on
  pipeline restart (HDMI-RX sleep/wake survives); watchdog still finds
  `audioqueue`; mute element `vol` still present; ALSA zombie detection
  intact.
- **Decision-engine integration**: ASR veto suppresses VLM-alone start;
  ASR unknown lets VLM fire alone; ASR confirm upgrades source label to
  `vlm+asr`; OCR-driven blocks unaffected.

End-to-end live test runs as part of `tests/asr_corpus/bench.py` which
invokes the real whisper.cpp binary on synthesized audio.

## Known limitations

- **Music-heavy ads** with no spoken copy (pure jingle): no transcript
  markers → ASR returns `unknown`, VLM fires alone (current behaviour
  preserved).
- **Foreign-language content**: tiny.en is English-only. Spanish ads
  produce 0-marker transcripts → ASR `unknown` or `veto` (false veto).
  Spanish support is deferred — see task #22.
- **Multi-speaker scenes** with overlapping dialogue: whisper-tiny
  transcript quality degrades, marker precision drops. Keyword set is
  designed to err on the side of `unknown` (not falsely confirm).
- **Whisper warm-up**: the FIRST inference after a model-load takes
  ~1.5s longer than subsequent calls. The ASR thread sleeps 5s after
  start before the first snapshot, so this only delays the first
  verdict — never blocks the audio pipeline.
