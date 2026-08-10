# Minus - Development Notes

## Overview

HDMI passthrough with real-time ML-based ad detection and blocking using dual NPUs + CPU ASR:
- **PaddleOCR** on RK3588 NPU (~400ms per frame, 1.0s timeout)
- **LFM2.5-VL-450M (ft-v2-fused-v2)** on Axera LLM 8850 NPU — **prefill-only on 16 fused decoder layers, ~0.37s per frame deterministic** (1.5s soft / 2s hard timeout). Replaced FastVLM-0.5B iter4 (May 2026): 97.0% holdout accuracy / 99.2% non-ad-recall vs iter4's 94.75% / 95.25%, structurally simpler (no KV cache, no autoregressive decode for `detect_ad` OR autonomous-mode `query_image`, no ml_dtypes bfloat16 ceremony). Both inference paths share one model — no FastVLM dependency anymore. See *FastVLM iter4 → LFM2.5-VL Migration* under Known Issues.
- **Moonshine tiny-en (ONNX) ASR** on 3 pinned CPU cores (~1.6s per **2s** audio window, max <2s even on dense continuous speech). Runs in a multiprocessing worker subprocess (mirrors OCR/VLM worker pattern) for hard-timeout safety. **CONFIRM-ONLY** audio signal on top of OCR+VLM: decorates the block label (`+asr`) and does a gated mid-block rescue, but **never suppresses a block at start** (the old VETO was removed in 2026-05 — it was killing real ads VLM was sure about). Never fires blocking alone. Engine-selectable via `MINUS_ASR_ENGINE` (faster-whisper fallback for cool/idle hosts; on the thermally-throttled production box faster-whisper's fixed 30s encoder is ~3.3-5s/window — too slow, hence Moonshine which processes audio proportionally). Was whisper.cpp → faster-whisper → Moonshine. See [docs/ASR.md](docs/ASR.md) and *Moonshine ASR migration + decision-engine retune* under Known Issues.
- **Spanish vocabulary practice** during ad blocks!

## Documentation

| Document | Description |
|----------|-------------|
| [docs/FEATURES.md](docs/FEATURES.md) | Complete feature list and capabilities |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and data flow |
| [docs/AESTHETICS.md](docs/AESTHETICS.md) | Visual design guide for UI/overlays |
| [MARISOL.md](MARISOL.md) | AI agent context guide |
| [docs/DEBUG_GLITCHES.md](docs/DEBUG_GLITCHES.md) | Video glitch debugging notes |
| [docs/FPS_DEBUGGING.md](docs/FPS_DEBUGGING.md) | FPS tracking and optimization |
| [docs/AUDIO.md](docs/AUDIO.md) | Audio passthrough documentation |
| [docs/ASR.md](docs/ASR.md) | Moonshine (ONNX) ASR — audio-based ad CONFIRM signal on top of OCR+VLM (worker process, 2s window, veto removed 2026-05) |
| [docs/VLM_NPU_DEGRADATION.md](docs/VLM_NPU_DEGRADATION.md) | Investigation of "NPU degradation" — root cause is per-image output-length variance; fix is `max_new_tokens` cap |
| [docs/IR_TRANSMITTER.md](docs/IR_TRANSMITTER.md) | IR transmitter for the REI 8K HDMI switch (PWM3 on pin 38) — wiring, NEC codes, API, troubleshooting |
| [docs/IR_RECEIVER.md](docs/IR_RECEIVER.md) | IR receiver eval on pin 3 (`gpiochip4 11`) — bench-tested decode of NEC remotes, gotchas, sketch for a future `IRReceiver` module |
| [docs/STATUS_LEDS.md](docs/STATUS_LEDS.md) | WS2812B status strip on SPI0 MOSI (pin 19) — wiring, state catalogue, API, encoding rationale |

## Visual Design

See **[docs/AESTHETICS.md](docs/AESTHETICS.md)** for the complete visual design guide including:
- Color palette (black background, matrix green, danger red, purple accents)
- Typography (VT323 for display, IBM Plex Mono for body, DejaVu for TV overlays)
- Component styling and animations
- TV overlay layout specifications

## Architecture

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│   HDMI-RX    │────▶│     ustreamer      │────▶│  GStreamer Pipeline │
│ /dev/video0  │     │ (MJPEG encoding)   │     │  (queue + kmssink)  │
│  4K@30fps    │     │                    │     │                     │
│              │     │   :9090/stream     │     │                     │
│              │     │   :9090/snapshot   │     │                     │
└──────────────┘     └────────┬───────────┘     └─────────────────────┘
                              │
                              ▼ HTTP snapshot (~150ms, non-blocking)
              ┌───────────────┴───────────────┐
              │                               │
     ┌────────┴────────┐           ┌──────────┴──────────┐
     │   OCR Worker    │           │    VLM Worker       │
     │  ┌───────────┐  │           │  ┌───────────────┐  │
     │  │ PaddleOCR │  │           │  │  LFM2.5-VL    │  │
     │  │ RK3588 NPU│  │           │  │ Axera LLM 8850│  │
     │  │ ~400ms    │  │           │  │ ~0.37s        │  │
     │  └───────────┘  │           │  └───────────────┘  │
     └────────┬────────┘           └──────────┬──────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                     ┌────────┴────────┐
                     │ Blocking Mode   │
                     │ (ustreamer API) │
                     └─────────────────┘
```

**Key Architecture Points:**
- Simple GStreamer pipeline with `queue max-size-buffers=3 leaky=downstream`
- All blocking overlay rendering done in ustreamer's MPP encoder at 60fps
- No X11 required - uses DRM/KMS directly via kmssink
- **Auto-detects HDMI output, resolution, and DRM plane** at startup
- Works with both 4K and 1080p displays (uses display's preferred resolution)
- Both ML workers run concurrently on separate NPUs
- Display runs independently at 30fps without any stutter

## Key Files

| File | Purpose |
|------|---------|
| `minus.py` | Main entry point - orchestrates everything |
| `minus.spec` | PyInstaller spec for building executable |
| `src/ad_blocker.py` | GStreamer video pipeline, blocking API client |
| `src/audio.py` | GStreamer audio passthrough with mute control |
| `src/ocr.py` | PaddleOCR on RKNN NPU, keyword detection |
| `src/ocr_worker.py` | Process-based OCR with hard timeout, warmup, and keepalive |
| `src/vlm.py` | LFM2.5-VL-450M on Axera NPU — prefill-only `detect_ad` (argmax YES/NO logits) + prefill-only `query_image` (first-token class logit) |
| `src/vlm_worker.py` | Process-based VLM with hard timeout, warmup, and keepalive |
| `src/autonomous_mode.py` | Autonomous mode - VLM-guided YouTube playback |
| `src/health.py` | Unified health monitor for all subsystems |
| `src/webui.py` | Flask web UI for remote monitoring/control |
| `src/fire_tv.py` | Fire TV ADB remote control for ad skipping |
| `src/roku.py` | Roku ECP remote control |
| `src/ir_transmitter.py` | NEC IR transmitter over PWM3 (REI 8K HDMI switch). Thread-safe, 1.5 s cooldown |
| `src/status_leds.py` | Raw WS2812B SPI driver. 8 LEDs, 10% brightness cap, Adafruit-canonical 8-bit-per-WS-bit encoding at 6.4 MHz |
| `src/status_led_controller.py` | State machine + animation thread on top of `status_leds.py`. States: off/initializing/idle/blocking/no_signal/autonomous/error |
| `src/device_config.py` | Streaming device type configuration and persistence |
| `src/fire_tv_setup.py` | Fire TV auto-setup flow with overlay notifications |
| `src/wifi_manager.py` | WiFi captive portal and AP mode management |
| `src/overlay.py` | Notification overlay via ustreamer API |
| `src/vocabulary.py` | Spanish vocabulary — original `SPANISH_VOCABULARY` (~550 entries, 4-tuples) plus `SPANISH_VOCABULARY_EXTENDED` (~200 entries, 5-tuples with two example sentences). `VOCABULARY_COMBINED` is the unified list the ad overlay iterates. |
| `src/console.py` | Console blanking/restore functions |
| `src/drm.py` | DRM output probing, adaptive bandwidth fallback |
| `src/v4l2.py` | V4L2 device probing (format, resolution) |
| `src/config.py` | MinusConfig dataclass |
| `src/capture.py` | UstreamerCapture class for snapshot capture |
| `src/screenshots.py` | ScreenshotManager with dHash dedup + blank rejection |
| `src/skip_detection.py` | Skip button detection (regex patterns) |
| `test_fire_tv.py` | Fire TV controller test and interactive remote |
| `ir_transmit.py` | Standalone CLI for the IR transmitter (`sudo python3 ir_transmit.py <button>`) |
| `tests/test_modules.py` | Comprehensive test suite (300+ tests) |
| `tools/ad_block_monitor.py` | Log-driven ad-block health monitor (recovery latency, weak-FP / overlong / query-error triage); run periodically/by a recurring agent |
| `tests/test_autonomous_mode.py` | Autonomous mode unit tests |
| `tests/test_review_ui.py` | Playwright UI tests for screenshot review |
| `tests/test_ir_transmitter.py` | Unit tests for IR transmitter (mocked sysfs, 20 tests) |
| `tests/test_ir_ui.py` | Playwright UI tests for IR remote panel |
| `tests/test_status_led_controller.py` | Unit tests for status-LED state machine (mocked hardware, 31 tests) |
| `tests/test_status_leds_ui.py` | Playwright UI tests for status-LED toggle + state palette |
| `tests/test_status_led_states.py` | Hardware walk: every controller state across all 8 LEDs, 5 s each |
| `test_status_leds.py` | Hardware walk/flash test for the WS2812B strip (R/G/B/W) |
| `tests/test_ocr_ad_detection.py` | OCR ad pattern detection tests (143+ cases) |
| `src/templates/index.html` | Web UI single-page app |
| `src/static/style.css` | Web UI dark theme styles |
| `install.sh` | Install as systemd service |
| `uninstall.sh` | Remove systemd service |
| `stop.sh` | Graceful shutdown script |
| `minus.service` | systemd service file |
| `screenshots/ads/` | OCR-detected ads (for training) |
| `screenshots/non_ads/` | User paused = false positives (for training). For VLM-only blocks, the saved frame is the *VLM-triggering frame* (cached on each `is_ad=True` verdict) rather than the current frame at pause time. |
| `screenshots/vlm_spastic/` | VLM uncertainty cases (for analysis) |
| `screenshots/static/` | Static screen suppression (still frames) |

## Running

```bash
python3 minus.py
```

**Command-line options:**
```bash
--device /dev/video1      # Custom capture device
--ocr-timeout 1.5         # OCR timeout in seconds (default: 1.5)
--max-screenshots 100     # Keep N recent screenshots (default: 50, 0=unlimited)
--check-signal            # Just check HDMI signal and exit
--connector-id 231        # DRM connector ID (auto-detected if not specified)
--plane-id 192            # DRM plane ID (auto-detected if not specified)
--webui-port 80         # Web UI port (default: 80)
```

**Auto-detection at startup:**
- **Connected HDMI output** - Works with either HDMI-A-1 (connector 215) or HDMI-A-2 (connector 231)
- **Preferred resolution** - Reads EDID to get the display's preferred mode (e.g., 4K@60Hz or 1080p@60Hz)
- **NV12-capable overlay plane** - Finds a suitable DRM plane that supports NV12 format for video output
- **Audio output device** - Matches ALSA device to the connected HDMI output (hw:0,0 for HDMI-A-1, hw:1,0 for HDMI-A-2)

This allows Minus to work with different displays without manual configuration.

**Adaptive HDMI Bandwidth Fallback:**

4K@60Hz RGB/YCbCr 4:4:4 requires 18 Gbps HDMI bandwidth. Some cables, adapters, or display paths can't handle this, resulting in "No Signal" on the TV even though the kernel reports success.

Minus includes adaptive bandwidth detection via `src/drm.py`:

| Function | Purpose |
|----------|---------|
| `get_color_format(connector_id)` | Read current color format (RGB, YCbCr 4:4:4, 4:2:2, 4:2:0) |
| `set_color_format(connector_id, format)` | Set color format with retry logic |
| `check_hdmi_i2c_errors(threshold, window)` | Detect signal problems via dmesg |

**Detection heuristic:** When HDMI signal fails at high bandwidth, the dwhdmi driver floods dmesg with `i2c read err!` messages. This is more reliable than kernel connector status (which shows "connected" even when signal fails).

**Color format values:**
- `COLOR_FORMAT_RGB` (0) - Full bandwidth
- `COLOR_FORMAT_YCBCR444` (1) - Full bandwidth
- `COLOR_FORMAT_YCBCR422` (2) - Reduced bandwidth
- `COLOR_FORMAT_YCBCR420` (3) - **Half bandwidth (9 Gbps)** - use for problematic cables

**Manual fallback:**
```bash
# Stop minus first (it holds DRM master lock)
sudo systemctl stop minus

# Set YCbCr 4:2:0 for half bandwidth
sudo modetest -M rockchip -w 215:color_format:3

# Restart minus
sudo systemctl start minus
```

**Environment variables:**
```bash
# Paths (override defaults for different installations)
MINUS_USTREAMER_PATH=/path/to/ustreamer     # Default: /home/radxa/ustreamer-patched
MINUS_VLM_MODEL_DIR=/path/to/vlm/models     # Default: /home/radxa/axera_models/minus-v0.1
MINUS_OCR_MODEL_DIR=/path/to/ocr/models     # Default: /home/radxa/rknn-llm/.../paddleocr

# Timing thresholds
MINUS_ANIMATION_START=0.3        # Blocking animation duration (seconds)
MINUS_ANIMATION_END=0.25         # Unblocking animation duration (seconds)
MINUS_FRAME_STALE_THRESHOLD=5.0  # Health check frame freshness (seconds)
MINUS_DYNAMIC_COOLDOWN=0.5       # Wait after screen becomes dynamic (seconds)
MINUS_SCENE_CHANGE_THRESHOLD=0.01  # Frame difference threshold for scene change
MINUS_VLM_ALONE_THRESHOLD=5      # Consecutive VLM detections needed to trigger alone
```

## Performance

| Metric | Value |
|--------|-------|
| Display (video) | **30fps** (GStreamer kmssink, MJPEG → NV12 → DRM plane) |
| Display (blocking) | **60fps** (ustreamer MPP blocking mode with FreeType) |
| Preview window | **60fps** (hardware-scaled in MPP encoder) |
| Blocking composite | **~0.5ms** per frame overhead |
| Audio mute/unmute | **INSTANT** (volume element mute property) |
| ustreamer MJPEG stream | **~60fps** (MPP hardware encoding at 4K) |
| OCR latency | **100-200ms** capture + **250-400ms** inference |
| VLM latency | **~0.37s per frame deterministic** (LFM2.5-VL fused-prefill; vision ~185ms + 16 fused layers ~185ms; no decode) |
| VLM model load | **~9-11s** (17 axengine sessions + 256MB embeds mmap + 4 warmup inferences + keepalive thread) |
| Snapshot capture | **~150ms** (4K JPEG download) |
| OCR image size | 960x540 (downscaled from 4K for speed) |
| ustreamer quality | 80% JPEG (MPP encoder) |
| Animation start | **0.3s** (fast blocking response) |
| Animation end | **0.25s** (fast unblocking) |

**FPS Tracking:**
- GStreamer identity element with pad probe counts frames
- FPS logged every 60 seconds via health monitor
- Warning logged if FPS drops below 25

## ustreamer-patched (NV12 + MPP Hardware Encoding)

We use a patched version of ustreamer from `garagehq/ustreamer` that adds:
- **NV12/NV16/NV24 format support** for RK3588 HDMI-RX devices
- **MPP hardware JPEG encoding** using RK3588 VPU (~60fps at 4K!)
- **Blocking mode system** with FreeType TrueType rendering for ad blocking overlays
- **Extended timeouts** for RK3588 HDMI-RX driver compatibility
- **Multi-worker MPP support** (4 parallel encoders optimal)
- **Cache sync fix** for DMA-related visual artifacts
- **Thread-safe FreeType** mutex for multi-worker encoding

**Why patched ustreamer?**
The stock PiKVM ustreamer doesn't support NV12 format or RK3588 hardware encoding.
Our fork adds NV12→JPEG encoding via Rockchip MPP (Media Process Platform) that
achieves ~60fps on 4K input with minimal CPU usage.

**Dynamic Format Detection:**
Minus automatically probes the V4L2 device to detect its current format and resolution. Supported formats:
- **NV12** - RK3588 HDMI-RX native (uses MPP hardware encoder directly)
- **NV24** - Some devices like Roku (converted to NV12 for MPP, ~60fps)
- **BGR24/BGR3** - Google TV and similar devices (converted to NV12 for MPP, ~42fps at 4K)
- **YUYV/UYVY** - Webcam-style devices
- **MJPEG** - Pre-compressed JPEG sources

Format conversions (NV24→NV12, BGR24→NV12) are done in software in the MPP encoder before hardware JPEG encoding.

**Performance comparison (4K HDMI input):**

| Mode | ustreamer FPS | CPU Usage | Notes |
|------|---------------|-----------|-------|
| CPU encoding | ~4 fps | ~100% | CPU can't keep up with 4K JPEG encoding |
| MPP hardware | **~60 fps** | **~5%** | `--encoder=mpp-jpeg` (default) |

**ustreamer command (used by Minus):**
```bash
/home/radxa/ustreamer-patched \
  --device=/dev/video0 \
  --format=NV12 \
  --resolution=3840x2160 \
  --persistent \
  --port=9090 \
  --host=0.0.0.0 \
  --encoder=mpp-jpeg \
  --encode-scale=passthrough \
  --quality=80 \
  --workers=4 \
  --buffers=5
```

**Installation:**
```bash
# Clone and build with MPP support
git clone https://github.com/garagehq/ustreamer.git /home/radxa/ustreamer-garagehq
cd /home/radxa/ustreamer-garagehq
make WITH_MPP=1
cp ustreamer /home/radxa/ustreamer-patched

# Minus uses /home/radxa/ustreamer-patched automatically
```

**Key changes in garagehq/ustreamer:**
- `src/ustreamer/encoders/mpp/encoder.c` - MPP hardware JPEG encoder with cache sync, blocking composite, NV24→NV12 and BGR24→NV12 format conversion
- `src/libs/capture.c` - NV12/NV16/NV24/BGR24 format support, extended timeouts
- `src/libs/blocking.c` - FreeType text rendering, NV12 compositing, thread-safe mutex
- `src/ustreamer/http/server.c` - Blocking API endpoints (`/blocking`, `/blocking/set`, `/blocking/background`)
- `src/ustreamer/encoder.c` - MPP encoder integration, multi-worker support
- `src/ustreamer/options.c` - `--encoder=mpp-jpeg` CLI option

## Audio Passthrough

**Hardware:**
- Capture: `hw:4,0` (rockchip,hdmiin) - HDMI-RX audio input
- Playback: `hw:0,0` (rockchip-hdmi0) - HDMI-TX0 output
- Format: 48kHz, stereo, S16LE

**GStreamer Pipeline:**
```
alsasrc (HDMI) ──┐
                 ├──► audiomixer ──► volume ──► alsasink
audiotestsrc ────┘
(silent keepalive)
```

The `audiotestsrc wave=silence` provides a silent keepalive that prevents pipeline stalls when the HDMI source has no audio (between songs, during video silence, etc.).

**Mute Control:**
- `ad_blocker.show()` calls `audio.mute()` - instant mute during ads
- `ad_blocker.hide()` calls `audio.unmute()` - restore audio after ads
- Uses GStreamer `volume` element's `mute` property (no pipeline restart)

**Why separate pipeline?**
- Audio runs independently from video - simpler debugging
- If audio fails, video continues unaffected
- No sync issues for live passthrough

**Error Recovery:**
- GStreamer bus monitors for pipeline errors and EOS
- Buffer probe tracks audio flow (detects stalls)
- Watchdog thread checks every 3s, restarts if no buffer for 6s
- Exponential backoff for restarts (1s → 2s → 4s → ... → 60s max)
- No maximum restart limit - always tries to recover
- Backoff resets after 5 seconds of sustained audio flow
- Mute state is preserved across restarts

**Testing:**
```bash
# Test passthrough manually
gst-launch-1.0 alsasrc device=hw:4,0 ! \
  "audio/x-raw,rate=48000,channels=2,format=S16LE" ! \
  audioconvert ! audioresample ! \
  alsasink device=hw:0,0 sync=false

# Check if HDMI source has audio
v4l2-ctl -d /dev/video0 --get-ctrl audio_present
```

## Ad Detection Logic (Weighted Model)

**OCR (Primary - Authoritative, with triangulation safety net):**
- Triggers blocking after a brief dwell — `OCR_TRANSIENCE_MIN_FRAMES=2` consecutive OCR-matched frames (~500-1000ms penalty at OCR's ~500ms cadence). **Fast-fires on 1 frame (no dwell)** when ANY of: (a) the matched keyword is a *definitive ad-UI keyword* (`DEFINITIVE_AD_KEYWORD_NAMES` = `STRONG_AD_KEYWORD_NAMES` minus `sponsored`: `skip ad`/`skip ads`/`skip in`/`skip ad (fuzzy*)`/`video will play after ad`/`visit advertiser`/`ad X of Y`/`ad countdown`/`ad with timestamp(+cross-element)`); (b) VLM is asserting ad; (c) ASR confirms. Rationale: a single-frame OCR misread (movie billboard with "SKIP", actor holding a sign reading "Sponsored", caption containing 'BUY') shouldn't trigger blocking — but those artifact strings match **none** of the definitive keywords (which only ever appear inside an active ad overlay), so fast-firing them reintroduces no FP risk while cutting ~one OCR cycle (~0.8-1.5s) off the most common ad-break activation. `sponsored` is excluded from the definitive set (it legitimately appears on home/promo tiles and as show-content text) and keeps the 2-frame dwell, as do weak keywords. Real ad UIs keep the keyword visible continuously, so 2 consecutive frames is trivially cleared. env-overridable: `MINUS_OCR_TRANSIENCE_MIN_FRAMES`. See *Definitive-keyword single-frame fast-fire* under Known Issues.
- Stops blocking after **2 consecutive no-ads** (`OCR_STOP_THRESHOLD=2`, was 4 — tuned via `tests/block_latency_harness.py`)
- **Authoritative for stopping** when OCR triggered the block — UNLESS the triangulation veto fires (see below).
- Tracks `last_ocr_ad_time` for VLM context
- Handles common OCR misreads in ad timestamps (see below)

**Triangulation veto (OCR ↔ VLM ↔ ASR cross-check):** an OCR-source (or "both"-source) block is force-stopped when ALL of these hold for ≥`OCR_TRIANGULATION_MIN_BLOCK_S` (4s) into the block:
- VLM sliding-window agreement says **≥80% no-ad** over ≥`vlm_min_decisions` decisions (`OCR_TRIANGULATION_VLM_NOAD_RATIO`)
- ASR `verdict() == 'veto'` — clear show dialog with no marketing markers in the rolling window
- OCR is **NOT** sustained — `ocr_ad_detection_count < OCR_TRUSTED_DWELL_FRAMES` (3). Sustained OCR overpowers the veto: a continuously-visible "Skip in 15" or "Ad 2 of 3" UI is ground truth, transient VLM/ASR noise cannot override it.

Motivating case: OCR matches an ad keyword that's actually a TV-show artifact — a billboard with "SKIP" in a movie, a news ticker passing through "BUY", a movie title containing the word "Sponsored", a paused-on-pause-screen tile. The transience guard catches the 1-frame version; this veto catches the 2-4-frame version when VLM (seeing a regular show frame) and ASR (hearing actor dialogue) both clearly disagree. env-overridable: `MINUS_OCR_TRIANGULATION_MIN_BLOCK_S`, `MINUS_OCR_TRIANGULATION_VLM_NOAD_RATIO`, `MINUS_OCR_TRUSTED_DWELL_FRAMES`.

**VLM (Secondary - Anti-Waffle Protected):**
- Uses sliding window of last **8 seconds** of VLM decisions (`vlm_history_window`). **Why 8s, not 45s:** a long window keeps stale *content* (no-ad) votes that mathematically prevent a VLM-alone ad from ever reaching the start-agreement ratio until they age out — measured VLM-only detect was ~38s with 81% of VLM-only ads missed. 8s collapses that to ~6-7s with ~0 misses and 0 phantom blocks (swept over 1920 param combos × thousands of holdout-bootstrapped scenarios; see *FastVLM-1.5B → 0.5B iter4 Logit-Threshold Migration* and `tests/test_vlm_decision_sim.py`). Stop responsiveness is governed by the consecutive counter, not the window, so shrinking it has no recovery downside.
- Only triggers blocking alone if **90%+ effective agreement** (`vlm_start_agreement` 80% + `vlm_hysteresis_boost` 10%) — hardened from 80% after real mid-Netflix-show false VLM-only triggers
- Hysteresis: capped at 95% via `vlm_start_threshold_cap` so a few stragglers can't block forever
- Minimum **5 decisions** in window before VLM can act (`vlm_min_decisions`; hardened from 3 — ~5s of sustained ad-agreement, not a ~3s transient burst)
- 8-second cooldown after state changes prevents rapid flip-flopping (`vlm_min_state_duration`)
- **Sliding window only for starting** - stopping uses simple consecutive count (`VLM_STOP_THRESHOLD=2`)

**Sliding Window Parameters:**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `vlm_history_window` | 8s | How far back to look at VLM decisions (was 45s; collapsed for iter4 — see migration note) |
| `vlm_min_decisions` | 3 | Min decisions before VLM-only acts (4→3→5→**3**; iter4 hardened to 5 vs iter4-era mid-show FPs, LFM2 reverted to 3 because LFM2's ~4× lower per-frame FP rate makes that hardening unnecessary — see *FastVLM iter4 → LFM2.5-VL Migration*) |
| `vlm_start_agreement` | 70% | VLM-only start agreement (90→80→70→80→**70**; +10% hysteresis = **80% effective**. Same LFM2 retune rationale as `vlm_min_decisions`) |
| `vlm_hysteresis_boost` | 10% | Extra agreement needed to change state |
| `vlm_start_threshold_cap` | 95% | Maximum effective start threshold (so hysteresis can't make it unreachable) |
| `vlm_min_state_duration` | 8s | Cooldown after VLM state change |
| `VLM_STOP_THRESHOLD` | 2 | Consecutive no-ad votes for fast-stop path |

**Transition Frame Detection:**
When blocking is active, black/solid-color frames are detected as transitions between ads and held in blocking state to prevent premature unblocking and re-blocking flicker. The `_is_transition_frame()` method analyzes:
- Mean brightness < 30 with low std deviation → black screen
- Low std deviation across frame → solid color
- >95% pixels within 20 values of median → uniform/static

**Transition-hold time cap (`_transition_hold_active`, `TRANSITION_HOLD_MAX_SECONDS`=3s, env `MINUS_TRANSITION_HOLD_MAX`):** the hold is only meant to bridge a *brief* (≤~2s) black/solid gap *between* ads. A dark/low-detail lofi music video (very common in autonomous YouTube, e.g. "WYS | Comforting You") reads as "uniform" *indefinitely*, so an uncapped hold froze `ocr_no_ad_count` / `vlm_no_ad_count` and a block never recovered — observed: a 46.9s VLM-source block held ~10s of benign content after the ad ended while VLM was firmly NO-AD. The hold is now capped: after `TRANSITION_HOLD_MAX_SECONDS` of *continuous* transition frames the no-ad counters resume so the block stops. Shared by the OCR and VLM loops; the timer resets on any non-transition or ad frame (so each real inter-ad gap gets a fresh full window). A true >3s black gap (rare) at worst causes a brief flicker — far better than multi-second false holds on uniform content. See *iter4 query_image p128 Overflow + Production FP / Slow-Recovery Fixes* under Known Issues.

**Starting Blocking:**
1. OCR detects ad → blocking starts immediately (unless home screen detected)
2. VLM detects ad (no OCR) → needs 80%+ agreement in sliding window (4+ decisions)
3. VLM with recent OCR → trusted, triggers blocking
4. Home screen detection suppresses both OCR and VLM blocking on streaming interfaces

**Stopping Blocking:**
1. **If OCR triggered alone** (source=ocr): OCR says stop (`OCR_STOP_THRESHOLD=2` no-ads) → ends (~1s). VLM dissent must NOT stop early here (OCR is authoritative). **Triangulation veto override:** if VLM agrees no-ad (≥80%) AND ASR=veto AND OCR dwell < `OCR_TRUSTED_DWELL_FRAMES` AND block ≥4s old, force-stop early — catches OCR FPs on TV-show artifacts.
2. **If BOTH triggered** (source=both): stop on whichever clears first — `ocr_says_stop OR vlm_says_stop` (2 consecutive no-ad). Both detected the ad, so either clearing is a correct "ad ended" signal; this decouples recovery from slow OCR snapshot capture (~2.5s/frame headless) so recovery is ~1s instead of ~3s. The triangulation veto also applies here for the rare case where OCR+VLM briefly agreed on an artifact then both flipped. See *iter4 query_image p128 Overflow + Production FP / Slow-Recovery Fixes* under Known Issues.
3. **If VLM triggered alone** (source=vlm): VLM says stop (`VLM_STOP_THRESHOLD=2` no-ads) → ends (~1-2s); 90s VLM-only safeguard.
4. **Universal cap:** any source, `MAX_BLOCKING_DURATION` (150s, env `MINUS_MAX_BLOCKING_DURATION`) force-stops and clears all detection state — bounds the worst-case static-weak-keyword false positive. **Frozen-stream guard:** when the cap fires it sets `_safeguard_freeze_active` + snapshots the frozen OCR text (`_safeguard_freeze_text`), which suppresses *re-blocking* until the OCR text **meaningfully changes** (difflib ratio <0.7 vs the snapshot). This handles the upstream stream freezing on an ad frame (observed: stuck on "Sponsored…31 Skip in", countdown frozen, OCR byte-identical for 150s → it's a genuine ad frame so OCR+VLM correctly keep flagging it and `skip in` correctly disables static-suppression, but capping then immediately re-blocking the *same frozen frame* produced a 150s→150s churn). **Note:** the first implementation cleared on pixel `is_scene_changed()` and failed — a frozen stream still pixel-jitters (buffering spinner / compression noise) so scene-change tripped ~1s after the cap and the churn continued; the reliable "stream resumed" signal is the OCR *text* changing. A real long ad pod is unaffected (its text changes, clearing within a cycle); a stuck source keeps ~identical text → stays suppressed so autonomous mode can recover it. **Early frozen-stream detection (`FROZEN_EARLY_SECONDS`=30s, env `MINUS_FROZEN_EARLY_SECONDS`):** the 150s cap bounds a freeze but a single ~150s hold still violates the zero-multi-minute-holds goal and recurred ~daily. The OCR loop tracks text stability (`_ocr_text_frozen_for`: seconds the normalised OCR text has been unchanged, difflib >0.93, non-empty only); when a block has been active with frozen OCR text for ≥30s, the SAME proven force-stop+freeze path fires early (only the trigger time is new — reuses `_norm_alnum`/difflib/`_safeguard_freeze_*`). A real skippable ad's "Skip in N" countdown decrements every ≤3s so its text never stays identical 30s; bumpers end well before 30s — so real ads don't trip it. Worst-case rare ≥30s fully-static-text no-countdown ad unblocks at 30s (vs 150s), an acceptable trade for eliminating the multi-minute hold.
5. VLM history cleared on stop → prevents immediate re-trigger
6. VLM stop uses simple consecutive count, NOT sliding window (for responsiveness)

**Why This Design:**
- VLM sliding window prevents erratic false-positive blocking when acting alone
- OCR is authoritative for stopping OCR-triggered blocks (fast unblock)
- VLM-triggered blocks require VLM to confirm ad ended (since OCR never saw it)
- Clearing VLM history on stop prevents "waffle memory" from causing re-triggers
- VLM stopping uses simple consecutive count (not sliding window) for responsiveness

**Anti-flicker:**
- Minimum blocking duration starts at 3.0s (`MIN_BLOCKING_DURATION_BASE`) and falls off by `MIN_BLOCKING_DURATION_STEP` (0.5s) on each consecutive ad: 3.0 → 2.5 → 2.0 → 1.5 → 1.0s. Floor is 1.0s for OCR-only, 1.5s for OCR+VLM both agreeing, and **0.5s for VLM-only** (`MIN_BLOCKING_DURATION_FLOOR_VLM`, applied regardless of the falloff toggle) so the rare residual false VLM-only block clears the instant VLM flips to no-ad (~1-2s) instead of being held the 3.0s base. Counter resets after `MIN_DURATION_RESET_GAP` (30s) without a block. Toggleable via Settings → Blocking Optimizations → *Block-duration Falloff*.
- VLM history cleared on stop prevents false re-triggers
- Transition frame detection holds blocking through black screens between ads
- After TV reconnect, ad blocking is suppressed for `HDMI_RECONNECT_GRACE_SECONDS` (90s) so the user can navigate without overlays jumping in. The health monitor calls `Minus.notify_hdmi_reconnect()` when it sees the HDMI-TX link return. Toggleable via Settings → Blocking Optimizations → *HDMI Reconnect Grace*.

**Static Screen Suppression:**
- Prevents blocking on paused video screens (Netflix/YouTube show ads when paused)
- After 2.5s of static screen (`STATIC_TIME_THRESHOLD`), blocking is suppressed
- When video resumes, 0.5s cooldown (`DYNAMIC_COOLDOWN`) before re-enabling blocking
- Detection state (OCR/VLM) cleared on cooldown complete to prevent false positives
- Static ad screenshots saved to `screenshots/static/` for analysis
- **Strong-ad-signal override (`STRONG_AD_KEYWORD_NAMES`):** suppression refuses to activate, and force-clears mid-suppression, when OCR has matched a strong keyword within the last `STRONG_AD_HOLD_SECONDS` (5s). Strong keywords: `skip ad` / `skip ads` / `skip in` / `skip ad (fuzzy*)` / `video will play after ad` / `visit advertiser` / `visitadvertiser` / `ad X of Y` / `ad countdown` / `ad with timestamp` / `ad with timestamp (cross-element)` / **`sponsored`** (promoted to STRONG 2026-05 per product decision — see *Sponsored promoted to strong* under Known Issues; bounded against static-promo holds by FROZEN_EARLY/MAX caps + home-screen detection). `Learn more` / `Shop now` / `Buy now` stay weak — those legitimately appear on home screens / paused-on-ad tiles.
- **Weak-keyword-only OCR suppression (detection layer, not just static):** if *every* matched keyword is in `WEAK_AD_KEYWORD_NAMES` (`learn more`, `shop now` (+fuzzy), `buy now` — `sponsored` was removed when promoted to STRONG) and no `STRONG_AD_KEYWORD` was seen within `STRONG_AD_HOLD_SECONDS`, the frame is suppressed AND routed into no-ad accounting so it neither starts nor sustains a block, and an active block decays. Replaced the old `_hdmi_audio_present()` discriminator (home/promo screens carry audio → it failed, holding a 591s block on a static "Sponsored · Peel to collect" promo). Originally a bare-`'sponsored'`-only check; **generalised to the full weak set** after a 150s VLM+OCR hold on a static "Learn more · Sponsored" promo — the keyword *pair* evaded the sponsored-only test (each weak keyword alone is suppressed but two together wasn't). Real video ads always also surface a strong keyword within the hold window, and VLM independently catches genuine ad video, so OCR can stay strict. See *iter4 query_image p128 Overflow + Production FP / Slow-Recovery Fixes* under Known Issues.

**OCR Timestamp Pattern Handling:**
OCR frequently misreads characters in ad timestamps. The detection handles these common confusions:

| Intended | OCR Misreads | Example |
|----------|--------------|---------|
| `0` (zero) | `o`, `O` | "Ad0:30" → "Ado:30", "AdO:30" |
| `1` (one) | `l`, `L`, `I`, `i` | "Ad1:30" → "Adl:30", "AdI:30" |
| `:` (colon) | `;`, `.` | "Ad0:30" → "Ad0;30", "Ad0.30" |

Combined misreads are also handled (e.g., "Adl;lo" for "Ad1:10"). The timestamp pattern matches:
- Standard: `Ad 0:30`, `Ad0:30`, `Ad1:45`
- Zero misreads: `Ado:30`, `Ad0:3o`, `Ado:oo`, `Ado:o5` (zeros misread on both sides of the colon)
- One misreads: `Adl:30`, `Ad1:l5`, `Adl:lo`
- Separator misreads: `Ad0;30`, `Ad0.30`, `Ado;3o`

The pattern lives in **two places** that must stay in sync: `src/ocr.py:595` (PaddleOCR class) and `src/ocr_worker.py:404` (OCRProcess, which is what production actually calls — `self.ocr = OCRProcess()` in `minus.py:563`). Each side carries a `Mirrors src/ocr.py:NNN — keep in sync` / vice-versa comment. The deeper fix is to delete the duplicate in `ocr_worker.py` and have it call `PaddleOCR.check_ad_keywords` directly; until then, any change to one file's pattern must be mirrored to the other. See *OCR Worker Keyword-Pattern Drift* under Known Issues for the past failure mode.

**Ad-keyword policy:**
- Bare `Ad` / `Ads` at a word boundary triggers blocking. Past false positives from words like `Loading`, `reading`, `Adobe` are handled via the word-boundary regex (`\bad\b` / `\bads\b`) and the `AD_EXCLUSIONS` list — bare `Ad` inside a longer word will not match.
- `Visit advertiser` (YouTube pre-roll CTA) is treated as an exact ad keyword.

**Fuzzy "Skip Intro" exclusion:**
Streaming UIs render a `Skip Intro` button that OCR sometimes reads as `Sk1p Intro`, `Skip 1ntro`, `Sk1p 1ntro`, `Sk1p1ntro`, etc. (`i` ↔ `1` ↔ `l` ↔ `I` swaps). A compiled regex `s[kK][i1lI]p\s*[i1lI]ntro` (in `src/ocr.py` as `SKIP_INTRO_FUZZY_RE` and mirrored in `src/ocr_worker.py`) covers all permutations. It's applied as part of the exclusion gate at the top of the per-text and cross-element matching paths, before either exact-keyword or word-boundary detection runs — important because `skip in` (inside `AD_KEYWORDS_EXACT`) is a substring of `skip intro` and would otherwise match first.

`Skip Ad` is **not** excluded — it still triggers ad detection (via the `skip ad` exact keyword) and is independently recognized as a skip button by `src/skip_detection.py`, so Minus will press it to dismiss the ad.

## Blocking Overlay

When ads are detected, the screen shows a full blocking overlay **rendered at 60fps via ustreamer's native MPP blocking mode**:
- **Pixelated Background**: Blurred/pixelated version of the screen from ~6 seconds before the ad
- **Header** (debug only): `[ BLOCKING // OCR ]`, `[ BLOCKING // VLM ]`, or `[ BLOCKING // OCR+VLM ]`
- **Spanish vocabulary**: Random intermediate-level word with translation
- **Example sentence**: Shows the word in context
- **Rotation**: New vocabulary every 11-15 seconds
- **Ad Preview Window**: Live preview of the blocked ad in bottom-right corner (60fps!)
- **Debug stats** (debug only): bottom-left dashboard with uptime, blocks, time saved, ad countdown bar, audio level
- **OCR trigger snippet** (debug only): top-right `(Ad) 0:30 left` style — the OCR text that fired the block, with the matched keyword wrapped in parens. Empty for VLM-only blocks. Capped at 50 chars.

**Multi-color Text Per Line:**
- **Purple** - Spanish word (IBM Plex Mono Bold font)
- **White** - Header and translation (DejaVu Sans Bold font)
- **Gray** - Pronunciation and example sentence (DejaVu Sans Bold font)

**Font Configuration:**
- `FONT_PATH_VOCAB_PRIMARY` = DejaVu Sans Bold (vocabulary text, centered)
- `FONT_PATH_WORD_PRIMARY` = IBM Plex Mono Bold (Spanish word, purple)
- `FONT_PATH_STATS_PRIMARY` = IBM Plex Mono Regular (debug stats, monospace)

**Rendering Pipeline:**
All overlay rendering is done inside ustreamer's MPP encoder, NOT GStreamer:
1. `ad_blocker.py` captures pre-ad frame and creates pixelated NV12 background
2. Background uploaded via `POST /blocking/background` (async, non-blocking)
3. Text and preview configured via `GET /blocking/set`
4. FreeType renders TrueType fonts directly to NV12 planes at encoder resolution
5. Composite runs at 60fps with ~0.5ms overhead per frame

**Pixelated Background:**
Instead of a plain black background, the blocking overlay shows a heavily pixelated (20x downscale) and darkened (60% brightness) version of what was on screen before the ad appeared. This provides visual context while clearly indicating blocking is active.

Implementation (`src/ad_blocker.py`):
- Rolling 6-second snapshot buffer (3 frames at 2-second intervals)
- Uses oldest frame when blocking starts (ensures pre-ad content)
- OpenCV pixelation: downscale by 20x, upscale with INTER_NEAREST
- Converted to NV12 and uploaded via `/blocking/background` POST API
- Upload runs in background thread for non-blocking operation

**Preview Window:**
Unlike the old GStreamer approach (limited to ~4fps), the ustreamer blocking mode provides:
- Full 60fps live preview of the blocked ad
- Hardware-accelerated scaling in the MPP encoder
- Automatic resolution handling (works at 1080p, 2K, 4K)

**Web UI Toggles:** *Ad Preview Window* and *Debug* toggleable via Settings (both default ON). The unified *Debug* toggle controls all three on-screen debug elements together — header, bottom-left stats dashboard, and top-right OCR trigger snippet — and is persisted to `~/.minus_system_settings.json` (`debug_overlay`) so off survives a service restart.

**Recursion safety for the OCR snippet:** OCR consumes `/snapshot/raw` (`src/capture.py:134`), which the patched ustreamer serves from `us_blocking_store_raw_frame()` *before* the blocking composite is applied. The new top-right text — and every other element on the blocking overlay — is therefore invisible to OCR, so the displayed `(Ad) 0:30 left` cannot loop back into detection. Don't break this: if you ever route OCR through `/snapshot` (the composited path), all of these debug texts will become self-triggering.

## Spanish Vocabulary

120+ intermediate-level words and phrases including:
- **Common verbs**: aprovechar, lograr, desarrollar, destacar, enfrentar...
- **Reflexive verbs**: comprometerse, enterarse, arrepentirse, darse cuenta...
- **Adjectives**: disponible, imprescindible, agotado, capaz, dispuesto...
- **Nouns**: desarrollo, comportamiento, conocimiento, ambiente, herramienta...
- **Expressions**: sin embargo, a pesar de, de repente, hoy en dia, cada vez mas...
- **False friends**: embarazada, exito, sensible, libreria, asistir...
- **Subjunctive triggers**: es importante que, espero que, dudo que, ojala...
- **Time expressions**: hace poco, dentro de poco, a la larga, de antemano...

## Housekeeping

**Log File:**
- Location: `/tmp/minus.log`
- Max 5MB per log file
- Keeps 3 backup files (minus.log.1, .2, .3)

**Screenshot Truncation:**
- Keeps only last 50 screenshots by default
- Configurable via `--max-screenshots`

## VLM Model

**LFM2.5-VL-450M-ft-v2-fused-v2** on Axera LLM 8850 NPU (fused-layer prefill, no decode):
- `detect_ad()` is **prefill-only**: vision encoder (~185ms) → 16 fused
  decoder layers (~185ms) → post (LM head) → last-token vocab logits.
  Decision is `argmax of max(YES_logits) vs max(NO_logits)` over the 4
  spelling variants each (`Yes`/`yes`/` Yes`/` yes` etc.). Calibrated on
  an 800-image holdout: **97.0% accuracy, 94.8% ad-recall, 99.2% non-ad-recall**.
  Also exposes a softmax-normalized `p_yes_norm` over the {YES,NO}
  subspace for an optional tunable threshold via env
  `MINUS_VLM_AD_THRESHOLD` (default `0.5` ≡ argmax; gain from tuning is
  small since argmax is already at 97%).
- `query_image()` is **also prefill-only** — autonomous-mode screen
  classification (`PLAYING / PAUSED / DIALOG / MENU / SCREENSAVER`)
  uses the same prefill loop and looks up the first-token logit for
  each class (max over no-leading-space and leading-space spellings),
  returns the class with the highest logit. The `max_new_tokens`
  parameter is kept for API compatibility but **ignored** — there is
  no decode loop and per-layer decode axmodels are not shipped with
  this v2 build (only `post_d.axmodel`). Latency ~370ms, same as
  `detect_ad`.
- **~0.37s** deterministic inference (vision ~185ms + prefill ~185ms;
  the descriptive-paragraph latency pathology that plagued the 1.5B
  is structurally impossible — both inference paths are fixed-length
  prefill, no decode).
- **~9-11s** model load: 17 axengine sessions (1 vision + 16 fused
  layers + 1 post) + 256MB embed.npy mmap + tokenizer.json. Faster
  than FastVLM iter4's ~14s.
- Uses Python axengine + `PreTrainedTokenizerFast` (no transformers
  AutoTokenizer / CLIPImageProcessor / ml_dtypes — those FastVLM
  dependencies are gone).
- Vision preprocessing: direct bilinear resize to 512×512, normalize
  `(x/255 - 0.5)/0.5`, patchify into `(1, 1024, 768)` for the vision
  encoder. DO NOT substitute FastVLM's `expand2square` + CLIPImageProcessor —
  the fine-tune was on this exact preprocessing.
- The classification prompt MUST stay byte-exact (LFM chat template
  with BOS + IM_START + system + IM_END):
  `"Is this an advertisement? Answer Yes or No."` with system
  `"You are a helpful multimodal assistant by Liquid AI."`. Likewise
  for the screen prompt — see the comment block on `SCREEN_QUERY_PROMPT`
  in `src/autonomous_mode.py`.
- `confidence` returned by `detect_ad` is `p_yes_norm` if ad else
  `1 - p_yes_norm`, feeding the existing confidence-weighted sliding
  window directly.

**Process-based architecture (`src/vlm_worker.py`):**
- VLM runs in a separate process for hard timeout capability
- Uses 'spawn' multiprocessing method to avoid "can only join a child process" errors from axengine
- **Soft/Hard timeout strategy** to avoid unnecessary restarts:
  - Soft timeout (1.5s): Returns immediately with "TIMEOUT", but worker keeps running
  - Hard timeout (2.0s): Only kills worker if inference is truly stuck
  - Restart threshold: 3 consecutive soft timeouts trigger a hard kill
  - Late responses are drained on next request and counters reset
- 4 warmup inferences at startup with varied content (noise, gradients, edges, mixed)
- Keepalive thread runs dummy inference every 20s during idle to prevent NPU cold-start
- Worker process loads model once (~9-11s for LFM2), processes requests via Queue

**Two inference modes (both prefill-only on the same model):**
- `detect_ad(image_path)` → `(is_ad, response_text, elapsed, confidence)` — ad/not-ad classification via `argmax(max(YES_logits), max(NO_logits))` (with optional `MINUS_VLM_AD_THRESHOLD` on `p_yes_norm`). ~370ms deterministic.
- `query_image(image_path, prompt, max_new_tokens=8)` → `(response_text, elapsed)` — autonomous-mode screen classification. Same prefill, but the last-position logits are looked up at the FIRST-TOKEN of each of `PLAYING / PAUSED / DIALOG / MENU / SCREENSAVER` (max over no-leading-space and leading-space variants), argmax over the 5 classes returns the class name. `max_new_tokens` ignored. **Hard 320-token prefill window:** the full tokenised prompt (chat template + 256 image tokens + question) MUST stay ≤320 tokens — the axmodels were compiled for `PREFILL_LEN=320` and the prompt is right-padded; overflow silently truncates the `[IM_START] assistant\n` suffix → garbage last-position logits. `load_model()` fails loud if either cached prompt overflows; `query_image` returns `"PROMPT_TOO_LONG"` for on-the-fly oversized prompts. The current screen prompt tokenises to 312 — 8 tokens of headroom; do not lengthen `SCREEN_QUERY_PROMPT` without re-measuring.

Both modes share the same model. Concurrent callers (detection loop calling `detect_ad`, autonomous mode calling `query_image`) are serialized by `VLMProcess._call_lock` so they cannot cross responses on the shared queue or race on the timeout / latency state. See *VLMProcess Cross-Thread Race* under Known Issues for the full rationale.

minus-v0.1 on-disk layout (fine-tuned LFM2.5-VL-450M iter28, published at https://huggingface.co/TheGarageDev/Minus-v0.1; renamed from `LFM2/LFM2-450M-ft-iter28` in Aug 2026):
```
/home/radxa/axera_models/minus-v0.1/
├── vision_encoder_512.axmodel              # Vision encoder (input "pixel_values")
├── fused_models/
│   ├── l{0,1,3,4,6,7,9,11,13,15}_conv_fused.axmodel   # 10× conv layers
│   └── l{2,5,8,10,12,14}_attn_fused.axmodel           # 6× attn layers
├── decode_models/
│   └── post_d.axmodel                      # Post/LM-head (no per-layer decode models shipped)
├── embed.npy                               # Embedding table (FP32, mmap'd, 256MB)
└── tokenizer.json                          # PreTrainedTokenizerFast format
```

**Why LFM2.5-VL instead of FastVLM-0.5B iter4?** (benchmarks in `/home/radxa/axera_models/BENCHMARKS.md`)
| Aspect | LFM2.5-VL fused-v2 | FastVLM-0.5B iter4 |
|--------|--------------------|---------------------|
| Inference time | **~0.37s deterministic** | ~0.44s |
| Holdout accuracy / ad-rec / non-ad-rec | **97.0% / 94.8% / 99.2%** | 94.75% / 94.25% / 95.25% |
| Decoder architecture | 16 fused-layer axmodels (10 conv + 6 attn) | 24 separate p128 layers + KV cache |
| KV-cache management | **none — no decode loop** | required (max_seq_len=1023 hard constraint) |
| Custom utils dir / bfloat16 | **none** | `LlavaConfig`/`InferManager`/`expand2square` + ml_dtypes |
| Image tokens | 256 (16×16 grid) | 64 (8×8 grid) |
| `query_image` (autonomous mode) | **same model, prefill-only logit lookup** | decode-based, ~1.0s, p128-limited |
| Prefill window | 320 (axmodel-locked) | 128 (p128 single-chunk) |
| Parameters | 0.45B | 0.5B |

Non-ad-recall jumped from 95.25% → 99.2% — the iter4 home-screen false-positive class is essentially eliminated at the model level. `query_image` no longer needs decode infrastructure, so there is no longer a separate FastVLM model loaded for autonomous mode — one model serves both paths.

### Latency-based auto-recovery

The Axera NPU can drift into a degraded state (observed: ~15–18s inference with descriptive responses instead of the structured short answer) that outlasts simple worker restarts. This is **not thermal** — temps are similar (~70°C) when healthy and when slow. Most likely accumulated NPU memory or axengine context state.

`VLMProcess` keeps a rolling window of the last 10 successful inference latencies. After each success it computes P95 and triggers recovery if `P95 > 3.0s`:

| Step | Trigger | Action |
|------|---------|--------|
| 1 | P95 > 3.0s, no recovery in last 60s | `restart()` — kill worker + 2s NPU-release + start |
| 2 | Still degraded within 180s of step 1 | **Deep restart** — kill + 8s release + start |

60s cooldown prevents thrashing. Latency window and recoveries surface on `/api/health` at `subsystems.vlm.latency` and as Prometheus gauge `minus_axera_temperature_celsius` / `minus_axera_npu_usage_percent` / `minus_axera_cmm_used_kib`.

Query axcl directly for live telemetry:
```bash
axcl-smi info --temp   # milli-°C, divide by 1000
axcl-smi info --npu    # utilization %
axcl-smi info --cmm    # CMM memory used / total
```

## Dependencies

```bash
# System packages
sudo apt install -y imagemagick ffmpeg curl v4l-utils

# GStreamer and plugins for video pipeline
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-rockchip1 \
  gir1.2-gst-plugins-base-1.0 \
  libgstreamer1.0-dev

# Build ustreamer with MPP hardware encoding and FreeType fonts
sudo apt install -y librockchip-mpp-dev libfreetype-dev libjpeg-dev libevent-dev
git clone https://github.com/garagehq/ustreamer.git /home/radxa/ustreamer-garagehq
cd /home/radxa/ustreamer-garagehq && make WITH_MPP=1
cp ustreamer /home/radxa/ustreamer-patched

# Fonts for blocking overlay
sudo apt install -y fonts-dejavu-core fonts-ibm-plex

# Python dependencies
pip3 install --break-system-packages \
  pyclipper shapely numpy opencv-python \
  pexpect PyGObject flask requests androidtv \
  rknnlite  # RKNN NPU runtime for OCR (may need Rockchip's pip repo)
```

**Note:** The `rknnlite` package is provided by Rockchip and may need to be installed from their SDK or a custom repository. On the Radxa board with NPU support, it may already be pre-installed.

**Axera NPU (for VLM):**
The LFM2.5-VL-450M model runs on the Axera LLM 8850 NPU. Required Python packages:
```bash
pip3 install --break-system-packages axengine
```
The `axengine` package requires the Axera AXCL runtime to be installed - see the Axera documentation. LFM2.5-VL does **not** need `transformers` or `ml_dtypes` (it uses `PreTrainedTokenizerFast` direct from `tokenizer.json` and FP32 throughout) — those were FastVLM-era dependencies.

## Troubleshooting

**ustreamer fails to start:**
```bash
fuser -k /dev/video0  # Kill processes using device
pkill -9 ustreamer    # Kill orphaned ustreamer
```

**VLM not loading:**
- Check Axera card: `axcl_smi`
- Verify model files exist in `/home/radxa/axera_models/minus-v0.1/`
- Ensure Python dependencies: `pip3 show axengine`

**OCR not detecting:**
- Test snapshot: `curl http://localhost:9090/snapshot -o test.jpg`
- Check HDMI: `v4l2-ctl -d /dev/video0 --query-dv-timings`

**Display issues:**
- Check DRM plane: `modetest -M rockchip -p | grep -A5 "plane\[72\]"`
- Verify connector: `modetest -M rockchip -c | grep HDMI`

## CRITICAL: Blocking Mode Architecture

**NEVER REVERT TO GSTREAMER TEXTOVERLAY FOR BLOCKING OVERLAYS.**

The blocking overlay system uses ustreamer's native MPP blocking mode (`/blocking/*` API), NOT GStreamer's input-selector or textoverlay. This is a one-way migration - we only move forward.

**Current Architecture:**
- Simple GStreamer pipeline with `queue max-size-buffers=3 leaky=downstream` for smooth video
- All blocking compositing (background, preview, text) done in ustreamer's MPP encoder at 60fps
- Control via HTTP API: `/blocking/set`, `/blocking/background`
- FreeType TrueType font rendering:
  - **IBM Plex Mono Bold** for Spanish word (purple, centered)
  - **DejaVu Sans Bold** for vocabulary text (white/gray, centered)
  - **IBM Plex Mono Regular** for stats dashboard (bottom-left, monospace)
- Per-line multi-color text matching web UI aesthetic (see AESTHETICS.md)
- Thread-safe with mutex protection for 4 parallel MPP encoder workers

**Resolution Flexibility:**
The blocking system automatically handles resolution mismatches:
- API calls may specify 4K dimensions (3840x2160)
- With `--encode-scale passthrough`, encoder uses source resolution directly
- Preview dimensions are scaled proportionally to fit
- Positions are clamped to valid ranges
- All coordinates aligned to even values for NV12

**Thread Safety:**
FreeType is NOT thread-safe. With 4 parallel MPP encoder workers, a `pthread_mutex_t _ft_mutex` serializes all FreeType calls in the composite function to prevent crashes. Without this, concurrent FT_Set_Pixel_Sizes/FT_Load_Glyph calls corrupt FreeType's internal state.

**Why NOT GStreamer textoverlay:**
- Caused pipeline stalls every ~12 seconds
- NV12 format incompatibility issues
- 4K→1080p resolution mismatch problems
- gdkpixbufoverlay limited to ~4fps for preview updates
- Complex input-selector switching logic

**Key files:**
- `ustreamer-garagehq/src/libs/blocking.c` - NV12 compositing with FreeType, mutex protection
- `ustreamer-garagehq/src/libs/blocking.h` - Blocking mode API
- `src/ad_blocker.py` - Python client using blocking API

## ustreamer Overlay and Blocking API

**Notification Overlay** (for Fire TV setup messages, etc.):
- `GET /overlay` - Get current overlay configuration
- `GET /overlay/set?params` - Set overlay configuration

| Parameter | Description |
|-----------|-------------|
| `text` | Text to display (URL-encoded, supports newlines) |
| `enabled` | `true` or `1` to enable overlay |
| `position` | 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right, 4=center |
| `scale` | Text scale factor (1-10) |
| `color_y`, `color_u`, `color_v` | Text color in YUV |
| `bg_enabled` | Enable background box |
| `bg_alpha` | Background transparency (0-255) |
| `clear` | Clear overlay |

**Example:**
```bash
curl "http://localhost:9090/overlay/set?text=LIVE&position=1&scale=3&enabled=true"
curl "http://localhost:9090/overlay/set?clear=true"
```

**Blocking Mode** (for ad blocking overlays):

**Blocking Mode Endpoints:**
- `GET /blocking` - Get current config (enabled, preview, colors, etc.)
- `GET /blocking/set?enabled=true&text_vocab=...&text_ocr=...&preview_enabled=true&preview_grayscale=true&word_y=140&word_u=175&word_v=145` - Configure. Includes `preview_grayscale` to desaturate the corner preview, `word_y/word_u/word_v` for cycling the Spanish word color per rotation, and `text_ocr` for the top-right OCR-trigger snippet (renders in IBM Plex Mono Regular at the same scale as `text_stats`; empty string clears it).
- `POST /blocking/background` - Upload pixelated NV12 background (width*height*1.5 bytes)

**Multi-color text auto-detection:** Lines starting with `[` → white (header), `(` → gray (pronunciation), `=` → white (translation), `"` → gray (example), other → purple (Spanish word)

## Overlay Priority System

The overlay system includes a priority mechanism to handle multiple overlays gracefully:

**Persistent Overlays:**
- Setup instructions (Roku Limited Mode, Fire TV ADB Enable) are "persistent"
- Registered with duration > 60 seconds
- Have a background monitor thread that checks every 5 seconds
- Auto-restore if overwritten by short notifications (VLM status, etc.)

**Short Overlays:**
- Status notifications (VLM Ready, Connected, etc.) are short-lived (5-10s)
- Can temporarily interrupt persistent overlays
- After they expire, the persistent overlay is automatically restored

**State Changes:**
- Successful device connection calls `_clear_persistent()` to dismiss setup instructions
- This prevents stale setup overlays from reappearing after connection

**Implementation:**
- Module-level singleton state in `src/overlay.py` (`_overlay_state` dict)
- Monitor thread spawned by `_set_persistent()` polls ustreamer overlay API
- Compares current overlay text to expected text, restores if different

## Health Monitoring

The health monitor (`src/health.py`) runs in a background thread and checks:

| Subsystem | Check | Recovery |
|-----------|-------|----------|
| HDMI signal | v4l2-ctl --query-dv-timings | Show "NO SIGNAL" overlay, mute audio |
| No HDMI at startup | check_hdmi_signal() | Show bouncing "NO SIGNAL" screensaver |
| ustreamer | HTTP HEAD to :9090/snapshot | Restart ustreamer + video pipeline |
| Video pipeline | Buffer flow + FPS monitoring | Restart pipeline with exponential backoff |
| Output FPS | GStreamer pad probe | Log warning if < 25fps |
| VLM | Consecutive timeouts < 5 | Degrade to OCR-only, retry VLM after 30s |
| Memory | Usage < 90% | Force GC, clear frame buffers |
| Disk | Free > 500MB | Log warning |

**HDMI Disconnect/Reconnect Recovery:**
- Detects HDMI signal loss via ustreamer's `/state` API (`captured_fps` field)
- Signal considered lost if FPS is 0 for more than 5 seconds (handles source going to sleep)
- Shows "NO SIGNAL" overlay and mutes audio immediately
- On signal restoration: restarts ustreamer → restarts video pipeline → restores display
- Full recovery typically completes in ~7 seconds

**Display Output Resilience (HDMI-TX Disconnected):**
- Service continues running even if HDMI-TX display output is disconnected
- ustreamer runs independently for web preview and ML detection
- Web UI shows "DISPLAY DISCONNECTED" overlay with grayscale video feed underneath
- Display retry loop attempts reconnection every 7 seconds (only display pipeline, not ustreamer)
- OCR/VLM ad detection continues working without display output
- API exposes `display_connected` and `display_error` fields in `/api/status`

**Video Pipeline Watchdog:**
- Buffer watchdog detects stalls (10 seconds without buffer)
- Monitors GStreamer pipeline state (must be PLAYING)
- Handles HTTP connection errors from souphttpsrc
- Handles unexpected EOS (end-of-stream) events
- Exponential backoff for restarts (1s → 2s → 4s → ... → 30s max)
- Backoff resets after 10 seconds of sustained buffer flow

**Startup grace period:**
- 30-second grace period before ustreamer health checks begin
- Prevents false positives during VLM model loading

**Graceful degradation (startup):**
- OCR initialization: 3 retries with 2s delay, continues without OCR if all fail
- VLM model loading: 3 retries with 5s delay, continues without VLM if all fail
- Both failures are non-fatal — the system runs with whatever subsystems loaded
- `ocr_ready` and `ocr_disabled` fields in `/api/status` (matching existing `vlm_ready`/`vlm_disabled`)
- OCR status badge in web UI header: `OCR: Ready / Disabled / Failed`

**Graceful degradation (runtime):**
- If VLM fails 5+ times consecutively, switches to OCR-only mode
- VLM restart is attempted after 30 seconds in background
- OCR continues working independently

**Scene skip cap:**
- OCR: Force run after 30 consecutive skips
- VLM: Force run after 10 consecutive skips
- Prevents missing ads that appear without scene change

**Periodic logging:**
- FPS logged every 60 seconds
- Full status logged every 5 minutes (uptime, fps, hdmi, video, audio, vlm, mem, disk)

## Web UI

Minus includes a lightweight Flask-based web UI for remote monitoring and control, accessible via Tailscale from desktop or mobile devices.

**Features:**
- **Live video feed** - MJPEG stream proxied from ustreamer (CORS bypass)
- **Status display** - Blocking state, FPS, HDMI info, uptime. Footer shows the block source incl. ASR (`OCR`/`VLM`/`OCR+VLM`/`OCR+ASR`/`VLM+ASR`/`OCR+VLM+ASR`).
- **Pause controls** - 1/2/5/10 minute presets + custom input, **up to 600 minutes** (`/api/pause/<minutes>` validates 1-600; was 60).
- **Detection history** - Recent OCR/VLM detections with timestamps
- **ASR Live panel** (Home) - collapsible dropdown: live transcript, verdict (confirm/veto/unknown), marker hits, latency, engine; on/off toggle + Test (ad)/Test (show) buttons (`/api/asr/test`).
- **OCR Live panel** (Home) - collapsible dropdown mirroring ASR Live: live OCR text flowing, matched ad keywords, frame count.
- **Settings** - Toggle preview window, debug dashboard, **ASR enable/disable** (Detection Settings, persisted `asr_enabled`)
- **Log viewer** - Collapsible log output for debugging

**Key API Routes:**
- `GET /`, `/api/status`, `/api/detections`, `/api/logs`
- `POST /api/pause/N`, `/api/resume` — pause/resume all blocking. **Special case for VLM-only blocks**: a pause issued while `blocking_source == "vlm"` is treated as a VLM-misclassification signal (the user is saying "this isn't an ad") and triggers two additional actions: (a) the VLM-triggering frame (the last `is_ad=True` frame cached by the dispatch loop) is saved to `screenshots/non_ads/` as training data, (b) VLM inference is paused for **5 min** (`VLM_FALSE_POSITIVE_COOLDOWN`) independently of the general pause timer — so even if the user clicks Resume early, VLM stays cold until the cooldown expires. OCR keeps running. `/api/status` exposes `vlm_user_paused` + `vlm_user_pause_remaining`. Scope is strict: `"ocr"` and `"both"` blocks fall through to the standard "save current frame, no VLM cooldown" path.
- `GET/POST /api/preview/*`, `/api/debug-overlay/*` (the debug-overlay route is the unified *Debug* toggle: header + bottom-left stats + top-right OCR snippet, persisted to `~/.minus_system_settings.json` as `debug_overlay`)
- `POST /api/test/trigger-block`, `/api/test/stop-block`
- `GET /stream`, `/snapshot` - Proxy to ustreamer
- `GET /api/health` - Health check for uptime monitors
- `POST /api/video/restart` - Force restart video pipeline
- `GET/POST /api/video/color` - Get/set color settings (saturation, brightness, contrast, hue)
- `POST /api/ocr/test` - Run OCR on current frame (no screenshot save)
- `POST /api/vlm/test` - Run VLM on current frame (no screenshot save)
- `GET /api/vlm/status` - Get VLM status (disabled, model_loaded, etc.)
- `POST /api/vlm/disable` - Disable VLM and unload model from NPU
- `POST /api/vlm/enable` - Re-enable VLM and load model
- `POST /api/blocking/skip` - Trigger Fire TV skip button
- `POST /api/audio/sync-reset` - Reset A/V sync (~300ms dropout)
- `GET /api/autonomous` - Autonomous mode status
- `POST /api/autonomous/enable` / `disable` / `toggle` / `start` - Control autonomous mode
- `POST /api/autonomous/schedule` - Set schedule (start_hour, end_hour, always_on)
- `GET /api/autonomous/logs` - Autonomous mode log entries
- `GET /api/screenshots/review/<category>` - Unreviewed screenshots for swipe classification
- `POST /api/screenshots/approve` - Mark screenshot as correctly labeled
- `POST /api/screenshots/classify` - Move screenshot between categories
- `POST /api/screenshots/undo` - Undo last review action
- `GET /api/ir/status` - IR transmitter status (`enabled`, `available`, `initialized`, `codes`)
- `POST /api/ir/enable` / `disable` - Toggle the IR remote feature (gates the UI and `/command`)
- `POST /api/ir/command` - Send a captured button. Body: `{"button": "power"|"input_1"|"input_2"|"input_3"|"next"|"auto"}`. `403` when disabled, `429` with `retry_after` inside the 1.5 s cooldown. See `docs/IR_TRANSMITTER.md`.
- `GET /api/leds/status` - Status LEDs status (`available`, `enabled`, `running`, `state`, `states`, `last_error`, `gated`)
- `POST /api/leds/enable` / `disable` - Toggle the WS2812B status strip; persists; starts/stops the animation thread
- `POST /api/leds/state` - Switch animation state. Body: `{"state": "<name>"}`. `403` when disabled, `400` for unknown state. States: `off / initializing / idle / blocking / paused / no_signal / autonomous / wifi_setup / error`. See `docs/STATUS_LEDS.md`.
- `GET /api/leds/require_display` - Display-gate status (`leds_require_display`, live `display_connected`)
- `POST /api/leds/require_display` - Body `{"enabled": true|false}` — when on (default), the strip stays dark while the HDMI-TX display is disconnected or powered off.

**Test API Endpoints:**
For development and testing ad blocking without waiting for real ads:
```bash
# Trigger blocking for 20 seconds (max 60)
curl -X POST -H "Content-Type: application/json" \
  -d '{"duration": 20, "source": "ocr"}' \
  http://localhost:80/api/test/trigger-block

# Stop blocking immediately
curl -X POST http://localhost:80/api/test/stop-block
```

Parameters for trigger-block:
- `duration`: seconds to block (default: 10, max: 60)
- `source`: detection source - 'ocr', 'vlm', 'both', or 'default'
- `kind`: optional forced replacement kind - 'vocab', 'fact', or 'photos'

Test mode prevents the detection loop from canceling the blocking, allowing full testing of pixelated background, animations, and audio muting. When `source` is `ocr` or `both`, the endpoint also injects a synthetic `(Ad) 0:30 left` snippet into the top-right OCR-trigger slot so you can exercise that rendering path without waiting for real OCR.

**Access URLs:**
- Local: `http://localhost:80`
- Tailscale: `http://<tailscale-hostname>:80`
- Direct stream: `http://<hostname>:9090/stream`

**Security:**
- No authentication (relies on Tailscale network security)
- Read-mostly API with minimal attack surface
- Binds to 0.0.0.0 for remote access

## VLM Training Data Collection

Minus automatically collects training data for future VLM improvements, organized by type:

**Screenshot directories:**
- `screenshots/ads/` - OCR-detected ads
- `screenshots/non_ads/` - User paused = false positives. **VLM-only blocks** save the *VLM-triggering frame* (the last `is_ad=True` frame cached by the dispatch loop), not the current frame at pause time — see *User-Pause Feedback Loop for VLM Misclassifications* below. OCR-only and OCR+VLM-both blocks save the current frame (unchanged behaviour).
- `screenshots/vlm_spastic/` - VLM uncertainty cases (detected 2-5x then changed)
- `screenshots/static/` - Static screen suppression

**User-Pause Feedback Loop for VLM Misclassifications:**

When the user pauses ad blocking *during a VLM-only block* (`blocking_source == "vlm"`), Minus reads that as an explicit "this isn't actually an ad" signal and reacts with two coordinated actions:

1. **Save the trigger frame as training data.** The VLM dispatch loop caches the most recent frame that produced an `is_ad=True` verdict (`last_vlm_ad_frame`, cleared when a block ends). On pause-during-VLM-block, that cached frame goes to `screenshots/non_ads/`. The current frame is *not* used because by the time the user clicks pause the underlying video may have advanced past the misclassified moment — the trigger frame is what VLM actually got wrong.
2. **5-min VLM cooldown** (`VLM_FALSE_POSITIVE_COOLDOWN = 300s`). The VLM dispatch loop skips inference entirely while paused, so the same misclassified content can't immediately re-trigger if the user clicks Resume before the cooldown expires. OCR keeps running normally — the cooldown is VLM-only. The general blocking-paused timer is independent of this and uses the duration the user picked (1/2/5/10 min).

Scope is strict: only `blocking_source == "vlm"` triggers the feedback path. `"ocr"` (OCR detected alone) and `"both"` (OCR + VLM agreed) fall through to the existing "save current frame, no VLM cooldown" behaviour — those aren't VLM-alone misclassifications. Window is the entire block: works at any point from start to end of a VLM-only block.

Falls back to the current frame if no VLM-AD frame was cached (defensive — would require a block to fire without any `is_ad=True` verdict in the dispatch loop, which shouldn't happen in practice but is logged at WARN if it does).

`/api/status` exposes `vlm_user_paused` and `vlm_user_pause_remaining` so the UI can surface the cooldown state. See `tests/test_recent_features.py::TestVLMFalsePositiveFeedback` for the 9 unit tests covering trigger-frame save, cooldown duration, VLM state clearing, source-scope guards, accessor edge cases, and the missing-frame fallback.

**Screenshot Quality Filtering (all categories):**

Every save goes through `_should_save()` which applies three layers of filtering:

| Layer | What it catches | Threshold |
|-------|----------------|-----------|
| Rate limiting | Rapid-fire saves | 5s minimum between saves per category |
| Blank rejection | Black/solid-color frames | Mean brightness < 15 or std dev < 10 |
| dHash dedup | Near-duplicate frames | Hamming distance < 10 bits (~85% similar) |

**dHash (Difference Hash):**
- Resize frame to 9x8 grayscale, compare adjacent pixels → 64-bit hash
- Two frames of the same ad with slightly different timestamps: hamming distance ~1-5
- A genuinely different scene: hamming distance ~20-30
- Keeps last 200 hashes per category for rolling dedup window

**Screenshot Review System (Tinder-style):**

The web UI includes a swipe-based review system for classifying screenshots:
- Each screenshot tab (Ads, Non-Ads, VLM Spastic, Static) has a 👀 review button
- Opens a full-screen modal with a 3-card visual stack
- **Swipe right** (or arrow key) = approve / classify as ad
- **Swipe left** (or arrow key) = reclassify / classify as not ad
- **Undo** (Ctrl+Z or button) reverses the last action
- Progress tracked in `/home/radxa/.minus_reviewed_screenshots.json` — shows oldest unreviewed first

| Category | Swipe Right | Swipe Left |
|----------|-------------|------------|
| Ads | Approved (correct) | Move to Non-Ads |
| Non-Ads | Approved (correct) | Move to Ads |
| VLM Spastic | Move to Ads | Move to Non-Ads |
| Static | Move to Ads | Move to Non-Ads |

**Review API:**
- `GET /api/screenshots/review/<category>` - Unreviewed items, oldest first
- `POST /api/screenshots/approve` - Mark as correctly labeled
- `POST /api/screenshots/classify` - Move between categories
- `POST /api/screenshots/undo` - Undo last action

## Autonomous Mode

Autonomous Mode keeps YouTube playing on streaming devices during scheduled hours so Minus can collect ad detection training data unattended. Device-agnostic design supports Fire TV, Roku, and Google TV. Uses VLM to understand screen state and take intelligent actions.

**How it works:**
1. **Schedule** — Configurable start/end hours (e.g., 22:00–06:00), or 24/7 mode
2. **OCR-based screen detection** — Before VLM, checks OCR text for login/home screen keywords (VLM often misclassifies these static screens as "PLAYING")
3. **VLM-guided keepalive** — Every 2 minutes, captures a frame and asks VLM to classify the screen state
4. **Roku ECP active app check** — Before VLM, queries Roku's `/query/active-app` API to detect if YouTube exited or screensaver activated (more reliable than VLM for Roku)
5. **Frame-change + audio verification** — After VLM says "PLAYING", verifies with dHash frame comparison + audio flow check to catch paused videos VLM misclassifies
6. **Smart actions** — Based on combined signals, takes the minimum necessary action:

| Signal | Action | Command |
|--------|--------|---------|
| OCR: login screen keywords | Select account | `down` + `select` |
| OCR: home screen keywords + static | Select video | `down` + `select` |
| VLM: PLAYING + frames changing | None | Video is fine |
| VLM: PLAYING + static + no audio | Play | `play_pause` (paused video VLM missed) |
| VLM: PLAYING + static + audio flowing | None | Music stream with static image (lo-fi) |
| VLM: PAUSED + **audio flowing** | **None (audio veto)** | VLM misclassified — see *Autonomous Mode VLM-Misclassification Traps* in Known Issues |
| VLM: PAUSED + no audio | Play | `play_pause` key |
| VLM: DIALOG | Dismiss | `back` (avoids selecting Sign in / toggling play_pause) |
| VLM: MENU + **audio flowing** | **None (audio veto)** | Video is playing, refuses to interrupt — see Known Issues |
| VLM: MENU + no audio + **player overlay visible** | **`play_pause`** | Paused video showing player overlay (Description/cc/Up next + `\d+:\d{2}`) → resume with play_pause (NOT `down+select`, would land on Sign in; NOT `back`, would exit). See Known Issues. |
| VLM: MENU + no audio + no overlay | Select video | `down` + `select` (real menu) |
| VLM: SCREENSAVER | Wake + launch | `wakeup` + launch YouTube |
| Roku: screensaver overlay | Dismiss | `select` (wake from screensaver) |
| Roku: not on YouTube | Relaunch | `launch_app('youtube')` |

**Device-agnostic design:**
- `set_device_controller(controller, device_type)` accepts any controller
- Device type auto-detected from controller class name
- YouTube launch uses device-specific methods (Roku ECP `launch_app`, Android ADB intent)
- Skip command routes through active device controller

**Roku-specific features:**
- **Active app check** via ECP `/query/active-app` — definitively knows if YouTube is running
- **Screensaver detection** — checks for `<screensaver>` element in active-app response (Roku City screensaver overlays YouTube without closing it)
- **YouTube app ID**: 837

**OCR-based screen detection:**
VLM often misclassifies static YouTube screens (login, home) as "PLAYING". OCR keywords provide more reliable detection:

| Screen | Keywords | Action |
|--------|----------|--------|
| Login/account selection | `watch as guest`, `watchas guest`, `add a kid account`, `kid account`, `choose account`, `switch account` | `down` + `select` to choose account |
| Home/browse | `new to you`, `newtoyou`, `trending`, `subscriptions`, `library`, `views`, `year ago`, `month ago` | `down` + `select` to pick a video |

Login screen detection runs before VLM query. Home screen detection runs when VLM says "PLAYING" but frames are static.

**Frame-change verification (pause detection):**
- dHash (difference hash) compares two frames 3 seconds apart
- Hamming distance < 3 = truly static (paused or stuck)
- Audio flow check via ad_blocker's audio module (`0 <= last_buffer_age < 3s`) or ALSA `/proc/asound` status
- Note: `buffer_age = -1` means no audio ever received (not flowing), fixed to prevent false "audio flowing" detection
- Static frames + audio flowing = music stream (not paused) — prevents false play_pause
- Static frames + no audio = truly paused — sends play_pause after 2 consecutive checks

**VLM Screen Query Prompt:**
```
Look at this TV screen and classify it into exactly one category.
Answer with ONLY one of these words:
PLAYING, PAUSED, DIALOG, MENU, SCREENSAVER
```
This structured prompt returns in ~1.0s (vs 5-22s with descriptive prompts).

**Settings persistence:** `/home/radxa/.minus_autonomous_mode.json`
```json
{"enabled": true, "start_hour": 22, "end_hour": 6, "always_on": false}
```

**System settings:** `/home/radxa/.minus_system_settings.json`
```json
{"vlm_preload": true}
```
VLM preload loads the model at startup before HDMI signal arrives (configurable in Settings tab).

**API endpoints:**
- `GET /api/autonomous` - Current status (active, schedule, stats, device_type, device_connected)
- `POST /api/autonomous/enable` / `disable` / `toggle`
- `POST /api/autonomous/start` - Start immediately (manual override)
- `POST /api/autonomous/schedule` - Set hours and always_on flag
- `GET /api/autonomous/logs` - Recent log entries
- `GET/POST /api/settings/vlm-preload` - VLM preload toggle
- `GET/POST /api/settings/optimization` - Toggle block-duration falloff, HDMI reconnect grace, and greyscale preview. POST body: `{"key": "block_falloff"|"hdmi_reconnect_grace"|"greyscale_preview", "enabled": true|false}`. Persisted to `~/.minus_system_settings.json`. Setting `greyscale_preview` here propagates to the running ad_blocker immediately via `/blocking/set?preview_grayscale=...` so the current block updates on the fly.
- `GET/POST /api/settings/replacement-modes` - Which content kinds the blocking overlay rolls into. POST body: `{"modes": ["vocab","fact","haiku","photos"]}`. Server enforces at least one text kind (vocab/fact/haiku) remains enabled. Persisted with the rest of system settings.
- `GET/POST /api/media/photos` - List all uploaded photos (GET) or upload a new one (POST multipart with `file` field). Server re-encodes to JPEG (max 1920px long edge, quality 85) under `~/.minus_media/photos/`. Count cap 200, size cap 200 MB (oldest evicted on add).
- `GET/DELETE /api/media/photos/<id>` - Download JPEG bytes inline (GET) or remove by id (DELETE). Id is sanitized to hex to prevent path traversal.

**Web UI:** Toggle button, schedule time selectors, 24/7 checkbox (auto-enables mode), stats display in Settings tab, VLM preload toggle.

**24h stability test results (Apr 10-11, 2026):**
- Memory: stable at ~1.65GB RSS, no leak (tested 21+ hours continuous)
- FD count: stable at ~35, no leak
- Autonomous actions: 10+ DIALOG dismissals, 6+ screensaver auto-dismissals, all successful
- Ads blocked: 15+ ad breaks (OCR+VLM), all legitimate
- Audio-aware static detection: prevented 100+ false play_pause commands on lo-fi streams
- Audio restarts: 3 total (isolated, all self-recovered)
- Zero errors throughout

## WiFi Captive Portal

Minus includes a WiFi captive portal system for easy network configuration when no WiFi is connected.

**How it works:**
1. If WiFi disconnects for 30+ seconds, Minus creates a "Minus" hotspot AP
2. Users connect to the hotspot and get redirected to a setup page
3. Setup page shows available networks with signal strength
4. User selects network and enters password
5. Minus connects and stops the AP automatically

**Hotspot Configuration:**
- **SSID:** `Minus`
- **Password:** `minus123`
- **IP:** `10.42.0.1`
- **Band:** 2.4GHz (802.11 b/g)

**Captive Portal Detection:**
The portal supports automatic detection on mobile devices:
- `GET /generate_204` - Android captive portal check
- `GET /hotspot-detect.html` - Apple captive portal check
- `GET /connecttest.txt` - Windows captive portal check

**API Endpoints:**
- `GET /api/wifi/status` - Current connection status, AP mode state
- `GET /api/wifi/scan` - Scan for available networks
- `POST /api/wifi/connect` - Connect to a network (ssid, password)
- `POST /api/wifi/disconnect` - Disconnect from current network
- `POST /api/wifi/ap/start` - Start AP mode manually
- `POST /api/wifi/ap/stop` - Stop AP mode
- `GET /wifi-setup` - Captive portal setup page

**Settings Tab Integration:**
The Settings tab in the web UI shows:
- Current WiFi status (SSID, IP, signal strength)
- Disconnect button for current network
- Manual AP mode start/stop buttons

**Files:**
- `src/wifi_manager.py` - WiFi/AP management module
- `src/templates/wifi_setup.html` - Captive portal page
- `tests/test_wifi_portal.py` - Playwright tests (30 tests)

**Note:** The Radxa's internal WiFi antenna has limited range. For better AP coverage in production, consider using a USB WiFi adapter with external antenna.

## IR Transmitter (REI 8K HDMI Switch)

An IR LED wired to Rock Pi 5B header pin **38** (`GPIO3_B2` / Linux GPIO **106**, muxed to `PWM3_IR_M1`) lets Minus control a REI 8K 3-port HDMI switch. The target use case is autonomous mode rotating between streaming devices (Roku / Fire TV / Google TV) on a schedule so training data covers multiple home-screen layouts.

**Hardware setup (one-time):** enable the `rk3588-pwm3-m1` overlay, reboot. After reboot a new `/sys/class/pwm/pwmchipN` appears whose `device` symlink points to `fd8b0030.pwm`. See `docs/IR_TRANSMITTER.md` for overlay install steps and wiring.

**Protocol:** NEC at 38 kHz carrier. Captured codes (all address `0x80`, via Flipper Zero): `input_1=0x07`, `input_2=0x1B`, `input_3=0x08`, `power=0x05`, `next=0x1F` (cycles 1→2→3→1), `auto=0x09`.

**API:** `/api/ir/status | enable | disable | command`. See the Web UI *Key API Routes* section above. Server enforces a **1.5 s cooldown** between successful sends (`IRCooldownError` → HTTP `429` with `retry_after`).

**UI:** toggle + 6-button remote (Input 1/2/3, Power, Next, Auto) inside the *Autonomous Mode* section of the Settings tab. Panel hidden until toggled on. Buttons auto-disable during cooldown and a status line shows `sent power` or `cooldown — wait 0.74s`. When IR is enabled, the **same 6-button remote also appears on the Home tab** next to the live feed (gated on a `body.ir-home-enabled` class JS sets only when IR is available+enabled): on **desktop (≥768px)** it sits beside the main streaming remote inside the "Remote" dropdown and rides its open state (no separate toggle); on **mobile (<768px)** it's its **own separate dropdown** with a dedicated *HDMI Switch* toggle (opening the main remote does not reveal it). Both placements share `sendIRCommand()`, the cooldown, and the status line. The desktop/mobile split is pure CSS (`.main-open` honored ≥768px, `.ir-open` honored <768px).

**Standalone CLI:** `sudo python3 ir_transmit.py <button>` sends one button; `--list` prints all valid names. Uses the same `IRTransmitter` class as the webui so there is one source of truth.

**Key gotchas (the ones that burned us once already):**
- The Radxa pinout labels GPIO3_B2 with the RK3588 pin-function `PWM3_IR_M1`, not PWM14. Only the `rk3588-pwm3-m1` overlay wires pin 38.
- On a fresh PWM export, `polarity` defaults to `inversed` on this chip. That flips mark/space at the LED. `IRTransmitter.initialize()` sets `polarity=normal` while the PWM is disabled, before enabling.
- Writing to `duty_cycle` returns `EINVAL` while `period` is still 0. Always set `period` before `duty_cycle` on a fresh export.

**Files:**
- `src/ir_transmitter.py` — `IRTransmitter` class, NEC encoder, cooldown, PWM sysfs wiring
- `ir_transmit.py` — standalone CLI shim
- `minus.py` — instantiates `self.ir_transmitter`, persists `ir_enabled` in `~/.minus_system_settings.json`
- `src/webui.py` — `/api/ir/*` endpoints, cooldown → 429
- `src/templates/index.html` — Settings toggle + remote panel, plus the Home-tab IR remote (desktop side-by-side / mobile dropdown)
- `src/static/style.css` — responsive Home-tab IR remote rules (`.inline-remote-wrap`, `.inline-ir-caption`, `body.ir-home-enabled` gating, `.main-open`/`.ir-open` breakpoint behavior)
- `tests/test_ir_transmitter.py` — 20 unit tests (mocked sysfs)
- `tests/test_ir_ui.py` — Playwright UI tests (live service); `TestIRHomeRemoteUI` covers the Home-tab remote
- `docs/IR_TRANSMITTER.md` — full hardware, protocol, API, and troubleshooting docs

**Future work:** hook `minus.ir_transmitter.send("next")` into autonomous mode's scheduler on a 12 h or 24 h cadence. The boilerplate (flag, endpoints, UI, cooldown) is in place so the autonomous-mode change is a single call site.

## IR Receiver (Bench-Tested, Not Wired Into App)

A 3-pin IR receiver (TSOP38238 / VS1838B class) was evaluated on header pin **3** (`GPIO4_B3` / `gpiochip4` line **11**, Linux GPIO 139). Decoded the REI remote's NEC frames cleanly — `0x80 / 0x07,1B,08,1F` plus REPEAT codes — using `gpiomon` + a Python decoder in `test_ir_receiver.py`. **No production code yet**, just exploratory.

**Why pin 3 instead of pin 38 (alongside the transmitter):** the `rk3588-pwm3-m1` overlay parks pin 38's pad-mux on PWM3 at boot. `gpiomon` will *claim* the line but the GPIO controller is electrically disconnected from the pad — `gpioget` reads a constant `0` and no edges fire. Pin 3 / `GPIO4_B3` has no overlay claiming it, so default GPIO mux applies and it Just Works. Sanity check: `gpioget gpiochip4 11` returns `1` with the receiver powered and idle.

**Two gotchas burned dev time, captured here so we don't re-discover:**
- `gpiomon -B both` is invalid in libgpiod 1.6 — `-B` is *bias*, not edge. Default already monitors both edges; pass nothing.
- After a falling edge the line is LOW (a MARK), not a SPACE. Get the polarity backwards and every frame appears to start with a `~4500/~600 µs` "leader" because the real 9 ms leader mark gets filtered by the empty-buffer guard.

**Status:** test script only. Decoder is a copy-able starting point if/when we want a real `IRReceiver` module — see `docs/IR_RECEIVER.md` for the full sketch including threading model, suggested API surface, and integration ideas (closed-loop transmitter verification, external hardware trigger, remote learning, post-send confirmation for autonomous-mode scheduling).

**Files:**
- `test_ir_receiver.py` — standalone bench-test script (gpiomon subprocess + NEC decoder + `--raw` mode for non-NEC remotes)
- `docs/IR_RECEIVER.md` — findings, gotchas, future-module sketch

## Status LED Strip (WS2812B on SPI0 MOSI)

8× WS2812B addressable strip on header pin 19 (`GPIO1_B2` muxed as `SPI0_MOSI_M2`). All 8 LEDs are user-addressable.

**Why SPI MOSI:** WS2812B's 800 kHz protocol needs sub-µs timing. Userspace GPIO can't deliver that on Linux; the SPI controller can. We clock SPI at **6.4 MHz** and encode each WS bit as one full SPI byte (`0b11110000` = WS-1, `0b11000000` = WS-0) — the canonical Adafruit `NeoPixel_SPI` pattern. The frame is wrapped with **80 µs zero-byte resets on both sides** and sent via `writebytes2(bytes)`. We initially tried 3-SPI-bits-per-WS-bit at 2.4 MHz; the `spi-rockchip` driver's PIO mode inserts inter-byte gaps when its FIFO refills, and the tighter scheme didn't have enough skew tolerance — visible symptom was "solid green decoded as cycling red/blue/white". `rpi_ws281x` and friends depend on Broadcom PWM+DMA hardware and don't work on RK3588.

**Hardware:** bare-wire data line direct from header pin 19 to the strip — no level shifter, no inline resistor, no bulk cap needed for reliable operation on this board (verified: removed both the Adafruit-recommended 470 Ω series resistor and the 1000 µF V+/GND electrolytic, decoding stayed clean across all 8 LEDs). Keep the data wire ≤ 10 cm. (We previously shipped a "sacrificial first pixel" workaround that exposed only 7 LEDs; the encoding switch made it unnecessary and it has been removed.)

**Brightness cap (load-bearing):** `BRIGHTNESS = 0.10` is applied inside `set_pixel()` before storage; every other setter funnels through it. Caps peak draw at ~48 mA across all 8 LEDs — small enough to keep current swings from corrupting the data line on the marginal 3.3V signalling. Don't bypass it from the application layer; if you need more brightness, add external 5 V power to the strip first.

**Controller (`src/status_led_controller.py`):** `StatusLEDController` runs a 200 ms (5 fps) animation thread. Each renderer self-paces in seconds via the shared `_to_ticks()` helper, so per-animation cadence is preserved if the global tick rate is changed. State transitions are atomic and thread-safe; the lifecycle holds `_thread` populated until `join()` returns so a racing `start()` can't open a second SPI handle.

**State catalogue:**
| State | Visual | Trigger |
|---|---|---|
| `off` | dark | feature disabled |
| `initializing` | white pulse 1% → 10% → 1% (1 step/500 ms; 14 s/breath) | `Minus.run()` start, HDMI restoration |
| `idle` | solid green | `ad_blocker.start()`, `ad_blocker.hide()`, recovery complete |
| `blocking` | bouncing red Cylon eye + 2-pixel tail (~200 ms/step) | `ad_blocker.show(...)` |
| `paused` | slow yellow breathing (3 s) | `Minus.pause_blocking(...)` |
| `no_signal` | slow amber breathing (4 s) | `_on_hdmi_lost`, `start_no_signal_mode` |
| `autonomous` | slow blue breathing (4 s) | autonomous-mode active callback |
| `wifi_setup` | cyan alternating sweep (~250 ms/swap) | WiFi AP-mode started |
| `error` | fast red blink (2 Hz) | manual / subsystem failure |

**Persistence:** the on/off toggle is in `~/.minus_status_leds.json`. State itself is runtime-only and gets re-asserted by the next event.

**Display gating:** by default the strip stays dark while the HDMI-TX display is disconnected or powered off — keeps a dark room dark when the TV is off. State machine still ticks; only the wire output is suppressed, so animations resume seamlessly within ~200 ms of the display coming back. Implemented as an optional `drive_predicate` on the controller that `Minus` wires to `health_monitor._check_hdmi_output_connected()`. The `leds_require_display` flag (default True) toggles the gate from the WebUI; persisted in `~/.minus_system_settings.json`.

**Hardware setup (one-time):** enable `rk3588-spi0-m2-cs0-spidev` overlay, install `python3-spidev`, add user to `spi` group, reboot. `./install.sh` does all of that idempotently.

**API endpoints:** see the *Web UI* section above.

**Files:**
- `src/status_leds.py` — raw SPI driver, brightness cap, encoding
- `src/status_led_controller.py` — state machine + animation thread + persistence
- `minus.py` — instantiates `self.status_leds`, wires `_set_led_state` helper, hooks `_on_hdmi_lost` / `_on_hdmi_restored`
- `src/ad_blocker.py` — calls `_set_led_state` from `show()`/`hide()`/`start()`/`start_no_signal_mode()`
- `src/webui.py` — `/api/leds/*` endpoints
- `src/templates/index.html` — toggle + state palette in Autonomous Mode section
- `test_status_leds.py` — hardware walk/flash test
- `tests/test_status_led_controller.py` — 26 unit tests (mocked hardware)
- `tests/test_status_leds_ui.py` — Playwright UI tests (live service)
- `docs/STATUS_LEDS.md` — full docs

**Future work:** per-LED subsystem indicators (OCR / VLM / audio / HDMI / wifi / autonomous each get one LED), one-shot detection-event flashes, automatic `autonomous` state on autonomous-mode entry/exit.

## Streaming Device Configuration

Minus supports multiple streaming device types with device-specific remote control:

**Supported Devices:**
| Device | Protocol | Status |
|--------|----------|--------|
| Fire TV | ADB over WiFi | Full support |
| Roku | ECP (External Control Protocol) | Full support |
| Google TV / Android TV | ADB over WiFi | Full support |
| Apple TV | MRP/AirPlay | Coming soon |
| Generic | None | Ad blocking only |

**Web UI Setup:**
The Remote tab provides a device selector where users can:
1. Select their streaming device type
2. Follow device-specific setup instructions
3. Scan for devices on the network (Fire TV, Roku, Google TV)
4. Manually enter device IP address
5. Connect and control their device

**Device Configuration Persistence:**
- Configuration stored in `~/.minus_device_config.json`
- Persists device type, IP address, and setup state
- Survives service restarts

**API Endpoints:**
- `GET /api/device/config` - Get current configuration
- `GET /api/device/types` - List available device types
- `POST /api/device/select` - Select a device type
- `POST /api/device/ip` - Set device IP address
- `POST /api/device/setup-complete` - Mark setup complete
- `POST /api/device/reset` - Reset configuration

**Roku API Endpoints:**
- `GET /api/roku/status` - Connection status and device info
- `GET /api/roku/discover` - Scan network via SSDP multicast
- `POST /api/roku/connect` - Connect to Roku by IP
- `POST /api/roku/command` - Send remote command
- `POST /api/roku/launch/<app>` - Launch app (youtube, netflix, etc.)

**Roku Features:**
- Discovery via SSDP multicast
- ECP commands over HTTP to port 8060
- Control mode detection (Limited vs Full)
- Supports all navigation, media, and volume controls
- App launching: YouTube, Netflix, Prime, Disney+, Hulu, Plex, HBO, Peacock

**Fire TV API Endpoints:**
- `GET /api/firetv/status` - Connection status
- `GET /api/firetv/scan` - Scan network for Fire TV devices
- `POST /api/firetv/connect` - Connect to Fire TV by IP
- `POST /api/firetv/command` - Send remote command

**Google TV / Android TV API Endpoints:**
- `GET /api/googletv/status` - Connection status
- `GET /api/googletv/scan` - Scan network for devices (port 5555)
- `POST /api/googletv/connect` - Connect by IP:PORT (Wireless debugging uses dynamic port)
- `POST /api/googletv/command` - Send remote command (includes `assistant` for Google Assistant)

**Google TV Setup Notes:**
- Uses "Wireless debugging" (not USB debugging) for network ADB
- Settings > System > Developer options > Wireless debugging
- Shows IP:PORT on TV screen when enabled (e.g., 192.168.1.100:37421)
- Enter the full IP:PORT in web UI Remote tab
- First connection requires approving the ADB dialog on TV

## Fire TV Remote Control

Minus can control Fire TV devices over WiFi via ADB for ad skipping and playback control.

**Auto-setup:** Fire TV is automatically discovered and connected 5 seconds after Minus starts. First-time connection requires approving the ADB authorization dialog on the TV screen (OCR detects when it appears). ADB keys are saved for future connections.

**Features:**
- Auto-discovery of Fire TV devices on local network
- Verification that discovered device is actually a Fire TV
- ADB key generation and persistent storage for pairing
- Auto-reconnect on connection drops
- Full remote control: play, pause, select, back, d-pad, etc.
- Async-compatible interface

**Requirements:**
- Fire TV must have ADB debugging enabled
- First connection requires approving RSA key on TV screen
- Both devices must be on the same WiFi network

**Enabling ADB on Fire TV:** Settings > My Fire TV > Developer Options > ADB Debugging ON (enable Dev Options first via About > click device name 7x)

**Testing:** `python3 test_fire_tv.py [--setup|--interactive|--scan|IP]`

**Commands:** Navigation (up/down/left/right/select/back/home), Media (play/pause), Volume, Power

**Usage:** `quick_connect()` → `skip_ad()` / `go_back()` → `disconnect()`

**Setup States:** `idle` → `scanning` → `waiting_adb_enable` → `waiting_auth` → `connected`

## Google TV / Android TV Remote Control

Minus can control Google TV and Android TV devices over WiFi via ADB's Wireless debugging feature.

**Setup Flow:**
1. Select "Google TV / Android TV" in the Remote tab
2. On-screen overlay guides you through enabling Wireless debugging
3. Enter the IP:PORT shown on your TV's Wireless debugging screen
4. Approve the connection dialog on your TV

**Key Differences from Fire TV:**
- Uses "Wireless debugging" instead of "ADB debugging" (USB debugging)
- Dynamic port (not fixed 5555) - must enter IP:PORT format
- Found in Developer options after enabling developer mode

**Enabling Wireless Debugging:**
1. Settings > System > About > click Build number 7 times
2. Go back to System > Developer options
3. Turn ON "Wireless debugging"
4. Note the IP address and port shown on screen

**Commands:** Same as Fire TV plus `assistant` for Google Assistant button

**Setup States:** Same as Fire TV: `idle` → `scanning` → `waiting_adb_enable` → `waiting_auth` → `connected`

## Color Correction

Color correction is done via GStreamer's `videobalance` element in the pipeline.

**Why not ustreamer/V4L2?**
The HDMI-RX device doesn't support V4L2 image controls (saturation, contrast, brightness).
Only read-only controls are available: `audio_sampling_rate`, `audio_present`, `power_present`.

**Default settings (in `src/ad_blocker.py`):**
```
videobalance saturation=1.25 brightness=0.0 contrast=1.0 hue=0.0
```

**Web UI Controls:**
Color settings can be adjusted in real-time via the Settings tab in the web UI:
- **Saturation**: 0.5-1.5 slider (default 1.25, higher = more vivid)
- **Brightness**: -0.5 to 0.5 slider (default 0.0)
- **Contrast**: 0.5-1.5 slider (default 1.0)
- **Hue**: -0.5 to 0.5 slider (default 0.0)

**API Endpoints:**
```bash
# Get current color settings
curl http://localhost/api/video/color

# Set color settings (any combination)
curl -X POST -H "Content-Type: application/json" \
  -d '{"saturation": 1.3, "brightness": 0.1}' \
  http://localhost/api/video/color
```

**GStreamer ranges (for advanced use):**
- `saturation`: 0.0-2.0 (default 1.0)
- `contrast`: 0.0-2.0 (default 1.0)
- `brightness`: -1.0 to 1.0 (default 0.0)
- `hue`: -1.0 to 1.0 (default 0.0)

## Running as a Service

```bash
# Install
sudo ./install.sh

# View logs
journalctl -u minus -f

# Stop
sudo systemctl stop minus
./stop.sh  # Alternative with optional X11 restart

# Uninstall
sudo ./uninstall.sh
```

The service:
- Starts on boot (`multi-user.target`)
- Conflicts with display managers (gdm, lightdm, sddm)
- Restarts on crash (5 attempts per 5 minutes)
- Runs as root for DRM/device access

## Development Notes

### CRITICAL: Testing and Debugging Methodology

**Finding the Root Cause is ESSENTIAL:**
- Do NOT implement band-aid fixes that mask symptoms without understanding the cause
- Investigate WHY something is failing, not just WHAT is failing
- Example: If audio restarts constantly, don't just limit restart attempts - find out WHY it's restarting
- Use logs, `/proc` filesystem, API responses, and system state to trace the actual problem
- A fix that doesn't address root cause will likely cause other issues or recur

**Test Fixes BEFORE Pushing:**
- After implementing a fix, TEST it immediately by observing actual behavior
- Focus testing specifically on the ORIGINAL PROBLEM - verify the symptom is gone
- Do NOT push fixes without verification - iterate until the fix demonstrably works
- Run prolonged tests (30-60 seconds minimum) to catch intermittent issues
- Watch for the specific symptom that was reported (e.g., "frame jumps every 2-3 seconds")

**Testing Methodology:**
1. Understand the symptom clearly (what exactly is failing and how often)
2. Identify potential causes through log analysis and code review
3. Implement a fix targeting the root cause
4. Restart the service and observe behavior
5. Check logs for the specific error patterns that were occurring
6. Run a prolonged test (60 seconds) watching for the original symptom
7. Only commit/push after confirming the symptom is resolved

**Verification Techniques:**
- Check logs: `sudo journalctl -u minus --since "60 seconds ago" | grep -E "error|restart|fail"`
- Check FPS: `curl -s http://localhost/api/status | jq .fps`
- Check ALSA status: `cat /proc/asound/card*/pcm*/sub*/status`
- Check pipeline state: API responses, GStreamer state queries
- Record video samples for visual issues: `ffmpeg -i http://localhost:9090/stream -t 10 test.mp4`

**Common Pitfalls to Avoid:**
- Limiting retry attempts instead of fixing why retries are needed
- Assuming a fix works without observing the system under the original conditions
- Pushing multiple untested changes at once (makes debugging harder)
- Not checking if the "fix" introduced new problems

**Git commits:**
- Do NOT add "Co-Authored-By" lines to commits
- Do NOT add "Generated with Claude Code" lines to commits
- Keep commit messages clean and professional - just the message, no AI attribution

- Do NOT create v2, v3, v4 files - update existing files directly
- VLM uses Python axengine for inference (not pexpect/C++ binary)
- Both NPUs run in parallel without resource contention
- No X11 required - pure DRM/KMS display
- Color correction via GStreamer videobalance (not V4L2 controls)
- Health monitor runs every 5 seconds in background thread
- VLM frame files use PID-based naming to avoid permission conflicts
- Snapshots scaled to 960x540 before OCR (model uses 960x960 anyway, smaller = faster)
- ustreamer quality set to 80% for balance of quality and CPU load
- FPS tracked via GStreamer identity element with pad probe
- Startup cleanup removes stale frame files and kills orphaned processes
- Background upload is async to prevent blocking main thread
- Animation times optimized: 0.3s start, 0.25s end for fast response
- DYNAMIC_COOLDOWN reduced to 0.5s for faster ad detection

## Building Executable

```bash
pip3 install pyinstaller
pyinstaller minus.spec
# Output: dist/minus
```

Note: Models are external and must be present at runtime.

## Testing

The project includes a comprehensive test suite for all extracted modules.

**Running Tests:**
```bash
python3 tests/test_modules.py                  # 300+ unit tests
python3 tests/test_autonomous_mode.py          # Autonomous mode tests
python3 tests/test_recent_features.py          # Recent feature tests
python3 tests/test_block_decision_engine.py    # Blocking state-machine regressions
python3 tests/test_vlm_decision_sim_lfm2.py    # Monte-Carlo sliding-window eval (LFM2.5-VL; --sweep to retune; no NPU)
python3 tests/test_vlm_iter4_parity.py         # HISTORICAL: iter4 logit parity vs 800-img holdout — kept for reference, iter4 model no longer shipped
python3 tests/test_vlm_decision_sim.py         # HISTORICAL: iter4-bootstrapped sim, superseded by test_vlm_decision_sim_lfm2.py
python3 tests/test_review_ui.py                # Playwright UI tests (requires chromium)
python3 tests/test_ir_transmitter.py           # IR transmitter unit tests (mocked sysfs)
python3 tests/test_ir_ui.py                    # Playwright UI tests for IR remote panel
python3 tests/test_status_led_controller.py    # Status LED state-machine tests (mocked hardware)
python3 tests/test_status_leds_ui.py           # Playwright UI tests for status-LED panel
```

**Block-latency test harness (`tests/block_latency_harness.py`):**

Headless rig for tuning the blocking decision engine. Plays Big Buck Bunny
in a Python loop, lets the test orchestrator inject "AD"-style overlay
text on/off at controlled timestamps, and measures detect / recover
latency end-to-end through the production OCR + VLM workers + a faithful
mirror of `minus.py`'s blocking decision logic. No HDMI, no ustreamer,
no DRM, no audio.

```bash
# Place a video file at /home/radxa/test_assets/bbb.mp4 first.
python3 tests/block_latency_harness.py round1   # 9 detect/recover combos
python3 tests/block_latency_harness.py round4   # realistic ad-break shapes
python3 tests/block_latency_harness.py round5   # VLM state machine (injected verdicts)
python3 tests/block_latency_harness.py round6   # user-bug pause-on-ad regression
python3 tests/block_latency_harness.py round7   # production-shaped, OCR + VLM corroborated
```

`use_real_vlm=False` mode uses injected VLM verdicts so the engine's
sliding-window state machine can be driven deterministically without the
~30s real-VLM model load. Override `PARAMS` from a small wrapper script
to A/B-test tuning candidates; the in-rig defaults mirror the locked-in
production values.

**Test Coverage:**

| Module | Test Class | Tests |
|--------|------------|-------|
| `src/vocabulary.py` | TestVocabulary | Format validation, content checks, common words |
| `src/config.py` | TestConfig | Dataclass defaults, custom values |
| `src/skip_detection.py` | TestSkipDetection | Pattern matching, countdown parsing, edge cases |
| `src/screenshots.py` | TestScreenshots | Deduplication, file saving, truncation |
| `src/console.py` | TestConsole | Console blanking/restore commands |
| `src/capture.py` | TestCapture | Snapshot capture, cleanup |
| `src/drm.py` | TestDRM | DRM probing, fallback values |
| `src/v4l2.py` | TestV4L2 | V4L2 format detection, error handling |
| `src/overlay.py` | TestOverlay | NotificationOverlay, positions, show/hide |
| `src/health.py` | TestHealth | HealthMonitor, HealthStatus, HDMI detection |
| `src/fire_tv.py` | TestFireTV | Controller, key codes, device detection |
| `src/vlm.py` | TestVLM | VLMManager, response parsing, 4-tuple returns |
| `src/ocr.py` | TestOCR | Keywords, exclusions, terminal detection |
| `src/webui.py` | TestWebUI, TestWebUIExtended | Flask routes, all API endpoints |
| `src/ad_blocker.py` | TestAdBlocker, TestAdBlockerExtended | Blocking modes, color controls, animations |
| `src/audio.py` | TestAudio, TestAudioExtended | A/V sync, pipeline controls, mute/unmute |
| `src/fire_tv.py` | TestFireTV, TestFireTVExtended | Connection, commands, device discovery |
| `src/vlm.py` | TestVLM, TestVLMExtended | Response parsing, confidence detection |
| `src/ocr.py` | TestOCR, TestOCRExtended | Keywords, exclusions, terminal detection |
| `src/skip_detection.py` | TestSkipDetection, TestSkipDetectionExtended | Pattern matching, countdown parsing |
| `src/screenshots.py` | TestScreenshots, TestScreenshotsExtended | Deduplication, categories, truncation |
| `src/config.py` | TestConfig, TestConfigValidation | Defaults, custom values |
| `src/health.py` | TestHealth, TestHealthExtended | Monitoring, callbacks, status |
| `src/overlay.py` | TestOverlay, TestOverlayExtended | Positions, show/hide, text formatting |
| `src/drm.py` | TestDRM, TestDRMExtended | DRM probing, fallback values |
| `src/v4l2.py` | TestV4L2, TestV4L2Extended | Format detection, error handling |
| `src/console.py` | TestConsole, TestConsoleExtended | Console blanking/restore |
| `src/capture.py` | TestCapture, TestCaptureExtended | Snapshot capture, cleanup |
| Integration | TestIntegration | Cross-module tests |
| Memory | TestMemoryLeaks | Resource cleanup, executor reuse |
| Blocking | TestBlockingModeIntegration | State transitions, API format |
| Error Handling | TestErrorHandling | Missing subsystems, graceful failures |
| Concurrency | TestConcurrency | Thread safety, locks |
| Vocabulary | TestVocabulary, TestVocabularyContent | Format, content, duplicates |
| API Responses | TestAPIResponseFormats | Consistent response structure |
| `src/vlm.py` | TestVLMQueryImage | Custom prompt queries, error paths |
| `src/ocr.py` | TestOCRResilience | NPU failure handling, graceful degradation |
| `src/screenshots.py` | TestScreenshotDedup | dHash, blank rejection, rate limiting, per-category |
| Memory | TestMemoryManagement | Hash buffer caps, resource cleanup |
| HDCP | TestHDCPHandling | Encrypted frame handling, blank frame rejection |
| `src/autonomous_mode.py` | TestAutonomousMode | Schedule, VLM actions, state, persistence (separate file) |
| Review UI | TestReviewModal* | Playwright: desktop/mobile swipe, modal, API (separate file) |

**Test Design:**
- Tests are self-contained with temporary directories
- Mock subprocess calls to avoid system dependencies
- Fallback to manual test runner if pytest not installed
- All 300+ tests should pass on a clean system
- Playwright tests require chromium: `python3 -m playwright install chromium`

## Module Structure

The codebase has been refactored from monolithic files into smaller, focused modules:

**Extracted from `minus.py`:**
- `src/console.py` - Console blanking functions (`blank_console`, `restore_console`)
- `src/drm.py` - DRM probing (`probe_drm_output`)
- `src/v4l2.py` - V4L2 probing (`probe_v4l2_device`)
- `src/config.py` - Configuration dataclass (`MinusConfig`)
- `src/capture.py` - Snapshot capture (`UstreamerCapture`)
- `src/screenshots.py` - Screenshot management (`ScreenshotManager`)
- `src/skip_detection.py` - Skip button detection (`check_skip_opportunity`)

**Extracted from `ad_blocker.py`:**
- `src/vocabulary.py` - Spanish vocabulary list (`SPANISH_VOCABULARY`)

**Benefits:**
- Easier to test individual components
- Better code organization and discoverability
- Reduced file sizes (minus.py ~1700 lines, ad_blocker.py ~950 lines)
- Clear separation of concerns

## Known Issues / Fixed

### GStreamer Video Path Overlay (Historical - FIXED)

**Previous problem:** Adding a `textoverlay` element to the GStreamer video path caused pipeline stalls every ~12 seconds due to NV12 format incompatibility and 4K→1080p resolution mismatch.

**Solution implemented:** Text overlay is now rendered directly in ustreamer's MPP encoder via the blocking mode API. This:
- Composites directly on NV12 frames in the encoder
- Has minimal CPU impact (~0.5ms per frame)
- Works at any resolution without GStreamer pipeline changes
- Supports pixelated background, live preview window, and text overlays
- Uses FreeType for proper TrueType font rendering

### Memory Management (Fixed)

**Issue:** Long-running sessions (several hours) could accumulate memory due to RKNN inference output buffers not being explicitly released.

**Solution implemented:**
- RKNN inference outputs are now explicitly copied and dereferenced in `src/ocr.py`
- Periodic `gc.collect()` runs every 100 OCR frames and every 50 VLM frames
- Health monitor triggers emergency cleanup at 90% memory usage
- Frame buffers (`prev_frame`, `vlm_prev_frame`) are cleared during memory critical events

**ThreadPoolExecutor fix (Jan 2026):**
- **CRITICAL:** The OCR worker was creating a new `ThreadPoolExecutor` on every iteration, causing massive file descriptor and memory leaks (~12GB after 12 hours)
- Fixed by creating a single `ocr_executor` before the loop and reusing it
- Symptom: "Too many open files" errors, display goes blank, memory exhaustion

**Memory monitoring:**
- Health monitor checks memory every 5 seconds
- Warning logged at 80% usage
- Critical cleanup triggered at 90% usage

### Fire TV Setup (Fixed)

**Status:** Fire TV auto-setup is ENABLED with notification overlays working via ustreamer API.

**Startup timing:**
- Fire TV setup starts 5 seconds after service start (runs in parallel with VLM loading)
- Total time from start to connection: ~13 seconds (5s delay + ~8s scan/connect)

**Bug fixed:** Auth retry interval was 3 seconds, causing multiple auth dialogs on the TV before user could respond. Fixed to 35 seconds (longer than AUTH_TIMEOUT of 30s) in `fire_tv_setup.py`.

### Audio Watchdog Restart Loop (Fixed - Apr 2026)

**Symptom:** Frame jumps every 2-3 seconds due to constant GStreamer audio pipeline restarts.

**Root Cause:** When HDMI signal was restored, `resume_watchdog()` tried to create a new audio pipeline without:
1. Checking if the existing pipeline was already working
2. Cleaning up the old pipeline first

This caused the new pipeline to fail with "device in use" because the old pipeline still held the ALSA device. The watchdog then repeatedly tried to restart every 3 seconds.

**Why band-aid fixes don't work:** Initially tried limiting restart attempts, but this just disabled audio after 5 restarts instead of fixing the underlying issue. The correct approach was to find WHY restarts were happening.

**Solution implemented:**
- Added `_is_alsa_device_running()` helper that checks `/proc/asound/cardX/pcmYp/sub0/status` to verify if ALSA device is actually running with our PID
- This is more reliable than GStreamer state queries when PipeWire/WirePlumber is involved
- Modified watchdog loop to skip restarts when ALSA confirms audio is flowing
- Modified `resume_watchdog()` to check if pipeline is already PLAYING before restart
- Added proper cleanup of old pipeline before creating new one

**Key insight:** The `/proc/asound` status showed the device was RUNNING with minus as owner, proving audio WAS working. GStreamer state queries were unreliable due to PipeWire interference, but the kernel-level ALSA status was authoritative.

### MPP Decoder Stuck After HDMI Signal Drop (Fixed - Apr 2026)

**Symptom:** After a brief HDMI signal loss (even 8 seconds), the video pipeline stalls every ~12 seconds with `mpp_buffer: check buffer found NULL pointer from mpp_dec_advanced_thread`. Restarting the GStreamer pipeline alone doesn't help - MPP stays stuck.

**Root Cause:** The RK3588 MPP JPEG decoder holds resources that don't get properly freed when the GStreamer pipeline is destroyed. After the HDMI source briefly drops and recovers, the decoder enters a corrupt state that persists across pipeline restarts.

**Solution implemented:**
- After 3+ consecutive pipeline failures, the system now kills ustreamer (`pkill -9 ustreamer`) to force-release MPP resources
- The health monitor detects ustreamer is down and restarts it + the video pipeline with clean MPP state
- This auto-recovers from stuck MPP decoder without manual service restart

### Audio Device Mismatch on Display Reconnect (Fixed - Apr 2026)

**Symptom:** No audio after TV wakes up from standby. Audio pipeline starts on wrong HDMI output (e.g., `hw:0,0` instead of `hw:1,0`).

**Root Cause:** When the display retry loop detects a DRM output change (TV connected to different HDMI port than at boot), it updated the config but not the audio object's playback device. Audio would start on the old device.

**Solution implemented:**
- Display retry loop now checks if `drm_info['audio_device'] != self.audio.playback_device`
- If changed, stops the audio pipeline and updates the playback device before restarting
- Ensures audio always matches the active HDMI output

### Netflix Ad Countdown Detection (Fixed - Apr 2026)

**Symptom:** Netflix ads showing "Ad 10", "Ad 5" (countdown timer format) were not detected by OCR.

**Root Cause:** Existing OCR patterns only matched "Ad X of Y" format. Netflix uses standalone "Ad NN" where NN is seconds remaining.

**Solution:** Added regex pattern `^ad\s*\d+$` to match the countdown format.

### Skip-to-Unblock Delay (Fixed - Apr 2026)

**Symptom:** After successfully skipping an ad, the blocking overlay stayed for 2-3+ seconds waiting for OCR to detect the ad was gone.

**Solution:** After a successful skip command (auto or manual via web UI), blocking is now removed after a 1.5s delay instead of waiting for 3 OCR cycles. The delay allows the skip animation to complete, then force-unblocks by resetting all detection state. Skip command is device-agnostic — routes to Fire TV (`skip_ad()`), Roku (`send_command('select')`), or Google TV based on the configured device type.

**Follow-up — post-skip re-arm (Fixed - May 2026):** the 1.5s force-unblock above reset detection state but had **no grace window**, so the very next OCR frame re-read the skipped ad's lingering sponsored end-card / transition and *immediately re-armed the block*. Observed on minus-2: skip sent at T, `[SKIP] Forcing unblock` at T+1s, but OCR still `[BLOCKING OCR]` at T+2s and the block did not actually clear until **T+12s** (a Mint Mobile "Sponsored ·$15/Month … skip" end-card kept matching; `'skip in'` was recent so weak-`'sponsored'` suppression didn't engage). Root cause: the reset is a one-shot; nothing stops the lingering end-card from re-triggering. Fix: `SKIP_UNBLOCK_GRACE_SECONDS` (env `MINUS_SKIP_UNBLOCK_GRACE`). For that window after a successful skip, OCR ad frames are routed into no-ad accounting (so the block **decays and cannot re-arm**, not merely reset once) and the `_update_blocking_state` start gate refuses to (re)start — mirrors the `is_in_hdmi_reconnect_grace()` pattern.

**Follow-up 2 — grace was too long, delayed pod ad #2 (Fixed - May 2026):** at 8s the grace caused a **detection-latency regression**: ad pods (skip ad 1 → ad 2 starts ~1-2s later) are extremely common, and the grace suppressed the *next* ad's start for up to 8s — logs showed VLM flagging ad 2 at `agreement: 100% of 3` but `AD BLOCKING STARTED` withheld 2-3s by the grace. Root cause: the 8s was sized as a multi-minute backstop, now redundant (the universal `MAX_BLOCKING_DURATION` cap + weak-keyword suppression + transition-hold cap independently prevent long false holds). Fix: (1) grace **8s → 3s** (covers the typical skipped-ad end-card only); (2) the `_update_blocking_state` post-skip gate now bypasses suppression when `self.vlm_ad_detected` is set — a VLM sliding-window-confirmed detection (3+ decisions ≥80%) right after a skip is a real new pod ad, not the dying end-card (which can't sustain that); the OCR-accounting-site grace still neutralises the lingering end-card text. Files: `minus.py` (`SKIP_UNBLOCK_GRACE_SECONDS`, OCR-accounting `suppress_reason`, `_update_blocking_state` gate).

### GStreamer Bus Signal Watch FD Leak (Fixed - Apr 2026)

**Symptom:** After running for 12+ hours with no HDMI signal, the web server becomes unresponsive. Logs show `[Errno 24] Too many open files` errors. The service cannot open new files or sockets.

**Root Cause:** When the no-signal or loading GStreamer pipelines failed to start, the cleanup code did not remove the bus signal watch before destroying the pipeline. Each failed attempt leaked a file descriptor from `bus.add_signal_watch()`. With retries every 10 seconds, the 1024 FD limit was reached in ~3 hours.

**Solution:** Added proper bus cleanup in all pipeline failure paths:
```python
# Before destroying failed pipeline:
if self.bus:
    self.bus.remove_signal_watch()
    self.bus = None
```

Fixed in `src/ad_blocker.py`: `start_no_signal_mode()` and `start_loading_mode()` failure paths and exception handlers.

### Audio Pipeline Zombie State After Sleep/Wake (Fixed - Apr 2026)

**Symptom:** After TV/display sleeps for several hours and wakes up, there is no audio output even though the health monitor reports `audio=OK` and the ALSA device shows `state: RUNNING`.

**Root Cause:** The GStreamer audio pipeline runs in a separate thread. When the display sleeps, this thread can crash or die (e.g., due to ALSA device disconnection), but:
1. The Python `AudioPassthrough` object retains a stale reference to the dead pipeline
2. The ALSA device shows `owner_pid` pointing to the dead thread's PID
3. The health check only queries the Python GStreamer state, not the actual ALSA device ownership
4. Result: Health reports `audio=OK` while no actual audio is flowing

**Detection:** Check if the ALSA playback device's `owner_pid` corresponds to a live process:
```bash
# Get owner PID
cat /proc/asound/card1/pcm0p/sub0/status | grep owner_pid
# owner_pid   : 179247

# Check if process exists
ps -p 179247
# Returns empty = zombie audio state!
```

**Solution:** Enhanced `_check_audio_pipeline()` in `src/health.py` to:
1. Read the ALSA device status from `/proc/asound/cardX/pcm0p/sub0/status`
2. Verify the `owner_pid` corresponds to a live process (check `/proc/{pid}/` exists)
3. If owner is dead but device shows RUNNING, trigger full `_restart_pipeline()` (not just queue flush)
4. 10-second cooldown after any restart before zombie detection runs again (prevents restart loops)
5. Skip zombie detection if restart is already in progress
6. This runs every health check cycle (5 seconds), so recovery happens automatically

**Files modified:**
- `src/health.py` - Added `_check_alsa_zombie_state()` method with full restart and cooldown logic

### OCR Ad Timestamp Pattern Fix (Fixed - Apr 2026)

**Symptom:** Ad blocking would flicker on/off during ads because OCR sometimes reads "Ad 0:42" (with space) and sometimes "Ad0:42" (no space) or "Ado:55" (OCR misreads '0' as 'o').

**Root Cause:** The OCR pattern used word boundaries (`\bad\b`) which required a space between "Ad" and the timestamp. When OCR dropped the space, the pattern didn't match, counting as "no ad". After 3 "no ads", blocking ended, then immediately re-triggered when a frame with space was detected.

**Solution:** Updated `src/ocr.py` to match OCR variants:
- `ad[0o]:` pattern catches "Ad0:" and "Ado:" (no space, or 'o' misread)
- `[0-9o]:\d{2}` timestamp pattern handles 'o' misread as '0'
- Both per-element and cross-element checks updated

**Test cases now matched:**
- `Ad 0:42` - standard format ✓
- `Ad0:42` - no space ✓
- `Ado:55` - OCR misread '0' as 'o' ✓
- `0:30 | Ad` - Hulu style ✓

### HDMI PHY Not Reinitializing After TV Restart (Fixed - Apr 2026)

**Symptom:** After TV restart/power cycle, the GStreamer pipeline reports "No-signal display started successfully" but the TV shows its own "HDMI 1 No Signal" message (meaning no video signal from RK3588).

**Root Cause:** When the TV restarts, the HDMI hotplug event is detected and the sysfs status changes from "disconnected" to "connected", but the HDMI PHY (physical layer) doesn't properly reinitialize. The DRM connector shows as connected, but no actual video signal is being transmitted.

**Discovery:** Physically unplugging and replugging the HDMI cable made the display work, indicating the HDMI PHY needed reinitialization that wasn't happening on TV restart.

**Solution:** Force HDMI PHY reinitialization via DPMS (Display Power Management Signaling) cycle:
1. When TV reconnects, health monitor detects status change and waits 2s for link stabilization
2. DPMS Off (value 3) sent via `modetest -M rockchip -w {connector}:DPMS:3`
3. Wait 300ms
4. DPMS On (value 0) sent via `modetest -M rockchip -w {connector}:DPMS:0`
5. This forces the HDMI transmitter to reinitialize, equivalent to cable replug

**Implementation:**
- `src/health.py` - Health monitor detects TV reconnection and calls `ad_blocker.restart(hdmi_reconnect=True)`
- `src/ad_blocker.py` - `_restart_pipeline(hdmi_reconnect=True)` does:
  1. Stop existing pipeline
  2. DPMS cycle via `_force_hdmi_reinit()`
  3. Re-probe DRM to detect connector/plane changes
  4. Restart audio pipeline (required after TV power cycle)
  5. Start new video pipeline
- For no-signal mode, DPMS cycle is done in `start_no_signal_mode()` directly

**Key heuristics for detecting working vs broken state:**
| Heuristic | Working | Broken (needs DPMS) |
|-----------|---------|---------------------|
| sysfs status | connected | connected |
| sysfs dpms | On | On |
| Video output | Visible | TV shows "No Signal" |

Note: All sysfs values look identical in both states - the only difference is whether video is actually being transmitted. The DPMS cycle is applied preemptively on every TV reconnection.

### ALSA Zombie Detection False Positives (Fixed - Apr 2026)

**Symptom:** Audio cuts out every ~15 seconds with logs showing "Audio zombie state detected - GStreamer playing but ALSA owner dead" followed by constant pipeline restarts.

**Root Cause:** The ALSA `owner_pid` in `/proc/asound/cardX/pcm0p/sub0/status` is actually a **thread ID (TID)**, not a process ID (PID). The zombie detection code was checking `/proc/{owner_pid}` which doesn't exist for threads - threads are listed under `/proc/{main_pid}/task/{tid}` instead.

**Solution:** Updated `_check_alsa_zombie_state()` in `src/health.py` to check both locations:
1. First check `/proc/{owner_pid}` (works if it's a PID)
2. If not found, check `/proc/{main_pid}/task/{owner_pid}` (works if it's a TID)

This prevents false zombie detection when the audio thread is actually alive and healthy.

### Minus Overlay Text Triggering False Positive Ad Detection (Fixed - Apr 2026)

**Symptom:** Screen stuck on "Initializing..." for 20+ minutes. GStreamer pipeline in restart loop (37+ attempts). ustreamer is capturing video correctly but display pipeline fails.

**Root Cause:** The Fire TV notification overlay shows "Ad skipping enabled." which contains the word "ad". When OCR read this overlay text, it triggered false positive ad detection. This activated the blocking mode, which caused MPP pipeline errors (`mpp_buffer: check buffer found NULL pointer`).

**Why overlay is visible to OCR:** The notification overlay is composited at the ustreamer encoder level BEFORE the snapshot, so `/snapshot/raw` includes overlay text. This is by design for the preview window in blocking mode, but it means OCR sees everything on screen including our overlays.

**Solution:** Added our overlay messages to the OCR exclusion lists:
- `'ad skipping enabled'`, `'ad skipping'`, `'adskipping'` added to `AD_EXCLUSIONS` in both `src/ocr.py` and `src/ocr_worker.py`

**Files modified:**
- `src/ocr.py` - Added Minus overlay exclusions
- `src/ocr_worker.py` - Added Minus overlay exclusions

### Autonomous Mode False-Pause When Display Disconnected (Fixed - Apr 2026)

**Symptom:** Running autonomous mode with HDMI-TX disconnected, music videos with static album art were being paused by autonomous mode every 20 seconds, interrupting legitimate playback.

**Root Cause:** With display disconnected, the audio pipeline's alsasink can't open HDMI-TX, so the pipeline never receives buffers (`last_buffer_age == -1`). `_is_audio_flowing()` returned False. On music videos with static art (hamming≈0), the pause detector concluded "static frames + no audio = PAUSED" and sent `play_pause`, actually pausing content that was playing.

**Solution:** Added `_is_audio_pipeline_available()` in `src/autonomous_mode.py`. When the audio pipeline has never received a buffer or its state is stopped, treat audio as "unknown" rather than "not flowing". `_is_screen_static()` returns False in that case so autonomous mode does not assume paused. VLM's direct `PAUSED` verdict still triggers play.

### Autonomous Mode Navigation During Ads (Fixed - Apr 2026)

**Symptom:** During real ads on YouTube, autonomous mode would fire `down` + `select` commands thinking it was on the home screen, navigating through the ad UI and occasionally switching to a different video.

**Root Cause:** `HOME_SCREEN_KEYWORDS` in `src/autonomous_mode.py` contained `'sponsored'` and `'views'`. "Sponsored · Visit advertiser" on YouTube pre-roll ads and "347M views" in any video's info panel both matched, triggering the home-screen action path.

**Solution:**
- Removed `'sponsored'` and `'views'` from `HOME_SCREEN_KEYWORDS`.
- Added `AD_ONLY_KEYWORDS` (`'visit advertiser'`, `'send to phone'`, `'skip in'`, `'skip ad'`) — if any are present, skip home-screen detection.
- Added `ad_blocker.is_visible` guard — if blocking is active, never classify as home screen.
- In `minus.py`, added a secondary audio-aware guard: if the OCR match is only `'sponsored'` **and** HDMI-IN `audio_present=0`, suppress the block. Real video ads transmit audio; home-screen sponsored tiles usually don't. Uses new `_hdmi_audio_present()` helper reading v4l2-ctl directly so it works even when our playback pipeline is down.

### OCR 'ad in' Keyword Matching Inside Words (Fixed - Apr 2026)

**Symptom:** False ad blocks triggered when OCR read `"LOADING"` or `"reading"` on screen.

**Root Cause:** `AD_KEYWORDS_EXACT` in `src/ocr_worker.py` contained `'ad in'`. The alphanumeric-normalized form is `'adin'` (4 chars), which appears as a substring in `'loading'` (lo**adin**g), `'reading'` (re**adin**g), and similar words.

**Solution:** Removed `'ad in'` from exact keywords. The specific patterns for `"Ad N of M"`, `"Ad N"` countdown, and `"ad with timestamp"` (in both `ocr.py` and `ocr_worker.py`) already cover legitimate ad timestamps.

### VLM Degraded State Auto-Recovery (Added - Apr 2026, ROOT CAUSE CORRECTED)

**Symptom:** After several hours of runtime, VLM inference degrades from ~0.7s to ~15–18s per query and returns descriptive responses to short-answer prompts. Each DISCARDED (>2s) entry makes the system effectively OCR-only. Not thermal — temperatures stayed around 70°C both when healthy and when slow.

**Original solution (kept as defense-in-depth):** Rolling latency window + auto-recovery in `src/vlm_worker.py`:
- `_record_latency()` / `_maybe_auto_recover()` called after each successful inference.
- If P95 over the last 10 queries exceeds 3.0s, trigger a worker restart.
- If a prior recovery happened within the last 3 minutes and we're degraded again, escalate to a **deep restart** with 8s NPU-release backoff.
- 60s cooldown prevents thrash.
- `get_latency_stats()` exposes samples/P50/P95/max via `/api/health` under `subsystems.vlm.latency`.

Axera telemetry (`axcl-smi info --temp / --npu / --cmm`) is wired into `/api/health` at `subsystems.vlm.axera` and exposed as Prometheus gauges `minus_axera_*` for alerting on temperature or memory pressure.

**⚠️ Correction (Apr 2026):** The "NPU drift to degraded state" framing turned out to be wrong. Controlled experiments (`docs/VLM_NPU_DEGRADATION.md`) confirmed:
- Latency is **deterministically image-dependent**, not a state that drifts in over time.
- Per-token decode rate is constant (~0.23 s/tok); the slow inferences are slow because the model generates 30–60 tokens of descriptive response instead of 1–3 tokens of `Yes.`/`No.`.
- The NPU, axcl driver, and Axera firmware are all healthy throughout. `axcl-smi reboot` and `rmmod` + `modprobe` of the host modules do **not** change behavior on the same image.

**Real fix:** Cap `max_new_tokens` at the model layer (5 for `detect_ad`, 8 for `query_image`). With the cap, worst-case latency drops from ~12 s to ~1.3 s and the entire restart-cycle pathology goes away. The auto-recovery logic above stays in as defense-in-depth for any genuine NPU pathology, but in normal operation it should never fire.

### VLMProcess Cross-Thread Race (Fixed - Apr 2026)

**Symptom:** Intermittent `too many values to unpack (expected 2)` from `[AutonomousMode] VLM screen query failed`, plus a sustained worker restart cycle (~15–40 hard kills per 15 min) that the soft/hard timeout logic could not damp on its own.

**Root Cause:** `VLMProcess.detect_ad` (called from the detection-loop thread) and `VLMProcess.query_image` (called from the autonomous-mode thread) shared the same request/response `multiprocessing.Queue` with no request-to-response correlation and no lock around the queue or the shared state (`_consecutive_timeouts`, `_pending_response`, `_recent_latencies`). When both threads called concurrently:

1. A `detect_ad` 4-tuple response could be `get()`-ed by the `query_image` caller (which expected a 2-tuple) — and vice versa — producing the unpack error.
2. Concurrent mutation of `_consecutive_timeouts` and `_pending_response` produced spurious threshold trips, triggering hard kills the system did not actually need. Each hard kill cost ~25s of model reload, during which more queued requests timed out, perpetuating the cycle.

**Solution:**
- Added `self._call_lock = threading.Lock()` to `VLMProcess.__init__`.
- Refactored `detect_ad` and `query_image` into thin wrappers that acquire the lock, then delegate to `_detect_ad_locked` / `_query_image_locked` with the original logic.
- This serializes the two callers across the entire request → response cycle, so cross-pollinated responses cannot happen and the shared timeout state stays consistent.

Upstream's tuple-shape defensive guards (introduced in commit `7c42e80`) are kept as belt-and-suspenders — they tolerate a stale leaked response if one ever does slip through. The lock prevents the leak; the guards handle it if prevention fails.

**Why a lock and not separate queues / request IDs:** simplest correct fix that is local to `VLMProcess`. Detection-loop calls are ~4 Hz and complete in ~0.7s; autonomous-mode calls are once per 2 minutes and complete in ~1.0s. The lock contention is negligible in practice. A dedicated request-ID protocol would be cleaner but invasive to both worker and callers.

**Files modified:**
- `src/vlm_worker.py` — `_call_lock`, `_detect_ad_locked`, `_query_image_locked`

### A/V Sync Flush Disabled (Apr 2026)

**Symptom:** Every 45 minutes of uptime, the `AudioPassthrough` watchdog ran its periodic sync-queue flush; `Sync queue flushed` was always followed ~12s later by `Pipeline issue detected: not in PLAYING state (paused)` and a full `Restarting pipeline (attempt N)`. Cumulative effect: ~32 spurious audio restarts per day, each a brief dropout. The feature that was supposed to *prevent* restarts was *causing* them.

**Root cause (investigated in 4 failed fix iterations):** the flush itself is unrecoverable without a full pipeline rebuild on this pipeline configuration.
1. `flush-start` event puts the sync queue and downstream into flushing mode.
2. `flush-stop` should resume streaming, but `syncqueue` has `min-threshold-time=300ms` that blocks downstream reads until the queue has refilled past the threshold.
3. While the queue is blocking, `alsasink` — having no data to consume — closes its PCM device. ALSA `state` transitions out of `RUNNING`, `hw_ptr` goes to 0.
4. `set_state(PLAYING)` on the pipeline cannot bring `alsasink` back up because the upstream queue is still blocked. The pipeline gets stuck in `PAUSED` for 10+ seconds until the watchdog gives up and restarts the whole pipeline.

Attempts that did **not** work:
- Same-iteration `continue` after flush (commit `3b4e0d0`) — subsequent iterations still trip on the lingering `PAUSED`.
- 10-second post-flush "grace window" — flush recovery takes longer than that.
- Explicit `pipeline.set_state(PLAYING)` with bounded 2s wait — `get_state` returns `PAUSED` regardless.
- Temporarily zeroing `syncqueue.min-threshold-time` across the flush + 400ms refill sleep — `alsasink` had already dropped the PCM by then.

**Solution:** flip `self._sync_reset_enabled = False` in `AudioPassthrough.__init__` (see `src/audio.py:151`). The periodic flush never runs, so it can never cascade. Drift isn't a real concern in this pipeline (`provide-clock=false` on alsasrc, `sync=false` on alsasink) and 48+ hours of runtime without a working flush showed no observable A/V desync.

To find the commit that made this change: `git log --all --oneline --grep='disable periodic A/V sync flush'` (commit subject is stable across amends).

**Kept as a side-benefit of the investigation:** rewrote `_is_alsa_device_running()` to sample `hw_ptr` across a 50ms window instead of comparing ALSA's `owner_pid` to the main process PID. The old check compared an ALSA-reported *thread TID* (often a stale one) against the main PID, so it could never return True under normal operation. The watchdog's "GStreamer reports PAUSED but ALSA is flowing — skip restart" rescue path has always been broken; now it works.

**If drift becomes a real problem in the future (easy revert):**
1. Write a flush mechanism that does not let `alsasink` close its PCM device — either by pausing→flushing→playing the whole pipeline in one atomic block, or by replacing `syncqueue` with an element that doesn't block on `min-threshold-time`.
2. Only after (1) works, flip `_sync_reset_enabled` back to `True` in `src/audio.py`.
3. Re-run the soak test (`_sync_interval = 2.5 * 60` + 5-min monitor for ~45 min) and confirm `audio.restart_count` stays at 0.

**Do NOT simply flip `_sync_reset_enabled` back to `True` without (1).** The bug will return.

**Files modified:**
- `src/audio.py` — `_sync_reset_enabled = False` + explanatory block comment; `_is_alsa_device_running()` rewrite

### VLM Queue Desync (Off-By-One on Soft Timeout) (Fixed - Apr 2026)

**Symptom:** After pausing on an ad on Netflix and unpausing, the blocking overlay stayed for ~20 seconds applied against frames where the show was clearly playing again. Other variants: VLM verdicts persistently lagging actual screen content by one frame; rare reports of "Ad 1:30 left" claims long after a real ad had ended.

**Root cause:** `VLMProcess._detect_ad_locked` and `_query_image_locked` (`src/vlm_worker.py`) shared the same MP request/response queues with no per-request correlation. When VLM hit a soft timeout (1.5s, ~15% of inferences in normal load), the request stayed in flight and `_pending_response` was set to `True`. On the next call:

1. The drain attempt was a single `get(timeout=0.1)`. If the worker had not yet pushed its response (still mid-inference), drain timed out.
2. The code then **fell through and `put`-ed a NEW request anyway**.
3. Now two requests were in flight. Worker finished the first → pushed result A → caller's `get(SOFT_TIMEOUT)` received result A as the answer for request B.
4. The queue was now permanently off-by-one. Every subsequent `get()` returned the prior frame's verdict.
5. After a pause-on-ad (where the queue accumulated several "ad" verdicts during the pause), the entire backlog was delivered against post-unpause "show is playing" frames → 10–20 seconds of phantom blocking.

The shared `/dev/shm/minus_vlm_frame_<pid>.jpg` path made it worse — the file was always the most recently written frame, so even the worker's view of "what was frame N" could be stale.

**Solution:**
1. **Drain ALL stale responses** at function entry using a `get_nowait()` loop (was a single `get(timeout=0.1)`).
2. **If a request is genuinely still in flight** after draining, do NOT queue another. Return `"PENDING"` (or `"KILLED"` after RESTART_THRESHOLD consecutive pendings). Caller treats this exactly like the existing `"TIMEOUT"` skip path — `is_ad=False`, `confidence=0.0`, no-op on the sliding window.

This guarantees only one request is ever in flight, which incidentally also fixes the file-content race because the worker dequeues and reads the file in tight succession.

**Files modified:**
- `src/vlm_worker.py` — `_detect_ad_locked` and `_query_image_locked` rewritten with multi-drain + don't-double-queue

### HDMI Restored Recovery Leaves Display Dead Forever (Fixed - Apr 2026)

**Symptom:** TV stays frozen on a stale frame for hours. Web app shows live content. `subsystems.video.status` reads `error`/`reason: no_pipeline`. `fps_capture` is healthy (~42 fps) but `fps_display` is essentially 0. Service uptime can be many hours; restart is the only recovery.

**Root cause (two coupled defects in `Minus._on_hdmi_restored()` at `minus.py:634`):**

1. **`ad_blocker.start()`'s return value was ignored.** When HDMI input recovers but HDMI-OUT (the TV) is still disconnected, kmssink can't open the DRM plane and `start()` returns `False`. The recovery handler proceeded as if all was well and logged `[Recovery] HDMI recovery complete`.

2. **`self.display_connected` was left stuck at `True`.** The display retry loop (`_start_display_retry_loop`) is the only thing that can recreate a dead pipeline post-startup, but it gates on `not self.display_connected`. Since recovery never set the flag to `False` on failure, the retry loop never ran. Pipeline stayed dead until the next service restart.

Observed once today across a 17-hour run: at 08:19:59 HDMI input recovered after a 550-second loss while the TV was off. Recovery declared success. `Attempting to reconnect display pipeline` log line count for the entire 17-hour run: **0**. The display sat frozen on its last decoded frame all the way until a manual restart at 12:46.

**Solution:** check `start()`'s return value; on failure, set `display_connected=False`, populate `display_error`, and call `_start_display_retry_loop()`. The retry loop already exists and works correctly — it just needs to be armed. Audio remains paused/muted on the failure path; it'll be resumed by the normal start path inside the retry loop when the pipeline finally comes up.

**Why the failure mode is sticky without this fix:** there is no other code path that ever flips `display_connected` from `True` back to `False` post-startup. Initial startup (`minus.py:3000`) is the only place. The retry loop never fires because its gate (`not self.display_connected`) stays `False`.

**The NO SIGNAL behavior is unchanged:** the HDMI-LOST path still calls `start_no_signal_mode()` and the health monitor's "Continuous NO SIGNAL mode enforcement" loop still re-triggers it whenever HDMI input is absent. So the desired "TV shows NO SIGNAL when input is gone" behavior is preserved end-to-end.

**Files modified:**
- `minus.py` — `_on_hdmi_restored()`: capture return value, branch on success/failure, arm retry loop on failure

### Phantom Re-Block After Pause-On-Ad (Fixed - Apr 2026)

**Symptom:** User pauses on a real ad on Netflix, ad ends offscreen during the pause, user unpauses to actual show content — and Minus shows the blocking overlay for ~5 more seconds on the show content. Reproduced via the block-latency harness (`round6`): with the OLD parameters, **3/3 scenarios** observed phantom re-blocks at ~0.9s after unpause.

**Root cause:** three coupled defects in the static-suppression / cooldown machinery, each individually plausible but combining badly:

1. **`OCR_STOP_THRESHOLD = 4`** meant blocking took 4 OCR cycles × 0.5s = 2s to clear once the ad ended — already over the 1.5s responsiveness target the user wanted.
2. **`scene_change_threshold = 0.01`** misclassified ~26% of natural low-motion frames in real video content as "static" (measured against BBB's actual inter-frame mean-abs-diff distribution: p5=0.002, p50=0.017, max=0.31). Static suppression therefore flapped on/off mid-content during slow scenes.
3. **`dynamic_cooldown = 0.5s`** was too short for the post-pause AD overlay to actually transition off-screen. The cooldown completed → state was cleared → the very next OCR cycle re-detected the still-lingering AD text → blocking re-fired immediately.

The user only saw symptom 3 in the worst form (the phantom re-block), but symptoms 1 and 2 amplified its visibility — symptom 2 was also responsible for the related "blocking flips off mid-content" issue earlier in the same investigation.

**Solution:** three coordinated tuning changes, locked in via `tests/block_latency_harness.py` measurements (rounds 1, 4, 6, 7):

| Parameter | Old | New | Effect |
|---|---|---|---|
| `OCR_STOP_THRESHOLD` (`minus.py`) | 4 | **2** | recover 2.0s → 1.0s |
| `scene_change_threshold` (`config.py`) | 0.01 | **0.001** | only truly-frozen frames (~1.7% of BBB) register as static; natural low-motion content (~98%) keeps flowing |
| `dynamic_cooldown` (`config.py`) | 0.5s | **1.5s** | post-pause AD overlay actually finishes transitioning off-screen before state is cleared |

**Verification:** `round6` of the harness re-runs the user's scenario 3× per parameter set:
- OLD params: 3/3 phantom re-blocks, max 0.90s after unpause
- NEW params: **0/3** phantom re-blocks ✓

**Final scenario performance with locked-in params:**
- detect: mean 0.59s, max 0.66s, **9/9 clean** across all round-1 ad shapes
- recover: mean 0.97s, max 1.15s, **all under 1.5s goal**
- 0 false-positive blocking events across 15s of clean content (round 7)
- 0 mid-block flaps across a 30s sustained ad (round 7)

**Defense-in-depth:** `tests/test_block_decision_engine.py` adds 11 lightweight unit tests for the DecisionEngine state machine (cooldown clearing, OCR stop threshold, VLM-only fast-stop, the user-bug regression itself with both OLD and NEW params asserted). Runs as part of the standard test suite.

**Files modified:**
- `minus.py` — `OCR_STOP_THRESHOLD = 2` + comment with link to harness
- `src/config.py` — `scene_change_threshold = 0.001` + measurement-derived comment, `dynamic_cooldown = 1.5` (already changed in the cooldown-fix commit earlier this session)
- `tests/block_latency_harness.py` — new ~700-line headless harness (BBB source, OCR/VLM workers, decision-engine mirror, 7 rounds of scenarios)
- `tests/test_block_decision_engine.py` — new 11 unit tests

### OCR Worker Keyword-Pattern Drift (Fixed - May 2026)

**Symptom:** During real ad breaks, blocking flapped on/off every 5–15 seconds even though OCR was reading the ad timer cleanly every frame. Logs showed sequences like:

```
00:29:00 [BLOCKING OCR] - Ad 1:11        ← match (boundary)
00:29:01 [BLOCKING OCR] - RATED TV-MA    ← no_ad #1
00:29:03 OCR #62188            - Ad1:09   ← no_ad #2 — silently!
00:29:03 OCR: ad no longer detected (after 2 no-ads)
00:29:03 AD BLOCKING ENDED after 3.1s
00:29:08 - Ad 1:02 → AD BLOCKING STARTED again
```

OCR's text output was literally the running ad timer, but `check_ad_keywords` was returning `ad_detected=False`, so the no-ad counter incremented and tripped `OCR_STOP_THRESHOLD=2`.

**Root cause:** there are two `check_ad_keywords` implementations — `src/ocr.py:515` on the `PaddleOCR` class, and `src/ocr_worker.py:310` on `OCRProcess`. Production wires `self.ocr = OCRProcess()` in `minus.py:563`, so `OCRProcess.check_ad_keywords` is what actually runs. The two have drifted: `ocr.py` was updated months ago to handle the OCR-drops-the-space variant ("Ad1:09") and looser separator/digit misreads, but `ocr_worker.py` was never updated.

The drifted worker pattern was:

```python
# src/ocr_worker.py (pre-fix) — ONLY matches when there's a word boundary after "ad"
if re.search(r'\bad\b', text_lower) and re.search(r'[0-9o]:[0-9o]{2}', text_lower):
    matched.append(('ad with timestamp', text))
```

`\bad\b` requires a non-word char after `d`. "Ad1:09" puts a digit (word char) right after `d`, so the boundary doesn't exist and the pattern fails. The timestamp side was also stricter: `[0-9o]` only (no `l/I/i`), and `:` only (no `;/.`).

So every frame OCR'd as `Ad1:09` was silently a no-ad, and a streaming service that briefly replaces the timer with a rating card ("RATED TV-MA") at ad-to-ad transitions was enough to chain two consecutive no-ads and trip the unblock — even though the same ad break was still running.

**Fix:** `src/ocr_worker.py:404` and the cross-element check at `src/ocr_worker.py:418` now mirror `src/ocr.py:595` exactly:

```python
has_ad = (re.search(r'\bad\b', text_lower)
          or re.search(r'ad[0-9oOlIi][:;.]', text_lower))
has_timestamp = re.search(r'[0-9oOlIi][:;.][0-9oOlIi][0-9oOlIi]', text_lower)
if has_ad and has_timestamp:
    matched.append(('ad with timestamp', text))
```

Verified against actual log samples: `Ad1:09`, `Ad 1:11`, `Ad0:30`, `Ado:30`, `Adl:l0`, `Ad1:02`, `Ad0:55` all match; bare `Ad`, `RATED TV-MA`, `loading`, `reading` correctly do not.

Both files now carry a `Mirrors src/ocr.py:NNN — keep in sync` comment to make the next drift visible at the patch site.

**Why this isn't *just* a tighter mirror:** the duplication exists at all because `OCRProcess` runs `check_ad_keywords` locally in the main process (it's just string matching, no NPU work) instead of in the worker subprocess where `PaddleOCR` lives. Deleting the duplicate would require either (a) importing `PaddleOCR` from `ocr.py` into `ocr_worker.py` and calling its method, or (b) sending `ocr_results` back into the worker for keyword check. (a) is the right fix and a small refactor — open task for next session. Until then, the mirror comments are the guardrail.

**Files modified:**
- `src/ocr_worker.py` — per-element + cross-element keyword patterns updated; mirror comments added
- `CLAUDE.md` — *OCR Timestamp Pattern Handling* section now calls out the dual-source requirement

### Static Suppression Catches Real Video Ads (Fixed - May 2026)

**Symptom:** A real ad break was playing (Michelob Ultra "Skip MI 15 Sponsored ZeRo Meo ULTRA USA"). OCR detected every frame correctly — every log line read `[AD DETECTED - STATIC SUPPRESSED]`. But the block never fired. Sequence:

```
18:36:30  ad starts ("Sponsored | ALENOL | 0.0%")
18:36:34  OCR matches keywords ('skip in', 'sponsored')
18:36:39  [Static] Screen static for 2.2s — suppressing blocking
18:36:46  AD BLOCKING STARTED (OCR) → next frame "[AD DETECTED - STATIC SUPPRESSED]"
... 102 seconds of suppressed frames ...
18:38:21  [Static] Screen became dynamic — cooldown 1.5s
18:38:23  block finally fires
```

The user paused the ad mid-stream expecting the overlay to be there — it never appeared. From their side it looked like OCR was broken.

**Root cause:** the static-screen suppressor was originally added (CLAUDE.md `Static Screen Suppression` section) to prevent blocking when the user pauses on a YouTube/Netflix home screen that *happens* to show a "Sponsored" tile. The implementation treats any static screen + ad-ish text as "probably paused, don't block." But it can't distinguish:

- Paused-on-home-screen with a sponsored corner tile (don't block — original intent)
- **Real graphic video ad with low motion** (Michelob's static brand frame + tiny "Skip in 15" countdown — needs to be blocked)

Both look static + have ad-ish text. The 2.5s `STATIC_TIME_THRESHOLD` is too tight for low-motion ad creative to ever beat the suppressor.

**Fix:** add a "strong ad signal" override. Keywords that only appear in *active video-ad UIs* — never on paused home screens or recommendation grids — override the suppressor. Implemented in `minus.py`:

- New class attribute: `STRONG_AD_KEYWORD_NAMES` (frozenset of keyword names) — `skip ad`, `skip ads`, `skip in`, `skip ad (fuzzy*)`, `video will play after ad`, `visit advertiser`, `visitadvertiser`, `ad X of Y`, `ad countdown`, `ad with timestamp`, `ad with timestamp (cross-element)`.
- New state: `last_strong_ad_time = 0.0` plus `STRONG_AD_HOLD_SECONDS = 5.0`.
- In the OCR loop, when `check_ad_keywords` returns matched keywords, if any name is in the strong set, update `last_strong_ad_time = time.time()`.
- In the static suppression check, before activating:
  - If suppression is currently on AND `(now - last_strong_ad_time) < STRONG_AD_HOLD_SECONDS` → force-clear suppression immediately. Don't wait for a scene change + cooldown; a low-motion ad might never trigger a scene change.
  - Otherwise, only activate suppression if `not strong_ad_recent` — so the suppressor can't engage while a real video ad is being detected.

**Why "strong" vs "weak" keywords:**
- `Sponsored`, `Shop now`, `Learn more`, bare `Ad`, bare `Ads` — all legitimately appear on home screens (Fire TV recommendation rows, YouTube home tiles, Roku featured grids). Including them as override would re-introduce the pause-on-home-screen false positive that suppression was built to prevent.
- `Skip in N`, `Ad N of M`, `0:NN Ad`, `Visit advertiser`, `Send to phone` — only appear during active video-ad playback. A paused home screen never shows "Skip in 14".

**Verified intent preserved:** pausing on a YouTube home screen with a "Sponsored" recommendation tile still suppresses correctly because none of the strong keywords are present.

**Edge case:** pausing on an in-progress video ad (real ad break that the user paused themselves) — the strong keyword is still on screen, so the block fires. Is that the right behavior? Probably yes: if the screen shows a real ad with a Skip button, the user almost certainly wants it blocked regardless of pause state. Better to err toward blocking real ads than away from them.

**Files modified:**
- `minus.py` — `STRONG_AD_KEYWORD_NAMES` + `STRONG_AD_HOLD_SECONDS` + `last_strong_ad_time` in `__init__`; OCR-loop update at the matched-keywords site; static-suppression check restructured to consult `strong_ad_recent` for both activation gating and mid-suppression lift.
- `CLAUDE.md` — *Static Screen Suppression* section now documents the override; this Known-Issue entry captures the root cause.

### Unified Debug Toggle + Top-Right OCR Snippet (Added - Apr 2026)

The blocking overlay grew a third debug element: a top-right `(Ad) 0:30 left` snippet showing the OCR text that triggered the block, with the matched keyword wrapped in parens. The existing *Debug Dashboard* settings toggle was unified into a single *Debug* toggle that gates three things together: the `[ BLOCKING // ... ]` header (top), the bottom-left stats dashboard, and this new top-right OCR snippet.

**Persistence:** the toggle is a system setting (`debug_overlay`, default `True`) in `~/.minus_system_settings.json`. Pushed into `ad_blocker.set_debug_overlay_enabled()` at startup so off survives a service restart.

**Recursion concern (resolved by existing architecture):** the natural worry is that putting the OCR trigger text back on screen would make OCR keep seeing "Ad" forever. That cannot happen because OCR consumes `/snapshot/raw` (`src/capture.py:134`), which the patched ustreamer serves from `us_blocking_store_raw_frame()` *before* the blocking composite runs (`ustreamer-garagehq/src/ustreamer/http/server.c:1026`). The new top-right text — and every other element on the blocking overlay — is therefore invisible to OCR. **Do not change OCR to read `/snapshot` (the composited path)** without first stripping the debug texts; otherwise the displayed snippet becomes self-triggering. The `Minus Overlay Text Triggering False Positive Ad Detection` fix in this same Known Issues list is the cautionary tale — the *notification* overlay (`/overlay`, distinct from `/blocking`) DOES composite before the snapshot and required keyword exclusions to suppress recursion.

**ustreamer C-side change:** added a third text region. Files in `ustreamer-garagehq`:
- `src/libs/blocking.h` — `text_ocr` field on `us_blocking_config_s`, `US_BLOCKING_TEXT_OCR_SIZE = 256`, declaration of `us_blocking_set_text_ocr()`
- `src/libs/blocking.c` — setter, clear/snapshot/composite all extended; render block draws at `text_x = dst_width - text_w - 30, text_y = 30` using the same IBM Plex Mono Regular face as `text_stats`. Reuses the existing `_ft_mutex` since FreeType is not thread-safe across the 4 MPP workers.
- `src/ustreamer/http/server.c` — `text_ocr` URL param parsing in `_http_callback_blocking_set`. `text_stats_scale` is reused for the OCR text size (no separate scale param needed).

**Python wiring:**
- `src/ad_blocker.py` — `_ocr_trigger_text` instance, `_format_ocr_trigger(raw, source)` builds the `(Ad) 0:30 left` snippet (paren-wraps the matched keyword inside the OCR text snippet, ASCII-collapsed, ≤50 chars), `_render_ocr_text()` returns empty when debug is off so the C side draws nothing. `show(source, ocr_trigger_text="")` accepts the trigger payload from minus; only overwrites the stored snippet when given a non-empty value (or when transitioning to vlm-only) so the top-right does not flicker as OCR text comes and goes during a block. `set_debug_overlay_enabled()` re-renders `text_vocab` (to add/strip the header) and pushes `text_ocr` in the right direction without waiting for the next rotation.
- `minus.py` — stashes `last_matched_keywords` in the OCR loop, helper `_first_match_for_overlay()` returns `(keyword, snippet_text)` for the most recent match. `_load_system_settings` adds `debug_overlay: True` default + `set_debug_overlay_enabled(enabled)` persists and propagates. Cleared in the block-end branch so the next block starts fresh.
- `src/webui.py` — `/api/debug-overlay/{enable,disable}` route through `minus.set_debug_overlay_enabled()` for persistence. The `POST /api/test/trigger-block` endpoint injects a synthetic `("Ad", "Ad 0:30 left")` snippet when `source` is `ocr`/`both` so the top-right slot can be exercised without real ads.
- `src/templates/index.html` — toggle relabeled "Debug Dashboard" → "Debug" with a tooltip listing what it controls.

**Files modified:**
- `ustreamer-garagehq/src/libs/blocking.{h,c}`, `src/ustreamer/http/server.c` — new `text_ocr` API + top-right render
- `minus.py`, `src/ad_blocker.py`, `src/webui.py`, `src/templates/index.html`

### FastVLM-1.5B → 0.5B iter4 Logit-Threshold Migration (May 2026)

**What changed:** the VLM ad detector was swapped from FastVLM-1.5B
(decode-based, parse "Yes"/"No" text) to the fine-tuned **FastVLM-0.5B
ad-classifier iter4** using **logit-based thresholding** — prefill only,
softmax the first-position logits over the full vocab, compare
normalized `P(Yes)` to `AD_THRESHOLD=0.76`. Per the implementation guide
`/home/radxa/axera_models/LOGIT_THRESHOLD_IMPLEMENTATION.md`.

**Why:** iter4 is **same/better accuracy at ~3× the speed** and removes
an entire failure class:
- Holdout (800 img, 2026-05-15): **F1 94.72, ad-recall 94.25%,
  non-ad-recall 95.25%** — beats the 1.5B (the task saturates at 0.5B;
  the 1.5B never won on device, see `BENCHMARKS.md`).
- Latency **~0.33s deterministic** (p95 0.34s) vs the 1.5B's ~0.9–1.1s
  with a 10–15s descriptive-paragraph tail. `detect_ad` has **no decode
  loop**, so that pathology (the whole reason for `max_new_tokens` caps +
  aggressive auto-recovery in `docs/VLM_NPU_DEGRADATION.md`) is now
  *structurally impossible*. `query_image` (autonomous mode) is
  unchanged — still decode-based, still capped.
- The threshold is tunable post-hoc from logged `p_yes_norm` without
  re-running inference.

**Parity proof:** `tests/test_vlm_iter4_parity.py` runs the production
`VLMManager.detect_ad` over the full 800-image holdout and compares to
the calibration script's scores: **0/800 classification flips**, max
|Δ p_yes_norm| = 0.00005, confusion matrix identical to `BENCHMARKS.md`
(TP=377 TN=381 FP=19 FN=23). The in-app pipeline is byte-faithful, so
the 0.76 threshold is valid — *provided the prompt stays byte-for-byte
identical to `threshold_sweep.py`* (system = "You are a helpful
assistant."). `VLMManager.AD_SYSTEM_PROMPT`/`AD_PROMPT` enforce this;
do not edit them without recalibrating.

**Path resolution:** iter4 ships a flat dir with no tokenizer/utils.
`src/vlm.py` auto-detects flat-0.5B vs legacy-1.5B layout; tokenizer
falls back to the canonical `FastVLM-0.5B/fastvlm_tokenizer` (dims MUST
match the model: 896 hidden / 24 layers) and utils to the patched
`FastVLM-1.5B/utils` (`infer_func` has the `max_new_tokens` cap that
`query_image` needs; `llava_qwen` is byte-identical to the 0.5B copy).
Overridable via `MINUS_VLM_TOKENIZER_DIR` / `MINUS_VLM_UTILS_DIR`.
Vision encoder input is read from the session (`pixel_values` for
iter4, `images` for the 1.5B) so both layouts work unchanged.

**Worker-timeout retune (`src/vlm_worker.py`):** since `detect_ad` is
now deterministic ~0.33s with no runaway-token failure mode,
`HARD_TIMEOUT` 5.0→3.0→**2.0s** and `LATENCY_P95_TRIGGER` 3.0→2.0s — a
real hang is the only thing that can exceed these now, so recovery is
faster with zero false-restart risk. The timeouts are **shared** by
`detect_ad` and `query_image`, and the floor is set by `query_image`,
not `detect_ad`: measured `detect_ad` p95 0.33s / max 0.33s (0 events
>1s over a full day) vs `query_image` (decode-based, autonomous mode)
typical 1.3s / max 1.5s with `max_new_tokens=8`. So `HARD_TIMEOUT=2.0`
keeps a 0.5s margin over `query_image`'s observed max; `SOFT_TIMEOUT`
stays 1.5s — `query_image` legitimately reaches ~1.5s, so a lower soft
timeout would spuriously time out screen queries and (3 consecutive)
hard-kill the worker (~15s reload). Going below `HARD_TIMEOUT=2.0`
would require per-request-type timeouts (not worth the complexity while
`detect_ad` is this stable).

**Sliding-window retune — the load-bearing fix.** The anti-waffle
window was built for the 1.5B's ~36% home-screen FP rate. iter4 has
near-perfect per-frame separation (clean video p_yes≈0.05, ad text
p_yes≈0.85), so the window is now over-conservative. `tests/test_vlm_
decision_sim.py` is a Monte-Carlo simulator that drives the faithful
`DecisionEngine` mirror with a virtual clock and VLM verdicts
**bootstrapped from the real 800-image holdout scores** (so per-frame
error rate + calibrated confidence are statistically identical to
production iter4) across 64 scenario shapes (pre/mid-roll, multi-ad
breaks, back-to-back, pause-on-ad, content-only, rapid alternation,
tiny/long ads × OCR strong/absent/delayed/flaky). Sweeping **1920
param combos** found:
- **`vlm_history_window` 45→8s is the decisive lever.** A 45s window
  keeps stale *content* no-ad votes that mathematically prevent a
  VLM-alone ad from reaching the start ratio until they age out:
  VLM-only detect **~38s, 81% of VLM-only ads missed**. At 8s:
  VLM-only detect **~7s, ~0% missed**, with OCR-path metrics unchanged
  (OCR-triggered detect ~0.9s, 0 miss) and **0 phantom content-blocks**
  preserved. Stop/recovery uses the consecutive counter, not the
  window, so shrinking it has no recovery downside.
- `vlm_start_agreement` 0.90→0.80→**0.70** (+0.10 hysteresis = 0.80
  effective): with a short window you can't afford a high bar; the
  sweep's feasible optimum is 0.65–0.70 and 0.70 stays phantom-free.
- `vlm_min_decisions` 4→**3**, others unchanged.

Validated on the real OCR+VLM rig (`block_latency_harness.py`,
`tests/harness_iter4_retune_ab.py`): OCR detect/recover and
false-positive/phantom behaviour unchanged; VLM-only transition sharply
faster. The earlier-in-this-migration retune (4→3 decisions, 0.90→0.80)
was validated only on the clean-injection rig which *resets state
before each VLM-only test* — that masked the stale-vote dilution; the
simulator (content precedes the ad, as in reality) exposed it.

**New test scripts** (added to the suite):
- `tests/test_vlm_iter4_parity.py` — 800-image holdout parity/accuracy.
- `tests/test_vlm_decision_sim.py` — Monte-Carlo sliding-window sweep
  (`--sweep`) and current-param eval; verdicts bootstrapped from real
  holdout scores; class-aware feasibility (OCR-present must be perfect;
  VLM-only is the optimised soft tail; multi-ad-gap flaps tracked
  separately as a mirror artifact — production holds those via
  `_is_transition_frame`).
- `tests/harness_iter4_retune_ab.py` — A/B wrapper over the real rig.

**Files modified:** `src/vlm.py` (logit path, dual-layout resolution,
calibration-exact prompt, dynamic vision-input name, mmap embeds),
`src/config.py` (`VLM_MODEL_DIR` → iter4, env-overridable),
`src/vlm_worker.py` (timeouts), `minus.py` (sliding-window params),
`tests/block_latency_harness.py` (PARAMS mirror — kept 1:1 with
production), CLAUDE.md.

### iter4 query_image p128 Overflow + Production FP / Slow-Recovery Fixes (May 2026)

Found by live monitoring on minus-2 (autonomous mode + Roku-driven ad
tests) right after the iter4 migration deployed. Four coupled defects:

**1. `query_image` crashed every call → autonomous mode fully blind.**
Symptom: `[AutonomousMode] VLM screen query: list index out of range`
every ~20s. Root cause: the iter4 LLM is **p128 — a single 128-token
prefill chunk** (`qwen2_p128_l*` axmodels expose only shape-group 0
=decode and 1=p128). `detect_ad`'s calibrated prompt is ~94 tokens
(fits). `query_image` built a verbose system message *plus* the verbose
`SCREEN_QUERY_PROMPT` (per-category descriptions) = **187 tokens** →
`infer_func.prefill` needs `slice_idx=1` → asks axengine for
`shape_group=2` → `self._outputs[2]` IndexError on every call.
`detect_ad` is prefill-only-logit and short so it never hit this.
Fixes: `src/autonomous_mode.py` `SCREEN_QUERY_PROMPT` reverted to the
minimal form (≈119 tok with the short system prompt — **do not
re-expand; the p128 budget is image-64 + ~64 text**); `src/vlm.py`
`query_image` uses the short `AD_SYSTEM_PROMPT` and a hard guard that
returns `PROMPT_TOO_LONG` (fail-soft, callers already treat non-category
replies as "unknown screen") instead of crashing if any caller exceeds
128 tokens again.

**2. `query_image` decode shape mismatch (surfaced after fix 1).**
`K_cache expect [1,1023,128], got [1,1024,128]`. iter4's decoder K/V
cache is compiled for **seq-len 1023**, not the 1.5B's 1024 (the
reference `test_ad_classifier.py` / `threshold_sweep.py` build
`InferManager(max_seq_len=1023)`). `src/vlm.py` now sets a layout-aware
`LLM_MAX_SEQ_LEN` (1023 for the flat 0.5B-iter* layout, 1024 for the
legacy 1.5B subdir layout). `detect_ad` is prefill-only so it never hit
this; only `query_image`'s decode path did.

**3. Multi-minute false-positive blocks (observed: 591s).** A static
"Sponsored · Peel to collect" promo held an OCR+VLM block for ~10 min.
Three causes: (a) bare `'sponsored'` (a *weak* keyword) was triggering
**and sustaining** OCR blocking whenever HDMI-IN audio was present — but
home/promo screens carry audio (autoplay previews, music), so the
`_hdmi_audio_present()` discriminator was wrong; (b) suppressed
`'sponsored'` frames fell through *without* feeding the no-ad counters,
so an active block's `ocr_no_ad_count` froze and it never decayed; (c)
the only max-duration safeguard was on `source=="vlm"` (90s) — `ocr` and
`both` had **no cap**. Fixes in `minus.py`: bare-`'sponsored'`-only is
suppressed unless a `STRONG_AD_KEYWORD` was seen within
`STRONG_AD_HOLD_SECONDS` (VLM still independently catches genuine
sponsored-only video ads); suppressed frames now route into the same
no-ad accounting so blocks **decay**; new universal
`MAX_BLOCKING_DURATION` (150s, env `MINUS_MAX_BLOCKING_DURATION`) clears
all detection state on cap.

**4. Slow ad→content recovery (~3s, over the 1.5–2s target).** OCR
snapshot capture runs ~2.5s/frame on a headless box (HDMI-OUT
disconnected). A `both`-source block waited on OCR's 2 consecutive
no-ad frames (~3s+) even though VLM had already cleared in ~0.3s. Fix:
for `source=="both"` (both signals detected the ad) stop on whichever
clears first — `ocr_says_stop OR vlm_says_stop`. Pure `source=="ocr"`
still requires OCR (VLM dissent must not stop early); `source=="vlm"`
unchanged. Measured live: recovery ~3s → **~1s** across Target / Ford /
HBS / Acura / `skip in` / `ads` ad breaks, 0 multi-minute holds, 0
`query_image` errors, 0 safeguard fires.

**Monitoring:** `tools/ad_block_monitor.py` parses `journalctl -u minus`
and reports per-block source / duration / recovery-latency / trigger
keywords and flags `WEAK_FP` (sponsored-only/no-keyword block lingering
>20s), `OVERLONG`, `SLOW_RECOVER(>3.5s)`, `query_errs`. Self-elevates
via `sudo -n journalctl` when not root. Run periodically (a recurring
agent re-runs it, root-causes any flag, tunes, restarts, commits) —
target: zero false-positive blocks, zero multi-minute holds, recovery
≤1.5–2s.

**Files modified:** `src/autonomous_mode.py`, `src/vlm.py`, `minus.py`,
`tools/ad_block_monitor.py` (new), CLAUDE.md.

### FastVLM iter4 → LFM2.5-VL Migration (May 2026)

**What changed:** the ad-detection VLM was swapped from FastVLM-0.5B
iter4 (Qwen2-decoder, p128 prefill + KV-cache decode) to
**LFM2.5-VL-450M-ft-v2-fused-v2** (16 fused-layer axmodels,
prefill-only, no KV cache). Autonomous-mode `query_image` was also
moved off FastVLM — there is now **no FastVLM dependency anywhere**.
Per `/home/radxa/axera_models/LFM2/MINUS_INTEGRATION_GUIDE.md`.

**Why:**
- **Accuracy:** holdout 97.0% / 94.8% ad-rec / **99.2% non-ad-rec** vs
  iter4's 94.75% / 94.25% / 95.25%. The remaining home-screen-FP class
  is essentially eliminated at the model layer (-3.95% FP rate).
- **Latency:** ~0.37s deterministic (vision 185ms + 16 fused layers
  185ms) vs iter4's ~0.44s. Both prefill-only, both immune to the
  descriptive-paragraph latency pathology that plagued the 1.5B.
- **Code simplification:** no `LlavaConfig` / `InferManager` /
  `expand2square` / `CLIPImageProcessor` / `ml_dtypes.bfloat16` /
  `llava_qwen.py` / `infer_func.py`. No `_reset_kv_cache()` work
  between inferences (conv state is per-call, freshly allocated). The
  `sys.path.insert` to a `utils/` dir is gone. ~590 lines of
  FastVLM-specific code in `src/vlm.py` collapsed to ~340 LOC for the
  full LFM2 implementation.
- **One model serves both paths.** FastVLM iter4 was p128 — a
  single 128-token prefill chunk — which forced `query_image` to use
  decode-based generation (~1.0s) with a 5-class chat prompt that
  flirted with the 128-token ceiling. LFM2's 320-token prefill window
  + first-token logit-lookup multi-class classification lets the same
  prefill loop serve `detect_ad` and the autonomous-mode screen query.

**Architecture:**
- `detect_ad` returns `(is_ad, response, elapsed, p_yes_norm-derived
  confidence)`. Decision is `argmax(max(YES_logits), max(NO_logits))`
  over the 4 spelling variants each (Yes/yes/ Yes/ yes) — matches
  `infer_vlm_fused.py:classify_image` exactly. `MINUS_VLM_AD_THRESHOLD`
  (default 0.5 ≡ argmax) gates `p_yes_norm` instead if set != 0.5;
  argmax already gives 97% holdout accuracy so the threshold knob is
  rarely useful.
- `query_image` does the same prefill, then looks up the first-token
  logit for each of the 5 screen-state classes (max over the
  no-leading-space and leading-space spellings of each) and returns
  the argmax. `max_new_tokens` parameter retained for API compat but
  IGNORED — there is no decode loop. Per-layer decode axmodels are not
  shipped with v2-fused; only `post_d.axmodel` is.
- Both paths use a single `VLMManager._lock` for serialisation. The
  outer `VLMProcess._call_lock` is unchanged.

**Hard 320-token prefill window:** the axmodels were compiled for a
fixed `[1, 320, 1024]` input shape. The chat-template + 256-image-token
overhead is ~37 text tokens (BOS, IM_START × 3, IM_END × 2, system =
"You are a helpful multimodal assistant by Liquid AI.", `user\n`,
`assistant\n`, etc.); the user question can be max ~30 tokens. The
`ad-prompt` ("Is this an advertisement? Answer Yes or No.") tokenises
to 293 total; `screen-prompt` ("Classify this TV screen: PLAYING,
PAUSED, DIALOG, MENU, or SCREENSAVER?") tokenises to 312. **An earlier
draft of the screen prompt tokenised to 326 — over by 6** — and
silently truncated the `[IM_START] assistant\n` suffix → the
last-position logits were garbage. `load_model()` now fails loud if
either cached prompt overflows; `query_image` returns
`"PROMPT_TOO_LONG"` for on-the-fly oversized prompts. The 326-token
prompt was committed and quietly broken until the load-time check
caught it during this migration.

**Sliding-window / decision-engine retune (DONE, middle-ground):**
`vlm_min_decisions: 5 → 3` and `vlm_start_agreement: 0.80 → 0.70`
(+0.10 hysteresis → 0.80 effective). `vlm_history_window=8s` kept.
The iter4-era hardening to 5 / 0.80 was sized against iter4's
mid-show VLM-only FPs — a failure class LFM2 mostly does not produce
(non-ad-recall 99.2% vs 95.25% → ~4× lower per-frame FP rate, with
tighter logit margins on confident cases: clean p_yes ≈ 0.001–0.01,
ad p_yes ≈ 0.97–0.99). At the iter4-hardened params, the LFM2
holdout-bootstrapped simulator (`tests/test_vlm_decision_sim_lfm2.py`,
2560-combo sweep × 64 scenarios × 30 seeds) measured the old params
as **infeasible**: O_rec p95 = 18s, V_det mean 9.3s / p95 20s,
V_miss 7.5%, Vs_miss 76% — the start gate could not accumulate
enough votes fast enough on real ads. New params: **O_rec p95
18s → 3.2s** (5.6× better), **V_det mean 9.3s → 7.4s, p95
20s → 9s**, **V_miss 7.5% → 0.3%**, phantom blocks remain **0**.
Middle-ground vs the sweep's most-aggressive winner (`min_dec=2,
agree=0.60`): the winner halves V_det further (4.5s mean) and cuts
Vs_miss to 29%, but pushes phantom-block math closer to the edge
(~0.7/day estimated on holdout-bootstrapped content vs ~0.1/day at
the middle ground). Chose the middle ground — it captures the
biggest wins (recovery tail + VLM-only miss rate) without crowding
the phantom margin. **Rollback path** if real-world VLM-only
mid-show false triggers reappear: revert to `min_dec=5, agree=0.80`
(see `minus.py` comment block around `self.vlm_min_decisions` — keep
the iter4 hardening rationale documented for whoever does the
rollback). Sweep raw log: `/tmp/lfm2_sweep.log` (until reboot);
rerun via `python3 tests/test_vlm_decision_sim_lfm2.py --sweep`.

**Gotchas (from the integration guide, kept here):**
- Vision preprocessing is direct bilinear-resize + `(x/255 - 0.5)/0.5`
  + patchify into `(1, 1024, 768)`. **Do not** reuse FastVLM's
  `expand2square` + CLIPImageProcessor — the fine-tune is on the
  patchify path, accuracy degrades otherwise.
- `indices` input to the attention layers must be `int32` at runtime
  (axengine asserts shape + dtype).
- After running the vision encoder you get 256 feature vectors. The
  prompt has 256 copies of `IMG_TOKEN_ID = 396` inserted at the
  image slot; the `_prefill_last_logits` step then OVERWRITES those
  256 prefill-data positions with the actual vision features. Forgetting
  the splice leaves the model with the embedding for token 396 — garbage.
- All axmodel I/O is FP32. The `ml_dtypes.bfloat16` casts from
  FastVLM are gone.
- NPU3 mode is baked into the axmodel files (Pulsar2 build flag) —
  no runtime flag needed. If a rebuild loses NPU3, latency rises ~40%.

**Files modified:**
- `src/vlm.py` — full rewrite around `VLMManager` with LFM2 prefill,
  fused-layer loop, multi-class logit lookup. `FASTVLM_MODEL_DIR`
  kept as a backwards-compat alias pointing at the new model dir.
- `src/config.py` — `VLM_MODEL_DIR` default → LFM2 dir, env var
  `MINUS_VLM_MODEL_DIR` unchanged.
- `src/autonomous_mode.py` — `SCREEN_QUERY_PROMPT` shortened from
  the verbose 326-token form to 312-token "Classify this TV screen:
  PLAYING, PAUSED, DIALOG, MENU, or SCREENSAVER?". Comment refreshed.
- `minus.py` — VLM-loading log line "FastVLM-1.5B" → "LFM2.5-VL-450M".
- CLAUDE.md — *Overview*, *Performance*, *VLM Model* sections and
  this Known Issues entry.

**Verification:** `sudo python3 minus.py` boot-to-ready in ~11s
(model load 9.3s + 4 warmup inferences). Sustained inference at
~360-400ms per frame. Live ad pods classified with `p_yes` 0.97-0.99
(confident ad) and content `p_yes` 0.0009-0.01 (confident no-ad).
Autonomous-mode `query_image` returns one of the 5 class names with
clean logit margins (e.g. MENU=22.74 vs second-best 14.26).
Zero `PROMPT_TOO_LONG`, zero IndexError, zero VLM-side hard-kills
over the first ~70 inferences. Parity with the standalone
`infer_vlm_fused.py` confirmed: on ad_0001 / nonad_0001 the
in-process VLMManager produces logits identical to the standalone
script (yes/no logits match to 3 decimals).

### Autonomous Mode VLM-Misclassification Traps (May 2026)

LFM2 classifies any screen showing the YouTube TV / Roku player overlay
(Description / Subscribe / cc / Up next buttons + a `\d+:\d{2}` time
marker) as **MENU**, regardless of whether the underlying video is
playing, paused, or stuck. This drove three production bugs in a
single 24-hour window. All three share a root cause — the
**MENU-action branch was acting on VLM alone, ignoring authoritative
playback signals (audio, screen activity, player-overlay markers)** —
and the fixes are layered on top of each other. Documented together
because the layering is non-obvious and easy to break.

**Pitfall #1 — Sign-in trap via `down + select`.**
Original code: `MENU → down + select`. On a video player with the
overlay visible, `down` navigates from the play button to the "Sign
in" CTA at the bottom of the overlay (always visible on Roku
YouTube), and `select` confirms it. That opens the Google sign-in
flow (keyboard / QR code / yt.be/activate code) — an unrecoverable
trap that costs ~2 min per occurrence before the keyboard-stuck
detector escapes. Observed loop: dozens of times per hour during
ad-data collection runs.

Fix: `_is_video_player_overlay()` veto BEFORE the down+select. Requires
BOTH an overlay-only keyword (`description` / `up next` / `autoplay` /
whole-token `cc`) AND a `\b\d{1,2}:\d{2}\b` time marker. All real menus
(home page, account picker, settings, signed-out prompt) lack the time
marker, so they pass through to the legitimate select action.

**Pitfall #2 — `back` escalation exits paused videos.**
First version of the overlay-veto used a 3-tier escalation:
veto-wait → veto-wait → **`back`** → veto-wait × 3 → full reset. `back`
on Roku/YouTube TV during a paused video EXITS to the recommendations
page — the opposite of what the user wants when a video is paused.
User reported live: "still fully paused. Autonomous mode was working
before we made the last few commits ... audio not playing and being on
Menu is a HUGE clue we are paused."

Fix: replace `back` escalation with `play_pause`. play_pause is
universally safe:
- Paused video → resumes (the case the user reported).
- Playing-but-silent video → pauses; next iteration's overlay-veto
  re-toggles. Self-correcting flap rather than permanent EXIT.
- Real menu → no-op (play_pause does nothing on a menu UI).
- Crucially: play_pause does NOT navigate UI, so the Sign-in trap
  from Pitfall #1 is structurally impossible from this action.

The 3-tier wait/back/reset ladder collapsed to a single decision based
on the audio + overlay signals.

**Pitfall #3 — audio-blind interruption of playing videos.**
With Pitfall #1 fixed but the audio guard absent, the MENU branch's
down+select still interrupted videos that were genuinely playing —
just because VLM misclassified the screen as MENU. Symptom: user
selects a music video / talking-head / lo-fi stream, ~30s later
autonomous mode selects a different video, repeat. The screen
content was changing under the visible-but-not-overlay UI, so the
overlay-veto didn't fire either. Same regression risk for the
PAUSED → play_pause branch when VLM misclassifies a playing frame
as PAUSED.

Fix: `_is_audio_flowing()` guard at the top of BOTH the `select`
and `play` branches. If HDMI-RX (`card4`) is receiving audio buffers
within the last 3s, the video is genuinely playing — veto every
interrupting action. Mirrors the precedent in `_is_screen_static()`
("static frames + audio flowing = music stream = don't pause").

**The full Action matrix that emerged from the three fixes:**

| Signal | Action |
|---|---|
| audio flowing | none (always wins) |
| no audio + overlay visible | `play_pause` (likely paused, resume) |
| no audio + no overlay (VLM PAUSED) | `play_pause` (normal pause-recovery) |
| no audio + no overlay (VLM MENU) | `down + select` (real menu) |

**Architectural invariant to preserve:** any action that could
interrupt or change playback (`down+select`, `back`, `play_pause`,
`launch`) MUST consult an authoritative playback signal first. The
hierarchy is:
1. `_is_audio_flowing()` — HDMI-RX audio is ground truth when
   HDMI-TX is connected.
2. `_is_video_player_overlay()` — disambiguates overlay-on-video
   vs real menu when audio is unavailable.
3. `_is_screen_static()` — the slow (3s sleep) frame-change check,
   used by the PLAYING-static branch but too expensive for the
   select/play branches.

VLM verdicts alone are not sufficient. The 99.2% non-ad-recall of
LFM2 is for the ad/non-ad classifier; the screen-state classifier
(`query_image`) is markedly noisier on overlay-heavy frames because
its training signal was the visible UI, which on YouTube TV is
indistinguishable from a menu by appearance alone.

**Files:** `src/autonomous_mode.py` — `_is_video_player_overlay`,
`_is_audio_flowing`, action dispatch in `_handle_screen_state`.

**OCR-dynamics guard (`_ocr_text_active`) — tried and rejected.**
An interim attempt to detect "screen is actively changing" via
difflib similarity between consecutive OCR-text snapshots was added
during Pitfall #3 debugging on the dev box (where audio guard can't
fire because HDMI-TX is disconnected and the audio pipeline never
opens). It worked on the music-video scenario but vetoed every
legitimate MENU verdict on any slight OCR drift — including paused
videos with timer ticks in the overlay — *exactly the regression
Pitfall #2 documented*. Removed in the play_pause-escalation
commit. If a future case demands a similar signal, the OCR
similarity check is structurally OK but it MUST be combined with a
positive-pause signal so it doesn't suppress recovery from real
pauses.

### Moonshine ASR migration + decision-engine retune (May 2026)

A cluster of coupled changes after live testing showed ASR was both too
slow and actively suppressing real ads.

**1. Self-deadlock froze ASR under load (root cause of "ASR does nothing").**
`ASRProcess._call_lock` was a non-reentrant `threading.Lock`. `transcribe()`
holds it and, on the hard-timeout escalation, calls `restart()` →
`stop()`/`start()`, which re-acquire it → deadlock. Under sustained load
every inference hit the timeout → restart → permanent wedge (symptom:
`inference_count` stuck, `killed_count=0`, empty transcript). Fix: RLock.
(VLM's `restart`/`kill`/`start` are lock-free, so VLMProcess never had this.)

**2. Engine swapped faster-whisper → Moonshine for sub-2s latency.** On the
thermally-throttled production box (4K passthrough pins the SoC at the
~85°C trip, cores throttled to 408MHz-1.6GHz — fan maxed, GPU also
throttled and exposes no usable compute provider so GPU offload is out)
faster-whisper's **fixed 30s-padded encoder** costs ~3.3-5s/window
*regardless of audio length* — a floor that streaming / smaller chunks
does NOT reduce (benchmarked: 5s=5.3s … 1s=4.0s). Moonshine
(`moonshine-voice`, ONNX, tiny-en) processes audio **proportionally**, so
a **2s window** is ~1.6s (p50, max <2s) even on wall-to-wall continuous
speech. Engine is env-selectable (`MINUS_ASR_ENGINE`); faster-whisper
stays the fallback for cool/idle hosts (more accurate: ~10/10 corpus vs
Moonshine ~8/10, and Moonshine misses some quiet/sung speech). Pinned to 3
cores (`MINUS_ASR_CPU_AFFINITY={3,4,5}`) to honour the "≤3 CPUs" budget.
Earlier "faster-whisper is 4-5s" benchmarks were CONTAMINATED by running
two whisper engines at once; single-engine faster-whisper is ~2s but
**starves OCR** (OCR's snapshot capture exceeds its 1s hard timeout → OCR
flap) and its p95 under full load is ~3.7s — so Moonshine wins on both
counts. Moonshine returns a result OBJECT, not a str — extract with
`str(raw)` (a `' '.join(raw)` assumption errored 100% of inferences once).

**3. ASR start-VETO removed; mid-block rescue gated on VLM weakening.** The
veto used to SUPPRESS a VLM-alone block when ASR heard speech with no
marketing markers. Live, it suppressed REAL ads VLM was sure about
(a Hotels.com ad at 80%, an insurance ad "What's important to you?" at
100% — the spoken copy just lacked explicit markers). ASR now NEVER
suppresses at start (a VLM-alone detection always blocks; ASR `confirm`
only upgrades the label). The mid-block product-placement rescue
(`source==vlm` + ASR veto ≥4s → force-stop) is now GATED on `ad_ratio <
0.5` so it can only end a block once **VLM itself has weakened** — a
confident visual detection is always trusted. The `tests/test_asr.py`
decision-engine cases were updated to assert the new behaviour
(`TestDecisionEngineASRGate`: VLM-alone blocks despite ASR veto with
base source `vlm`; `confirm` only flips `blocking_asr_confirmed` so the
DISPLAY label becomes `vlm+asr`); `test_get_status_keys` now matches the
default engine (`MINUS_ASR_ENGINE`, default `moonshine`) instead of
hard-coding `faster-whisper`.

**4. ASR markers expanded (~45 phrases + written-URL regex).**
`src/asr_keywords.py` gained urgency/CTA/guarantee/sale/code/pharma/
sponsor phrases plus a written-URL regex (`brand.com`) — the spoken
"dot com" regex missed the transcribed "Hotels.com". Markers now only
feed CONFIRM (veto is gone), so additions are low-risk.

**5. ASR block labels.** New `blocking_asr_confirmed` flag (kept SEPARATE
from `blocking_source`, which stays the base `ocr`/`vlm`/`both` for all
stop-logic). Display source → `ocr+asr` / `vlm+asr` / `both+asr`; overlay
headers `[ BLOCKING // OCR+ASR ]` etc.; set at start from the verdict and
**upgraded mid-block** when ASR later confirms (the common path). Also
fixed the prior `vlm+asr` source value that fell through to a blank
`[ BLOCKING ]` and didn't match the `== "vlm"` stop checks. Exposed as
`blocking_source_display` in `/api/status`.

**6. ASR audio without HDMI-TX (partial).** Audio pipeline falls back to
`fakesink` when HDMI-TX is disconnected (`AudioPassthrough._init_pipeline`)
and `start_display_pipeline` was decoupled to start audio+ASR even when
the display fails — so ASR runs off HDMI-RX with the TV off. BUT the
RK3588 HDMI-RX delivers **digital silence** when HDMI-TX is off (the
source mutes when its downstream sink/EDID drops: measured 0 non-zero
samples over 12s while `audio_present=1`), so live ASR has no real audio
without the TV. The `/api/asr/test` endpoint sidesteps this for testing.

**Files:** `src/asr_worker.py` (RLock, engine select, affinity, env
timeouts, Moonshine path), `src/asr.py` (2s window, env interval/window,
engine label, `test_transcribe`), `src/asr_keywords.py` (markers),
`minus.py` (veto removed, gated rescue, `blocking_asr_confirmed` +
`_display_source`, `sponsored`→STRONG, ASR enable/disable persistence,
audio/ASR startup decoupling), `src/audio.py` (fakesink fallback),
`src/webui.py` (`/api/asr/test`, pause cap 60→600), `src/templates/index.html`
+ `src/static/style.css` (ASR/OCR Live panels, ASR settings toggle).

### Sponsored promoted to strong; pause cap 60→600; misc UI (May 2026)

- **`sponsored` moved WEAK→STRONG** (`STRONG_AD_KEYWORD_NAMES`) per product
  decision so it decisively triggers ad blocking. Trade-off: STRONG
  overrides static-screen suppression, so a STATIC "Sponsored" promo/tile
  off a detected home screen can now block and hold — but bounded by
  `FROZEN_EARLY_SECONDS` (30s) and `MAX_BLOCKING_DURATION` (150s), and
  home-screen detection still guards home/browse tiles. Real *video* ads
  almost always also show a Skip/Visit-advertiser strong keyword, so the
  main *new* effect is catching static sponsored promos. Revert by moving
  `sponsored` back to WEAK.
- **Pause cap 60→600 min** in `/api/pause/<minutes>` validation, the
  Home custom-pause `max`, and the `pauseCustom()` JS clamp.
- **Replace-mode photos no longer stretched.** `_push_photo_background`
  decodes the photo, letterboxes it (fit-within + black bars) to the
  display dimensions, and re-encodes before POST — ustreamer's
  scale-to-composite is then aspect-preserving. Was sending the raw photo,
  which ustreamer stretched to full screen (distorting non-16:9 photos).
- **Vocabulary unpack fix.** `get_current_vocabulary()` unpacked a 4-tuple
  but `VOCABULARY_COMBINED` also has 5-tuple extended entries → "too many
  values to unpack (expected 4)" spamming during blocks. Now `[:4]`.

### Definitive-keyword single-frame fast-fire (May 2026)

**Symptom:** ad-break activation felt like ~3s. Measured from live logs
(e.g. a `skip in`/`sponsored` break): OCR first matched the keyword at
T, logged `OCR ad-detection pending dwell (1/2 frames)`, and the block
did not fire until T+~1s after the *second* OCR-matched frame. The
2-frame transience guard (`OCR_TRANSIENCE_MIN_FRAMES=2`) was costing a
full extra OCR cycle (~0.8-1.5s, capture cadence is noisy 120-2160ms)
on the most common ad-break case.

**Root cause:** the transience guard was added to reject *single-frame
OCR misreads* of an ad-keyword *word* appearing in show content — a
billboard reading "SKIP", a caption with "BUY", a movie title with
"Sponsored". But it applied uniformly to ALL keywords, including
*definitive ad-UI strings* (`skip in`, `skip ad`, `ad countdown`, `ad N
of M`, `ad with timestamp`, `visit advertiser`, `video will play after
ad`) that only ever render inside an active ad overlay and are
implausible as a 1-frame artifact. So the guard bought nothing but
latency for exactly the keywords that fire most often.

**Fix:** `DEFINITIVE_AD_KEYWORD_NAMES = STRONG_AD_KEYWORD_NAMES -
{'sponsored'}` (`minus.py.__init__`). In the OCR-loop transience guard,
`fast_fire` is now true when ANY matched keyword is in that set (in
addition to the pre-existing VLM-asserting / ASR-confirm corroboration
paths) → blocks on the FIRST matched frame, no dwell. The guard's cited
artifact strings ("SKIP"/"Sponsored"/"BUY") match **none** of the
definitive names, so no FP risk is reintroduced. `sponsored` is
deliberately EXCLUDED (it legitimately appears on home/promo tiles and
as show-content text — kept at the 2-frame dwell), as are all weak
keywords. Cuts the typical OCR ad-break activation under 2s.

**Verification:** `tests/test_asr.py::TestOCRTransienceGuard` updated to
mirror the new logic + 2 new cases (`test_fast_fire_on_definitive_ocr_keyword`,
`test_sponsored_alone_still_requires_dwell`); 68 tests pass. Deployed
and confirmed clean boot (OCR/ASR/VLM up, no errors).

**Files modified:** `minus.py` (`DEFINITIVE_AD_KEYWORD_NAMES` +
transience-guard `fast_fire`), `tests/test_asr.py`, CLAUDE.md.

### Moonshine 0.0.59 transcript leak — periodic worker restart (May 2026)

**Symptom:** after ~20 hours of continuous runtime, memory hit 88% and
the box was at risk of OOM. The leaker was PID 1643 — the ASR worker
subprocess — at **11.3 GB anonymous heap** (`smaps_rollup` PSS_Anon
11.19 GB) after 44,494 inferences. No timeouts, no kills, no
restart_count growth — just steady inference with monotonically growing
RSS.

**Root cause:** `moonshine_voice 0.0.59`'s
`Transcriber.transcribe_without_streaming` allocates a `TranscriptC`
struct (lines + per-line text/audio_data/words + per-word text) via the
C library on every call and **never exposes a `moonshine_free_transcript`
symbol** to release it. Measured leak rate: **~3.08 MB per inference** on
real speech (44k iters × 3MB ≈ 132 GB; the process was capped at 11 GB
by physical RAM pressure). The Python `_parse_transcript` does deep-copy
the fields into Python objects (`ctypes.string_at` for text,
`list(audio_array)` for audio) so the source data isn't aliased — the
sub-structures are pure leak.

Three mitigations were probed empirically before settling on the right
one (`tests/asr_corpus/` scripts in /tmp during diagnosis):

1. **Manual deep free** via `moonshine_free(addr)` (= `libc.free`) on
   each sub-pointer → `free(): invalid pointer` crash on the first
   `audio_data` free. The library uses C++ `new` for these
   allocations, not libc `malloc`, so libc `free` is undefined
   behaviour. The exported symbol list confirms this: there are
   per-type free helpers for `intent_matches`, `tts_synthesizer`,
   `grapheme_to_phonemizer`, `stream`, `tensor`, etc. — but
   conspicuously no `moonshine_free_transcript`. Upstream forgot to
   expose it.
2. **Top-level pointer free** only (`libc.free(out_transcript)`,
   leaking the sub-structures but recovering the outer struct) → same
   `free(): invalid pointer` crash. Even the TOP struct isn't a libc
   malloc.
3. **Periodic `del Transcriber + gc.collect()`** (recreate the
   Transcriber object every N calls, hoping its destructor releases
   accumulated state) → measured `RSS=320.7 MB` *after* 25 leaky iters
   then `RSS=320.7 MB` after `del+gc` and `RSS=335.1 MB` after
   recreate. The destructor does NOT release the leaked memory — the
   leak lives in **globally-cached ONNX runtime / libmoonshine.so
   state**, not in the Transcriber instance. So Transcriber recreation
   is useless.

**Fix (the only viable one without forking upstream):**
parent-side **periodic subprocess restart** in `ASRProcess`. Added
`RESTART_AFTER_INFERENCES = int(os.environ.get(
'MINUS_ASR_RESTART_AFTER_INFERENCES', '500'))` and a counter in
`__init__`. In the `'ok'` branch of `transcribe()` the counter
increments; when it reaches the threshold, we log + call
`self.restart()` (which already exists for hard-timeout escalation,
`RLock`-safe). The OS frees ALL process memory on exit — the only
mechanism that reliably reclaims the leak.

At the default 500 inferences: with the production cadence of ~1.7s/
inference + ~1.5s gap = ~3.2s per cycle, restarts every ~27 min.
~1.5 GB of leaked memory recovered per cycle. Restart costs ~1.4 s of
ASR downtime (Moonshine tiny.en reload from disk cache); ASR is a
confirm-only signal on top of OCR+VLM so the gap doesn't drop any
ad blocks. Override via `MINUS_ASR_RESTART_AFTER_INFERENCES` (set 0 to
disable when upstream is fixed).

**Verification** (`/tmp/test_fix_integration.py`, threshold=5):
- 18 inferences → 3 restarts (at 5, 10, 15) as expected.
- RSS sawtooth: 228 → 236 (4 iters, leaked +8 MB) → **RESTART** → 226
  → 238 (5 iters, +12 MB) → **RESTART** → 225 → 236 (+11 MB) →
  **RESTART** → 227 → 244 (+17 MB).
- All 18 transcribes returned `ok`; ASR keeps working through every
  restart (~1.2-1.5 s reload, well under SOFT_TIMEOUT=4 s).
- Memory is bounded (sawtooth, no monotonic growth) — the actual user
  requirement.

**Files modified:** `src/asr_worker.py` (`RESTART_AFTER_INFERENCES`
constant + `_inferences_since_restart` counter + restart trigger in
`transcribe()`), CLAUDE.md.

### 60fps pipeline fix — ustreamer zero-copy VPU + RGA (Jul 2026)

The HDMI pipeline capped at **27-30fps** (dipping to 23) at every
resolution despite the VPU being rated 4K60. Diagnosed with ustreamer's
`--perf` logging + standalone MPP/RGA benchmarks; fixed on the
`fix/fps-early-buffer-release` branch of `ustreamer-garagehq`
(commit `32ab470`), deployed to `/home/radxa/ustreamer-patched`
(backup: `ustreamer-patched.bak-pre-60fps`). Measured: **29.4 → 60.0 fps**
served at 1080p60 NV24, grab-to-expose latency **110ms → 17ms**.

Three stacked root causes, all in the ustreamer fork:
1. **Deferred V4L2 buffer release starved capture to half rate.** A
   worker's capture buffer was only decref'd when the dispatcher next
   picked that worker (~100ms later), permanently pinning 4 of 5 V4L2
   buffers — `captured_fps` dropped 60→30 the moment a client attached.
   Buffers are now released inside the worker right after encode.
2. **The CPU touched every pixel of UNCACHED V4L2 mmap memory.** The
   per-frame copy/convert into the MPP buffer read uncached DMA memory:
   ~90ms per 1080p NV24 frame (vs ~5ms actual VPU encode; benchmarked
   213fps pure VPU at 1080p). Replaced with hardware paths:
   - **NV12 sources (4K)**: V4L2 DMABUF imported directly into MPP
     (`MPP_BUFFER_TYPE_EXT_DMA`) — true zero-copy, VPU reads via IOMMU.
     `dma_export` enabled for the MPP encoder type in `stream.c`.
   - **NV24 sources (1080p YCbCr 4:4:4, e.g. Roku)**: two RGA passes —
     Y plane copied as an RGBA-reinterpreted image; the NV24 UV plane
     viewed as RGBA (each px = U0V0U1V1) downscaled 2× == exactly the
     NV12 UV plane layout. ~2.4ms/frame. NOTE: RGA has no native
     YUV444SP support, and feeding YUV444SP/BGR888 directly to the JPEG
     VEPU produces CORRUPT bitstreams (tested — the old "only NV12 is
     reliable" comment was right); the RGA plane tricks are the way.
   - **BGR24 sources (Google TV)**: RGA `imcvtcolor` (~2ms, benchmarked
     478fps).
   Import handles are cached per fd; ANY hardware-path failure logs once
   and permanently falls back to the old CPU path (`zero_copy_broken` /
   `rga_broken`).
3. Minor: per-frame full-buffer `memset` removed (cleared once at
   configure); releaser poll 5ms → 1ms.

**Feature preservation (verified live):** blocking composite and the
notification overlay REQUIRE CPU pixel access, so the encoder checks
`us_blocking_is_enabled_fast()` and the new `us_overlay_is_enabled()`
per frame and routes through the legacy CPU path while either is active
(fps drops to CPU rates during a block — acceptable, the screen shows
the vocab card). Verified: overlay text renders, full blocking overlay
(header/vocab/preview/stats/OCR-snippet) renders, path switching is
clean, `/snapshot` + `/snapshot/raw` + OCR + VLM all work, stream
returns to 60fps after the block ends.

**4K resilience validation (no 4K HDMI source available — validated
with real 4K content in dma-heap buffers, the same memory class as
V4L2 capture buffers):**
- **Encoder**: the exact `MPP_BUFFER_TYPE_EXT_DMA` import path was
  exercised at 3840×2160 with 16 real (upscaled-capture) NV12 frames in
  CMA dmabufs: import OK, output JPEGs valid (~380KB @q80),
  single-context 18.0ms/frame (55.4fps), **4-worker sustained aggregate
  66.7fps over 10s** — 4K60 encode headroom confirmed
  (`scratchpad/test4k_extdma.c` pattern).
- **Display decode**: `mppjpegdec` at 4K does **120fps** to fakesink;
  neutral `videobalance` passes through at 120fps; **non-neutral
  videobalance (sat=1.4/bri=0.2, the saved user settings) caps 4K at
  ~56fps** — the only remaining sub-60 link, CPU per-pixel.
- **Dead ends tested**: GStreamer-GL (`glupload!glcolorbalance`)
  SIGSEGVs on this BSP/Mali stack; VOP2 exposes no BCSH/saturation DRM
  props (COLOR_ENCODING/COLOR_RANGE only); RGA has no saturation op;
  feeding YUV444SP/BGR888 straight to the JPEG VEPU corrupts bitstreams.
- **Mitigation shipped**: `ad_blocker._init_pipeline` now OMITS
  `videobalance` entirely when saved color settings are neutral (pure
  zero-copy DMABuf mppjpegdec→kmssink, 120fps headroom);
  `set_color_settings` rebuilds the pipeline when settings cross
  neutral↔non-neutral.
- **Open lead for full 60fps at 4K WITH color correction** (needs the
  TV attached to verify visually): the saved sat=1.4/bri=0.2 looks like
  compensation for a **limited-range YCbCr HDMI capture being encoded
  into JPEG (full-range container) and displayed as full-range** —
  i.e., washed-out picture. If so, the proper fix is range expansion,
  not saturation: either set the kmssink plane's `COLOR_RANGE` prop /
  caps colorimetry, or do the limited→full range expansion in the
  ustreamer RGA pass (RGA supports range conversion in `imcvtcolor`
  modes). Then color settings can go neutral → zero-copy display path.
  Test with the TV: `modetest -M rockchip` plane COLOR_RANGE, compare
  picture with videobalance neutral.

**Display-side follow-up — kmssink double-vsync (Jul 2026, TV attached):**
with the TV connected (CRTC genuinely at 3840x2160p60) the display
pipeline pinned at **exactly 30.00fps** while the encoder served 60 to
both clients and a parallel decode-to-fakesink consumer got 60.
Isolated to kmssink: GST_DEBUG=kmssink:7 showed dmabuf import + SetPlane
taking microseconds but ~33ms between frames. Two measured facts:
(1) `drmWaitVBlank` fires at a clean 60Hz; (2) **the rockchip BSP's
legacy `drmModeSetPlane` ioctl BLOCKS until the flip latches at vblank**
(measured 16.67ms/call in a bare-DRM loop). kmssink doesn't know that
and does its own internal vsync wait after SetPlane → 2 vblanks/frame →
30fps. Fix: `skip-vsync=true` on kmssink (upstream property since 1.22,
description literally says "avoid double vsync") — frame pacing is
still vsync-locked by the blocking SetPlane, so no tearing. Applied to
all three kmssink pipelines in `ad_blocker.py` (main / no-signal /
loading). Verified live: **fps_display 30.00 → 59.99** at 4K60 output.
Notes from the same session: videobalance was NOT the 30fps culprit at
1080p (passthrough test stayed at 30); the conditional-videobalance
omission still matters for 4K sources (videobalance @4K ~56fps ceiling).
VOP2 debugfs (`/sys/kernel/debug/dri/0/summary`) shows the video on the
Esmart3 plane as BT.601/Limited — matching the limited-range data the
RGA path preserves verbatim, so neutral color settings should now look
correct; the old sat=1.4/bri=0.2 likely compensated for the pre-RGA CPU
conversion path.

### Roku sticky-disconnect + autonomous Shorts/YouTube-TV avoidance (Jul 2026)

**1. Roku connection died ~daily and never came back (sticky disconnect).**
`RokuController._reconnect_loop` only called `connect()` when the live
HTTP health check was *currently failing*. Roku devices reboot nightly
(update check): during the reboot the check failed, `_connected` was
cleared, the single immediate reconnect attempt failed while the device
was still booting — and once the Roku came back the health check passed
again, so `connect()` was never called and the controller stayed
"disconnected" until the user reconnected via the web UI. A transient
keypress error (`_send_keypress` sets `_connected = False`) hit the same
trap. Fixes in `src/roku.py`:
- Reconnect tick (`_reconnect_tick`, split out of the loop for tests) now
  reconnects when **either** the health check fails **or** the
  `_connected` flag was dropped.
- **DHCP-change recovery:** every 3rd consecutive failure the loop falls
  back to SSDP rediscovery (`_rediscover_and_connect`), preferring the
  same physical device by **serial number** (a different-serial Roku is
  never hijacked). On success at a new IP it fires the new
  `set_ip_change_callback`, which `minus.py._persist_roku_ip` uses to
  save the address to `~/.minus_device_config.json` for the next restart.
- **Startup resilience:** `_start_roku_connection` now wires autonomous
  mode + the ip-change callback immediately after constructing the
  controller, and on total startup failure arms
  `start_monitoring(saved_ip)` so the background loop keeps retrying /
  rescanning until the Roku appears. User-initiated `disconnect()` (web
  UI device switch) permanently stops the loop — auto-reconnect never
  fights an explicit disconnect. `disconnect()` also now joins the
  reconnect thread *outside* the lock (the loop's `connect()` takes the
  same lock; joining under it could orphan a thread that reconnected
  after the user disconnected).
- Tests: `tests/test_roku_reconnect.py` (12 tests incl. the
  dropped-flag regression and serial-match rediscovery).

**2. Autonomous mode kept landing in YouTube Shorts and on YouTube TV
upsell prompts.** Fixes in `src/autonomous_mode.py`:
- **YouTube TV prompts:** new `_is_youtube_tv_prompt()` +
  `YOUTUBE_TV_PROMPT_KEYWORDS`/`_MARKERS`. Exact promo phrases
  ("cable-free live tv", "try youtube tv", …) OR `"youtube tv"` +
  a promo marker (free trial / per month / sign up / live tv) — plain
  "youtube tv" alone is NOT enough (video titles contain it). Dispatch
  (right after the keyboard-stuck check) presses Back, escalating to
  `_full_reset_to_youtube()` after `_STUCK_THRESHOLD` consecutive hits.
  Suppressed while the ad blocker is visible (a YouTube TV *ad* belongs
  to ad handling; Back mid-ad could exit the video).
- **Shorts:** the old OCR duration check was a tautology
  (`any(c.isdigit() and ':' in combined for c in combined)` ≡ "any digit
  + any colon anywhere") so `@handle + subscribe + no duration` almost
  never fired. Now a real `\b\d{1,2}:\d{2}\b` time-marker regex. Added a
  second, OCR-independent signal: `_is_vertical_video_frame()` detects
  the pillarboxed 9:16 format (outer 22% side bands dark `<25` AND flat
  `std<6` — true bars are flat; gray-converted noise in a dark movie
  scene is ~8 — plus a center ≥4× brighter), **double-confirmed on two
  frames ~2s apart** so a single dark movie shot can't trigger a Back
  press on a playing video. Skipped while blocking is active. The Shorts
  escape also gained the same stuck-escalation → full reset. Existing
  home-screen "push past the Shorts row" navigation is unchanged.
- Tests: `TestYouTubeTVPromptDetection` + `TestShortsDetection` in
  `tests/test_autonomous_mode.py` (also repaired the stale
  `test_is_audio_flowing_with_ad_blocker`, broken since the June
  `recent_level` silence-threshold change).

**3. 48-hour live-watch follow-ups (Jul 2026).** A monitored soak run
immediately after the fixes above surfaced four more trap classes, all
the same root shape — **an interrupting action firing on a VLM
misclassification without consulting an authoritative signal** (the
CLAUDE.md invariant). Every VLM-driven interrupting action is now
guarded; each was observed live before being fixed:
- **"Sign in to YouTube TV" activation screen** (Enter this code /
  tv.youtube.com/start / scan with your phone) matched NO detector and
  stalled a session 40 min. Its markers were added to
  `KEYBOARD_STUCK_KEYWORDS` (incl. OCR-merged `enterthis code`) and
  `'sign in to youtube tv'` to `YOUTUBE_TV_PROMPT_KEYWORDS`.
- **MENU skip-loop watchdog** (`_menu_skip_count`,
  `_MENU_SKIP_ESCAPE_AT=5`): the "MENU select skipped — home OCR not
  confirmed" guard could no-op forever on unrecognized dead-end
  screens. After 5 consecutive skips (~3 min, no audio/overlay/home) it
  runs `_escape_stuck_state()`. Verified live twice: dead-end → escape →
  ECP home detect → relaunch → new video in ~3 min.
- **SCREENSAVER→launch guarded** by audio + a Roku ECP re-check
  (screensaver absent + app 837 ⇒ dark playing frame): unguarded, VLM's
  SCREENSAVER misreads of dark scenes ran `_wake_device` (power+home) +
  relaunch, killing playback every ~2 min.
- **Keyboard-stuck escape audio-guarded**: the character-pattern
  heuristic (many short OCR fragments + digits) false-positived on a
  playing video and the 4-Back escape exited YouTube. Real sign-in
  screens are silent, so audio flowing vetoes the escape.
- **DIALOG→dismiss audio-guarded** (the last unguarded action): VLM
  misread playing frames as DIALOG twice in one hour; each `back` exited
  the video into a ~3-min watchdog recovery. Real blocking dialogs pause
  playback (no audio) and are still dismissed; a banner over a playing
  video just lingers.
- Also: Roku ECP `202 Accepted` now counts as keypress/launch success
  (was logged as failure under load).
- **Stuck-on-Roku-home loop (Jul 2026, 50-min stall):** Roku OS 15.2
  reports the home screen as `<app id="native-ui">`; `get_active_app_id`'s
  digits-only regex (`id="(\d+)"`) returned None → callers treated it as
  "query failed, don't interfere" → the authoritative unguarded ECP
  recovery never fired. The OCR home-tile fallback DID detect home every
  cycle, but the Roku home screen AUTOPLAYS promo audio in its side pane,
  so the audio-flowing veto blocked the relaunch indefinitely (92
  vetoed matches/hour observed). Two fixes: (a) regex now accepts any id
  (`id="([^"]+)"`) so ECP is authoritative again; (b) the OCR fallback
  escalates past the audio veto after `_ROKU_HOME_VETO_ESCAPE_AT` (6)
  consecutive vetoed matches (~3 min) — persistent home-tile OCR for
  minutes IS the home screen, promo audio notwithstanding. Tests:
  `TestActiveAppIdParsing` (roku), streak covered by dispatch tests.
Tests: `TestYouTubeTVActivationScreen`, `TestMenuSkipWatchdog`,
`TestScreensaverLaunchGuards`, `TestKeyboardStuckAudioGuard`,
`TestDialogDismissAudioGuard` in `tests/test_autonomous_mode.py`;
`TestKeypressStatusCodes` in `tests/test_roku_reconnect.py`.

### Axera libaxcl_*.so broken symlinks — VLM dead at boot (May 2026)

**Symptom:** VLM marked `disabled` in `/api/health.subsystems.vlm`
while `vlm_disabled: false` in `/api/status` (i.e., VLM was supposed
to be ENABLED but never loaded). Every VLM load attempt in the
journal showed:

```
[E] Failed to load LFM2.5-VL: cannot load library 'libaxcl_rt.so.1':
libaxcl_pkg.so: cannot open shared object file: No such file or directory.
```

Three load attempts on three different worker PIDs over ~3 minutes
(graceful-degradation retry path), then VLM gave up and the service
ran OCR-only for 20+ hours.

**Root cause:** `/usr/lib/axcl/` has the pattern `libaxcl_NAME.so` →
`libaxcl_NAME.so.1` → `libaxcl_NAME.so.1.0.0` (three dots). The
`.so.1` symlinks (radxa-owned, Mar 26) point correctly at the
`.so.1.0.0` file. But the top-level `.so` symlinks (root-owned, May
29 17:34 — created in some prior provisioning step) all point at
non-existent `libaxcl_NAME.so.1.0` (TWO dots, missing). All 17
top-level symlinks were broken (`libaxcl_comm.so`, `libaxcl_pkg.so`,
`libaxcl_rt.so`, etc.).

`ldconfig` couldn't stat them → the linker cache had no `libaxcl_*.so`
entries → `dlopen("libaxcl_rt.so.1")` couldn't resolve its
`DT_NEEDED libaxcl_pkg.so` → VLM load fails. The `.so.1` file IS
present and resolvable directly, but dlopen still has to resolve the
DT_NEEDED chain, and the cache miss on `libaxcl_pkg.so` (because of
the broken `libaxcl_pkg.so` symlink) breaks the load.

**Fix:** re-point each of the 17 broken `libaxcl_*.so` symlinks to
the working `libaxcl_*.so.1`, then `ldconfig` to rebuild the cache.
After: `ldd /usr/lib/axcl/libaxcl_rt.so.1` resolves every dependency,
and VLM loads cleanly on restart (10.1 s model load + 1.6 s warmup).

```bash
sudo bash -c '
for f in /usr/lib/axcl/libaxcl_*.so; do
  if [ -L "$f" ] && [ ! -e "$f" ]; then
    ln -sfn "$(basename "$f").1" "$f"
  fi
done
ldconfig'
```

If the box ever gets re-imaged from the same source the symlinks may
break again; the snippet above is idempotent.
