# Minus Features

## Overview

Minus is an HDMI passthrough device that detects and blocks advertisements in real-time using dual NPU machine learning.

## Core Features

### Ad Detection

**Dual-NPU ML Pipeline:**
- **PaddleOCR** on RK3588 NPU (~400ms per frame) — detects text-based ad indicators
- **[minus-v0.1](https://huggingface.co/TheGarageDev/Minus-v0.1)** (fine-tuned LFM2.5-VL-450M) on Axera LLM 8850 NPU (~0.37s per frame, prefill-only on 16 fused decoder layers, no decode loop) — visual content analysis
- Both workers run in dedicated **subprocesses** (`src/ocr_worker.py`, `src/vlm_worker.py`) with hard timeouts so a stuck NPU inference can never freeze the detection loop. The worker processes ship with warmup inferences, keepalive pings, and soft/hard timeout escalation.

**Detection Methods:**
- OCR keyword matching (Skip, Ad, Advertisement, timestamp patterns, etc.)
- OCR self-overlay exclusions: Minus' own on-screen notifications (e.g. "Ad skipping enabled") are excluded from ad detection so the system does not self-trigger
- VLM scene understanding with confidence scoring, `max_new_tokens` cap to keep short-answer prompts short (the root cause of long inference latency was descriptive responses, not NPU pathology — see `docs/VLM_NPU_DEGRADATION.md`)
- Weighted voting with anti-waffle protection
- Home screen detection to prevent false positives (with `AD_ONLY_KEYWORDS` guard to avoid home-screen triggers during ads that also contain "Shorts"/"Search" text)
- Transition frame detection for smoother blocking

### Blocking Overlay

**When ads are detected:**
- Full-screen blocking overlay at 60fps
- Pixelated background from pre-ad content (optional)
- Live preview window showing blocked content — **desaturated** (greyscale chroma) by default so the Spanish overlay is more eye-catching than the ad
- Spanish vocabulary practice during blocks
- Debug dashboard with stats

**Overlay Features:**
- Smooth animations (0.3s start, 0.25s end)
- Multi-color text rendering via FreeType — Spanish word cycles through a 10-color YUV palette (purple / magenta / cyan / lime / amber / pink / mint / sky / coral / teal) each rotation so the overlay doesn't feel static
- Hardware-accelerated compositing in ustreamer MPP encoder
- Configurable preview window, debug overlay, and greyscale-preview flag (all toggleable in the web UI Settings tab)

**Block-duration falloff:**
Minimum block time starts at 3.0s for the first ad in a sequence and shrinks by 0.5s on each consecutive ad (3.0 → 2.5 → 2.0 → 1.5 → 1.0s). Floor is 1.5s when OCR+VLM both agree, 1.0s when OCR is alone. Counter resets after 30s without a block. The goal: unblock as soon as the ad actually ends instead of holding a fixed 3-4s every time. Can be disabled from Settings → Blocking Optimizations.

**HDMI reconnect grace period:**
For 90 seconds after the TV reconnects (detected by the health monitor), ad blocking is suppressed so the user can navigate menus without overlays jumping in. Toggleable from Settings → Blocking Optimizations.

**Replacement modes (Settings → Replace):**
The blocking overlay rolls a single *replacement mode* at the start of each ad break and sticks with it for the whole break (plus a 30-second cooldown so back-to-back ads reuse the same style). Available kinds:
- **Vocabulary** — Spanish words with 1-2 example sentences (default)
- **Did You Know?** — short trivia cards (`src/facts.py`)
- **Haikus** — classical + modern short poems (`src/haiku.py`)
- **Photo Screensaver** — cycles user-uploaded photos as the blocking background every 5 seconds. Photos are uploaded via the web UI Settings → Replace tab, stored under `~/.minus_media/photos/` (re-encoded to 1920px max, JPEG quality 85, capped at 200 photos / 200 MB).

The overlay also gets:
- **Pixelated pre-ad background** — heavy pixelation + 60% darken of the screen as it looked ~6s before the ad. Gives context without competing for attention. Falls back to a dark radial gradient when the snapshot buffer is empty (e.g. within the first seconds after a restart).
- **Ad countdown bar** — when OCR reads "Ad 0:30" or "Ad 10", a `[###...] 12s` bar drains in real time so the user sees how long is left.
- **Audio-reactive visualizer** — a short ASCII ramp bar (`' .,-;+ox*#@'`) driven by RMS of the alsasrc capture in `src/audio.py`. Bounded 16-sample deque — no memory growth over a 24h run.
- **Rotating Spanish word color** — 10-color YUV palette cycled per rotation.

### Audio Passthrough

**Features:**
- Full HDMI audio passthrough (48kHz stereo)
- Instant mute/unmute during ad blocking
- Automatic A/V sync reset every 45 minutes
- Silent keepalive to prevent pipeline stalls
- Exponential backoff restart on failures

### Streaming Device Remote Control

**Supported devices:**
- **Fire TV / Amazon** — ADB over Wi-Fi (`src/fire_tv.py`)
- **Roku** — ECP over HTTP (`src/roku.py`)
- **Google TV / Android TV** — ADB Wireless debugging (port is dynamic; users paste the `IP:PORT` printed on the TV)
- **Generic / No remote** — ad blocking still works, only skip-automation is disabled

**Common behavior:**
- Discovery (network scan) where the protocol supports it
- Persistent device config at `~/.minus_device_config.json`
- Auto-reconnect on drops
- Skip button detection via OCR
- Guided setup flow with overlay notifications

**Web UI skip** (`POST /api/blocking/skip`) is device-agnostic: the handler calls `minus.try_skip_ad()` which dispatches to whichever controller is connected. There is no Fire TV-only path.

### Autonomous Mode

**VLM-guided YouTube playback** — keeps content rolling overnight so OCR/VLM keep generating training data.

- Device-agnostic: runs against any of the streaming controllers above
- Configurable schedule (start/end hours, or 24/7 mode)
- VLM screen-state classification every 2 minutes (`PLAYING` / `PAUSED` / `DIALOG` / `MENU` / `SCREENSAVER`)
- OCR screen-state pre-check before VLM: explicit keyword lists for login screens, home/browse rows, ad banners (`AD_ONLY_KEYWORDS`), and a "signed-out" fallback. OCR is cheaper and more accurate on static UI chrome than VLM.
- Roku-specific active-app query via ECP — more reliable than VLM for "YouTube closed" or "Roku City screensaver on top of YouTube"
- Pause detection combines dHash frame comparison with audio state:
  - Static frames + audio flowing = music stream, NOT paused (no action)
  - Static frames + no audio + pipeline healthy = truly paused (send play)
  - Static frames + audio pipeline unavailable (e.g. display disconnected) = suspicious; increments a `PERSISTENT_STATIC_LIMIT` counter. Escalates to STUCK only after ~5–7 minutes of unchanged frames, which prevents misfiring on live streams whose output path is temporarily down.
- Dismiss action sends a single `back` (previously `select + play_pause`, which could confirm unwanted buttons and toggle the player)
- Stats tracking: videos played, ads detected, ads skipped, session duration
- Settings persist at `~/.minus_autonomous_mode.json`
- Web UI controls: enable/disable, schedule time selectors, 24/7 checkbox, "start now" manual override

**Commands (all controllers):**
- Navigation: up, down, left, right, select, back, home
- Media: play, pause, fast_forward, rewind
- Volume: volume_up, volume_down, mute
- Power: power, wakeup, sleep
- Google Assistant (Google TV only): `assistant`

### Web UI

**Remote Monitoring:**
- Live MJPEG video stream
- Status display (blocking state, FPS, HDMI info, uptime)
- Detection history with timestamps
- Log viewer

**Controls:**
- Pause ad blocking (1/2/5/10 minute presets)
- Toggle preview window and debug dashboard
- Test trigger/stop blocking
- Fire TV skip button
- A/V sync reset

**Video Color Controls:**
- Real-time saturation adjustment (0.5-1.5)
- Brightness adjustment (-0.5 to 0.5)
- Contrast adjustment (0.5-1.5)
- Hue adjustment (-0.5 to 0.5)

### Health Monitoring

**Automatic Recovery:**
- HDMI signal detection and recovery
- HDMI PHY reinit via DPMS cycle on TV reconnect (works around silent PHY stalls after TV power cycles)
- Adaptive HDMI bandwidth fallback: when 4K@60Hz RGB/4:4:4 fails, we fall back to 4:2:0 (half bandwidth) automatically. See `docs/ARCHITECTURE.md` and `src/drm.py`.
- ustreamer health checks with restart
- After 3+ consecutive video pipeline failures we also kill ustreamer to force a clean MPP decoder state (fixes the "stuck after brief HDMI drop" class of bugs)
- Video pipeline watchdog
- Audio pipeline watchdog with **ALSA zombie detection**: if the GStreamer audio pipeline thread dies but the ALSA device still reports RUNNING, we detect via `/proc/asound` owner-TID check and restart. The detector understands that `owner_pid` is a thread ID, not a process ID.
- Memory monitoring with cleanup
- VLM rolling P95 latency check: if P95 of the last 10 inferences exceeds 3s, we restart the worker. If a second trigger follows quickly, we escalate to a deep restart with a longer NPU-release backoff.
- VLM degradation to OCR-only mode after consecutive hard timeouts

**Graceful Degradation:**
- OCR init: 3 retries with 2s delay, continues without OCR if all fail
- VLM model load: 3 retries with 5s delay, continues without VLM if all fail
- OCR + VLM status badges in web UI (Ready/Disabled/Failed)
- System continues running with whatever subsystems loaded

**Status Tracking:**
- FPS monitoring (logged every 60s)
- Full status logged every 5 minutes
- Startup grace period for VLM loading

## Spanish Vocabulary Practice

**Content:**
- 500+ intermediate-level words and phrases
- Common verbs, nouns, adjectives, expressions
- False friends and subjunctive triggers
- Pronunciation guides
- Example sentences in context

**Display:**
- Random vocabulary rotation every 11-15 seconds
- Purple Spanish word (IBM Plex Mono Bold)
- White translation (DejaVu Sans Bold)
- Gray pronunciation and example

## Screenshot Collection

**Training Data:**
- `screenshots/ads/` - OCR-detected ads
- `screenshots/non_ads/` - User paused (false positives). For **VLM-only blocks**, saves the VLM-triggering frame (cached on each `is_ad=True` verdict) rather than the current frame — see *User-Feedback Cooldown* below.
- `screenshots/vlm_spastic/` - VLM uncertainty cases
- `screenshots/static/` - Static screen suppression

**User-Feedback Cooldown (VLM-only blocks):**
Pausing ad-blocking during a VLM-only block tells Minus the VLM misclassified the frame. Response: save the VLM-triggering frame to `non_ads/` and put VLM inference on a **5-min cooldown** so the same content can't immediately re-trigger if the user resumes early. OCR keeps running. Only fires when `blocking_source == "vlm"` strictly — `"ocr"` and `"both"` blocks use the existing save-current-frame behaviour. Status exposed at `/api/status` as `vlm_user_paused` + `vlm_user_pause_remaining`.

**Quality Filtering (all categories):**
- dHash perceptual deduplication (hamming distance < 10 bits = ~85% similar)
- Black/blank frame rejection (mean brightness < 15)
- Solid-color frame rejection (std deviation < 10)
- Rate limiting (5s minimum between saves per category)
- Rolling dedup window (last 200 hashes per category)
- Configurable max screenshots per folder with automatic truncation

**Review System (Tinder-style):**
- Swipe-based screenshot classification in web UI
- Per-category review buttons (Ads, Non-Ads, VLM Spastic, Static)
- Swipe right = approve/classify as ad, swipe left = reclassify/not ad
- 3-card visual stack with fly-out animations
- Progress tracking (oldest unreviewed first)
- Undo support (Ctrl+Z)
- Keyboard shortcuts (arrow keys, Escape)

## Configuration

**Command Line Options:**
```bash
--device /dev/video1      # Custom capture device
--ocr-timeout 1.5         # OCR timeout in seconds
--max-screenshots 100     # Keep N recent screenshots
--check-signal            # Just check HDMI signal and exit
--connector-id 231        # DRM connector ID
--plane-id 192            # DRM plane ID
--webui-port 80           # Web UI port
```

**Auto-Detection:**
- HDMI output (HDMI-A-1 or HDMI-A-2)
- Display resolution (4K or 1080p)
- NV12-capable DRM plane
- Audio output device

## Systemd Service

**Installation:**
```bash
sudo ./install.sh
```

**Service Features:**
- Starts on boot
- Conflicts with display managers
- Auto-restart on crash
- Journal logging
