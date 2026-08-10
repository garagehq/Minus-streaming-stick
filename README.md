# Minus

Minus is an open-source HDMI box that sits between your streaming device and your TV. It watches the video feed, detects ads with ML models running locally on its own NPUs, then mutes and covers them until your show comes back. Instead of the ad you get something useful, like a Spanish vocabulary card.

Nothing leaves your living room. There is no cloud service, no subscription, and no account. The OCR, the vision model, and the speech recognition all run on the device.

![Minus blocking an ad and showing a Spanish vocabulary card instead](docs/images/blocking-overlay.jpg)

*An ad break as seen through Minus. The ad is muted and covered by a vocabulary card, with a small greyscale preview in the corner so you can tell when it's over.*

## How it works

Minus is a 4K@60fps HDMI passthrough. Your Roku, Fire TV, or Google TV plugs into Minus, and Minus plugs into your TV. Three detectors watch the stream and vote on whether an ad is playing:

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│   HDMI IN    │────▶│     ustreamer      │────▶│  HDMI OUT to TV     │
│ (Roku, etc.) │     │ (MPP hw encoding)  │     │  (DRM/KMS, 60fps)   │
└──────────────┘     └────────┬───────────┘     └─────────────────────┘
                              │ snapshots + audio
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
     ┌────────────────┐ ┌──────────────┐ ┌──────────────┐
     │  PaddleOCR     │ │  minus-v0.1  │ │  Moonshine   │
     │  RK3588 NPU    │ │  (VLM)       │ │  ASR (CPU)   │
     │  reads ad UI   │ │  Axera NPU   │ │  hears ad    │
     │  text ~400ms   │ │  ~370ms      │ │  language    │
     └────────────────┘ └──────────────┘ └──────────────┘
              └───────────────┼────────────────┘
                              ▼
                    blocking decision engine
              (mute + overlay, typically < 2s to react)
```

- OCR (PaddleOCR on the RK3588's NPU) reads on-screen ad UI like "Skip in 5", "Ad 2 of 3", and countdown timers, including the character misreads OCR makes on real TVs.
- [minus-v0.1](https://huggingface.co/TheGarageDev/Minus-v0.1), a 450M-parameter vision-language model fine-tuned for this exact job, looks at the raw frame and answers "is this an advertisement?" in about 370ms on an Axera AX8850 NPU. It needs no text on screen to work.
- ASR (Moonshine tiny-en on CPU) transcribes the audio and listens for marketing language as a confirmation signal.

When the votes say "ad", audio mutes immediately and the screen switches to an overlay rendered at 60fps inside the hardware JPEG encoder (no desktop stack, no X11). When the ad ends, usually within a second, the show comes back.

## The model

minus-v0.1 is published on Hugging Face: [TheGarageDev/Minus-v0.1](https://huggingface.co/TheGarageDev/Minus-v0.1).

It's a fine-tune of [LiquidAI's LFM2.5-VL-450M](https://huggingface.co/LiquidAI) trained on frames captured from real TV streams and compiled into fused-layer models for the Axera NPU. Inference is prefill-only: the yes/no answer is read straight from the logits without generating any tokens, so latency is deterministic.

| Metric | Value |
|---|---|
| Accuracy (800-image holdout) | 97.0% |
| Ad recall | 94.8% |
| Non-ad recall (false-positive resistance) | 99.2% |
| Inference latency | ~0.37s per frame |
| Parameters | 450M |

The same model also classifies screen state (playing / paused / menu / dialog / screensaver) for autonomous mode.

## Features

- Real-time ad blocking via OCR + VLM + ASR triangulation. The decision engine handles the ugly cases: flicker between ads, paused screens that look like ads, black transition frames, frozen streams.
- Ads get replaced, not just blacked out. Choose Spanish vocabulary cards (750+ entries), trivia, haikus, or a photo screensaver of your own pictures.
- 4K@60fps passthrough with a zero-copy VPU/RGA pipeline in a [patched ustreamer](https://github.com/garagehq/ustreamer), around 5% CPU.
- Audio mutes the instant a block starts, with watchdogs that recover the pipeline if anything wedges.
- Web UI with the live feed, pause controls, detection history, screenshot review, and settings, usable from any device on your network.
- Remote control integration for Fire TV (ADB), Roku (ECP), and Google TV (ADB), including pressing "Skip Ad" for you.
- Autonomous mode keeps YouTube playing unattended on a schedule to collect training data, using the VLM to navigate.
- Odds and ends: IR blaster for HDMI switches, WS2812B status LED strip, WiFi captive portal for first-time setup, HDCP 1.4 capture support.

## Web UI

| Desktop | Mobile |
|---|---|
| ![Minus web UI on desktop](docs/images/webui-home.png) | ![Minus web UI on mobile](docs/images/webui-mobile.png) |

The web UI is served by the device itself (Flask). Live feed, block/pause controls, per-detection history, a swipe-based screenshot review flow for cleaning up training data, device remotes, and every setting. No app to install.

## Hardware

| Component | Notes |
|---|---|
| [Radxa Rock 5B+](https://radxa.com/products/rock5/5bp/) | RK3588 SoC with an HDMI input port, NPU for OCR, VPU for 4K60 JPEG encoding |
| Axera AX8850 accelerator (M.2) | Runs minus-v0.1 at ~370ms per inference |
| HDMI cables | Source → Minus → TV |
| *Optional:* IR LED on GPIO | Controls an HDMI switch for multi-device setups |
| *Optional:* WS2812B 8-LED strip | Status indicator (idle / blocking / error / ...) |
| *Optional:* HDCP 1.4 sink key | For capturing HDCP-protected sources, see [hdcp/](hdcp/README.md) |

There is no display server or desktop environment involved. Minus talks straight to DRM/KMS and the hardware encoders.

## Getting started

```bash
git clone https://github.com/garagehq/Minus-streaming-stick.git /home/radxa/Minus
cd /home/radxa/Minus

# Install system + Python dependencies (GStreamer, fonts, rknnlite, axengine, ...)
# then build the patched ustreamer:
git clone https://github.com/garagehq/ustreamer.git /home/radxa/ustreamer-garagehq
cd /home/radxa/ustreamer-garagehq && make WITH_MPP=1
cp ustreamer /home/radxa/ustreamer-patched

# Download the model
# → https://huggingface.co/TheGarageDev/Minus-v0.1
#   into /home/radxa/axera_models/minus-v0.1/

# Run directly:
cd /home/radxa/Minus && sudo python3 minus.py

# ...or install as a service that starts on boot:
sudo ./install.sh
```

The full dependency list and deployment walkthrough are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Model paths and detection thresholds are set with environment variables (`MINUS_VLM_MODEL_DIR`, `MINUS_OCR_MODEL_DIR`, etc.); [CLAUDE.md](CLAUDE.md) documents all of them.

Minus auto-detects the connected HDMI output, resolution, DRM plane, and audio device at startup, and works with both 4K and 1080p displays.

## Documentation

| Document | Description |
|---|---|
| [docs/FEATURES.md](docs/FEATURES.md) | Complete feature list |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and data flow |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Setting up a new device from scratch |
| [docs/API.md](docs/API.md) | Web UI REST API |
| [docs/ASR.md](docs/ASR.md) | Audio-based ad confirmation |
| [docs/AUDIO.md](docs/AUDIO.md) | Audio passthrough pipeline |
| [docs/AESTHETICS.md](docs/AESTHETICS.md) | Visual design guide |
| [docs/IR_TRANSMITTER.md](docs/IR_TRANSMITTER.md) | IR blaster for HDMI switches |
| [docs/STATUS_LEDS.md](docs/STATUS_LEDS.md) | WS2812B status strip |
| [hdcp/README.md](hdcp/README.md) | HDCP 1.4 setup for the HDMI input |
| [CLAUDE.md](CLAUDE.md) | Engineering notes: detection tuning and the full bug history |

## Related projects

- [Minus-chrome-extension](https://github.com/garagehq/Minus-chrome-extension): the same idea for your browser, as a Chrome extension.
- [garagehq/ustreamer](https://github.com/garagehq/ustreamer): our ustreamer fork with RK3588 MPP hardware encoding, NV12/NV24/BGR24 capture, and the 60fps blocking-overlay compositor.
- [TheGarageDev/Minus-v0.1](https://huggingface.co/TheGarageDev/Minus-v0.1): the model, on Hugging Face.

## Support

Minus is free, open-source, and runs entirely on your own hardware. No ads, no tracking, no server bills passed on to you. If it has saved you from a few unskippable ad breaks, you can buy me a coffee ☕:

<a href="https://buymeacoffee.com/cyrilengmann" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50" width="210"></a>

> **[buymeacoffee.com/cyrilengmann](https://buymeacoffee.com/cyrilengmann)**

## License

- Code: [GPL-3.0](LICENSE)
- Documentation: [CC BY-SA 4.0](LICENSE-DOCS)
- Hardware designs: [CERN-OHL-S-2.0](LICENSE-HARDWARE)
