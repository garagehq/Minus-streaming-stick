# Minus Deployment Guide

This guide covers deploying Minus as a systemd service on Radxa/RK3588 hardware.

---

## Prerequisites

### Hardware
- Radxa ROCK 5B+ or ROCK 5B (RK3588 SoC with the 4K@60 HDMI receiver), 8 GB RAM recommended, 4 GB minimum
- Radxa AICore AX-M1 (Axera AX8850) in an M.2 M-key slot, for the VLM
- Heatsink and fan, plus a 30 W USB-C PD supply
- Source device on the HDMI **input** (appears as `/dev/video0`), TV on the HDMI **output**

Full bill of materials with part links and sizing notes: [HARDWARE.md](HARDWARE.md).

### Software
- **Debian 12 (bookworm) arm64 from a Radxa official image**, running the **Radxa BSP 6.1 kernel** (`uname -r` should end in `-rk<NNNN>`, e.g. `6.1.84-8-rk2410`). A mainline kernel will not work: `rk_hdmirx`, Rockchip MPP, RGA and the RKNN NPU runtime are all BSP-only, and without them `/dev/video0` never appears. Verify with `v4l2-ctl -d /dev/video0 --info` (expect driver `rk_hdmirx`).
- Python 3.11+
- GStreamer 1.0 with Rockchip plugins (`gstreamer1.0-rockchip1`)
- ustreamer with MPP support (garagehq fork)
- `rknpu2-rk3588` + `rknnlite` for OCR, `axclhost` + `axengine` for the VLM (see steps 5 and 6)

---

## Installation

### Quick path: `install.sh` on a fresh image (KingBlocker9000 units)

On a freshly flashed Radxa Debian 12 image this is the whole install. It
was first run end to end on the second unit (`kingblocker9000`, ROCK 5B+,
KDE image) on 2026-11-10.

```bash
# On the new board's own keyboard/screen, once, so it can be reached over SSH:
sudo apt install -y openssh-server && sudo systemctl enable --now ssh
hostname -I

# Then (over SSH is fine; the script survives a dropped session):
git clone -b feature/kingblocker-9000 https://github.com/garagehq/Minus-streaming-stick.git ~/Minus
cd ~/Minus
sudo env MINUS_HOSTNAME=kingblocker9000 nohup ./install.sh > ~/install.log 2>&1 &
tail -f ~/install.log          # ends with "=== Installation complete ==="
sudo reboot                    # overlays / kernel command line take effect
```

It is idempotent: re-run it after a failure and it only does what is
missing (`MINUS_SKIP_APT=1` skips the package step on a re-run). Steps:
apt packages, Python packages (`requirements.txt`, pip `--user` for
radxa), ustreamer built from garagehq/ustreamer `asr-first`, the stock
SenseVoice model from Hugging Face if no model is installed, the 24-30 Hz
HDMI-RX EDID, groups, hostname + mDNS, silent boot / auto-login / no
sleep, and the systemd service (enabled, started at the next boot).

Options (environment variables): `MINUS_HOSTNAME`, `MINUS_START=1` (start
now), `MINUS_KEEP_DESKTOP=1`, `MINUS_VERBOSE_BOOT=1`, `MINUS_STATUS_LEDS=1`
(WS2812B strip overlay), `MINUS_IR=1` (IR transmitter overlay),
`MINUS_USTREAMER_BRANCH`, `MINUS_SKIP_APT=1`.

**The fine-tuned speech model is not public.** Copy
`~/asr_models/sensevoice-rknn-minus-ft/` from a box that has it and check
it with `sha256sum -c SHA256SUMS` (skip the `.onnx` line, which is not
copied). Without it the stock model is used (82.5% vs 94.6% of names
muted from the first syllable, docs/ASR_BENCHMARKS.md).

#### What the first fresh unit taught us

- **The KDE image suspends itself when idle.** A box left alone during the
  install went to sleep (`PM: suspend entry (deep)` in the journal), which
  looks like the network dropping or the user being logged out. The
  installer masks `sleep/suspend/hibernate.target` and turns off logind's
  idle action; on a box that has not been through the installer yet, run
  `sudo systemctl mask sleep.target suspend.target hibernate.target
  hybrid-sleep.target` before any long job.
- **`adb` fails with "held broken packages".** Radxa images ship
  `android-libcutils` from bookworm-backports, and bookworm's `adb` wants
  the older one. The installer takes `adb` from backports
  (`apt-get install -t bookworm-backports adb`). One failing package used
  to send the installer into a slow one-by-one retry of everything.
- **An interrupted apt run leaves dpkg half-configured.** Run
  `sudo dpkg --configure -a` before re-running the installer.
- **ASR was silently off without faster-whisper.** The ASR gate used to
  require the `faster_whisper` package whatever the engine; a unit with
  SenseVoice but no faster-whisper logged "ASR module loaded but
  faster-whisper not available" and never muted from speech. It now
  checks the selected engine (`src/asr.py:is_asr_available`). After an
  install, check `curl -s localhost/api/status` -> `asr.engine` is
  `sensevoice`.
- **Verify every model copy against its checksums.** Copies from minus-2
  arrived corrupted twice (SSH "message authentication code incorrect",
  then a bad hash over plain HTTP). The file on minus-2's SD card was
  fine; its in-memory cached copy was not (dropping the page cache gave the
  right hash). Treat anything served from a box with suspect RAM as
  untrusted and check `SHA256SUMS` on the receiving side.
- **Training clips are off by default.** `MINUS_ASR_CLIPS=1` turns the
  collector on (it budgets 8 GB of recordings); customer units leave it off.
- **No desktop on HDMI-out.** Minus takes over the display, so the
  installer disables the display manager, boots to `multi-user.target`
  and keeps the screen black: kernel and systemd text go to the serial
  port only (`console=tty1` removed, `quiet loglevel=0
  vt.global_cursor_default=0 logo.nologo systemd.show_status=false`), no
  getty on tty1, and an auto-logged-in shell on tty2 (Ctrl+Alt+F2) for a
  keyboard. The original command line is saved to
  `/etc/kernel/cmdline.bak-minus`; restore it, `sudo u-boot-update`,
  reboot. The bootloader's own logo/text before Linux starts is unchanged.
- **KingBlocker units have only the red mute LED** (pin 40, ground on pin
  39, plain GPIO). The WS2812B strip and IR transmitter overlays are
  opt-in, and the strip is saved as disabled
  (`/root/.minus_status_leds.json`).

### Manual install

The steps below do the same by hand.

### 1. Clone Repository

```bash
cd /home/radxa
git clone https://github.com/garagehq/Minus-streaming-stick.git
cd Minus
```

### 2. Install System Dependencies

```bash
# GStreamer and media packages
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-rockchip1 \
  gir1.2-gst-plugins-base-1.0 \
  libgstreamer1.0-dev

# Build tools for ustreamer
sudo apt install -y \
  librockchip-mpp-dev \
  libfreetype-dev \
  libjpeg-dev \
  libevent-dev

# Fonts for blocking overlay
sudo apt install -y fonts-dejavu-core fonts-ibm-plex

# Other utilities
sudo apt install -y imagemagick ffmpeg curl v4l-utils
```

### 3. Install Python Dependencies

```bash
pip3 install --break-system-packages \
  pyclipper shapely numpy opencv-python \
  pexpect PyGObject flask requests androidtv \
  rknnlite
```

**Note:** `rknnlite` may require Rockchip's custom repository.

### 4. Build ustreamer with MPP Support

```bash
git clone https://github.com/garagehq/ustreamer.git /home/radxa/ustreamer-garagehq
cd /home/radxa/ustreamer-garagehq
make WITH_MPP=1
cp ustreamer /home/radxa/ustreamer-patched
```

### 5. OCR Models (nothing to do)

The PaddleOCR models that read on-screen ad text (PP-OCRv6 by default, with
PP-OCRv3 kept as a rollback via `MINUS_OCR_MODEL_VERSION=v3`) run on the RK3588's
own NPU and **ship inside this repository** at `models/paddleocr/`, so a fresh
clone already has them. `src/config.py` points `OCR_MODEL_DIR` there
automatically, falling back to the legacy `rknn-llm/.../paddleocr` path only if
the repo copy is missing. Override with `MINUS_OCR_MODEL_DIR` if you keep them
elsewhere.

The NPU runtime itself comes from Radxa's `rk3588-bookworm` repository and is
present on a current Radxa image:

```bash
dpkg -l | grep rknpu2-rk3588      # expect 2.3.0 or newer
pip3 show rknnlite                # expect 2.3.0 or newer
```

Confirm OCR works once the service is up:

```bash
curl -s -X POST http://localhost/api/ocr/test | python3 -m json.tool
```

See `models/paddleocr/README.md` for what each file does, checksums, and how to
re-convert them.

### 6. Install the VLM (Optional)

For ad detection using minus-v0.1 (a fine-tuned LFM2.5-VL-450M) on the Axera
AX-M1. Minus runs without it, in OCR-only mode, but detection quality drops
substantially.

```bash
# Install the AXCL host driver (M5Stack APT repository), then the Python binding
dpkg -l | grep axclhost                     # expect 3.6.6 or newer
pip3 install --break-system-packages axengine

# Confirm the card is present and healthy
lspci | grep -i axera                       # expect 1f4b:0650
axcl-smi info

# Download the model from https://huggingface.co/TheGarageDev/Minus-v0.1
# Models should be at: /home/radxa/axera_models/minus-v0.1/
```

If the VLM fails to load with an error about `libaxcl_*.so`, see the broken
symlink entry in [CLAUDE.md](../CLAUDE.md) for the one-line fix.

---

## Configuration

### Environment Variables

All configuration can be done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINUS_USTREAMER_PATH` | `/home/radxa/ustreamer-patched` | Path to ustreamer binary |
| `MINUS_VLM_MODEL_DIR` | `/home/radxa/axera_models/minus-v0.1` | VLM model directory |
| `MINUS_OCR_MODEL_DIR` | `models/paddleocr` in the repo | OCR model directory |
| `MINUS_ANIMATION_START` | `0.3` | Blocking animation duration (seconds) |
| `MINUS_ANIMATION_END` | `0.25` | Unblocking animation duration (seconds) |
| `MINUS_FRAME_STALE_THRESHOLD` | `5.0` | Frame freshness threshold for health checks |
| `MINUS_DYNAMIC_COOLDOWN` | `0.5` | Cooldown after screen becomes dynamic |
| `MINUS_SCENE_CHANGE_THRESHOLD` | `0.01` | Scene change detection threshold |
| `MINUS_VLM_ALONE_THRESHOLD` | `5` | Consecutive VLM detections to trigger alone |

### Command Line Options

```bash
python3 minus.py [OPTIONS]

Options:
  --device PATH         Capture device (default: /dev/video0)
  --ocr-timeout SECS    OCR timeout in seconds (default: 1.5)
  --max-screenshots N   Keep N recent screenshots (default: 50, 0=unlimited)
  --check-signal        Check HDMI signal and exit
  --connector-id ID     DRM connector ID (auto-detected)
  --plane-id ID         DRM plane ID (auto-detected)
  --webui-port PORT     Web UI port (default: 80)
```

---

## Systemd Service

### Install Service

```bash
cd /home/radxa/Minus
sudo ./install.sh
```

This creates `/etc/systemd/system/minus.service` and enables it.

### Service File

The service file (`minus.service`) configures:

```ini
[Unit]
Description=Minus Ad Blocker
After=network.target
Conflicts=gdm.service lightdm.service sddm.service

[Service]
Type=simple
User=root
WorkingDirectory=/home/radxa/Minus
ExecStart=/usr/bin/python3 /home/radxa/Minus/minus.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Note:** Runs as root for DRM/device access. Conflicts with display managers.

### Service Commands

```bash
# Start service
sudo systemctl start minus

# Stop service
sudo systemctl stop minus

# Restart service
sudo systemctl restart minus

# View status
sudo systemctl status minus

# View logs
journalctl -u minus -f

# Enable at boot
sudo systemctl enable minus

# Disable at boot
sudo systemctl disable minus
```

### Uninstall Service

```bash
sudo ./uninstall.sh
```

---

## Verification

### Check HDMI Signal

```bash
python3 minus.py --check-signal
```

### Check ustreamer

```bash
curl -s http://localhost:9090/snapshot -o /tmp/test.jpg && echo "OK"
```

### Check Web UI

Open in browser: `http://<device-ip>:80`

### Check Health

```bash
curl http://localhost:80/api/health
```

### Check Metrics

```bash
curl http://localhost:80/api/metrics
```

---

## Troubleshooting

### Box drops off the network during a long job
Check `journalctl -b -1 | tail` for `Reached target sleep.target`: it
suspended. Mask the sleep targets (see *What the first fresh unit taught
us* above) or run `install.sh`, which does it.

### No name muting from speech
`curl -s localhost/api/status` -> `asr` should show `engine: sensevoice`
and `enabled: true`. If the log says "no ASR engine is available", the
SenseVoice model dir or `rknnlite`/`kaldi_native_fbank` is missing.

### Service Won't Start

1. Check logs: `journalctl -u minus -n 50`
2. Check HDMI signal: `v4l2-ctl -d /dev/video0 --query-dv-timings`
3. Check device permissions: `ls -la /dev/video0 /dev/dri/card0`

### No Video Output

1. Check DRM connector: `modetest -M rockchip -c | grep HDMI`
2. Check plane availability: `modetest -M rockchip -p`
3. Restart video: `curl -X POST http://localhost:80/api/video/restart`

### ustreamer Fails

```bash
# Kill orphaned processes
pkill -9 ustreamer
fuser -k /dev/video0

# Restart service
sudo systemctl restart minus
```

### VLM Not Working

1. Check Axera card: `axcl_smi`
2. Check model files: `ls /home/radxa/axera_models/minus-v0.1/`
3. Check dependencies: `pip3 show axengine`

### Audio Issues

1. Check capture device: `aplay -l`
2. Check ALSA: `cat /proc/asound/cards`
3. Test manually:
   ```bash
   gst-launch-1.0 alsasrc device=hw:4,0 ! \
     "audio/x-raw,rate=48000,channels=2,format=S16LE" ! \
     alsasink device=hw:0,0 sync=false
   ```

### Memory Issues

1. Check usage: `curl http://localhost:80/api/health`
2. Check for leaks: Watch memory over time in health endpoint
3. Force cleanup: Service automatically triggers GC at 90% memory

---

## Monitoring

### Prometheus Integration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'minus'
    static_configs:
      - targets: ['<device-ip>:80']
    metrics_path: '/api/metrics'
    scrape_interval: 15s
```

### Uptime Monitoring

Use the simple health endpoint:

```bash
curl http://localhost:80/api/health?simple=1
# Returns: {"status": "ok", "timestamp": 1234567890}
```

### Log Files

- Location: `/tmp/minus.log`
- Max size: 5MB per file
- Rotation: 3 backup files (minus.log.1, .2, .3)

---

## Security Considerations

1. **Network Access**: Web UI binds to 0.0.0.0 - use firewall or Tailscale
2. **No Authentication**: Relies on network-level security
3. **Root Access**: Runs as root for hardware access
4. **ADB Keys**: Fire TV ADB keys stored in `~/.android/`

### Recommended: Use Tailscale

```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Connect
sudo tailscale up

# Access via Tailscale hostname
http://minus:80
```

---

*Last updated: 2026-04-02*
