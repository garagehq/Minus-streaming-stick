# Minus Deployment Guide

This guide covers deploying Minus as a systemd service on Radxa/RK3588 hardware.

---

## Prerequisites

### Hardware
- Radxa board with RK3588 SoC (e.g., Radxa Rock 5B)
- Axera LLM 8850 NPU card (for VLM)
- HDMI capture card connected to `/dev/video0`
- HDMI output connected to display

### Software
- Ubuntu 22.04 or similar Linux distribution
- Python 3.11+
- GStreamer 1.0 with Rockchip plugins
- ustreamer with MPP support (garagehq fork)

---

## Installation

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

### 5. Install VLM Models (Optional)

For ad detection using LFM2.5-VL-450M on Axera NPU:

```bash
# Install Axera runtime
pip3 install --break-system-packages axengine

# Copy model files to expected location
# Models should be at: /home/radxa/axera_models/LFM2/LFM2-450M-ft-v2-fused-v2/
```

---

## Configuration

### Environment Variables

All configuration can be done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINUS_USTREAMER_PATH` | `/home/radxa/ustreamer-patched` | Path to ustreamer binary |
| `MINUS_VLM_MODEL_DIR` | `/home/radxa/axera_models/LFM2/LFM2-450M-ft-v2-fused-v2` | VLM model directory |
| `MINUS_OCR_MODEL_DIR` | `/home/radxa/rknn-llm/.../paddleocr` | OCR model directory |
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
2. Check model files: `ls /home/radxa/axera_models/LFM2/LFM2-450M-ft-v2-fused-v2/`
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
