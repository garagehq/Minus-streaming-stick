# Minus Architecture

## System Overview

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

## Hardware Requirements

### Primary Board: RK3588
- **SoC**: Rockchip RK3588
- **NPU**: 6 TOPS for PaddleOCR inference
- **HDMI-RX**: 4K@30fps video capture
- **HDMI-TX**: 4K@60fps video output
- **Memory**: 8GB+ recommended

### Secondary NPU: Axera LLM 8850
- **Purpose**: minus-v0.1 (fine-tuned LFM2.5-VL-450M) inference — prefill-only, no decode
- **Connection**: USB/PCIe to RK3588
- **Memory**: Dedicated NPU memory
- **Performance**: ~0.37s per inference (deterministic)

## Software Components

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| Main | `minus.py` | Entry point, orchestration |
| Ad Blocker | `src/ad_blocker.py` | GStreamer pipeline, blocking API |
| Audio | `src/audio.py` | Audio passthrough, mute control |
| OCR Client | `src/ocr.py` | PaddleOCR model + `AD_EXCLUSIONS`, keyword scan |
| OCR Worker | `src/ocr_worker.py` | Process-based OCR with hard 1.0s timeout, warmup, keepalive |
| VLM Client | `src/vlm.py` | LFM2.5-VL inference wrapper — argmax logit thresholding for `detect_ad`, first-token logit lookup for `query_image`; both prefill-only on 16 fused decoder layers |
| VLM Worker | `src/vlm_worker.py` | Process-based VLM with soft (1.5s) / hard (2.0s) timeout, P95 latency auto-recovery, `_call_lock` serializing detect_ad and query_image |
| Health | `src/health.py` | Health monitoring, recovery, ALSA zombie detection, HDMI DPMS reinit |
| Web UI | `src/webui.py` | Flask web interface |

### Support Modules

| Module | File | Purpose |
|--------|------|---------|
| Fire TV | `src/fire_tv.py` | ADB remote control |
| Fire TV Setup | `src/fire_tv_setup.py` | Auto-setup flow |
| Roku | `src/roku.py` | ECP (External Control Protocol) remote control |
| Device Config | `src/device_config.py` | Device type selection + persistence (Fire TV / Roku / Google TV / generic) |
| WiFi Manager | `src/wifi_manager.py` | Captive portal and AP-mode fallback |
| Overlay | `src/overlay.py` | Notification overlays |
| Vocabulary | `src/vocabulary.py` | Spanish vocabulary list |
| Screenshots | `src/screenshots.py` | Training data with dHash dedup |
| Autonomous Mode | `src/autonomous_mode.py` | Device-agnostic VLM-guided playback |
| Skip Detection | `src/skip_detection.py` | Skip button detection |
| Config | `src/config.py` | Configuration dataclass |
| Capture | `src/capture.py` | Snapshot capture |
| Console | `src/console.py` | Console blanking |
| DRM | `src/drm.py` | DRM output probing, adaptive 4K bandwidth fallback |
| V4L2 | `src/v4l2.py` | V4L2 device probing |

### Detection Loop Execution Model

OCR and VLM do *not* run on pooled threads inside the main loop. Each is
a long-lived child process:

- `OCRProcess` (`src/ocr_worker.py`) loads the RKNN model once, processes
  requests via a `multiprocessing.Queue`, and has a **hard 1.0s timeout**
  backed by worker-process kill-and-restart with exponential backoff.
- `VLMProcess` (`src/vlm_worker.py`) loads minus-v0.1 once (17
  axengine sessions: 1 vision + 16 fused decoder layers + 1 post, plus
  a 256MB mmap of the embedding table), and uses a **soft/hard timeout**
  split: soft (1.5s) returns `"TIMEOUT"` to the caller while letting the
  worker keep finishing; hard (2.0s) kills the worker. In parallel, a
  rolling 10-sample P95 latency check triggers an auto-recovery restart
  when variance creeps above 2.0s. If a recent restart did not clear the
  state, the next trigger escalates to a *deep* restart with a longer
  NPU-release backoff. Inference is prefill-only (~0.37s deterministic);
  the runaway-decode pathology that drove FastVLM's auto-recovery
  thresholds is structurally impossible here.
- Both processes run warmup inferences at start and a periodic keepalive
  to avoid NPU cold-start penalties on the first real frame.
- `VLMProcess._call_lock` serializes `detect_ad` and `query_image` so the
  autonomous-mode thread and the main detection-loop thread cannot race
  on the shared request/response queue.

## Data Flow

### Video Pipeline

1. **Capture**: HDMI-RX captures video at 4K@30fps
2. **Encoding**: ustreamer encodes to MJPEG via MPP hardware
3. **Streaming**: HTTP stream available at :9090/stream
4. **Display**: GStreamer pipeline decodes and displays via DRM/KMS
5. **Blocking**: ustreamer composites blocking overlay at 60fps

### ML Detection Pipeline

1. **Snapshot**: HTTP GET to :9090/snapshot (~150ms)
2. **Parallel Processing**:
   - OCR: Downscale to 960x540, run PaddleOCR
   - VLM: Send to Axera NPU for scene analysis
3. **Voting**: Combine results with weighted logic
4. **Action**: Trigger/release blocking based on consensus

### Audio Pipeline

```
alsasrc (HDMI-RX) ──┐
                    ├──► audiomixer ──► volume ──► alsasink (HDMI-TX)
audiotestsrc ───────┘
(silent keepalive)
```

## Threading Model

### Main Thread
- Orchestrates startup/shutdown
- Handles signals (SIGINT, SIGTERM)

### Background Threads

| Thread | Purpose | Interval |
|--------|---------|----------|
| OCR Dispatcher | Send snapshots to `OCRProcess` and collect results | ~500ms |
| VLM Dispatcher | Send snapshots to `VLMProcess` and collect results | ~1s |
| Health Monitor | Check subsystem health | 5s |
| Vocabulary Rotation | Rotate displayed word | 11-15s |
| Debug Update | Update debug overlay | 2s |
| Video Watchdog | Detect pipeline stalls | 3s |
| Audio Watchdog | Detect audio stalls | 3s |
| Fire TV Keepalive | Maintain ADB connection | 5min |
| Autonomous Mode | VLM screen check + keepalive | 60s check, 2min VLM |

### Thread Safety

**Lock Usage:**

| Module | Lock | Protects |
|--------|------|----------|
| `ad_blocker.py` | `_lock` | `is_visible`, `current_source`, animation state |
| `audio.py` | `_lock` | `is_muted`, pipeline state, restart flag |
| `fire_tv.py` | `_lock` | `_connected`, `_device`, connection state |
| `health.py` | `_status_lock` | Health status updates |
| `capture.py` | `_capture_lock` | Rate limiting between workers |
| `capture.py` | `_session_lock` | HTTP session creation |
| `autonomous_mode.py` | `_lock` | Active state, schedule, settings |

**Thread-Safe Patterns:**

1. **State Reads**: Use `with self._lock:` for compound reads
2. **State Updates**: Always lock before modifying shared state
3. **API Calls**: HTTP calls are inherently thread-safe
4. **GStreamer**: Pipeline accessed from single thread only
5. **Atomic Flags**: `threading.Event()` for stop signals

**Critical Sections:**

```python
# Example: ad_blocker.py show/hide
def show(self, source='default'):
    with self._lock:
        if self.is_visible:
            return
        self.is_visible = True
        self.current_source = source
    # API calls outside lock to prevent blocking
    self._send_blocking_request(...)
```

**Watchdog Pattern:**

```python
# Watchdogs use Event for clean shutdown
self._stop_event = threading.Event()

def _watchdog_loop(self):
    while not self._stop_event.is_set():
        self._check_health()
        self._stop_event.wait(timeout=5.0)  # Interruptible sleep
```

**Rate Limiting:**

The capture module uses a global lock to prevent HTTP contention:
- Minimum 500ms between captures during normal operation
- Minimum 1s between captures during blocking (MPP encoder busy)

## API Endpoints

### ustreamer API (port 9090)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/stream` | GET | MJPEG video stream |
| `/snapshot` | GET | Single JPEG frame |
| `/state` | GET | Device state JSON |
| `/blocking` | GET | Blocking mode config |
| `/blocking/set` | GET | Configure blocking |
| `/blocking/background` | POST | Upload NV12 background |
| `/overlay` | GET | Notification overlay config |
| `/overlay/set` | GET | Configure overlay |

### Web UI API (port 80)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/status` | GET | System status |
| `/api/health` | GET | Health check |
| `/api/detections` | GET | Detection history |
| `/api/logs` | GET | Recent logs |
| `/api/pause/<mins>` | POST | Pause blocking |
| `/api/resume` | POST | Resume blocking |
| `/api/video/restart` | POST | Restart video pipeline |
| `/api/video/color` | GET/POST | Color settings |
| `/api/audio/sync-reset` | POST | Reset A/V sync |
| `/api/firetv/status` | GET | Fire TV status |
| `/api/firetv/command` | POST | Send Fire TV command |
| `/api/blocking/skip` | POST | Trigger Fire TV skip |
| `/api/autonomous` | GET | Autonomous mode status |
| `/api/autonomous/enable` | POST | Enable autonomous mode |
| `/api/autonomous/schedule` | POST | Set schedule |
| `/api/screenshots/review/<cat>` | GET | Unreviewed screenshots |
| `/api/screenshots/classify` | POST | Move between categories |
| `/api/screenshots/approve` | POST | Mark as reviewed |
| `/api/screenshots/undo` | POST | Undo last action |

## Configuration

### MinusConfig Dataclass

```python
@dataclass
class MinusConfig:
    device: str = "/dev/video0"
    screenshot_dir: str = "screenshots"
    ocr_timeout: float = 1.5
    ustreamer_port: int = 9090
    webui_port: int = 80
    max_screenshots: int = 0
```

### Environment Detection

At startup, Minus auto-detects:
1. Connected HDMI output (HDMI-A-1 or HDMI-A-2)
2. Display's preferred resolution
3. NV12-capable DRM plane
4. Audio output device matching HDMI output
5. V4L2 device format (NV12, BGR24, etc.)

## Error Recovery

### HDMI Signal Loss
1. Health monitor detects signal loss
2. Show "NO SIGNAL" overlay
3. Mute audio
4. On signal restore: restart ustreamer → restart pipeline

### Video Pipeline Stall
1. Watchdog detects no buffers for 10s
2. Stop current pipeline
3. Wait with exponential backoff (1s → 30s max)
4. Create new pipeline
5. Reset backoff after 10s of stable flow

### VLM Failures
1. Track consecutive timeouts
2. After 5 failures: degrade to OCR-only mode
3. Attempt VLM restart after 30s background
4. OCR continues independently

## Build & Deployment

### Dependencies

```bash
# System packages
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-* \
  librockchip-mpp-dev libfreetype-dev fonts-dejavu-core fonts-ibm-plex

# Python packages
pip3 install pyclipper shapely numpy opencv-python flask requests \
  androidtv rknnlite axengine transformers
```

### ustreamer Build

```bash
git clone https://github.com/garagehq/ustreamer.git
cd ustreamer && make WITH_MPP=1
```

### PyInstaller Build

```bash
pip3 install pyinstaller
pyinstaller minus.spec
```

### Service Installation

```bash
sudo ./install.sh   # Install service
sudo ./uninstall.sh # Remove service
```

## Performance Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Video FPS | 30fps | 30fps |
| Blocking FPS | 60fps | 60fps |
| OCR latency | <500ms | 300-400ms |
| VLM latency | <1.5s | ~0.9s |
| Blocking start | <500ms | ~300ms |
| Blocking end | <300ms | ~250ms |
| Memory usage | <2GB | ~1.5GB |
