# Audio Passthrough - Implementation Plan

## Current Hardware Setup

```
HDMI-RX Input → Rock5T → HDMI-TX Output
                 │
                 └── Card 4: rockchip,hdmiin (ALSA capture)
                     └── Device hw:4,0
```

**Audio Capture Device:**
```
card 4: rockchiphdmiin [rockchip,hdmiin], device 0: rockchip,hdmiin i2s-hifi-0
```

**Audio Output Devices:**
```
card 0: rockchiphdmi0 [rockchip-hdmi0]  ← HDMI-TX0 (main output)
card 1: rockchiphdmi1 [rockchip-hdmi1]  ← HDMI-TX1
card 2: rockchipes8316                   ← Onboard codec (headphones)
card 3: rockchiphdmi2 [rockchip-hdmi2]  ← Additional HDMI
```

**V4L2 Audio Info (from HDMI-RX):**
```
audio_sampling_rate: 48000 (read-only, volatile)
audio_present: 0/1 (read-only, volatile)
```

## Architecture Options

### Option A: Separate Audio Pipeline with Silent Keepalive (IMPLEMENTED)

Audio runs in its own GStreamer pipeline with `audiomixer` combining HDMI input with a silent keepalive tone. This prevents pipeline stalls when the HDMI source has no audio.

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIDEO PIPELINE (existing)                    │
│  souphttpsrc → mppjpegdec → input-selector → kmssink            │
│                               ▲                                  │
│                               │ (switch on ad)                  │
│                 videotestsrc ─┘                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO PIPELINE (implemented)                  │
│                                                                  │
│  alsasrc (HDMI) ──────┐                                         │
│                       ├──► audiomixer ──► volume ──► alsasink   │
│  audiotestsrc ────────┘         ▲                               │
│  (silent keepalive)             │ mute=true on ad               │
└─────────────────────────────────────────────────────────────────┘
```

**Why silent keepalive?**
- HDMI sources can have gaps in audio (between songs, during silence)
- Without keepalive, `alsasrc` may stall or error when source goes silent
- `audiotestsrc wave=silence` provides continuous (inaudible) data flow
- Pipeline never stalls because `audiomixer` always has input

**Pros:**
- Independent of video pipeline
- Easy mute control via `volume` element
- Never stalls during source silence gaps
- Auto-recovery with watchdog if issues occur

**Cons:**
- Two pipelines to manage (but simpler to debug)
- Small potential for audio/video drift (unnoticeable for live)

### Option B: Unified Audio+Video Pipeline

Add audio to the existing video pipeline with synchronized input-selectors.

```
input-selector (video) ─────────────┬───→ kmssink
                                    │
input-selector (audio) ─────────────┴───→ alsasink
     ▲
     │ mute during ads (switch to audiotestsrc silence)
```

**Pros:**
- Single pipeline, synchronized switching
- Guaranteed audio/video sync

**Cons:**
- More complex pipeline
- If audio fails, takes down video
- Harder to debug

## Recommended Approach: Option A

### GStreamer Audio Pipeline

```python
# Audio passthrough with mute control
audio_pipeline_str = (
    "alsasrc device=hw:4,0 ! "
    "queue max-size-buffers=10 leaky=downstream ! "
    "audioconvert ! "
    "volume name=vol mute=false ! "
    "alsasink device=hw:0,0 sync=false"
)
```

### Muting During Ads

```python
class AudioPassthrough:
    def __init__(self):
        self.pipeline = Gst.parse_launch(audio_pipeline_str)
        self.volume = self.pipeline.get_by_name('vol')

    def mute(self):
        """Mute audio during ad blocking."""
        self.volume.set_property('mute', True)

    def unmute(self):
        """Restore audio after ad ends."""
        self.volume.set_property('mute', False)
```

**A block's mute survives a pipeline rebuild (Oct 2026).** The real pipeline
(`src/audio.py`) is built with the *current* mute state (`volume ... mute=true`
while `is_muted`), and the volume element is set muted again as soon as it is
found. Every rebuild path (the watchdog restart, `restart()`, and `stop()` +
`start()` on an HDMI reconnect) therefore comes up muted mid-block. Before
this, `start()` built a fresh `mute=false` pipeline while `is_muted` stayed
`True`: a TV that re-hotplugged during a blocked ad let the ad's sound play for
the rest of the block, and every check still read "muted"
(`tests/test_audio_mute_survives_rebuild.py`).

### Integration with Ad Blocker

```python
# In ad_blocker.py show() method:
def show(self, source='default'):
    # ... existing video switch code ...

    # Mute audio during ads
    if self.audio_passthrough:
        self.audio_passthrough.mute()

# In ad_blocker.py hide() method:
def hide(self):
    # ... existing video switch code ...

    # Restore audio
    if self.audio_passthrough:
        self.audio_passthrough.unmute()
```

## Key Considerations

### 1. Audio Format Handling

The `audioconvert` element handles format conversion between capture and playback devices. May need `audioresample` if sample rates differ:

```python
"alsasrc device=hw:4,0 ! audioconvert ! audioresample ! alsasink device=hw:0,0"
```

### 2. Latency

For live passthrough, use `sync=false` on alsasink to minimize latency:

```python
"alsasink device=hw:0,0 sync=false"
```

### 3. Audio Signal Detection

Check if HDMI-RX has audio before starting:

```python
import subprocess

def check_audio_present():
    result = subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video0", "--get-ctrl", "audio_present"],
        capture_output=True, text=True
    )
    return "value=1" in result.stdout
```

### 4. Buffer Sizes

Queue with leaky buffer prevents audio buildup if pipeline stalls:

```python
"queue max-size-buffers=10 leaky=downstream"
```

## Testing Commands

**Test audio capture:**
```bash
# Record 5 seconds from HDMI-RX
arecord -D hw:4,0 -d 5 -f cd -r 48000 -c 2 -t wav /tmp/hdmiin_test.wav
```

**Test audio playback:**
```bash
# Play to HDMI-TX0
aplay -D hw:0,0 /tmp/hdmiin_test.wav
```

**Test live passthrough:**
```bash
gst-launch-1.0 alsasrc device=hw:4,0 ! audioconvert ! alsasink device=hw:0,0 sync=false
```

**Test with mute control:**
```bash
gst-launch-1.0 alsasrc device=hw:4,0 ! audioconvert ! volume mute=false ! alsasink device=hw:0,0 sync=false
```

## Potential Issues

### No Audio on HDMI-RX

- Check `audio_present` via V4L2
- Some sources may not send audio (check source device)
- May need to wait for stable signal before starting audio pipeline

### Audio Crackling/Pops

- Increase buffer size: `buffer-time=200000` on alsasrc/alsasink
- Add `audioresample` if sample rates mismatch
- Check for CPU contention with ML workers

### Sync with Video

For live passthrough, sync shouldn't be an issue. Both pipelines process data as it arrives. If noticeable drift occurs:
- Use same clock source
- Consider unified pipeline (Option B)

## Implementation Steps

1. **Create `src/audio.py`** - AudioPassthrough class
2. **Add audio pipeline** - alsasrc → audioconvert → volume → alsasink
3. **Integrate with DRMAdBlocker** - mute on show(), unmute on hide()
4. **Handle edge cases** - no audio signal, pipeline errors
5. **Test with various sources** - different sample rates, stereo/mono

## File Structure After Implementation

```
minus/
├── minus.py              # Orchestrator (initialize audio)
├── src/
│   ├── ad_blocker.py     # Video pipeline + mute integration
│   ├── audio.py          # NEW: Audio passthrough pipeline
│   ├── ocr.py            # OCR detection
│   └── vlm.py            # VLM detection
```

## References

- [Radxa HDMI-RX Documentation](https://docs.radxa.com/en/rock5/rock5t/app-development/hdmi-rx)
- [GStreamer alsasrc](https://gstreamer.freedesktop.org/documentation/alsa/alsasrc.html)
- [GStreamer alsasink](https://gstreamer.freedesktop.org/documentation/alsa/alsasink.html)
- [GStreamer volume element](https://gstreamer.freedesktop.org/documentation/volume/index.html)
- [GStreamer input-selector](https://gstreamer.freedesktop.org/documentation/coreelements/input-selector.html)
