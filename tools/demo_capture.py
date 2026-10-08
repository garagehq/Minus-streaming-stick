#!/usr/bin/env python3
"""Capture the live input video and Minus's state for a demo recording.

Writes to OUT_DIR:
  frames/NNNNNN.jpg   every MJPEG frame from ustreamer (the live HDMI input)
  frames.idx          "index mono_time" per frame (box monotonic clock)
  timeline.jsonl      ~5 Hz: name-mute state, ASR/caption text, temperatures,
                      CPU clocks, all stamped with time.monotonic()

The output audio comes from Minus itself (MINUS_RECORD_OUTPUT_DIR, see
src/output_recorder.py); tools/demo_compose.py lines the three up.

usage: python3 tools/demo_capture.py OUT_DIR SECONDS
"""
import glob
import json
import os
import sys
import threading
import time
import urllib.request

STREAM = 'http://localhost:9090/stream'
API = 'http://localhost'


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def sysfs_state():
    temps = {}
    for z in glob.glob('/sys/class/thermal/thermal_zone*'):
        try:
            with open(z + '/type') as f:
                name = f.read().strip().replace('-thermal', '')
        except OSError:
            continue
        t = read_int(z + '/temp')
        if t is not None:
            temps[name] = t / 1000.0
    clocks = {}
    for p in sorted(glob.glob('/sys/devices/system/cpu/cpufreq/policy*')):
        clocks[os.path.basename(p)] = {
            'cur': read_int(p + '/scaling_cur_freq'),
            'max': read_int(p + '/cpuinfo_max_freq'),
            'cap': read_int(p + '/scaling_max_freq')}
    return temps, clocks


def api(path):
    try:
        with urllib.request.urlopen(API + path, timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return None


def capture_frames(out, until, stats):
    os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
    idx = open(os.path.join(out, 'frames.idx'), 'w')
    r = urllib.request.urlopen(STREAM, timeout=5)
    buf = b''
    n = 0
    while time.monotonic() < until:
        chunk = r.read(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b'\xff\xd8')
            b = buf.find(b'\xff\xd9', a + 2) if a >= 0 else -1
            if a < 0 or b < 0:
                break
            jpg, buf = buf[a:b + 2], buf[b + 2:]
            t = time.monotonic()
            with open(os.path.join(out, 'frames', f'{n:06d}.jpg'), 'wb') as f:
                f.write(jpg)
            idx.write(f'{n} {t:.4f}\n')
            n += 1
    idx.close()
    stats['frames'] = n


def capture_timeline(out, until):
    with open(os.path.join(out, 'timeline.jsonl'), 'w') as f:
        next_sys = 0
        temps, clocks = sysfs_state()
        while time.monotonic() < until:
            t = time.monotonic()
            if t >= next_sys:
                temps, clocks = sysfs_state()
                next_sys = t + 1.0
            nm = api('/api/name-mute') or {}
            st = api('/api/status') or {}
            asr = st.get('asr') or {}
            f.write(json.dumps({
                't': round(t, 3), 'muted': nm.get('muted'), 'mute_count': nm.get('mute_count'),
                'asr_text': nm.get('last_asr_text') or asr.get('last_transcript'),
                'caption_text': nm.get('last_caption_text'), 'delay_s': nm.get('delay_s'),
                'asr_engine': asr.get('engine') or asr.get('model'),
                'asr_latency_s': asr.get('last_latency_s'), 'cpu_percent': st.get('cpu_percent'),
                'thermal': st.get('thermal'), 'thermal_degraded': st.get('thermal_degraded'),
                'temps': temps, 'clocks': clocks}) + '\n')
            time.sleep(max(0.0, 0.2 - (time.monotonic() - t)))


def main():
    out, seconds = sys.argv[1], float(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    until = time.monotonic() + seconds
    stats = {}
    th = [threading.Thread(target=capture_frames, args=(out, until, stats)),
          threading.Thread(target=capture_timeline, args=(out, until))]
    for x in th:
        x.start()
    for x in th:
        x.join()
    print(json.dumps({'out': out, 'start_mono': round(until - seconds, 3), **stats}))


if __name__ == '__main__':
    main()
