#!/usr/bin/env python3
"""Real-time replay test for the name muter.

Streams a 16 kHz mono WAV into an AudioASRTap at real-time speed, runs the
production ASRManager + ASR worker + NameMuteController against it, and
records every mute window. Each ground-truth mention (from a YouTube json3
caption track) counts as covered when a mute window, mapped back to clip
time through the A/V delay, contains it.

usage: python3 tests/name_mute_replay.py clip16k.wav captions.json3 [delay_s]
"""
import json
import os
import re
import sys
import threading
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np  # noqa: E402
from audio import AudioASRTap  # noqa: E402
from asr import ASRManager  # noqa: E402
from name_mute import NameMuteScheduler, NameMuteController  # noqa: E402


class RecordingAudio:
    def __init__(self):
        self.windows = []
        self._on = None

    def set_name_mute(self, muted):
        now = time.monotonic()
        if muted and self._on is None:
            self._on = now
        elif not muted and self._on is not None:
            self.windows.append((self._on, now))
            self._on = None


def feed(tap, audio, t0, stop):
    """Write audio into the tap in 20 ms chunks at real-time pace."""
    chunk = 320
    for i in range(0, len(audio), chunk):
        if stop.is_set():
            return
        target = t0 + i / 16000.0
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        samples = audio[i:i + chunk]
        with tap._lock:
            n = len(samples)
            end = tap._write_pos + n
            if end <= tap._buffer_samples:
                tap._ring[tap._write_pos:end] = samples
            else:
                split = tap._buffer_samples - tap._write_pos
                tap._ring[tap._write_pos:] = samples[:split]
                tap._ring[:n - split] = samples[split:]
            tap._write_pos = end % tap._buffer_samples
            tap._samples_written += n
            tap._last_write_mono = time.monotonic()


def main():
    wav_path, cap_path = sys.argv[1], sys.argv[2]
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else 4.36
    w = wave.open(wav_path)
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    dur = len(audio) / 16000.0

    truth = []
    for ev in json.load(open(cap_path)).get('events', []):
        for sg in ev.get('segs', []) or []:
            if re.search(r'lebron', sg.get('utf8', ''), re.I):
                truth.append((ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000.0)

    tap = AudioASRTap(wav_path='/dev/shm/name_mute_replay.wav')
    rec = RecordingAudio()
    sched = NameMuteScheduler(rec, lambda: delay)
    ctl = NameMuteController(sched)
    asr = ASRManager(tap)
    ctl.window_s = asr.WINDOW_SECONDS
    calls = []   # (clip time, worker latency)

    def on_words(transcript, words, window_start):
        lat = asr._process._recent_latencies[-1] if asr._process._recent_latencies else 0.0
        calls.append((time.monotonic() - t0, lat))
        ctl.on_asr_words(transcript, words, window_start)
    asr.on_words = on_words
    asr.enabled = True
    asr.start()   # blocks until the worker has loaded the model
    if not asr.is_running:
        print("ASR failed to start")
        return 1
    sched.start()
    stop = threading.Event()
    t0 = time.monotonic() + 0.5
    th = threading.Thread(target=feed, args=(tap, audio, t0, stop), daemon=True)
    th.start()
    print(f"replaying {dur:.0f}s of audio in real time (delay {delay:.2f}s)...", flush=True)
    th.join()
    time.sleep(delay + 3.0)   # let the last mutes play out
    sched.stop()
    asr.stop()

    # Mute windows in clip time: a playback instant p plays clip time p - delay - t0.
    clip_windows = [(a - delay - t0, b - delay - t0) for a, b in rec.windows]
    covered = [x for x in truth if any(a <= x <= b for a, b in clip_windows)]
    near = [x for x in truth if x not in covered and
            any(a - 1.0 <= x <= b + 1.0 for a, b in clip_windows)]
    stray = [(a, b) for a, b in clip_windows
             if not any(a - 1.5 <= x <= b + 1.5 for x in truth)]
    total_muted = sum(b - a for a, b in clip_windows)
    st = asr.get_status()
    print(f"ASR inferences={st.get('inference_count')} p50={st.get('p50_latency_s')} "
          f"p95={st.get('p95_latency_s')}s")
    print(f"mentions: {len(truth)}  covered by a mute: {len(covered)} "
          f"({100 * len(covered) / max(1, len(truth)):.0f}%)  "
          f"within 1s of a mute: {len(near)}")
    print(f"mute windows: {len(clip_windows)}  late: {sched.late_count}  "
          f"not near a captioned mention: {len(stray)}  "
          f"total muted {total_muted:.1f}s of {dur:.0f}s")
    print("ASR calls per 30s / median latency:", [
        (int(b0), sum(1 for t, _ in calls if b0 <= t < b0 + 30),
         round(float(np.median([l for t, l in calls if b0 <= t < b0 + 30] or [0])), 2))
        for b0 in range(0, int(dur) + 1, 30)])
    print("missed:", [round(x, 1) for x in truth if x not in covered])
    print("stray:", [(round(a, 1), round(b, 1)) for a, b in stray])
    return 0


if __name__ == '__main__':
    sys.exit(main())
