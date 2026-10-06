#!/usr/bin/env python3
"""Evaluate Moonshine streaming ASR for name-detection latency, in real time.

Feeds a 16 kHz WAV into a Moonshine Stream at real-time pace (the library
transcribes on its own background thread, so only real-time feeding shows
real latency). Polls the transcript every chunk; for each target-name word,
records the wall-clock delay between the word ending in the audio and the
word first appearing in the transcript. Scores recall against a YouTube
json3 caption track.

usage: python3 tests/asr_stream_eval.py clip16k.wav captions.json3 ARCH \
         [seconds] [start_s] [opt=value ...]
"""
import json
import re
import statistics
import sys
import time
import wave

import numpy as np
import moonshine_voice as mv
from moonshine_voice import download as dl

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/src')
from name_mute import NameMatcher  # noqa: E402


def main():
    wav, cap, arch_name = sys.argv[1], sys.argv[2], sys.argv[3]
    secs = float(sys.argv[4]) if len(sys.argv) > 4 else 120
    start = float(sys.argv[5]) if len(sys.argv) > 5 else 0
    opts = dict(a.split('=', 1) for a in sys.argv[6:])
    opts.setdefault('word_timestamps', 'true')
    # Not a library option: how often the stream re-transcribes the
    # in-progress line (seconds). Too small and it falls behind real time.
    update_interval = float(opts.pop('update_interval', 0.5))
    arch = getattr(mv.ModelArch, arch_name)
    path, _ = dl.download_model_from_info(dl.find_model_info(language='en', model_arch=arch))
    t = mv.Transcriber(model_path=path, model_arch=arch, options=opts)
    w = wave.open(wav)
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    audio = audio[int(start * 16000):int((start + secs) * 16000)]
    dur = len(audio) / 16000

    truth = []
    for ev in json.load(open(cap)).get('events', []):
        for sg in ev.get('segs', []) or []:
            if re.search(r'lebron|\bking\b', sg.get('utf8', ''), re.I):
                tt = (ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000 - start
                if 0 <= tt < dur - 1:
                    truth.append(tt)

    matcher = NameMatcher()
    stream = t.create_stream(update_interval=update_interval)
    stream.start()
    chunk = 1600   # 0.1 s
    seen = []      # (word_start, word_end, latency_after_word_end)
    t0 = time.monotonic()
    i = 0
    while i < len(audio) or time.monotonic() - t0 < dur + 3:
        while i < len(audio) and time.monotonic() >= t0 + i / 16000:
            stream.add_audio(audio[i:i + chunk].tolist(), sample_rate=16000)
            i += chunk
        tr = stream.update_transcription()
        now_audio = time.monotonic() - t0          # audio time that has played
        for ln in tr.lines:
            words = [(x.word, x.start, x.end) for x in (ln.words or [])]
            for s, e in matcher.find_word_spans(words):
                if not any(abs(s - ps) < 0.5 for ps, _, _ in seen):
                    seen.append((s, e, now_audio - e))
        time.sleep(0.25)
    stream.stop()

    tol = 1.0
    hits = [x for x in truth if any(abs(s - x) <= tol for s, _, _ in seen)]
    lat = [l for _, _, l in seen]
    q = lambda xs: (f"p50 {statistics.median(xs):.2f} / p90 {statistics.quantiles(xs, n=10)[-1]:.2f}"
                    f" / max {max(xs):.2f}s" if len(xs) >= 3 else str([round(x, 2) for x in xs]))
    print(f"{arch_name} {opts} interval={update_interval} {dur:.0f}s@{start:.0f}: recall {len(hits)}/{len(truth)} "
          f"({100 * len(hits) / max(1, len(truth)):.0f}%), detections {len(seen)}")
    print(f"  name visible after word end: {q(lat)}")


if __name__ == '__main__':
    main()
