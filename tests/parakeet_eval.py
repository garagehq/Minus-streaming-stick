#!/usr/bin/env python3
"""Evaluate NVIDIA Parakeet (sherpa-onnx, CPU) for name detection.

Same method as tests/sensevoice_npu_eval.py: sliding windows over a 16 kHz
WAV, per-token timestamps grouped into words, "LeBron"/"King" matched with
the production NameMatcher, scored against a YouTube json3 caption track.

Offline models (window_s/hop_s apply):
  ctc   MODEL_DIR  -- e.g. sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8
  tdt   MODEL_DIR  -- e.g. sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
Streaming model (fed in real-time-sized chunks, latency = audio fed past the
word end when the name first appears):
  stream MODEL_DIR -- e.g. sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-streaming-240ms

usage: python3 tests/parakeet_eval.py KIND MODEL_DIR clip16k.wav captions.json3 \
         [window_s] [hop_s] [threads]
"""
import json
import os
import re
import statistics
import sys
import time
import wave

import numpy as np
import sherpa_onnx

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/src')
from name_mute import NameMatcher  # noqa: E402


def words_from(result, offset=0.0):
    """Group sherpa token timestamps into (word, start, end)."""
    toks, ts = list(result.tokens), list(result.timestamps)
    words = []
    for i, (t, s) in enumerate(zip(toks, ts)):
        if not t.strip() and not t.startswith(' '):
            continue
        if t.startswith((' ', '▁')) or not words:
            words.append([t.strip(' ▁'), s, s])
        else:
            words[-1][0] += t
        words[-1][2] = ts[i + 1] if i + 1 < len(ts) else s + 0.2
    return [(w, s + offset, e + offset) for w, s, e in words if w]


def load_truth(cap):
    truth = []
    for ev in json.load(open(cap)).get('events', []):
        for sg in ev.get('segs', []) or []:
            if re.search(r'lebron|\bking\b(?!s)', sg.get('utf8', ''), re.I):
                truth.append((ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000)
    return truth


def main():
    kind, mdir, wav, cap = sys.argv[1:5]
    win = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    hop = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    threads = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    f = lambda n: os.path.join(mdir, n)
    w = wave.open(wav)
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    dur = len(audio) / 16000
    truth = load_truth(cap)
    matcher = NameMatcher()
    seen, lat, texts, word_lag = [], [], [], []

    if kind in ('ctc', 'tdt'):
        if kind == 'ctc':
            rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=f('model.int8.onnx'), tokens=f('tokens.txt'), num_threads=threads)
        else:
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=f('encoder.int8.onnx'), decoder=f('decoder.int8.onnx'),
                joiner=f('joiner.int8.onnx'), tokens=f('tokens.txt'),
                num_threads=threads, model_type='nemo_transducer')
        s = 0.0
        while s + win <= dur:
            st = rec.create_stream()
            st.accept_waveform(16000, audio[int(s * 16000):int((s + win) * 16000)])
            t0 = time.time()
            rec.decode_stream(st)
            lat.append(time.time() - t0)
            ws = words_from(st.result, s)
            for a, b in matcher.find_word_spans(ws):
                if not any(abs(a - p) < 0.5 for p, _ in seen):
                    seen.append((a, b))
                    texts.append((round(a, 1), st.result.text[:60]))
            s += hop
    elif kind == 'stream':
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=f('encoder.int8.onnx'), decoder=f('decoder.int8.onnx'),
            joiner=f('joiner.int8.onnx'), tokens=f('tokens.txt'),
            num_threads=threads, decoding_method='greedy_search')
        st = rec.create_stream()
        chunk = 1600
        t_all = time.time()
        for i in range(0, len(audio), chunk):
            st.accept_waveform(16000, audio[i:i + chunk])
            t0 = time.time()
            while rec.is_ready(st):
                rec.decode_stream(st)
            lat.append(time.time() - t0)
            fed = (i + chunk) / 16000
            r = rec.get_result_all(st) if hasattr(rec, 'get_result_all') else None
            res = r if r is not None else st.result if hasattr(st, 'result') else None
            if res is None:
                continue
            ws = words_from(res)
            for a, b in matcher.find_word_spans(ws):
                if not any(abs(a - p) < 0.5 for p, _ in seen):
                    seen.append((a, b))
                    word_lag.append(fed - b)
                    texts.append((round(a, 1), ' '.join(x[0] for x in ws[-8:])[:60]))
        print(f"  streaming RTF {(time.time() - t_all) / dur:.3f}")
    else:
        sys.exit(f"unknown kind {kind}")

    found = [x for x in truth if any(abs(a - x) <= 1.0 for a, _ in seen)]
    false = [round(a, 1) for a, _ in seen if not any(abs(a - x) <= 1.5 for x in truth)]
    off = [min((a - x for a, _ in seen), key=abs) for x in found]
    q = lambda xs, u='s': (f"p10 {statistics.quantiles(xs, n=10)[0]:+.2f} / p50 {statistics.median(xs):+.2f}"
                           f" / p90 {statistics.quantiles(xs, n=10)[-1]:+.2f}{u}" if len(xs) >= 3 else str(xs))
    print(f"{kind} {os.path.basename(mdir.rstrip('/'))} window {win}s hop {hop}s threads {threads}: "
          f"recall {len(found)}/{len(truth)} ({100 * len(found) / max(1, len(truth)):.0f}%), "
          f"false {len(false)} {false[:8]}")
    print(f"  compute per call: p50 {statistics.median(lat) * 1000:.0f}ms "
          f"p90 {statistics.quantiles(lat, n=10)[-1] * 1000:.0f}ms")
    print(f"  detected word start vs caption word start: {q(off)}")
    if word_lag:
        print(f"  name visible after word end (audio-time): {q(word_lag)}")
    print("  samples:", texts[:5])


if __name__ == '__main__':
    main()
