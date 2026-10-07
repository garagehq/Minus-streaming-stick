#!/usr/bin/env python3
"""CPU streaming harness (pip sherpa-onnx), same TSV output as zf_stream:
fed_s  decode_ms  token  token_ts   -- one line per newly emitted token.

usage: taskset -c 4 python3 stream_cpu.py kind MODELDIR WAV [max_s] [threads]
kind: zipformer | nemo_ctc
"""
import os
import sys
import time
import wave

import numpy as np
import sherpa_onnx

kind, d, wav = sys.argv[1:4]
max_s = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 1
common = dict(tokens=f'{d}/tokens.txt', num_threads=threads, enable_endpoint_detection=os.environ.get('NOEP') is None,
              rule1_min_trailing_silence=1.0, rule2_min_trailing_silence=0.6,
              rule3_min_utterance_length=15)
if kind == 'zipformer':
    r = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=f'{d}/encoder.onnx', decoder=f'{d}/decoder.onnx', joiner=f'{d}/joiner.onnx', **common)
else:
    r = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(model=f'{d}/model.int8.onnx', **common)

w = wave.open(wav)
sr = w.getframerate()
audio = np.frombuffer(w.readframes(min(w.getnframes(), int(max_s * sr))), dtype=np.int16).astype(np.float32) / 32768
step = sr // 50
s = r.create_stream()
emitted, seg_off, total, n_dec, mx = 0, 0.0, 0.0, 0, 0.0
for i in range(0, len(audio), step):
    s.accept_waveform(sr, audio[i:i + step])
    fed = (i + len(audio[i:i + step])) / sr
    ms = 0.0
    while r.is_ready(s):
        t0 = time.perf_counter()
        r.decode_stream(s)
        dt = (time.perf_counter() - t0) * 1000
        ms += dt; total += dt; n_dec += 1; mx = max(mx, dt)
    res = r.get_result_all(s)
    toks, ts = res.tokens, res.timestamps
    while emitted < len(toks):
        print(f"{fed:.3f}\t{ms:.1f}\t{toks[emitted]}\t{seg_off + (ts[emitted] if emitted < len(ts) else 0):.3f}")
        emitted += 1
    if r.is_endpoint(s):
        r.reset(s)
        emitted, seg_off = 0, fed
print(f"decodes {n_dec}, mean {total / max(n_dec, 1):.1f} ms, max {mx:.1f} ms, audio {len(audio) / sr:.1f} s, "
      f"total {total / 1000:.1f} s, RTF {total / 1000 / (len(audio) / sr):.3f}", file=sys.stderr)
