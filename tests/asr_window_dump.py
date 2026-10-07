#!/usr/bin/env python3
"""Dump per-window ASR words for offline matcher experiments.

Runs the production-style sliding window (WINDOW_S long, every HOP_S) over
a 16 kHz WAV and writes one JSON line per window:
  {"end": window end (s), "ms": inference ms, "words": [[w, start, end], ...]}
Word times are absolute (seconds into the WAV).

usage: python3 tests/asr_window_dump.py sensevoice|parakeet WAV OUT.jsonl \
         [start_s] [end_s] [hop_s] [window_s] [npu_core]
"""
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))

PARAKEET_DIR = os.environ.get(
    'MINUS_PARAKEET_DIR',
    '/home/radxa/asr_models/parakeet/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8')


def main():
    kind, wav, out = sys.argv[1:4]
    start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    end = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
    hop = float(sys.argv[6]) if len(sys.argv) > 6 else (0.25 if kind == 'parakeet' else 0.5)
    win = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
    core = int(sys.argv[8]) if len(sys.argv) > 8 else 2

    w = wave.open(wav)
    sr = w.getframerate()
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    end = min(end, len(audio) / sr)

    if kind == 'sensevoice':
        from sensevoice_npu import SenseVoiceNPU
        sv = SenseVoiceNPU(npu_core=core)

        def run(chunk, off):
            return [(x, s + off, e + off) for x, s, e in sv.words(chunk)]
    else:
        import sherpa_onnx
        sys.path.insert(0, HERE)
        from parakeet_eval import words_from
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=f'{PARAKEET_DIR}/model.int8.onnx', tokens=f'{PARAKEET_DIR}/tokens.txt',
            num_threads=1)

        def run(chunk, off):
            st = rec.create_stream()
            st.accept_waveform(sr, chunk)
            rec.decode_stream(st)
            return words_from(st.result, off)

    with open(out, 'w') as f:
        t = start + win
        while t <= end:
            off = t - win
            chunk = audio[int(off * sr):int(t * sr)]
            t0 = time.perf_counter()
            words = run(chunk, off)
            ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({'end': round(t, 3), 'ms': round(ms, 1),
                                'words': [[x, round(s, 3), round(e, 3)] for x, s, e in words]}) + '\n')
            t += hop


if __name__ == '__main__':
    main()
