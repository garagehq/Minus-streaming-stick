#!/usr/bin/env python3
"""Evaluate sherpa-onnx keyword spotting for the name muter.

Streams a 16 kHz WAV through a sherpa-onnx KeywordSpotter in 0.1 s chunks
and scores the detections against a YouTube json3 caption track: recall,
false alarms (detections > 1 s from any caption mention), and latency
(audio consumed when the keyword fired, minus the caption word start).

usage: python3 tests/kws_eval.py MODEL_DIR clip16k.wav captions.json3 [threshold] [score]
"""
import json
import re
import statistics
import sys
import time
import wave

import numpy as np
import sentencepiece as spm
import sherpa_onnx

KEYWORDS = ['LEBRON', 'LEBRON JAMES', 'KING', 'KING JAMES', 'LE BRON']


def main():
    mdir, wav, cap = sys.argv[1], sys.argv[2], sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    score = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    sp = spm.SentencePieceProcessor(model_file=f'{mdir}/bpe.model')
    kw_path = '/tmp/claude-1000/kws_keywords.txt'
    with open(kw_path, 'w') as f:
        for k in KEYWORDS:
            f.write(' '.join(sp.encode(k, out_type=str)) + f' @{k.replace(" ", "_")}\n')
    sfx = 'epoch-12-avg-2-chunk-16-left-64'
    kws = sherpa_onnx.KeywordSpotter(
        tokens=f'{mdir}/tokens.txt',
        encoder=f'{mdir}/encoder-{sfx}.int8.onnx',
        decoder=f'{mdir}/decoder-{sfx}.onnx',
        joiner=f'{mdir}/joiner-{sfx}.int8.onnx',
        keywords_file=kw_path, num_threads=2, provider='cpu',
        keywords_score=score, keywords_threshold=threshold, max_active_paths=4)
    w = wave.open(wav)
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    dur = len(audio) / 16000

    truth = []
    for ev in json.load(open(cap)).get('events', []):
        for sg in ev.get('segs', []) or []:
            if re.search(r'lebron|\bking\b(?!s)', sg.get('utf8', ''), re.I):
                truth.append((ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000)

    s = kws.create_stream()
    hits = []   # (audio time when fired, keyword)
    t0 = time.time()
    n = 1600
    for i in range(0, len(audio), n):
        s.accept_waveform(16000, audio[i:i + n])
        while kws.is_ready(s):
            kws.decode_stream(s)
            r = kws.get_result(s)
            if r:
                hits.append(((i + n) / 16000, r))
                kws.reset_stream(s)
    wall = time.time() - t0

    tol = 1.5
    found = [x for x in truth if any(0 <= h - x <= 3.0 for h, _ in hits)]
    false = [(round(h, 1), k) for h, k in hits if not any(-tol <= h - x <= 3.0 for x in truth)]
    lat = [min(h - x for h, _ in hits if 0 <= h - x <= 3.0) for x in found]
    q = lambda xs: (f"p50 {statistics.median(xs):.2f} / p90 {statistics.quantiles(xs, n=10)[-1]:.2f}"
                    f" / max {max(xs):.2f}s" if len(xs) >= 3 else str(xs))
    print(f"threshold {threshold} score {score}: recall {len(found)}/{len(truth)} "
          f"({100 * len(found) / max(1, len(truth)):.0f}%), false alarms {len(false)} "
          f"in {dur / 60:.1f} min, RTF {wall / dur:.3f}")
    print(f"  fired after caption word start: {q(lat)}")
    if false:
        print("  false:", false[:10])


if __name__ == '__main__':
    main()
