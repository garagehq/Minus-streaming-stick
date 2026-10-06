#!/usr/bin/env python3
"""Evaluate SenseVoice-small on the RK3588 NPU for name detection.

Runs the RKNN SenseVoice encoder (happyme531/SenseVoiceSmall-RKNN2) on
sliding windows of a 16 kHz WAV, decodes CTC with per-token frame times
(60 ms LFR frames), and scores "LeBron"/"King" detections against a YouTube
json3 caption track: recall, false alarms, inference time, and how far the
detected word start sits from the caption word start.

usage: python3 tests/sensevoice_npu_eval.py MODEL_DIR clip16k.wav captions.json3 \
         [window_s] [hop_s] [core]
"""
import json
import re
import statistics
import sys
import time
import wave

import numpy as np

MODEL_DIR = sys.argv[1]
sys.path.insert(0, MODEL_DIR)
import sensevoice_rknn as sv  # noqa: E402
from rknnlite.api import RKNNLite  # noqa: E402

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/src')
from name_mute import NameMatcher  # noqa: E402

FRAME_S = 0.06          # LFR frame: 6 x 10 ms
N_PROMPT = 4            # language, event, emotion, text-norm queries
LANG_EN = 4


class SenseVoiceNPU:
    def __init__(self, model_dir, core=RKNNLite.NPU_CORE_1):
        self.front = sv.WavFrontend(f'{model_dir}/am.mvn')
        self.emb = np.load(f'{model_dir}/embedding.npy')
        self.rk = RKNNLite(verbose=False)
        assert self.rk.load_rknn(f'{model_dir}/sense-voice-encoder.rknn') == 0
        assert self.rk.init_runtime(core_mask=core) == 0
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=f'{model_dir}/chn_jpn_yue_eng_ko_spectok.bpe.model')

    def words(self, audio):
        """Return [(word, start_s, end_s)] for a mono float32 16 kHz clip."""
        feats = self.front.get_features(audio)[None, ...]
        q = np.concatenate([self.emb[[[LANG_EN]]], self.emb[[[1, 2]]], self.emb[[[15]]],
                            feats * sv.SPEECH_SCALE], axis=1).astype(np.float32)
        n = q.shape[1]
        q = np.pad(q, ((0, 0), (0, sv.RKNN_INPUT_LEN - n), (0, 0)))
        out = self.rk.inference(inputs=[q])[0][0][:, 0, :]   # (1, vocab, 1, seq) -> (vocab, seq)
        ids = out.argmax(axis=0)[:n]
        toks = []            # (piece, frame)
        prev = -1
        for f, t in enumerate(ids):
            if t != prev and t != 0:
                toks.append((self.sp.IdToPiece(int(t)), f - N_PROMPT))
            prev = t
        words = []
        for piece, f in toks:
            if f < 0 or piece.startswith('<|'):
                continue
            t = f * FRAME_S
            if piece.startswith('▁') or not words:
                words.append([piece.lstrip('▁'), t, t + FRAME_S])
            else:
                words[-1][0] += piece
                words[-1][2] = t + FRAME_S
        return [(w, s, e) for w, s, e in words if w]


def main():
    wav, cap = sys.argv[2], sys.argv[3]
    win = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    hop = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    core = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2][
        int(sys.argv[6]) if len(sys.argv) > 6 else 1]
    m = SenseVoiceNPU(MODEL_DIR, core)
    w = wave.open(wav)
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    dur = len(audio) / 16000
    truth = []
    for ev in json.load(open(cap)).get('events', []):
        for sg in ev.get('segs', []) or []:
            if re.search(r'lebron|\bking\b(?!s)', sg.get('utf8', ''), re.I):
                truth.append((ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000)
    matcher = NameMatcher()
    seen, lat, texts = [], [], []
    s = 0.0
    while s + win <= dur:
        seg = audio[int(s * 16000):int((s + win) * 16000)]
        t0 = time.time()
        words = m.words(seg)
        lat.append(time.time() - t0)
        for a, b in matcher.find_word_spans(words):
            ws = s + a
            if not any(abs(ws - p) < 0.5 for p, _ in seen):
                seen.append((ws, s + b))
                texts.append((round(ws, 1), ' '.join(x[0] for x in words)[:60]))
        s += hop
    tol = 1.0
    found = [x for x in truth if any(abs(a - x) <= tol for a, _ in seen)]
    false = [round(a, 1) for a, _ in seen if not any(abs(a - x) <= 1.5 for x in truth)]
    off = [min((a - x for a, _ in seen), key=abs) for x in found]
    q = lambda xs: (f"p10 {statistics.quantiles(xs, n=10)[0]:+.2f} / p50 {statistics.median(xs):+.2f}"
                    f" / p90 {statistics.quantiles(xs, n=10)[-1]:+.2f}s" if len(xs) >= 3 else str(xs))
    print(f"window {win}s hop {hop}s: recall {len(found)}/{len(truth)} "
          f"({100 * len(found) / max(1, len(truth)):.0f}%), false {len(false)} {false[:8]}")
    print(f"  NPU inference per window: p50 {statistics.median(lat) * 1000:.0f}ms "
          f"p90 {statistics.quantiles(lat, n=10)[-1] * 1000:.0f}ms")
    print(f"  detected word start vs caption word start: {q(off)}")
    print("  samples:", texts[:6])


if __name__ == '__main__':
    main()
