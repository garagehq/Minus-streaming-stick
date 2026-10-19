"""SenseVoice-small on the RK3588 NPU, with per-word timings.

Model: happyme531/SenseVoiceSmall-RKNN2 (RKNN conversion of
lovemefan/SenseVoice-onnx), installed in MINUS_SENSEVOICE_DIR. The encoder
takes a fixed 171-frame input (~10 s of 60 ms LFR frames), so a call costs
the same ~0.38 s on one NPU core whatever the window length.

CTC decoding keeps each token's frame index, which gives word start/end
times at 60 ms resolution. On NBA commentary, word starts land 0.10-0.22 s
after the YouTube caption word times (p10-p90), far tighter than Moonshine.
"""

import os
import sys

import numpy as np

SENSEVOICE_DIR = os.environ.get('MINUS_SENSEVOICE_DIR', '/home/radxa/asr_models/sensevoice-rknn')
FRAME_S = 0.06       # LFR frame: 6 x 10 ms
N_PROMPT = 4         # language, event, emotion and text-norm query frames
LANG_EN = 4          # embedding row for English
NO_ITN = 15          # embedding row for "no inverse text normalisation"
WORD_TAIL_S = 0.4    # max a word runs past its last CTC spike


def is_available(model_dir: str = SENSEVOICE_DIR) -> bool:
    if not os.path.isfile(os.path.join(model_dir, 'sense-voice-encoder.rknn')):
        return False
    try:
        import rknnlite  # noqa: F401
        import kaldi_native_fbank  # noqa: F401
        return True
    except ImportError:
        return False


class SenseVoiceNPU:
    def __init__(self, npu_core: int = 1, model_dir: str = SENSEVOICE_DIR):
        sys.path.insert(0, model_dir)
        import sensevoice_rknn as sv
        import sentencepiece as spm
        from rknnlite.api import RKNNLite
        self._sv = sv
        self.front = sv.WavFrontend(os.path.join(model_dir, 'am.mvn'))
        self.emb = np.load(os.path.join(model_dir, 'embedding.npy'))
        self.rk = RKNNLite(verbose=False)
        if self.rk.load_rknn(os.path.join(model_dir, 'sense-voice-encoder.rknn')) != 0:
            raise RuntimeError('load_rknn failed')
        mask = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1,
                2: RKNNLite.NPU_CORE_2}.get(int(npu_core), RKNNLite.NPU_CORE_AUTO)
        if self.rk.init_runtime(core_mask=mask) != 0:
            raise RuntimeError(f'init_runtime failed on NPU core {npu_core}')
        self.sp = spm.SentencePieceProcessor(
            model_file=os.path.join(model_dir, 'chn_jpn_yue_eng_ko_spectok.bpe.model'))

    def words(self, audio: np.ndarray):
        """[(word, start_s, end_s)] for mono float32 16 kHz audio (<= ~10 s)."""
        feats = self.front.get_features(audio)[None, ...]
        q = np.concatenate([self.emb[[[LANG_EN]]], self.emb[[[1, 2]]], self.emb[[[NO_ITN]]],
                            feats * self._sv.SPEECH_SCALE], axis=1).astype(np.float32)
        n = q.shape[1]
        q = np.pad(q, ((0, 0), (0, self._sv.RKNN_INPUT_LEN - n), (0, 0)))
        out = self.rk.inference(inputs=[q])[0][0][:, 0, :]   # (1, vocab, 1, seq) -> (vocab, seq)
        ids = out.argmax(axis=0)[:n]
        words, prev = [], -1
        for f, t in enumerate(ids):
            if t != prev and t != 0:
                piece = self.sp.IdToPiece(int(t))
                frame = f - N_PROMPT
                if frame >= 0 and not piece.startswith('<|'):
                    ts = frame * FRAME_S
                    if piece.startswith('▁') or not words:
                        words.append([piece.lstrip('▁'), ts, ts + FRAME_S])
                    else:
                        words[-1][0] += piece
                        words[-1][2] = ts + FRAME_S
            prev = t
        # CTC fires one short spike per token, so the last spike is not where
        # the word ends. Run each word up to the next word's start, capped.
        out = []
        for i, (w, s, e) in enumerate(words):
            if not w:
                continue
            nxt = words[i + 1][1] if i + 1 < len(words) else e + WORD_TAIL_S
            out.append((w, s, max(e, min(nxt, e + WORD_TAIL_S))))
        return out

    def release(self):
        try:
            self.rk.release()
        except Exception:
            pass
