"""Training-clip collector for the name muter's ASR.

Saves short 16 kHz mono WAV clips of the broadcast audio, each with a JSON
sidecar, for fine-tuning the ASR on TV/sports commentary later:

  * every name detection (ASR, caption, or both), from PRE_S before to
    POST_S after the mention;
  * caption mentions the ASR did not catch: kind "caption_only", the most
    useful examples, because the ASR missed a name the captions confirm;
  * a random clip every RANDOM_INTERVAL_S of non-silent audio, so the
    dataset also has ordinary speech;
  * on request (request_clip, POST /api/name-mute/clips {"clip_at": t}):
    kind "nickname", used by tools/clip_farm.py when the playing video's
    captions say a LeBron nickname (kept for later training, not muted).

The sidecar records every ASR window transcript and every OCR line seen
during the clip, with times relative to the clip start. Text labels come
later, offline (captions where present, or a larger teacher model).

Clips live in screenshots/asr_clips/ under a byte budget (oldest evicted).
Env: MINUS_ASR_CLIPS=0 disables; MINUS_ASR_CLIPS_BUDGET_MB (10000);
MINUS_ASR_CLIPS_RANDOM_S (300, 0 disables random clips).
"""
import collections
import json
import logging
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger('Minus.ASRClips')


class ASRClipCollector:
    PRE_S = 6.0          # audio kept before the mention
    POST_S = 4.0         # and after it
    MERGE_S = 3.0        # detections this close are one mention
    SILENCE_RMS = 0.003  # random clips quieter than this are skipped
    CONTEXT_S = 60.0     # how long transcripts / OCR lines are remembered

    def __init__(self, tap, base_dir, engine: str = '', delay_fn=None):
        self.tap = tap
        self.dir = Path(base_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self._delay_fn = delay_fn
        self.enabled = os.environ.get('MINUS_ASR_CLIPS', '1') != '0'
        self.budget_bytes = int(float(os.environ.get('MINUS_ASR_CLIPS_BUDGET_MB', '10000')) * 1e6)
        self.random_interval_s = float(os.environ.get('MINUS_ASR_CLIPS_RANDOM_S', '300'))
        self._lock = threading.Lock()
        self._asr = collections.deque()       # (window_start_mono, transcript, words)
        self._ocr = collections.deque()       # (capture_mono, [lines])
        self._pending = []                    # dicts, see on_detection
        self._last_random = time.monotonic()
        self._stop = threading.Event()
        self._thread = None
        self.saved = collections.Counter()
        self.evicted = 0
        self.last_error = ''

    # ---- inputs (called from ASR / OCR / scheduler threads) -------------

    def note_asr(self, transcript, words, window_start):
        with self._lock:
            self._asr.append((window_start, transcript or '', list(words or [])))
            self._prune_locked(time.monotonic())

    def note_ocr(self, lines, capture_time):
        lines = [str(t).strip() for t in (lines or []) if str(t).strip()]
        if not lines:
            return
        with self._lock:
            if self._ocr and self._ocr[-1][1] == lines:
                return            # same screen text as the last frame
            self._ocr.append((capture_time, lines))
            self._prune_locked(time.monotonic())

    def on_detection(self, entry: dict):
        """NameMuteScheduler.on_detection hook: one call per detection."""
        if not self.enabled:
            return
        anchor = float(entry.get('capture_start') or time.monotonic())
        src = entry.get('source', '')
        label = entry.get('label', '')
        with self._lock:
            for p in self._pending:
                if p['kind'] != 'random' and abs(p['anchor'] - anchor) < self.MERGE_S:
                    p['sources'].add(src)
                    p['labels'].append([src, label, round(anchor - p['anchor'], 3)])
                    return
            self._pending.append({'kind': 'mention', 'anchor': anchor,
                                  'due': anchor + self.POST_S + 0.5,
                                  'sources': {src}, 'labels': [[src, label, 0.0]]})

    def request_clip(self, anchor: float, label: str = ''):
        """Save a clip around capture time `anchor` (time.monotonic())."""
        if not self.enabled:
            return
        with self._lock:
            for p in self._pending:
                if p['kind'] != 'random' and abs(p['anchor'] - anchor) < self.MERGE_S:
                    p['sources'].add('request')
                    p['labels'].append(['request', label, round(anchor - p['anchor'], 3)])
                    return
            self._pending.append({'kind': 'mention', 'anchor': anchor,
                                  'due': anchor + self.POST_S + 0.5,
                                  'sources': {'request'}, 'labels': [['request', label, 0.0]]})

    # ---- worker ----------------------------------------------------------

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='ASRClips', daemon=True)
        self._thread.start()
        logger.info(f"[ASRClips] collecting to {self.dir} "
                    f"(budget {self.budget_bytes / 1e6:.0f} MB, "
                    f"random every {self.random_interval_s:.0f}s)")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.wait(0.5):
            try:
                self._tick(time.monotonic())
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"[ASRClips] tick failed: {e}")

    def _tick(self, now):
        if self.random_interval_s > 0 and now - self._last_random >= self.random_interval_s:
            self._last_random = now
            with self._lock:
                busy = any(p['kind'] != 'random' for p in self._pending)
                if not busy:
                    anchor = now - self.POST_S
                    self._pending.append({'kind': 'random', 'anchor': anchor, 'due': now,
                                          'sources': set(), 'labels': []})
        with self._lock:
            due = [p for p in self._pending if p['due'] <= now]
            self._pending = [p for p in self._pending if p['due'] > now]
        for p in due:
            self._save(p)

    def _save(self, p):
        got = self.tap.recent_samples(time.monotonic() - (p['anchor'] - self.PRE_S))
        if got is None:
            return
        data, end_mono = got
        clip_start = end_mono - len(data) / self.tap.SAMPLE_RATE
        n = int((self.PRE_S + self.POST_S) * self.tap.SAMPLE_RATE)
        data = data[:n]
        if len(data) < n // 2:
            return
        rms = float(np.sqrt(np.mean((data.astype(np.float32) / 32768) ** 2)))
        if p['kind'] == 'random' and rms < self.SILENCE_RMS:
            return
        clip_end = clip_start + len(data) / self.tap.SAMPLE_RATE

        srcs = p['sources'] - {'request'}     # a requested clip that is also a mention keeps the mention kind
        if p['kind'] == 'random':
            kind = 'random'
        elif not srcs:
            kind = 'nickname'
        elif srcs == {'caption'}:
            kind = 'caption_only'     # captions saw the name, the ASR did not
        elif 'caption' in srcs:
            kind = 'both'
        else:
            kind = 'asr'

        with self._lock:
            asr = [{'t': round(ws - clip_start, 3), 'text': tr,
                    'words': [[w, round(ws - clip_start + s, 3), round(ws - clip_start + e, 3)]
                              for w, s, e in words]}
                   for ws, tr, words in self._asr if clip_start - 3.5 <= ws <= clip_end]
            ocr = [{'t': round(t - clip_start, 3), 'lines': lines}
                   for t, lines in self._ocr if clip_start <= t <= clip_end]

        stamp = time.strftime('%Y%m%d_%H%M%S')
        base = self.dir / f"{stamp}_{int((time.time() % 1) * 1000):03d}_{kind}"
        tmp = str(base) + '.wav.tmp'
        with wave.open(tmp, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.tap.SAMPLE_RATE)
            wf.writeframes(data.tobytes())
        os.replace(tmp, str(base) + '.wav')
        meta = {
            'kind': kind, 'created': time.time(), 'engine': self.engine,
            # Capture time of the first sample on the system-wide monotonic
            # clock, so other processes (tools/clip_farm.py play log) can
            # map the clip to a video position for labelling.
            'clip_start_mono': round(clip_start, 3),
            'duration_s': round(len(data) / self.tap.SAMPLE_RATE, 3),
            'mention_t': None if kind == 'random' else round(p['anchor'] - clip_start, 3),
            'detections': p['labels'], 'rms': round(rms, 4),
            'av_delay_s': round(float(self._delay_fn() or 0), 3) if self._delay_fn else None,
            'asr_windows': asr, 'ocr': ocr,
        }
        with open(str(base) + '.json', 'w') as f:
            json.dump(meta, f)
        self.saved[kind] += 1
        logger.debug(f"[ASRClips] saved {base.name} ({kind})")
        if sum(self.saved.values()) % 20 == 0:
            self._enforce_budget()

    def _enforce_budget(self):
        files = sorted(self.dir.glob('*.wav'), key=lambda f: f.stat().st_mtime)
        total = sum(f.stat().st_size for f in self.dir.iterdir() if f.is_file())
        while files and total > self.budget_bytes:
            f = files.pop(0)
            for g in (f, f.with_suffix('.json')):
                try:
                    total -= g.stat().st_size
                    g.unlink()
                except FileNotFoundError:
                    pass
            self.evicted += 1

    def _prune_locked(self, now):
        while self._asr and now - self._asr[0][0] > self.CONTEXT_S:
            self._asr.popleft()
        while self._ocr and now - self._ocr[0][0] > self.CONTEXT_S:
            self._ocr.popleft()

    def set_random_interval(self, seconds: float):
        """Random-clip cadence (0 disables). tools/clip_farm.py raises it
        while it plays captioned videos, since every clip is labelable then."""
        self.random_interval_s = max(0.0, float(seconds))
        self._last_random = time.monotonic()
        logger.info(f"[ASRClips] random clip interval -> {self.random_interval_s:.0f}s")

    def get_status(self) -> dict:
        try:
            wavs = list(self.dir.glob('*.wav'))
            size = sum(f.stat().st_size for f in self.dir.iterdir() if f.is_file())
        except OSError:
            wavs, size = [], 0
        return {'enabled': self.enabled, 'dir': str(self.dir), 'clips_on_disk': len(wavs),
                'bytes_on_disk': size, 'budget_bytes': self.budget_bytes,
                'saved_this_run': dict(self.saved), 'evicted': self.evicted,
                'random_interval_s': self.random_interval_s,
                'pending': len(self._pending), 'last_error': self.last_error}
