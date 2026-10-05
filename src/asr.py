"""
ASR-based ad-content confirmation/veto via faster-whisper.

Role in the detection stack:
  OCR  → reads text *on the screen*           — authoritative for blocking
  VLM  → classifies the *visual frame*        — primary AD/NO-AD signal
  ASR  → reads marketing-language *in speech* — confirms or vetoes VLM

ASR is intentionally NOT a primary trigger. It only:
  (1) CONFIRMS a VLM-alone block when marketing keywords are heard
      (becomes blocking_source "vlm+asr" — higher confidence label).
  (2) VETOES a VLM-alone block start, OR force-stops an active VLM-only
      block, when the audio is clearly show dialog with no marketing
      language (zero marker hits over the rolling window).
  (3) Returns "unknown" when it has no signal yet (cold start, music-only,
      whisper hallucinations on silence). "unknown" lets VLM fire normally
      — we never let an ASR outage prevent legitimate blocking.

The OCR-driven paths (`blocking_source` "ocr" / "both") are NOT consulted
against ASR. OCR text on the screen is authoritative.

Engine: faster-whisper tiny.en (CTranslate2 int8). See
docs/ASR.md for the benchmark that drove this choice — 25% faster than
whisper.cpp at the same 10/10 corpus accuracy with cleaner transcripts.

Safety architecture (matches OCR/VLM worker pattern):
  - Audio comes from a parallel `tee` branch on the existing
    AudioPassthrough GStreamer pipeline (see AudioASRTap in src/audio.py).
    The tap is leaky so a slow ASR can never backpressure the audio
    passthrough to the TV.
  - faster-whisper runs in a CHILD PROCESS (src/asr_worker.py
    ASRProcess), not in this thread. This restores the hard-timeout
    safety story we previously got "for free" from invoking whisper.cpp
    as a binary subprocess: if the worker hangs, we kill the OS process.
  - This module runs a Python *thread* that pulls snapshots, calls the
    worker, and updates the rolling history.

Cost on RK3588 (measured, see docs/ASR.md):
  - faster-whisper tiny.en, 3 threads, 5-second window: ~1.14 s
    inference latency. With a 2-second snapshot cadence that leaves
    ~75% CPU idle margin in the ASR worker.
  - Model load: ~1 s for cached, ~3 s on first HF download.
  - Memory: model + CT2 runtime + ring buffer ≈ 250 MB.

whisper-tiny mis-transcribes like OCR mis-reads. The keyword set in
`asr_keywords.py` is designed for that constraint: phrase-level
matching with multiple variants per phrase, regex shape-matchers for
prices/URLs/phone numbers, and an exclusion list for show-dialog
mentions of money. If quality is insufficient with tiny.en, swap the
model via MINUS_ASR_MODEL=base.en — keyword module needs no change.
"""

import logging
import os
import re
import threading
import time
from collections import deque
from typing import Optional

from asr_keywords import count_marker_hits, explain_hits
from asr_worker import ASRProcess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — overridable via env vars matching the existing MINUS_* pattern.
# ---------------------------------------------------------------------------
# Model name accepted by faster_whisper.WhisperModel(). Built-in sizes
# include 'tiny.en', 'base.en', 'small.en', 'medium.en', and 'large'.
# Can also be a path to a local model directory.
DEFAULT_ASR_MODEL = 'tiny.en'
ASR_MODEL = os.environ.get('MINUS_ASR_MODEL', DEFAULT_ASR_MODEL)


def is_asr_available() -> bool:
    """Whether the ASR backend is importable. Installs without
    faster-whisper installed will skip the tap branch entirely,
    keeping the audio pipeline byte-identical to pre-ASR shape."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


class ASRManager:
    """faster-whisper-driven ad-content confirmation/veto.

    Public API used by Minus:
      start()              → spawn worker + start background inference thread
      stop()               → signal shutdown
      verdict()            → 'confirm' / 'veto' / 'unknown'
      get_status()         → dict for /api/status surface
      is_enabled           → bool (set externally)
      enabled              → settable; False short-circuits verdict to 'unknown'

    Thread safety: `_history`, `last_*` attributes are guarded by `_lock`.
    `verdict()` is safe to call from the OCR/VLM decision loop hot path
    (single lock acquisition, O(window-size) iteration, no I/O).
    """

    # Rolling history window. Chosen to match the VLM sliding-window
    # cadence — 8s is "the last few iterations" so a single noisy hit
    # doesn't carry forever, and a streak of cleanness over 4+ seconds
    # provides enough evidence to veto.
    HISTORY_WINDOW_S = 8.0

    # How often to invoke whisper. Overlapping windows catch short bursts
    # of marketing copy that a non-overlapping grid might split. Env-
    # overridable (MINUS_ASR_INTERVAL).
    # Name muting needs every word heard, so the default is back-to-back
    # inference (no pause): with a 2.5s window and ~1.1s inference the
    # windows overlap by more than a word, so nothing falls between them.
    INFERENCE_INTERVAL_S = float(os.environ.get('MINUS_ASR_INTERVAL', '0.0'))
    # Minimum time between inference starts. Moonshine returns an empty
    # transcript in ~0.03s for windows with no speech; without a floor the
    # loop spins ~30x/s through silence, burning CPU and racing through the
    # worker's periodic leak-restart budget (each restart is ~2s deaf).
    MIN_CYCLE_S = float(os.environ.get('MINUS_ASR_MIN_CYCLE', '0.6'))

    # Audio window length per inference. Moonshine latency scales with how
    # much SPEECH is in the window — on dense continuous speech a 5s window
    # takes ~3.3s on the throttled RK3588 (too slow). A 2s window is ~1.6s
    # (p50), comfortably under 2s even on wall-to-wall casual speech.
    # Since ASR is now a CONFIRM-ONLY signal (the veto was removed), shorter
    # transcripts are low-cost: a CTA split across two windows still lands in
    # the 8s rolling history, and a missed confirm only changes the source
    # LABEL, never whether a block fires. Env-overridable (MINUS_ASR_WINDOW).
    WINDOW_SECONDS = float(os.environ.get('MINUS_ASR_WINDOW', '2.5'))

    def __init__(self, audio_tap, *, model_name: str = None, cpu_threads: int = 3):
        """audio_tap must be an AudioASRTap (see src/audio.py).
        It exposes `snapshot_to_wav(seconds)` returning True on success.
        """
        self._tap = audio_tap
        # Optional callback(transcript, words, window_start_mono) after each
        # successful inference. words: [(word, start_s, end_s)] relative to
        # the window start. Used by the name muter.
        self.on_words = None
        self._model_name = model_name or ASR_MODEL
        # What the worker actually loads (for logs/status): the Moonshine
        # arch, or the faster-whisper model name.
        if os.environ.get('MINUS_ASR_ENGINE', 'moonshine').lower() == 'moonshine':
            self.model_label = 'moonshine-' + os.environ.get(
                'MINUS_ASR_MOONSHINE_ARCH', 'MEDIUM_STREAMING').lower()
        else:
            self.model_label = self._model_name
        self._cpu_threads = cpu_threads

        # Worker process (hard-timeout via process kill — see asr_worker.py)
        self._process = ASRProcess(model_name=self._model_name,
                                   cpu_threads=self._cpu_threads)

        # Runtime state
        self.is_running = False
        self.enabled = True
        self.inference_count = 0
        self.timeout_count = 0
        self.failure_count = 0
        self.killed_count = 0
        self.last_transcript = ''
        self.last_marker_hits = 0
        self.last_inference_time = 0.0
        self.last_inference_latency = 0.0

        # Rolling history: (timestamp, marker_hits, transcript_alpha_word_count)
        self._history = deque(maxlen=32)
        self._lock = threading.RLock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----- lifecycle -----

    def start(self):
        if self.is_running:
            return
        if not is_asr_available():
            logger.warning("[ASR] faster-whisper not installed — ASR disabled")
            return

        if not self._process.start():
            logger.warning("[ASR] worker process failed to start — ASR disabled")
            return

        self._stop_event.clear()
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='ASR')
        self._thread.start()
        logger.info(f"[ASR] started (model={self.model_label}, "
                    f"window={self.WINDOW_SECONDS}s, interval={self.INFERENCE_INTERVAL_S}s, "
                    f"threads={self._cpu_threads})")

    def stop(self):
        if not self.is_running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.is_running = False
        # Stop the worker process
        try:
            self._process.stop()
        except Exception as e:
            logger.debug(f"[ASR] worker stop error: {e}")
        logger.info(f"[ASR] stopped after {self.inference_count} inferences "
                    f"(timeouts={self.timeout_count}, killed={self.killed_count}, "
                    f"failures={self.failure_count})")

    # ----- main loop -----

    def _loop(self):
        # Give the audio tap a moment to fill the ring buffer with at
        # least one window's worth of data. Without this we'd start
        # producing empty/short transcripts immediately on boot.
        if not self._stop_event.wait(self.WINDOW_SECONDS + 0.5):
            pass

        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                if not self.enabled:
                    self._stop_event.wait(self.INFERENCE_INTERVAL_S)
                    continue

                if not self._tap.snapshot_to_wav(self.WINDOW_SECONDS):
                    # Tap not ready (not enough audio yet, or stalled).
                    self._stop_event.wait(self.INFERENCE_INTERVAL_S)
                    continue

                # Capture time of the first sample in this window.
                window_start = (self._tap.last_snapshot_end_mono
                                - self.WINDOW_SECONDS)
                status, transcript, latency = self._process.transcribe(
                    self._tap.wav_path)
                self._record_result(status, transcript, latency)
                if status == 'ok' and self.on_words is not None:
                    try:
                        self.on_words(transcript,
                                      list(getattr(self._process, 'last_words', []) or []),
                                      window_start)
                    except Exception as e:
                        logger.error(f"[ASR] on_words callback failed: {e}")
            except Exception as e:
                logger.error(f"[ASR] loop iteration failed: {e}")
                self.failure_count += 1

            self._stop_event.wait(max(
                self.INFERENCE_INTERVAL_S,
                self.MIN_CYCLE_S - (time.monotonic() - cycle_start)))

    def _record_result(self, status: str, transcript: str, latency: float):
        hits = count_marker_hits(transcript) if status == 'ok' else 0
        now = time.time()
        with self._lock:
            self.inference_count += 1
            if status == 'timeout':
                self.timeout_count += 1
            elif status == 'killed':
                self.killed_count += 1
            elif status == 'error':
                self.failure_count += 1
            self.last_inference_time = now
            self.last_inference_latency = latency
            if status == 'ok':
                self.last_transcript = transcript
                self.last_marker_hits = hits
                alpha_word_count = len(re.findall(r'[a-z]{2,}', transcript.lower()))
                self._history.append((now, hits, alpha_word_count))
                cutoff = now - self.HISTORY_WINDOW_S
                while self._history and self._history[0][0] < cutoff:
                    self._history.popleft()

        if status == 'ok' and (hits > 0 or len(transcript) >= 20):
            preview = transcript[:80].replace('\n', ' ')
            if hits > 0:
                markers = explain_hits(transcript)[:5]
                logger.info(f"[ASR] hits={hits} ({latency:.2f}s) markers={markers} text='{preview}'")
            else:
                logger.debug(f"[ASR] hits=0 ({latency:.2f}s) text='{preview}'")
        elif status != 'ok':
            logger.debug(f"[ASR] inference {status} ({latency:.2f}s)")

    # ----- decision API -----

    def verdict(self) -> str:
        """Three-state output: 'confirm' / 'veto' / 'unknown'.

        Used by Minus._update_blocking_state for the VLM-alone start
        gate and the mid-block force-stop. See the module-level docstring
        for the role contract.

        Decision rule:
          - disabled or not running        → 'unknown' (never blocks VLM)
          - any marker hits in window      → 'confirm' (marketing heard)
          - no hits AND ≥1 inference with
            transcribed speech in window   → 'veto'   (show audio, not ad)
          - otherwise                      → 'unknown' (no useful signal)
        """
        if not self.is_running or not self.enabled:
            return 'unknown'
        now = time.time()
        with self._lock:
            cutoff = now - self.HISTORY_WINDOW_S
            recent = [h for h in self._history if h[0] >= cutoff]
        if not recent:
            return 'unknown'
        total_hits = sum(h[1] for h in recent)
        if total_hits > 0:
            return 'confirm'
        total_words = sum(h[2] for h in recent)
        if total_words >= 10:
            return 'veto'
        return 'unknown'

    # ----- test / diagnostics -----

    def test_transcribe(self, wav_path: str) -> dict:
        """Run a WAV through the live ASR worker and fold the result into the
        rolling state (so /api/status + the webui ASR-Live panel reflect it).

        Lets ASR be exercised end-to-end without live HDMI-RX audio (e.g. when
        the TV/HDMI-TX is off and the source sends digital silence). Starts the
        worker on demand if it isn't running yet.
        """
        if not is_asr_available():
            return {'success': False, 'error': 'faster-whisper/moonshine not installed'}
        # The worker is independent of the audio tap — it just transcribes a
        # file — so we can start it even if the audio pipeline never came up.
        if not self._process.is_ready:
            if not self._process.start():
                return {'success': False, 'error': 'ASR worker failed to start'}
            self.is_running = True
        status, text, latency = self._process.transcribe(wav_path)
        self._record_result(status, text, latency)
        hits = count_marker_hits(text) if status == 'ok' else 0
        return {
            'success': status == 'ok',
            'status': status,
            'transcript': text,
            'latency_s': round(latency, 3),
            'marker_hits': hits,
            'verdict': self.verdict(),
        }

    # ----- status surface -----

    def get_status(self) -> dict:
        with self._lock:
            history_len = len(self._history)
            recent_hits_sum = sum(h[1] for h in self._history)
        latency_stats = self._process.get_latency_stats() if self._process else {}
        return {
            'available': is_asr_available(),
            'enabled': self.enabled,
            'running': self.is_running,
            'engine': os.environ.get('MINUS_ASR_ENGINE', 'moonshine'),
            'model': self.model_label,
            'inference_count': self.inference_count,
            'timeout_count': self.timeout_count,
            'killed_count': self.killed_count,
            'failure_count': self.failure_count,
            'verdict': self.verdict(),
            'history_window_s': self.HISTORY_WINDOW_S,
            'history_entries': history_len,
            'recent_hits_sum': recent_hits_sum,
            'last_transcript': self.last_transcript[:200],
            'last_marker_hits': self.last_marker_hits,
            'last_latency_s': round(self.last_inference_latency, 3),
            'p50_latency_s': latency_stats.get('p50_s', 0.0),
            'p95_latency_s': latency_stats.get('p95_s', 0.0),
            'max_latency_s': latency_stats.get('max_s', 0.0),
            'latency_samples': latency_stats.get('samples', 0),
        }
