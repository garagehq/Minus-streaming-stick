"""
ASR Worker Process — runs faster-whisper in a separate Python process so
we can hard-kill stuck inferences instead of just timing out.

Mirrors the OCR/VLM worker pattern (see src/vlm_worker.py for the
template that defines our hard-timeout safety story):
  - 'spawn' start method (no inherited fds / state from the parent)
  - load model once at startup, then process WAV file paths from a
    multiprocessing.Queue
  - parent uses a soft timeout (returns 'timeout' to caller but keeps
    the worker running, hoping it finishes) and a hard timeout (kill +
    restart). Three consecutive soft timeouts → hard kill.

Why this exists (history): the previous implementation in src/asr.py
shelled out to whisper.cpp's `whisper-cli` binary per inference. That
gave us hard-timeout for free (subprocess.run with timeout=). When we
swapped to faster-whisper for the speed/accuracy win documented in
docs/ASR.md, faster-whisper became an in-process Python lib — losing
that safety story. This worker restores it: faster-whisper runs in
a child process; if it hangs we kill the process; the parent thread
that requested the inference returns 'killed' and the next snapshot
goes through normally after the worker comes back up.

The keyword module (src/asr_keywords.py) is engine-agnostic, so the
swap from whisper.cpp to faster-whisper required no keyword changes.
"""

import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque
from multiprocessing import Process, Queue, Event

# 'spawn' so we don't inherit fds/state from the parent. Critical when
# the parent process uses multiprocessing internally (axengine for VLM)
# — 'fork' there causes "can only join a child process" errors.
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set by another worker import (VLM/OCR)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker process main loop
# ---------------------------------------------------------------------------

def resolve_engine() -> str:
    """MINUS_ASR_ENGINE, default 'sensevoice' (RK3588 NPU). Falls back to
    'moonshine' when the SenseVoice model or its runtime is missing."""
    engine = os.environ.get('MINUS_ASR_ENGINE', 'sensevoice').lower()
    if engine == 'sensevoice':
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sensevoice_npu import is_available
        if not is_available():
            log.warning("[ASRWorker] SenseVoice model/runtime missing; using moonshine")
            engine = 'moonshine'
    return engine


def _asr_worker_main(request_queue, response_queue, ready_event, shutdown_event,
                     model_name, cpu_threads, npu_core=None):
    """Worker process entrypoint.

    Args:
        request_queue:  multiprocessing.Queue carrying WAV file paths
                        (or None as a shutdown sentinel)
        response_queue: multiprocessing.Queue receiving
                        (status, transcript, elapsed_seconds) tuples
        ready_event:    set once the model is loaded and warmup is done
        shutdown_event: set by parent to request graceful shutdown
        model_name:     faster-whisper model identifier (e.g. 'tiny.en')
                        or a path to a pre-downloaded model directory
        cpu_threads:    threads for CTranslate2 inference
    """
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format='%(asctime)s [%(levelname)s] %(message)s')
    log = _logging.getLogger('ASRWorker')

    # Engine selection. Default 'faster-whisper' (10/10 corpus, ~1.1s on a
    # cool box). On a thermally-throttled box (4K passthrough pins the SoC
    # at the throttle trip → cores at 408MHz-1.6GHz) faster-whisper's fixed
    # 30s-padded encoder costs ~3.3s regardless of window. 'moonshine'
    # (moonshine-voice tiny-en, ONNX) processes audio PROPORTIONALLY (no 30s
    # pad), so a 5s window is ~1.1s even throttled — comfortably <2.5s. It
    # trades a little accuracy (~8/10 vs 10/10) but never false-positives on
    # show audio, which is what matters for the confirm/veto role.
    # DEFAULT IS 'moonshine': on the target hardware (RK3588 doing 4K
    # passthrough) the SoC thermally throttles cores to 408MHz-1.6GHz, where
    # faster-whisper's fixed 30s encoder costs ~4-5s (a hard floor that
    # streaming/smaller-chunks does NOT reduce — benchmarked). Moonshine is
    # the only engine that meets <2.5s here. Set MINUS_ASR_ENGINE=faster-whisper
    # on a cool/idle host to trade latency for the extra 2/10 corpus accuracy.
    engine = resolve_engine()

    # Honour "N CPUs max" by pinning the whole worker (and thus the engine's
    # threads) to a fixed core set. onnxruntime/CTranslate2 otherwise grab
    # every core. Overridable with MINUS_ASR_CPU_AFFINITY="3,4,5".
    # Default for the 8-core RK3588 (4×A55 little 0-3, 4×A76 big 4-7): pin to
    # {3,4,5}. Cores 4,5 (cpufreq policy4) stay the least thermally-throttled
    # big cores under load; core 6,7 (policy6) get parked at ~408MHz, so we
    # avoid them. Measured ~1.1s/window there vs slower on {5,6,7}.
    try:
        aff_env = os.environ.get('MINUS_ASR_CPU_AFFINITY', '')
        ncpu = os.cpu_count() or cpu_threads
        if aff_env.strip():
            cores = {int(c) for c in aff_env.split(',') if c.strip() != ''}
        elif ncpu >= 8:
            cores = set(range(3, 3 + cpu_threads))  # {3,4,5} on RK3588
        else:
            cores = set(range(max(0, ncpu - cpu_threads), ncpu))
        os.environ.setdefault('OMP_NUM_THREADS', str(cpu_threads))
        os.sched_setaffinity(0, cores)
        log.info(f"[ASRWorker] CPU affinity pinned to {sorted(cores)} (cpu_threads={cpu_threads})")
    except Exception as e:
        log.warning(f"[ASRWorker] could not set CPU affinity: {e}")

    try:
        load_start = time.time()
        import numpy as np

        if engine == 'sensevoice':
            # SenseVoice-small on the RK3588 NPU (src/sensevoice_npu.py):
            # ~0.38s per 3s window on one NPU core, near-zero CPU, per-word
            # timings from CTC frames.
            import wave as _wave
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from sensevoice_npu import SenseVoiceNPU
            log.info(f"[ASRWorker] Loading SenseVoice on NPU core {npu_core}...")
            _svn = SenseVoiceNPU(npu_core=npu_core if npu_core is not None
                                 else int(os.environ.get('MINUS_ASR_NPU_CORE', '1')))

            def _infer(wav_path):
                w = _wave.open(wav_path, 'rb')
                d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                w.close()
                words = _svn.words(d.astype(np.float32) / 32768.0)
                return ' '.join(x[0] for x in words), words

            warmup = lambda: _svn.words(np.zeros(16000, dtype=np.float32))
        elif engine == 'moonshine':
            import re as _re
            import wave as _wave
            import moonshine_voice as _mv
            from moonshine_voice import download as _mv_dl
            # Model size. Measured on LeBron commentary (2.5s windows, name
            # recall vs YouTube captions): TINY 42% @0.45s p50, BASE 58%
            # @0.53s, SMALL_STREAMING 81% @0.79s, MEDIUM_STREAMING 88% @1.10s
            # (p95 1.50s). Name muting needs recall, so MEDIUM is the default.
            _arch_name = os.environ.get('MINUS_ASR_MOONSHINE_ARCH',
                                        'MEDIUM_STREAMING').upper()
            _arch = getattr(_mv.ModelArch, _arch_name, _mv.ModelArch.MEDIUM_STREAMING)
            log.info(f"[ASRWorker] Loading moonshine ({_arch.name}, ONNX)...")
            _mpath, _ = _mv_dl.download_model_from_info(
                _mv_dl.find_model_info(language='en', model_arch=_arch))
            # word_timestamps: per-word (start, end) so the name muter can cut
            # just the name instead of the whole window.
            # vad_threshold: Moonshine gates audio through a Silero VAD that
            # is global (shared by every Transcriber) and stateful. On crowd-
            # heavy sports commentary it latches "no speech" and returns empty
            # transcripts in ~0.03s for whole minutes of play-by-play (50/64
            # windows empty at the default; 0.1-0.3 barely help). 0 hands
            # every window to the model: 9/64 empty, 16x the name hits.
            _ms = _mv.Transcriber(model_path=_mpath, model_arch=_arch, options={
                'word_timestamps': 'true',
                'vad_threshold': os.environ.get('MINUS_ASR_VAD_THRESHOLD', '0.0'),
            })
            _ts_re = _re.compile(r'\[[\d.]+s\]')  # moonshine inserts [1.23s] segment marks

            def _infer(wav_path):
                w = _wave.open(wav_path, 'rb')
                sr = w.getframerate()
                d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                w.close()
                audio = d.astype(np.float32) / 32768.0
                raw = _ms.transcribe_without_streaming(audio, sample_rate=sr)
                # moonshine returns a result object whose str() is the
                # transcript (with inline [1.23s] segment marks); str() works
                # whether raw is already a plain string or the result object.
                txt = _ts_re.sub(' ', str(raw)).replace('\n', ' ').strip()
                words = []
                for line in getattr(raw, 'lines', None) or []:
                    for w in getattr(line, 'words', None) or []:
                        words.append((w.word, float(w.start), float(w.end)))
                return txt, words

            warmup = lambda: _ms.transcribe_without_streaming(
                np.zeros(16000, dtype=np.float32), sample_rate=16000)
        else:
            log.info(f"[ASRWorker] Loading faster-whisper {model_name!r} "
                     f"(cpu_threads={cpu_threads}, compute_type=int8)...")
            from faster_whisper import WhisperModel
            _fw = WhisperModel(model_name, device='cpu', compute_type='int8',
                               cpu_threads=cpu_threads)

            def _infer(wav_path):
                segments, _info = _fw.transcribe(
                    wav_path, beam_size=1, language='en',
                    condition_on_previous_text=False, vad_filter=False)
                return ' '.join(s.text for s in segments).strip(), []

            warmup = lambda: list(_fw.transcribe(
                np.zeros(16000, dtype=np.float32), beam_size=1, language='en')[0])

        # Warmup primes the kernels so the first real call isn't slow.
        try:
            warmup()
        except Exception as e:
            log.warning(f"[ASRWorker] Warmup skipped: {e}")

        log.info(f"[ASRWorker] Model loaded in {time.time() - load_start:.1f}s "
                 f"(engine={engine}, ready)")
        ready_event.set()

        # Main request loop. `get(timeout=1.0)` returns control to check
        # shutdown_event between requests, so kill signals propagate
        # within ~1s even when no audio is arriving.
        while not shutdown_event.is_set():
            try:
                request = request_queue.get(timeout=1.0)
            except Exception:
                continue  # queue.Empty — loop back and re-check shutdown

            if request is None:
                # Sentinel: parent signaling shutdown via the queue.
                break

            wav_path = request
            start = time.time()
            try:
                text, words = _infer(wav_path)
                response_queue.put(('ok', text, time.time() - start, words))
            except FileNotFoundError as e:
                response_queue.put(('error', f'wav not found: {e}', time.time() - start))
            except Exception as e:
                response_queue.put(('error', f'inference failed: {e}',
                                    time.time() - start))

        log.info("[ASRWorker] Shutdown signaled, exiting cleanly")
    except Exception as e:
        log.error(f"[ASRWorker] Fatal during startup/loop: {e}", exc_info=True)
    finally:
        # Best-effort signal so the parent's start() wait returns even
        # on load failure (so it can retry/give up rather than block 60s).
        try:
            ready_event.set()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parent-side controller — hard-timeout via process kill
# ---------------------------------------------------------------------------


class ASRProcess:
    """Parent-side handle to the ASR worker. Provides transcribe() with
    soft + hard timeouts; kills + restarts the worker on hangs.

    Threading: all public methods are thread-safe via `_call_lock`.
    transcribe() can be called from the ASRManager's loop thread without
    additional locking on the caller side.
    """

    # Healthy faster-whisper tiny.en (CT2 int8) on 3 threads, measured
    # from tests/asr_corpus/bench.py against the 10-sample corpus:
    #   p50 ~1.10 s, max ~1.40 s per 5-second window — BUT that bench was
    #   run with the minus service STOPPED (cool SoC, cores at full 2.3GHz).
    # On a thermally-saturated box (e.g. 4K passthrough keeping the SoC at
    # ~84°C with the fan maxed → cores throttled to 408MHz-1.6GHz) the SAME
    # inference takes ~5-6s. Defaults are sized for the DEFAULT moonshine
    # engine, whose per-window latency on the throttled box is p50 ~1.1s /
    # p95 ~1.8s with occasional ~3.4s spikes during dense continuous speech.
    # SOFT 4.0 / HARD 6.0 give margin so real inferences COMPLETE (and
    # produce verdicts) rather than timing out, while still reaping a truly
    # hung worker. Env-overridable (e.g. tighten on a cool host running
    # faster-whisper, or widen further if you see timeout churn).
    #   MINUS_ASR_SOFT_TIMEOUT / MINUS_ASR_HARD_TIMEOUT (seconds)
    SOFT_TIMEOUT = float(os.environ.get('MINUS_ASR_SOFT_TIMEOUT', '4.0'))
    HARD_TIMEOUT = float(os.environ.get('MINUS_ASR_HARD_TIMEOUT', '6.0'))
    RESTART_THRESHOLD = 3  # Hard kill after this many consecutive soft timeouts
    WORKER_LOAD_TIMEOUT = 30.0  # tiny.en loads ~1s; allow plenty for first-time HF download
    # Periodic worker restart to bound the upstream moonshine_voice 0.0.59
    # transcript leak. Measured: ~3 MB of anonymous heap per
    # transcribe_without_streaming() call on real speech, because the C
    # library mallocs (well, `new`-allocates — confirmed: libc free crashes
    # with "invalid pointer", and Transcriber del+gc doesn't free either —
    # the leak lives in globally-cached ONNX runtime / libmoonshine.so
    # state, NOT in the Transcriber instance) a TranscriptC struct + line
    # text + audio_data float buffers + word structs on every call and
    # never exposes a `moonshine_free_transcript` symbol to free them.
    # Production observed: 11 GB anonymous heap after ~44k inferences
    # (~20 h continuous) which pushed the box to 88% memory and starved
    # other subsystems. The only way to release the memory is to exit the
    # process. We restart the worker every RESTART_AFTER_INFERENCES
    # successful calls: at 500 the worker peaks ~1.5 GB above baseline
    # before recycling (~28 min wall-clock at the production cadence),
    # well under any pressure threshold. Restart costs ~0.8 s of ASR
    # downtime (tiny.en reload from disk cache); ASR is a confirm-only
    # signal so OCR+VLM keep blocking ads through the gap. Override via
    # MINUS_ASR_RESTART_AFTER_INFERENCES (set to 0 to disable, e.g. for
    # tests or after the upstream bug is fixed).
    RESTART_AFTER_INFERENCES = int(
        os.environ.get('MINUS_ASR_RESTART_AFTER_INFERENCES', '500'))

    def __init__(self, model_name='tiny.en', cpu_threads=3, npu_core=None):
        self.model_name = model_name
        self.cpu_threads = cpu_threads
        self.npu_core = npu_core

        # Worker process state
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.ready_event = None
        self.shutdown_event = None
        self.is_ready = False

        # Bookkeeping for restart / timeout management
        self._restart_count = 0
        self._consecutive_timeouts = 0
        self._pending_response = False  # True if previous call timed out and worker may still answer
        # Counter for the periodic-restart leak workaround — see
        # RESTART_AFTER_INFERENCES.
        self._inferences_since_restart = 0
        # Per-word (word, start_s, end_s) timings from the last ok response.
        self.last_words = []
        # RLock (not Lock): transcribe() holds this lock and, on the hard-timeout
        # escalation path, calls restart() -> stop()/start(), which re-acquire it.
        # A plain Lock self-deadlocks there (the worker freezes mid-restart under
        # sustained load — observed as inference_count stuck + no 'killed' count).
        self._call_lock = threading.RLock()
        self._recent_latencies = deque(maxlen=20)

    # ----- lifecycle -----

    def start(self) -> bool:
        """Spawn the worker process and wait up to WORKER_LOAD_TIMEOUT
        seconds for the model to load. Returns True on success."""
        with self._call_lock:
            if self.process is not None and self.process.is_alive():
                if self.ready_event is not None and self.ready_event.is_set():
                    self.is_ready = True
                return self.is_ready

            self.request_queue = Queue()
            self.response_queue = Queue()
            self.ready_event = Event()
            self.shutdown_event = Event()

            self.process = Process(
                target=_asr_worker_main,
                args=(self.request_queue, self.response_queue,
                      self.ready_event, self.shutdown_event,
                      self.model_name, self.cpu_threads, self.npu_core),
                daemon=True,
                name=f'ASRWorker-{self.model_name}'
            )
            self.process.start()

        start = time.time()
        logger.info(f"[ASRProcess] Waiting for worker to load "
                    f"(PID {self.process.pid}, model={self.model_name})...")

        ok = self.ready_event.wait(timeout=self.WORKER_LOAD_TIMEOUT)
        if ok and self.process.is_alive():
            elapsed = time.time() - start
            self.is_ready = True
            logger.info(f"[ASRProcess] Worker ready in {elapsed:.1f}s")
            return True

        # Either timeout or worker died during load. Tear it down so
        # the next start() attempt gets a clean slate.
        logger.error(f"[ASRProcess] Worker failed to become ready "
                     f"after {self.WORKER_LOAD_TIMEOUT}s — killing")
        self._kill_process_locked()
        return False

    def stop(self):
        """Gracefully shut down the worker. Best-effort — falls through
        to terminate/kill if the worker is unresponsive."""
        with self._call_lock:
            if self.process is None:
                return

            if self.shutdown_event is not None:
                self.shutdown_event.set()
            # Queue a sentinel so a blocked `get(timeout=1)` loop wakes up
            try:
                self.request_queue.put_nowait(None)
            except Exception:
                pass

            if self.process.is_alive():
                self.process.join(timeout=3.0)
            if self.process.is_alive():
                logger.warning("[ASRProcess] Worker did not exit cleanly; terminating")
                self.process.terminate()
                self.process.join(timeout=2.0)
            if self.process.is_alive():
                logger.warning("[ASRProcess] Worker still alive after terminate; killing")
                self.process.kill()
                self.process.join(timeout=1.0)

            self.process = None
            self.is_ready = False

    def restart(self):
        """Kill + respawn the worker. Used on hard-timeout escalation."""
        logger.warning(f"[ASRProcess] Restarting worker "
                       f"(restart #{self._restart_count + 1})")
        self._restart_count += 1
        self.stop()
        return self.start()

    def _kill_process_locked(self):
        """Internal — must be called with `_call_lock` held."""
        if self.process is None:
            return
        try:
            self.process.kill()
            self.process.join(timeout=1.0)
        except Exception:
            pass
        self.process = None
        self.is_ready = False

    # ----- public inference API -----

    def transcribe(self, wav_path: str):
        """Synchronously transcribe a WAV file.

        Returns (status, transcript, elapsed_seconds) where:
          status == 'ok'      → inference completed
                 == 'timeout' → soft timeout fired; worker may still finish
                                later (we'll drain on next call). Returned
                                so the caller can keep going.
                 == 'killed'  → hard timeout fired; worker has been killed
                                and is restarting in the background. Caller
                                should not block on the next call.
                 == 'error'   → worker reported an inference exception OR
                                the worker isn't running. transcript holds
                                the error text for logging.

        Thread-safe via `_call_lock`. The lock also serializes the
        per-call bookkeeping (_pending_response, _consecutive_timeouts).
        """
        with self._call_lock:
            if not self.is_ready or self.process is None or not self.process.is_alive():
                return 'error', 'worker not running', 0.0

            # Drain any late response from a previous soft-timeout call.
            # If we don't, that response would be picked up as the answer
            # to the NEW request below — classic queue desync.
            self._drain_stale_responses_locked()

            if self._pending_response:
                # Worker is still busy with a previous inference. Don't
                # queue another; tell caller we're still waiting.
                if self._consecutive_timeouts >= self.RESTART_THRESHOLD:
                    logger.warning(
                        f"[ASRProcess] {self._consecutive_timeouts} consecutive "
                        f"soft timeouts — killing stuck worker")
                    self.restart()
                    self._pending_response = False
                    self._consecutive_timeouts = 0
                    return 'killed', '', 0.0
                return 'timeout', '', 0.0

            self.request_queue.put(wav_path)
            start = time.time()

            try:
                result = self.response_queue.get(timeout=self.SOFT_TIMEOUT)
            except Exception:
                # Soft timeout. Worker may still finish — leave
                # _pending_response True so we drain on next call.
                self._pending_response = True
                self._consecutive_timeouts += 1
                elapsed = time.time() - start

                # If we're past hard timeout, kill+restart immediately.
                if elapsed >= self.HARD_TIMEOUT or \
                        self._consecutive_timeouts >= self.RESTART_THRESHOLD:
                    logger.warning(
                        f"[ASRProcess] Hard timeout / restart threshold "
                        f"hit ({elapsed:.1f}s) — killing worker")
                    self.restart()
                    self._pending_response = False
                    self._consecutive_timeouts = 0
                    return 'killed', '', elapsed

                return 'timeout', '', elapsed

            # Got a response — reset timeout counter
            self._consecutive_timeouts = 0
            self._pending_response = False

            status, payload, latency = result[:3]
            # Worker 'ok' responses carry per-word timings as a 4th element.
            self.last_words = result[3] if len(result) > 3 else []
            if status == 'ok':
                self._recent_latencies.append(latency)
                self._inferences_since_restart += 1
                if (self.RESTART_AFTER_INFERENCES > 0 and
                        resolve_engine() == 'moonshine' and
                        self._inferences_since_restart
                        >= self.RESTART_AFTER_INFERENCES):
                    # Periodic restart to bound the upstream moonshine_voice
                    # 0.0.59 transcript leak (~3 MB/call, no exposed free).
                    # restart() is RLock-safe (the lock was upgraded to RLock
                    # specifically so stop()/start() can re-acquire it).
                    logger.info(
                        f"[ASRProcess] Periodic restart "
                        f"({self._inferences_since_restart} inferences) — "
                        f"bounds the upstream moonshine_voice transcript "
                        f"leak (~3 MB/call). ASR unavailable for ~0.8s.")
                    self._inferences_since_restart = 0
                    self.restart()
            return status, payload, latency

    # Hard cap on drain iterations. The drain loop SHOULD exit naturally
    # via queue.Empty after at most ~RESTART_THRESHOLD entries (one per
    # backlogged soft-timeout). But if the queue object misbehaves (a
    # tests-with-MagicMock surprise that killed the box once, or a real
    # multiprocessing.Queue corruption case), we never want an infinite
    # loop here — it'd OOM the parent + take down the whole service.
    _MAX_DRAIN_ITERATIONS = 32

    def _drain_stale_responses_locked(self):
        """Pull any leftover responses from a previous soft-timeout call.
        Returns immediately if the queue is empty. Hard-capped at
        _MAX_DRAIN_ITERATIONS to prevent infinite-loop OOM if the queue
        object somehow returns truthy non-empty results indefinitely."""
        drained = 0
        for _ in range(self._MAX_DRAIN_ITERATIONS):
            try:
                self.response_queue.get_nowait()
                drained += 1
            except Exception:
                break
        else:
            # Hit the cap without seeing Empty. Real-world this should
            # never happen — log loud so it's visible if it does.
            logger.warning(
                f"[ASRProcess] drain capped at {self._MAX_DRAIN_ITERATIONS} "
                f"iterations — queue may be misbehaving")
        if drained:
            logger.debug(f"[ASRProcess] Drained {drained} stale responses")
            # If we drained something, the previous pending_response is now resolved
            self._pending_response = False

    # ----- introspection -----

    def get_latency_stats(self) -> dict:
        latencies = list(self._recent_latencies)
        if not latencies:
            return {'samples': 0, 'p50_s': 0.0, 'p95_s': 0.0, 'max_s': 0.0}
        import statistics
        srt = sorted(latencies)
        p95_idx = max(0, int(0.95 * len(srt)) - 1)
        return {
            'samples': len(latencies),
            'p50_s': round(statistics.median(latencies), 3),
            'p95_s': round(srt[p95_idx], 3),
            'max_s': round(max(latencies), 3),
        }
