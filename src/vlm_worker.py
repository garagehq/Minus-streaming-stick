"""
VLM Worker Process - runs VLM in separate process for hard timeout capability.

This allows us to actually KILL stuck VLM inference instead of just timing out.
"""

import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Event

# Use 'spawn' start method to avoid inherited file descriptors and process state issues
# This is especially important when the parent process uses multiprocessing internally
# (like axengine's NPU runtime), as 'fork' can cause "can only join a child process" errors
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Set environment before any imports
os.environ['PYTORCH_MATCHER_LOGLEVEL'] = 'WARNING'
os.environ['TORCH_LOGS'] = '-all'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


def _vlm_worker_main(request_queue, response_queue, ready_event, shutdown_event):
    """
    Main function for VLM worker process.

    Loads model once, then processes requests until shutdown.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('VLMWorker')

    try:
        load_start = time.time()

        # Import VLM manager
        sys.path.insert(0, '/home/radxa/Minus/src')
        from vlm import VLMManager

        # Load model
        logger.info("[VLMWorker] Loading model...")
        vlm = VLMManager()
        if not vlm.load_model():
            logger.error("[VLMWorker] Failed to load model")
            return

        model_load_time = time.time() - load_start
        logger.info(f"[VLMWorker] Model loaded in {model_load_time:.1f}s, running warmup...")

        # Extended warmup to ensure NPU is fully ready for real frames
        # 4 inferences with varying content to warm all code paths
        try:
            import numpy as np
            from PIL import Image
            warmup_path = '/tmp/vlm_warmup.jpg'

            for i in range(4):
                # Create varied warmup images (noise, gradients, edges)
                if i == 0:
                    # Pure noise
                    warmup_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                elif i == 1:
                    # Gradient (simulates video content)
                    arr = np.zeros((512, 512, 3), dtype=np.uint8)
                    arr[:, :, 0] = np.linspace(0, 255, 512).reshape(1, -1).astype(np.uint8)
                    arr[:, :, 1] = np.linspace(255, 0, 512).reshape(-1, 1).astype(np.uint8)
                    warmup_img = Image.fromarray(arr)
                elif i == 2:
                    # High contrast edges (simulates text/UI)
                    arr = np.zeros((512, 512, 3), dtype=np.uint8)
                    arr[::2, :, :] = 255
                    warmup_img = Image.fromarray(arr)
                else:
                    # Mixed content
                    warmup_img = Image.fromarray(np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8))

                warmup_img.save(warmup_path, quality=80)
                start_w = time.time()
                _ = vlm.detect_ad(warmup_path)
                logger.debug(f"[VLMWorker] Warmup {i+1}/4: {time.time() - start_w:.2f}s")

            total_time = time.time() - load_start
            logger.info(f"[VLMWorker] Warmup complete (4 inferences), total startup: {total_time:.1f}s")
        except Exception as e:
            logger.warning(f"[VLMWorker] Warmup failed (non-fatal): {e}")

        logger.info("[VLMWorker] Model loaded, ready for requests")
        ready_event.set()

        # Process requests with keepalive to prevent NPU cold-start
        last_inference_time = time.time()
        KEEPALIVE_INTERVAL = 20.0  # Run keepalive if idle for 20s
        warmup_path = '/tmp/vlm_warmup.jpg'

        while not shutdown_event.is_set():
            try:
                # Wait for request with timeout so we can check shutdown and keepalive
                try:
                    request = request_queue.get(timeout=1.0)
                except:
                    # No request - check if we need keepalive
                    if time.time() - last_inference_time > KEEPALIVE_INTERVAL:
                        try:
                            _ = vlm.detect_ad(warmup_path)
                            last_inference_time = time.time()
                            logger.debug("[VLMWorker] Keepalive inference completed")
                        except:
                            pass
                    continue

                if request is None:  # Shutdown signal
                    break

                # detect_ad: (image_path, 'detect_ad')
                # query:     (image_path, prompt, max_new_tokens, 'query')
                request_type = request[-1]

                if request_type == 'detect_ad':
                    image_path = request[0]
                    result = vlm.detect_ad(image_path)
                    response_queue.put(('ok', result))
                    last_inference_time = time.time()
                elif request_type == 'query':
                    image_path, prompt, mnt, _ = request
                    result = vlm.query_image(image_path, prompt, max_new_tokens=mnt)
                    response_queue.put(('ok', result))
                    last_inference_time = time.time()
                else:
                    response_queue.put(('error', 'Unknown request type'))

            except Exception as e:
                logger.error(f"[VLMWorker] Error processing request: {e}")
                response_queue.put(('error', str(e)))

        logger.info("[VLMWorker] Shutting down")
        vlm.release()

    except Exception as e:
        import traceback
        print(f"[VLMWorker] Fatal error: {e}")
        traceback.print_exc()


class VLMProcess:
    """
    Manages VLM in a separate process with hard timeout capability.

    If inference takes longer than timeout, the process is KILLED and restarted.
    """

    # Timeouts sized for the LFM2.5-VL fused-prefill path. Both detect_ad
    # AND query_image are PREFILL-ONLY (no autoregressive decode loop):
    # detect_ad uses argmax over YES/NO last-token logits, query_image
    # picks the highest-logit class over PLAYING/PAUSED/DIALOG/MENU/
    # SCREENSAVER first-tokens. Compute is deterministic ~0.37s for both.
    #
    # Historical context: the previous FastVLM-1.5B path was decode-based
    # and could spend 0.7s..15s on a single inference (the descriptive-
    # paragraph pathology in docs/VLM_NPU_DEGRADATION.md), which forced
    # SOFT=1.5 / HARD=5.0 / P95=3.0. That pathology CANNOT occur without
    # a decode loop, so anything past the current bounds means a
    # genuinely stuck NPU which we want to reap faster.
    #
    # Measured on LFM2.5-VL:
    #   detect_ad   : p50 0.31s / p95 0.33s / max 0.33s, 0 events >1s/day
    #   query_image : p50 0.39s / p95 0.41s — same prefill loop, just
    #                 different last-token logit lookups
    # SOFT_TIMEOUT=1.5 leaves >3x headroom over both paths; HARD_TIMEOUT=
    # 2.0 still ~5x p95 so zero false-kill risk, but a genuinely stuck
    # NPU is reaped 3s sooner than the old 5.0.
    SOFT_TIMEOUT = 1.5   # Return "timeout" after this, but don't kill
    HARD_TIMEOUT = 2.0   # Only kill if inference takes longer than this
    RESTART_THRESHOLD = 3  # Restart after this many consecutive soft timeouts

    # Latency-based auto-recovery (defense-in-depth). With LFM2.5-VL's
    # prefill-only paths the NPU-drift-to-15s pathology can no longer
    # arise, so this should never fire in normal operation; it only
    # guards a genuinely hung NPU. Trigger 2.0s is unreachable without a
    # real hang at healthy P95 ~0.4s, so faster recovery carries no
    # false-restart risk.
    LATENCY_WINDOW = 10          # Rolling sample size for trend detection
    LATENCY_P95_TRIGGER = 2.0    # P95 latency (s) that triggers auto-recovery
    RECOVERY_COOLDOWN = 60.0     # Min seconds between auto-recoveries
    DEEP_RESTART_BACKOFF = 8.0   # Longer NPU-release delay when simple restart didn't help

    def __init__(self):
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.ready_event = None
        self.shutdown_event = None
        self.is_ready = False
        self._restart_count = 0
        self._last_restart_time = 0
        self._consecutive_timeouts = 0
        self._pending_response = False  # True if we're waiting for a slow response
        # Rolling latencies of successful inferences for auto-recovery detection
        from collections import deque
        self._recent_latencies = deque(maxlen=self.LATENCY_WINDOW)
        self._last_auto_recovery_time = 0.0
        # Serializes detect_ad and query_image calls from different threads
        # (detection loop vs. autonomous mode). Upstream's tuple-shape guards
        # already tolerate stale cross-pollinated responses, but the lock
        # also protects shared mutable state (_consecutive_timeouts,
        # _pending_response, _recent_latencies) from concurrent mutation.
        import threading
        self._call_lock = threading.Lock()

    def start(self):
        """Start the VLM worker process."""
        if self.process is not None and self.process.is_alive():
            # Process running - check if ready
            if self.ready_event is not None and self.ready_event.is_set():
                self.is_ready = True
            return self.is_ready

        # Create communication queues
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.ready_event = Event()
        self.shutdown_event = Event()

        # Start worker process
        self.process = Process(
            target=_vlm_worker_main,
            args=(self.request_queue, self.response_queue, self.ready_event, self.shutdown_event),
            daemon=True
        )
        self.process.start()

        import logging
        logger = logging.getLogger('Minus.VLM')
        # Monotonic, not wall time: at boot NTP can step the clock by days,
        # which made this report waits like "554611.9s".
        start_time = time.monotonic()
        logger.info(f"[VLMProcess] Waiting for model to load (PID {self.process.pid})...")

        # Wait for model to load (up to 60s - model + warmup can take 40s under load).
        # Poll so a worker that exits early (e.g. no Axera card plugged in) fails
        # in seconds instead of holding startup for the full 60s on every retry.
        while time.monotonic() - start_time < 60.0:
            if self.ready_event.wait(timeout=0.5):
                elapsed = time.monotonic() - start_time
                logger.info(f"[VLMProcess] Worker ready after {elapsed:.1f}s")
                self.is_ready = True
                return True
            if not self.process.is_alive():
                elapsed = time.monotonic() - start_time
                logger.error(f"[VLMProcess] Worker exited (code {self.process.exitcode}) "
                             f"before becoming ready after {elapsed:.1f}s")
                self.kill()
                return False

        elapsed = time.monotonic() - start_time
        logger.error(f"[VLMProcess] Worker failed to become ready after {elapsed:.1f}s - killing")
        self.kill()
        return False

    def kill(self):
        """Kill the VLM worker process."""
        if self.process is not None:
            self.shutdown_event.set()
            # Try graceful shutdown first
            self.process.terminate()
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                # Force kill if still running
                self.process.kill()
                self.process.join(timeout=1.0)
            self.process = None
        self.is_ready = False

    def restart(self):
        """Kill and restart the VLM worker.

        Includes delay to allow NPU resources to be released properly.
        The Axera NPU can get into a bad state if we restart too quickly
        after killing a process that was using NPU resources.
        """
        import logging
        logger = logging.getLogger('Minus.VLM')

        self._restart_count += 1

        # Fixed 2 second delay for NPU recovery
        backoff = 2.0

        self.kill()

        # Clear any stale responses from killed worker
        if self.response_queue is not None:
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except:
                    break

        # Reset timeout counter after restart
        self._consecutive_timeouts = 0

        # Wait for NPU resources to be released
        time.sleep(backoff)

        self._last_restart_time = time.time()
        return self.start()

    def _record_latency(self, elapsed_s):
        """Record a successful inference latency for trend analysis."""
        try:
            self._recent_latencies.append(float(elapsed_s))
        except (TypeError, ValueError):
            pass

    def _maybe_auto_recover(self):
        """If recent inference latencies show degraded performance, restart the worker.

        Triggered when P95 of the rolling window exceeds LATENCY_P95_TRIGGER.
        Subject to RECOVERY_COOLDOWN to avoid thrashing.

        If a recovery fired recently (within cooldown) and we're still slow, escalate
        to a "deep" restart with a longer NPU-release backoff.
        """
        if len(self._recent_latencies) < self.LATENCY_WINDOW:
            return
        samples = sorted(self._recent_latencies)
        p95_idx = max(0, int(len(samples) * 0.95) - 1)
        p95 = samples[p95_idx]
        if p95 <= self.LATENCY_P95_TRIGGER:
            return
        now = time.time()
        since_last = now - self._last_auto_recovery_time
        if since_last < self.RECOVERY_COOLDOWN:
            return
        import logging
        logger = logging.getLogger('Minus.VLM')
        # If our previous recovery was recent-ish (<3 min) and we're degraded
        # again, the simple restart isn't clearing NPU state — use deep restart.
        deep = since_last < 180.0 and self._last_auto_recovery_time > 0
        if deep:
            logger.warning(
                f"[VLMProcess] Auto-recovery: P95={p95:.1f}s over last "
                f"{len(samples)} queries; previous restart at {since_last:.0f}s "
                f"ago didn't help — DEEP restart (backoff={self.DEEP_RESTART_BACKOFF}s)"
            )
            self.kill()
            if self.response_queue is not None:
                while not self.response_queue.empty():
                    try:
                        self.response_queue.get_nowait()
                    except Exception:
                        break
            self._consecutive_timeouts = 0
            time.sleep(self.DEEP_RESTART_BACKOFF)
            self._restart_count += 1
            self._last_restart_time = time.time()
            self.start()
        else:
            logger.warning(
                f"[VLMProcess] Auto-recovery: P95={p95:.1f}s over last "
                f"{len(samples)} queries exceeds {self.LATENCY_P95_TRIGGER}s — restarting worker"
            )
            self.restart()
        self._last_auto_recovery_time = time.time()
        self._recent_latencies.clear()

    def get_latency_stats(self):
        """Return recent inference latency stats for observability."""
        if not self._recent_latencies:
            return {'samples': 0}
        samples = sorted(self._recent_latencies)
        n = len(samples)
        p50 = samples[n // 2]
        p95_idx = max(0, int(n * 0.95) - 1)
        p95 = samples[p95_idx]
        return {
            'samples': n,
            'p50_s': round(p50, 3),
            'p95_s': round(p95, 3),
            'max_s': round(samples[-1], 3),
            'auto_recoveries_last_time': self._last_auto_recovery_time,
        }

    def detect_ad(self, image_path):
        """
        Run ad detection with soft/hard timeout.

        Soft timeout (1.5s): Returns immediately but doesn't kill worker
        Hard timeout (5.0s): Kills and restarts worker

        Returns: (is_ad, response, elapsed, confidence)
        If soft timeout: returns (False, "TIMEOUT", timeout, 0.0)
        If hard timeout: returns (False, "KILLED", timeout, 0.0)
        """
        with self._call_lock:
            return self._detect_ad_locked(image_path)

    def _detect_ad_locked(self, image_path):
        import logging
        logger = logging.getLogger('Minus.VLM')

        if not self.is_ready or self.process is None or not self.process.is_alive():
            if not self.start():
                return False, "VLM not ready", 0, 0.0

        # Drain ALL stale responses before sending a new request.
        #
        # This is the critical fix for the queue-desync bug: a prior soft
        # timeout left a request in flight; that request eventually completed
        # and its result is sitting in response_queue. If we just put a new
        # request and call get(), FIFO semantics return the OLD result as
        # the answer to our NEW request — every subsequent call would then
        # be off-by-one indefinitely. Drain anything that's already there.
        drained = 0
        while True:
            try:
                self.response_queue.get_nowait()
                drained += 1
            except Exception:
                break
        if drained:
            logger.debug(f"[VLMProcess] Drained {drained} stale response(s)")
            self._pending_response = False
            self._consecutive_timeouts = 0

        # If a request is still genuinely in flight (worker hasn't pushed
        # its response yet), DO NOT send another. Sending now would create
        # two requests in flight and re-introduce the off-by-one. Caller
        # treats this as a skipped frame.
        if self._pending_response:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self.RESTART_THRESHOLD:
                logger.warning(
                    f"[VLMProcess] {self._consecutive_timeouts} consecutive pending — restarting worker"
                )
                self.restart()
                self._pending_response = False
                return False, "KILLED", 0, 0.0
            return False, "PENDING", 0, 0.0

        start_time = time.time()

        # Send request
        self.request_queue.put((image_path, 'detect_ad'))

        # Wait for response with soft timeout first
        try:
            status, result = self.response_queue.get(timeout=self.SOFT_TIMEOUT)
            elapsed = time.time() - start_time

            if status == 'ok':
                # Reset consecutive timeout counter on success
                self._consecutive_timeouts = 0
                self._pending_response = False
                # Record inference latency for degradation detection. Use the
                # elapsed time the worker reported if present (4-tuple or 2-tuple),
                # else our wall-clock elapsed.
                latency = elapsed
                if isinstance(result, tuple):
                    if len(result) >= 3:  # detect_ad: (is_ad, text, elapsed, conf)
                        latency = result[2]
                    elif len(result) == 2:  # query: (text, elapsed)
                        latency = result[1]
                self._record_latency(latency)
                self._maybe_auto_recover()
                # Guard: detect_ad returns 4-tuple; if we got a 2-tuple (stale query
                # response leaked through), synthesize a 4-tuple.
                if isinstance(result, tuple) and len(result) == 2:
                    response_text, r_elapsed = result
                    return False, response_text, r_elapsed, 0.0
                return result
            else:
                return False, f"Error: {result}", elapsed, 0.0

        except:
            # SOFT TIMEOUT - don't kill yet, just skip this frame
            elapsed = time.time() - start_time
            self._consecutive_timeouts += 1
            self._pending_response = True

            # Check if we should do hard kill (only after many consecutive timeouts)
            if self._consecutive_timeouts >= self.RESTART_THRESHOLD:
                # Try waiting a bit longer for hard timeout before killing
                try:
                    remaining = self.HARD_TIMEOUT - elapsed
                    if remaining > 0:
                        status, result = self.response_queue.get(timeout=remaining)
                        # Got a response, reset counters
                        self._consecutive_timeouts = 0
                        self._pending_response = False
                        if status == 'ok':
                            elapsed = time.time() - start_time
                            logger.info(f"[VLMProcess] Slow response arrived after {elapsed:.1f}s")
                            return result
                except:
                    pass

                # Still no response after hard timeout, restart
                elapsed = time.time() - start_time
                logger.warning(
                    f"[VLMProcess] HARD KILL after {elapsed:.1f}s ({self._consecutive_timeouts} timeouts) - restarting worker"
                )
                self.restart()
                self._pending_response = False
                return False, "KILLED", elapsed, 0.0
            else:
                logger.debug(
                    f"[VLMProcess] Soft timeout after {elapsed:.1f}s (#{self._consecutive_timeouts}) - skipping frame"
                )
                return False, "TIMEOUT", elapsed, 0.0

    def query_image(self, image_path, prompt, max_new_tokens=8):
        """
        Run a custom prompt against an image (e.g. autonomous mode screen classification).

        max_new_tokens defaults to 8 (fits the autonomous-mode multi-choice
        prompt). Raise explicitly for open-ended questions, knowing the
        end-to-end latency rises ~0.23s per allowed token.

        Returns: (response_text, elapsed)
        On soft timeout: ("TIMEOUT", elapsed)
        On hard timeout/error: ("KILLED", elapsed)
        """
        with self._call_lock:
            return self._query_image_locked(image_path, prompt, max_new_tokens)

    def _query_image_locked(self, image_path, prompt, max_new_tokens):
        import logging
        logger = logging.getLogger('Minus.VLM')

        if not self.is_ready or self.process is None or not self.process.is_alive():
            if not self.start():
                return "VLM not ready", 0.0

        # Drain ALL stale responses before sending a new request. See the
        # detailed explanation in _detect_ad_locked — same off-by-one queue
        # desync applies here, since detect_ad and query_image share the
        # same request/response queues.
        drained = 0
        while True:
            try:
                self.response_queue.get_nowait()
                drained += 1
            except Exception:
                break
        if drained:
            logger.debug(f"[VLMProcess] Drained {drained} stale response(s)")
            self._pending_response = False
            self._consecutive_timeouts = 0

        if self._pending_response:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self.RESTART_THRESHOLD:
                logger.warning(
                    f"[VLMProcess] {self._consecutive_timeouts} consecutive pending — restarting worker"
                )
                self.restart()
                self._pending_response = False
                return "KILLED", 0.0
            return "PENDING", 0.0

        start_time = time.time()
        self.request_queue.put((image_path, prompt, max_new_tokens, 'query'))

        try:
            status, result = self.response_queue.get(timeout=self.SOFT_TIMEOUT)
            elapsed = time.time() - start_time

            if status == 'ok':
                self._consecutive_timeouts = 0
                self._pending_response = False
                latency = elapsed
                if isinstance(result, tuple):
                    if len(result) >= 3:
                        latency = result[2]
                    elif len(result) == 2:
                        latency = result[1]
                self._record_latency(latency)
                self._maybe_auto_recover()
                # Guard: query returns 2-tuple; if we got a 4-tuple (stale detect_ad
                # response leaked through), unpack and keep only the text.
                if isinstance(result, tuple) and len(result) == 4:
                    _is_ad, response_text, r_elapsed, _conf = result
                    return response_text, r_elapsed
                return result
            else:
                return f"Error: {result}", elapsed

        except:
            elapsed = time.time() - start_time
            self._consecutive_timeouts += 1
            self._pending_response = True

            if self._consecutive_timeouts >= self.RESTART_THRESHOLD:
                try:
                    remaining = self.HARD_TIMEOUT - elapsed
                    if remaining > 0:
                        status, result = self.response_queue.get(timeout=remaining)
                        self._consecutive_timeouts = 0
                        self._pending_response = False
                        if status == 'ok':
                            elapsed = time.time() - start_time
                            logger.info(f"[VLMProcess] Slow query response arrived after {elapsed:.1f}s")
                            return result
                except:
                    pass

                elapsed = time.time() - start_time
                logger.warning(
                    f"[VLMProcess] HARD KILL after {elapsed:.1f}s ({self._consecutive_timeouts} timeouts) - restarting worker"
                )
                self.restart()
                self._pending_response = False
                return "KILLED", elapsed
            else:
                logger.debug(
                    f"[VLMProcess] Soft timeout after {elapsed:.1f}s (#{self._consecutive_timeouts}) - skipping query"
                )
                return "TIMEOUT", elapsed

    def release(self):
        """Release the VLM worker process."""
        self.kill()

    def load_model(self):
        """Compatibility method - starts the worker process."""
        return self.start()

    @property
    def restart_count(self):
        return self._restart_count
