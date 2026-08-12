"""
OCR Worker Process - runs OCR in separate process for hard timeout capability.

This allows us to actually KILL stuck OCR inference instead of just timing out.
"""

import os
import re
import sys
import time
import multiprocessing as mp
from multiprocessing import Process, Queue, Event

# Use 'spawn' start method to avoid inherited file descriptors and process state issues
# This prevents "can only join a child process" errors from RKNN runtime
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set


def _ocr_worker_main(request_queue, response_queue, ready_event, shutdown_event):
    """
    Main function for OCR worker process.

    Loads models once, then processes requests until shutdown.
    """
    # Reset inherited signal handlers so SIGTERM from parent just exits cleanly
    # instead of running minus.stop() (inherited via fork), which deadlocks
    # and corrupts RKNN runtime state across subsequent worker respawns.
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('OCRWorker')

    try:
        # Import OCR
        sys.path.insert(0, '/home/radxa/Minus/src')
        from ocr import PaddleOCR
        from config import OCR_MODEL_DIR
        from pathlib import Path

        # Find model paths dynamically
        base_path = Path(OCR_MODEL_DIR)
        det_models = list(base_path.glob('ppocrv3_det_*.rknn'))
        rec_models = list(base_path.glob('ppocrv3_rec_*.rknn'))
        dict_file = base_path / 'ppocr_keys_v1.txt'

        if not det_models or not rec_models or not dict_file.exists():
            logger.error(f"[OCRWorker] Models not found in {OCR_MODEL_DIR}")
            return

        det_model = str(det_models[0])
        rec_model = str(rec_models[0])
        dict_path = str(dict_file)

        # Load models
        logger.info("[OCRWorker] Loading models...")
        ocr = PaddleOCR(det_model, rec_model, dict_path)
        if not ocr.load_models():
            logger.error("[OCRWorker] Failed to load models")
            return

        # Extended warmup to ensure RKNN NPU is fully ready for real frames
        # 4 inferences with varying content to warm all code paths
        load_start = time.time()
        try:
            import numpy as np
            for i in range(4):
                # Create varied warmup images (noise, gradients, edges, text-like)
                if i == 0:
                    # Pure noise
                    warmup_img = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
                elif i == 1:
                    # Gradient (simulates video content)
                    arr = np.zeros((540, 960, 3), dtype=np.uint8)
                    arr[:, :, 0] = np.linspace(0, 255, 960).reshape(1, -1).astype(np.uint8)
                    arr[:, :, 1] = np.linspace(255, 0, 540).reshape(-1, 1).astype(np.uint8)
                    warmup_img = arr
                elif i == 2:
                    # High contrast edges (simulates text/UI elements)
                    arr = np.zeros((540, 960, 3), dtype=np.uint8)
                    arr[::2, :, :] = 255
                    warmup_img = arr
                else:
                    # Mixed content
                    warmup_img = np.random.randint(50, 200, (540, 960, 3), dtype=np.uint8)

                start_w = time.time()
                _ = ocr.ocr(warmup_img)
                logger.debug(f"[OCRWorker] Warmup {i+1}/4: {time.time() - start_w:.2f}s")

            total_time = time.time() - load_start
            logger.info(f"[OCRWorker] Warmup complete (4 inferences in {total_time:.1f}s)")
        except Exception as e:
            logger.warning(f"[OCRWorker] Warmup failed (non-fatal): {e}")

        logger.info("[OCRWorker] Models loaded, ready for requests")
        ready_event.set()

        # Process requests with keepalive
        last_inference_time = time.time()
        KEEPALIVE_INTERVAL = 20.0  # Run keepalive if idle for 20s

        while not shutdown_event.is_set():
            try:
                # Wait for request with timeout so we can check shutdown and keepalive
                try:
                    request = request_queue.get(timeout=1.0)
                except:
                    # No request - check if we need keepalive
                    if time.time() - last_inference_time > KEEPALIVE_INTERVAL:
                        try:
                            import numpy as np
                            warmup_img = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
                            _ = ocr.ocr(warmup_img)
                            last_inference_time = time.time()
                            logger.debug("[OCRWorker] Keepalive inference completed")
                        except:
                            pass
                    continue

                if request is None:  # Shutdown signal
                    break

                frame_data, request_type = request

                # Convert frame back to numpy array if needed (Queue may serialize as list)
                if isinstance(frame_data, list):
                    frame_rgb = np.array(frame_data, dtype=np.uint8)
                else:
                    frame_rgb = frame_data

                if request_type == 'ocr':
                    result = ocr.ocr(frame_rgb)
                    response_queue.put(('ok', result))
                    last_inference_time = time.time()
                elif request_type == 'check_ad':
                    # OCR + keyword check (check_ad_keywords returns a 4-tuple)
                    ocr_results = ocr.ocr(frame_rgb)
                    is_ad, keywords, _texts, _terminal = ocr.check_ad_keywords(ocr_results)
                    response_queue.put(('ok', (is_ad, keywords, ocr_results)))
                    last_inference_time = time.time()
                else:
                    response_queue.put(('error', 'Unknown request type'))

            except Exception as e:
                logger.error(f"[OCRWorker] Error processing request: {e}")
                response_queue.put(('error', str(e)))

        logger.info("[OCRWorker] Shutting down")
        ocr.release()

    except Exception as e:
        import traceback
        print(f"[OCRWorker] Fatal error: {e}")
        traceback.print_exc()


class OCRProcess:
    """
    Manages OCR in a separate process with hard timeout capability.

    If inference takes longer than timeout, the process is KILLED and restarted.
    """

    HARD_TIMEOUT = 1.0  # Kill OCR if it takes longer than this

    def __init__(self):
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.ready_event = None
        self.shutdown_event = None
        self.is_ready = False
        self._restart_count = 0
        self._consecutive_timeouts = 0

    def start(self):
        """Start the OCR worker process."""
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
            target=_ocr_worker_main,
            args=(self.request_queue, self.response_queue, self.ready_event, self.shutdown_event),
            daemon=True
        )
        self.process.start()

        # Wait for models to load (up to 30s)
        if self.ready_event.wait(timeout=30.0):
            self.is_ready = True
            return True
        else:
            self.kill()
            return False

    def kill(self):
        """Kill the OCR worker process."""
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
        """Kill and restart the OCR worker.

        Uses exponential backoff if restarting frequently (prevents restart loops).
        """
        import logging
        logger = logging.getLogger('Minus.OCR')

        self._restart_count += 1
        self._consecutive_timeouts += 1

        # Calculate backoff based on consecutive timeouts
        # 1st timeout: 1s, 2nd: 2s, 3rd+: 4s max
        if self._consecutive_timeouts >= 3:
            backoff = 4.0
            logger.warning(f"[OCRProcess] Multiple timeouts ({self._consecutive_timeouts}), using {backoff}s backoff")
        elif self._consecutive_timeouts >= 2:
            backoff = 2.0
        else:
            backoff = 1.0

        self.kill()

        # Clear any stale responses
        if self.response_queue is not None:
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except:
                    break

        # Wait for NPU resources to be released
        time.sleep(backoff)

        return self.start()

    def ocr(self, frame_rgb):
        """
        Run OCR with hard timeout.

        Returns: OCR results list or empty list on timeout
        """
        if not self.is_ready or self.process is None or not self.process.is_alive():
            if not self.start():
                return []

        start_time = time.time()

        # Send request
        self.request_queue.put((frame_rgb, 'ocr'))

        # Wait for response with hard timeout
        try:
            status, result = self.response_queue.get(timeout=self.HARD_TIMEOUT)

            if status == 'ok':
                # Reset consecutive timeout counter on success
                self._consecutive_timeouts = 0
                return result
            else:
                return []

        except:
            # TIMEOUT - kill the process
            elapsed = time.time() - start_time
            import logging
            logging.getLogger('Minus.OCR').warning(
                f"[OCRProcess] HARD KILL after {elapsed:.1f}s (timeout #{self._consecutive_timeouts + 1}) - restarting worker"
            )
            self.restart()
            return []

    def release(self):
        """Release the OCR worker process."""
        self.kill()

    def load_models(self):
        """Compatibility method - starts the worker process."""
        return self.start()

    @property
    def restart_count(self):
        return self._restart_count

    def check_ad_keywords(self, ocr_results):
        """
        Check OCR results for ad-related keywords.
        This runs locally (not in worker) since it's just string matching.

        Delegates to PaddleOCR.check_ad_keywords (src/ocr.py) — the single
        source of truth. This class used to carry its own copy of the
        keyword lists and patterns, and that copy silently drifted stale
        TWICE (missing "Ad1:09" timestamp handling in May 2026; missing
        the 'skip'/'sponsor'/'promoted'/Spanish keywords in Aug 2026 —
        real ads on screen matched nothing while the boot log printed the
        rich list from ocr.py). Do not reintroduce a local copy. See
        CLAUDE.md, "OCR Worker Keyword-Pattern Drift".

        Returns:
            Tuple of (found_ad, matched_keywords, all_texts, is_terminal)
        """
        from ocr import PaddleOCR
        return PaddleOCR.check_ad_keywords(ocr_results)
