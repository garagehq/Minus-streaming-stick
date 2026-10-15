"""
Anonymous ad-snapshot contribution for Minus (OPT-IN, default OFF).

The device-side twin of the Chrome extension's "Contribute anonymous ad
snapshots" setting, sending to the same ingest server and the same private
training dataset. Nothing here runs, queues or sends unless the user has
explicitly turned it on in the web UI (Settings -> Help improve Minus).

What a sample is
----------------
A downscaled PNG of a TV frame that Minus BLOCKED as an ad, plus:
  p_ad     the VLM's ad probability for that frame (null if only OCR fired)
  verdict  always "ad" (only blocked frames are offered)
  host     always "tv" (there is no URL on an HDMI input)
  w, h     the image size as sent
  engine   "minus-tv/<detector>" e.g. "minus-tv/vlm", "minus-tv/ocr"
  key      a fresh random id per sample (not linkable to other samples)

Never sent: the device's name, IP, serial, Wi-Fi, account, streaming device,
app/channel, time of viewing, OCR text, audio, or any frame that was not
blocked as an ad. No cookies, no user or device identifier of any kind.

The pause-retracts rule
-----------------------
Samples wait in a local queue for a cool-down (10 minutes) before upload. If
the user pauses blocking during that window while that block is on screen, or
within RETRACT_AFTER_END_S of it ending, every sample from that block is deleted
and never leaves the box. A pause means "that wasn't an ad I wanted
blocked", and we don't collect what the user didn't want blocked. (The
extension does the same thing when the user clicks X on an overlay.)

Resilience
----------
Contribution is best-effort and entirely severable: nothing here may raise
into the detection loops, the queue is capped (count and age), and a dead or
cancelled ingest server only causes exponential backoff (20 s -> 6 h).
Turning the setting off stops uploads immediately and deletes the local
queue.
"""

import base64
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Same endpoint and gate key as the Chrome extension (extension/background.js).
# The key only keeps random bots off the endpoint -- it ships with every
# install by design and is not a secret. The Hugging Face write token lives
# only on the ingest server.
DEFAULT_INGEST_URL = "https://minus-ingest-garage.fly.dev/ingest"
DEFAULT_INGEST_KEY = "bdc2d283edff8b961ddf5f235bd1ebeab6c4f98ca9a5e48a"

DEFAULT_QUEUE_DIR = Path.home() / ".minus_contrib_queue"

COOLDOWN_S = 10 * 60            # local wait before a sample may upload
RETRACT_AFTER_END_S = 60        # a pause this soon after a block ends still retracts it
MAX_QUEUE = 200                 # hard cap on queued samples (oldest dropped)
SAMPLE_TTL_S = 7 * 24 * 3600    # drop samples older than this
MAX_PER_BLOCK = 3               # at most this many frames from one ad block
MIN_SPACING_S = 8.0             # and at least this far apart
MAX_SIDE = 960                  # longest image side as sent (frames arrive 1080p-4K)
BATCH = 3                       # samples per POST (~1 MB each)
UPLOAD_TIMEOUT_S = 25           # the server scales to zero; allow a cold start
UPLOAD_INTERVAL_S = 60          # how often the uploader looks for due samples
BACKOFF_BASE_S = 20
MAX_BACKOFF_S = 6 * 3600
DHASH_DUP_BITS = 10             # near-identical frames within a block are skipped


def _dhash(frame) -> int:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


class SnapshotContributor:
    """Opt-in queue + uploader. Construct once; call `offer()` from the
    detection loops while a block is on screen, `retract_block()` when the
    user pauses, and `set_enabled()` from the web UI."""

    def __init__(self, enabled: bool = False, queue_dir: Path = None,
                 ingest_url: str = None, ingest_key: str = None,
                 cooldown_s: float = None, start_thread: bool = True):
        self.queue_dir = Path(queue_dir or os.environ.get("MINUS_CONTRIB_QUEUE_DIR", DEFAULT_QUEUE_DIR))
        self.ingest_url = ingest_url if ingest_url is not None else os.environ.get(
            "MINUS_INGEST_URL", DEFAULT_INGEST_URL)
        self.ingest_key = ingest_key if ingest_key is not None else os.environ.get(
            "MINUS_INGEST_KEY", DEFAULT_INGEST_KEY)
        self.cooldown_s = float(cooldown_s if cooldown_s is not None else os.environ.get(
            "MINUS_CONTRIB_COOLDOWN_S", COOLDOWN_S))
        self._enabled = bool(enabled)
        self._lock = threading.Lock()
        self._per_block = {}            # block_id -> {"n", "last", "hashes"}
        self._failures = 0
        self._next_upload_at = 0.0
        self.uploaded = 0               # this session
        self.retracted = 0              # this session
        self.last_upload_at = None
        self.last_error = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        if self._enabled:
            self._ensure_dir()
        if start_thread:
            self._thread = threading.Thread(target=self._run, name="contribute-uploader", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------- settings
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        """Turn contribution on or off. Off deletes the local queue so
        nothing that was waiting can ever be sent."""
        enabled = bool(enabled)
        with self._lock:
            self._enabled = enabled
            self._per_block.clear()
            self._failures = 0
            self._next_upload_at = 0.0
        if enabled:
            self._ensure_dir()
            logger.info("[Contribute] Anonymous ad-snapshot contribution ON (opt-in)")
        else:
            n = self._clear_queue()
            logger.info(f"[Contribute] Anonymous ad-snapshot contribution OFF "
                        f"({n} queued sample(s) deleted, nothing sent)")

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "queued": len(self._queued()) if self._enabled else 0,
            "uploaded_this_session": self.uploaded,
            "retracted_this_session": self.retracted,
            "last_upload_at": self.last_upload_at,
            "last_error": self.last_error,
            "cooldown_minutes": round(self.cooldown_s / 60, 1),
            "endpoint": self.ingest_url,
        }

    # ------------------------------------------------------------ the queue
    def offer(self, frame, block_id, source: str, p_ad=None) -> bool:
        """Queue a frame from an active ad block. Returns True if queued.
        Never raises -- this is called from the detection loops."""
        if not self._enabled or frame is None or block_id is None:
            return False
        try:
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return False
            now = time.time()
            h = _dhash(frame)
            with self._lock:
                st = self._per_block.setdefault(block_id, {"n": 0, "last": 0.0, "hashes": []})
                if st["n"] >= MAX_PER_BLOCK or now - st["last"] < MIN_SPACING_S:
                    return False
                if any(bin(h ^ o).count("1") < DHASH_DUP_BITS for o in st["hashes"]):
                    return False
                st["n"] += 1
                st["last"] = now
                st["hashes"].append(h)
                # forget old blocks so the dict never grows
                if len(self._per_block) > 64:
                    for k in sorted(self._per_block, key=lambda k: self._per_block[k]["last"])[:32]:
                        self._per_block.pop(k, None)

            img = frame
            ih, iw = img.shape[:2]
            scale = min(1.0, MAX_SIDE / float(max(ih, iw)))
            if scale < 1.0:
                img = cv2.resize(img, (max(1, int(iw * scale)), max(1, int(ih * scale))),
                                 interpolation=cv2.INTER_AREA)
            ok, png = cv2.imencode(".png", img)
            if not ok:
                return False
            key = uuid.uuid4().hex
            meta = {
                "key": key,
                "p_ad": None if p_ad is None else round(float(p_ad), 4),
                "verdict": "ad",
                "host": "tv",
                "w": int(img.shape[1]),
                "h": int(img.shape[0]),
                "engine": f"minus-tv/{source or 'unknown'}",
                # local-only fields (stripped before upload)
                "_queued_at": now,
                "_block": str(block_id),
            }
            self._ensure_dir()
            tmp = self.queue_dir / f".{key}.tmp"
            tmp.write_bytes(png.tobytes())
            os.replace(tmp, self.queue_dir / f"{key}.png")
            (self.queue_dir / f"{key}.json").write_text(json.dumps(meta))
            if not self._enabled:      # opted out while we were writing
                self._delete(key)
                return False
            self._enforce_caps()
            logger.info(f"[Contribute] Queued ad snapshot ({meta['w']}x{meta['h']}, {meta['engine']}, "
                        f"p_ad={meta['p_ad']}); uploads after {self.cooldown_s / 60:.0f} min unless retracted")
            return True
        except Exception as e:  # never surface into detection
            logger.debug(f"[Contribute] offer failed: {e}")
            return False

    def retract_block(self, block_id) -> int:
        """Delete every not-yet-uploaded sample from this block. Called when
        the user pauses blocking: a pause means the block was unwanted."""
        if block_id is None:
            return 0
        n = 0
        try:
            for meta in self._queued():
                if meta.get("_block") == str(block_id):
                    self._delete(meta["key"])
                    n += 1
            with self._lock:
                st = self._per_block.get(block_id)
                if st is not None:
                    st["n"] = MAX_PER_BLOCK   # and take nothing more from it
            if n:
                self.retracted += n
                logger.info(f"[Contribute] User paused: {n} queued snapshot(s) from that block "
                            f"deleted, never uploaded")
        except Exception as e:
            logger.debug(f"[Contribute] retract failed: {e}")
        return n

    def _ensure_dir(self):
        try:
            self.queue_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.queue_dir, 0o700)
        except Exception:
            pass

    def _queued(self) -> list:
        out = []
        try:
            for p in self.queue_dir.glob("*.json"):
                try:
                    meta = json.loads(p.read_text())
                    if (self.queue_dir / f"{meta['key']}.png").exists():
                        out.append(meta)
                    else:
                        p.unlink(missing_ok=True)
                except Exception:
                    p.unlink(missing_ok=True)
        except Exception:
            pass
        out.sort(key=lambda m: m.get("_queued_at", 0))
        return out

    def _delete(self, key: str):
        for ext in (".png", ".json"):
            try:
                (self.queue_dir / f"{key}{ext}").unlink(missing_ok=True)
            except Exception:
                pass

    def _clear_queue(self) -> int:
        n = len(self._queued())
        try:
            if self.queue_dir.exists():
                shutil.rmtree(self.queue_dir, ignore_errors=True)
        except Exception:
            pass
        return n

    def _enforce_caps(self):
        now = time.time()
        q = self._queued()
        keep = []
        for m in q:
            if now - m.get("_queued_at", 0) >= SAMPLE_TTL_S:
                self._delete(m["key"])
            else:
                keep.append(m)
        for m in keep[:max(0, len(keep) - MAX_QUEUE)]:
            self._delete(m["key"])

    # -------------------------------------------------------------- upload
    def upload_due(self) -> int:
        """Upload one batch of samples whose cool-down has passed. Returns
        the number uploaded. Never raises."""
        if not self._enabled or not self.ingest_url:
            return 0
        if time.time() < self._next_upload_at:
            return 0
        try:
            self._enforce_caps()
            now = time.time()
            due = [m for m in self._queued() if now - m.get("_queued_at", 0) >= self.cooldown_s][:BATCH]
            if not due:
                return 0
            samples = []
            for m in due:
                png = (self.queue_dir / f"{m['key']}.png").read_bytes()
                s = {k: v for k, v in m.items() if not k.startswith("_")}
                s["img"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                samples.append(s)
            body = json.dumps({"v": 1, "samples": samples}).encode("utf-8")
            headers = {"Content-Type": "application/json", "User-Agent": "Minus-TV"}
            if self.ingest_key:
                headers["x-minus-key"] = self.ingest_key
            req = urllib.request.Request(self.ingest_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=UPLOAD_TIMEOUT_S) as resp:
                if not (200 <= resp.status < 300):
                    raise RuntimeError(f"HTTP {resp.status}")
            # The user may have opted out while the POST was in flight; the
            # files are gone in that case and there is nothing to delete.
            for m in due:
                self._delete(m["key"])
            self.uploaded += len(due)
            self.last_upload_at = time.time()
            self.last_error = None
            self._failures = 0
            self._next_upload_at = 0.0
            logger.info(f"[Contribute] Uploaded {len(due)} anonymous ad snapshot(s)")
            return len(due)
        except Exception as e:
            self._failures += 1
            backoff = min(MAX_BACKOFF_S, BACKOFF_BASE_S * 2 ** min(self._failures - 1, 10))
            self._next_upload_at = time.time() + backoff
            self.last_error = str(e)[:200]
            logger.debug(f"[Contribute] upload failed ({e}); retry in {backoff:.0f}s")
            return 0

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(UPLOAD_INTERVAL_S)
            self._wake.clear()
            if self._stop.is_set():
                break
            if self._enabled:
                # drain everything that is due, a batch at a time
                while self._enabled and self.upload_due() == BATCH:
                    pass

    def stop(self):
        self._stop.set()
        self._wake.set()
