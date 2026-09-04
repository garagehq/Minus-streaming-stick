"""Headless block-latency test harness.

Plays Big Buck Bunny in a Python loop, lets the test orchestrator inject
"AD"-style overlay text on/off at controlled timestamps, and measures how
long the production OCR + VLM workers take to flip a blocking-state
machine that mirrors minus's logic.

No HDMI, no ustreamer, no DRM, no audio. The full HDMI pipeline is bypassed
intentionally — we only want to test the detection + decision flow.

The DecisionEngine here is a faithful subset of minus.py's blocking logic.
Knobs at the top of this file are the ones that exist (and that we tune)
in production.

Run as:   python3 tests/block_latency_harness.py [scenarios...]

Scenarios:
  detect    measure overlay-on -> blocking=on latency
  recover   measure overlay-off -> blocking=off latency
  burst     5 randomised on/off cycles with 5-20s ad windows
  pause     simulate pause-while-ad: freeze the source while overlay is on,
            then resume; measure how long the (now stale) ad reading lingers
"""
from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
import threading
import time

# Add minus's src/ to path so we can import the real production workers.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import cv2  # noqa: E402

from ocr_worker import OCRProcess  # noqa: E402  - real production OCR
from vlm_worker import VLMProcess  # noqa: E402  - real production VLM


# ---------------------------------------------------------------------------
# Tunable parameters — defaults mirror production (locked-in via testing in
# this harness). Change here AND in production for any new tuning to map 1:1.
# ---------------------------------------------------------------------------
PARAMS = {
    # OCR
    'OCR_INTERVAL_S': 0.5,         # how often the OCR worker is invoked
    'OCR_STOP_THRESHOLD': 3,       # 4 -> 2 -> 3; see minus.py for the flap data
    # VLM — retuned for FastVLM-0.5B iter4 (validated in
    # tests/harness_iter4_retune_ab.py: VLM-only detect 6.11s->2.11s,
    # 0 false-pos, 0 phantom re-blocks). iter4 inference is ~0.33s vs the
    # 1.5B's ~1s, so the production VLM loop's real cadence is ~1s, not ~2s.
    'VLM_INTERVAL_S': 1.0,         # was 2.0 — models iter4 real loop cadence
    'vlm_history_window': 8.0,     # was 45.0 — kills stale-content-vote dilution
    'vlm_min_decisions': 3,        # LFM2 retune 5→3 — LFM2's ~4× lower per-frame FP rate vs iter4 makes the iter4-era hardening unnecessary
    'vlm_start_agreement': 0.70,   # LFM2 retune 0.80→0.70 (+0.10 hyst = 0.80 eff). See minus.py comment block for math.
    'vlm_stop_agreement': 0.75,
    'vlm_hysteresis_boost': 0.10,
    'vlm_start_threshold_cap': 0.95,
    'VLM_STOP_THRESHOLD': 2,       # consecutive no-ad VLM votes (fast-stop path)
    # Static / cooldown / blocking
    'STATIC_TIME_THRESHOLD': 2.5,
    'STATIC_OCR_THRESHOLD': 4,
    'DYNAMIC_COOLDOWN': 1.5,       # tuned: was 0.5. cooldown window after dynamic
    'MIN_BLOCKING_DURATION': 3.0,
    # Scene-change detector (mean-abs-diff over 64x36 grey resize)
    'scene_change_threshold': 0.001,  # tuned: was 0.01. only true-static frames register
    # When True, static-suppression is a no-op. Used to measure pure OCR
    # detection/recovery latency without static-suppression interference.
    'disable_static_suppression': False,
}

VIDEO_PATH = '/home/radxa/test_assets/bbb.mp4'


# ---------------------------------------------------------------------------
# Decision engine — replicates the relevant parts of minus.py
# ---------------------------------------------------------------------------
class DecisionEngine:
    """In-memory state machine that mirrors minus.py's blocking decision."""

    def __init__(self, params):
        self.p = params
        # OCR
        self.ocr_ad_detected = False
        self.ocr_ad_detection_count = 0
        self.ocr_no_ad_count = 0
        # VLM
        self.vlm_ad_detected = False
        self.vlm_decision_history = []  # (time, is_ad, confidence)
        self.vlm_no_ad_count = 0  # consecutive no-ad VLM verdicts (fast-stop path)
        # Combined / blocking
        self.ad_detected = False
        self.blocking_source = None
        self.blocking_start_time = 0.0
        # Static screen
        self.static_since_time = 0.0
        self.static_ocr_count = 0
        self.static_blocking_suppressed = False
        self.screen_became_dynamic_time = 0.0
        # Scene-change reference
        self._prev_grey_small = None

    # ----- scene change -----
    def scene_changed(self, frame_bgr):
        small = cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_grey_small is None:
            self._prev_grey_small = grey
            return True, 0.0
        diff = cv2.absdiff(grey, self._prev_grey_small).mean() / 255.0
        self._prev_grey_small = grey
        return diff >= self.p['scene_change_threshold'], float(diff)

    # ----- static screen tracking -----
    def update_static(self, scene_did_change, now):
        if self.p.get('disable_static_suppression'):
            # No-op when disabled (used to baseline pure OCR latency).
            return False
        cooldown_completed = False
        if scene_did_change:
            if self.static_blocking_suppressed and self.screen_became_dynamic_time == 0:
                self.screen_became_dynamic_time = now
            self.static_since_time = 0.0
            self.static_ocr_count = 0
        else:
            self.static_ocr_count += 1
            if self.static_since_time == 0:
                self.static_since_time = now

        static_time = (now - self.static_since_time) if self.static_since_time > 0 else 0
        if (static_time >= self.p['STATIC_TIME_THRESHOLD']
                or self.static_ocr_count >= self.p['STATIC_OCR_THRESHOLD']):
            self.static_blocking_suppressed = True
            self.screen_became_dynamic_time = 0
        elif self.screen_became_dynamic_time > 0:
            cooldown_elapsed = now - self.screen_became_dynamic_time
            if cooldown_elapsed >= self.p['DYNAMIC_COOLDOWN']:
                self.static_blocking_suppressed = False
                self.screen_became_dynamic_time = 0
                # Clear stale detection state from static + cooldown
                if (self.ocr_ad_detected or self.vlm_ad_detected
                        or self.ocr_ad_detection_count > 0):
                    self.ocr_ad_detected = False
                    self.ocr_no_ad_count = 0
                    self.ocr_ad_detection_count = 0
                    self.vlm_ad_detected = False
                    self.vlm_decision_history.clear()
                    cooldown_completed = True
        return cooldown_completed

    # ----- OCR -----
    def on_ocr(self, found_ad):
        if found_ad:
            self.ocr_ad_detection_count += 1
            self.ocr_no_ad_count = 0
            if self.ocr_ad_detection_count >= 1 and not self.ocr_ad_detected:
                self.ocr_ad_detected = True
        else:
            self.ocr_no_ad_count += 1
            self.ocr_ad_detection_count = 0
            if self.ocr_ad_detected and self.ocr_no_ad_count >= self.p['OCR_STOP_THRESHOLD']:
                self.ocr_ad_detected = False

    # ----- VLM -----
    def on_vlm(self, is_ad, confidence):
        # Update consecutive-count counter (used for fast-stop path).
        # Mirrors production minus.py:2806/2813.
        if is_ad:
            self.vlm_no_ad_count = 0
        else:
            self.vlm_no_ad_count += 1

        # Update sliding-window state (used for VLM-internal start/stop).
        now = time.time()
        self.vlm_decision_history.append((now, is_ad, confidence))
        cutoff = now - self.p['vlm_history_window']
        self.vlm_decision_history = [e for e in self.vlm_decision_history if e[0] >= cutoff]

        ad_w = sum(c for _, ad, c in self.vlm_decision_history if ad)
        no_ad_w = sum(c for _, ad, c in self.vlm_decision_history if not ad)
        total_w = ad_w + no_ad_w
        if total_w == 0:
            return
        ad_ratio = ad_w / total_w
        no_ad_ratio = no_ad_w / total_w
        n = len(self.vlm_decision_history)
        if n < self.p['vlm_min_decisions']:
            return

        if not self.vlm_ad_detected:
            threshold = self.p['vlm_start_agreement'] + self.p['vlm_hysteresis_boost']
            threshold = min(threshold, self.p['vlm_start_threshold_cap'])
            if ad_ratio >= threshold:
                self.vlm_ad_detected = True
        else:
            threshold = self.p['vlm_stop_agreement'] + self.p['vlm_hysteresis_boost']
            if no_ad_ratio >= threshold:
                self.vlm_ad_detected = False

    # ----- combined / blocking -----
    def compute_blocking(self, now):
        # START
        ad_now_to_start = self.ocr_ad_detected or self.vlm_ad_detected
        if ad_now_to_start and not self.ad_detected:
            self.ad_detected = True
            self.blocking_start_time = now
            if self.ocr_ad_detected and self.vlm_ad_detected:
                self.blocking_source = 'both'
            elif self.ocr_ad_detected:
                self.blocking_source = 'ocr'
            else:
                self.blocking_source = 'vlm'
        # STOP — mirror production's two-path stop logic in minus.py:2270.
        # OCR-triggered (or both): OCR is authoritative, uses ocr_no_ad_count.
        # VLM-triggered alone: VLM consecutive-count is authoritative
        # (sliding-window self.vlm_ad_detected may lag; consecutive count
        # gives the fast recovery the user actually feels).
        elif self.ad_detected:
            duration = now - self.blocking_start_time
            if duration >= self.p['MIN_BLOCKING_DURATION']:
                if self.blocking_source == 'vlm':
                    should_stop = self.vlm_no_ad_count >= self.p['VLM_STOP_THRESHOLD']
                else:
                    should_stop = self.ocr_no_ad_count >= self.p['OCR_STOP_THRESHOLD']
                if should_stop:
                    self.ad_detected = False
                    self.blocking_source = None
                    # Production also clears the VLM internal state on stop.
                    self.vlm_ad_detected = False
                    self.vlm_decision_history.clear()
                    self.vlm_no_ad_count = 0
        is_blocking = self.ad_detected and not self.static_blocking_suppressed
        return is_blocking, self.blocking_source


# ---------------------------------------------------------------------------
# Frame source + overlay
# ---------------------------------------------------------------------------
def render_ad_overlay(frame, text):
    """High-contrast overlay top-right that OCR will trivially read."""
    if not text:
        return frame
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thick = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = w - tw - 30
    y = th + 30
    cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick)
    return frame


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class Harness:
    def __init__(self, params, video_path=VIDEO_PATH, use_real_vlm=True):
        self.p = params
        self.video_path = video_path
        self.engine = DecisionEngine(params)
        self.ocr = OCRProcess()
        self.use_real_vlm = use_real_vlm
        self.vlm = VLMProcess() if use_real_vlm else None
        # control
        self.overlay_text = None
        self.overlay_lock = threading.Lock()
        self._freeze_source = False  # for "pause" tests
        self.stop_event = threading.Event()
        # VLM injection: when set, the harness will use this verdict in place
        # of (or in addition to) the real VLM model. This lets us drive the
        # decision engine's VLM state machine deterministically without
        # depending on what the real model thinks of synthetic frames.
        self._vlm_inject = None  # None = real VLM; ('ad', conf) or ('no-ad', conf) = injected
        self._vlm_inject_lock = threading.Lock()
        # event log
        self.events = []  # list of dicts
        self.event_lock = threading.Lock()
        self.last_blocking = False
        # vlm async
        self._vlm_inflight = False
        self._vlm_lock = threading.Lock()
        # bookkeeping
        self.last_ocr_at = 0.0
        self.last_vlm_at = 0.0
        self.frames_processed = 0

    # public API
    def start(self):
        print("[harness] starting OCR worker...")
        if not self.ocr.start():
            raise RuntimeError("OCR worker failed to start")
        if self.use_real_vlm:
            print("[harness] starting VLM worker (~30s for first load)...")
            if not self.vlm.start():
                raise RuntimeError("VLM worker failed to start")
        else:
            print("[harness] VLM disabled — using injected verdicts only")
        print("[harness] workers ready")

    def reset_state(self):
        """Wipe the engine state so the next scenario starts clean.

        Without this, residual VLM history / static-suppression bookkeeping
        from one scenario leaks into the next and corrupts the measurements.
        """
        self.engine = DecisionEngine(self.p)
        self.last_blocking = False
        self.last_ocr_at = 0.0
        self.last_vlm_at = 0.0
        with self.overlay_lock:
            self.overlay_text = None
        with self._vlm_inject_lock:
            self._vlm_inject = None
        self._freeze_source = False
        self._log('reset_state', None)

    def stop(self):
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.ocr.kill()
        if self.vlm:
            with contextlib.suppress(Exception):
                self.vlm.kill()

    def inject_vlm(self, verdict, confidence=0.75):
        """Force the next VLM cycle (and any after) to use this verdict.

        verdict: 'ad' / 'no-ad' / None (None reverts to real VLM model).
        This bypasses the actual VLM model so we can drive the engine's
        sliding-window state machine deterministically. The injection
        applies on every VLM_INTERVAL_S tick until cleared.
        """
        with self._vlm_inject_lock:
            if verdict is None:
                self._vlm_inject = None
            else:
                assert verdict in ('ad', 'no-ad'), verdict
                self._vlm_inject = (verdict == 'ad', confidence)
        self._log('vlm_inject', {'verdict': verdict, 'confidence': confidence})

    def set_overlay(self, text):
        with self.overlay_lock:
            self.overlay_text = text
        self._log('overlay_set', {'text': text})

    def freeze_source(self, on):
        """Simulate user pause: hold the same frame until unfrozen."""
        self._freeze_source = on
        self._log('freeze', {'on': on})

    def get_blocking(self):
        return self.last_blocking

    def _log(self, kind, payload):
        with self.event_lock:
            self.events.append({'t': time.time(), 'kind': kind, 'data': payload})

    # the loop
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        period = 1.0 / fps
        next_t = time.time()
        frozen_frame = None

        while not self.stop_event.is_set():
            if self._freeze_source and frozen_frame is not None:
                frame = frozen_frame.copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self._freeze_source:
                    frozen_frame = frame.copy()
            self.frames_processed += 1

            with self.overlay_lock:
                ot = self.overlay_text
            if ot:
                frame = render_ad_overlay(frame, ot)

            now = time.time()

            # OCR (rate-limited). Scene-change + static tracking are gated to
            # this same cadence — that's what minus.py does (the static state
            # machine lives inside the OCR worker loop, not per-frame), and
            # tracking it per-frame at 24fps fires STATIC_OCR_THRESHOLD=4 in
            # 0.16s on BBB's slow scenes instead of the intended ~2s.
            if (now - self.last_ocr_at) >= self.p['OCR_INTERVAL_S']:
                self.last_ocr_at = now
                sc, diff_value = self.engine.scene_changed(frame)
                cooldown_done = self.engine.update_static(sc, now)
                if cooldown_done:
                    self._log('cooldown_complete', None)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.ocr.ocr(rgb)
                if results:
                    is_ad, matched, _, _ = self.ocr.check_ad_keywords(results)
                else:
                    is_ad, matched = False, []
                self.engine.on_ocr(is_ad)
                self._log('ocr', {
                    'is_ad': is_ad,
                    'kw': [k for k, _ in matched] if matched else [],
                    'scene_changed': sc,
                    'diff': round(diff_value, 5),
                    'static_suppressed': self.engine.static_blocking_suppressed,
                })

            # VLM (rate-limited, async — call_lock prevents overlap).
            # If a verdict has been injected, the engine sees the injection
            # without going through the real model; the real model is only
            # called for logging/observability when use_real_vlm=True.
            if (now - self.last_vlm_at) >= self.p['VLM_INTERVAL_S']:
                self.last_vlm_at = now
                with self._vlm_inject_lock:
                    inject = self._vlm_inject
                if inject is not None:
                    is_ad_inj, conf_inj = inject
                    self.engine.on_vlm(is_ad_inj, conf_inj)
                    self._log('vlm_injected', {
                        'is_ad': is_ad_inj, 'confidence': conf_inj,
                    })
                elif self.use_real_vlm:
                    with self._vlm_lock:
                        if not self._vlm_inflight:
                            self._vlm_inflight = True
                            path = '/dev/shm/test_vlm_frame.jpg'
                            cv2.imwrite(path, frame)
                            threading.Thread(
                                target=self._do_vlm,
                                args=(path,),
                                daemon=True,
                            ).start()

            # combined blocking state
            is_blocking, source = self.engine.compute_blocking(now)
            if is_blocking != self.last_blocking:
                self._log('blocking_changed', {'on': is_blocking, 'source': source})
                self.last_blocking = is_blocking

            # real-time pacing
            next_t += period
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.time()
        cap.release()

    def _do_vlm(self, path):
        try:
            is_ad, response, elapsed, confidence = self.vlm.detect_ad(path)
            if response and response.upper() in ('TIMEOUT', 'PENDING', 'KILLED'):
                self._log('vlm_skip', {'response': response})
            else:
                self.engine.on_vlm(is_ad, confidence)
                self._log('vlm', {
                    'is_ad': is_ad,
                    'response': (response or '')[:30],
                    'elapsed': round(elapsed, 2),
                    'confidence': confidence,
                })
        except Exception as e:
            self._log('vlm_error', repr(e))
        finally:
            with self._vlm_lock:
                self._vlm_inflight = False


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------
def _wait_until(predicate, timeout):
    """Wait for predicate() to return True, with 50ms polling. Returns elapsed or None."""
    start = time.time()
    while time.time() - start < timeout:
        if predicate():
            return time.time() - start
        time.sleep(0.05)
    return None


def scenario_detect_recover(harness, text, hold_seconds, settle_after=2.0, deadline=15.0):
    """One overlay-on / overlay-off cycle. Returns dict with timings.

    Also flags 'early_off' when blocking goes off DURING the hold (overlay
    still on) — that's the static-suppression / VLM-no-ad / OCR-stop artifact
    that produces a misleading recover=0.00 reading.
    """
    print(f"\n--- scenario: '{text}' on for {hold_seconds}s ---")
    # baseline: blocking should be off
    if harness.get_blocking():
        print("  [warn] baseline: blocking already on, waiting for off")
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    # overlay on
    t_on = time.time()
    harness.set_overlay(text)
    detect_latency = _wait_until(harness.get_blocking, timeout=deadline)
    if detect_latency is None:
        print(f"  ✗ DETECT FAILED (no blocking within {deadline}s)")
        # No detection — sleep through the hold so timing doesn't get weird,
        # then bail.
        time.sleep(max(0, hold_seconds - (time.time() - t_on)))
        harness.set_overlay(None)
        time.sleep(settle_after)
        return {'text': text, 'hold': hold_seconds, 'detect': None,
                'recover': None, 'early_off': False}
    print(f"  detect:  {detect_latency:.2f}s")

    # Hold and watch for early termination (blocking flipping off mid-hold).
    end_at = t_on + hold_seconds
    early_off = False
    early_off_at = None
    while time.time() < end_at:
        if not harness.get_blocking():
            early_off = True
            early_off_at = time.time() - t_on
            print(f"  ⚠ EARLY-OFF at {early_off_at:.2f}s (overlay still on; static-suppression / OCR-stop fired during hold)")
            break
        time.sleep(0.05)

    # If blocking ended early, wait until end_at anyway so scenarios stay aligned.
    remain = end_at - time.time()
    if remain > 0:
        time.sleep(remain)

    # overlay off
    t_off = time.time()
    harness.set_overlay(None)
    recover_latency = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    if recover_latency is None:
        print(f"  ✗ RECOVER FAILED (still blocking after {deadline}s)")
    elif early_off:
        # Already off when overlay cleared — recover_latency is meaningless.
        print(f"  recover: N/A (was already off due to early termination)")
        recover_latency = None
    else:
        print(f"  recover: {recover_latency:.2f}s")

    time.sleep(settle_after)
    return {'text': text, 'hold': hold_seconds, 'detect': detect_latency,
            'recover': recover_latency, 'early_off': early_off,
            'early_off_at': early_off_at}


def scenario_pause_during_ad(harness, text, hold_seconds, deadline=15.0):
    """Inject overlay, then freeze the source. Measure how the harness handles
    the unfreeze (the "pause-on-ad → unpause leaves blocking sticky" case)."""
    print(f"\n--- scenario PAUSE: '{text}' overlay + freeze for {hold_seconds}s ---")
    if harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    # set overlay, wait for detection
    harness.set_overlay(text)
    detect_latency = _wait_until(harness.get_blocking, timeout=deadline)
    print(f"  detect:  {detect_latency}")

    # freeze source (simulates pause). overlay still on, frame stops moving.
    time.sleep(1.0)  # let some "ad on real frames" go by first
    harness.freeze_source(True)
    time.sleep(hold_seconds)

    # unfreeze (simulate unpause). clear overlay simultaneously.
    harness.freeze_source(False)
    harness.set_overlay(None)
    t_unpause = time.time()
    recover_latency = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover after unfreeze: {recover_latency}")
    time.sleep(2.0)
    return {'text': text, 'detect': detect_latency, 'recover_after_unfreeze': recover_latency}


def run_round1(harness):
    """Round 1: detection + recovery latency for the 3 ad text patterns,
    each held for 5/10/20s windows. Pure latency baseline."""
    out = []
    for text in ['AD', 'AD 1 of 1', 'AD 0:15']:
        for hold in [5, 10, 20]:
            r = scenario_detect_recover(harness, text, hold)
            out.append(r)
    print("\n=== round 1 summary ===")
    print(f"{'text':12} {'hold':>5} {'detect':>9} {'recover':>9}  {'note'}")
    for r in out:
        d = f"{r['detect']:.2f}s" if r['detect'] is not None else "FAIL"
        if r['recover'] is not None:
            rc = f"{r['recover']:.2f}s"
        elif r['early_off']:
            rc = f"early@{r['early_off_at']:.1f}s"
        else:
            rc = "FAIL" if r['detect'] is not None else "-"
        note = ''
        if r['early_off']:
            note = '(static suppression flipped mid-hold)'
        print(f"{r['text']:12} {r['hold']:>4}s  {d:>8}  {rc:>9}  {note}")
    detects = [r['detect'] for r in out if r['detect'] is not None]
    recovers = [r['recover'] for r in out if r['recover'] is not None]
    if detects:
        print(f"\ndetect:  mean={statistics.mean(detects):.2f}s  max={max(detects):.2f}s  goal=1.5s ({len(detects)}/{len(out)} clean)")
    if recovers:
        print(f"recover: mean={statistics.mean(recovers):.2f}s  max={max(recovers):.2f}s  goal=1.5s ({len(recovers)}/{len(out)} clean)")
    early = [r for r in out if r['early_off']]
    if early:
        print(f"early-off: {len(early)}/{len(out)} scenarios had blocking flip off during hold")


def scenario_multi_ad_break(harness, deadline=15.0):
    """Real-world: 3 consecutive ads with brief content gaps between them.

    YouTube/Hulu ad breaks often look like:
      | AD 1 of 3 (15s) | -> 0.5s transition -> | AD 2 of 3 (10s) | ... | content |

    Per-ad detect latency measures the per-transition behavior. We expect
    blocking to *stay on* through the transition (held by MIN_BLOCKING_DURATION
    or by OCR re-matching the next AD overlay quickly) and *not flip off*
    in the 0.5s gaps.
    """
    print(f"\n--- scenario MULTI_AD_BREAK: 3 ads with 0.5s gaps ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    sequence = [
        ('AD 1 of 3 - 0:15', 8),
        (None, 0.5),
        ('AD 2 of 3 - 0:10', 6),
        (None, 0.5),
        ('AD 3 of 3 - 0:08', 5),
    ]
    t_start_break = time.time()
    detect_latencies = []
    flap_count = 0
    last_blocking = False

    for text, hold in sequence:
        harness.set_overlay(text)
        if text:
            t = time.time()
            d = _wait_until(harness.get_blocking, timeout=deadline)
            if d is not None:
                detect_latencies.append(d)
                print(f"  '{text}': detected in {d:.2f}s")
            else:
                print(f"  '{text}': DETECT FAILED")
        # poll for flapping during this slot
        end_at = time.time() + hold
        while time.time() < end_at:
            b = harness.get_blocking()
            if b != last_blocking:
                if not b and text is not None:
                    flap_count += 1
                last_blocking = b
            time.sleep(0.05)

    # Now turn it all off
    t_off = time.time()
    harness.set_overlay(None)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover after final clear: {recover}")
    time.sleep(2)
    return {
        'kind': 'multi_ad',
        'detects': detect_latencies,
        'flap_count': flap_count,
        'final_recover': recover,
        'total_break_seconds': time.time() - t_start_break,
    }


def scenario_pre_roll(harness, hold=5, deadline=15.0):
    """Short pre-roll ad then back to content.

    Tests the simplest ad-break shape: clean enter / clean exit. Should hit
    the detect <1.5s and recover <1.5s goals comfortably.
    """
    print(f"\n--- scenario PRE_ROLL: 5s 'AD 0:05' ad ---")
    return scenario_detect_recover(harness, 'AD 0:05', hold)


def scenario_long_ad(harness, deadline=15.0):
    """Long sustained ad (25s) — tests no spurious early-off over a long
    block, even with content fluctuations."""
    print(f"\n--- scenario LONG_AD: 25s 'AD 0:25' ad ---")
    return scenario_detect_recover(harness, 'AD 0:25', 25)


def scenario_real_pause_on_ad(harness, pause_seconds, ad_ends_during_pause=True, deadline=15.0):
    """User pauses ON an ad, ad ends offscreen during the pause, user unpauses.
    This is the exact case the user reported (paused on Netflix ad, unpaused,
    saw extra blocking on actual content).

    Sequence:
      1. content plays
      2. AD overlay appears
      3. detect blocking
      4. (after 3s) freeze source ← user pauses
      5. (after pause_seconds) unfreeze ← user unpauses
         If ad_ends_during_pause: clear overlay simultaneous with unfreeze
      6. measure recover-to-blocking-off latency
    """
    print(f"\n--- scenario REAL_PAUSE: ad seen, pause {pause_seconds}s, "
          f"{'ad cleared' if ad_ends_during_pause else 'ad still active'} on unpause ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    harness.set_overlay('AD 0:15')
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    if detect is None:
        print("  ✗ DETECT FAILED")
        harness.set_overlay(None)
        return {'kind': 'real_pause', 'detect': None}
    print(f"  detect: {detect:.2f}s")

    # let blocking run on real motion for a moment, then pause
    time.sleep(3)
    harness.freeze_source(True)
    print(f"  user paused (frame frozen)")
    time.sleep(pause_seconds)

    # unfreeze + (optionally) clear overlay simultaneously
    t_unpause = time.time()
    harness.freeze_source(False)
    if ad_ends_during_pause:
        harness.set_overlay(None)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    if recover is not None:
        print(f"  recover after unpause: {recover:.2f}s")
    else:
        print(f"  ✗ RECOVER FAILED")

    if not ad_ends_during_pause:
        # ad continued; we still need to clear it before scenario ends
        time.sleep(2)
        harness.set_overlay(None)
        time.sleep(2)

    return {'kind': 'real_pause', 'pause_seconds': pause_seconds,
            'ad_cleared_at_unpause': ad_ends_during_pause,
            'detect': detect, 'recover': recover}


def scenario_vlm_only_trigger(harness, deadline=20.0):
    """No OCR overlay; inject VLM='ad' votes only. Measure how long until
    vlm_ad_detected → True and blocking fires.

    With vlm_min_decisions=4 and VLM_INTERVAL=2s, theoretical floor is 8s
    (4 votes × 2s). Plus the cycle wait → ~8-10s typical."""
    print(f"\n--- scenario VLM_ONLY_TRIGGER: no OCR, inject VLM='ad' until block ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    harness.set_overlay(None)  # no OCR signal
    harness.inject_vlm('ad', confidence=0.75)
    t = time.time()
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    if detect is None:
        print(f"  ✗ DETECT FAILED — VLM didn't trigger blocking within {deadline}s")
    else:
        print(f"  detect (VLM-only): {detect:.2f}s")
    # cleanup
    harness.inject_vlm(None)
    harness.set_overlay(None)
    time.sleep(2)
    return {'kind': 'vlm_only_trigger', 'detect': detect}


def scenario_vlm_only_recover(harness, deadline=30.0):
    """Get VLM into ad_detected, then stop injecting ad votes (inject no-ad)
    and measure how long until VLM clears.

    With vlm_stop_agreement=0.75 + hysteresis 0.10 = 0.85, after N ad votes
    we need M no-ad votes such that M/(N+M) >= 0.85 → M >= 5.67*N. So with
    4 ad votes: ~23 no-ad votes ≈ 46s at 2s interval. Probably bad.
    """
    print(f"\n--- scenario VLM_ONLY_RECOVER: build ad votes then flip to no-ad ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    harness.set_overlay(None)
    harness.inject_vlm('ad', confidence=0.75)
    t_inj = time.time()
    detect = _wait_until(harness.get_blocking, timeout=20)
    if detect is None:
        print(f"  ✗ failed to set up — couldn't get VLM-only block")
        harness.inject_vlm(None)
        return {'kind': 'vlm_only_recover', 'setup': 'fail'}
    print(f"  set up VLM-only block in {detect:.2f}s")

    # Now flip to no-ad votes; measure recovery
    t_flip = time.time()
    harness.inject_vlm('no-ad', confidence=0.75)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    if recover is None:
        print(f"  ✗ RECOVER FAILED after {deadline}s of no-ad votes")
    else:
        print(f"  recover (VLM-only): {recover:.2f}s")
    harness.inject_vlm(None)
    time.sleep(2)
    return {'kind': 'vlm_only_recover', 'setup_detect': detect, 'recover': recover}


def scenario_ocr_plus_vlm_corroborate(harness, deadline=15.0):
    """OCR finds ad AND VLM votes ad. Both signals align — should fire fast
    (OCR triggers, VLM corroborates → source = 'both')."""
    print(f"\n--- scenario OCR+VLM corroborate: both fire ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    harness.inject_vlm('ad', confidence=0.75)
    harness.set_overlay('AD 0:15')
    t = time.time()
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    print(f"  detect: {detect}")
    # let the VLM corroboration accumulate (4+ votes → vlm_ad_detected=True)
    time.sleep(10)
    # Now check the source
    src = harness.engine.blocking_source
    vlm_det = harness.engine.vlm_ad_detected
    ocr_det = harness.engine.ocr_ad_detected
    print(f"  after 10s: ocr_ad_detected={ocr_det}, vlm_ad_detected={vlm_det}, source={src}")
    # cleanup
    harness.inject_vlm(None)
    harness.set_overlay(None)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover (after both stop): {recover}")
    time.sleep(2)
    return {'kind': 'ocr_vlm_corroborate', 'detect': detect, 'source_after_10s': src,
            'vlm_detected_after_10s': vlm_det, 'recover': recover}


def scenario_ocr_only_vlm_dissents(harness, deadline=15.0):
    """OCR finds ad. VLM votes no-ad. Blocking must persist via OCR."""
    print(f"\n--- scenario OCR-only + VLM dissents: OCR triggers, VLM disagrees ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    harness.inject_vlm('no-ad', confidence=0.75)
    harness.set_overlay('AD 0:15')
    t = time.time()
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    print(f"  detect: {detect}")
    # blocking should still be on after 8s (OCR keeps matching)
    time.sleep(8)
    still_on = harness.get_blocking()
    print(f"  after 8s of OCR-yes / VLM-no: blocking={still_on}")
    # cleanup
    harness.inject_vlm(None)
    harness.set_overlay(None)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover: {recover}")
    time.sleep(2)
    return {'kind': 'ocr_yes_vlm_no', 'detect': detect, 'still_on_at_8s': still_on,
            'recover': recover}


def scenario_vlm_stuck_after_pause(harness, pause_seconds=6, deadline=20.0):
    """The user-reported case but at the VLM layer: ad detected (OCR+VLM),
    user pauses, ad ends offscreen during pause, user unpauses on real
    content. Does VLM's sliding-window history keep blocking active?"""
    print(f"\n--- scenario VLM_STUCK_AFTER_PAUSE: build block, pause {pause_seconds}s, unpause to clean content ---")
    while harness.get_blocking():
        _wait_until(lambda: not harness.get_blocking(), timeout=20)

    # Set up: both OCR and VLM say ad
    harness.inject_vlm('ad', confidence=0.75)
    harness.set_overlay('AD 0:15')
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    print(f"  initial detect: {detect}")

    # Let VLM accumulate ad votes (4+) so it's truly vlm_ad_detected=True
    time.sleep(10)
    print(f"  before pause: vlm_ad_detected={harness.engine.vlm_ad_detected}, "
          f"history={len(harness.engine.vlm_decision_history)} entries")

    # User pauses
    harness.freeze_source(True)
    time.sleep(pause_seconds)

    # Unpause: simulate "ad ended during pause" — clear OCR overlay AND
    # flip VLM injection to no-ad
    t_unpause = time.time()
    harness.freeze_source(False)
    harness.set_overlay(None)
    harness.inject_vlm('no-ad', confidence=0.75)

    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    if recover is not None:
        print(f"  recover after unpause: {recover:.2f}s")
    else:
        print(f"  ✗ RECOVER FAILED after {deadline}s")
    harness.inject_vlm(None)
    time.sleep(2)
    return {'kind': 'vlm_stuck_after_pause', 'pause_seconds': pause_seconds,
            'detect': detect, 'recover': recover}


def run_round5(harness):
    """Round 5: VLM state machine via injected verdicts.

    State is reset between scenarios so they don't pollute each other.
    """
    print("\n=== ROUND 5: VLM state machine (injected verdicts) ===")
    out = []
    for fn in [
        scenario_vlm_only_trigger,
        scenario_vlm_only_recover,
        scenario_ocr_plus_vlm_corroborate,
        scenario_ocr_only_vlm_dissents,
    ]:
        harness.reset_state()
        out.append(fn(harness))
    for ps in [3, 6, 15]:
        harness.reset_state()
        out.append(scenario_vlm_stuck_after_pause(harness, pause_seconds=ps))
    print("\n=== ROUND 5 SUMMARY ===")
    for r in out:
        print(r)


# ---------------------------------------------------------------------------
# Round 6: regression — the user-reported bug case
# ---------------------------------------------------------------------------
def scenario_user_bug_pause_on_ad(harness, deadline=15.0):
    """Recreate the user's scenario:

      - Real ad detected (OCR + VLM agree), blocking on for ~5s
      - User pauses (3s — long enough to fire static suppression at the
        2.5s STATIC_TIME_THRESHOLD default, short enough to see the
        post-cooldown re-check behavior)
      - Ad ends offscreen during the pause (overlay cleared at unfreeze)
      - User unpauses → blocking should clear within 1.5s

    With ORIGINAL params (DYNAMIC_COOLDOWN=0.5, OCR_STOP=4,
    scene_change=0.01) this is where the user observed "5 seconds of
    blocking on the actual video".
    """
    harness.reset_state()
    print(f"\n--- USER BUG SCENARIO: 5s ad → 3s pause-during-ad → unpause to clean content ---")

    # Phase 1: ad starts, both signals say ad
    harness.inject_vlm('ad', confidence=0.75)
    harness.set_overlay('AD 0:15')
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    print(f"  detect: {detect}")
    # Let blocking run on actual ad for 5s (user watches the ad)
    time.sleep(5)

    # Phase 2: user pauses (the freeze fires static suppression after ~2.5s)
    harness.freeze_source(True)
    print(f"  user paused")
    time.sleep(3)

    # Phase 3: user unpauses; the ad has ended during the pause
    t_unpause = time.time()
    harness.freeze_source(False)
    harness.set_overlay(None)
    harness.inject_vlm('no-ad', confidence=0.75)

    # Measure: how long after unpause does the SCREEN actually become unblocked?
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover after unpause: {recover}")

    # Also measure if blocking ever phantom-fires AGAIN after unpause within
    # the next 5 seconds (the user-reported "5 seconds of blocking on real
    # content" pattern).
    end_at = time.time() + 5
    re_block_at = None
    while time.time() < end_at:
        if harness.get_blocking():
            re_block_at = time.time() - t_unpause
            break
        time.sleep(0.05)
    if re_block_at is not None:
        print(f"  ⚠ phantom RE-BLOCK at {re_block_at:.2f}s after unpause "
              f"(this is the user-reported bug)")

    harness.inject_vlm(None)
    time.sleep(2)
    return {'kind': 'user_bug', 'detect': detect, 'recover_after_unpause': recover,
            'phantom_re_block_at': re_block_at}


def scenario_no_ad_no_vlm_no_block(harness, hold=15.0):
    """False-positive resistance: BBB plays for 15s with no overlay and no
    VLM injection. Blocking must stay off the entire time."""
    harness.reset_state()
    print(f"\n--- scenario NO_AD_NO_VLM: {hold}s of clean BBB, must stay unblocked ---")
    harness.set_overlay(None)
    harness.inject_vlm(None)
    end_at = time.time() + hold
    blocked_at = None
    while time.time() < end_at:
        if harness.get_blocking():
            blocked_at = time.time() - (end_at - hold)
            break
        time.sleep(0.1)
    if blocked_at is None:
        print(f"  ✓ stayed clean for {hold}s (no false-positive blocking)")
    else:
        print(f"  ✗ FALSE POSITIVE: blocked at {blocked_at:.2f}s")
    return {'kind': 'no_ad_no_vlm', 'hold': hold, 'false_positive_at': blocked_at}


def scenario_realistic_multi_ad_with_vlm(harness, deadline=15.0):
    """3-ad break with both OCR overlay AND VLM injection 'ad' throughout
    the break. Mirrors what production sees on a real Netflix/Hulu break:
    OCR sees the timestamp + 'Skip Ad', VLM sees the visual ad content."""
    harness.reset_state()
    print(f"\n--- scenario REALISTIC_MULTI_AD: 3-ad break with VLM corroborating ---")
    harness.inject_vlm('ad', confidence=0.75)
    sequence = [('AD 1 of 3 - 0:15', 8), (None, 0.5),
                ('AD 2 of 3 - 0:10', 6), (None, 0.5),
                ('AD 3 of 3 - 0:08', 5)]
    detect_latencies = []
    flap_count = 0
    last = False
    t_break = time.time()
    for text, hold in sequence:
        harness.set_overlay(text)
        if text:
            t = time.time()
            d = _wait_until(harness.get_blocking, timeout=deadline)
            if d is not None:
                detect_latencies.append(d)
                print(f"  '{text}': detect {d:.2f}s  (source={harness.engine.blocking_source})")
        end_at = time.time() + hold
        while time.time() < end_at:
            b = harness.get_blocking()
            if b != last:
                if not b and text is not None:
                    flap_count += 1
                last = b
            time.sleep(0.05)
    # End the break
    harness.set_overlay(None)
    harness.inject_vlm('no-ad', confidence=0.75)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover after break end: {recover:.2f}s  (source was {harness.engine.blocking_source or 'cleared'})")
    harness.inject_vlm(None)
    time.sleep(2)
    return {'kind': 'realistic_multi_ad', 'detects': detect_latencies,
            'flaps': flap_count, 'recover': recover,
            'total_break': time.time() - t_break}


def scenario_long_ad_with_vlm(harness, deadline=15.0):
    """30s ad with both OCR + VLM. Should stay blocked the full 30s
    without spurious early-off."""
    harness.reset_state()
    print(f"\n--- scenario LONG_AD_WITH_VLM: 30s sustained ad + VLM corroborating ---")
    harness.inject_vlm('ad', confidence=0.75)
    harness.set_overlay('AD 0:30')
    detect = _wait_until(harness.get_blocking, timeout=deadline)
    print(f"  detect: {detect}")
    # Hold 30s, watch for flaps
    end_at = time.time() + 30
    flap_count = 0
    last = True
    while time.time() < end_at:
        b = harness.get_blocking()
        if b != last:
            if not b:
                flap_count += 1
            last = b
        time.sleep(0.05)
    print(f"  flaps during 30s: {flap_count}")
    harness.set_overlay(None)
    harness.inject_vlm('no-ad', confidence=0.75)
    recover = _wait_until(lambda: not harness.get_blocking(), timeout=deadline)
    print(f"  recover: {recover}")
    harness.inject_vlm(None)
    time.sleep(2)
    return {'kind': 'long_ad_with_vlm', 'detect': detect, 'flaps': flap_count,
            'recover': recover}


def run_round7(harness):
    """Round 7: realistic production scenarios with both OCR + VLM signals."""
    print("\n=== ROUND 7: realistic production-shaped scenarios ===")
    out = []
    out.append(scenario_no_ad_no_vlm_no_block(harness, hold=15))
    out.append(scenario_realistic_multi_ad_with_vlm(harness))
    out.append(scenario_long_ad_with_vlm(harness))
    print("\n=== ROUND 7 SUMMARY ===")
    for r in out:
        print(r)


def run_round6(harness, params_label='current'):
    print(f"\n=== ROUND 6: user bug regression ({params_label} params) ===")
    out = []
    for _ in range(3):
        r = scenario_user_bug_pause_on_ad(harness)
        out.append(r)
    print(f"\n=== ROUND 6 SUMMARY ({params_label}) ===")
    for r in out:
        recv = r.get('recover_after_unpause')
        rb = r.get('phantom_re_block_at')
        recv_s = f"{recv:.2f}s" if isinstance(recv, (int, float)) and recv > 0.01 else (
            "N/A (was off)" if recv is not None else "FAIL")
        rb_s = f"⚠ re-block at {rb:.2f}s" if rb is not None else "no re-block"
        print(f"  recover={recv_s:>14}   {rb_s}")
    re_blocks = [r for r in out if r['phantom_re_block_at'] is not None]
    print(f"\n  Phantom re-blocks: {len(re_blocks)}/{len(out)}")
    if re_blocks:
        max_rb = max(r['phantom_re_block_at'] for r in re_blocks)
        print(f"  Max phantom re-block latency: {max_rb:.2f}s")


def run_round4(harness):
    """Realistic ad-break shapes, including the user-reported pause case."""
    print("\n=== ROUND 4: realistic ad break patterns ===")
    out = []

    out.append(scenario_pre_roll(harness, hold=5))
    out.append(scenario_long_ad(harness))
    out.append(scenario_multi_ad_break(harness))
    # Pause-on-ad: short (under static threshold) and long (over)
    for ps, clear_during in [(2, True), (3, True), (6, True), (15, True), (6, False)]:
        out.append(scenario_real_pause_on_ad(harness, pause_seconds=ps,
                                             ad_ends_during_pause=clear_during))

    print("\n=== ROUND 4 SUMMARY ===")
    for r in out:
        print(r)


def run_pause(harness):
    """Pause-on-ad scenario.

    Two variants:
      short: 2s pause (under static threshold) — should keep blocking
             through the pause; recovery measured from overlay-clear.
      long:  6s pause (over 4s static threshold) — suppression fires
             during pause, cooldown clears state on unfreeze; recovery
             measured from clear-overlay & unfreeze.
    """
    out = []
    for text in ['AD 0:15', 'AD 1 of 1']:
        for hold, label in [(2, 'short'), (6, 'long')]:
            print(f"\n--- pause variant: {label} ({hold}s freeze) ---")
            r = scenario_pause_during_ad(harness, text, hold_seconds=hold)
            r['variant'] = label
            out.append(r)
    print("\n=== pause summary ===")
    print(f"{'text':12} {'variant':8} {'detect':>8} {'rec-unfreeze':>13}")
    for r in out:
        d = f"{r['detect']:.2f}s" if r['detect'] is not None else "FAIL"
        rc = (f"{r['recover_after_unfreeze']:.2f}s"
              if r['recover_after_unfreeze'] is not None else "N/A")
        print(f"{r['text']:12} {r['variant']:8} {d:>8} {rc:>13}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scenarios', nargs='*', default=['round1'],
                    help='scenarios to run: round1, pause')
    args = ap.parse_args()

    harness = Harness(PARAMS)
    harness.start()

    runner = threading.Thread(target=harness.run, daemon=True)
    runner.start()

    # warm-up: let video play for 2s before starting tests
    time.sleep(2)

    try:
        for scenario in args.scenarios:
            if scenario == 'round1':
                run_round1(harness)
            elif scenario == 'round4':
                run_round4(harness)
            elif scenario == 'round5':
                run_round5(harness)
            elif scenario == 'round6':
                run_round6(harness)
            elif scenario == 'round7':
                run_round7(harness)
            elif scenario == 'pause':
                run_pause(harness)
            else:
                print(f"unknown scenario: {scenario}")
    finally:
        # dump event log for forensics
        log_path = '/tmp/block_latency_events.log'
        with open(log_path, 'w') as f:
            for e in harness.events:
                f.write(f"{e['t']:.3f} {e['kind']} {e.get('data')}\n")
        print(f"\nevent log: {log_path}")
        harness.stop()


if __name__ == '__main__':
    main()
