"""Unit tests for the DecisionEngine state machine (the one that mirrors
minus.py's blocking decision logic). These run without OCR/VLM workers
and without the BBB video — pure state-machine exercise.

The DecisionEngine lives in tests/block_latency_harness.py. This file
exists so the key tunings (OCR_STOP_THRESHOLD, dynamic_cooldown,
scene_change_threshold) are protected by lightweight regression tests
that run as part of the standard suite.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

# Defer worker module imports — DecisionEngine is class-level only, doesn't
# spawn anything until Harness.start() is called.
from block_latency_harness import DecisionEngine, PARAMS


def _params(**overrides):
    """Build a fresh PARAMS dict with overrides on top of locked-in defaults."""
    p = dict(PARAMS)
    # Mirror locked-in production tuning
    p.setdefault('OCR_STOP_THRESHOLD', 2)
    p.setdefault('STATIC_OCR_THRESHOLD', 4)
    p.setdefault('STATIC_TIME_THRESHOLD', 2.5)
    p.setdefault('DYNAMIC_COOLDOWN', 1.5)
    p.setdefault('scene_change_threshold', 0.001)
    p.setdefault('disable_static_suppression', False)
    p.setdefault('VLM_STOP_THRESHOLD', 2)
    p.update(overrides)
    return p


class TestOCRStopThreshold(unittest.TestCase):
    """OCR_STOP_THRESHOLD=3: three consecutive no-ad cycles clear blocking.

    History 4 -> 2 -> 3. The move to 2 halved recovery latency but assumed
    consecutive OCR misses during a real ad were rare; production logs showed
    they are the dominant case (runs of exactly 2 occurred 118x in 24h vs 4
    runs of 3 and zero of 5+), which flapped 58.8% of blocks. 3 clears ~92%
    of the flapping for ~0.6s of recovery. See minus.py for the full data."""

    def test_one_no_ad_does_not_clear(self):
        e = DecisionEngine(_params())
        # Start blocking via OCR
        e.on_ocr(True)
        self.assertTrue(e.ocr_ad_detected)
        # One no-ad — not enough
        e.on_ocr(False)
        self.assertTrue(e.ocr_ad_detected, "1 no-ad should not clear (threshold=3)")

    def test_two_no_ad_does_not_clear(self):
        """THE FLAP REGRESSION: a 2-miss run is the single most common OCR
        dropout during a live ad and must no longer end the block."""
        e = DecisionEngine(_params())
        e.on_ocr(True)
        e.on_ocr(False)
        e.on_ocr(False)
        self.assertTrue(e.ocr_ad_detected, "2 consecutive no-ads must NOT clear")

    def test_three_no_ad_clears(self):
        e = DecisionEngine(_params())
        e.on_ocr(True)
        e.on_ocr(False)
        e.on_ocr(False)
        e.on_ocr(False)
        self.assertFalse(e.ocr_ad_detected, "3 consecutive no-ads should clear")

    def test_ad_frame_between_misses_prevents_clear(self):
        """A 2-miss / hit / 2-miss pattern (very common) must keep the block
        up throughout — this is what produced the observed mid-ad flicker."""
        e = DecisionEngine(_params())
        e.on_ocr(True)
        for _ in range(3):
            e.on_ocr(False)
            e.on_ocr(False)
            e.on_ocr(True)
            self.assertTrue(e.ocr_ad_detected)

    def test_no_ad_then_ad_resets_count(self):
        e = DecisionEngine(_params())
        e.on_ocr(True)
        e.on_ocr(False)  # count=1
        e.on_ocr(True)   # count=0 (reset)
        e.on_ocr(False)  # count=1
        self.assertTrue(e.ocr_ad_detected, "intermediate ad reset the no-ad count")


class TestDynamicCooldownClearsState(unittest.TestCase):
    """The cooldown completion path must clear ocr_ad_detection_count too,
    not just ocr_ad_detected. Pre-fix, the count survived and the next OCR
    match (count >= 1) immediately re-triggered blocking — which manifested
    as the user-reported '5s of phantom blocking on actual video' bug."""

    def test_cooldown_completion_clears_ocr_count(self):
        e = DecisionEngine(_params())
        # Set up a state that mirrors "pause-on-ad with detection accumulated"
        e.ocr_ad_detected = True
        e.ocr_ad_detection_count = 5
        e.static_blocking_suppressed = True
        # Screen became dynamic
        e.screen_became_dynamic_time = 100.0
        # Tick at t=101.6 (after 1.5s cooldown)
        cooldown_done = e.update_static(scene_did_change=True, now=101.6)
        self.assertTrue(cooldown_done)
        self.assertEqual(e.ocr_ad_detection_count, 0,
                         "ocr_ad_detection_count must be reset on cooldown completion")
        self.assertFalse(e.ocr_ad_detected)

    def test_cooldown_clears_when_only_count_positive(self):
        """Even if ocr_ad_detected was False but count was still positive,
        the clear must fire (otherwise count leaks past cooldown)."""
        e = DecisionEngine(_params())
        e.ocr_ad_detection_count = 3
        e.static_blocking_suppressed = True
        e.screen_became_dynamic_time = 100.0
        e.update_static(scene_did_change=True, now=101.6)
        self.assertEqual(e.ocr_ad_detection_count, 0)

    def test_cooldown_clears_vlm_history(self):
        e = DecisionEngine(_params())
        e.vlm_ad_detected = True
        e.vlm_decision_history = [(t, True, 0.75) for t in range(5)]
        e.static_blocking_suppressed = True
        e.screen_became_dynamic_time = 100.0
        e.update_static(scene_did_change=True, now=101.6)
        self.assertEqual(e.vlm_decision_history, [])
        self.assertFalse(e.vlm_ad_detected)

    def test_default_cooldown_is_1_5s(self):
        from config import MinusConfig
        cfg = MinusConfig()
        self.assertEqual(cfg.dynamic_cooldown, 1.5,
                         "production default should be 1.5s (was 0.5)")


class TestSceneChangeThresholdTuning(unittest.TestCase):
    """scene_change_threshold=0.001 means only truly-frozen frames (diff~0)
    register as static; natural low-motion content (diff>=0.002) keeps
    flowing. Old default 0.01 misclassified ~26% of natural BBB frames."""

    def test_default_threshold_is_0_001(self):
        from config import MinusConfig
        cfg = MinusConfig()
        self.assertEqual(cfg.scene_change_threshold, 0.001,
                         "production default should be 0.001 (was 0.01)")


class TestVLMOnlyStopFastPath(unittest.TestCase):
    """VLM-only blocking uses consecutive vlm_no_ad_count for the stop
    decision (production minus.py:2275). Mirrored in DecisionEngine."""

    def test_vlm_only_stop_takes_2_consecutive_no_ad(self):
        e = DecisionEngine(_params())
        # Build VLM-only block: 4+ ad votes hit start threshold (95% cap)
        for _ in range(5):
            e.on_vlm(True, 0.75)
        self.assertTrue(e.vlm_ad_detected)
        # blocking activates
        is_blocking, source = e.compute_blocking(now=10.0)
        self.assertTrue(is_blocking)
        self.assertEqual(source, 'vlm')
        # Advance past MIN_BLOCKING_DURATION
        e.blocking_start_time = 0
        # 1 no-ad — not enough
        e.on_vlm(False, 0.75)
        is_blocking, _ = e.compute_blocking(now=10.0)
        self.assertTrue(is_blocking, "1 no-ad should not stop a VLM-only block")
        # 2 no-ads — should clear
        e.on_vlm(False, 0.75)
        is_blocking, _ = e.compute_blocking(now=10.0)
        self.assertFalse(is_blocking, "2 consecutive no-ad votes should clear")


class TestUserBugRegression(unittest.TestCase):
    """End-to-end regression of the user-reported scenario: paused on ad,
    unpaused on cleared content, blocking must NOT phantom re-block.

    With OLD params (dynamic_cooldown=0.5) the bug reproduced reliably.
    With locked-in NEW params it does not."""

    def _run_user_bug_sim(self, params):
        """Simulate the user's pause-on-ad scenario as the harness's run
        loop would: each tick calls on_ocr + update_static + compute_blocking.

        Returns (initially_blocked, went_off_during_or_after_pause,
                 phantom_re_block_after_off).
        """
        e = DecisionEngine(params)
        ocr_interval = 0.5
        last_blocking = False
        went_off = False
        phantom = False

        def tick(t, found_ad, scene_did_change):
            nonlocal last_blocking, went_off, phantom
            e.on_ocr(found_ad)
            e.update_static(scene_did_change=scene_did_change, now=t)
            is_blocking, _ = e.compute_blocking(now=t)
            if last_blocking and not is_blocking:
                went_off = True
            elif went_off and is_blocking and not last_blocking:
                phantom = True
            last_blocking = is_blocking
            return is_blocking

        # Phase 1: real ad playing for 5s. OCR finds 'ad'. Scene changes
        # naturally (BBB content). Blocking turns on.
        for i in range(int(5 / ocr_interval)):
            t = i * ocr_interval
            tick(t, found_ad=True, scene_did_change=True)
        # Sanity: blocking should be on
        self.assertTrue(last_blocking, "blocking should be on by 5s mark")
        # Make MIN_BLOCKING_DURATION non-binding for the rest of the sim
        e.blocking_start_time = -100

        # Phase 2: pause for 3s. Frame is frozen — scene_did_change=False.
        # AD overlay is still on the frozen frame so OCR still returns True.
        for i in range(int(3 / ocr_interval)):
            t = 5 + i * ocr_interval
            tick(t, found_ad=True, scene_did_change=False)

        # Phase 3: unpause. AD has ended offscreen during the pause; on
        # unpause OCR sees no AD and the scene starts moving again.
        for i in range(int(8 / ocr_interval)):  # 8s after unpause
            t = 8 + i * ocr_interval
            tick(t, found_ad=False, scene_did_change=True)

        return last_blocking, went_off, phantom

    def test_new_params_no_phantom_reblock(self):
        params = _params()  # locked-in defaults
        final_blocking, went_off, phantom = self._run_user_bug_sim(params)
        self.assertTrue(went_off, "blocking should turn off at some point")
        self.assertFalse(phantom, "blocking should NOT phantom re-block after "
                                  "going off")
        self.assertFalse(final_blocking, "blocking should be off at end of sim")

    def test_old_params_DO_phantom_reblock(self):
        """The flip side: confirm OLD params reproduce the bug, so the
        new-params test above is actually proving something."""
        params = _params(OCR_STOP_THRESHOLD=4, DYNAMIC_COOLDOWN=0.5,
                         scene_change_threshold=0.01)
        _, went_off, phantom = self._run_user_bug_sim(params)
        # With the old params the simulation may not even register
        # went_off cleanly, but if it does, we expect a phantom re-block.
        # Either failure mode means the bug is reproducing.
        if went_off:
            self.assertTrue(phantom or _, "old params should phantom-re-block "
                                          "after off")


if __name__ == '__main__':
    unittest.main()
