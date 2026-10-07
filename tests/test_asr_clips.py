#!/usr/bin/env python3
"""Unit tests for the ASR training-clip collector (src/asr_clips.py)."""
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
os.environ.setdefault('MINUS_ASR_CLIPS_RANDOM_S', '0')
from asr_clips import ASRClipCollector  # noqa: E402


class FakeTap:
    SAMPLE_RATE = 16000

    def __init__(self, level=3000):
        self.level = level
        self.end = None

    def recent_samples(self, seconds):
        n = int(min(seconds, 30) * self.SAMPLE_RATE)
        rng = np.random.default_rng(0)
        data = (rng.standard_normal(n) * self.level).astype(np.int16)
        return data, self.end if self.end is not None else time.monotonic()


def make(level=3000, **env):
    d = tempfile.mkdtemp()
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        c = ASRClipCollector(FakeTap(level), d, engine='test', delay_fn=lambda: 1.2)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return c, d


def entries(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.json'):
            with open(os.path.join(d, f)) as fh:
                out.append(json.load(fh))
    return out


class TestASRClipCollector(unittest.TestCase):
    def test_mention_saved_after_post_window(self):
        c, d = make()
        now = time.monotonic()
        c.on_detection({'source': 'asr', 'label': 'for lebron james', 'capture_start': now - 10})
        c._tick(now)
        meta = entries(d)
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]['kind'], 'asr')
        self.assertAlmostEqual(meta[0]['duration_s'], c.PRE_S + c.POST_S, places=1)
        self.assertTrue(os.path.exists(os.path.join(d, [f for f in os.listdir(d) if f.endswith('.wav')][0])))

    def test_not_saved_before_due(self):
        c, d = make()
        now = time.monotonic()
        c.on_detection({'source': 'asr', 'label': 'lebron', 'capture_start': now})
        c._tick(now)
        self.assertEqual(entries(d), [])
        self.assertEqual(c.get_status()['pending'], 1)

    def test_caption_only_is_tagged(self):
        c, d = make()
        now = time.monotonic()
        c.on_detection({'source': 'caption', 'label': 'LeBron James', 'capture_start': now - 10})
        c._tick(now)
        self.assertEqual(entries(d)[0]['kind'], 'caption_only')

    def test_asr_and_caption_merge_into_one_clip(self):
        c, d = make()
        now = time.monotonic()
        c.on_detection({'source': 'caption', 'label': 'LeBron', 'capture_start': now - 10})
        c.on_detection({'source': 'asr', 'label': 'lebron', 'capture_start': now - 9.5})
        c._tick(now)
        meta = entries(d)
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]['kind'], 'both')
        self.assertEqual(len(meta[0]['detections']), 2)

    def test_far_apart_detections_are_separate_clips(self):
        c, d = make()
        now = time.monotonic()
        c.on_detection({'source': 'asr', 'label': 'a', 'capture_start': now - 20})
        c.on_detection({'source': 'asr', 'label': 'b', 'capture_start': now - 10})
        c._tick(now)
        self.assertEqual(len(entries(d)), 2)

    def test_context_relative_times(self):
        c, d = make()
        now = time.monotonic()
        c.tap.end = now
        mention = now - 6.0
        c.note_asr('here is lebron', [('here', 0.1, 0.3), ('lebron', 1.0, 1.5)], mention - 1.0)
        c.note_ocr(['and LeBron drives'], mention + 0.5)
        c.on_detection({'source': 'asr', 'label': 'lebron', 'capture_start': mention})
        c._tick(now)
        meta = entries(d)[0]
        clip_start = now - (now - (mention - c.PRE_S))
        self.assertAlmostEqual(meta['mention_t'], mention - clip_start, places=2)
        self.assertEqual(meta['asr_windows'][0]['text'], 'here is lebron')
        self.assertAlmostEqual(meta['asr_windows'][0]['words'][1][1],
                               mention - 1.0 + 1.0 - clip_start, places=2)
        self.assertEqual(meta['ocr'][0]['lines'], ['and LeBron drives'])
        self.assertEqual(meta['av_delay_s'], 1.2)

    def test_repeated_ocr_screen_stored_once(self):
        c, _ = make()
        t = time.monotonic()
        c.note_ocr(['same line'], t)
        c.note_ocr(['same line'], t + 0.3)
        c.note_ocr(['new line'], t + 0.6)
        self.assertEqual(len(c._ocr), 2)

    def test_random_clip_and_silence_skip(self):
        c, d = make(MINUS_ASR_CLIPS_RANDOM_S=1)
        c._last_random -= 5
        c._tick(time.monotonic())
        self.assertEqual(entries(d)[0]['kind'], 'random')
        self.assertIsNone(entries(d)[0]['mention_t'])
        q, dq = make(level=0, MINUS_ASR_CLIPS_RANDOM_S=1)
        q._last_random -= 5
        q._tick(time.monotonic())
        self.assertEqual(entries(dq), [])

    def test_sidecar_has_monotonic_clip_start(self):
        c, d = make()
        now = time.monotonic()
        c.tap.end = now
        c.on_detection({'source': 'asr', 'label': 'x', 'capture_start': now - 10})
        c._tick(now)
        meta = entries(d)[0]
        self.assertAlmostEqual(meta['clip_start_mono'], now - 10 - c.PRE_S, delta=0.05)

    def test_set_random_interval(self):
        c, d = make()
        c.set_random_interval(20)
        self.assertEqual(c.get_status()['random_interval_s'], 20)
        c._last_random -= 25
        c._tick(time.monotonic())
        self.assertEqual(entries(d)[0]['kind'], 'random')
        c.set_random_interval(-5)
        self.assertEqual(c.random_interval_s, 0)

    def test_disabled_collects_nothing(self):
        c, d = make(MINUS_ASR_CLIPS=0)
        now = time.monotonic()
        c.on_detection({'source': 'asr', 'label': 'x', 'capture_start': now - 10})
        c._tick(now)
        self.assertEqual(entries(d), [])

    def test_budget_evicts_oldest(self):
        c, d = make(MINUS_ASR_CLIPS_BUDGET_MB=0.7)   # ~2 clips of 320 KB
        now = time.monotonic()
        for i in range(4):
            c.on_detection({'source': 'asr', 'label': str(i), 'capture_start': now - 40 + i * 5})
            c._tick(now)
            time.sleep(0.01)
        c._enforce_budget()
        wavs = [f for f in os.listdir(d) if f.endswith('.wav')]
        self.assertLessEqual(len(wavs), 2)
        self.assertGreater(c.evicted, 0)
        jsons = [f for f in os.listdir(d) if f.endswith('.json')]
        self.assertEqual(len(wavs), len(jsons))


class TestTapRecentSamples(unittest.TestCase):
    def test_recent_samples_matches_snapshot(self):
        from audio import AudioASRTap
        tap = AudioASRTap(wav_path=tempfile.mktemp(suffix='.wav'))
        n = tap.SAMPLE_RATE * 12
        sig = (np.arange(n) % 30000).astype(np.int16)
        with tap._lock:
            tap._ring[:n] = sig
            tap._write_pos = n
            tap._samples_written = n
            tap._last_write_mono = 123.0
        data, end = tap.recent_samples(10)
        self.assertEqual(end, 123.0)
        self.assertTrue(np.array_equal(data, sig[-tap.SAMPLE_RATE * 10:]))
        self.assertIsNone(tap.recent_samples(20))
        self.assertEqual(tap.snapshot_window(3, tap.wav_path), 123.0)

    def test_ring_holds_clip_context(self):
        from audio import AudioASRTap
        self.assertGreaterEqual(AudioASRTap.BUFFER_SECONDS,
                                ASRClipCollector.PRE_S + ASRClipCollector.POST_S + 5)


class TestSchedulerHook(unittest.TestCase):
    def test_on_detection_called_for_every_detection(self):
        from name_mute import NameMuteScheduler

        class A:
            def set_name_mute(self, m):
                pass
        sch = NameMuteScheduler(A(), lambda: 2.0)
        seen = []
        sch.on_detection = seen.append
        now = time.monotonic()
        sch.schedule(now, now + 0.5, 'asr', 'lebron')
        sch.schedule(now, now + 0.5, 'caption', 'LeBron')   # duplicate, still reported
        self.assertEqual([e['source'] for e in seen], ['asr', 'caption'])

    def test_hook_failure_does_not_break_muting(self):
        from name_mute import NameMuteScheduler

        class A:
            def set_name_mute(self, m):
                pass
        sch = NameMuteScheduler(A(), lambda: 2.0)
        sch.on_detection = lambda e: 1 / 0
        now = time.monotonic()
        sch.schedule(now, now + 0.5, 'asr', 'lebron')
        self.assertEqual(sch.mute_count, 1)


if __name__ == '__main__':
    unittest.main()
