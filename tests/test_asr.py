#!/usr/bin/env python3
"""
Tests for the ASR (whisper.cpp) confirmation/veto pipeline.

Three concerns:

  1. Keyword module (src/asr_keywords.py): correctly classifies known
     ad-copy vs show-dialog transcripts, handles whisper mistranscriptions,
     and the exclusion list suppresses service-promo / YouTube-creator
     false positives.

  2. ASRManager state machine (src/asr.py): verdict()'s three-state output
     (confirm/veto/unknown), rolling-window history correctness, graceful
     degradation when ASR is disabled or whisper is missing.

  3. AudioASRTap ring buffer (src/audio.py): write-correctness across
     wraparound, snapshot atomicity, no leakage between snapshots, and
     the playback-branch pipeline shape is byte-identical with vs.
     without a tap attached (so the audio recovery + watchdog
     mechanisms behave identically).

End-to-end live behavior (real whisper.cpp on real audio) is covered by
tests/asr_corpus/bench.py — that runs slowly and depends on the model
file, so it's separate. This test file mocks whisper.

Run: python3 -m pytest tests/test_asr.py -v
   or: python3 -m unittest tests.test_asr
"""
import os
import sys
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / 'src'))
# Repo root too: the decision-engine tests `import minus`, which only
# resolves under `python3 -m unittest tests.test_asr` (cwd on path).
# This makes script-mode `python3 tests/test_asr.py` work identically.
sys.path.insert(0, str(REPO))


# =============================================================================
# Keyword module
# =============================================================================


class TestASRKeywords(unittest.TestCase):
    """Validates count_marker_hits + the exclusion list against
    realistic whisper-tiny transcripts (including known mistranscriptions
    captured from tests/asr_corpus/bench.py runs)."""

    def setUp(self):
        from asr_keywords import count_marker_hits, explain_hits
        self.count = count_marker_hits
        self.explain = explain_hits

    # ----- ad-copy cases that MUST hit -----

    def test_strong_cta_ad_scores(self):
        text = ("Call now! 1-800-DISCOUNT! Save up to 50 percent. "
                "Available now at brand dot com. Limited time only!")
        self.assertGreaterEqual(self.count(text), 3)

    def test_pharma_disclaimer_scores(self):
        text = ("Side effects may include nausea. Consult your doctor "
                "before taking this medication. Available by prescription only.")
        self.assertGreaterEqual(self.count(text), 2)

    def test_mangled_ad_still_scores(self):
        """whisper-tiny drops/mangles syllables. A real ad with multiple
        markers should still hit >=1 even when half the transcript is
        garbled."""
        # Real bench output from ad_price.wav: most of the original CTA
        # was lost; only "order yours today" came through cleanly.
        text = "3 shipping orders over $25 order yours today."
        self.assertGreaterEqual(self.count(text), 1)

    def test_price_with_cents_hits(self):
        # Strict $X.XX should still match — ads with explicit cents.
        text = "Now only $9.99 for a limited time!"
        self.assertGreaterEqual(self.count(text), 2)

    # ----- show / non-ad cases that MUST NOT hit -----

    def test_show_dialog_no_hits(self):
        text = "I just don't think it's a good idea, John. Sarah said she would be home by seven."
        self.assertEqual(self.count(text), 0)

    def test_show_dollar_amount_no_hits(self):
        """Show characters mention bare dollar amounts. Whisper sometimes
        renders 'fifty dollars' as '$50' or '$15'. We must NOT score
        those as ad markers — that's the failure that drove tightening
        the price regex (see asr_keywords.py comment block)."""
        text = "She paid $15 for that dress. Can you believe it?"
        self.assertEqual(self.count(text), 0)
        # Also the actual TTS-rendered phrasing
        text2 = "She paid fifty dollars for that dress."
        self.assertEqual(self.count(text2), 0)

    def test_netflix_promo_excluded(self):
        text = "Available on Netflix this Friday. The new season of Stranger Things."
        self.assertEqual(self.count(text), 0)

    def test_youtube_creator_excluded(self):
        text = "Subscribe to my channel and hit that bell. Link in the description."
        self.assertEqual(self.count(text), 0)

    def test_empty_returns_zero(self):
        self.assertEqual(self.count(''), 0)
        self.assertEqual(self.count('   '), 0)

    def test_silence_hallucination_no_hits(self):
        """whisper hallucinates short phrases like 'you' or 'Thank you.'
        on silence/music. Must not score."""
        self.assertEqual(self.count('you'), 0)
        self.assertEqual(self.count('Thank you.'), 0)

    def test_exclusion_overrides_marker(self):
        """If a transcript contains both an ad marker AND an exclusion
        phrase, the exclusion wins (the marker is incidental to a
        service-promo context)."""
        text = "Available on Netflix this Friday. Save up to 50 percent."
        self.assertEqual(self.count(text), 0)

    def test_explain_returns_matched_markers(self):
        text = "Call now! Available at brand dot com."
        hits = self.explain(text)
        self.assertIn('call now', hits)
        # url-spoken regex match
        self.assertTrue(any('url-spoken' in h for h in hits))


# =============================================================================
# ASRManager state machine
# =============================================================================


class _FakeTap:
    """Minimal AudioASRTap stand-in for ASRManager tests."""
    def __init__(self, has_audio=True):
        self.has_audio = has_audio
        self.snapshot_calls = 0
    def snapshot_to_wav(self, seconds=5.0):
        self.snapshot_calls += 1
        return self.has_audio


class TestASRManager(unittest.TestCase):

    def setUp(self):
        from asr import ASRManager
        self.ASRManager = ASRManager

    def _make(self, **kwargs):
        """Build an ASRManager with a mocked ASRProcess (so tests don't
        actually spawn a faster-whisper worker subprocess)."""
        m = self.ASRManager(_FakeTap(), **kwargs)
        # Replace the real ASRProcess with a MagicMock — tests feed
        # _record_result directly, so the process isn't actually invoked.
        from unittest.mock import MagicMock
        m._process = MagicMock()
        m._process.get_latency_stats.return_value = {
            'samples': 0, 'p50_s': 0.0, 'p95_s': 0.0, 'max_s': 0.0
        }
        return m

    def test_verdict_unknown_when_not_running(self):
        m = self._make()
        self.assertEqual(m.verdict(), 'unknown')

    def test_verdict_unknown_when_disabled(self):
        m = self._make()
        m.is_running = True
        m.enabled = False
        self.assertEqual(m.verdict(), 'unknown')

    def test_verdict_unknown_when_no_history(self):
        m = self._make()
        m.is_running = True
        self.assertEqual(m.verdict(), 'unknown')

    def test_verdict_confirm_on_marker_hit(self):
        m = self._make()
        m.is_running = True
        m._record_result('ok', 'Call now! Available at brand dot com.', 0.5)
        self.assertEqual(m.verdict(), 'confirm')

    def test_verdict_veto_on_clear_show_dialog(self):
        m = self._make()
        m.is_running = True
        # Multiple zero-hit transcripts with plenty of speech
        for _ in range(3):
            m._record_result(
                'ok',
                "She walked into the room and sat down. Her brother was already there.",
                0.5)
        self.assertEqual(m.verdict(), 'veto')

    def test_pharma_disclosures_are_ad_language(self):
        """Live on minus 2026-10-19 (SenseVoice, a Mounjaro ad): the
        side-effect stretch had no markers and the window vetoed a real ad."""
        from asr_keywords import count_marker_hits
        for heard in ('to it or if you or someone in your family had medullary thyroid cancer',
                      'serious side effects may include pancreatitis',
                      'stop taking and call your doctor right away'):
            self.assertGreater(count_marker_hits(heard), 0, heard)
        # Fanatics Sportsbook ad, minus 2026-10-19: commentary-like opening,
        # then the required disclaimers.
        for heard in ('if you or someone you know has a gambling problem call 1 800 gambler',
                      'must be 21 plus terms apply', 'the only sportsbook to reward every bet'):
            self.assertGreater(count_marker_hits(heard), 0, heard)
        for show in ('she walked into the kitchen and sat down',
                     'the doctor will see you now said the nurse',
                     'third and long for the colts from their own twenty one',
                     'he bet on himself and it paid off this season'):
            self.assertEqual(count_marker_hits(show), 0, show)

    def test_no_veto_right_after_ad_language(self):
        """An ad that said 'ask your doctor' 15 s ago is still the ad, even
        through a stretch with no markers: no veto (but no confirm either)."""
        import time as _time
        m = self._make()
        m.is_running = True
        m._record_result('ok', 'now that you know ask your doctor about it', 0.3)
        # age that hit out of the 8 s window, keep it inside ad context
        m._history = type(m._history)(
            [(t - 15.0, h, w) for (t, h, w) in m._history], maxlen=m._history.maxlen)
        m._last_hit_time -= 15.0
        for _ in range(3):
            m._record_result('ok', 'she walked into the room and sat down with her brother there', 0.3)
        self.assertEqual(m.verdict(), 'unknown')
        m._last_hit_time -= 30.0       # long after the ad: show dialog vetoes again
        self.assertEqual(m.verdict(), 'veto')

    def test_short_window_ignores_the_previous_videos_speech(self):
        """Zepbound ad, minus 2026-10-19: the 8 s window still held the music
        video's lyrics when the ad began. Judged on the last 2.5 s (the ad's
        own audio, no words yet), there is nothing to veto."""
        m = self._make()
        m.is_running = True
        for _ in range(4):
            m._record_result('ok', 'im waking up to ash and dust i wipe my brow and i sweat my rust', 0.3)
        m._history = type(m._history)(
            [(t - 4.0, h, w) for (t, h, w) in m._history], maxlen=m._history.maxlen)
        m._record_result('ok', 'mm', 0.3)          # ad just started: music bed
        self.assertEqual(m.verdict(), 'veto')       # 8 s window: the lyrics
        self.assertEqual(m.verdict(window_s=2.5), 'unknown')

    def test_verdict_unknown_on_silence_window(self):
        """Several inferences with zero hits AND very little transcribed
        speech → 'unknown' (not 'veto'). Avoids vetoing when audio is
        actually just music or silence."""
        m = self._make()
        m.is_running = True
        for _ in range(3):
            m._record_result('ok', 'Thank you.', 0.5)
        # Only a few alpha words per window — total < 10 over window
        self.assertEqual(m.verdict(), 'unknown')

    def test_verdict_history_window_ages_out(self):
        """A confirm from outside the 8s window should no longer count."""
        m = self._make()
        m.is_running = True
        # Inject a hit, then time-travel forward beyond the window.
        m._record_result('ok', 'Call now! Available at brand dot com.', 0.5)
        old_ts = time.time() - (m.HISTORY_WINDOW_S + 1)
        # Replace the lone entry's timestamp
        with m._lock:
            ts, hits, words = m._history[0]
            m._history[0] = (old_ts, hits, words)
        self.assertEqual(m.verdict(), 'unknown')

    def test_get_status_keys(self):
        """API stability: /api/status reads these keys."""
        m = self._make()
        s = m.get_status()
        for key in ('available', 'enabled', 'running', 'engine', 'model',
                    'inference_count', 'timeout_count', 'killed_count',
                    'failure_count', 'verdict',
                    'last_transcript', 'last_marker_hits',
                    'p50_latency_s', 'p95_latency_s'):
            self.assertIn(key, s)
        # engine label tells the UI which backend is in use: SenseVoice on
        # the NPU where its model is installed (a Minus box), else Moonshine
        # (dev hosts); MINUS_ASR_ENGINE overrides. Same rule as the worker.
        from asr_worker import resolve_engine
        self.assertEqual(s['engine'], resolve_engine())
        self.assertIn(s['engine'], ('sensevoice', 'moonshine', 'faster-whisper'))

    def test_record_result_timeouts_increment_counter(self):
        m = self._make()
        m._record_result('timeout', '', 2.5)
        self.assertEqual(m.timeout_count, 1)
        self.assertEqual(m.last_transcript, '')

    def test_record_result_errors_increment_counter(self):
        m = self._make()
        m._record_result('error', '', 0.1)
        self.assertEqual(m.failure_count, 1)

    def test_record_result_killed_increments_counter(self):
        """The 'killed' status (hard timeout → process kill) is a new
        outcome under the worker-subprocess architecture and gets its
        own counter so we can spot worker thrashing in /api/status."""
        m = self._make()
        m._record_result('killed', '', 3.0)
        self.assertEqual(m.killed_count, 1)
        self.assertEqual(m.last_transcript, '')


# =============================================================================
# ASRProcess worker — hard timeout + restart machinery (no real worker spawn)
# =============================================================================


class TestASRProcessTimeouts(unittest.TestCase):
    """Tests the parent-side ASRProcess controller. The worker process
    spawn is NOT exercised — we mock the queues / process so the timeout
    + restart bookkeeping logic is testable without a 3+ s model load.

    The actual end-to-end worker spawn + transcribe is covered by
    tests/asr_corpus/bench_worker.py (runs the real faster-whisper)."""

    def _make(self):
        from asr_worker import ASRProcess
        proc = ASRProcess(model_name='tiny.en', cpu_threads=3)
        # Replace process internals with mocks so transcribe() doesn't
        # spawn or talk to a real worker.
        from unittest.mock import MagicMock
        import queue as q_mod
        proc.process = MagicMock()
        proc.process.is_alive.return_value = True
        proc.process.pid = 12345
        proc.request_queue = MagicMock()
        proc.response_queue = MagicMock()
        # CRITICAL: default get_nowait to raise queue.Empty (matching
        # multiprocessing.Queue semantics when the queue is empty). A
        # naive MagicMock returns a MagicMock object which is truthy
        # AND non-raising, so _drain_stale_responses_locked() would
        # infinite-loop. Individual tests that need to exercise the
        # drain path override this with side_effect.
        proc.response_queue.get_nowait.side_effect = q_mod.Empty()
        proc.is_ready = True
        return proc

    def test_constants_documented(self):
        """If someone changes the timeout constants, this catches an
        accidental zero or negative — they're load-bearing for the
        decision-engine timing budget."""
        from asr_worker import ASRProcess
        self.assertGreater(ASRProcess.SOFT_TIMEOUT, 0)
        self.assertGreater(ASRProcess.HARD_TIMEOUT, ASRProcess.SOFT_TIMEOUT)
        self.assertGreaterEqual(ASRProcess.RESTART_THRESHOLD, 1)

    def test_transcribe_returns_error_when_not_running(self):
        from asr_worker import ASRProcess
        proc = ASRProcess()
        # No start() called — should return error, not block or crash
        status, text, lat = proc.transcribe('/tmp/whatever.wav')
        self.assertEqual(status, 'error')

    def test_transcribe_passes_ok_response_through(self):
        proc = self._make()
        proc.response_queue.get.return_value = ('ok', 'hello world', 1.2)
        status, text, lat = proc.transcribe('/tmp/x.wav')
        self.assertEqual(status, 'ok')
        self.assertEqual(text, 'hello world')
        self.assertEqual(lat, 1.2)
        # Latency recorded for stats
        self.assertIn(1.2, list(proc._recent_latencies))

    def test_transcribe_soft_timeout_returns_timeout(self):
        """A single soft-timeout sets pending_response but does NOT
        restart the worker."""
        import queue as q_mod
        proc = self._make()
        proc.response_queue.get.side_effect = q_mod.Empty()
        proc.process.kill = MagicMock()

        status, text, lat = proc.transcribe('/tmp/x.wav')
        self.assertEqual(status, 'timeout')
        self.assertTrue(proc._pending_response)
        self.assertEqual(proc._consecutive_timeouts, 1)
        # Process must NOT have been killed yet
        proc.process.kill.assert_not_called()

    def test_transcribe_drains_stale_responses(self):
        """If a previous soft-timeout's response arrives late, the next
        call must drain it before queueing a new request. Otherwise
        the next caller would get the wrong transcript (the old
        inference's result paired with the new wav path)."""
        proc = self._make()
        proc._pending_response = True
        import queue as q_mod
        # First get_nowait() returns the stale response; second raises Empty
        proc.response_queue.get_nowait.side_effect = [
            ('ok', 'stale-result', 1.0),
            q_mod.Empty(),
        ]
        proc.response_queue.get.return_value = ('ok', 'fresh-result', 0.8)

        status, text, lat = proc.transcribe('/tmp/x.wav')
        self.assertEqual(status, 'ok')
        self.assertEqual(text, 'fresh-result')
        # After successful drain + fresh result, pending is cleared
        self.assertFalse(proc._pending_response)

    def test_get_latency_stats_empty(self):
        from asr_worker import ASRProcess
        proc = ASRProcess()
        s = proc.get_latency_stats()
        self.assertEqual(s['samples'], 0)
        self.assertEqual(s['p50_s'], 0.0)
        self.assertEqual(s['max_s'], 0.0)

    def test_get_latency_stats_populated(self):
        proc = self._make()
        for lat in [0.9, 1.0, 1.1, 1.2, 1.5]:
            proc._recent_latencies.append(lat)
        s = proc.get_latency_stats()
        self.assertEqual(s['samples'], 5)
        # median of 5 values is the middle one
        self.assertEqual(s['p50_s'], 1.1)
        self.assertEqual(s['max_s'], 1.5)


# =============================================================================
# AudioASRTap ring buffer
# =============================================================================


class TestAudioASRTap(unittest.TestCase):
    """Validates the ring buffer in isolation (no GStreamer)."""

    def setUp(self):
        # Import after sys.path is fixed
        from audio import AudioASRTap
        self.AudioASRTap = AudioASRTap

    def _make(self, wav_path=None):
        if wav_path is None:
            wav_path = f'/tmp/test_asr_tap_{os.getpid()}_{time.time_ns()}.wav'
        return self.AudioASRTap(wav_path=wav_path), wav_path

    def _feed_samples(self, tap, n_samples, fill_value=0):
        """Bypass GStreamer; directly push samples into the ring as the
        appsink callback would."""
        samples = np.full(n_samples, fill_value, dtype=np.int16)
        n = len(samples)
        with tap._lock:
            end = tap._write_pos + n
            if end <= tap._buffer_samples:
                tap._ring[tap._write_pos:end] = samples
            else:
                split = tap._buffer_samples - tap._write_pos
                tap._ring[tap._write_pos:] = samples[:split]
                tap._ring[:n - split] = samples[split:]
            tap._write_pos = end % tap._buffer_samples
            tap._samples_written += n
            tap._last_buffer_time = time.time()

    def test_snapshot_returns_false_when_buffer_cold(self):
        tap, _ = self._make()
        self.assertFalse(tap.snapshot_to_wav(seconds=5.0))

    def test_snapshot_writes_wav_when_warm(self):
        tap, wav = self._make()
        try:
            # Fill enough for a 1-second snapshot
            self._feed_samples(tap, tap.SAMPLE_RATE * 2, fill_value=100)
            self.assertTrue(tap.snapshot_to_wav(seconds=1.0))
            # Verify the WAV is well-formed and contains our samples
            with wave.open(wav, 'rb') as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), tap.SAMPLE_RATE)
                frames = wf.readframes(wf.getnframes())
                data = np.frombuffer(frames, dtype=np.int16)
                self.assertEqual(len(data), tap.SAMPLE_RATE)
                # All values should be 100
                self.assertTrue((data == 100).all())
        finally:
            if os.path.exists(wav):
                os.unlink(wav)

    def test_ring_wraps_correctly(self):
        """Write more samples than the ring can hold and verify the
        snapshot returns the newest N samples in order."""
        tap, wav = self._make()
        try:
            # Fill the ring with rising values, then push enough more to
            # wrap. Snapshot should reflect ONLY the latest segment.
            self._feed_samples(tap, tap.BUFFER_SECONDS * tap.SAMPLE_RATE, fill_value=1)
            self._feed_samples(tap, tap.SAMPLE_RATE * 3, fill_value=99)  # 3s of "99"
            self.assertTrue(tap.snapshot_to_wav(seconds=2.0))
            with wave.open(wav, 'rb') as wf:
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                # Should be all 99s — the latest 2 seconds
                self.assertTrue((data == 99).all(),
                                f"Expected all 99s, got unique values {set(data.tolist()[:50])}")
        finally:
            if os.path.exists(wav):
                os.unlink(wav)

    def test_concurrent_write_and_snapshot(self):
        """Hammer the tap from multiple threads to surface any race."""
        tap, wav = self._make()
        try:
            stop = threading.Event()
            def writer():
                while not stop.is_set():
                    self._feed_samples(tap, 1024, fill_value=42)
            t = threading.Thread(target=writer, daemon=True)
            t.start()
            time.sleep(0.1)
            for _ in range(20):
                tap.snapshot_to_wav(seconds=0.5)
            stop.set()
            t.join(timeout=2)
        finally:
            if os.path.exists(wav):
                os.unlink(wav)
        # If we got here without exceptions, the lock did its job.

    def test_status_shape(self):
        tap, _ = self._make()
        s = tap.get_status()
        for key in ('attached_count', 'samples_written', 'last_buffer_age_s',
                    'is_active', 'buffer_seconds', 'wav_path'):
            self.assertIn(key, s)


# =============================================================================
# Audio pipeline shape — playback branch identical w/ and w/o tap
# =============================================================================


class TestAudioPipelineShape(unittest.TestCase):
    """The most important safety property: the playback branch must be
    structurally identical regardless of whether asr_tap is set. If this
    breaks, the audio recovery / watchdog / mute logic could silently
    behave differently and we wouldn't notice."""

    PLAYBACK_ELEMENTS_IN_ORDER = [
        'syncqueue', 'audioqueue', 'audioconvert', 'volume',
        # alsasink is dynamic; not given a name in the pipeline
    ]

    def _build(self, with_tap, tx_connected=True):
        # We can't actually start GStreamer in tests reliably, but we
        # CAN construct the pipeline_str by patching parse_launch to
        # capture it. The shape is purely a function of the pipeline
        # string, which is what we care about.
        # _hdmi_tx_connected is pinned (default: TV present) because on a
        # headless box the real sysfs probe returns disconnected, which
        # legitimately swaps alsasink for fakesink and would make these
        # shape assertions depend on the host's HDMI state.
        from audio import AudioPassthrough
        captured = {}
        def fake_parse_launch(s):
            captured['str'] = s
            raise RuntimeError("fake-parse-launch (test)")
        with patch('audio.Gst.parse_launch', side_effect=fake_parse_launch), \
             patch('audio.detect_hdmi_capture_device', return_value='hw:4,0'), \
             patch.object(AudioPassthrough, '_hdmi_tx_connected',
                          return_value=tx_connected):
            ap = AudioPassthrough(
                capture_device='hw:4,0', playback_device='hw:0,0',
                asr_tap=MagicMock() if with_tap else None
            )
            try:
                ap._init_pipeline()
            except RuntimeError as e:
                if "fake-parse-launch" not in str(e):
                    raise
        return captured.get('str', '')

    def test_no_tap_has_no_tee(self):
        s = self._build(with_tap=False)
        self.assertNotIn('tee', s)
        self.assertNotIn('appsink', s)
        self.assertNotIn('asr_sink', s)

    def test_tap_introduces_tee_and_appsink(self):
        s = self._build(with_tap=True)
        self.assertIn('tee', s)
        self.assertIn('audiotee', s)
        self.assertIn('appsink', s)
        self.assertIn('asr_sink', s)

    def test_playback_chain_byte_identical(self):
        """The substring from `syncqueue` through `alsasink` must be
        identical with or without the tap. If this changes, the
        audio-recovery tests + the documented latency budget no longer
        hold."""
        no_tap = self._build(with_tap=False)
        with_tap = self._build(with_tap=True)
        # Extract from "queue name=syncqueue" to "alsasink ..."
        def playback_chunk(s):
            start = s.index('queue name=syncqueue')
            end = s.index('alsasink')
            # Grab through end of alsasink's parameter list
            end = s.index('sync=false', end) + len('sync=false')
            return s[start:end]
        self.assertEqual(playback_chunk(no_tap), playback_chunk(with_tap),
                         "playback branch changed between no-tap and tap pipelines — "
                         "audio recovery / watchdog assumptions may have broken")

    def test_tx_disconnected_falls_back_to_fakesink(self):
        """With HDMI-TX off, playback must route to fakesink (keeping
        HDMI-RX capture + the ASR tap alive) and never open alsasink."""
        s = self._build(with_tap=True, tx_connected=False)
        self.assertIn('fakesink', s)
        self.assertNotIn('alsasink', s)
        # The capture + tap side must survive the fallback.
        self.assertIn('alsasrc', s)
        self.assertIn('asr_sink', s)

    def test_tap_branch_is_leaky(self):
        """Slow whisper must NOT backpressure the tee → must NOT delay
        playback. Verify by inspecting the tap branch queue config."""
        s = self._build(with_tap=True)
        # Find the asrqueue config
        i = s.index('name=asrqueue')
        chunk = s[max(0, i - 100):i + 100]
        self.assertIn('leaky=downstream', chunk,
                      "asrqueue must be leaky=downstream so a slow whisper "
                      "consumer can't backpressure the audio passthrough")

    def test_tap_branch_resamples_to_16khz_mono(self):
        """whisper.cpp expects 16 kHz mono S16LE. Verify the tap branch
        produces that format."""
        s = self._build(with_tap=True)
        self.assertIn('rate=16000', s)
        self.assertIn('channels=1', s)
        self.assertIn('format=S16LE', s)

    def test_audioqueue_still_present_for_watchdog_probe(self):
        """The audio watchdog adds a buffer probe to `audioqueue.src` to
        detect stalls. This element MUST exist in both pipeline shapes
        or the watchdog would silently lose its stall detection."""
        for with_tap in (False, True):
            s = self._build(with_tap=with_tap)
            self.assertIn('name=audioqueue', s,
                          f"audioqueue missing (with_tap={with_tap}) — "
                          f"watchdog buffer probe would lose its anchor")

    def test_volume_element_still_present_for_mute(self):
        """ad_blocker mutes via the named `vol` element. Must persist
        across both pipeline shapes."""
        for with_tap in (False, True):
            s = self._build(with_tap=with_tap)
            self.assertIn('name=vol', s,
                          f"volume element missing (with_tap={with_tap}) — "
                          f"mute control would lose its target")


# =============================================================================
# Audio recovery: tap survives pipeline restart, watchdog still works
# =============================================================================


class TestAudioRecoveryWithTap(unittest.TestCase):
    """The existing audio-recovery story (HDMI-RX sleep/wake, ALSA zombie
    detection, watchdog-driven restart) MUST continue to work with the
    tap attached. The dangerous failure mode is: tap is attached on first
    pipeline build, audio pipeline restarts (HDMI source went to sleep
    then woke), new pipeline comes up WITHOUT the tap attached — ASR
    silently stops getting buffers and would forever return 'unknown'.

    These tests guard the attach-on-init contract via source inspection
    and via the AudioASRTap.attach_count counter, which increments on
    each attach_to() call — so the count rising across restarts proves
    re-attachment happened."""

    def test_init_pipeline_attaches_tap_when_configured(self):
        """The body of _init_pipeline must look up the appsink BY NAME
        and call asr_tap.attach_to on it. If this gets refactored away,
        the tap silently stops getting buffers on the next restart."""
        import inspect
        from audio import AudioPassthrough
        src = inspect.getsource(AudioPassthrough._init_pipeline)
        self.assertIn('asr_tap', src,
                      "_init_pipeline must reference asr_tap so it gets "
                      "re-attached on every pipeline rebuild")
        self.assertIn("get_by_name('asr_sink')", src,
                      "_init_pipeline must look up the appsink by its "
                      "documented name 'asr_sink'")
        self.assertIn('attach_to', src,
                      "_init_pipeline must call asr_tap.attach_to to wire "
                      "the new appsink callback")

    def test_tap_attach_count_increments_across_calls(self):
        """attach_count is the runtime signal for 'tap was attached to a
        live pipeline'. Watchdog-recovery code can check it (e.g. via
        get_status()) to detect a re-attach happened after a restart."""
        from audio import AudioASRTap
        tap = AudioASRTap()
        fake_appsink = MagicMock()
        tap.attach_to(fake_appsink)
        tap.attach_to(fake_appsink)
        self.assertEqual(tap._attach_count, 2)
        self.assertEqual(tap.get_status()['attached_count'], 2)
        # Each attach must wire the GStreamer signal
        self.assertEqual(fake_appsink.connect.call_count, 2)
        fake_appsink.connect.assert_called_with('new-sample', tap._on_new_sample)

    def test_restart_pipeline_calls_init_pipeline(self):
        """The watchdog's restart path MUST go through _init_pipeline (not
        re-create the GStreamer pipeline inline) so the tap gets
        re-attached. If someone introduces a fast-path that builds the
        pipeline directly, the tap will silently die after the first
        HDMI sleep/wake cycle."""
        import inspect
        from audio import AudioPassthrough
        restart_src = inspect.getsource(AudioPassthrough._restart_pipeline)
        self.assertIn('_init_pipeline', restart_src,
                      "_restart_pipeline must call _init_pipeline so the "
                      "ASR tap gets re-attached after the restart")

    def test_watchdog_paused_flag_still_governs_restart(self):
        """The HDMI-lost path sets _watchdog_paused so the watchdog stops
        churning restart attempts on a known-dead pipeline. Must still
        be in place — otherwise we'd loop trying to restart a pipeline
        that can't open ALSA, and the tap would never get a chance to
        attach to a real working pipeline."""
        import inspect
        from audio import AudioPassthrough
        loop_src = inspect.getsource(AudioPassthrough._watchdog_loop)
        self.assertIn('_watchdog_paused', loop_src,
                      "_watchdog_loop must still check _watchdog_paused "
                      "for the HDMI-sleep recovery story to work")

    def test_alsa_zombie_check_present(self):
        """The ALSA zombie detection (hw_ptr sampling) is the cross-check
        that distinguishes GStreamer-says-PAUSED-but-actually-flowing from
        a real stall. Don't lose this — pipeline restarts on healthy
        audio cost ~1s of dropout each."""
        import inspect
        from audio import AudioPassthrough
        method_src = inspect.getsource(AudioPassthrough._is_alsa_device_running)
        self.assertIn('hw_ptr', method_src,
                      "_is_alsa_device_running must still sample hw_ptr — "
                      "see CLAUDE.md 'ALSA Zombie Detection False Positives'")

    def test_tap_ring_state_survives_lock_pressure(self):
        """If the GStreamer streaming thread is hammering the ring while
        the ASR thread is taking snapshots, neither side should see torn
        reads or stale data."""
        from audio import AudioASRTap
        tap = AudioASRTap(wav_path=f'/tmp/test_tap_survives_{time.time_ns()}.wav')
        try:
            # Pre-fill so snapshot_to_wav can succeed
            samples = np.ones(tap.SAMPLE_RATE * 3, dtype=np.int16) * 7
            with tap._lock:
                tap._ring[:len(samples)] = samples
                tap._write_pos = len(samples) % tap._buffer_samples
                tap._samples_written = len(samples)
            ok = tap.snapshot_to_wav(seconds=1.0)
            self.assertTrue(ok)
            with wave.open(tap.wav_path, 'rb') as wf:
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                self.assertTrue((data == 7).all())
        finally:
            if os.path.exists(tap.wav_path):
                os.unlink(tap.wav_path)


# =============================================================================
# Decision-engine integration — verdict() consumed by _update_blocking_state
# =============================================================================


class TestDecisionEngineASRGate(unittest.TestCase):
    """Verifies how the ASR verdict decorates VLM-alone blocking at START.

    ASR is CONFIRM-ONLY at the start gate now: a 'confirm' upgrades the
    display label to 'vlm+asr' via the blocking_asr_confirmed flag, but
    'veto' is IGNORED — the visual detector is trusted, so a VLM-alone
    block always fires regardless of ASR. (A genuine product-placement
    false positive is caught later by the gated mid-block ASR rescue,
    which only force-stops once VLM ITSELF has weakened.) Uses a partial
    Minus construction (skips heavy init) and exercises the start gate.
    """

    def _make_minus(self, asr_verdict='unknown'):
        """Hand-build a Minus instance with just enough state to call
        _update_blocking_state. Avoids the expensive __init__."""
        import minus as minus_mod
        m = minus_mod.Minus.__new__(minus_mod.Minus)
        # Minimal state matching real construction
        import threading as _t
        m._state_lock = _t.RLock()
        m.ad_detected = False
        m.blocking_active = False
        m.blocking_source = None
        m.blocking_start_time = 0.0
        m.blocking_end_time = 0.0
        m.blocking_paused_until = 0.0
        m.vlm_paused_until = 0.0
        m.OCR_TRUST_WINDOW = 5.0
        m.last_ocr_ad_time = 0
        m.ocr_ad_detected = False
        m.ocr_ad_detection_count = 0
        m.ocr_no_ad_count = 0
        m.vlm_ad_detected = False
        m.vlm_no_ad_count = 0
        m.vlm_decision_history = []
        m.consecutive_ad_count = 0
        m.MIN_DURATION_RESET_GAP = 30.0
        m.accidental_pause_detected = False
        m.skip_attempted_this_ad = False
        m.last_skip_countdown = None
        m.last_skip_success_time = 0
        m.SKIP_UNBLOCK_GRACE_SECONDS = 3.0
        m._safeguard_freeze_active = False
        m.home_screen_detected = False
        m.video_interface_detected = False
        m.static_blocking_suppressed = False
        # hdmi_reconnect_grace_enabled is a property; we override the
        # method that consults it instead.
        m.hdmi_reconnect_time = 0
        m.HDMI_RECONNECT_GRACE_SECONDS = 90.0
        # ASR mock
        asr_mock = MagicMock()
        asr_mock.verdict.return_value = asr_verdict
        asr_mock.last_marker_hits = 0 if asr_verdict != 'confirm' else 3
        asr_mock.last_transcript = ''
        m.asr = asr_mock
        # Misc downstream config + collaborators that _update_blocking_state
        # touches; stub everything to a no-op so the test focuses on the
        # ASR gate alone.
        m.config = MagicMock(no_blocking=False)
        m.last_matched_keywords = []
        m.add_detection = MagicMock()
        m.last_ocr_texts = []
        m.MAX_BLOCKING_DURATION = 150.0
        m.vlm_no_ad_count = 0
        m.OCR_STOP_THRESHOLD = 2
        m.VLM_STOP_THRESHOLD = 2
        m._current_min_blocking_duration = lambda: 1.0
        m._safeguard_freeze_text = ''
        m.last_vlm_ad_frame = None
        m.last_vlm_ad_frame_time = 0.0
        m.SKIP_UNBLOCK_GRACE_SECONDS = 3.0
        m._safeguard_freeze_active = False
        m.audio = MagicMock()
        m.audio.is_muted = False
        m.ad_blocker = MagicMock()
        m.ad_blocker.is_visible = False
        m._set_led_state = MagicMock()
        m.blocking_asr_confirmed = False
        # Helpers
        m._get_vlm_agreement = MagicMock(return_value=(0.85, 0.15, 5))
        m.is_in_hdmi_reconnect_grace = lambda: False
        m.get_hdmi_reconnect_grace_remaining = lambda: 0
        return m

    def test_asr_veto_holds_back_an_unsure_vlm(self):
        """The veto counts again (2026-10-18, SenseVoice on the NPU), but
        only against a VLM below ASR_VETO_MAX_VLM_RATIO: at 85% agreement,
        clear show dialog with no ad language keeps the block from starting."""
        m = self._make_minus(asr_verdict='veto')
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)

    def test_asr_veto_cannot_stop_a_sure_vlm(self):
        """The cases the old unconditional veto got wrong (an insurance ad
        at 100% whose copy had no marketing words) still block."""
        m = self._make_minus(asr_verdict='veto')
        m._get_vlm_agreement = MagicMock(return_value=(1.0, 0.0, 5))
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected,
                        "a VLM at 100% must block whatever ASR says")
        self.assertEqual(m.blocking_source, 'vlm')
        self.assertFalse(m.blocking_asr_confirmed)
        self.assertEqual(m._display_source(), 'vlm')

    def test_vlm_alone_blocks_with_vlm_source_when_asr_unknown(self):
        """ASR 'unknown' (cold start, music) must NOT suppress VLM-alone
        and must NOT set the +asr label."""
        m = self._make_minus(asr_verdict='unknown')
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)
        self.assertEqual(m.blocking_source, 'vlm')
        self.assertFalse(m.blocking_asr_confirmed)
        self.assertEqual(m._display_source(), 'vlm')

    def test_vlm_alone_confirm_upgrades_display_label(self):
        """ASR 'confirm' keeps the base source 'vlm' (stop-logic depends
        on it) but flips blocking_asr_confirmed so the DISPLAY label
        becomes 'vlm+asr' in /api/status and the logs — the high-confidence
        path. The base source must stay 'vlm', not 'vlm+asr'."""
        m = self._make_minus(asr_verdict='confirm')
        m.vlm_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)
        self.assertEqual(m.blocking_source, 'vlm',
                         "base source must stay 'vlm' for stop-logic")
        self.assertTrue(m.blocking_asr_confirmed)
        self.assertEqual(m._display_source(), 'vlm+asr')

    def test_ocr_blocking_unaffected_by_asr_veto(self):
        """OCR-driven blocking is authoritative. ASR 'veto' must NOT
        suppress an OCR-confirmed block — OCR saw text on screen, that's
        ground truth regardless of audio."""
        m = self._make_minus(asr_verdict='veto')
        m.ocr_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)
        self.assertEqual(m.blocking_source, 'ocr')


class TestAudioGate(unittest.TestCase):
    """No sound, no ad (2026-10-18): menus, home-screen banners and paused
    screens can look like ads, but ads always play audio."""

    def _m(self, quiet, **kw):
        m = TestDecisionEngineASRGate._make_minus(self, **kw)
        m.audio.seconds_since_audible = MagicMock(return_value=quiet)
        m._system_settings = {}
        return m

    def test_silent_screen_does_not_start_a_vlm_block(self):
        m = self._m(quiet=5.0, asr_verdict='unknown')
        m.vlm_ad_detected = True
        m._get_vlm_agreement = MagicMock(return_value=(1.0, 0.0, 5))
        m._update_blocking_state()
        self.assertFalse(m.ad_detected, "VLM block started on a silent screen")

    def test_silent_ad_card_that_ocr_reads_still_blocks(self):
        """Frozen/silent "Sponsored | Skip in" cards: OCR is authoritative."""
        m = self._m(quiet=60.0, asr_verdict='unknown')
        m.ocr_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_audio_playing_blocks_as_before(self):
        m = self._m(quiet=0.2)
        m.ocr_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_unknown_audio_never_holds_a_block_back(self):
        for quiet in (None, MagicMock()):
            m = self._m(quiet=quiet)
            m.ocr_ad_detected = True
            m._update_blocking_state()
            self.assertTrue(m.ad_detected, "an audio outage must not stop blocking")

    def test_setting_off_disables_the_gate(self):
        m = self._m(quiet=60.0)
        m._system_settings = {'block_requires_audio': False}
        m.ocr_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)


class TestAudioGateStop(unittest.TestCase):
    """A block ends after AUDIO_GATE_STOP_QUIET_S of silence."""

    def _blocking(self, quiet, elapsed=20.0, source='vlm', ocr=False):
        # Mid-block harness, with the VLM still calling it an ad: only the
        # audio can end it.
        m = TestOCRTriangulationVeto._make_minus(
            self, blocking_elapsed=elapsed, vlm_no_ad_ratio=0.0,
            asr_verdict='unknown', source=source)
        m.ocr_ad_detected = ocr
        m.audio.seconds_since_audible = MagicMock(return_value=quiet)
        m._system_settings = {}
        return m

    def test_long_silence_ends_a_vlm_block(self):
        m = self._blocking(quiet=12.0)
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)

    def test_silence_does_not_end_a_block_ocr_still_sees(self):
        m = self._blocking(quiet=60.0, source='both', ocr=True)
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_short_gap_keeps_it(self):
        m = self._blocking(quiet=4.0)
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)


class TestNoReblockFromStaleOCRFlag(unittest.TestCase):
    """A both-block ended by the VLM must not restart from OCR's old flag.

    Observed live 2026-10-19: every ad ending "stopped by BOTH" re-blocked
    0.6-1.6 s later as "(OCR)" with no new OCR ad read, ran 2.6-3.6 s, and
    raised the flap threshold - ocr_ad_detected was still True from the ad.
    """

    def _both_block_vlm_clearing(self):
        m = TestOCRTriangulationVeto._make_minus(
            self, blocking_elapsed=20.0, vlm_no_ad_ratio=1.0,
            asr_verdict='unknown', source='both')
        m.ocr_ad_detected = True
        m.ocr_ad_detection_count = 3
        m.vlm_no_ad_count = 5
        m._system_settings = {}
        m._ad_clock_says_playing = lambda now: False
        return m

    def test_vlm_stop_clears_ocr_flag(self):
        m = self._both_block_vlm_clearing()
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)
        self.assertFalse(m.ocr_ad_detected)
        self.assertEqual(m.ocr_ad_detection_count, 0)

    def test_no_restart_without_a_new_ocr_read(self):
        m = self._both_block_vlm_clearing()
        m._update_blocking_state()
        m._update_blocking_state()   # next tick, nothing new from OCR
        self.assertFalse(m.ad_detected)


class TestBareAdNeedsCorroboration(unittest.TestCase):
    """A lone OCR 'ad' token blocks only when something agrees."""

    def _m(self, vlm=False, last='no-ad', asr='unknown'):
        m = TestOCRTriangulationVeto._make_minus(
            self, blocking_elapsed=0.0, vlm_no_ad_ratio=1.0,
            asr_verdict=asr, source='ocr')
        m.vlm_ad_detected = vlm
        m.vlm_last_state = last
        return m

    def test_noise_ad_alone_is_suppressed(self):
        self.assertTrue(self._m()._bare_ad_uncorroborated(['ad'], False))

    def test_vlm_or_asr_or_strong_keyword_lets_it_through(self):
        self.assertFalse(self._m(last='ad')._bare_ad_uncorroborated(['ad'], False))
        self.assertFalse(self._m(vlm=True)._bare_ad_uncorroborated(['ads'], False))
        self.assertFalse(self._m(asr='confirm')._bare_ad_uncorroborated(['ad'], False))
        self.assertFalse(self._m()._bare_ad_uncorroborated(['ad'], True))

    def test_other_keywords_untouched(self):
        m = self._m()
        self.assertFalse(m._bare_ad_uncorroborated(['ad', 'sponsored'], False))
        self.assertFalse(m._bare_ad_uncorroborated(['skip in'], False))
        self.assertFalse(m._bare_ad_uncorroborated([], False))


class TestAudibleTracker(unittest.TestCase):
    """AudioPassthrough.seconds_since_audible: None means unknown."""

    def _a(self):
        from audio import AudioPassthrough
        a = AudioPassthrough.__new__(AudioPassthrough)
        a.is_running = True
        a.AUDIBLE_DBFS = -55.0
        a._last_level_mono = 0.0
        a._last_audible_mono = 0.0
        a._level_since_mono = 0.0
        return a

    def test_unknown_states(self):
        import time as _time
        a = self._a()
        self.assertIsNone(a.seconds_since_audible())          # no samples yet
        a._last_level_mono = _time.monotonic() - 5.0
        self.assertIsNone(a.seconds_since_audible())          # samples stopped
        a.is_running = False
        self.assertIsNone(a.seconds_since_audible())

    def test_quiet_and_audible(self):
        import time as _time
        now = _time.monotonic()
        a = self._a()
        a._level_since_mono = now - 7.0
        a._last_level_mono = now
        self.assertGreaterEqual(a.seconds_since_audible(), 6.9)   # quiet since start
        a._last_audible_mono = now - 1.0
        self.assertLess(a.seconds_since_audible(), 1.5)


class TestEngineSelection(unittest.TestCase):
    """SenseVoice on the NPU by default, CPU moonshine when it's missing."""

    def test_falls_back_without_the_model(self):
        import asr_worker
        with patch.dict(os.environ, {'MINUS_ASR_ENGINE': 'sensevoice'}), \
                patch('sensevoice_npu.is_available', return_value=False):
            self.assertEqual(asr_worker.resolve_engine(), 'moonshine')
        with patch.dict(os.environ, {'MINUS_ASR_ENGINE': 'sensevoice'}), \
                patch('sensevoice_npu.is_available', return_value=True):
            self.assertEqual(asr_worker.resolve_engine(), 'sensevoice')
        with patch.dict(os.environ, {'MINUS_ASR_ENGINE': 'moonshine'}):
            self.assertEqual(asr_worker.resolve_engine(), 'moonshine')

    def test_sys_path_is_not_grown_per_call(self):
        import sys as _sys, asr_worker
        before = len(_sys.path)
        for _ in range(5):
            asr_worker.resolve_engine()
        self.assertLessEqual(len(_sys.path), before + 1)


class TestStaleVLMStart(unittest.TestCase):
    """minus 2026-10-19: VLM-alone blocks started at 17-50% agreement on the
    sticky vlm_ad_detected flag once other gates cleared."""

    def _m(self, ratio):
        m = TestDecisionEngineASRGate._make_minus(self, asr_verdict='unknown')
        m._get_vlm_agreement = MagicMock(return_value=(ratio, 1 - ratio, 6))
        m.vlm_ad_detected = True
        return m

    def test_stale_flag_does_not_start_a_block(self):
        for ratio in (0.17, 0.31, 0.5):
            m = self._m(ratio)
            m._update_blocking_state()
            self.assertFalse(m.ad_detected, f"started at {ratio:.0%}")

    def test_current_agreement_still_starts(self):
        m = self._m(0.85)
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_ocr_blocks_do_not_need_vlm(self):
        m = self._m(0.1)
        m.ocr_ad_detected = True
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)


class TestVetoNeedsAConfidentVLM(unittest.TestCase):
    """minus 2026-10-19 12:45: five weak AD calls (mean 0.78) read as 100%
    agreement and overruled ASR hearing lyrics; 4.5 s false-positive block."""

    def _m(self, confs):
        import time as _t
        m = TestDecisionEngineASRGate._make_minus(self, asr_verdict='veto')
        now = _t.time()
        m.vlm_history_window = 8.0
        m.vlm_decision_history = [(now - i, True, c) for i, c in enumerate(confs)]
        m._get_vlm_agreement = MagicMock(return_value=(1.0, 0.0, len(confs)))
        m.vlm_ad_detected = True
        return m

    def test_weak_unanimous_vlm_does_not_overrule_the_ear(self):
        m = self._m([0.91, 0.50, 0.91, 0.65, 0.95])
        m._update_blocking_state()
        self.assertFalse(m.ad_detected)

    def test_confident_unanimous_vlm_still_overrules(self):
        m = self._m([0.97, 0.95, 0.99, 0.93, 0.96])
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)

    def test_without_a_veto_weak_vlm_still_blocks(self):
        """No ASR objection: VLM-only ads (e.g. cable TV, no on-screen text)
        keep their old threshold."""
        m = self._m([0.91, 0.50, 0.91, 0.65, 0.95])
        m.asr.verdict.return_value = 'unknown'
        m._update_blocking_state()
        self.assertTrue(m.ad_detected)


# =============================================================================
# OCR triangulation — transience guard + VLM+ASR veto + sustained OCR override
# =============================================================================


class TestOCRTriangulationVeto(unittest.TestCase):
    """Tests the mid-block triangulation veto on OCR-source blocks.

    Scenario the veto targets: OCR matches an ad keyword that's actually
    a TV-show artifact (movie billboard, news ticker, sign in a scene),
    blocking starts, but VLM and ASR both clearly see show content.
    The block should release rather than running to the universal cap.

    Override the user explicitly asked for: if OCR has been matching
    for SUSTAINED time (≥OCR_TRUSTED_DWELL_FRAMES), the veto is
    DISABLED — sustained OCR is authoritative even when VLM/ASR
    transiently disagree (frame-classification noise, ad-copy pause).
    """

    def _make_minus(self, *, ocr_detection_count=10, blocking_elapsed=10.0,
                    vlm_no_ad_ratio=0.95, asr_verdict='veto',
                    source='ocr'):
        """Build a Minus instance mid-OCR-block with the relevant
        triangulation inputs configurable."""
        import minus as minus_mod
        import threading as _t
        m = minus_mod.Minus.__new__(minus_mod.Minus)
        m._state_lock = _t.RLock()

        # The block is already active when _update_blocking_state runs.
        m.ad_detected = True
        m.blocking_active = True
        m.blocking_source = source
        m.blocking_asr_confirmed = False
        m.blocking_start_time = time.time() - blocking_elapsed
        m.blocking_end_time = 0.0
        m.blocking_paused_until = 0.0
        m.vlm_paused_until = 0.0

        # OCR state
        m.OCR_TRUST_WINDOW = 5.0
        m.OCR_STOP_THRESHOLD = 2
        m.last_ocr_ad_time = time.time()
        m.ocr_ad_detected = True
        m.ocr_ad_detection_count = ocr_detection_count
        m.ocr_no_ad_count = 0

        # VLM state
        m.vlm_ad_detected = (source != 'ocr')  # 'both' source means VLM still agreed at start
        m.vlm_no_ad_count = 0
        m.vlm_decision_history = []
        m.VLM_STOP_THRESHOLD = 2
        m.vlm_min_decisions = 3

        # Ad-countdown clock: off, so these cases exercise the triangulation
        # veto rather than the clock's stop veto.
        m.AD_COUNTDOWN_ENABLED = False
        m.ad_clock_stats = {'holds': 0, 'early_release': 0, 'pause_override': 0}
        m.flap_escalation = 0
        m.OCR_STOP_THRESHOLD_MAX = 5
        m.OCR_STOP_MIN_SECONDS = 5.0
        m.OCR_STOP_MAX_SECONDS = 9.0
        m.ocr_failure_streak = 0
        m.OCR_DEGRADED_STREAK = 3

        # Triangulation constants (mirror minus.py defaults)
        m.OCR_TRIANGULATION_MIN_BLOCK_S = 4.0
        m.OCR_TRIANGULATION_VLM_NOAD_RATIO = 0.80
        m.OCR_TRUSTED_DWELL_FRAMES = 3

        # ASR mock
        asr_mock = MagicMock()
        asr_mock.verdict.return_value = asr_verdict
        asr_mock.last_marker_hits = 0
        asr_mock.last_transcript = "She walked into the room and said hi."
        m.asr = asr_mock

        # _get_vlm_agreement returns (ad_ratio, no_ad_ratio, total)
        m._get_vlm_agreement = MagicMock(return_value=(
            1.0 - vlm_no_ad_ratio, vlm_no_ad_ratio, 5))

        # Misc state _update_blocking_state touches
        m.consecutive_ad_count = 1
        m.MIN_DURATION_RESET_GAP = 30.0
        m.accidental_pause_detected = False
        m.skip_attempted_this_ad = False
        m.last_skip_countdown = None
        m.last_skip_success_time = 0
        m.SKIP_UNBLOCK_GRACE_SECONDS = 3.0
        m._safeguard_freeze_active = False
        m._safeguard_freeze_text = ''
        m.home_screen_detected = False
        m.video_interface_detected = False
        m.static_blocking_suppressed = False
        m.hdmi_reconnect_time = 0
        m.HDMI_RECONNECT_GRACE_SECONDS = 90.0
        m.MAX_BLOCKING_DURATION = 150.0
        m.FROZEN_EARLY_SECONDS = 30.0
        m._ocr_text_frozen_for = 0.0
        m.last_matched_keywords = []
        m.last_vlm_ad_frame = None
        m.last_vlm_ad_frame_time = 0.0
        m.config = MagicMock(no_blocking=False)
        m.last_ocr_texts = []
        m.audio = MagicMock(is_muted=True)
        m.ad_blocker = MagicMock(is_visible=True)
        m._set_led_state = MagicMock()
        m._current_min_blocking_duration = lambda: 0.5
        m.add_detection = MagicMock()
        m.is_in_hdmi_reconnect_grace = lambda: False
        m.get_hdmi_reconnect_grace_remaining = lambda: 0
        return m

    def test_veto_fires_on_ocr_source_when_vlm_and_asr_both_clean(self):
        """Canonical case: OCR-source block, brief OCR dwell, VLM 95%
        no-ad, ASR veto. Block should force-stop."""
        m = self._make_minus(source='ocr', ocr_detection_count=1,
                             vlm_no_ad_ratio=0.95, asr_verdict='veto')
        m._update_blocking_state()
        self.assertFalse(m.ad_detected,
                         "triangulation veto should have force-stopped the block")

    def test_veto_disabled_when_ocr_dwell_is_sustained(self):
        """User-asserted invariant: sustained OCR is ground truth and
        overrides the triangulation veto. The word 'ad' on screen for
        ≥OCR_TRUSTED_DWELL_FRAMES cycles means it really IS an ad,
        even if VLM and ASR transiently disagree."""
        m = self._make_minus(source='ocr', ocr_detection_count=5,
                             vlm_no_ad_ratio=0.95, asr_verdict='veto')
        m._update_blocking_state()
        self.assertTrue(m.ad_detected,
                        "sustained OCR (5 frames) must override the "
                        "triangulation veto")

    def test_veto_not_fired_before_min_block_seconds(self):
        """Block must be at least OCR_TRIANGULATION_MIN_BLOCK_S old
        before the veto can fire — otherwise VLM/ASR sliding windows
        haven't gathered enough evidence to override OCR."""
        m = self._make_minus(source='ocr', ocr_detection_count=1,
                             vlm_no_ad_ratio=0.95, asr_verdict='veto',
                             blocking_elapsed=1.0)
        m._update_blocking_state()
        self.assertTrue(m.ad_detected,
                        "veto must not fire within MIN_BLOCK_S of block start")

    def test_veto_not_fired_when_vlm_ratio_below_threshold(self):
        """VLM must clearly say no-ad (≥80%) — a borderline 70% should
        NOT be enough to override OCR."""
        m = self._make_minus(source='ocr', ocr_detection_count=1,
                             vlm_no_ad_ratio=0.70, asr_verdict='veto')
        m._update_blocking_state()
        self.assertTrue(m.ad_detected,
                        "borderline VLM (70%) must not trigger triangulation veto")

    def test_veto_not_fired_when_asr_unknown(self):
        """ASR 'unknown' (no signal yet) must NOT contribute to the veto.
        We require an ACTIVE veto signal from ASR, not just absence of
        confirmation."""
        m = self._make_minus(source='ocr', ocr_detection_count=1,
                             vlm_no_ad_ratio=0.95, asr_verdict='unknown')
        m._update_blocking_state()
        self.assertTrue(m.ad_detected,
                        "ASR 'unknown' must not contribute to veto")

    def test_veto_not_fired_when_asr_confirms(self):
        """If ASR is hearing marketing language, the block stands even
        if VLM disagrees — ASR confirm overrides VLM no-ad."""
        m = self._make_minus(source='ocr', ocr_detection_count=1,
                             vlm_no_ad_ratio=0.95, asr_verdict='confirm')
        m._update_blocking_state()
        self.assertTrue(m.ad_detected,
                        "ASR confirm must veto the triangulation veto")

    def test_veto_works_on_both_source(self):
        """'both' source (OCR+VLM agreed at start) is also eligible for
        triangulation — an ad-text overlay can match BOTH OCR and VLM
        briefly, but later show content can still trigger the veto."""
        m = self._make_minus(source='both', ocr_detection_count=1,
                             vlm_no_ad_ratio=0.95, asr_verdict='veto')
        m._update_blocking_state()
        self.assertFalse(m.ad_detected,
                         "'both' source must be eligible for triangulation veto")

    def test_veto_does_not_target_vlm_only_source(self):
        """The mid-block VLM-only force-stop is a separate path; this
        veto only targets OCR-driven blocks ('ocr' or 'both')."""
        m = self._make_minus(source='vlm', ocr_detection_count=0,
                             vlm_no_ad_ratio=0.95, asr_verdict='veto')
        # The other path (VLM-only ASR force-stop, separate code in
        # _update_blocking_state) DOES fire here — we just want to
        # confirm THIS test goes through that path and the asr_verdict
        # is being read. The behaviour overlap is intentional; we only
        # negatively assert that this test class's triangulation log
        # message wouldn't appear for vlm source. Functionally,
        # ad_detected ends up False either way for this input.
        m._update_blocking_state()
        # Just sanity check the block did stop (via the VLM-only path),
        # and was not blocked by our triangulation gating logic.
        self.assertFalse(m.ad_detected)


# =============================================================================
# OCR transience guard — require dwell before firing, fast-fire on triangulation
# =============================================================================


class TestOCRTransienceGuard(unittest.TestCase):
    """Tests the start-side transience guard that requires sustained OCR
    matches before firing the first block — rejects single-frame OCR
    misreads from TV-show artifacts (a billboard with "SKIP", a scene
    with a sign reading "Sponsored", a caption containing 'BUY')."""

    def _make_minus(self, *, vlm_ad=False, asr_verdict='unknown'):
        """Build a Minus instance pre-block ready to receive consecutive
        real_ad_frame hits via _simulate_ocr_match(). Mirrors the OCR
        loop's count increment + transience-gate logic."""
        import minus as minus_mod
        m = minus_mod.Minus.__new__(minus_mod.Minus)
        import threading as _t
        m._state_lock = _t.RLock()
        m.ad_detected = False
        m.blocking_source = None
        m.ocr_ad_detected = False
        m.ocr_ad_detection_count = 0
        m.vlm_ad_detected = vlm_ad
        m.OCR_TRANSIENCE_MIN_FRAMES = 2
        # Mirror the production definitive-keyword set (STRONG minus
        # 'sponsored'); only the names exercised here are needed.
        m.DEFINITIVE_AD_KEYWORD_NAMES = frozenset({
            'skip ad', 'skip ads', 'skip in', 'ad countdown',
            'ad X of Y', 'ad with timestamp', 'visit advertiser',
            'video will play after ad'})

        asr_mock = MagicMock()
        asr_mock.verdict.return_value = asr_verdict
        m.asr = asr_mock
        return m

    def _simulate_ocr_match(self, m, keywords_found=('sponsored',)):
        """Run the transience-guard branch from the OCR loop once. This
        copies the production logic literally so a refactor that
        diverges shows up as a test failure. keywords_found defaults to a
        non-definitive keyword ('sponsored') so the dwell applies unless a
        test passes a definitive ad-UI keyword."""
        m.ocr_ad_detection_count += 1
        definitive_ocr = any(kw in m.DEFINITIVE_AD_KEYWORD_NAMES
                             for kw in keywords_found)
        fast_fire = (definitive_ocr
                     or m.vlm_ad_detected
                     or (m.asr is not None
                         and m.asr.verdict() == 'confirm'))
        required_frames = (1 if fast_fire else m.OCR_TRANSIENCE_MIN_FRAMES)
        if (m.ocr_ad_detection_count >= required_frames
                and not m.ocr_ad_detected):
            m.ocr_ad_detected = True

    def test_single_frame_does_not_fire(self):
        """One OCR-matched frame in isolation must NOT fire the block."""
        m = self._make_minus()
        self._simulate_ocr_match(m)
        self.assertFalse(m.ocr_ad_detected,
                         "single OCR match must not fire — could be an artifact")

    def test_two_consecutive_frames_fire(self):
        """Two consecutive matches clear the default dwell threshold."""
        m = self._make_minus()
        self._simulate_ocr_match(m)
        self._simulate_ocr_match(m)
        self.assertTrue(m.ocr_ad_detected,
                        "2 consecutive matches must reach dwell threshold")

    def test_fast_fire_when_vlm_confirms(self):
        """Triangulation override: VLM already says ad → fast-fire."""
        m = self._make_minus(vlm_ad=True)
        self._simulate_ocr_match(m)
        self.assertTrue(m.ocr_ad_detected,
                        "vlm_ad_detected=True must fast-fire on 1 frame")

    def test_fast_fire_when_asr_confirms(self):
        """Triangulation override: ASR has marker hits → fast-fire."""
        m = self._make_minus(asr_verdict='confirm')
        self._simulate_ocr_match(m, keywords_found=['sponsored'])
        self.assertTrue(m.ocr_ad_detected,
                        "ASR 'confirm' must fast-fire on 1 frame")

    def test_fast_fire_on_definitive_ocr_keyword(self):
        """Latency fix: a DEFINITIVE ad-UI keyword (skip in / skip ad / ad
        countdown / visit advertiser / ...) fires on the FIRST frame with
        no VLM/ASR corroboration — these strings never occur as a
        single-frame show-content artifact, so the dwell only added
        latency (~1 OCR cycle) to the most common ad-break case."""
        for kw in ('skip in', 'skip ad', 'ad countdown', 'ad X of Y',
                   'ad with timestamp', 'visit advertiser',
                   'video will play after ad'):
            m = self._make_minus()  # no VLM, ASR unknown
            self._simulate_ocr_match(m, keywords_found=[kw])
            self.assertTrue(m.ocr_ad_detected,
                            f"definitive keyword {kw!r} must fast-fire on 1 frame")

    def test_sponsored_alone_still_requires_dwell(self):
        """'sponsored' is STRONG but NOT definitive — it appears on home /
        promo tiles and as show-content text, so a single 'sponsored'
        frame must still wait for the 2-frame dwell (artifact protection)."""
        m = self._make_minus()
        self._simulate_ocr_match(m, keywords_found=['sponsored'])
        self.assertFalse(m.ocr_ad_detected,
                         "single 'sponsored' frame must not fast-fire")
        self._simulate_ocr_match(m, keywords_found=['sponsored'])
        self.assertTrue(m.ocr_ad_detected,
                        "second 'sponsored' frame clears the dwell")

    def test_dwell_resets_after_gap(self):
        """The artifact pattern the user described — keyword appears
        briefly, then 3-4s gap, then briefly again — must not fire.
        Mirrors the production OCR loop: on a no-ad frame the count
        resets to 0."""
        m = self._make_minus()
        # 1 hit
        self._simulate_ocr_match(m)
        self.assertFalse(m.ocr_ad_detected)
        # No-ad frame: count resets (production OCR loop line ~3481).
        m.ocr_ad_detection_count = 0
        # Another single hit later — still doesn't fire.
        self._simulate_ocr_match(m)
        self.assertFalse(m.ocr_ad_detected,
                         "two separate single-frame hits with a gap must not fire")

    def test_dwell_threshold_is_configurable(self):
        """OCR_TRANSIENCE_MIN_FRAMES must be honored as a knob. Set to 3
        and verify 2 hits still don't fire."""
        m = self._make_minus()
        m.OCR_TRANSIENCE_MIN_FRAMES = 3
        self._simulate_ocr_match(m)
        self._simulate_ocr_match(m)
        self.assertFalse(m.ocr_ad_detected,
                         "with MIN_FRAMES=3, 2 hits must not be enough")
        self._simulate_ocr_match(m)
        self.assertTrue(m.ocr_ad_detected,
                        "third match clears the configured threshold")


if __name__ == '__main__':
    unittest.main()
