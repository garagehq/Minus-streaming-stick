#!/usr/bin/env python3
"""Unit tests for tools/label_clips.py (clip -> caption labels)."""
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import label_clips as lc  # noqa: E402

SR = 16000


def sync(t, pos, vid='vid1'):
    # Device clock == box clock here; 'updated' is the device time of `pos`.
    return (t, vid, {'board_mid': t, 'device_uptime': t, 'position': pos,
                     'updated': t, 'speed': 1.0})


def write_json3(path, words):
    """words: [(text, start_s)] -> one json3 event per word."""
    events = [{'tStartMs': int(s * 1000), 'dDurationMs': 2000,
               'segs': [{'utf8': w}]} for w, s in words]
    path.write_text(json.dumps({'events': events}))


class TestPositionSpan(unittest.TestCase):
    def test_steady_stretch_maps_linearly(self):
        syncs = [sync(100, 50), sync(105, 55), sync(110, 60), sync(115, 65)]
        vid, v0, v1 = lc.position_span(syncs, 101.0, 111.0)
        self.assertEqual(vid, 'vid1')
        self.assertAlmostEqual(v0, 51.0)
        self.assertAlmostEqual(v1, 61.0)

    def test_ad_inside_clip_rejected(self):
        # Video clock stops for an ad between t=105 and t=110.
        syncs = [sync(100, 50), sync(105, 55), sync(110, 55.2), sync(115, 60.2)]
        self.assertIsNone(lc.position_span(syncs, 101.0, 111.0))

    def test_needs_syncs_on_both_sides(self):
        syncs = [sync(100, 50), sync(105, 55)]
        self.assertIsNone(lc.position_span(syncs, 101.0, 111.0))
        self.assertIsNone(lc.position_span(syncs, 90.0, 101.0))

    def test_video_change_rejected(self):
        syncs = [sync(100, 50), sync(105, 55), sync(110, 3, 'vid2'), sync(115, 8, 'vid2')]
        self.assertIsNone(lc.position_span(syncs, 101.0, 111.0))

    def test_sync_gap_rejected(self):
        syncs = [sync(100, 50), sync(140, 90)]
        self.assertIsNone(lc.position_span(syncs, 101.0, 111.0))


class TestCaptionWords(unittest.TestCase):
    def setUp(self):
        lc._caption_cache.clear()
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / 'captions').mkdir()

    def test_json3_word_times_and_tags_dropped(self):
        write_json3(self.dir / 'captions' / 'a.json3',
                    [('and', 1.0), ('lebron', 1.3), ('james', 1.7), ('[Music]', 2.5)])
        w = lc.caption_words(self.dir, 'a')
        self.assertEqual([x[0] for x in w], ['and', 'lebron', 'james'])
        self.assertAlmostEqual(w[1][1], 1.3)
        self.assertAlmostEqual(w[1][2], 1.7)    # ends where the next word starts

    def test_json3_line_segment_words_spread(self):
        # Uploaded captions: one segment per line, no word offsets.
        (self.dir / 'captions' / 'c.json3').write_text(json.dumps({'events': [
            {'tStartMs': 10000, 'dDurationMs': 4000, 'segs': [{'utf8': 'here comes\nLeBron James'}]},
            {'tStartMs': 14000, 'dDurationMs': 2000, 'segs': [{'utf8': '>> again'}]}]}))
        w = lc.caption_words(self.dir, 'c')
        self.assertEqual([x[0] for x in w], ['here', 'comes', 'LeBron', 'James', 'again'])
        self.assertAlmostEqual(w[0][1], 10.0)
        self.assertAlmostEqual(w[2][1], 12.0)
        self.assertAlmostEqual(w[3][2], 14.0)
        self.assertAlmostEqual(w[4][1], 14.0)   # a speaker mark takes no time

    def test_multi_word_tags_dropped_and_censoring_marked(self):
        w = lc.drop_tags([(t, k, k + 1) for k, t in enumerate(
            ['so', '[clears', 'throat]', 'lebron', '[', '__', ']', 'yeah', '>>', '[Music]'])])
        self.assertEqual([x[0] for x in w], ['so', 'lebron', lc.CENSORED, 'yeah'])
        self.assertEqual(w[2][1:], (4, 7))

    def test_stray_bracket_keeps_words(self):
        words = ['[oh'] + ['word'] * 8
        w = lc.drop_tags([(t, k, k + 1) for k, t in enumerate(words)])
        self.assertEqual(len(w), 9)

    def test_vtt_cue_words_spread(self):
        (self.dir / 'captions' / 'b.vtt').write_text(
            "WEBVTT\n\n00:00:10.000 --> 00:00:12.000\nhere comes LeBron James\n\n")
        w = lc.caption_words(self.dir, 'b')
        self.assertEqual([x[0] for x in w], ['here', 'comes', 'LeBron', 'James'])
        self.assertAlmostEqual(w[0][1], 10.0)
        self.assertAlmostEqual(w[-1][2], 12.0)

    def test_missing_captions(self):
        self.assertEqual(lc.caption_words(self.dir, 'nope'), [])


class TestLabelAll(unittest.TestCase):
    def setUp(self):
        lc._caption_cache.clear()
        root = Path(tempfile.mkdtemp())
        self.clips, self.farm, self.ds = root / 'clips', root / 'farm', root / 'ds'
        for d in (self.clips, self.farm / 'captions', self.farm / 'playlog'):
            d.mkdir(parents=True)
        # Video vid1 plays from t=1000 (position 0) for 60 s, synced every 5 s.
        with open(self.farm / 'playlog' / 'x.jsonl', 'w') as f:
            for k in range(13):
                f.write(json.dumps({'type': 'sync', 'video': 'vid1', 'board_mid': 1000 + 5 * k,
                                    'device_uptime': 1000 + 5 * k, 'position': 5.0 * k,
                                    'updated': 1000 + 5 * k, 'speed': 1.0}) + '\n')
        words = [(w, 20.0 + 0.4 * i) for i, w in
                 enumerate('and a strong move by lebron james to the rim'.split())]
        write_json3(self.farm / 'captions' / 'vid1.json3', words)

    HEARD = 'and a strong move by lebron james to the rim'

    def clip(self, name, start_mono, kind='random', seconds=10.0, heard=HEARD):
        n = int(seconds * SR)
        ramp = (np.arange(n) % 20000).astype(np.int16)
        with wave.open(str(self.clips / f'{name}.wav'), 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(ramp.tobytes())
        (self.clips / f'{name}.json').write_text(json.dumps(
            {'kind': kind, 'duration_s': seconds, 'clip_start_mono': start_mono,
             'asr_windows': [{'t': 0.0, 'text': heard}] if heard else []}))

    def run_label(self):
        return lc.label_all(self.clips, self.farm, self.ds, log=lambda *_: None)

    def test_clip_labelled_and_trimmed(self):
        self.clip('c1', 1018.0)            # video 18..28 s, words at 20..23.6 s
        st = self.run_label()
        self.assertEqual(st['labeled'], 1)
        row = json.loads((self.ds / 'manifest.jsonl').read_text())
        self.assertEqual(row['text'], 'and a strong move by lebron james to the rim')
        self.assertTrue(row['has_name'])
        self.assertEqual(row['video_id'], 'vid1')
        self.assertAlmostEqual(row['video_start'], 20.0 - lc.LEAD_S, places=2)
        with wave.open(row['audio_filepath']) as w:
            dur = w.getnframes() / SR
            first = np.frombuffer(w.readframes(1), np.int16)[0]
        self.assertAlmostEqual(dur, row['duration'], places=2)
        # Trim starts LEAD_S before the first word: sample index (2.0-0.15)*SR.
        self.assertEqual(first, int((2.0 - lc.LEAD_S) * SR) % 20000)

    def test_clip_with_censored_word_rejected(self):
        words = [(w, 20.0 + 0.4 * i) for i, w in
                 enumerate('and a strong [ __ ] move by lebron james to the rim'.split())]
        write_json3(self.farm / 'captions' / 'vid1.json3', words)
        self.clip('c1', 1018.0)
        st = self.run_label()
        self.assertEqual((st['labeled'], st['rejected']), (0, 1))

    def test_clip_without_speech_rejected_and_not_retried(self):
        self.clip('c2', 1001.0)            # video 1..11 s: no caption words
        self.assertEqual(self.run_label()['rejected'], 1)
        self.assertEqual(self.run_label()['rejected'], 0)   # remembered as done

    def test_clip_past_play_log_waits(self):
        self.clip('c3', 1058.0)            # ends at 1068 > last sync 1060
        st = self.run_label()
        self.assertEqual(st['pending'], 1)
        self.assertFalse((self.ds / 'manifest.jsonl').read_text().strip())

    def test_pre_farm_clip_without_mono_time_rejected(self):
        (self.clips / 'old.json').write_text(json.dumps({'kind': 'asr', 'duration_s': 10}))
        self.assertEqual(self.run_label()['rejected'], 1)

    def test_words_cut_by_clip_edge_excluded(self):
        self.clip('c4', 1012.4)            # video 12.4..22.4 s: clip ends during "lebron"
        self.run_label()
        row = json.loads((self.ds / 'manifest.jsonl').read_text())
        self.assertEqual(row['text'], 'and a strong move by')
        self.assertFalse(row['has_name'])

    def test_held_out_video_never_labelled(self):
        with open(self.farm / 'playlog' / 'x.jsonl') as f:
            lines = f.read().replace('"vid1"', '"2nC9z57MuaI"')
        (self.farm / 'playlog' / 'x.jsonl').write_text(lines)
        (self.farm / 'captions' / 'vid1.json3').rename(self.farm / 'captions' / '2nC9z57MuaI.json3')
        self.clip('c5', 1018.0)
        st = self.run_label()
        self.assertEqual(st['labeled'], 0)
        self.assertEqual(st['rejected'], 1)

    def test_label_not_matching_asr_rejected(self):
        # Wrong video on screen (autoplay, ad): ASR heard something else.
        self.clip('c6', 1018.0, heard='welcome back to the show folks')
        st = self.run_label()
        self.assertEqual((st['labeled'], st['rejected']), (0, 1))

    def test_clip_without_asr_text_rejected(self):
        self.clip('c7', 1018.0, heard='')
        self.assertEqual(self.run_label()['rejected'], 1)

    def test_partial_asr_agreement_kept(self):
        # ASR misheard the name but got most words: still a good label.
        self.clip('c8', 1018.0, heard='and a strong move by le bron james to rim')
        self.assertEqual(self.run_label()['labeled'], 1)
        row = json.loads((self.ds / 'manifest.jsonl').read_text())
        self.assertGreaterEqual(row['asr_overlap'], lc.MIN_ASR_OVERLAP)

    def test_summary(self):
        self.clip('c1', 1018.0)
        self.run_label()
        s = lc.summary(self.ds)
        self.assertEqual(s['clips'], 1)
        self.assertEqual(s['with_name'], 1)


class TestNormalize(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(lc.normalize("LeBron's 3-pointer — WOW!"), "lebron's 3 pointer wow")


class TestPlayLogBoot(unittest.TestCase):
    """Sync times are time.monotonic(): a play log from before this boot is
    on another clock and must not be matched against today's clips."""

    def _farm(self):
        farm = Path(tempfile.mkdtemp())
        (farm / 'playlog').mkdir()
        for name, t in (('old.jsonl', 5000), ('new.jsonl', 1000)):
            (farm / 'playlog' / name).write_text(json.dumps(
                {'type': 'sync', 'video': name[:3], 'board_mid': t, 'device_uptime': t,
                 'position': 0.0, 'updated': t, 'speed': 1.0}) + '\n')
        booted = lc._boot_time()
        os.utime(farm / 'playlog' / 'old.jsonl', (booted - 3600, booted - 3600))
        return farm

    def test_log_from_before_boot_skipped(self):
        syncs = lc.load_syncs(self._farm())
        self.assertEqual([v for _, v, _ in syncs], ['new'])


class TestExcludeFiles(unittest.TestCase):
    """CLIP_FARM_EXCLUDE_FILES: the live bench's video lists are held out."""

    def test_reads_vid_id_and_plain_entries(self):
        d = Path(tempfile.mkdtemp())
        (d / 'holdout.json').write_text(json.dumps([{'vid': 'aaa', 't0': 5}, {'vid': 'bbb'}]))
        (d / 'neg.json').write_text(json.dumps([{'id': 'ccc'}, 'ddd', {'title': 'no id'}]))
        self.assertEqual(lc._ids_from_files([d / 'holdout.json', d / 'neg.json']),
                         {'aaa', 'bbb', 'ccc', 'ddd'})

    def test_missing_or_bad_files_are_skipped(self):
        d = Path(tempfile.mkdtemp())
        (d / 'bad.json').write_text('{not json')
        self.assertEqual(lc._ids_from_files([d / 'bad.json', d / 'nope.json']), set())


if __name__ == '__main__':
    unittest.main()


class TestVttCaptions(unittest.TestCase):
    PLAIN = ("WEBVTT\nKind: captions\nLanguage: en\n\n"
             "00:00:01.000 --> 00:00:03.000 line:7%\n&gt;&gt; LEBRON. I'M SURE WE'LL GET\n\n"
             "00:00:03.000 --> 00:00:05.000 line:7%\nTO FUTURE QUESTIONS SOON.\n\n")
    ROLLING = ("WEBVTT\nKind: captions\nLanguage: en\n\n"
               "00:00:08.641 --> 00:00:30.262 align:start position:0%\n \n"
               "AT<00:00:08.674><c> H</c><00:00:08.707><c>OM</c><00:00:08.741><c>E </c>\n\n"
               "00:00:30.262 --> 00:00:30.396 align:start position:0%\nAT HOME \n \n\n")

    def _farm(self, vid, text):
        d = Path(tempfile.mkdtemp())
        (d / 'captions').mkdir()
        (d / 'captions' / f'{vid}.vtt').write_text(text)
        return d

    def test_entities_decoded_and_speaker_marks_dropped(self):
        d = self._farm('plainvid0001', self.PLAIN)
        words = [w for w, _, _ in lc.caption_words(d, 'plainvid0001')]
        self.assertEqual(words[:3], ['LEBRON.', "I'M", 'SURE'])
        self.assertNotIn('gt', lc.normalize(' '.join(words)).split())

    def test_rolling_vtt_gives_no_words(self):
        d = self._farm('rollvid00001', self.ROLLING)
        self.assertTrue(lc.is_rolling_vtt(d / 'captions' / 'rollvid00001.vtt'))
        self.assertEqual(lc.caption_words(d, 'rollvid00001'), [])
        self.assertFalse(lc.is_rolling_vtt(self._farm('plainvid0002', self.PLAIN)
                                           / 'captions' / 'plainvid0002.vtt'))
