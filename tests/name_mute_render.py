#!/usr/bin/env python3
"""Render what the viewer heard: apply a live run's mutes to the source audio.

For every name mention in a live run (tests/name_mute_live_measure.py output,
scored on stretches where the video clock was steady, like the analyzer):
  * cut a clip from 1.5 s before to 1.5 s after the name, with the run's mute
    windows applied (silence) -- this is what came out of the TV;
  * measure how much of the name was audible (by the caption word timing);
  * run SenseVoice (NPU) on the muted clip and on the original clip, and
    check whether the name is still recognisable.
Writes one WAV per mention, a concatenated listening file (0.7 s gaps) and
an index of clip order.

usage: python3 tests/name_mute_render.py run.json captions.json3 source16k.wav OUTDIR [npu_core]
"""
import bisect
import importlib.util
import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from name_mute import NameMatcher  # noqa: E402
from sensevoice_npu import SenseVoiceNPU  # noqa: E402

spec = importlib.util.spec_from_file_location('analyze', os.path.join(HERE, 'name_mute_live_analyze.py'))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

SR = 16000
PRE, POST = 1.5, 1.5


def main():
    run, cap, src, out = sys.argv[1:5]
    core = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    os.makedirs(out, exist_ok=True)
    data = json.load(open(run))
    syncs = [s[0] for s in data['syncs'] if s]
    mids = [s['board_mid'] for s in syncs]

    def vpos(m):
        s = syncs[max(0, bisect.bisect_right(mids, m) - 1)]
        return s['position'] + (m + s['device_uptime'] - s['board_mid'] - s['updated']) * s['speed']

    def implied(s, m):
        return s['position'] + (m + s['device_uptime'] - s['board_mid'] - s['updated']) * s['speed']
    trusted = []
    for a, b in zip(syncs, syncs[1:]):
        if abs(implied(b, b['board_mid']) - implied(a, b['board_mid'])) < 0.3:
            if trusted and abs(trusted[-1][1] - a['board_mid']) < 1e-6:
                trusted[-1][1] = b['board_mid']
            else:
                trusted.append([a['board_mid'], b['board_mid']])
    in_trusted = lambda m: any(x <= m <= y for x, y in trusted)
    vranges = [(vpos(x) + 2.0, vpos(y) - 2.0) for x, y in trusted if y - x > 5]

    mutes = []
    for d in data['detections']:
        if d['play_start'] is None:
            continue
        ps, pe = d['play_start'] - d['delay'], d['play_end'] - d['delay']
        if in_trusted(ps) and in_trusted(pe):
            mutes.append((vpos(ps), vpos(pe)))
    mentions = [m for m in analyze.load_mentions(cap, -1e9, 1e9)
                if any(x <= m[0] and m[1] <= y for x, y in vranges)]

    w = wave.open(src)
    assert w.getframerate() == SR and w.getnchannels() == 1
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    muted = audio.copy()
    for s, e in mutes:
        muted[max(0, int(s * SR)):max(0, int(e * SR))] = 0.0

    sv = SenseVoiceNPU(npu_core=core)
    matcher = NameMatcher()
    rows, listen = [], []
    gap = np.zeros(int(0.7 * SR), dtype=np.float32)
    for i, (a, b, text) in enumerate(mentions):
        lo, hi = int((a - PRE) * SR), int((b + POST) * SR)
        clip_m, clip_o = muted[lo:hi], audio[lo:hi]
        # audible part of the name, by caption timing
        n = int((b - a) * SR)
        name_m = muted[int(a * SR):int(a * SR) + n]
        audible_ms = 1000 * float(np.count_nonzero(name_m)) / SR
        lead_ms = 1000 * float(np.argmax(name_m == 0) if (name_m == 0).any() else n) / SR
        heard_m = ' '.join(x[0] for x in sv.words(clip_m))
        heard_o = ' '.join(x[0] for x in sv.words(clip_o))
        rec_m, rec_o = bool(matcher.find(heard_m)), bool(matcher.find(heard_o))
        status = 'full' if audible_ms < 1 else ('partial' if audible_ms < (b - a) * 1000 - 1 else 'unmuted')
        name = f"{i:02d}_{a:07.1f}s_{status}.wav"
        with wave.open(os.path.join(out, name), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
            f.writeframes((np.clip(clip_m, -1, 1) * 32767).astype(np.int16).tobytes())
        listen += [clip_m, gap]
        rows.append({'clip': name, 'video_s': round(a, 2), 'text': text, 'status': status,
                     'audible_ms': round(audible_ms), 'audible_lead_ms': round(lead_ms),
                     'asr_muted': heard_m, 'name_heard_muted': rec_m,
                     'asr_original': heard_o, 'name_heard_original': rec_o})
    with wave.open(os.path.join(out, 'listen_all.wav'), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((np.clip(np.concatenate(listen or [gap]), -1, 1) * 32767).astype(np.int16).tobytes())
    json.dump(rows, open(os.path.join(out, 'index.json'), 'w'), indent=1)

    part = [r for r in rows if r['status'] == 'partial']
    print(f"{os.path.basename(run)}: {len(rows)} mentions, full {sum(r['status'] == 'full' for r in rows)}, "
          f"partial {len(part)}, unmuted {sum(r['status'] == 'unmuted' for r in rows)}")
    if part:
        ms = sorted(r['audible_ms'] for r in part)
        print(f"  partial: audible {ms} ms (lead-in before the mute: {[r['audible_lead_ms'] for r in part]})")
    ctl = [r for r in rows if r['name_heard_original']]
    print(f"  SenseVoice hears the name in {len(ctl)}/{len(rows)} original clips; of those, still hears it "
          f"after muting in {sum(r['name_heard_muted'] for r in ctl)} "
          f"(full-muted {sum(r['name_heard_muted'] for r in ctl if r['status'] == 'full')}, "
          f"partial {sum(r['name_heard_muted'] for r in ctl if r['status'] == 'partial')})")
    for r in part:
        print(f"   partial {r['video_s']:7.1f}s {r['audible_ms']:4d}ms audible | muted: {r['asr_muted'][:60]!r}")


if __name__ == '__main__':
    main()
