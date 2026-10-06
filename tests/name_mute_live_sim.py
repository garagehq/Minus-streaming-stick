#!/usr/bin/env python3
"""Re-run name-mute windowing rules over a recorded live session.

Takes the detections recorded by name_mute_live_measure.py and rebuilds the
mute windows under candidate parameters, then scores them against the
YouTube caption track (same scoring as name_mute_live_analyze.py). Lets the
padding / dedup / filtering rules be tuned without replaying the video.

usage: python3 tests/name_mute_live_sim.py data.json captions.json3 [--sweep]
"""
import bisect
import difflib
import itertools
import json
import re
import sys

sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/tests')
from name_mute_live_analyze import load_mentions  # noqa: E402

NAME_RE = re.compile(r"(le ?bron+|lebrun|la ?bron|the bron james|king james)", re.I)


def norm(t):
    t = re.sub(r"[^a-z' ]+", ' ', t.lower().replace("’", "'"))
    return re.sub(r'\s+', ' ', t).strip()


def build(dets, p):
    """Rebuild capture-time mute windows from recorded detections."""
    wins = []           # [start, end, source]
    recent_caps = []    # (time, context)
    for d in sorted(dets, key=lambda d: d['detected']):
        meta = d.get('meta') or {}
        if d['source'] == 'asr':
            ws, we = meta.get('word_start'), meta.get('word_end')
            if ws is None:
                continue
            if p['asr_min_word'] and (we - ws) < p['asr_min_word']:
                continue   # zero-length timings: repetition hallucinations
            base = d['capture_start'] - ws          # window start in capture time
            trunc = we >= 2.45
            s = base + ws - p['asr_before']
            e = base + we + (0.4 if trunc else 0.0) + p['asr_after']
            if p['asr_replaces_caption']:
                # A precise ASR hit replaces a caption window for the same
                # mention, but only one that has not started playing yet
                # (live, the scheduler cannot take back a mute in progress).
                wins = [w for w in wins
                        if not (w[2] == 'caption' and w[0] + d['delay'] > d['detected']
                                and w[0] < e + 0.8 and w[1] > s - 0.8)]
            if any(w[0] <= s and e <= w[1] for w in wins):
                continue
            wins.append([s, e, 'asr'])
        else:
            label = d['label']
            if p['cap_min_words'] and len(norm(label).split()) < p['cap_min_words']:
                continue
            if p['cap_no_allcaps'] and label.upper() == label and any(c.isalpha() for c in label):
                continue
            # Frame capture time. Older logs stored the caption window
            # [frame - 0.7, frame + 0.6]; newer ones store the frame time as a
            # zero-length span and pass the pads separately.
            if d['capture_end'] - d['capture_start'] > 0.01:
                cap = d['capture_start'] + 0.7
            else:
                cap = d['capture_start']
            m = NAME_RE.search(norm(label))
            ctx = norm(label)[:m.start()][-20:] if m else norm(label)
            if p['cap_fuzzy']:
                recent_caps = [(t, c) for t, c in recent_caps if cap - t < 10]
                if any(difflib.SequenceMatcher(None, c, ctx).ratio() >= p['cap_fuzzy']
                       for t, c in recent_caps):
                    recent_caps.append((cap, ctx))
                    continue
                recent_caps.append((cap, ctx))
            s, e = cap - p['cap_before'], cap + p['cap_after']
            if p['cap_skip_if_asr'] and any(w[2] == 'asr' and w[0] < e + 0.8 and w[1] > s - 0.8 for w in wins):
                continue
            if any(w[0] <= s and e <= w[1] for w in wins):
                continue
            wins.append([s, e, 'caption'])
    return wins


def score(data, mentions, wins, vpos):
    mutes = sorted((vpos(s), vpos(e), src) for s, e, src in wins)
    # Overlapping windows mute once: score coverage and seconds on the union.
    merged = []
    for s, e, _ in mutes:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    full = part = 0
    for a, b, _ in mentions:
        if any(s <= a and b <= e for s, e in merged):
            full += 1
        elif any(s < b and e > a for s, e in merged):
            part += 1
    muted = sum(e - s for s, e in merged)
    extra = sum(1 for s, e, _ in mutes if not any(s - 0.5 < b and e + 0.5 > a for a, b, _ in mentions))
    return full, part, muted, extra


DEFAULT = dict(asr_before=0.6, asr_after=0.35, asr_min_word=0.0, asr_replaces_caption=False,
               cap_before=1.3, cap_after=0.95, cap_min_words=0, cap_no_allcaps=False,
               cap_fuzzy=0.0, cap_skip_if_asr=False)


def main():
    data = json.load(open(sys.argv[1]))
    syncs = [s[0] for s in data['syncs'] if s]
    mids = [s['board_mid'] for s in syncs]

    def vpos(m):
        s = syncs[max(0, bisect.bisect_right(mids, m) - 1)]
        return s['position'] + (m + s['device_uptime'] - s['board_mid'] - s['updated']) * s['speed']

    lo, hi = vpos(data['start'] + 8), vpos(data['end'] - 8)
    mentions = load_mentions(sys.argv[2], lo, hi)
    speech = sum(b - a for a, b, _ in mentions)
    dets = data['detections']

    def run(p, tag):
        full, part, muted, extra = score(data, mentions, build(dets, p), vpos)
        n = len(mentions)
        print(f"{tag:58s} full {full}/{n} ({100*full/n:.0f}%) part {part} "
              f"muted {muted:6.1f}s ({muted/speech:.1f}x) extra {extra}")
        return full, muted

    run(DEFAULT, 'current rules')
    if '--sweep' not in sys.argv:
        return
    best = []
    grid = dict(asr_before=[0.3, 0.45, 0.6], asr_after=[0.1, 0.2, 0.35],
                cap_before=[0.9, 1.1, 1.3], cap_after=[0.1, 0.3, 0.5],
                cap_fuzzy=[0.0, 0.6], asr_replaces_caption=[False, True])
    fixed = dict(DEFAULT, asr_min_word=0.08, cap_min_words=3, cap_no_allcaps=True)
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        p = dict(fixed, **dict(zip(keys, vals)))
        full, part, muted, extra = score(data, mentions, build(dets, p), vpos)
        best.append((full, -muted, p, part, extra))
    best.sort(key=lambda x: (x[0], x[1]), reverse=True)
    n = len(mentions)
    print(f"\nname speech {speech:.1f}s over {n} mentions. Top settings by coverage, then least muted:")
    for full, negm, p, part, extra in best[:12]:
        print(f"full {full}/{n} part {part} muted {-negm:6.1f}s ({-negm/speech:.1f}x) extra {extra}  "
              + ' '.join(f"{k}={p[k]}" for k in keys))


if __name__ == '__main__':
    main()
