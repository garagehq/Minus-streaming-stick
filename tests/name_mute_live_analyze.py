#!/usr/bin/env python3
"""Analyse name_mute_live_measure.py output against a YouTube json3 track.

usage: python3 tests/name_mute_live_analyze.py data.json captions.json3 [hdmi_latency_s]

All times are mapped to video position. hdmi_latency_s (default 0) is how
long after the player reports a position that audio reaches the capture.
"""
import bisect
import json
import re
import statistics
import sys

NAME_RE = re.compile(r"\b(le ?bron|lebrun|king\b(?!s))", re.I)


def load_mentions(path, lo, hi):
    """Caption mentions in [lo, hi]: (start, end, text). 'LeBron James' is one span."""
    words = []
    for ev in json.load(open(path)).get('events', []):
        for sg in ev.get('segs', []) or []:
            w = sg.get('utf8', '').strip()
            if w:
                words.append(((ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000.0, w))
    words.sort()
    out = []
    for i, (t, w) in enumerate(words):
        if not (lo <= t <= hi) or not NAME_RE.search(w):
            continue
        nxt = words[i + 1] if i + 1 < len(words) else (t + 0.6, '')
        end = min(nxt[0], t + 0.8)
        text = w
        if nxt[1].lower().startswith('james') and i + 2 < len(words):
            # The track has word starts only. "James" is ~0.45s spoken; the
            # next word's start can be seconds later after a pause.
            end = min(words[i + 2][0], nxt[0] + 0.45)
            text = f"{w} {nxt[1]}"
        out.append((t, max(end, t + 0.25), text))
    return out


def main():
    data = json.load(open(sys.argv[1]))
    lat = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    syncs = [s[0] for s in data['syncs'] if s]
    mids = [s['board_mid'] for s in syncs]

    def vpos(m):
        s = syncs[max(0, bisect.bisect_right(mids, m) - 1)]
        off = s['device_uptime'] - s['board_mid']
        return s['position'] + (m + off - s['updated']) * s['speed'] - lat

    dets = data['detections']
    mutes = []   # (start, end, source, label, meta) in video time
    for d in dets:
        if d['play_start'] is None:
            continue
        mutes.append((vpos(d['play_start'] - d['delay']), vpos(d['play_end'] - d['delay']),
                      d['source'], d['label'], d.get('meta', {})))
    mutes.sort()
    lo, hi = vpos(data['start'] + 8), vpos(data['end'] - 8)
    mentions = load_mentions(sys.argv[2], lo, hi)

    def covered_by(a, b, srcs=None):
        # Against the union of windows: two overlapping mutes cover a mention
        # together even if neither covers it alone.
        segs = []
        for s, e, src, _, _ in mutes:
            if srcs is not None and src not in srcs:
                continue
            if segs and s <= segs[-1][1]:
                segs[-1][1] = max(segs[-1][1], e)
            else:
                segs.append([s, e])
        return any(s <= a and b <= e for s, e in segs)

    full = part = 0
    leads, tails = [], []
    missed = []
    by_src = {'asr': 0, 'caption': 0}
    for a, b, text in mentions:
        over = [(s, e) for s, e, *_ in mutes if s < b and e > a]
        if covered_by(a, b):
            full += 1
            s, e = min(over)[0], max(e for _, e in over)
            leads.append(a - s)
            tails.append(e - b)
        elif over:
            part += 1
            missed.append((round(a, 1), text, 'partial'))
        else:
            missed.append((round(a, 1), text, 'missed'))
        for src in by_src:
            if covered_by(a, b, {src}):
                by_src[src] += 1

    extra = [(round(s, 1), round(e - s, 2), src, lab[:50], meta)
             for s, e, src, lab, meta in mutes
             if not any(s - 0.5 < b and e + 0.5 > a for a, b, _ in mentions)]
    merged = []   # overlapping windows mute once
    for s, e, *_ in mutes:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    muted = sum(e - s for s, e in merged)
    speech = sum(b - a for a, b, _ in mentions)

    # Detector timing vs the caption word start (signed: + = after).
    def offsets(src, key):
        out = []
        for d in dets:
            if d['source'] != src:
                continue
            t = vpos(d[key])
            near = min(mentions, key=lambda m: abs(m[0] - t), default=None)
            if near and abs(near[0] - t) < 2.0:
                out.append(t - near[0])
        return out

    n = len(mentions)
    q = lambda xs: (f"p10 {statistics.quantiles(xs, n=10)[0]:+.2f} / median "
                    f"{statistics.median(xs):+.2f} / p90 {statistics.quantiles(xs, n=10)[-1]:+.2f}s"
                    if len(xs) >= 3 else str([round(x, 2) for x in xs]))
    print(f"video {lo:.0f}-{hi:.0f}s ({(hi - lo) / 60:.1f} min), hdmi latency {lat:+.2f}s")
    print(f"mentions: {n}  fully muted {full} ({100 * full / max(1, n):.0f}%)  "
          f"partial {part}  missed {n - full - part}")
    print(f"  fully muted by asr alone {by_src['asr']}, by captions alone {by_src['caption']}")
    print(f"  mute starts before word: {q(leads)}")
    print(f"  mute ends after word:    {q(tails)}")
    print(f"mutes: {len(mutes)} windows, {muted:.1f}s muted for {speech:.1f}s of name speech "
          f"({muted / max(speech, 0.01):.1f}x)")
    print(f"  not near any spoken mention: {len(extra)} "
          f"(asr {sum(1 for x in extra if x[2] == 'asr')}, "
          f"caption {sum(1 for x in extra if x[2] == 'caption')})")
    print(f"asr word start vs caption word: {q(offsets('asr', 'capture_start'))}")
    cap_off = [o + (0.7 if any(d['capture_end'] - d['capture_start'] > 0.01
                               for d in dets if d['source'] == 'caption') else 0.0)
               for o in offsets('caption', 'capture_start')]
    print(f"caption seen on screen vs word: {q(cap_off)}")
    # Bare "James" (not part of "LeBron James"): not a target by default
    # (name_mute_surname), but in LeBron footage it is usually him.
    words = []
    for ev in json.load(open(sys.argv[2])).get('events', []):
        for sg in ev.get('segs', []) or []:
            w = sg.get('utf8', '').strip()
            if w:
                words.append(((ev['tStartMs'] + sg.get('tOffsetMs', 0)) / 1000.0, w))
    words.sort()
    bare = [t for i, (t, w) in enumerate(words)
            if lo <= t <= hi and re.match(r"james\b", w, re.I)
            and not (i and NAME_RE.search(words[i - 1][1]))]
    bare_muted = sum(1 for t in bare if covered_by(t, t + 0.4))
    print(f"bare 'James' (not targeted by default): {len(bare)}, inside a mute anyway: {bare_muted}")
    print("missed/partial:", missed[:15])
    print("extra:", extra[:15])


if __name__ == '__main__':
    main()
