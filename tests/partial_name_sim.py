#!/usr/bin/env python3
"""Offline simulation of name-mute rules on dumped ASR windows.

Replays tests/asr_window_dump.py output through matcher variants and
scores, at several effective delays, how many caption name mentions would
be fully muted, partly muted or missed, and how much non-name audio gets
muted. A detection is usable once its window has been inferred
(window end + inference time); a mute can only cover audio whose playback
(capture time + delay) is still in the future at that moment.

Variants:
  base     production NameMatcher spans (whole window when only the text
           matches)
  partial  + a cut-off "le…"/"lebr…" fragment that is the last word of a
           window mutes provisionally
  cancel   partial, and a later window that hears that stretch whole
           without a name ends the provisional mute early

usage: tests/partial_name_sim.py DUMP.jsonl captions.json3 [DUMP2 CAP2 ...]
"""
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from name_mute import NameMatcher, NameMuteController, NameMuteScheduler, _norm  # noqa: E402

spec = importlib.util.spec_from_file_location('an', os.path.join(HERE, 'name_mute_live_analyze.py'))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

PAD_BEFORE = NameMuteScheduler.PAD_BEFORE_S
PAD_AFTER = NameMuteScheduler.PAD_AFTER_S
MIN_LATE = NameMuteScheduler.MIN_LATE_MUTE_S
TRUNC = NameMuteController.TRUNCATED_WORD_EXTRA_S
WINDOW_S = 3.0

PARTIAL_RE = re.compile(r"^(?:le|leb|lebr|lebr[aou]\w*|lebo\w*|ler|lero|leron|lbr\w*)$")
PARTIAL_EDGE_S = 0.5      # fragment must start within this of the window end
PARTIAL_MUTE_S = 0.8      # provisional mute length past the fragment start


def load(dump):
    return [json.loads(line) for line in open(dump)]


def detections(windows, variant, matcher):
    """[(available_at, cap_start, cap_end, kind)] in capture/audio time."""
    out = []
    for w in windows:
        avail = w['end'] + w['ms'] / 1000.0
        words = [(x[0], x[1], x[2]) for x in w['words']]
        ws = w['end'] - WINDOW_S
        rel = [(x, s - ws, e - ws) for x, s, e in words]
        spans = matcher.find_word_spans(rel) if rel else []
        text = ' '.join(x for x, _, _ in words)
        if not spans and matcher.find(text):
            spans = [(0.0, max((e for _, _, e in rel), default=WINDOW_S))]
        for s, e in spans:
            if e - s < 0.08:
                continue
            if s <= 0.05:
                s -= TRUNC
            if e >= WINDOW_S - 0.05:
                e += TRUNC
            out.append((avail, ws + s, ws + e, 'name'))
        if variant != 'base' and not spans and rel:
            x, s, e = rel[-1]
            if s >= WINDOW_S - PARTIAL_EDGE_S and PARTIAL_RE.match(_norm(x)):
                out.append((avail, ws + s, ws + s + PARTIAL_MUTE_S, 'partial'))
    return out


def cancellations(windows):
    """For the cancel variant: [(available_at, cap_t0, cap_t1)] stretches a
    window heard whole (not at its edges) with no name word."""
    out = []
    m = NameMatcher()
    for w in windows:
        avail = w['end'] + w['ms'] / 1000.0
        words = w['words']
        if len(words) < 2:
            continue
        ws = w['end'] - WINDOW_S
        rel = [(x, s - ws, e - ws) for x, s, e in words]
        if m.find_word_spans(rel) or m.find(' '.join(x for x, _, _ in rel)):
            continue
        # Words fully inside the window (not cut by either edge).
        inner = [(s, e) for x, s, e in words if s > ws + 0.1 and e < w['end'] - 0.2]
        if inner:
            out.append((avail, min(s for s, _ in inner), max(e for _, e in inner)))
    return out


def mute_intervals(dets, cancels, delay, variant):
    """Capture-time intervals actually muted."""
    ivs = []
    for avail, s, e, kind in dets:
        earliest = avail - delay           # audio captured before this already played
        start = max(s - PAD_BEFORE, earliest)
        end = e + PAD_AFTER
        if start >= end:
            end = start + MIN_LATE
        elif s - PAD_BEFORE < earliest:
            end = max(end, earliest + MIN_LATE)
        if kind == 'partial' and variant == 'cancel':
            # End the provisional mute at the first later window that heard
            # this stretch whole without a name (cut at what is playing then).
            for c_avail, c0, c1 in cancels:
                if c_avail > avail and c0 <= s + 0.05 and c1 >= s + 0.3:
                    end = min(end, max(start, c_avail - delay))
                    break
        if end > start:
            ivs.append((start, end))
    ivs.sort()
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def score(merged, mentions, span):
    full = part = miss = 0
    leak = []
    for a, b, _ in mentions:
        covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e in merged)
        if covered >= (b - a) - 0.01:
            full += 1
        elif covered > 0:
            part += 1
            leak.append((b - a) - covered)
        else:
            miss += 1
    near = [(a - 1.0, b + 1.0) for a, b, _ in mentions]
    false_s = 0.0
    false_n = 0
    for s, e in merged:
        overlap = sum(max(0.0, min(e, y) - max(s, x)) for x, y in near)
        if overlap < 1e-6:
            false_n += 1
        false_s += max(0.0, (e - s) - overlap)
    hours = span / 3600.0
    return full, part, miss, leak, false_n / hours, false_s / hours


def main():
    args = sys.argv[1:]
    sets = []
    for dump, cap in zip(args[0::2], args[1::2]):
        w = load(dump)
        lo, hi = w[0]['end'] - WINDOW_S + 2, w[-1]['end'] - 1
        sets.append((os.path.basename(dump), w, an.load_mentions(cap, lo, hi), hi - lo))
    matcher = NameMatcher()
    print(f"{'set':10} {'variant':8} {'delay':>5}  {'full':>9} {'part':>4} {'miss':>4} "
          f"{'leak ms (median)':>16}  {'false mutes/h':>13} {'false s/h':>9}")
    for delay in (0.8, 1.0, 1.2, 1.6, 2.35):
        for variant in ('base', 'partial', 'cancel'):
            tot = [0, 0, 0, [], 0.0, 0.0]
            for name, w, ms, span in sets:
                dets = detections(w, variant, matcher)
                canc = cancellations(w) if variant == 'cancel' else []
                f, p, m, lk, fn, fs = score(mute_intervals(dets, canc, delay, variant), ms, span)
                tot[0] += f; tot[1] += p; tot[2] += m; tot[3] += lk
                tot[4] += fn * span / 3600; tot[5] += fs * span / 3600
                n = f + p + m
                print(f"{name:10} {variant:8} {delay:5.2f}  {f:3d}/{n:<3d}({100 * f / max(n, 1):3.0f}%) {p:4d} {m:4d} "
                      f"{(sorted(lk)[len(lk) // 2] * 1000 if lk else 0):16.0f}  {fn:13.1f} {fs:9.1f}")
            if len(sets) > 1:
                hours = sum(s[3] for s in sets) / 3600
                n = tot[0] + tot[1] + tot[2]
                lk = sorted(tot[3])
                print(f"{'ALL':10} {variant:8} {delay:5.2f}  {tot[0]:3d}/{n:<3d}({100 * tot[0] / max(n, 1):3.0f}%) "
                      f"{tot[1]:4d} {tot[2]:4d} {(lk[len(lk) // 2] * 1000 if lk else 0):16.0f}  "
                      f"{tot[4] / hours:13.1f} {tot[5] / hours:9.1f}")
        print()


if __name__ == '__main__':
    main()
