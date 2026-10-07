#!/usr/bin/env python3
"""Score zf_stream asr output against caption name mentions.

usage: zf_analyze.py zf.tsv captions.json3
For each caption mention: what the zipformer wrote around it, whether the
name matcher fires on that text, and the detection lag (time the matching
word was emitted + its decode time - caption word start).
"""
import importlib.util
import os
import sys

MINUS = os.path.expanduser('~/Minus')
sys.path.insert(0, os.path.join(MINUS, 'src'))
from name_mute import NameMatcher  # noqa: E402

spec = importlib.util.spec_from_file_location(
    'analyze', os.path.join(MINUS, 'tests', 'name_mute_live_analyze.py'))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

words = []  # [text, start_ts, emit_time]
for line in open(sys.argv[1]):
    parts = line.rstrip('\n').split('\t')
    if len(parts) != 4:
        continue
    fed, ms, tok, ts = parts
    emit = float(fed) + float(ms) / 1000
    if tok.startswith(' ') or not words:
        words.append([tok.strip().lower(), float(ts), emit])
    else:
        words[-1][0] += tok.lower()
        words[-1][2] = emit

mentions = analyze.load_mentions(sys.argv[2], -1e9, 1e9)
m = NameMatcher()
lags, hit = [], 0
for a, b, text in mentions:
    near = [w for w in words if a - 1.5 <= w[1] <= b + 1.5]
    found = None
    for i, w in enumerate(near):
        ctx = ' '.join(x[0] for x in near[max(0, i - 1):i + 1])
        if m.find(w[0]) or m.find(ctx):
            found = w
            break
    if found:
        hit += 1
        lags.append(found[2] - a)
    print(f"{a:7.1f}s {text!r:22} -> {' '.join(w[0] for w in near)[:70]!r}"
          + (f"  HIT lag {found[2] - a:+.2f}s" if found else ''))
lags.sort()
print(f"\n{hit}/{len(mentions)} mentions matched")
if lags:
    print(f"lag after name start: p50 {lags[len(lags) // 2]:.2f} / p90 {lags[int(len(lags) * 0.9)]:.2f} / max {lags[-1]:.2f}s")
# all non-mention triggers
print("\nmatches away from any caption mention:")
for i, w in enumerate(words):
    if m.find(w[0]) and not any(a - 1.5 <= w[1] <= b + 1.5 for a, b, _ in mentions):
        print(f"  {w[1]:7.1f}s {' '.join(x[0] for x in words[max(0, i - 3):i + 2])!r}")
