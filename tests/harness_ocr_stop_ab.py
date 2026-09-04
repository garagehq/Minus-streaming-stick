#!/usr/bin/env python3
"""A/B evaluation of OCR_STOP_THRESHOLD against the real block-flap failure.

WHY NOT THE OVERLAY RIG: block_latency_harness.py injects clean, static
overlay text that production OCR reads on essentially every frame. The
failure we are fixing is the opposite — OCR *intermittently* misses the ad
keyword while the ad is still on screen, so the clean rig cannot reproduce
it. Instead we drive the harness's real DecisionEngine (a faithful mirror of
minus.py's blocking state machine) with the miss pattern measured from
production logs, the same bootstrapping approach used by
tests/test_vlm_decision_sim.py.

EMPIRICAL INPUT (24h of production logs, 101 ad breaks, 2309 OCR frames
inside those breaks — measured uncensored, i.e. over the raw OCR keyword
stream rather than inside blocks, because a run of N misses ends the block
at threshold N and would censor every longer run):

    overall miss rate inside an ad break : 12.4%
    consecutive-miss run lengths         : 1:15  2:118  3:4  4:6  5+:0

Runs of exactly 2 dominate, which is precisely what OCR_STOP_THRESHOLD=2
trips on — hence 58.8% of blocks re-blocking within 5s and 300 mute/unmute
cycles overnight.

Usage:  python3 tests/harness_ocr_stop_ab.py
"""

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from block_latency_harness import DecisionEngine, PARAMS  # noqa: E402

# Empirically measured run-length distribution of consecutive OCR misses
# inside a live ad break (see module docstring).
MISS_RUN_WEIGHTS = {1: 15, 2: 118, 3: 4, 4: 6}
MISS_RATE = 0.124
OCR_CYCLE_S = 0.85          # measured production cadence (cap p50 581ms + ocr ~200ms)
AD_BREAK_S = 30.0           # typical pod length
POST_AD_FRAMES = 12         # frames after the ad genuinely ends


def _sample_run(rng):
    lengths = list(MISS_RUN_WEIGHTS)
    weights = [MISS_RUN_WEIGHTS[k] for k in lengths]
    return rng.choices(lengths, weights=weights, k=1)[0]


def build_ad_break(rng, n_frames):
    """Return a list of booleans: True = OCR saw an ad keyword this frame.

    Reproduces the measured intermittency: mostly hits, punctuated by
    miss-runs drawn from the empirical distribution, at the measured rate.
    """
    seq = []
    while len(seq) < n_frames:
        # run of hits
        seq.extend([True] * max(1, int(rng.expovariate(MISS_RATE) * 0.5)))
        if len(seq) >= n_frames:
            break
        seq.extend([False] * _sample_run(rng))
    return seq[:n_frames]


def run_scenario(threshold, rng):
    """Drive the real DecisionEngine through one ad break.

    Returns (flaps, recovery_s) — flaps = premature block stops while the ad
    is still on screen; recovery_s = time from the ad genuinely ending to
    blocking clearing.
    """
    params = dict(PARAMS)
    params['OCR_STOP_THRESHOLD'] = threshold
    params['disable_static_suppression'] = True   # isolate the OCR stop path
    eng = DecisionEngine(params)

    n = int(AD_BREAK_S / OCR_CYCLE_S)
    seq = build_ad_break(rng, n)

    t = 0.0
    was_blocking = False
    flaps = 0
    # --- during the ad ---
    for found in seq:
        eng.on_ocr(found)
        blocking = eng.ocr_ad_detected
        if was_blocking and not blocking:
            flaps += 1          # block dropped while the ad is still playing
        was_blocking = blocking
        t += OCR_CYCLE_S

    # --- ad ends: OCR now genuinely sees no keyword ---
    ad_end_t = t
    recovery = None
    for _ in range(POST_AD_FRAMES):
        eng.on_ocr(False)
        t += OCR_CYCLE_S
        if not eng.ocr_ad_detected:
            recovery = t - ad_end_t
            break
    if recovery is None:
        recovery = POST_AD_FRAMES * OCR_CYCLE_S
    return flaps, recovery


def main():
    print(__doc__.split('Usage:')[0].strip())
    print("\n" + "=" * 74)
    print(f"{'threshold':>9} | {'flaps/break':>12} | {'breaks w/ flap':>15} | {'recovery s':>12}")
    print("-" * 74)
    results = {}
    for threshold in (2, 3, 4, 5):
        rng = random.Random(1234)      # same scenarios for every threshold
        flaps, recs, clean = [], [], 0
        for _ in range(400):
            f, r = run_scenario(threshold, rng)
            flaps.append(f)
            recs.append(r)
            if f == 0:
                clean += 1
        results[threshold] = (statistics.mean(flaps), 100 * (400 - clean) / 400,
                              statistics.mean(recs), max(recs))
        m, pct, rmean, rmax = results[threshold]
        print(f"{threshold:>9} | {m:>12.2f} | {pct:>14.1f}% | "
              f"{rmean:>5.2f} (max {rmax:.2f})")
    print("=" * 74)

    base_flap = results[2][0]
    print("\nInterpretation:")
    for threshold in (3, 4, 5):
        m, pct, rmean, _ = results[threshold]
        drop = 100 * (base_flap - m) / base_flap if base_flap else 0
        cost = rmean - results[2][2]
        print(f"  threshold={threshold}: {drop:5.1f}% fewer flaps, "
              f"recovery +{cost:.2f}s vs threshold=2")
    print("\nProduction recovery target is <=1.5-2.0s (CLAUDE.md).")


if __name__ == '__main__':
    main()
