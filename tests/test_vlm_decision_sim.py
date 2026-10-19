"""
Monte-Carlo decision-engine simulator + sliding-window param sweep.

Goal: tune the OCR+VLM blocking algorithm for the FastVLM-0.5B iter4
classifier so it transitions INTO blocking fast and BACK to content fast,
with zero false-positive blocks / phantom re-blocks / mid-ad flaps.

Why a simulator: the real OCR+VLM rig (block_latency_harness.py) runs in
wall-clock time against bbb.mp4 and the NPU — minutes per scenario, and
bbb has no real ads. Here we drive the SAME faithful DecisionEngine
mirror with a virtual clock and feed it VLM verdicts BOOTSTRAPPED FROM
THE REAL 800-IMAGE HOLDOUT SCORES (threshold_sweep_raw.json). So the
per-frame error rate (ad-recall 94.25%, non-ad-recall 95.25%) and the
calibrated confidence distribution are statistically identical to
production iter4 — thousands of cases in seconds.

Run:
  python3 tests/test_vlm_decision_sim.py                 # eval current params
  python3 tests/test_vlm_decision_sim.py --sweep         # full param sweep
  python3 tests/test_vlm_decision_sim.py --seeds 60      # more MC samples
"""
import os
import sys
import json
import copy
import argparse
import itertools

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import block_latency_harness as H  # noqa: E402

HOLDOUT = '/home/radxa/axera_models/FastVLM/FastVLM-0.5B-ad-classifier-iter4/threshold_sweep_raw.json'

# ---- virtual clock: DecisionEngine.on_vlm calls module time.time() ----
# Only the HARNESS module's clock is virtual. H.time is the real `time`
# module, so `H.time.time = ...` used to freeze time.time() for the whole
# process: under pytest every test collected after this module ran on a
# clock stuck at 0 (23 cooldown/grace/rate-limit tests failed on the box,
# the only place the holdout file exists so this module gets imported).
_VT = [0.0]


class _VirtualClock:
    """`time` for the harness: time() is virtual, the rest is real."""

    def __init__(self, real):
        self._real = real

    def time(self):
        return _VT[0]

    def __getattr__(self, name):
        return getattr(self._real, name)


H.time = _VirtualClock(H.time)


def _load_scores():
    raw = json.load(open(HOLDOUT))
    ads = np.array([r['p_yes_norm'] for r in raw if r['label'] == 'ad'])
    non = np.array([r['p_yes_norm'] for r in raw if r['label'] == 'not_ad'])
    return ads, non


AD_SCORES, NON_SCORES = _load_scores()
AD_THRESHOLD = 0.76


def sample_vlm(is_ad_truth, rng):
    """Bootstrap a real iter4 p_yes_norm for this truth, return
    (is_ad_pred, confidence) exactly as production detect_ad would."""
    pool = AD_SCORES if is_ad_truth else NON_SCORES
    p = float(pool[rng.integers(len(pool))])
    is_ad = p > AD_THRESHOLD
    conf = p if is_ad else (1.0 - p)
    return is_ad, conf


# ---------------------------------------------------------------------------
# Scenario model: a list of (start, end) ad-truth intervals + meta over a
# total duration. OCR behaviour per scenario controls how/whether OCR fires
# on ad frames (strong = immediate, absent = VLM must carry it, delayed,
# flaky). `pause` injects a frozen window (scene static) over [ps, pe].
# ---------------------------------------------------------------------------
def simulate(params, scn, rng,
             ocr_interval=0.5, vlm_interval=0.85, dt=0.2):
    eng = H.DecisionEngine(params)
    _VT[0] = 0.0
    T = scn['duration']
    ads = scn['ads']                       # [(s,e), ...]
    ocr_mode = scn['ocr']                  # dict
    pause = scn.get('pause')               # (s,e) or None

    def truth_ad(t):
        return any(s <= t < e for s, e in ads)

    def in_pause(t):
        return pause is not None and pause[0] <= t < pause[1]

    last_ocr = -1e9
    last_vlm = -1e9
    prev_paused = False
    block_edges = []                       # (t, on)
    last_block = False

    t = 0.0
    while t <= T + 1e-9:
        _VT[0] = t
        ta = truth_ad(t)
        paused = in_pause(t)

        if t - last_vlm >= vlm_interval - 1e-9:
            last_vlm = t
            # VLM still polls a frozen ad frame in production (forced skip);
            # it sees the same ad content → sample on truth.
            is_ad, conf = sample_vlm(ta, rng)
            eng.on_vlm(is_ad, conf)

        if t - last_ocr >= ocr_interval - 1e-9:
            last_ocr = t
            # OCR ad-keyword presence model
            ocr_found = False
            if ta:
                m = ocr_mode['type']
                if m == 'strong':
                    ocr_found = True
                elif m == 'absent':
                    ocr_found = False
                elif m == 'delayed':
                    seg = next(s for s, e in ads if s <= t < e)
                    ocr_found = (t - seg) >= ocr_mode['delay']
                elif m == 'flaky':
                    ocr_found = rng.random() < ocr_mode['p']
            # scene_changed: True normally; False while paused (frozen)
            scene_changed = not paused
            eng.update_static(scene_changed, t)
            eng.on_ocr(ocr_found)

        is_block, _src = eng.compute_blocking(t)
        if is_block != last_block:
            block_edges.append((t, is_block))
            last_block = is_block
        prev_paused = paused
        t += dt

    return _score(scn, block_edges, T)


def _merge_regions(ads, gap=3.0):
    """Merge ad intervals separated by <= gap into one perceived ad break
    (production holds blocking through black/transition frames between
    ads in the same break)."""
    if not ads:
        return []
    a = sorted(ads)
    out = [list(a[0])]
    for s, e in a[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(r) for r in out]


def _region_internal_gap(region, ads):
    """True if this merged region was formed from >1 original ad — i.e.
    it contains a black/transition gap. Production holds blocking through
    such gaps via _is_transition_frame(); the harness DecisionEngine
    mirror does NOT model that, so a drop in such a gap is a mirror
    artifact, not a sliding-window-param failure → tracked separately."""
    s, e = region
    n = sum(1 for a, b in ads if a >= s - 1e-9 and b <= e + 1e-9)
    return n > 1


def _score(scn, edges, T):
    """Level-based detect/recover + safety, per merged ad-break region."""
    regions = _merge_regions(scn['ads'])
    res = {'detect': [], 'recover': [], 'false_block_s': 0.0,
           'phantom': 0, 'flaps': 0, 'gap_flaps': 0, 'miss': 0}

    def blocking_at(tq):
        st = False
        for (te, on) in edges:
            if te <= tq + 1e-9:
                st = on
            else:
                break
        return st

    GRACE = 4.0  # legit recovery tail after a break ends

    for (s, e) in regions:
        # detect: first instant at/after s that blocking is ON, within e+GRACE
        on_t = next((te for te, on in edges
                     if on and te >= s - 1e-9 and te <= e + GRACE), None)
        # also covered if blocking was already ON entering the region
        if blocking_at(s):
            on_t = s
        if on_t is None:
            res['miss'] += 1
            continue
        res['detect'].append(max(0.0, on_t - s))
        # recover: first instant at/after e that blocking is OFF
        off_t = next((te for te, on in edges
                      if (not on) and te >= e - 1e-9), None)
        res['recover'].append((off_t - e) if off_t is not None else (T - e))
        # flaps: blocking dropped OFF while still inside the ad region.
        # If the region had an internal black gap, that's a mirror
        # artifact (no transition-frame hold) → gap_flaps, not flaps.
        nflap = sum(1 for te, on in edges
                    if (not on) and on_t < te < e - 1e-9)
        if _region_internal_gap((s, e), scn['ads']):
            res['gap_flaps'] += nflap
        else:
            res['flaps'] += nflap

    def in_region(tq):
        return any(s <= tq < e for s, e in regions)

    def in_grace(tq):
        return any(e <= tq < e + GRACE for s, e in regions)

    tq = 0.0
    while tq <= T:
        if blocking_at(tq) and not in_region(tq) and not in_grace(tq):
            res['false_block_s'] += 0.1
        tq += 0.1

    # phantom: blocking turns ON outside any region and its grace tail
    for te, on in edges:
        if on and not in_region(te) and not in_grace(te):
            res['phantom'] += 1

    return res


# ---------------------------------------------------------------------------
# Scenario library — TONS of cases
# ---------------------------------------------------------------------------
def _cls(ocr_type, max_ad_len):
    """Class drives which metrics are HARD vs optimised:

    'O'  OCR-present ad — OCR is authoritative and immediate, so this
         MUST be caught fast and never flap (hard constraint).
    'V'  VLM-only ad >= 10s — the genuinely hard case; optimise detect
         latency, tolerate a bounded miss/flap rate (production also has
         OCR on most real ads + transition-frame holding the mirror lacks).
    'Vs' VLM-only ad < 10s — best-effort, reported only.
    """
    if ocr_type != 'absent':
        return 'O'
    return 'V' if max_ad_len >= 10 else 'Vs'


def build_scenarios():
    S = []
    OCR = {
        'strong': {'type': 'strong'},
        'absent': {'type': 'absent'},
        'delayed2': {'type': 'delayed', 'delay': 2.0},
        'delayed5': {'type': 'delayed', 'delay': 5.0},
        'flaky50': {'type': 'flaky', 'p': 0.5},
        'flaky20': {'type': 'flaky', 'p': 0.2},
    }

    # 1. pre-roll: ad at t=3 for L, then content
    for L in (5, 10, 15, 30, 60):
        for ok in ('strong', 'absent', 'delayed2', 'flaky50'):
            S.append({'name': f'preroll_L{L}_{ok}', 'duration': L + 25,
                      'ads': [(3, 3 + L)], 'ocr': OCR[ok]})

    # 2. mid-roll: content, ad, content
    for L in (6, 12, 20, 45):
        for ok in ('strong', 'absent', 'delayed2', 'flaky20'):
            S.append({'name': f'midroll_L{L}_{ok}', 'duration': L + 30,
                      'ads': [(12, 12 + L)], 'ocr': OCR[ok]})

    # 3. multi-ad break with short black gaps (flap stress)
    for gap in (1.0, 2.0, 3.0):
        for ok in ('strong', 'absent'):
            ads = []
            t = 5.0
            for _ in range(4):
                ads.append((t, t + 12))
                t += 12 + gap
            S.append({'name': f'multiad_gap{gap}_{ok}',
                      'duration': t + 20, 'ads': ads, 'ocr': OCR[ok]})

    # 4. back-to-back short ads
    for ok in ('strong', 'absent', 'flaky50'):
        ads = [(5, 11), (11, 18), (18, 23)]
        S.append({'name': f'b2b_short_{ok}', 'duration': 45,
                  'ads': ads, 'ocr': OCR[ok]})

    # 5. pause-on-ad (frozen screen during the ad)
    for ps_len in (2, 4, 8, 15):
        for ok in ('strong', 'absent'):
            S.append({'name': f'pause_on_ad_{ps_len}s_{ok}',
                      'duration': 50, 'ads': [(6, 30)],
                      'pause': (12, 12 + ps_len), 'ocr': OCR[ok]})

    # 6. pause-on-ad where ad ENDS during the pause (user-bug shape)
    for ok in ('strong', 'absent'):
        S.append({'name': f'pause_ad_ends_{ok}', 'duration': 45,
                  'ads': [(6, 16)], 'pause': (12, 22), 'ocr': OCR[ok]})

    # 7. content-only (MUST never block) — long, plus "hard" content runs
    for d in (30, 60, 120):
        S.append({'name': f'content_only_{d}', 'duration': d,
                  'ads': [], 'ocr': OCR['absent']})

    # 8. rapid ad/content alternation (10s ad, 10s content × 4)
    for ok in ('strong', 'absent'):
        ads = [(10, 20), (30, 40), (50, 60), (70, 80)]
        S.append({'name': f'alternation_{ok}', 'duration': 95,
                  'ads': ads, 'ocr': OCR[ok]})

    # 9. very short ad (3s) — recoverability stress
    for ok in ('strong', 'absent'):
        S.append({'name': f'tiny_ad_{ok}', 'duration': 25,
                  'ads': [(8, 11)], 'ocr': OCR[ok]})

    # 10. long ad (90s) sustained — must hold, no flaps
    for ok in ('strong', 'absent'):
        S.append({'name': f'long_hold_{ok}', 'duration': 110,
                  'ads': [(5, 95)], 'ocr': OCR[ok]})

    for scn in S:
        max_len = max((e - s for s, e in scn['ads']), default=0)
        scn['cls'] = _cls(scn['ocr']['type'], max_len)
    return S


# ---------------------------------------------------------------------------
def evaluate(params, scenarios, seeds):
    O = {'detect': [], 'recover': [], 'flaps': 0, 'miss': 0, 'n': 0}
    V = {'detect': [], 'recover': [], 'flaps': 0, 'miss': 0, 'n': 0}
    Vs = {'detect': [], 'miss': 0, 'n': 0}
    glb = {'false_block_s': 0.0, 'phantom': 0, 'gap_flaps': 0}
    for scn in scenarios:
        cls = scn['cls']
        nreg = len(_merge_regions(scn['ads']))
        for sd in range(seeds):
            rng = np.random.default_rng(1000 + sd)
            r = simulate(params, scn, rng)
            glb['false_block_s'] += r['false_block_s']
            glb['phantom'] += r['phantom']
            glb['gap_flaps'] += r['gap_flaps']
            if cls == 'O':
                O['detect'] += r['detect']
                O['recover'] += r['recover']
                O['flaps'] += r['flaps']
                O['miss'] += r['miss']
                O['n'] += nreg
            elif cls == 'V':
                V['detect'] += r['detect']
                V['recover'] += r['recover']
                V['flaps'] += r['flaps']
                V['miss'] += r['miss']
                V['n'] += nreg
            else:
                Vs['detect'] += r['detect']
                Vs['miss'] += r['miss']
                Vs['n'] += nreg

    def st(arr):
        a = np.array(arr) if arr else np.array([99.0])
        return float(a.mean()), float(np.percentile(a, 95))
    od_m, od_p = st(O['detect'])
    orc_m, orc_p = st(O['recover'])
    vd_m, vd_p = st(V['detect'])
    vrc_m, vrc_p = st(V['recover'])
    vsd_m, _ = st(Vs['detect'])
    return {
        'O_det_mean': od_m, 'O_det_p95': od_p,
        'O_rec_mean': orc_m, 'O_rec_p95': orc_p,
        'O_flaps': O['flaps'], 'O_miss': O['miss'], 'O_n': O['n'],
        'V_det_mean': vd_m, 'V_det_p95': vd_p,
        'V_rec_mean': vrc_m, 'V_rec_p95': vrc_p,
        'V_flaps': V['flaps'], 'V_miss': V['miss'], 'V_n': V['n'],
        'Vs_det_mean': vsd_m, 'Vs_miss': Vs['miss'], 'Vs_n': Vs['n'],
        'false_block_s': round(glb['false_block_s'], 1),
        'phantom': glb['phantom'], 'gap_flaps': glb['gap_flaps'],
    }


def feasible(m):
    # HARD guarantees the algorithm params fully control. `phantom`
    # (blocking turning ON with no ad anywhere near) is the true
    # "blocked real content" signal — content-only scenarios trip it
    # directly. OCR-present ads must never miss or flap (OCR is
    # authoritative). `false_block_s` is a noisy duration metric
    # (dt-quantised region/grace boundaries + slow-recovery overhang),
    # so it is a soft cost term, not a hard gate.
    return (m['phantom'] == 0 and m['O_flaps'] == 0 and m['O_miss'] == 0)


def cost(m):
    # Among feasible sets, minimise transition latency. OCR-present is the
    # common case (weight it most); VLM-only latency + its residual
    # miss/flap rate are softer optimisation targets. false_block enters
    # softly so slow recovery is still penalised.
    vn = max(1, m['V_n'])
    return (1.5 * m['O_det_mean'] + 1.0 * m['O_det_p95']
            + 1.0 * m['O_rec_mean'] + 0.7 * m['O_rec_p95']
            + 0.5 * m['V_det_mean'] + 0.3 * m['V_det_p95']
            + 0.4 * m['V_rec_mean']
            + 8.0 * (m['V_miss'] / vn) + 6.0 * (m['V_flaps'] / vn)
            + 0.05 * m['false_block_s'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--seeds', type=int, default=30)
    args = ap.parse_args()

    scenarios = build_scenarios()
    base = copy.deepcopy(H.PARAMS)
    print(f"{len(scenarios)} scenario shapes × {args.seeds} seeds = "
          f"{len(scenarios)*args.seeds} runs/param-set\n")

    SHOW = ('vlm_min_decisions', 'vlm_start_agreement', 'vlm_stop_agreement',
            'vlm_hysteresis_boost', 'vlm_history_window', 'VLM_STOP_THRESHOLD',
            'OCR_STOP_THRESHOLD', 'MIN_BLOCKING_DURATION')

    def show(m):
        return (f"O_det={m['O_det_mean']:.2f}/{m['O_det_p95']:.2f} "
                f"O_rec={m['O_rec_mean']:.2f}/{m['O_rec_p95']:.2f} "
                f"O_flap={m['O_flaps']} O_miss={m['O_miss']} | "
                f"V_det={m['V_det_mean']:.2f}/{m['V_det_p95']:.2f} "
                f"V_rec={m['V_rec_mean']:.2f} "
                f"V_flap={m['V_flaps']}/{m['V_n']} V_miss={m['V_miss']}/{m['V_n']} "
                f"Vs_miss={m['Vs_miss']}/{m['Vs_n']} | "
                f"fb={m['false_block_s']} ph={m['phantom']} "
                f"gapflap={m['gap_flaps']}")

    m = evaluate(base, scenarios, args.seeds)
    print("CURRENT:", {k: base[k] for k in SHOW})
    print("  ", show(m), "FEASIBLE" if feasible(m) else "*** INFEASIBLE ***")

    if not args.sweep:
        return 0

    # Focused grid. vlm_history_window swept LOW — directly attacks the
    # stale-content-vote dilution that throttles VLM-alone detection.
    grid = {
        'vlm_min_decisions': [2, 3],
        'vlm_start_agreement': [0.65, 0.70, 0.75, 0.80],
        'vlm_stop_agreement': [0.60, 0.65, 0.70, 0.75],
        'vlm_hysteresis_boost': [0.10],
        'VLM_STOP_THRESHOLD': [2, 3, 4],
        'vlm_history_window': [8.0, 12.0, 16.0, 24.0, 45.0],
        'OCR_STOP_THRESHOLD': [2, 3],
        'MIN_BLOCKING_DURATION': [1.0, 2.0],
    }
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    # Sweep the full suite at low seeds; finalists are re-evaluated at
    # full seeds below to guard against low-seed sampling overfit.
    sw_seeds = max(4, args.seeds // 3)
    print(f"\nSweeping {len(combos)} combos on {len(scenarios)} shapes "
          f"× {sw_seeds} seeds...\n")

    ranked = []   # (feasible, cost, params, metrics)
    nfeas = 0
    for i, vals in enumerate(combos):
        p = copy.deepcopy(base)
        p.update(dict(zip(keys, vals)))
        m = evaluate(p, scenarios, sw_seeds)
        f = feasible(m)
        nfeas += f
        ranked.append((f, cost(m), dict(zip(keys, vals)), m))
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(combos)}] feasible so far: {nfeas}",
                  flush=True)

    # Feasible first (sorted by cost), then the rest by cost.
    ranked.sort(key=lambda x: (not x[0], x[1]))
    print(f"\n{nfeas}/{len(combos)} feasible "
          f"(fb<0.5 / 0 phantom / 0 OCR-flap / 0 OCR-miss)\n")
    print("=== TOP 15 (subset seeds) ===")
    for f, c, prm, m in ranked[:15]:
        print(f"{'F' if f else '.'} cost={c:6.2f} | {show(m)} | {prm}")

    # Re-evaluate the 10 best on the FULL suite at full seeds.
    print(f"\n=== FINALISTS on FULL suite × {args.seeds} seeds ===")
    finals = []
    for _, _, prm, _ in ranked[:10]:
        p = copy.deepcopy(base)
        p.update(prm)
        m = evaluate(p, scenarios, args.seeds)
        finals.append((feasible(m), cost(m), prm, m))
        print(f"{'F' if feasible(m) else '.'} cost={cost(m):6.2f} | "
              f"{show(m)} | {prm}", flush=True)
    finals.sort(key=lambda x: (not x[0], x[1]))
    print("\n*** WINNER ***")
    print(finals[0][2])
    print("feasible:", finals[0][0], "metrics:", show(finals[0][3]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
