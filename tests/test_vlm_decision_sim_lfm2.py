"""
Monte-Carlo decision-engine sim using LFM2.5-VL holdout scores.

Re-uses every scenario / DecisionEngine / scoring / sweep helper from
`test_vlm_decision_sim.py` (which was built for FastVLM-0.5B iter4) and
monkey-patches the per-frame VLM model: instead of bootstrapping from
iter4's `threshold_sweep_raw.json` at threshold 0.76, it bootstraps from
the LFM2 holdout dump at `~/axera_models/nontrained_test_data/
eval_results_fused_rmsfp32.json` (800 images; write it with the installed
model's `eval_fused.py`: minus-v0.2 measures 99.1% accuracy / 98.5% non-ad /
99.8% ad on the NPU, minus-v0.1 95.6% / 97.5% / 93.8%; set MINUS_LFM2_HOLDOUT
to use a results file somewhere else) using `p_yes_norm = sigmoid(yes_logit - no_logit)`
and the argmax decision rule (threshold = 0.5).

Run:
  python3 tests/test_vlm_decision_sim_lfm2.py            # eval current PARAMS on LFM2
  python3 tests/test_vlm_decision_sim_lfm2.py --sweep    # full param sweep
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

import block_latency_harness as H        # noqa: E402
import test_vlm_decision_sim as S        # noqa: E402

LFM2_HOLDOUT = os.environ.get(
    'MINUS_LFM2_HOLDOUT',
    '/home/radxa/axera_models/nontrained_test_data/eval_results_fused_rmsfp32.json')


def _load_lfm2_scores():
    raw = json.load(open(LFM2_HOLDOUT))
    res = raw['results']
    ads = []
    non = []
    for r in res:
        y = float(r['p_yes_logit'])
        n = float(r['p_no_logit'])
        # softmax-normalized P(yes) in the {yes,no} subspace
        # = 1 / (1 + exp(n - y)) (numerically stable)
        diff = y - n
        if diff >= 0:
            p_yes_norm = 1.0 / (1.0 + np.exp(-diff))
        else:
            ex = np.exp(diff)
            p_yes_norm = ex / (1.0 + ex)
        if r['true_label'] == 'ad':
            ads.append(p_yes_norm)
        else:
            non.append(p_yes_norm)
    return np.array(ads), np.array(non)


# ---- monkey-patch the iter4-pinned globals in S ----
S.AD_SCORES, S.NON_SCORES = _load_lfm2_scores()
S.AD_THRESHOLD = 0.5   # LFM2 uses argmax (== threshold 0.5 on p_yes_norm)

print(f"[LFM2 sim] holdout: {len(S.AD_SCORES)} ads, {len(S.NON_SCORES)} non-ads")
print(f"  ad p_yes_norm: p5={np.percentile(S.AD_SCORES, 5):.3f} "
      f"p50={np.percentile(S.AD_SCORES, 50):.3f} "
      f"p95={np.percentile(S.AD_SCORES, 95):.3f}")
print(f"  non p_yes_norm: p5={np.percentile(S.NON_SCORES, 5):.3f} "
      f"p50={np.percentile(S.NON_SCORES, 50):.3f} "
      f"p95={np.percentile(S.NON_SCORES, 95):.3f}")
# These reproduce the holdout accuracy/recall under the argmax decision:
ad_recall = float((S.AD_SCORES > S.AD_THRESHOLD).mean())
non_recall = float((S.NON_SCORES <= S.AD_THRESHOLD).mean())
print(f"  derived ad-recall={ad_recall*100:.1f}% non-ad-recall={non_recall*100:.1f}% "
      f"(holdout reports 94.75% / 99.25%)\n")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--seeds', type=int, default=30)
    args = ap.parse_args()

    scenarios = S.build_scenarios()
    base = copy.deepcopy(H.PARAMS)
    print(f"{len(scenarios)} scenario shapes × {args.seeds} seeds = "
          f"{len(scenarios)*args.seeds} runs/param-set\n")

    SHOW = ('vlm_min_decisions', 'vlm_start_agreement', 'vlm_stop_agreement',
            'vlm_hysteresis_boost', 'vlm_history_window', 'VLM_STOP_THRESHOLD',
            'OCR_STOP_THRESHOLD', 'MIN_BLOCKING_DURATION')

    m = S.evaluate(base, scenarios, args.seeds)
    print("CURRENT (LFM2):", {k: base[k] for k in SHOW})
    print("  ", show(m), "FEASIBLE" if S.feasible(m) else "*** INFEASIBLE ***")

    if not args.sweep:
        return 0

    # Same grid as iter4 sim. LFM2's confident-case distribution is
    # sharper (clean p_yes ~0.001-0.01, ad p_yes ~0.97-0.99), so the
    # win-region for `vlm_min_decisions`/`vlm_start_agreement` is likely
    # broader and shifted toward less-conservative.
    grid = {
        'vlm_min_decisions': [2, 3, 4, 5],
        'vlm_start_agreement': [0.60, 0.65, 0.70, 0.75, 0.80],
        'vlm_stop_agreement': [0.60, 0.65, 0.70, 0.75],
        'vlm_hysteresis_boost': [0.10],
        'VLM_STOP_THRESHOLD': [2, 3],
        'vlm_history_window': [6.0, 8.0, 12.0, 16.0],
        'OCR_STOP_THRESHOLD': [2, 3],
        'MIN_BLOCKING_DURATION': [1.0, 2.0],
    }
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    sw_seeds = max(4, args.seeds // 3)
    print(f"\nSweeping {len(combos)} combos on {len(scenarios)} shapes "
          f"× {sw_seeds} seeds...\n")

    ranked = []
    nfeas = 0
    for i, vals in enumerate(combos):
        p = copy.deepcopy(base)
        p.update(dict(zip(keys, vals)))
        m = S.evaluate(p, scenarios, sw_seeds)
        f = S.feasible(m)
        nfeas += f
        ranked.append((f, S.cost(m), dict(zip(keys, vals)), m))
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(combos)}] feasible so far: {nfeas}",
                  flush=True)

    ranked.sort(key=lambda x: (not x[0], x[1]))
    print(f"\n{nfeas}/{len(combos)} feasible "
          f"(0 phantom / 0 OCR-flap / 0 OCR-miss)\n")
    print("=== TOP 15 (subset seeds) ===")
    for f, c, prm, m in ranked[:15]:
        print(f"{'F' if f else '.'} cost={c:6.2f} | {show(m)} | {prm}")

    print(f"\n=== FINALISTS on FULL suite × {args.seeds} seeds ===")
    finals = []
    for _, _, prm, _ in ranked[:10]:
        p = copy.deepcopy(base)
        p.update(prm)
        m = S.evaluate(p, scenarios, args.seeds)
        finals.append((S.feasible(m), S.cost(m), prm, m))
        print(f"{'F' if S.feasible(m) else '.'} cost={S.cost(m):6.2f} | "
              f"{show(m)} | {prm}", flush=True)
    finals.sort(key=lambda x: (not x[0], x[1]))
    print("\n*** WINNER ***")
    print(finals[0][2])
    print("feasible:", finals[0][0], "metrics:", show(finals[0][3]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
