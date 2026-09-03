"""Paired-bootstrap summary for the registered policy 2x2 (10-seed rerun).

Adjudicates the pre-registered decision tree
(research/DECISION-TREE-2X2-PREREGISTRATION.md) on the primary endpoint
``u_rank_corr`` and reports the secondary endpoints with the repo's
conventions: mean paired delta, wins/N, and a 5,000-draw percentile
bootstrap 95% CI over seeds.

Verdicts: "confirmed" = CI excludes 0 and >= 7/10 seeds in the predicted
direction; "absent" = the opposite is confirmed; "inconclusive" otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTRASTS = [
    ("hybrid", "ordinary"),                     # reproduction of the published collapse
    ("lagged_hybrid", "hybrid"),                # does lagging fix the u-rank collapse?
    ("identified_hybrid", "hybrid"),            # identified target under the LIVE critic
    ("lagged_identified", "hybrid"),            # combined
    ("lagged_identified", "identified_hybrid"), # does lag add anything once identified?
]

ENDPOINTS = [
    "u_rank_corr", "w_rank_corr", "u_rmse", "w_rmse",
    "next_state_mse", "final_return", "selective_rank_corr", "selective_recall_bad",
]

# Positive delta is "better" for these; negative delta is "better" for the rest.
HIGHER_IS_BETTER = {"u_rank_corr", "w_rank_corr", "selective_rank_corr", "selective_recall_bad", "final_return"}

RNG = np.random.default_rng(20260903)


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def paired_delta(rows, arm_a, arm_b, key):
    a = {int(r["seed"]): r.get(key) for r in rows if r["variant"] == arm_a}
    b = {int(r["seed"]): r.get(key) for r in rows if r["variant"] == arm_b}
    seeds = sorted(set(a) & set(b))
    vals = np.asarray([float(a[s]) - float(b[s]) for s in seeds if a[s] is not None and b[s] is not None])
    return seeds, vals


def bootstrap_ci(deltas, n_draws=5000):
    if deltas.size == 0:
        return None, None
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = RNG.choice(deltas, size=deltas.size, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="runs/policy_identifiability_2x2_10seed.json")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    data = load_rows(Path(args.data))
    rows = data["rows"]
    pairing = data.get("teacher_pairing", {})
    seeds = args.seeds or sorted({int(r["seed"]) for r in rows})
    rows = [r for r in rows if int(r["seed"]) in seeds]

    print(f"file: {args.data}  rows: {len(rows)}  seeds: {len(seeds)}")
    for s in seeds:
        p = pairing.get(str(s), {})
        print(f"  seed {s}: n_arms={p.get('n_arms')} exact={p.get('exact_teacher_match')} gap={p.get('max_teacher_checksum_gap')}")

    # Per-arm means (10 seeds).
    arms = sorted({r["variant"] for r in rows})
    print("\nper-arm means (n=%d):" % len(seeds))
    hdr = f"{'arm':<20}" + "".join(f"{e:>16}" for e in ENDPOINTS)
    print(hdr)
    for arm in arms:
        rs = [r for r in rows if r["variant"] == arm]
        line = f"{arm:<20}"
        for e in ENDPOINTS:
            v = np.asarray([r.get(e, np.nan) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            line += f"{v.mean():>16.4f}" if v.size else f"{'-':>16}"
        print(line)

    # Paired contrasts.
    print("\npaired contrasts (mean delta, wins/N, bootstrap 95%):")
    for a, b in CONTRASTS:
        print(f"\n{a} - {b}")
        for e in ENDPOINTS:
            seeds_used, d = paired_delta(rows, a, b, e)
            if d.size < 2:
                print(f"  {e:<22} n<2 -> skip")
                continue
            lo, hi = bootstrap_ci(d, args.n_bootstrap)
            wins = int((d > 0).sum())
            better = ">" if e in HIGHER_IS_BETTER else "<"
            good = (d > 0).mean() if e in HIGHER_IS_BETTER else (d < 0).mean()
            verdict = "inconclusive"
            excludes_zero = (lo > 0) if e in HIGHER_IS_BETTER else (hi < 0)
            if d.size >= 7 and (wins >= 7 or wins <= 3):
                if (wins >= 7 and e in HIGHER_IS_BETTER) or (wins <= 3 and e not in HIGHER_IS_BETTER):
                    verdict = "confirmed" if excludes_zero else "inconclusive"
                elif excludes_zero:
                    verdict = "absent"
            print(f"  {e:<22} mean={d.mean():+12.4f}  wins={wins:>2}/{d.size:<2} 95%[{lo:+10.4f},{hi:+10.4f}]  -> {verdict}")


if __name__ == "__main__":
    main()
