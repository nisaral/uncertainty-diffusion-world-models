# -*- coding: utf-8 -*-
"""Adjudication summary for the corrected-weight policy N-study (2026-09-05).

Registration: research/CORRECTED-WEIGHT-POLICY-PREREGISTRATION-2026-09-05.md.
Contrasts and bars below are fixed by that registration (written before any
row ran). Primary endpoint: u_rank_corr.

Bars:
  1. paired u_rank_corr identified_eq - identified_hybrid: wins >= 7/10 AND
     5,000-draw bootstrap 95% CI excludes 0;
  2. identified_eq u_rank_corr >= 0.70 on >= 7/10 seeds.
Branches: A = both bars; B = bar 1 only; C = bar 1 not met (see registration).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ENDPOINTS = [
    "u_rank_corr", "w_rank_corr", "u_rmse", "w_rmse",
    "next_state_mse", "final_return", "selective_rank_corr", "selective_recall_bad",
]
HIGHER_IS_BETTER = {"u_rank_corr", "w_rank_corr", "selective_rank_corr", "selective_recall_bad", "final_return"}

CONTRASTS = [
    ("identified_eq", "identified_hybrid"),   # bar 1 (primary mechanism contrast)
    ("identified_eq", "ordinary"),
    ("identified_eq", "hybrid"),
    ("identified_eq", "lagged_hybrid"),
    ("hybrid", "ordinary"),                    # reproducibility of the N=30 hybrid dip
    ("identified_wonly", "identified_eq"),
]

RNG = np.random.default_rng(20260905)


def load_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["rows"]


def paired_delta(rows, arm_a, arm_b, key):
    a = {int(r["seed"]): r.get(key) for r in rows if r["variant"] == arm_a}
    b = {int(r["seed"]): r.get(key) for r in rows if r["variant"] == arm_b}
    seeds = sorted(set(a) & set(b))
    vals = np.asarray([float(a[s]) - float(b[s]) for s in seeds
                       if a[s] is not None and b[s] is not None])
    return seeds, vals


def bootstrap_ci(deltas, n_draws=5000):
    if deltas.size == 0:
        return None, None
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = RNG.choice(deltas, size=deltas.size, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/policy_corrected_weights_10seed.json")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)
    rows = load_rows(args.data)
    seeds = args.seeds or sorted({int(r["seed"]) for r in rows})
    rows = [r for r in rows if int(r["seed"]) in seeds]
    print(f"file: {args.data}  rows: {len(rows)}  seeds: {len(seeds)}")

    arms = sorted({r["variant"] for r in rows})
    print("\nper-arm means (n=%d):" % len(seeds))
    print(f"{'arm':<20}" + "".join(f"{e:>16}" for e in ENDPOINTS))
    for arm in arms:
        rs = [r for r in rows if r["variant"] == arm]
        line = f"{arm:<20}"
        for e in ENDPOINTS:
            v = np.asarray([r.get(e, np.nan) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            line += f"{v.mean():>16.4f}" if v.size else f"{'-':>16}"
        print(line)

    print("\npaired contrasts (mean delta, wins/N, bootstrap 95%):")
    for a, b in CONTRASTS:
        if a not in arms or b not in arms:
            continue
        print(f"\n{a} - {b}")
        for e in ENDPOINTS:
            seeds_used, d = paired_delta(rows, a, b, e)
            if d.size < 2:
                print(f"  {e:<22} n<2 -> skip")
                continue
            lo, hi = bootstrap_ci(d, args.n_bootstrap)
            wins = int((d > 0).sum())
            frac_up = (d > 0).mean()
            higher_better = e in HIGHER_IS_BETTER
            thr = 0.7
            verdict = "~ inconclusive"
            if d.size >= 7:
                if higher_better and lo > 0 and frac_up >= thr:
                    verdict = "+ confirmed"
                elif higher_better and hi < 0 and frac_up <= 1 - thr:
                    verdict = "- confirmed (harm)"
                elif (not higher_better) and hi < 0 and (1 - frac_up) >= thr:
                    verdict = "+ confirmed"
                elif (not higher_better) and lo > 0 and (1 - frac_up) <= 1 - thr:
                    verdict = "- confirmed (harm)"
            print(f"  {e:<22} mean={d.mean():+12.4f}  up={wins:>2}/{d.size:<2} 95%[{lo:+10.4f},{hi:+10.4f}]  -> {verdict}")

    eq = [r for r in rows if r["variant"] == "identified_eq"]
    ordi = [r for r in rows if r["variant"] == "ordinary"]
    eq_u = np.asarray([r.get("u_rank_corr", np.nan) for r in eq])
    print("\nregistration bars:")
    print(f"  bar 1: eq - identified_hybrid u_rank  (see contrast above)")
    print(f"  bar 2: eq u_rank>=0.70 on {int((eq_u >= 0.70).sum())}/{eq_u.size} "
          f"(need >= 7/10); eq mean {eq_u.mean():.3f}")
    if ordi and len(ordi) == len(eq):
        om = np.nanmean([r.get("next_state_mse", np.nan) for r in ordi])
        em = np.nanmean([r.get("next_state_mse", np.nan) for r in eq])
        print(f"  next_state_mse: eq {em:.3f} vs ordinary {om:.3f} (2x bound {2 * om:.3f})")


if __name__ == "__main__":
    main()
