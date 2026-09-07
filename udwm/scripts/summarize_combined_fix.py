# -*- coding: utf-8 -*-
"""Adjudication summary for the combined-fix N-study (registered 2026-09-07).

Registration: research/COMBINED-FIX-PREREGISTRATION-2026-09-07.md
(incl. Amendment 1: fresh full 7-arm table). Primary endpoint: u_rank_corr.

Bars (N=10, repo conventions: wins >= 7/10 AND 5,000-draw bootstrap 95% CI
excludes 0):
  1. paired u_rank_corr lagged_identified_eq - lagged_hybrid (>=, confirmed);
  2. paired u_rank_corr lagged_identified_eq - identified_eq (>=, confirmed);
  3. lagged_identified_eq u_rank >= 0.70 on >= 7/10 seeds.
Reference contrasts vs ordinary / hybrid / identified_hybrid are reported
without a bar.
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
    ("lagged_identified_eq", "lagged_hybrid"),   # bar 1 (adds on top of lag?)
    ("lagged_identified_eq", "identified_eq"),    # bar 2 (adds on top of eq?)
    ("lagged_identified_eq", "ordinary"),          # parity reference
    ("lagged_identified_eq", "hybrid"),            # vs naive live
    ("lagged_identified_eq", "identified_hybrid"), # vs EMA control
    ("lagged_identified", "lagged_identified_eq"), # historical broken combo vs combined fix (if present)
]

RNG = np.random.default_rng(20260907)


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
    p.add_argument("--data", default="runs/policy_combined_fix_10seed.json")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    args = p.parse_args(argv)
    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = payload["rows"]
    seeds = args.seeds or sorted({int(r["seed"]) for r in rows})
    rows = [r for r in rows if int(r["seed"]) in seeds]
    print(f"file: {args.data}  rows: {len(rows)}  seeds: {len(seeds)}")
    pairing = payload.get("teacher_pairing", {})
    for s in seeds:
        p_info = pairing.get(str(s), {})
        print(f"  seed {s}: n_arms={p_info.get('n_arms')} exact={p_info.get('exact_teacher_match')} gap={p_info.get('max_teacher_checksum_gap')}")

    arms = sorted({r["variant"] for r in rows})
    print("\nper-arm means (n=%d):" % len(seeds))
    print(f"{'arm':<22}" + "".join(f"{e:>16}" for e in ENDPOINTS))
    for arm in arms:
        rs = [r for r in rows if r["variant"] == arm]
        line = f"{arm:<22}"
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

    lid = [r for r in rows if r["variant"] == "lagged_identified_eq"]
    lu = np.asarray([r.get("u_rank_corr", np.nan) for r in lid], dtype=float)
    lu = lu[np.isfinite(lu)]
    print("\nregistration bars:")
    print(f"  bar 1: lagged_identified_eq - lagged_hybrid u_rank  (see contrast above)")
    print(f"  bar 2: lagged_identified_eq - identified_eq u_rank  (see contrast above)")
    print(f"  bar 3: lagged_identified_eq u_rank >= 0.70 on {int((lu >= 0.70).sum())}/{lu.size} "
          f"(need >= 7/10); mean {lu.mean():.3f}")


if __name__ == "__main__":
    main()