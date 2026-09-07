# -*- coding: utf-8 -*-
"""Adjudication summary for the G9 normalization-controlled re-adjudication.

Registration: research/NORMALIZATION-CONTROL-READJUDICATION-PREREGISTRATION-
2026-09-07.md. Runs the combined-fix arm-to-term mapping with in-file
baselines plus the four G7 normalization cells in one file.

Bars (N=10; wins >= 7/10 AND 20,000-draw percentile bootstrap 95% CI
excludes 0, fixed RNG seed 20260907):
  1. within-cell lag null replication: D - A and C - B on u_rank;
  2. identified_eq_norm - identified_eq u_rank (standardization on live);
  3. identified_eq_norm u_rank >= 0.70 on >= 7/10 seeds;
  4. identified_eq - hybrid u_rank (file-level mechanism sanity).
w_rmse is NOT adjudicated (G7 verdict: rank-only); printed for record only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ENDPOINTS = [
    "u_rank_corr", "w_rank_corr", "u_rmse", "w_rmse",
    "next_state_mse", "final_return", "selective_rank_corr", "selective_recall_bad",
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


def bootstrap_ci(deltas, n_draws=20000):
    if deltas.size == 0:
        return None, None
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = RNG.choice(deltas, size=deltas.size, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/normalization_readjudication_10seed.json")
    p.add_argument("--n-bootstrap", type=int, default=20000)
    args = p.parse_args(argv)
    payload = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = payload["rows"]
    seeds = sorted({int(r["seed"]) for r in rows})
    print(f"file: {args.data}  rows: {len(rows)}  seeds: {len(seeds)}")
    pairing = payload.get("teacher_pairing", {})
    for s in seeds:
        p_info = pairing.get(str(s), {})
        print(f"  seed {s}: n_arms={p_info.get('n_arms')} exact={p_info.get('exact_teacher_match')} gap={p_info.get('max_teacher_checksum_gap')}")

    arms = sorted({r["variant"] for r in rows})
    print("\nper-arm means (n=%d):" % len(seeds))
    print(f"{'arm':<28}" + "".join(f"{e:>13}" for e in ENDPOINTS))
    for arm in arms:
        rs = [r for r in rows if r["variant"] == arm]
        line = f"{arm:<28}"
        for e in ENDPOINTS:
            v = np.asarray([r.get(e, np.nan) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            line += f"{v.mean():>13.4f}" if v.size else f"{'-':>13}"
        print(line)

    def contrast(a, b, key, desc=""):
        if a not in arms or b not in arms:
            print(f"  {a} - {b}: missing arm -> skip")
            return None
        _, d = paired_delta(rows, a, b, key)
        lo, hi = bootstrap_ci(d, args.n_bootstrap)
        wins = int((d > 0).sum())
        print(f"  {a} - {b}  {key:<16} mean={d.mean():+10.4f}  up={wins:>2}/{d.size:<2} 95%[{lo:+9.4f},{hi:+9.4f}]  {desc}")
        return d, lo, hi, wins

    print("\nBar 1: within-cell lag null replication on u_rank (C - B, D - A):")
    for label, (a, b) in {"D - A": ("lagged_identified_eq_nonorm", "identified_eq"),
                          "C - B": ("lagged_identified_eq", "identified_eq_norm")}.items():
        contrast(a, b, "u_rank_corr", f"({label})")

    print("\nBar 2: standardization on the live critic, u_rank (identified_eq_norm - identified_eq):")
    contrast("identified_eq_norm", "identified_eq", "u_rank_corr", "(bar 2)")

    print("\nBar 3: identified_eq_norm u_rank >= 0.70 per seed:")
    eqn = np.asarray([r.get("u_rank_corr", np.nan) for r in rows
                      if r["variant"] == "identified_eq_norm"], dtype=float)
    eqn = eqn[np.isfinite(eqn)]
    print(f"  {int((eqn >= 0.70).sum())}/{eqn.size} seeds >= 0.70; mean {eqn.mean():.4f}")

    print("\nBar 4: file-level mechanism sanity (identified_eq - hybrid, u_rank):")
    contrast("identified_eq", "hybrid", "u_rank_corr", "(bar 4)")

    print("\nReference contrasts (no bar; ceiling-bound or noisy):")
    for a, b in [("identified_eq_norm", "ordinary"),
                 ("identified_eq_norm", "lagged_hybrid"),
                 ("lagged_identified_eq", "lagged_hybrid")]:
        for e in ["u_rank_corr", "u_rmse", "next_state_mse", "final_return"]:
            contrast(a, b, e)


if __name__ == "__main__":
    main()
