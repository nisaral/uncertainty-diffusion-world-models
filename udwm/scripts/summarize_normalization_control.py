# -*- coding: utf-8 -*-
"""Adjudication summary for the G7 normalization-controlled w-channel contrast.

Registration: research/NORMALIZATION-CONTROL-PREREGISTRATION-2026-09-07.md.
Cells (all DelayedBimodal, M=2 identified, equal weights, no EMA):
  A identified_eq                (live critic,  normalize_values false)
  B identified_eq_norm           (live critic,  normalize_values true)
  C lagged_identified_eq         (target critic, normalize_values true, guard/warmup)
  D lagged_identified_eq_nonorm  (target critic, normalize_values false, guard/warmup)

Primary (pre-registered bars): paired w_rmse contrast D - A (both
unnormalized) and C - B (both normalized). If BOTH exclude 0 in the
lagged-better (negative) direction with >= 7/10 wins: lagging reduces w_rmse
independent of normalization -> the w claim is a controlled mechanism claim.
If NEITHER excludes 0: normalization drove the 0.92 -> 0.28 gap; the w claim
is dropped to rank endpoints only. If exactly one excludes 0: interaction,
w claim conditional.

Secondary (never adjudicative): normalization effect within lag cell
(B - A, D - C) and u_rank_corr for all four cells.

Stats: paired deltas over shared seeds, 20,000-draw percentile bootstrap 95%
CI with fixed RNG seed, wins/N (repo convention).
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

CELLS = {
    "A": "identified_eq",
    "B": "identified_eq_norm",
    "C": "lagged_identified_eq",
    "D": "lagged_identified_eq_nonorm",
}

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
    p.add_argument("--data", default="runs/normalization_control_10seed.json")
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
    missing = [name for name in CELLS.values() if name not in arms]
    if missing:
        print("!! missing cells:", missing)
    print("\nper-arm means (n=%d):" % len(seeds))
    print(f"{'cell':<4}{'arm':<26}" + "".join(f"{e:>14}" for e in ENDPOINTS))
    for label, arm in CELLS.items():
        if arm not in arms:
            continue
        rs = [r for r in rows if r["variant"] == arm]
        line = f"{label:<4}{arm:<26}"
        for e in ENDPOINTS:
            v = np.asarray([r.get(e, np.nan) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            line += f"{v.mean():>14.4f}" if v.size else f"{'-':>14}"
        print(line)

    print("\nPrimary cell-pair contrasts on w_rmse (lagged - live within a normalization cell; negative = lagged better):")
    for label, (a, b) in {
        "D - A (both unnormalized)": ("D", "A"),
        "C - B (both normalized)": ("C", "B"),
    }.items():
        _, d = paired_delta(rows, CELLS[a], CELLS[b], "w_rmse")
        lo, hi = bootstrap_ci(d, args.n_bootstrap)
        wins = int((d < 0).sum())  # lagged arm better (lower w_rmse)
        print(f"  {label:<24} mean={d.mean():+12.4f}  lagged-better={wins:>2}/{d.size:<2} 95%[{lo:+10.4f},{hi:+10.4f}]")

    print("\nWithin-cell lag contrasts on u_rank_corr / next_state_mse (descriptive; normalization held fixed):")
    for label, (a, b) in {
        "D - A (both unnormalized)": ("D", "A"),
        "C - B (both normalized)": ("C", "B"),
    }.items():
        for e in ["u_rank_corr", "next_state_mse"]:
            _, d = paired_delta(rows, CELLS[a], CELLS[b], e)
            lo, hi = bootstrap_ci(d, args.n_bootstrap)
            wins = int((d > 0).sum())
            print(f"  {label:<24} {e:<16} mean={d.mean():+10.4f}  up={wins:>2}/{d.size:<2} 95%[{lo:+9.4f},{hi:+9.4f}]")

    print("\nSecondary contrasts (descriptive):")
    for label, (a, b) in {
        "B - A (normalization, live cell)": ("B", "A"),
        "D - C (normalization, lagged cell)": ("D", "C"),
    }.items():
        for e in ["w_rmse", "u_rank_corr", "w_rank_corr", "next_state_mse"]:
            _, d = paired_delta(rows, CELLS[a], CELLS[b], e)
            lo, hi = bootstrap_ci(d, args.n_bootstrap)
            wins = int((d > 0).sum())
            print(f"  {label:<34} {e:<16} mean={d.mean():+10.4f}  up={wins:>2}/{d.size:<2} 95%[{lo:+9.4f},{hi:+9.4f}]")

    print("\nAll four cells, u_rank_corr per seed (must not move the combined-fix u-rank bars):")
    print("seed   A          B          C          D")
    for s in seeds:
        vals = []
        for arm in CELLS.values():
            v = [r.get("u_rank_corr", np.nan) for r in rows if int(r["seed"]) == s and r["variant"] == arm]
            vals.append(float(v[0]) if v else float("nan"))
        print(f"{s:<6}" + "".join(f"{v:>10.4f} " for v in vals))


if __name__ == "__main__":
    main()
