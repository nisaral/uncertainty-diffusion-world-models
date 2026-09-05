# -*- coding: utf-8 -*-
"""Summary for the corruption-probe N=10 replication (RESULTS-CORRUPTION-2026-09-05).

Arms (all M=2 latents, live SAC critic, schedule corruptions unless noted):
  eq     : equal-weight identified (reweight_ema=false, aleatoric_weight=1.0), lr 1e-3
  ema    : EMA-both reweighted identified (historical config), lr 1e-3
  emalr1 : EMA-both reweighted identified, grad_clip=1e3, lr 0.1 (B2 endpoint)

Files: seeds 0-1 of eq/ema come from the registered corruption probe
(probe_u_corrupt_{schedule,ema}_s{0,1}.json); seeds 2-9 and all emalr1 rows
come from probe_u_n10_{eq,ema,emalr1}_s{seed}.json (fixed probe semantics).

Prints the per-seed table and paired bootstrap 95% CIs (5,000 percentile
draws over seeds, repo convention) for the registered comparisons:
  - u_rank eq - ema            (does removing the EMA recover the decision object?)
  - student g eq - ema         (level of the aleatoric channel)
  - student g emalr1           (headline B2 replication: g pinned at 0?)
No verdicts are issued here; adjudication follows the registration rules in
research/CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md and the result doc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

ARM_FILES = {
    "eq": {0: "probe_u_corrupt_schedule_s{seed}.json", 1: "probe_u_corrupt_schedule_s{seed}.json"},
    "ema": {0: "probe_u_corrupt_ema_s{seed}.json", 1: "probe_u_corrupt_ema_s{seed}.json"},
    "emalr1": {},
}
ARM_TMPL = {
    "eq": "probe_u_n10_eq_s{seed}.json",
    "ema": "probe_u_n10_ema_s{seed}.json",
    "emalr1": "probe_u_n10_emalr1_s{seed}.json",
}

RNG = np.random.default_rng(20260905)


def load_metric(arm: str, seed: int) -> dict:
    name = ARM_FILES[arm].get(seed) or ARM_TMPL[arm]
    path = RUNS / name.format(seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    fm = d["final_metrics"]
    ds = d.get("decomp_summary", {})
    return {
        "arm": arm,
        "seed": int(seed),
        "u_rank": float(fm.get("u_rank_corr", float("nan"))),
        "s_u": float(fm.get("student_u_mean", float("nan"))),
        "t_u": float(fm.get("teacher_u_mean", float("nan"))),
        "s_g": float(fm.get("student_g_mean", float("nan"))),
        "t_g": float(fm.get("teacher_g_mean", float("nan"))),
        "s_w": float(fm.get("student_w_mean", float("nan"))),
        "t_w": float(fm.get("teacher_w_mean", float("nan"))),
        "s_delta": float(ds.get("student", {}).get("delta_spread_mean", float("nan"))),
        "s_mse": float(fm.get("next_state_mse", float("nan"))),
    }


def bootstrap_ci(deltas: np.ndarray, n_draws: int = 5000):
    if deltas.size == 0:
        return None, None
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = RNG.choice(deltas, size=deltas.size, replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["eq", "ema", "emalr1"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--n-bootstrap", type=int, default=5000)
    args = p.parse_args(argv)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            try:
                rows.append(load_metric(arm, seed))
            except FileNotFoundError as e:
                print(f"  !! {e}")

    if rows:
        print(f'{"arm":<8}{"seed":<5}{"u_rank":>8}{"s_u":>10}{"t_u":>9}{"s_g":>9}{"t_g":>9}'
              f'{"s_w":>9}{"t_w":>9}{"s_delta":>8}{"s_mse":>8}')
        for r in rows:
            print(f'{r["arm"]:<8}{r["seed"]:<5}{r["u_rank"]:>8.3f}{r["s_u"]:>10.2f}{r["t_u"]:>9.2f}'
                  f'{r["s_g"]:>9.3f}{r["t_g"]:>9.2f}{r["s_w"]:>9.4f}{r["t_w"]:>9.4f}'
                  f'{r["s_delta"]:>8.3f}{r["s_mse"]:>8.3f}')

    by = {(r["arm"], r["seed"]): r for r in rows}

    def paired(arm_a, arm_b, key):
        vals = []
        for seed in args.seeds:
            if (arm_a, seed) in by and (arm_b, seed) in by:
                a, b = by[(arm_a, seed)], by[(arm_b, seed)]
                va, vb = a[key], b[key]
                if np.isfinite(va) and np.isfinite(vb):
                    vals.append(va - vb)
        return np.asarray(vals)

    print("\npaired contrasts (mean delta, wins/N, bootstrap 95%):")
    for arm_a, arm_b, key, label in [
        ("eq", "ema", "u_rank", "u_rank: eq - ema"),
        ("eq", "ema", "s_g", "student g: eq - ema"),
        ("eq", "ema", "s_mse", "next_state_mse: eq - ema"),
        ("ema", "emalr1", "s_g", "student g: ema - emalr1"),
    ]:
        d = paired(arm_a, arm_b, key)
        if d.size < 2:
            print(f"  {label:<36} n<2 -> skip")
            continue
        lo, hi = bootstrap_ci(d, args.n_bootstrap)
        print(f"  {label:<36} mean={d.mean():+9.4f}  up={int((d > 0).sum()):>2}/{d.size:<2}"
              f"  95%[{lo:+10.4f},{hi:+10.4f}]")

    for arm, key, label in [
        ("emalr1", "s_g", "emalr1 student g (B2 endpoint: ==0?)"),
    ]:
        vals = np.asarray([by[(arm, s)][key] for s in args.seeds if (arm, s) in by])
        if vals.size:
            print(f"\n  {label:<38} n={vals.size} mean={vals.mean():.4f} "
                  f"exact0={(vals == 0.0).sum()}/{vals.size} max={vals.max():.2e}")


if __name__ == "__main__":
    main()
