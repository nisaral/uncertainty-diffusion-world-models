# -*- coding: utf-8 -*-
"""Additive effect-size + power/MDE report over an existing adjudicated run file.

Registration context: research/MASTER-PLAN-V3-EXECUTION-2026-09-09.md item 5a.
This is an ADDITIVE report over already-adjudicated rows: it never changes a
verdict, it only adds (i) the paired standardized effect (d_z = mean / SD of
the paired deltas) next to the raw delta + bootstrap CI + wins/N that the
existing summarizers already print, and (ii) a minimum-detectable-effect
(MDE) estimate for a *planned* study at a given seed count, computed from the
paired-delta SD observed on the chosen endpoint of this file.

Default contrast list mirrors the registered DMC 30-seed payoff contrasts so
the output lines up with summarize_dmc_payoff.py; pass --pairs to use the
tool on any other adjudicated file (e.g. the DelayedBimodal combined-fix file
or a future CRN-bias probe file).

MDE formula (paired design, two-sided, alpha=0.05, power 0.8):
    MDE = (z_{1-alpha/2} + z_power) * sd_delta / sqrt(n)
where sd_delta is the observed SD of the paired per-seed deltas on this
file's endpoint (the same variance a similarly scaled new endpoint would
have). z values come from the standard normal quantiles (no t correction is
applied; at n=10 the t vs z difference is a few percent and is called out in
the per-row output).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Registered headline contrasts of the DMC 30-seed payoff study (u_rank).
DEFAULT_PAIRS = [
    ("identified_eq", "ordinary", "eq - ordinary"),
    ("identified_eq", "lagged_hybrid", "eq - lagged_hybrid"),
    ("identified_eq", "hybrid", "eq - hybrid"),
    ("identified_eq", "identified_hybrid", "eq - EMA"),
    ("lagged_identified_eq", "ordinary", "lagged_eq - ordinary"),
    ("lagged_identified_eq", "hybrid", "lagged_eq - hybrid"),
    ("lagged_identified_eq", "identified_eq", "lagged_eq - eq"),
    ("lagged_hybrid", "hybrid", "lagged_hybrid - hybrid"),
    ("lagged_hybrid", "ordinary", "lagged_hybrid - ordinary"),
    ("hybrid", "ordinary", "hybrid - ordinary"),
    ("ordinary", "identified_hybrid", "ordinary - EMA"),
    ("hybrid", "identified_hybrid", "hybrid - EMA"),
]


def load_rows(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d["rows"]


def by_seed(rows, arm):
    return {int(r["seed"]): r for r in rows if r["variant"] == arm}


def paired_deltas(rows_a, rows_b, key):
    seeds = sorted(set(rows_a) & set(rows_b))
    vals = np.asarray([float(rows_a[s][key]) - float(rows_b[s][key]) for s in seeds])
    return seeds, vals


def z_quantile(p: float) -> float:
    """Standard-normal quantile via bisection on Phi(z) (no scipy needed)."""
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mde(sd_delta: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    return (z_quantile(1.0 - alpha / 2.0) + z_quantile(power)) * sd_delta / math.sqrt(max(n, 1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="runs/dmc_payoff_30seed_15k_gpu.json")
    p.add_argument("--key", default="u_rank_corr",
                   help="row field the contrasts are read on (default u_rank_corr)")
    p.add_argument("--pairs", default=None, nargs="*",
                   help="contrasts as 'armA,armB[,label]'; default = registered DMC list")
    p.add_argument("--n-bootstrap", type=int, default=100000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--planned-n", type=int, nargs="+", default=[10, 30])
    args = p.parse_args(argv)

    rows = load_rows((ROOT / args.data).resolve())
    rng = np.random.default_rng(args.rng_seed)
    arms = {a: by_seed(rows, a) for a in sorted({r["variant"] for r in rows})}
    seeds = sorted(set().union(*[set(a) for a in arms.values()])) if arms else []

    pairs = []
    if args.pairs:
        for raw in args.pairs:
            parts = [x.strip() for x in raw.split(",")]
            if len(parts) < 2:
                raise SystemExit(f"--pairs item must be 'armA,armB[,label]', got {raw!r}")
            label = parts[2] if len(parts) > 2 else f"{parts[0]} - {parts[1]}"
            pairs.append((parts[0], parts[1], label))
    else:
        pairs = DEFAULT_PAIRS

    za = z_quantile(1.0 - args.alpha / 2.0)
    zp = z_quantile(args.power)
    print(f"rows={len(rows)} seeds={len(seeds)} key={args.key} "
          f"alpha={args.alpha} power={args.power}")
    print(f"paired-design MDE = ({za:.2f} + {zp:.2f}) * sd_delta / sqrt(n) "
          f"(z-based, no t correction)\n")
    print(f"{'contrast':<46}{'mean':>8}{'d_z':>7}{'CI_lo':>9}{'CI_hi':>9}"
          f"{'wins':>7}{'sd_delta':>9}")
    out = {}
    for a, b, label in pairs:
        if a not in arms or b not in arms:
            print(f"[skip] {label}: missing arm(s) in {args.data}")
            continue
        _, deltas = paired_deltas(arms[a], arms[b], args.key)
        n = deltas.size
        sd = float(deltas.std(ddof=1)) if n > 1 else 0.0
        means = rng.choice(deltas, size=(args.n_bootstrap, n), replace=True).mean(axis=1)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        dz = float(deltas.mean() / sd) if sd > 0 else float("nan")
        wins = int(np.sum(deltas > 0))
        print(f"{label:<46}{deltas.mean():>+8.3f}{dz:>7.2f}{lo:>+9.3f}{hi:>+9.3f}"
              f"{wins:>5}/{n:<2}{sd:>9.4f}")
        out[label] = {"mean": float(deltas.mean()), "d_z": dz, "ci": [lo, hi],
                      "wins": wins, "n": n, "sd_delta": sd}

    print("\n== MDE at planned seed counts (same endpoint variance) ==")
    print(f"{'contrast':<46}" + "".join(f"n={nn:<12}" for nn in args.planned_n))
    for label, rec in out.items():
        cells = "".join(f"{mde(rec['sd_delta'], nn, args.alpha, args.power):<12.4f}"
                        for nn in args.planned_n)
        print(f"{label:<46}{cells}")

    # Reference variances for planning NEW similarly scaled endpoints: the
    # per-arm cross-seed SD of the endpoint (what a one-arm descriptive study
    # would use) plus the pooled within-arm SD.
    print("\n== endpoint reference variance (per-arm cross-seed) ==")
    for arm in sorted(arms):
        vals = np.asarray([r[args.key] for r in arms[arm].values()], dtype=float)
        if vals.size == 0:
            continue
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        print(f"{arm:<24} n={vals.size:<3} mean={vals.mean():+.4f} sd={sd:.4f}")


if __name__ == "__main__":
    main()
