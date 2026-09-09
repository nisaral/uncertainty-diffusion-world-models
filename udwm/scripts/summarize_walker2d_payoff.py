# -*- coding: utf-8 -*-
"""Walker2d payoff adjudicator (Hypothesis B) - staged 2026-09-09.

Registration: research/WALKER2D-PAYOFF-PREREGISTRATION-2026-09-09.md. The
registered endpoint set (section 5) mirrors the DMC payoff study: per-arm
u_rank_corr / w_rmse / next_state_mse / final_return, plus paired contrasts
vs ordinary on u_rank_corr and final_return.

NO registered bars are printed here: the DMC bars (DMC-PAYOFF prereg) are
Hopper-registered and Walker2d bars are locked at launch in an addendum to
the staging registration. Return is read with the pre-committed floor rule
from the DMC study: if per-arm return medians are ~0 and means are driven by
a few seeds, the return axis is floor-bound at this budget and the payoff
question is deferred - never adjudicated here.

Usage: python -m udwm.scripts.summarize_walker2d_payoff --data runs/<file>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
U_RANK_BAR = 0.70


def load_rows(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d["rows"]


def by_seed(rows, arm):
    return {int(r["seed"]): r for r in rows if r["variant"] == arm}


def paired_delta(rows_a, rows_b, key):
    seeds = sorted(set(rows_a) & set(rows_b))
    vals = np.asarray(
        [float(rows_a[s][key]) - float(rows_b[s][key]) for s in seeds], dtype=float
    )
    return seeds, vals


def bootstrap_ci(deltas, rng, n_draws):
    means = rng.choice(deltas, size=(n_draws, deltas.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report_pair(rng, rows_a, rows_b, key, n_draws, label):
    _, deltas = paired_delta(rows_a, rows_b, key)
    if deltas.size == 0:
        print(f"{label:<52} arm(s) absent from file - skipped")
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = bootstrap_ci(deltas, rng, n_draws)
    wins = int(np.sum(deltas > 0))
    print(f"{label:<52} {deltas.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  "
          f"wins {wins}/{len(deltas)}")
    return deltas.mean(), lo, hi, wins


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="runs/walker2d_payoff_30seed_15k_gpu.json")
    p.add_argument("--n-bootstrap", type=int, default=100000)
    p.add_argument("--rng-seed", type=int, default=0)
    args = p.parse_args(argv)

    rows = load_rows((ROOT / args.data).resolve())
    rng = np.random.default_rng(args.rng_seed)
    present = []
    for r in rows:
        if r["variant"] not in present:
            present.append(r["variant"])
    d = {a: by_seed(rows, a) for a in present}
    seeds = sorted(d.get("ordinary", {}))
    print(f"rows={len(rows)} arms={present} seeds={seeds}")
    if not seeds:
        raise SystemExit("no ordinary rows - cannot adjudicate paired contrasts")

    print(f"\n== per-arm final eval (n={len(seeds)}) ==")
    print(f"{'arm':<24}{'u_mean':>8}{'u_med':>8}{'n>=0.70':>9}"
          f"{'w_rmse_med':>13}{'ns_mse':>10}{'ret_mean':>10}{'ret_med':>10}")
    for a in present:
        rs = list(d[a].values())
        ur = np.asarray([r["u_rank_corr"] for r in rs])
        ret = np.asarray([r["final_return"] for r in rs])
        print(f"{a:<24}{ur.mean():>8.3f}{float(np.median(ur)):>8.3f}"
              f"{int(np.sum(ur >= U_RANK_BAR)):>7}/{len(ur):<2}"
              f"{float(np.median([r['w_rmse'] for r in rs])):>13.3g}"
              f"{float(np.mean([r['next_state_mse'] for r in rs])):>10.4f}"
              f"{float(ret.mean()):>10.4f}{float(np.median(ret)):>10.4f}")

    print("\n== paired contrasts vs ordinary (bootstrap 95% CI, house style) ==")
    for a in present:
        if a == "ordinary":
            continue
        report_pair(rng, d[a], d["ordinary"], "u_rank_corr", args.n_bootstrap,
                    f"{a} - ordinary u_rank_corr")
        report_pair(rng, d[a], d["ordinary"], "final_return", args.n_bootstrap,
                    f"{a} - ordinary final_return")

    print("\n== pre-committed floor rule (never adjudicates) ==")
    for a in present:
        ret = np.asarray([r["final_return"] for r in d[a].values()])
        med = float(np.median(ret))
        mean = float(ret.mean())
        driven = (mean - med) > max(0.5 * abs(mean), 1e-6) and med < 0.05
        flag = "FLOOR-BOUND -> return deferred" if driven else "return usable"
        print(f"{a:<24} ret med {med:+.4f} mean {mean:+.4f}  -> {flag}")


if __name__ == "__main__":
    main()