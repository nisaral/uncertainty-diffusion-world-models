# -*- coding: utf-8 -*-
"""CRN-bias probe adjudicator (Hypothesis A) - registration 2026-09-09.

Registration: research/CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md. Loads the
probe output of udwm/scripts/probe_crn_bias.py (one row per (seed, variant)
with nested "crn_checkpoints": step-level M1/M3/M4 records) and reads the
registered endpoints E1-E5 plus the compute-normalized-style slope regression
of estimation error on logged drift. Verdict language follows the house
style: wins/N and bootstrap 95% CIs only, no editorializing.

Cell map (registration section 3):
  A = identified_eq          (live critic, no normalization)
  B = identified_eq_norm     (live critic, normalized values)
  C = lagged_identified_eq   (lagged target critic, normalized)
  D = lagged_identified_eq_nonorm (lagged target critic, no normalization)
plus hybrid / lagged_hybrid (M=1 single-eval control family) and
identified_hybrid (EMA collapse sanity arm).

E4 (placebo correction) is deliberately NOT implemented here: per the queue,
the Wu-style correction is only built after the core probe confirms the
mechanism. This script prints that gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

U_RANK_BAR = 0.70
ARMS = ["identified_eq", "identified_eq_norm", "lagged_identified_eq",
        "lagged_identified_eq_nonorm", "hybrid", "lagged_hybrid",
        "identified_hybrid"]


def load_rows(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d["rows"]


def by_seed(rows, arm):
    return {int(r["seed"]): r for r in rows if r["variant"] == arm}


def final_cp(row):
    """Last (fresh) checkpoint record: max step wins; ties -> later record."""
    cps = row.get("crn_checkpoints") or []
    if not cps:
        return None
    best = cps[0]
    for cp in cps[1:]:
        if cp.get("step", 0) >= best.get("step", 0):
            best = cp
    return best


def seed_final(rows_arm, key):
    out = {}
    for seed, row in rows_arm.items():
        cp = final_cp(row)
        if cp is not None and key in cp:
            out[seed] = float(cp[key])
    return out


def paired(rows_a, rows_b, key):
    seeds = sorted(set(rows_a) & set(rows_b))
    vals = np.asarray([float(rows_a[s][key]) - float(rows_b[s][key]) for s in seeds])
    return seeds, vals


def bootstrap_ci(stats, rng, n_draws):
    stats = np.asarray(stats, dtype=float)
    means = rng.choice(stats, size=(n_draws, stats.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def rho_bootstrap_ci(keys_a, keys_b, rng, n_draws):
    keys_a = np.asarray(keys_a, dtype=float)
    keys_b = np.asarray(keys_b, dtype=float)
    rng = np.random.default_rng(0)
    rhos = []
    for _ in range(n_draws):
        idx = rng.integers(0, keys_a.size, size=keys_a.size)
        rhos.append(spearman(keys_a[idx], keys_b[idx]))
    rhos = np.asarray(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def per_seed_slope(seed_row, x_key, y_key, min_points=3):
    """OLS slope of y on x over the seed's checkpoint series (rank form uses
    spearman on the series when requested by caller)."""
    cps = [cp for cp in (seed_row.get("crn_checkpoints") or [])
           if x_key in cp and y_key in cp]
    cps.sort(key=lambda cp: cp["step"])
    if len(cps) < min_points:
        return None
    xs = np.asarray([cp[x_key] for cp in cps], dtype=float)
    ys = np.asarray([cp[y_key] for cp in cps], dtype=float)
    if np.all(xs == xs[0]):
        return None
    slope = float(np.polyfit(xs, ys, 1)[0])
    return slope


def slope_report(rng, rows_by_seed, x_key, y_key, label, n_draws):
    slopes = {seed: per_seed_slope(row, x_key, y_key)
              for seed, row in rows_by_seed.items()}
    slopes = {s: v for s, v in slopes.items() if v is not None}
    if len(slopes) < 3:
        print(f"{label:<52} insufficient per-seed series "
              f"({len(slopes)}/{(len(rows_by_seed))})")
        return slopes
    vals = np.asarray(list(slopes.values()), dtype=float)
    lo, hi = bootstrap_ci(vals, rng, n_draws)
    print(f"{label:<52} mean slope {vals.mean():+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]  n_seeds {len(vals)}")
    return slopes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="runs/crn_bias_probe_15k_n10_gpu.json")
    p.add_argument("--n-bootstrap", type=int, default=100000)
    p.add_argument("--rng-seed", type=int, default=0)
    args = p.parse_args(argv)
    rows = load_rows((ROOT / args.data).resolve())
    rng = np.random.default_rng(args.rng_seed)
    d = {a: by_seed(rows, a) for a in ARMS}
    present = [a for a in ARMS if d[a]]
    seeds = sorted(set().union(*[set(d[a]) for a in present])) if present else []
    print(f"rows={len(rows)} arms={present} seeds={seeds}")

    print(f"\n== per-arm final checkpoint ==")
    print(f"{'arm':<28}{'u_mean':>8}{'u_med':>8}{'n>=0.70':>9}{'B_hat_med':>11}"
          f"{'drift_med':>11}{'w_rmse_med':>12}")
    for a in present:
        rs = list(d[a].values())
        urs = np.asarray([final_cp(r)["u_rank_corr"] for r in rs if final_cp(r)])
        bhs = np.asarray([final_cp(r)["b_hat"] for r in rs if final_cp(r) and "b_hat" in final_cp(r)])
        drs = np.asarray([final_cp(r)["drift"] for r in rs if final_cp(r) and "drift" in final_cp(r)])
        wrm = np.median([r["final_metrics"]["w_rmse"] for r in rs])
        print(f"{a:<28}{urs.mean():>8.3f}{float(np.median(urs)):>8.3f}"
              f"{int(np.sum(urs >= U_RANK_BAR)):>7}/{len(urs):<2}"
              f"{float(np.median(bhs)) if bhs.size else float('nan'):>11.3g}"
              f"{float(np.median(drs)) if drs.size else float('nan'):>11.3g}"
              f"{float(wrm):>12.3g}")

    def e1_cell(cell_arm):
        u = seed_final(d[cell_arm], "u_rank_corr")
        b = seed_final(d[cell_arm], "b_hat")
        common = sorted(set(u) & set(b))
        if len(common) < 3:
            print(f"E1 {cell_arm}: <3 seeds with both endpoints -> INCONCLUSIVE")
            return
        rho = spearman([u[s] for s in common], [b[s] for s in common])
        lo, hi = rho_bootstrap_ci([u[s] for s in common], [b[s] for s in common],
                                  rng, args.n_bootstrap)
        print(f"E1 {cell_arm}: Spearman rho(u_rank, B_hat) = {rho:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] n={len(common)} "
              f"(negative with CI excl 0 -> SUPPORTED)")

    print("\n== E1 (primary, mechanism) within-cell ==")
    for cell in ("lagged_identified_eq", "lagged_identified_eq_nonorm"):
        e1_cell(cell)
    for cell in ("identified_eq", "identified_eq_norm"):
        b = seed_final(d[cell], "b_hat")
        print(f"E1 {cell}: B_hat(final) mean = "
              f"{float(np.mean(list(b.values()))) if b else float('nan'):.3g} "
              f"(live-critic cell: registered within-cell null, B_hat ~ 0)")

    print("\n== E2 (decomposition; descriptive at n=10, adjudicated at n=30) ==")
    for lag, live, cell in (
        ("lagged_identified_eq_nonorm", "identified_eq", "no-norm cell"),
        ("lagged_identified_eq", "identified_eq_norm", "norm cell"),
    ):
        _, deltas = paired(d[lag], d[live], "u_rank_corr")
        lo, hi = bootstrap_ci(deltas, rng, args.n_bootstrap)
        wins = int(np.sum(deltas > 0))
        print(f"E2 {cell}: {lag} - {live} u_rank {deltas.mean():+.3f} "
              f"[{lo:+.3f},{hi:+.3f}] wins {wins}/{len(deltas)} "
              f"(negative = staleness hurts the paired family)")

    print("\n== E3 (control family, no direction predicted) ==")
    u_h = seed_final(d["lagged_hybrid"], "u_rank_corr")
    u_h0 = seed_final(d["hybrid"], "u_rank_corr")
    b_h = seed_final(d["lagged_hybrid"], "b_hat")
    common = sorted(set(u_h) & set(u_h0) & set(b_h))
    if len(common) >= 3:
        benefit = np.asarray([u_h[s] - u_h0[s] for s in common])
        bvals = np.asarray([b_h[s] for s in common])
        rho = spearman(benefit, bvals)
        print(f"E3: Spearman rho(hybrid-family lag benefit, B_hat) = {rho:+.3f} "
              f"n={len(common)} (registered read: ~0, no sign predicted)")

    print("\n== 1c regression: estimation error vs logged drift (per-seed slope, pooled) ==")
    slope_report(rng, d["lagged_identified_eq"], "drift", "e_u",
                 "C lagged_eq (norm): slope of e_u on drift", args.n_bootstrap)
    slope_report(rng, d["lagged_identified_eq_nonorm"], "drift", "e_u",
                 "D lagged_eq (nonorm): slope of e_u on drift", args.n_bootstrap)
    slope_report(rng, d["lagged_hybrid"], "drift", "e_u",
                 "lagged_hybrid (placebo): slope of e_u on drift", args.n_bootstrap)
    slope_report(rng, d["hybrid"], "drift", "e_u",
                 "hybrid (placebo): slope of e_u on drift", args.n_bootstrap)

    print("\n== E5 (collapse sanity, not a bar) ==")
    if d["identified_hybrid"]:
        u = seed_final(d["identified_hybrid"], "u_rank_corr")
        print(f"E5: identified_hybrid u_rank mean {np.mean(list(u.values())):+.3f} "
              f"(registered: ~0 reproduces the EMA collapse)")

    print("\n== pre-committed decision tree (registration section 8) ==")
    print("E1 SUPPORTED + E4 CLEAN -> Hypothesis A confirmed; next: Wu-style")
    print("  correction as a new eq_crn arm, pre-registered DMC adjudication.")
    print("E1 INCONCLUSIVE + E2 within-cell normalization story -> attribution")
    print("  corrected to normalization, not only CRN staleness.")
    print("E1 INCONCLUSIVE + E2 flat -> Hypothesis A not supported on DMC/15k;")
    print("  Section 7 written as an open problem.")
    print("E4 is gated on the mechanism confirming (built after, not before).")


if __name__ == "__main__":
    main()
