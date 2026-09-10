# -*- coding: utf-8 -*-
"""Adjudicator for the Hypothesis-C gradient-interference probe.

Registration: research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md.  Reports the
registered endpoints exactly, with paired bootstrap CIs and wins/N, and does
not editorialize beyond the measured numbers.

Registered endpoints (fixed before the probe ran):

  E1 (environment contrast).  Mean within-seed interference
     ``-cos(g_point, g_unc)`` for ``identified_eq``, averaged over the probe
     checkpoints after the warmup fraction.  Prediction: DMC/hopper-hop >>
     DelayedBimodal.  Bar: the DMC-minus-DB difference has a bootstrap 95% CI
     excluding zero AND is positive.

  E2 (within-environment dose-response).  Across seeds within one env, the
     correlation between a seed's mean interference (identified_eq) and that
     seed's u-rank gap ``u_rank(ordinary) - u_rank(identified_eq)``.
     Prediction: positive (more interference -> larger eq deficit).

  P  (placebo).  E1 and E2 run on the M=1 ``hybrid`` family, which never pairs
     M>=2 latents.  Prediction: no consistent env contrast, no dose-response.

  Falsification.  If E1's CI includes zero (or the sign is wrong) or E2 is
     uncorrelated, Hypothesis C is refuted and must be reported that way.

Usage:
    python -m udwm.scripts.summarize_gradient_interference \
        --data db=runs/gi_probe_db.json --data dmc=runs/gi_probe_dmc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOOT = 100000
BOOT_SEED = 0
WARMUP_FRAC = 0.2          # registered: drop the first 20% of checkpoints
TEST_ARM = "identified_eq"
REF_ARM = "ordinary"
PLACEBO_ARMS = ("hybrid", "lagged_hybrid")


def _load(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows", []), payload.get("protocol", {})


def seed_interference(row, warmup_frac=WARMUP_FRAC):
    cps = [c for c in row.get("gi_checkpoints", []) if c.get("applicable")]
    if not cps:
        return None, 0
    n_drop = int(len(cps) * warmup_frac)
    used = cps[n_drop:] or cps
    return float(np.mean([c["interference"] for c in used])), len(used)


def seed_u_rank(row, key="u_rank_corr"):
    v = row.get("final_metrics", {}).get(key)
    return float(v) if v is not None else None


def mean_by_seed(rows, arm, fn):
    out = {}
    for r in rows:
        if r.get("variant") != arm:
            continue
        v = fn(r)
        if isinstance(v, tuple):
            v = v[0]
        if v is not None:
            out[int(r["seed"])] = float(v)
    return out


def boot_ci(values, n=BOOT, seed=BOOT_SEED, stat=np.mean):
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    draws = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_two_sample_ci(a, b, n=BOOT, seed=BOOT_SEED):
    """CI for mean(a) - mean(b), unpaired resample with replacement."""
    if len(a) == 0 or len(b) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    da = A[rng.integers(0, A.size, size=(n, A.size))].mean(axis=1)
    db = B[rng.integers(0, B.size, size=(n, B.size))].mean(axis=1)
    d = da - db
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_corr_ci(x, y, n=BOOT, seed=BOOT_SEED):
    """Vectorised bootstrap CI for Pearson r (paired resample of (x, y))."""
    X, Y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if X.size < 3:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.size, size=(n, X.size))
    XS, YS = X[idx], Y[idx]
    XS = XS - XS.mean(axis=1, keepdims=True)
    YS = YS - YS.mean(axis=1, keepdims=True)
    den = np.sqrt((XS ** 2).sum(axis=1) * (YS ** 2).sum(axis=1))
    ok = den > 1e-12
    if not ok.any():
        return None, None
    r = (XS * YS).sum(axis=1)[ok] / den[ok]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse_env(rows):
    inter = {a: mean_by_seed(rows, a, seed_interference) for a in set(r["variant"] for r in rows)}
    u = {a: mean_by_seed(rows, a, lambda r: seed_u_rank(r)) for a in set(r["variant"] for r in rows)}
    return inter, u


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True,
                    help="env_name=path (repeatable)")
    ap.add_argument("--out", default="runs/hypothesis_c_adjudication.json")
    args = ap.parse_args(argv)

    envs = {}
    for spec in args.data:
        name, _, path = spec.partition("=")
        envs[name] = _load((ROOT / path) if not Path(path).is_absolute() else Path(path))

    report = {"registration": "research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md",
              "test_arm": TEST_ARM, "ref_arm": REF_ARM,
              "warmup_frac_dropped": WARMUP_FRAC, "bootstrap_draws": BOOT,
              "envs": {}, "endpoints": {}}

    per_env = {}
    for name, (rows, protocol) in envs.items():
        inter, u = analyse_env(rows)
        per_env[name] = (inter, u)
        entry = {"n_rows": len(rows), "protocol_steps": protocol.get("steps"),
                 "arms": {}}
        for arm, vals in inter.items():
            if not vals:
                entry["arms"][arm] = {"n_seeds": 0}
                continue
            arr = np.asarray(list(vals.values()), float)
            entry["arms"][arm] = {
                "n_seeds": int(arr.size),
                "interference_mean": float(arr.mean()),
                "interference_median": float(np.median(arr)),
                "interference_per_seed": {str(k): float(v) for k, v in vals.items()},
                "u_rank_mean": (float(np.mean(list(u.get(arm, {}).values())))
                                if u.get(arm) else None),
            }
        report["envs"][name] = entry

    # ---- E1: environment contrast on the test arm (and the placebo) --------
    env_names = list(envs)
    e1 = {}
    for arm in (TEST_ARM,) + PLACEBO_ARMS:
        table = {}
        for name in env_names:
            vals = per_env[name][0].get(arm, {})
            table[name] = np.asarray(list(vals.values()), float)
        if len(env_names) >= 2:
            # Registered direction is DMC - DB; use it whenever both are present.
            order = env_names
            if "dmc" in env_names and "db" in env_names:
                order = ["dmc", "db"]
            hi, lo = order[0], order[1]
            a, b = table.get(hi, np.array([])), table.get(lo, np.array([]))
            if a.size and b.size:
                lo_ci, hi_ci = boot_two_sample_ci(a, b)
                table["contrast"] = {
                    "minus": "{}_minus_{}".format(hi, lo),
                    "delta": float(a.mean() - b.mean()), "ci95": [lo_ci, hi_ci],
                    "wins": int(np.sum(a[:, None] > b[None, :])),
                    "n_pairs": int(a.size * b.size),
                }
            else:
                table["contrast"] = {"minus": "{}_minus_{}".format(hi, lo),
                                     "delta": None, "ci95": [None, None],
                                     "wins": None, "n_pairs": 0}
        e1[arm] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in table.items()}
    report["endpoints"]["E1_env_contrast"] = e1

    # ---- E2: within-env dose-response -------------------------------------
    e2 = {}
    for arm in (TEST_ARM,) + PLACEBO_ARMS:
        arm_e2 = {}
        for name in env_names:
            inter, u = per_env[name]
            if arm not in inter or REF_ARM not in u or arm not in u:
                arm_e2[name] = {"n": 0}
                continue
            seeds = sorted(set(inter[arm]) & set(u[REF_ARM]) & set(u[arm]))
            xs = [inter[arm][s] for s in seeds]
            ys = [u[REF_ARM][s] - u[arm][s] for s in seeds]
            arm_e2[name] = {
                "n": len(seeds), "seeds": seeds,
                "interference": xs, "u_rank_gap": ys,
                "pearson": pearson(xs, ys), "spearman": spearman(xs, ys),
                "pearson_ci95": list(boot_corr_ci(xs, ys)),
                "wins": int(np.sum(np.asarray(ys) > 0)) if ys else None,
            }
        e2[arm] = arm_e2
    report["endpoints"]["E2_dose_response"] = e2

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- console summary ---------------------------------------------------
    print("== Hypothesis C: gradient interference ==")
    for name in env_names:
        e = report["envs"][name]
        for arm, a in sorted(e["arms"].items()):
            if a.get("n_seeds"):
                print("  {:>4} {:>20}: interference {:+.4f} (n={}, u_rank {})".format(
                    name, arm, a["interference_mean"], a["n_seeds"],
                    "n/a" if a["u_rank_mean"] is None else "{:.3f}".format(a["u_rank_mean"])))
            else:
                print("  {:>4} {:>20}: not applicable".format(name, arm))
    for arm in (TEST_ARM,) + PLACEBO_ARMS:
        c = e1[arm].get("contrast")
        if not c:
            continue
        if c["delta"] is None or c["ci95"][0] is None:
            print("  E1 {}: {} not computable (missing arm rows)".format(arm, c["minus"]))
        else:
            print("  E1 {}: {} delta {:+.4f} CI [{:+.4f}, {:+.4f}] wins {}/{}".format(
                arm, c["minus"], c["delta"], c["ci95"][0], c["ci95"][1],
                c["wins"], c["n_pairs"]))
    for arm in (TEST_ARM,) + PLACEBO_ARMS:
        for name in env_names:
            d = e2[arm][name]
            if d.get("n", 0) >= 3 and d.get("pearson") is not None \
                    and d["pearson_ci95"][0] is not None:
                print("  E2 {} {}: pearson {:+.3f} CI [{:+.3f}, {:+.3f}] spearman {:+.3f} (n={})".format(
                    arm, name, d["pearson"], d["pearson_ci95"][0], d["pearson_ci95"][1],
                    d["spearman"] if d.get("spearman") is not None else float("nan"), d["n"]))
            else:
                print("  E2 {} {}: n={} (not adjudicable)".format(arm, name, d.get("n", 0)))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
