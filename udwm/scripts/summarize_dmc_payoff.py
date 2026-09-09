# -*- coding: utf-8 -*-
"""Adjudication summary for the DMC 30-seed payoff run (2026-09-08).

Registration: research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md (Amendment 2
activated 2026-09-08: 15,000-step budget). Data:
runs/dmc_payoff_30seed_15k_gpu.json (six arms x 30 seeds) and, when given,
runs/dmc_payoff_30seed_15k_gpu_ctrl.json (ordinary_gate_off, Addendum 4).

Registered bars (mirror the DelayedBimodal study):
  1. identified_eq u_rank_corr >= 0.70 on >= 21/30 seeds;
  2. paired identified_eq - hybrid u_rank_corr wins >= 21/30 AND bootstrap
     95% CI excludes 0 (the DelayedBimodal partial-transfer confirm).
Mechanism-transfer control (expected, not a bar): identified_eq - EMA
(identified_hybrid) wins >= 21/30 with CI excluding 0.
Return endpoints at 15k are NOT adjudicated (Amendment 2 item 5: floor-bound
on hopper-hop; deferred to the conditional 30k extension) and are printed as
descriptive only. The gate-off control is read with its own pre-committed
rule (Addendum 4): CI excludes 0 -> measurable gating effect; includes 0 ->
gating is return-neutral at this budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ARMS = [
    "ordinary", "hybrid", "lagged_hybrid", "identified_hybrid",
    "identified_eq", "lagged_identified_eq",
]

U_RANK_BAR = 0.70
SEED_BAR = 21  # 70% of 30


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
        print(f"{label:<46} arm(s) absent from file - skipped")
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = bootstrap_ci(deltas, rng, n_draws)
    wins = int(np.sum(deltas > 0))
    print(f"{label:<46} {deltas.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  wins {wins}/{len(deltas)}")
    return deltas.mean(), lo, hi, wins


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="runs/dmc_payoff_30seed_15k_gpu.json")
    p.add_argument("--ctrl", default="runs/dmc_payoff_30seed_15k_gpu_ctrl.json")
    p.add_argument("--n-bootstrap", type=int, default=100000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--no-ctrl", action="store_true",
                   help="skip the gate-off control block (arm-subset files)")
    p.add_argument("--no-bars", action="store_true",
                   help="skip the DMC-registered bar block (non-DMC files)")
    args = p.parse_args(argv)

    rows = load_rows((ROOT / args.data).resolve())
    rng = np.random.default_rng(args.rng_seed)
    d = {a: by_seed(rows, a) for a in ARMS}
    for _rv in sorted({r["variant"] for r in rows}):
        d.setdefault(_rv, by_seed(rows, _rv))
    seeds = sorted(d["ordinary"])
    if not seeds:
        seeds = sorted({int(r["seed"]) for r in rows})
        print(f"NOTE: no ordinary rows; arm-subset mode, arms="
              f"{sorted({r['variant'] for r in rows})}")
    if not seeds:
        raise SystemExit("file has no rows to summarize")
    print(f"rows={len(rows)} seeds={len(seeds)} range=[{seeds[0]},{seeds[-1]}]")

    print(f"\n== per-arm final eval (n={len(seeds)}) ==")
    print(
        f"{'arm':<22}{'u_rank_mean':>11}{'u_rank_med':>11}{'n>=0.70':>9}"
        f"{'w_rmse_med':>13}{'ns_mse':>10}{'ret_mean':>10}{'ret_med':>10}"
    )
    table_arms = list(ARMS) + sorted(
        {r["variant"] for r in rows if r["variant"] not in ARMS}
    )
    for a in table_arms:
        rs = list(d[a].values())
        if not rs:
            print(f"{a:<22}  (no rows in file - skipped)")
            continue
        ur = np.asarray([r["u_rank_corr"] for r in rs])
        print(
            f"{a:<22}{ur.mean():>11.3f}{float(np.median(ur)):>11.3f}"
            f"{int(np.sum(ur >= U_RANK_BAR)):>9}"
            f"{float(np.median([r['w_rmse'] for r in rs])):>13.3g}"
            f"{float(np.mean([r['next_state_mse'] for r in rs])):>10.4f}"
            f"{float(np.mean([r['final_return'] for r in rs])):>10.4f}"
            f"{float(np.median([r['final_return'] for r in rs])):>10.4f}"
        )

    print("\n== registered bars / contrasts (paired u_rank_corr) ==")
    if len(seeds) != 30:
        print(f"NOTE: bars registered for the DMC 30-seed study; this file has "
              f"n={len(seeds)} -> bar verdicts NOT applicable (read contrasts only).")
    if args.no_bars:
        print("(skipped: --no-bars)")
        eq, hy = {}, {}
    else:
        eq, hy = d["identified_eq"], d["hybrid"]
    if not eq:
        n_ge = 0
        verdict1 = "n/a (identified_eq absent)"
        print(f"bar 1: identified_eq u_rank >= {U_RANK_BAR} -> {verdict1}")
    else:
        n_ge = int(np.sum([r["u_rank_corr"] >= U_RANK_BAR for r in eq.values()]))
        verdict1 = "MET" if n_ge >= SEED_BAR else "NOT MET"
        print(f"bar 1: identified_eq u_rank >= {U_RANK_BAR} on {n_ge}/{len(seeds)} "
              f"seeds (need >= {SEED_BAR}) -> {verdict1}")
    if not eq or not hy:
        print("bar 2: eq - hybrid -> n/a (one or both arms absent from file)")
    else:
        _, lo, hi, w = report_pair(
            rng, eq, hy, "u_rank_corr", args.n_bootstrap,
            "bar 2: eq - hybrid (need >=21/30, CI excl 0)",
        )
        ok = (w >= SEED_BAR) and not (lo <= 0 <= hi)
        print(f"      -> {'MET' if ok else 'NOT MET'}")
    for a, b, label in [
        ("identified_eq", "ordinary", "eq - ordinary (transfer gap)"),
        ("identified_eq", "lagged_hybrid", "eq - lagged_hybrid (transfer gap)"),
        ("identified_eq", "identified_hybrid", "eq - EMA (mechanism transfer)"),
        ("lagged_identified_eq", "ordinary", "lagged_eq - ordinary (combined-fix gap)"),
        ("lagged_identified_eq", "hybrid", "lagged_eq - hybrid"),
        ("lagged_identified_eq", "identified_eq", "lagged_eq - eq (DB reversal check)"),
        ("lagged_hybrid", "hybrid", "lagged_hybrid - hybrid (lag axis, hybrid family)"),
        ("lagged_hybrid", "ordinary", "lagged_hybrid - ordinary"),
        ("hybrid", "ordinary", "hybrid - ordinary"),
        ("ordinary", "identified_hybrid", "ordinary - EMA (collapse control)"),
        ("hybrid", "identified_hybrid", "hybrid - EMA"),
    ]:
        report_pair(rng, d[a], d[b], "u_rank_corr", args.n_bootstrap, label)

    print("\n== return deltas vs ordinary (DESCRIPTIVE ONLY at 15k; deferred) ==")
    for a in ["hybrid", "lagged_hybrid", "identified_hybrid",
              "identified_eq", "lagged_identified_eq"]:
        report_pair(rng, d[a], d["ordinary"], "final_return",
                    args.n_bootstrap, f"{a} - ordinary")

    if args.ctrl and not args.no_ctrl:
        ctrl_path = (ROOT / args.ctrl).resolve()
        if ctrl_path.exists():
            ctrl = by_seed(load_rows(ctrl_path), "ordinary_gate_off")
            o = d["ordinary"]
            common = sorted(set(ctrl) & set(o))
            print(f"\n== gate-off control (ordinary_gate_off, n={len(common)}) ==")
            cur = np.asarray([ctrl[s]["u_rank_corr"] for s in common])
            our = np.asarray([o[s]["u_rank_corr"] for s in common])
            print(f"descriptive u_rank: gate_off mean {cur.mean():.3f} vs ordinary "
                  f"{our.mean():.3f} (never adjudicative)")
            report_pair(rng, ctrl, o, "final_return", args.n_bootstrap,
                        "gate_off - ordinary final_return")

    # MACURA field baseline (registered 2026-09-09): guarded extra contrasts,
    # only printed when the run file actually carries macura_gate rows. MACURA
    # (ICML 2024) produces no w/g split, so the fair endpoints are downstream
    # return and rollout-gating quality; u_rank_corr is a distillation sanity
    # read only (macura_gate uses ordinary distillation, so parity is expected).
    macura_rows = [r for r in rows if r["variant"] == "macura_gate"]
    if macura_rows:
        mg = by_seed(macura_rows, "macura_gate")
        print("\n== MACURA baseline contrasts (macura_gate) ==")
        report_pair(rng, mg, d["ordinary"], "final_return", args.n_bootstrap,
                    "macura_gate - ordinary final_return")
        report_pair(rng, mg, d["identified_eq"], "final_return", args.n_bootstrap,
                    "macura_gate - identified_eq final_return")
        report_pair(rng, mg, d["ordinary"], "u_rank_corr", args.n_bootstrap,
                    "macura_gate - ordinary u_rank (distill sanity)")
        report_pair(rng, mg, d["identified_eq"], "u_rank_corr", args.n_bootstrap,
                    "macura_gate - identified_eq u_rank")


if __name__ == "__main__":
    main()
