# -*- coding: utf-8 -*-
"""Compute-normalized re-analysis of already-adjudicated run files (no new runs).

Registration: research/COMPUTE-NORMALIZED-REANALYSIS-2026-09-09.md. Pure
re-indexing of existing rows: each arm's u_rank learning curve (the eval
checkpoints already logged in ``eval_history``) is re-plotted/tabulated
against cumulative *teacher-sample-equivalents* instead of raw env steps, and
the arm ordering is re-read at fixed teacher-sample spend.

Cost multipliers are derived from the loss implementation, not assumed
(udwm/models/consistency.py):
  - ordinary        -> distill_loss:               1 teacher-member fwd/epoch
  - hybrid family   -> decision_preserving_loss:   5 teacher-member fwd/epoch
  - identified eq   -> identified_*_loss (M=2):   10 teacher-member fwd/epoch
Each model-train epoch draws ONE minibatch and runs the arm's full loss
forward once (see _train_world_model, epochs = num_model_epochs = 12 per
model-train call, one call every model_train_freq=100 env steps from
warmup_steps on). "teacher-sample-equivalent" units below = ONE ordinary
epoch's teacher cost (1 teacher + 1 student diffusion forward on a minibatch
of model_batch_size=128 states).

Critic value_fn evals are NOT folded into the unit (they are small MLP state
evaluations with no teacher-sample interpretation); their raw counts per
epoch are printed for transparency (hybrid: 2 x N x B state evals, identified:
4 x N x B) so the unit definition is auditable.

Fresh-eval milestone extraction follows the registered forward-fill
convention: the LAST eval_history record at each duplicated step is the fresh
eval; the row-level final metrics append the terminal point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# teacher-member diffusion forwards per model-train EPOCH, read from
# udwm/models/consistency.py (distill_loss / decision_preserving_distill_loss
# / identified_*_loss schedule path, ensemble_size N=5, m_latents M=2).
# Ordinary samples ONE random member; every decision arm runs all N members;
# the identified family repeats the member pass M=2 times.
TEACHER_FWD_PER_EPOCH = {
    "ordinary": 1,
    "ordinary_gate_off": 1,
    "hybrid": 5,
    "lagged_hybrid": 5,
    "identified_hybrid": 10,
    "identified_eq": 10,
    "identified_eq_norm": 10,
    "lagged_identified_eq": 10,
    "lagged_identified_eq_nonorm": 10,
    "lagged_identified": 10,
    "identified_wonly": 10,
}

VALUE_STATE_EVALS_PER_EPOCH = {
    "ordinary": 0,
    "ordinary_gate_off": 0,
    "hybrid": 2,
    "lagged_hybrid": 2,
    "identified_hybrid": 4,
    "identified_eq": 4,
    "identified_eq_norm": 4,
    "lagged_identified_eq": 4,
    "lagged_identified_eq_nonorm": 4,
    "lagged_identified": 4,
    "identified_wonly": 4,
}  # value_fn calls per epoch; each call evaluates N x B states.

KEYS = ("u_rank_corr", "w_rmse", "final_return")


def load_payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(d):
    return d if isinstance(d, list) else d["rows"]


def fresh_milestones(history):
    """[(step, record)] with the last record at each duplicated step winning.

    The trainer appends log(step) then eval(step) at eval_freq multiples, so
    the surviving record at those steps is the fresh eval; single records at
    non-eval steps (forward-filled copies of the previous eval) are dropped.
    """
    collapsed = []
    for r in history:
        if collapsed and collapsed[-1][0] == r["step"]:
            collapsed[-1] = (r["step"], r)
        else:
            collapsed.append((r["step"], r))
    # Keep only records that carry full-eval fields (u_rank_corr), which
    # forward-filled log records after the FIRST eval also carry -> restrict
    # to the surviving record of each duplicated step (the fresh eval).
    step_counts = {}
    for r in history:
        step_counts[r["step"]] = step_counts.get(r["step"], 0) + 1
    dupe_steps = {s for s, _ in collapsed if step_counts.get(s, 0) > 1}
    out = [(s, r) for s, r in collapsed if s in dupe_steps and "u_rank_corr" in r]
    return out


def per_seed_milestones(row, final_step):
    hist = row.get("eval_history") or []
    ms = fresh_milestones(hist)
    steps = sorted({s for s, _ in ms})
    by_step = {s: r for s, r in ms}
    out = []
    for s in steps:
        out.append((s, by_step[s]))
    # terminal point from the row-level final metrics (fresh full eval at the
    # end of train()); skip if the same step is already present.
    if final_step not in by_step and "u_rank_corr" in row:
        out.append((int(final_step), row))
    out.sort(key=lambda x: x[0])
    return out


def model_calls_up_to(step, warmup=300, freq=100):
    if step < warmup:
        return 0
    return 1 + (int(step) - warmup) // freq


def cumulative_units(step, arm, warmup=300, freq=100, epochs=12):
    calls = model_calls_up_to(step, warmup, freq)
    mult = TEACHER_FWD_PER_EPOCH.get(arm)
    if mult is None:
        return None
    return calls * epochs * mult


def mean_curve(rows_arm, final_step, key="u_rank_corr"):
    """x = cumulative units, y = mean endpoint over seeds at each milestone."""
    pts = []
    for row in rows_arm:
        for s, rec in per_seed_milestones(row, final_step):
            if key not in rec or cumulative_units(s, row["variant"]) is None:
                continue
            pts.append((cumulative_units(s, row["variant"]), float(rec[key])))
    if not pts:
        return [], []
    xs = sorted({x for x, _ in pts})
    ys = []
    for x in xs:
        vals = [v for u, v in pts if u == x]
        ys.append(float(np.mean(vals)))
    return xs, ys


def interp(xs, ys, x):
    if x < xs[0] or x > xs[-1]:
        return None
    if len(xs) == 1:
        return float(ys[0])
    return float(np.interp(x, xs, ys))


def report_file(path, label, final_step, rng_seed=0):
    d = load_payload(path)
    rows = rows_of(d)
    proto = d.get("protocol") or {}
    if final_step is None:
        final_step = int(proto.get("steps", 0)) or max(
            (int(r.get("step", 0)) for r in rows), default=0
        ) or int(proto.get("total_env_steps", 0))
    arms = sorted({r["variant"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    print(f"\n{'=' * 100}\n{label}\nsource: {path}  rows={len(rows)} "
          f"seeds={len(seeds)} final_step={final_step}\n{'=' * 100}")

    print(f"\nper-arm forward counts per epoch (source: udwm/models/consistency.py)")
    print(f"{'arm':<26}{'teacher_fwd':>12}{'value_fn_calls':>16}{'mult vs ordinary':>18}")
    for a in arms:
        mult = TEACHER_FWD_PER_EPOCH.get(a)
        print(f"{a:<26}{str(mult):>12}{str(VALUE_STATE_EVALS_PER_EPOCH.get(a, '?')):>16}"
              f"{(str(mult) + 'x') if mult else 'unknown':>18}")

    print(f"\nlearning curves at eval milestones: mean u_rank_corr at cumulative "
          f"teacher-sample units")
    curves = {}
    for a in arms:
        xs, ys = mean_curve([r for r in rows if r["variant"] == a], final_step)
        curves[a] = (xs, ys)
        if not xs:
            continue
        units = int(xs[-1])
        final = ys[-1]
        eff = 1000.0 * final / max(units, 1)
        row_label = f"{a:<26}"
        cells = " ".join(f"{y:.3f}@{int(x)}" for x, y in zip(xs, ys))
        print(f"{row_label} final={final:.3f} spend={units:>6}u  "
              f"u_rank/1000u={eff:.3f}   {cells}")

    # Fixed-compute dominance: is any arm measured at a strictly lower spend
    # with a >= final u_rank of each other arm? (measured points only)
    print("\nfixed-compute read (measured points only, no extrapolation):")
    for a in arms:
        xs, ys = curves.get(a, ([], []))
        if not xs:
            continue
        for b in arms:
            if b == a:
                continue
            xb, yb = curves.get(b, ([], []))
            if not xb:
                continue
            if xs[-1] < xb[-1] and ys[-1] >= yb[-1]:
                print(f"  {a}: final u_rank {ys[-1]:.3f} at {int(xs[-1])}u dominates "
                      f"{b}: {yb[-1]:.3f} at {int(xb[-1])}u "
                      f"(lower spend, >= u_rank)")

    # Overlap domain across arms with data: can ANY common-budget interpolation
    # be reported?
    spans = [(xs[0], xs[-1]) for a in (arms) for xs, _ in [curves[a]] if xs]
    if spans:
        lo = max(s[0] for s in spans)
        hi = min(s[1] for s in spans)
        print(f"\ncommon teacher-sample domain with eval data for ALL arms: "
              f"[{int(lo)}, {int(hi)}]u -> "
              f"{'interpolation on the common domain is possible' if lo <= hi else 'EMPTY: no fixed-compute head-to-head is readable at this protocol without extrapolation'}")
        if lo <= hi:
            budgets = [int(lo), int((lo + hi) / 2), int(hi)]
            print(f"interpolated u_rank means at common budgets:")
            for budget in budgets:
                row_s = f"  budget {budget:>7}u: "
                for a in arms:
                    xs, ys = curves[a]
                    if not xs:
                        continue
                    v = interp(xs, ys, budget)
                    row_s += f"{a}={'-' if v is None else f'{v:.3f}'}  "
                print(row_s)
    else:
        print("\nno arm has eval milestones in this file")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="runs/dmc_payoff_30seed_15k_gpu.json")
    p.add_argument("--second-data", default=None,
                   help="optional second file (e.g. runs/policy_combined_fix_10seed.json)")
    p.add_argument("--final-step", type=int, default=None,
                   help="total env steps; default: from file protocol")
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--model-freq", type=int, default=100)
    p.add_argument("--epochs", type=int, default=12)
    args = p.parse_args(argv)

    report_file((ROOT / args.data).resolve(), "DMC (or --data) file", args.final_step)
    if args.second_data:
        report_file((ROOT / args.second_data).resolve(), "second file",
                    args.final_step)


if __name__ == "__main__":
    main()
