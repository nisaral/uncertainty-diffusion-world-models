"""Paired bootstrap summary for the identified-distillation stress A/B.

Reports, per arm and per endpoint, the paired delta versus `ordinary`, the
win count, and a percentile bootstrap 95% interval over seeds -- the same
protocol as research/RESULTS-STRESS-LARGE-2026-08-21.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOWER_IS_BETTER = {
    "w_rmse", "w_deb_rmse", "g_rmse", "member_value_rmse", "paired_state_mse",
    "conflated_rmse",
}
ENDPOINTS = [
    "w_rank_corr", "w_rmse", "top_decile_recall",
    "w_deb_rank_corr", "w_deb_rmse",
    "g_rank_corr", "g_rmse",
    "conflated_rank_corr",
    "paired_state_mse", "member_value_rmse",
    "split_distortion",
]


def boot_ci(x: np.ndarray, n_boot: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="runs/identified_stress_20seed.json")
    ap.add_argument("--baseline", default="ordinary")
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    rows = payload["rows"]
    base = {r["seed"]: r for r in rows if r["variant"] == args.baseline}
    arms = [v for v in sorted({r["variant"] for r in rows}) if v != args.baseline]

    checksums = {}
    for r in rows:
        checksums.setdefault(r["seed"], set()).add(round(r["teacher_checksum"], 6))
    gap = max(len(v) for v in checksums.values())
    print(f"file: {args.path}")
    print(f"seeds: {len(base)}   arms: {arms}")
    print(f"teacher checksum: {'EXACT pairing (1 value per seed)' if gap == 1 else f'MISMATCH ({gap} values)'}")
    print()

    # absolute means, for context
    print("=" * 104)
    print("ABSOLUTE MEANS")
    print("=" * 104)
    print(f"{'endpoint':<24}" + "".join(f"{a[:18]:>19}" for a in [args.baseline] + arms))
    for ep in ENDPOINTS:
        line = f"{ep:<24}"
        for a in [args.baseline] + arms:
            vals = [r[ep] for r in rows if r["variant"] == a and ep in r]
            line += f"{np.mean(vals):19.6f}" if vals else f"{'-':>19}"
        print(line)
    print()

    for arm in arms:
        sel = [r for r in rows if r["variant"] == arm and r["seed"] in base]
        print("=" * 104)
        print(f"{arm}  vs  {args.baseline}   (n={len(sel)} paired seeds)")
        print("=" * 104)
        print(f"{'endpoint':<24}{'mean delta':>13}{'wins':>8}{'bootstrap 95%':>30}{'verdict':>14}")
        for ep in ENDPOINTS:
            d = np.asarray(
                [r[ep] - base[r["seed"]][ep] for r in sel if ep in r and ep in base[r["seed"]]],
                dtype=float,
            )
            if d.size == 0:
                continue
            lo, hi = boot_ci(d)
            wins = int((d < 0).sum()) if ep in LOWER_IS_BETTER else int((d > 0).sum())
            better = (hi < 0) if ep in LOWER_IS_BETTER else (lo > 0)
            worse = (lo > 0) if ep in LOWER_IS_BETTER else (hi < 0)
            verdict = "BETTER" if better else ("WORSE" if worse else "inconclusive")
            print(f"{ep:<24}{d.mean():>13.6f}{wins:>5}/{d.size:<3}"
                  f"{f'[{lo:+.6f}, {hi:+.6f}]':>30}{verdict:>14}")
        print()


if __name__ == "__main__":
    main()
