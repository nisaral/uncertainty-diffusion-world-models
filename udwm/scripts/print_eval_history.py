# -*- coding: utf-8 -*-
"""Print per-eval training curves from a split-seed policy json.

Rows written after 2026-09-07 carry ``eval_history`` (the trainer's in-loop
eval log: step, return_mean, u_rank_corr, u_rmse, next_state_mse, ...). This
reader is for diagnostics only (e.g., the registered DMC budget probe); it
never adjudicates.

Usage:
    python -m udwm.scripts.print_eval_history --data runs/dmc_budget_probe_gpu.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ["return_mean", "u_rank_corr", "u_rmse", "next_state_mse"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/dmc_budget_probe_gpu.json")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=None)
    args = p.parse_args(argv)
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))["rows"]
    seeds = args.seeds or sorted({int(r["seed"]) for r in rows})
    variants = args.variants or sorted({r["variant"] for r in rows})
    hdr = f'{"step":>8}' + "".join(f"{f:>16}" for f in FIELDS)
    for seed in seeds:
        for variant in variants:
            rows_for = [r for r in rows
                        if int(r["seed"]) == seed and r["variant"] == variant]
            if not rows_for:
                continue
            logs = rows_for[0].get("eval_history") or []
            print(f"\nseed {seed} | {variant}  (n_evals={sum(1 for e in logs if 'return_mean' in e)})")
            print(hdr)
            for e in logs:
                if "return_mean" not in e:
                    continue
                line = f'{int(e["step"]):>8}'
                for f in FIELDS:
                    v = e.get(f)
                    line += f"{float(v):>16.4f}" if v is not None else f'{"-":>16}'
                print(line)


if __name__ == "__main__":
    main()