# -*- coding: utf-8 -*-
"""DMC diagnostic-gate reader (registered in DMC-PAYOFF-PREREGISTRATION).

Reads the ordinary-arm rows of a pilot policy run and prints the teacher's
g*/w* ratio under the live SAC critic per seed (row.teacher_g_mean /
row.teacher_w_mean, real-buffer eval states) plus the median, and maps it to
the pre-committed regime:

  g*/w* >> 1      aleatoric-dominated (expect the DelayedBimodal pattern)
  O(1) .. 1e1     balanced (balance-window theory predicts both channels)
  << 1            epistemic-dominated (unlikely under a Q-value map)

Gate protocol (Amendment 1 of the DMC preregistration): pilot seeds {0, 1}
with the ordinary arm only, same runner/protocol as the main study. The gate
must be recorded BEFORE the full comparison is launched.

Usage: python -m udwm.scripts.dmc_gate_ratio --data runs/dmc_gate_pilot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/dmc_gate_pilot.json")
    args = p.parse_args(argv)
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))["rows"]
    rows = [r for r in rows if r["variant"] == "ordinary" and r.get("teacher_w_mean")]
    if not rows:
        raise SystemExit(f"no ordinary-arm rows with teacher_w_mean in {args.data}")
    ratios = []
    print(f'{"seed":<6}{"teacher_g":>12}{"teacher_w":>12}{"g*/w*":>10}')
    for r in sorted(rows, key=lambda x: int(x["seed"])):
        g, w = float(r["teacher_g_mean"]), float(r["teacher_w_mean"])
        ratios.append(g / max(w, 1e-12))
        print(f'{int(r["seed"]):<6}{g:>12.3f}{w:>12.5f}{g / max(w, 1e-12):>10.2f}')
    med = float(np.median(ratios))
    regime = "aleatoric-dominated" if med > 10 else ("balanced" if med >= 0.1 else "epistemic-dominated")
    print(f"\nmedian g*/w* = {med:.3f}  -> regime: {regime}")
    print("pre-committed consequence (DMC-PAYOFF-PREREGISTRATION-2026-09-05.md):")
    if regime == "aleatoric-dominated":
        print("  expect DelayedBimodal pattern: EMA collapse, equal-weight partial transfer,")
        print("  payoff question open at long horizon.")
    elif regime == "balanced":
        print("  balance-window theory predicts both w and g recover under equal-weight")
        print("  identified; transfer-to-parity question open.")
    else:
        print("  equal-weight should match w; u ~ w; report as-is.")


if __name__ == "__main__":
    main()
