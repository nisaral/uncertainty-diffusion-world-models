# -*- coding: utf-8 -*-
"""
Estimator-noise scaling for the identified (w, g) split (reviewer item 2).

How large is the batch-to-batch estimator noise of the two channels the
identified loss matches, relative to the coherent signal (the teacher-student
gap, ~ the teacher scale at init), as a function of the number of shared
latents M?  Two numbers decide whether a channel can be *matched from per-batch
estimates* and how much the other channel must be up-weighted before it
dominates the shared parameters:

  std(g_hat)/g*   ~ sqrt(2 / (N (M-1)))          (Gaussian within-member)
  std(w_hat)/w*   ~ sqrt(A_w / M) / w*           (A_w from ESTIMATOR-VARIANCE)

The per-channel learnability of each term is weight-independent (weights cancel
in the per-channel SNR), so weights only set the *competition* between the
channels.  This script measures the M-scaling of both relative noises and then
prints the re-adjudication of the repo's ground-truth run
(runs/ground_truth_w_g.json) for BOTH w and g per arm -- the equal-weight,
EMA-both and hybrid cells -- which is the operating-envelope evidence.

Run: python theory/identified_balance_window.py  (CPU, <1 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from theory.ground_truth_w_g import (  # noqa: E402
    N, RHO, analytic_w_g, member_state_means, sample_values, sigma_fn,
)
from udwm.models.consistency import coupled_w_g  # noqa: E402


def make_teacher(seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(N, 4, 2, generator=g) * 0.8
    eta = torch.randn(4, generator=g) * 0.5
    return theta, eta, 0.0


def main():
    t0 = time.time()
    gen = torch.Generator().manual_seed(1)
    theta, eta, c = make_teacher(0)
    x = torch.randn(512, 4, generator=gen)
    sigma2 = sigma_fn(x, eta, c) ** 2
    s = member_state_means(theta, x)
    w_star, g_star = analytic_w_g(theta, x, sigma2)
    w_star, g_star = w_star.squeeze(-1), g_star.squeeze(-1)
    print("=" * 84)
    print("P0  estimator-noise scaling of (w_hat, g_hat) with shared latents M")
    print("=" * 84)
    print(f'{"M":>4} {"std(g_hat)/g*":>14} {"pred sqrt(2/(N(M-1)))":>22} {"std(w_hat)/w*":>14} {"w* mean":>10} {"g* mean":>10}')
    out = {}
    for m in (2, 4, 8, 16, 32):
        w_hats, g_hats = [], []
        for _ in range(120):
            y = sample_values(s, sigma2.sqrt(), RHO, m, gen)  # [N,M,B,1]
            w_h, g_h = coupled_w_g(y, m)
            w_hats.append(w_h.squeeze(-1))
            g_hats.append(g_h.squeeze(-1))
        w_hats = torch.stack(w_hats)
        g_hats = torch.stack(g_hats)
        row = {
            "std_g_rel": float(g_hats.std(dim=0).mean() / g_star.mean()),
            "pred_std_g_rel": float(np.sqrt(2.0 / (N * (m - 1)))),
            "std_w_rel": float(w_hats.std(dim=0).mean() / max(float(w_star.mean()), 1e-9)),
        }
        out[m] = row
        print(f"{m:>4} {row['std_g_rel']:>14.3f} {row['pred_std_g_rel']:>22.3f} "
              f"{row['std_w_rel']:>14.3f} {float(w_star.mean()):>10.4f} {float(g_star.mean()):>10.4f}")
    print("  -> both relative noises fall ~1/sqrt(M); at M=2 the w-channel's")
    print("     per-batch noise is >100% of w* itself, so an equal-weight loss")
    print("     cannot resolve w from per-batch estimates, while the g-channel")
    print("     noise is ~g* (SNR ~ 1).  Up-weighting w (EMA) is required to")
    print("     resolve w -- and the EMA ratio (g*/w*)^2 is what then starves g.")

    print()
    print("=" * 84)
    print("Re-adjudication of runs/ground_truth_w_g.json: BOTH channels per arm")
    print("=" * 84)
    gt = json.load(open(ROOT / "runs" / "ground_truth_w_g.json", encoding="utf-8"))
    header = f'{"part":<16}{"arm":<16}{"w_rank":>8}{"w_hat_mean":>12}{"w*":>8}{"g_rank":>8}{"g_hat_mean":>12}{"g*":>10}'
    print(header)
    for part in ("part1_benign", "part2_collapsed", "part3_hole"):
        for arm, r in gt[part].items():
            print(f"{part:<16}{arm:<16}{r['w_rank_corr']:>8.3f}{r['w_hat_mean']:>12.1f}{r['w_mean']:>8.1f}"
                  f"{r['g_rank_corr']:>8.3f}{r['g_hat_mean']:>12.1f}{r['g_mean']:>10.1f}")
    print("  -> equal-weight identified recovers g (g_rank 0.90-0.97) but leaves")
    print("     w at the hole (w_rank 0.40 in part3); EMA-both fixes w (0.97-0.99)")
    print("     and annihilates g (g_rank ~0.1, g_hat ~ 0.03-0.4 vs g* 46-3966).")
    print("     The toy reproduces the real-stack collapse: the 'fix' adjudicated")
    print("     only w.  No single scalar weight recovers both at g* >> w*.")
    out_path = ROOT / "runs" / "identified_balance_window.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"P0": out}, indent=2, default=float))
    print(f"wrote {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
