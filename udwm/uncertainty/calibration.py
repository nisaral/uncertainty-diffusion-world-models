from __future__ import annotations

from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def reliability_summary(
    predicted_std: torch.Tensor,
    residuals: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Simple calibration summary: correlation and binned |err| vs predicted std.

    predicted_std, residuals: [B] or [B,1]
    """
    ps = predicted_std.detach().cpu().reshape(-1).numpy()
    res = residuals.detach().cpu().reshape(-1).numpy()
    abs_err = np.abs(res)
    if ps.std() < 1e-12 or abs_err.std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ps, abs_err)[0, 1])
    # bin by predicted std
    edges = np.quantile(ps, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bin_errs = []
    bin_stds = []
    for i in range(len(edges) - 1):
        mask = (ps >= edges[i]) & (ps <= edges[i + 1] if i == len(edges) - 2 else ps < edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_stds.append(float(ps[mask].mean()))
        bin_errs.append(float(abs_err[mask].mean()))
    return {
        "corr_std_abserr": corr,
        "mean_pred_std": float(ps.mean()),
        "mean_abs_err": float(abs_err.mean()),
        "n_bins_used": float(len(bin_stds)),
    }
