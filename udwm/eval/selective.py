"""Selective prediction / abstention diagnostics for √U.

SOP stance (do not overclaim): a ranking score is not automatically a
decision threshold. This module *measures* what happens if we treat √U
as a threshold — risk–coverage, over-rejection, empirical reliability.

It does **not** implement conformal prediction or give finite-sample
coverage guarantees. Those are the theory the SOP says you want to *learn*.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


@torch.no_grad()
def collect_score_and_error(
    agent,
    u_net,
    buffer,
    batch_size: int = 256,
    n_batches: int = 4,
) -> Dict[str, np.ndarray]:
    """Pairs √U(s,a) with |TD residual| on buffer data."""
    scores, errors = [], []
    gamma = agent.gamma
    n = min(batch_size, max(1, len(buffer)))
    for _ in range(n_batches):
        b = buffer.sample(n)
        obs, act = b["obs"], b["actions"]
        rew, nxt, done = b["rewards"], b["next_obs"], b["dones"]
        q = agent.q_min(obs, act)
        next_a = agent.policy_tensor(nxt, deterministic=True)
        q_next = agent.critic_target.min_q(nxt, next_a)
        td = rew + (1.0 - done) * gamma * q_next - q
        u = u_net(obs, act)
        scores.append(torch.sqrt(u + 1e-8).reshape(-1).cpu().numpy())
        errors.append(td.abs().reshape(-1).cpu().numpy())
    return {
        "score": np.concatenate(scores, 0),
        "abs_td": np.concatenate(errors, 0),
    }


def risk_coverage_curve(
    score: np.ndarray,
    abs_error: np.ndarray,
    n_points: int = 11,
) -> List[Dict[str, float]]:
    """Keep the lowest-score fraction ``coverage``; report mean |error| on kept set.

    coverage=1.0 → keep all (no abstention).
    coverage=0.5 → abstain on the top 50% scores (highest √U).
    """
    score = np.asarray(score).reshape(-1)
    abs_error = np.asarray(abs_error).reshape(-1)
    order = np.argsort(score)  # low score first = more trusted
    err_sorted = abs_error[order]
    n = len(score)
    rows = []
    for frac in np.linspace(1.0, 0.1, n_points):
        k = max(1, int(round(frac * n)))
        kept = err_sorted[:k]
        abstained = err_sorted[k:] if k < n else np.array([])
        rows.append(
            {
                "coverage": float(k / n),
                "abstention_rate": float(1.0 - k / n),
                "risk_mean_abs_td": float(kept.mean()),
                "abstained_mean_abs_td": float(abstained.mean()) if len(abstained) else float("nan"),
            }
        )
    return rows


def threshold_sweep(
    score: np.ndarray,
    abs_error: np.ndarray,
    error_high_quantile: float = 0.8,
) -> List[Dict[str, float]]:
    """Treat score as an *absolute* threshold (the DIO failure mode).

    Define a 'bad' transition as |TD| above a quantile of |TD|.
    At each τ, abstain if score > τ.
    Report:
      - recall of bad transitions (caught)
      - over_rejection: fraction of *good* transitions abstained
      - precision of abstention
    This is measurement, not a guarantee.
    """
    score = np.asarray(score).reshape(-1)
    abs_error = np.asarray(abs_error).reshape(-1)
    bad_cut = float(np.quantile(abs_error, error_high_quantile))
    bad = abs_error >= bad_cut
    good = ~bad
    taus = np.quantile(score, np.linspace(0.2, 0.95, 8))
    rows = []
    for tau in taus:
        abstain = score > tau
        tp = float(np.logical_and(abstain, bad).sum())
        fp = float(np.logical_and(abstain, good).sum())
        fn = float(np.logical_and(~abstain, bad).sum())
        n_good = float(good.sum()) or 1.0
        n_bad = float(bad.sum()) or 1.0
        n_abs = float(abstain.sum()) or 1.0
        rows.append(
            {
                "tau": float(tau),
                "abstention_rate": float(abstain.mean()),
                "recall_bad": tp / n_bad,
                "over_rejection": fp / n_good,
                "precision_abstain": tp / n_abs,
                "missed_bad": fn / n_bad,
                "bad_td_cut": bad_cut,
            }
        )
    return rows


def spearman_like(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without scipy (Spearman via Pearson on ranks)."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def selective_report(score: np.ndarray, abs_error: np.ndarray) -> Dict[str, object]:
    return {
        "n": int(len(score)),
        "score_mean": float(np.mean(score)),
        "abs_td_mean": float(np.mean(abs_error)),
        "rank_corr_score_vs_error": spearman_like(score, abs_error),
        "risk_coverage": risk_coverage_curve(score, abs_error),
        "threshold_sweep": threshold_sweep(score, abs_error),
        "disclaimer": (
            "Empirical selective-prediction diagnostics only. "
            "Not conformal coverage; ranking √U ≠ a calibrated decision threshold."
        ),
    }
