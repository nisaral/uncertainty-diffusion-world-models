"""Fast, deterministic checks for the ground-truth (w*, g*) toy.

Run: python -m pytest tests/test_ground_truth_w_g.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "theory"))

import ground_truth_w_g as gt  # noqa: E402


def _setup(seed=0, grid=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    theta, eta, c = gt.make_teacher(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(grid, gt.FDIM, generator=gen)
    sigma2 = gt.sigma_fn(x, eta, c) ** 2
    return theta, eta, c, gen, x, sigma2


def _rel_mae(a, b):
    return float((a - b).abs().mean() / b.abs().mean())


def test_estimator_recovers_analytic_w_g():
    theta, eta, c, gen, x, sigma2 = _setup(0)
    w_true, g_true = gt.analytic_w_g(theta, x, sigma2)
    y = gt.sample_values(gt.member_state_means(theta, x), sigma2.sqrt(), gt.RHO, 64, gen)
    w_hat, g_hat = gt.coupled_w_g(y, 64)
    w_hat, g_hat = w_hat.squeeze(-1), g_hat.squeeze(-1)
    assert _rel_mae(w_hat, w_true) < 0.15
    assert _rel_mae(g_hat, g_true) < 0.15


def test_coupling_identity_holds_under_nonlinear_value_map():
    theta, eta, c, gen, x, sigma2 = _setup(1, grid=64)
    w_true, g_true = gt.analytic_w_g(theta, x, sigma2)
    trials = 128
    ev, means = 0.0, []
    for _ in range(trials):
        y = gt.sample_values(gt.member_state_means(theta, x), sigma2.sqrt(), gt.RHO, 1, gen)
        yb = y.squeeze(2).squeeze(-1)
        ev += float(yb.var(dim=0, unbiased=False).mean())
        means.append(yb.mean(dim=0))
    means = torch.stack(means)
    ev /= trials
    sigma_bar = float(means.var(dim=0, unbiased=False).mean())
    pred = float((w_true + g_true - sigma_bar).mean())
    assert abs(ev - pred) / float(g_true.mean()) < 0.05


def _train_arms(seed, arms, init="benign", scale=4.0, steps=350, pretrain=200):
    gt.STEPS = steps
    gt.PRETRAIN = pretrain
    gt.BATCH = 32
    torch.manual_seed(seed)
    np.random.seed(seed)
    theta, eta, c = gt.make_teacher(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    x_src = torch.randn(256, gt.FDIM, generator=gen)
    s_m_src = gt.member_state_means(theta, x_src)
    x_grid = torch.randn(128, gt.FDIM, generator=torch.Generator().manual_seed(seed + 77))
    sigma2_src = gt.sigma_fn(x_src, eta, c) ** 2 * scale ** 2
    sigma2_grid = gt.sigma_fn(x_grid, eta, c) ** 2 * scale ** 2
    return gt.run_arms(x_src, s_m_src, sigma2_src, x_grid, sigma2_grid, theta, gen,
                       init=init, arms=arms)


def test_identified_recovers_w_better_than_hybrid_from_collapsed_init():
    r = _train_arms(7, ("hybrid", "identified"), init="collapsed", scale=1.0)
    assert r["identified"]["w_rel_rmse"] < r["hybrid"]["w_rel_rmse"]
    assert r["identified"]["w_rank_corr"] > r["hybrid"]["w_rank_corr"] - 0.05


def test_equal_weighting_hole_and_ema_fix():
    r = _train_arms(11, ("identified", "identified_ema"), init="benign", scale=4.0)
    # Hole: with g* >> w*, equal weights leave w poorly recovered; the EMA
    # per-term normaliser (the fix in udwm.models.consistency.TermScaleEMA)
    # restores w ranking and magnitude.
    assert r["identified"]["w_rank_corr"] < 0.65
    assert r["identified_ema"]["w_rank_corr"] > 0.85
    assert r["identified_ema"]["w_rel_rmse"] < r["identified"]["w_rel_rmse"]


def test_w_rank_robust_to_weighting_once_normalised():
    """P4 mini-sweep: w ranking stays high across the weight range."""
    gt.STEPS = 350
    gt.PRETRAIN = 200
    gt.BATCH = 32
    torch.manual_seed(21)
    np.random.seed(21)
    theta, eta, c = gt.make_teacher(21)
    gen = torch.Generator().manual_seed(22)
    x_src = torch.randn(256, gt.FDIM, generator=gen)
    s_m_src = gt.member_state_means(theta, x_src)
    x_grid = torch.randn(128, gt.FDIM, generator=torch.Generator().manual_seed(121))
    sigma2_src = gt.sigma_fn(x_src, eta, c) ** 2 * 16.0
    sigma2_grid = gt.sigma_fn(x_grid, eta, c) ** 2 * 16.0
    ranks = []
    for r in (0.01, 0.5, 0.99):
        st = gt.Student()
        ema = gt.TermScaleEMA()
        gt.train(st, gt.ordinary_loss, x_src, s_m_src, sigma2_src, 200, gen)
        fn = lambda st_, x, s_m, sg, g, r=r: gt.identified_loss(
            st_, x, s_m, sg, g, m=2, reweight="ema", ema=ema, kw=r, kg=1.0 - r)
        gt.train(st, fn, x_src, s_m_src, sigma2_src, 350, gen)
        ranks.append(gt.evaluate(st, x_grid, theta, sigma2_grid, gen)["w_rank_corr"])
    assert all(x > 0.8 for x in ranks), ranks
