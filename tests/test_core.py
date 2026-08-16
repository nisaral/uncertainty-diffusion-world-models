"""Unit tests for shared core components."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udwm.data.replay_buffer import ReplayBuffer
from udwm.models.world_model import WorldModel
from udwm.uncertainty.mc_ube import MCUBELocalRewards, UNetwork


def test_replay_buffer():
    device = torch.device("cpu")
    buf = ReplayBuffer(100, obs_dim=3, action_dim=1, device=device)
    for i in range(10):
        buf.add(np.zeros(3), np.zeros(1), 0.0, np.ones(3), 0.0)
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 3)


def test_diffusion_forward_backward():
    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(32, 32), diffusion_steps=4, sample_steps=2
    )
    obs = torch.randn(8, 3)
    act = torch.randn(8, 1)
    nxt = obs + 0.05 * torch.randn_like(obs)
    rew = torch.randn(8, 1)
    done = torch.zeros(8, 1)
    loss = wm.train_loss(obs, act, nxt, rew, done)["total"]
    loss.backward()
    assert torch.isfinite(loss)


def test_mc_ube_debias_removes_finite_m_bias():
    """u_hat must be ~unbiased for u* = w* - g*; the naive form is biased +O(1/M)."""
    torch.manual_seed(0)
    n, m, trials = 5, 8, 4000
    mu = torch.tensor([0.0, 0.5, 1.2, -0.3, 0.8])
    sigma2 = torch.tensor([0.25, 0.50, 0.10, 0.80, 0.35])
    w_star = float(((mu - mu.mean()) ** 2).mean())
    g_star = float(sigma2.mean())
    u_star = w_star - g_star

    y = mu[None, :, None] + sigma2.sqrt()[None, :, None] * torch.randn(trials, n, m)
    means = y.mean(-1).T.unsqueeze(-1)                    # [N, trials, 1]
    vars_ub = y.var(-1, unbiased=True).T.unsqueeze(-1)

    naive = MCUBELocalRewards(u_min=-1e9, debias=False)
    deb = MCUBELocalRewards(u_min=-1e9, debias=True)
    u_naive = float(naive.combine(means, vars_ub, m_eff=m)["u"].mean())
    u_deb = float(deb.combine(means, vars_ub, m_eff=m)["u"].mean())

    predicted = g_star * (2 * n - 1) / (n * m)
    # Finite-trial MC noise around the O(1/M) formula; check order, not digits.
    assert abs((u_naive - u_star) - predicted) < 0.08, "naive bias must match O(1/M) formula"
    assert abs(u_deb - u_star) < 0.02, "debiased estimator must be ~unbiased"
    assert abs(u_deb - u_star) < abs(u_naive - u_star)


def test_mc_ube_debias_tracks_mean_noise_scale():
    """The correction must scale with c^2 where Var(mean) = c^2 sigma^2/M.

    Guards the Gaussian branch: an exact plug-in mean (c=0) must not be
    corrected at all, and a (1-b)*plug_in + b*mc blend must be corrected by b^2,
    not by 1.  Using a bool here silently over/under-subtracts.
    """
    torch.manual_seed(0)
    n, m, trials = 5, 8, 20000
    mu = torch.tensor([0.0, 0.5, 1.2, -0.3, 0.8])
    sigma2 = torch.tensor([0.25, 0.50, 0.10, 0.80, 0.35])
    w_star = float(((mu - mu.mean()) ** 2).mean())
    u_star = w_star - float(sigma2.mean())

    y = mu[None, :, None] + sigma2.sqrt()[None, :, None] * torch.randn(trials, n, m)
    vars_ub = y.var(-1, unbiased=True).T.unsqueeze(-1)
    mc_mean = y.mean(-1).T.unsqueeze(-1)
    exact = mu[:, None, None].expand(n, trials, 1)        # closed-form plug-in

    est = MCUBELocalRewards(u_min=-1e9, debias=True)
    # c = 0: means carry no MC noise, so only the Bessel correction applies.
    u_exact = float(est.combine(exact, vars_ub, m_eff=m, mean_noise_scale=0.0)["u"].mean())
    assert abs(u_exact - u_star) < 0.01

    # c = beta: blended mean. Correcting as if c=1 over-subtracts by (1-beta^2).
    beta = 0.5
    blend = (1.0 - beta) * exact + beta * mc_mean
    u_blend = float(est.combine(blend, vars_ub, m_eff=m, mean_noise_scale=beta)["u"].mean())
    u_wrong = float(est.combine(blend, vars_ub, m_eff=m, mean_noise_scale=1.0)["u"].mean())
    assert abs(u_blend - u_star) < 0.01, "blend must be unbiased at c=beta"
    assert abs(u_wrong - u_star) > abs(u_blend - u_star), "c=1 must be visibly worse"


def _correlated_draws(mu, sigma2, rho, m, trials, gen):
    """Y[i,m] with Var(Y_i)=sigma2_i and Corr(Y_i,Y_j)=rho for i!=j.

    Shape [N, m, trials, 1] to match combine_coupled's [N,M,B,1].
    """
    n = mu.shape[0]
    common = torch.randn(1, m, trials, 1, generator=gen)
    idio = torch.randn(n, m, trials, 1, generator=gen)
    sd = sigma2.sqrt()[:, None, None, None]
    shared = float(rho) ** 0.5
    return mu[:, None, None, None] + sd * (
        shared * common + (1.0 - float(rho)) ** 0.5 * idio
    )


def test_combine_coupled_matches_combine_when_independent():
    """The coupled estimator must reduce to the independent one at rho=0."""
    gen = torch.Generator().manual_seed(0)
    n, m, trials = 5, 8, 6000
    mu = torch.tensor([0.0, 0.5, 1.2, -0.3, 0.8])
    sigma2 = torch.tensor([0.25, 0.50, 0.10, 0.80, 0.35])
    u_star = float(((mu - mu.mean()) ** 2).mean()) - float(sigma2.mean())

    q = _correlated_draws(mu, sigma2, 0.0, m, trials, gen)
    est = MCUBELocalRewards(u_min=-1e9, debias=True)
    out = est.combine_coupled(q)
    assert abs(float(out["u"].mean()) - u_star) < 0.02
    # coupling diagnostic ~0 when members are independent
    assert abs(float(out["coupling"].mean())) < 0.05

    # and it must agree with combine() fed the same per-member statistics
    ref = est.combine(
        q.mean(dim=1), q.var(dim=1, unbiased=True), m_eff=m, mean_noise_scale=1.0
    )
    assert abs(float(out["u"].mean()) - float(ref["u"].mean())) < 0.01


def test_combine_coupled_unbiased_under_correlation():
    """Under a coupling, the independent correction over-subtracts; coupled does not.

    This is the whole point of estimate_coupled: with shared latents the members'
    MC errors are correlated, the true bias is (g* - mean(Sigma))/M rather than
    (N-1)/N * g*/M, and only the covariance-aware form stays unbiased.
    """
    gen = torch.Generator().manual_seed(1)
    n, m, trials, rho = 5, 8, 6000, 0.8
    mu = torch.tensor([0.0, 0.5, 1.2, -0.3, 0.8])
    sigma2 = torch.tensor([0.25, 0.50, 0.10, 0.80, 0.35])
    u_star = float(((mu - mu.mean()) ** 2).mean()) - float(sigma2.mean())

    q = _correlated_draws(mu, sigma2, rho, m, trials, gen)
    est = MCUBELocalRewards(u_min=-1e9, debias=True)

    u_coupled = float(est.combine_coupled(q)["u"].mean())
    u_indep_form = float(
        est.combine(
            q.mean(dim=1), q.var(dim=1, unbiased=True), m_eff=m, mean_noise_scale=1.0
        )["u"].mean()
    )
    assert abs(u_coupled - u_star) < 0.02, "coupled form must stay unbiased"
    assert abs(u_indep_form - u_star) > abs(u_coupled - u_star), (
        "independent correction must visibly over-subtract under correlation"
    )
    assert float(est.combine_coupled(q)["coupling"].mean()) > 0.5


def test_coupling_reduces_estimator_variance():
    """Correlated draws must give a lower-variance w than independent ones.

    The variance reduction — not the bias removal — is what would let M shrink,
    so it is the claim worth guarding.
    """
    gen = torch.Generator().manual_seed(2)
    m, trials = 4, 4000
    mu = torch.tensor([0.0, 0.5, 1.2, -0.3, 0.8])
    sigma2 = torch.tensor([0.25, 0.50, 0.10, 0.80, 0.35])
    est = MCUBELocalRewards(u_min=-1e9, debias=True)

    w_indep = est.combine_coupled(_correlated_draws(mu, sigma2, 0.0, m, trials, gen))["w"]
    w_coup = est.combine_coupled(_correlated_draws(mu, sigma2, 0.9, m, trials, gen))["w"]
    assert float(w_coup.std()) < float(w_indep.std())


def test_sample_next_coupled_shares_latents_across_members():
    """Mechanical check: identical members + shared latents => identical samples.

    Deterministic, so it pins the coupling mechanism itself rather than a
    statistical side effect of it.
    """
    torch.manual_seed(0)
    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(32, 32),
        diffusion_steps=4, sample_steps=2,
    )
    dyn = wm.dynamics
    dyn.members[1].load_state_dict(dyn.members[0].state_dict())
    obs, act = torch.randn(6, 3), torch.randn(6, 1)

    nxt, _ = dyn.sample_next_coupled(obs, act, m=3, coupled=True)
    assert nxt.shape == (2, 3, 6, 3)
    assert torch.allclose(nxt[0], nxt[1], atol=1e-5), "shared x_T must give equal draws"

    nxt_i, _ = dyn.sample_next_coupled(obs, act, m=3, coupled=False)
    assert not torch.allclose(nxt_i[0], nxt_i[1], atol=1e-5)


def test_diffusion_predict_mean_reduces_spread():
    """MC-mean prediction must be tighter than single stochastic draws."""
    torch.manual_seed(0)
    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(32, 32),
        diffusion_steps=4, sample_steps=2,
    )
    obs = torch.randn(16, 3)
    act = torch.randn(16, 1)
    mean_n, mean_r = wm.dynamics.predict_mean(obs, act, m=8)
    assert mean_n.shape == obs.shape
    assert mean_r.shape == (16, 1)
    assert torch.isfinite(mean_n).all()
    draws = torch.stack([wm.dynamics.sample_next(obs, act) for _ in range(8)], 0)
    assert mean_n.std() <= draws.std() + 1e-6


def test_gaussian_forward_backward():
    wm = WorldModel.build("gaussian", 3, 1, ensemble_size=2, hidden_dims=(32, 32))
    obs = torch.randn(8, 3)
    act = torch.randn(8, 1)
    nxt = obs + 0.05 * torch.randn_like(obs)
    rew = torch.randn(8, 1)
    done = torch.zeros(8, 1)
    loss = wm.train_loss(obs, act, nxt, rew, done)["total"]
    loss.backward()
    assert torch.isfinite(loss)


def test_mc_ube_shapes():
    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(32, 32), diffusion_steps=4, sample_steps=2
    )
    u_net = UNetwork(3, 1, (32, 32))
    mc = MCUBELocalRewards(m_samples=2)
    obs = torch.randn(4, 3)
    act = torch.randn(4, 1)

    def q_fn(o, a):
        return u_net(o, a)  # any scalar head

    def policy_fn(o):
        return torch.zeros(o.shape[0], 1)

    out = mc.estimate(wm, q_fn, policy_fn, obs, act)
    assert out["u"].shape == (4, 1)
    assert out["w"].shape == (4, 1)


def test_toy_ube_mdp_exact():
    """UBE fixed point matches Var(V) on the enumerable DAG MRP."""
    import importlib.util
    import sys

    path = ROOT / "theory" / "toy_ube_mdp.py"
    spec = importlib.util.spec_from_file_location("toy_ube_mdp", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["toy_ube_mdp"] = mod
    spec.loader.exec_module(mod)
    mrp = mod.EnsembleMRP(ps=[0.2, 0.5, 0.8], r1=10.0, r2=0.0, gamma=0.9)
    _, var_v, _ = mrp.posterior_mean_var()
    w = mrp.closed_form_w()
    ube_u = (mrp.gamma**2) * w
    assert abs(ube_u - var_v) < 1e-9


def test_joint_reward_diffusion():
    wm = WorldModel.build(
        "diffusion",
        3,
        1,
        ensemble_size=2,
        hidden_dims=(32, 32),
        diffusion_steps=4,
        sample_steps=2,
        joint_with_diffusion=True,
    )
    assert wm.joint_reward
    obs = torch.randn(8, 3)
    act = torch.randn(8, 1)
    nxt = obs + 0.05 * torch.randn_like(obs)
    rew = torch.randn(8, 1)
    done = torch.zeros(8, 1)
    loss = wm.train_loss(obs, act, nxt, rew, done)["total"]
    loss.backward()
    o2, r2 = wm.dynamics.sample_next_with_reward(obs[:4], act[:4])
    assert o2.shape == (4, 3)
    assert r2.shape == (4, 1)


def test_selective_and_audit():
    from udwm.data.audit_log import GateAuditLog
    from udwm.eval.selective import risk_coverage_curve, selective_report, threshold_sweep

    rng = np.random.default_rng(0)
    score = rng.random(200)
    err = score + 0.1 * rng.random(200)
    report = selective_report(score, err)
    assert report["n"] == 200
    assert "risk_coverage" in report
    assert "threshold_sweep" in report
    assert report["rank_corr_score_vs_error"] > 0.3
    curve = risk_coverage_curve(score, err, n_points=5)
    assert curve[0]["coverage"] > curve[-1]["coverage"]
    sweep = threshold_sweep(score, err)
    assert sweep[0]["over_rejection"] >= 0

    log = GateAuditLog(path=None, enabled=True)
    obs = torch.zeros(4, 2, 3)
    act = torch.zeros(4, 2, 1)
    su = torch.ones(4, 2, 1) * 2.0
    w = torch.ones(4, 2, 1) * 0.2
    d = torch.zeros(4, 2, 1)
    log.record_batch(1, obs, act, su, w, d, stop_threshold=1.0, mode="both")
    s = log.summary()
    assert s["n_logged"] > 0
    assert s["frac_abstain_stop"] > 0


def test_u_gated_rollout_and_adaptive_mc():
    from udwm.models.world_model import WorldModel
    from udwm.rl.u_gated_imagination import u_gated_rollout
    from udwm.uncertainty.adaptive_mc import AdaptiveMCUBELocalRewards
    from udwm.uncertainty.mc_ube import UNetwork

    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(32, 32),
        diffusion_steps=4, sample_steps=2,
    )
    u_net = UNetwork(3, 1, (32, 32))
    obs = torch.randn(8, 3)

    def policy_fn(o):
        return torch.zeros(o.shape[0], 1)

    roll = u_gated_rollout(
        wm, policy_fn, u_net, obs, horizon=2, mode="both",
        stop_percentile=0.7, weight_beta=1.0,
    )
    assert roll["obs"].shape == (8, 2, 3)
    assert roll["weights"].shape == (8, 2, 1)
    assert "stopped_frac" in roll

    def q_fn(o, a):
        return u_net(o, a)

    adaptive = AdaptiveMCUBELocalRewards(m_min=2, m_max=6, m_probe=2, enabled=True)
    act = torch.randn(8, 1)
    out = adaptive.estimate(wm, q_fn, policy_fn, obs, act)
    assert out["u"].shape == (8, 1)
    assert "m_mean" in out
    assert "refine_frac_used" in out


def test_consistency_distill_build():
    wm = WorldModel.build(
        "diffusion",
        3,
        1,
        ensemble_size=2,
        hidden_dims=(32, 32),
        diffusion_steps=4,
        sample_steps=2,
        use_consistency_distill=True,
    )
    from udwm.models.consistency import DistilledWorldModel

    assert isinstance(wm, DistilledWorldModel)
    obs = torch.randn(8, 3)
    act = torch.randn(8, 1)
    nxt = obs + 0.05 * torch.randn_like(obs)
    rew = torch.randn(8, 1)
    done = torch.zeros(8, 1)
    loss = wm.train_loss(obs, act, nxt, rew, done)["total"]
    loss.backward()
    assert torch.isfinite(loss)

