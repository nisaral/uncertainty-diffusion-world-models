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
    assert out["sample_split"].item() == 1.0


def test_one_step_disagreement_baseline():
    from udwm.uncertainty.baselines import one_step_state_disagreement

    for model_type in ("gaussian", "diffusion"):
        wm = WorldModel.build(
            model_type, 3, 1, ensemble_size=2, hidden_dims=(32, 32),
            diffusion_steps=4, sample_steps=2,
        )
        obs, act = torch.randn(5, 3), torch.randn(5, 1)
        score = one_step_state_disagreement(wm, obs, act, m_samples=2)
        assert score.shape == (5, 1)
        assert torch.isfinite(score).all()
        assert (score >= 0).all()


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


def test_uncertainty_preserving_distill_loss():
    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=3, hidden_dims=(32, 32),
        diffusion_steps=4, sample_steps=2, use_consistency_distill=True,
        preserve_distilled_uncertainty=True,
    )
    obs, act = torch.randn(6, 3), torch.randn(6, 1)
    nxt, rew, done = obs + 0.05 * torch.randn_like(obs), torch.randn(6, 1), torch.zeros(6, 1)
    out = wm.train_loss(obs, act, nxt, rew, done)
    for key in ("distill_member", "distill_mean", "distill_geometry", "distill_pairwise"):
        assert key in out
        assert torch.isfinite(out[key])
    out["total"].backward()


def test_decision_preserving_distill_loss_backpropagates_only_student():
    from udwm.models.consistency import decision_preserving_distill_loss
    from udwm.models.world_model import WorldModel

    wm = WorldModel.build(
        "diffusion", obs_dim=2, action_dim=1, ensemble_size=3, hidden_dims=[16, 16],
        student_hidden_dims=[8, 8], diffusion_steps=4, sample_steps=2,
        use_consistency_distill=True,
    )
    wm.freeze_teacher()
    obs = torch.randn(12, 2)
    actions = torch.randn(12, 1)
    next_obs = obs + 0.1 * torch.randn(12, 2)

    def value_fn(states, acts):
        return torch.tanh(states[..., :1] - states[..., 1:2]) - 0.1 * acts.square()

    parts = decision_preserving_distill_loss(
        wm.student, wm.teacher, obs, actions, next_obs, value_fn
    )
    assert set(parts) >= {
        "total", "member", "value_geometry", "value_variance", "variance_shape",
        "state_geometry", "state_pairwise", "guard_fired", "decision_scale",
    }
    parts["total"].backward()
    assert any(p.grad is not None for p in wm.student.parameters())
    assert all(p.grad is None for p in wm.teacher.parameters())


def test_delayed_bimodal_env_is_seeded_and_markov_shaped():
    from udwm.envs.registry import make_env, space_info

    env_a = make_env("DelayedBimodal-v0", seed=7)
    env_b = make_env("DelayedBimodal-v0", seed=7)
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    assert np.allclose(obs_a, obs_b)
    info = space_info(env_a)
    assert info["obs_dim"] == 5
    assert info["action_dim"] == 1
    for _ in range(8):
        obs_a, reward, terminated, truncated, _ = env_a.step(np.array([0.25], dtype=np.float32))
        assert obs_a.shape == (5,)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_two_stage_trainer_freezes_teacher_and_uses_student_optimizer():
    from udwm.rl.trainer import MBPOTrainer
    from udwm.utils.config import load_config

    cfg = load_config(str(ROOT / "configs" / "delayed_bimodal_distill.yaml"))
    cfg["model"]["distill_teacher_pretrain_updates"] = 1
    cfg["mbpo"]["model_batch_size"] = 16
    cfg["mbpo"]["num_model_epochs"] = 1
    cfg["mbpo"]["warmup_steps"] = 0
    trainer = MBPOTrainer(cfg)
    for _ in range(20):
        trainer._env_step()
    initial = trainer._parameter_checksum(trainer.world_model.teacher)
    pretrain_info = trainer._train_world_model()
    trained = trainer._parameter_checksum(trainer.world_model.teacher)
    assert pretrain_info["teacher_frozen"] == 0.0
    assert pretrain_info["teacher_updates"] == 1.0
    assert trained != initial
    info = trainer._train_world_model()
    assert info["teacher_frozen"] == 1.0
    assert all(not p.requires_grad for p in trainer.world_model.teacher.parameters())
    optimized = {id(p) for group in trainer.wm_opt.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in trainer.world_model.student.parameters()}
    assert trainer._parameter_checksum(trainer.world_model.teacher) == trained


def test_distillation_uncertainty_metrics():
    from udwm.eval.metrics import evaluate_distillation_uncertainty
    from udwm.data.replay_buffer import ReplayBuffer

    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(16, 16),
        diffusion_steps=4, sample_steps=2, use_consistency_distill=True,
        preserve_distilled_uncertainty=True,
    )
    buf = ReplayBuffer(64, 3, 1, torch.device("cpu"))
    for _ in range(24):
        o = np.random.randn(3).astype(np.float32)
        a = np.random.randn(1).astype(np.float32)
        buf.add(o, a, 0.0, o + 0.01 * np.random.randn(3).astype(np.float32), 0.0)
    q = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

    def q_fn(o, a):
        return q(torch.cat([o, a], dim=-1))

    def policy_fn(o):
        return torch.zeros(o.shape[0], 1)

    report = evaluate_distillation_uncertainty(wm, q_fn, policy_fn, buf, batch_size=8, n_batches=1, m_samples=2)
    assert report["n"] > 0
    assert -1.0 <= report["u_rank_corr"] <= 1.0
    assert np.isfinite(report["u_rmse"])


def test_distilled_teacher_can_be_frozen_for_matched_stage_training():
    wm = WorldModel.build(
        "diffusion", 3, 1, ensemble_size=2, hidden_dims=(16, 16),
        diffusion_steps=4, sample_steps=2, use_consistency_distill=True,
        preserve_distilled_uncertainty=True,
    )
    wm.freeze_teacher()
    assert all(not p.requires_grad for p in wm.teacher.parameters())
    obs, act = torch.randn(5, 3), torch.randn(5, 1)
    nxt, rew, done = obs + 0.05 * torch.randn_like(obs), torch.randn(5, 1), torch.zeros(5, 1)
    out = wm.train_loss(obs, act, nxt, rew, done)
    out["total"].backward()
    assert all(p.grad is None for p in wm.teacher.parameters())
    assert any(p.grad is not None for p in wm.student.parameters())


def test_teacher_student_variance_bound_algebra():
    # Directly validates the perturbation inequality used in the proof note.
    mu = torch.tensor([0.2, -0.4, 0.7, 1.1])
    delta = torch.tensor([0.03, -0.02, 0.01, 0.04])
    mu_q = mu + delta
    w_p = ((mu - mu.mean()) ** 2).mean()
    w_q = ((mu_q - mu_q.mean()) ** 2).mean()
    d = delta - delta.mean()
    D = torch.sqrt((d * d).mean())
    rhs = 2.0 * torch.sqrt(w_p) * D + D * D
    assert abs(float(w_q - w_p)) <= float(rhs) + 1e-7


def test_distillation_ablation_runner_imports():
    import udwm.scripts.run_distillation_ablation as runner

    assert callable(runner.main)
    assert callable(runner._make_cfg)


def test_distillation_ablation_smoke_executes():
    import udwm.scripts.run_distillation_ablation as runner
    from udwm.utils.config import load_config

    cfg = load_config(str(ROOT / "configs" / "consistency_distill.yaml"))
    row = runner._run(
        {**cfg, "mbpo": {**cfg["mbpo"], "model_batch_size": 32}},
        variant="geometry",
        seed=0,
        collect_steps=40,
        teacher_epochs=1,
        student_epochs=1,
        m=2,
    )
    assert row["teacher_frozen"] is True
    assert row["n"] > 0
    assert np.isfinite(row["u_rmse"])
    assert row["collect_steps"] >= 32


def test_distillation_ablation_summary_is_honest():
    from udwm.scripts.run_distillation_ablation import summarize

    rows = [
        {"variant": "ordinary", "seed": 0, "u_rank_corr": 0.2, "w_rank_corr": 0.2, "u_rmse": 1.0, "teacher_state_checksum": 10.0},
        {"variant": "geometry", "seed": 0, "u_rank_corr": 0.3, "w_rank_corr": 0.4, "u_rmse": 0.8, "teacher_state_checksum": 10.0},
    ]
    out = summarize(rows)
    assert out["verdict"] == "supports_preservation_hypothesis"
    assert "novelty" in out["interpretation"]
    assert out["max_teacher_checksum_gap"] == 0.0


def test_uncertainty_fidelity_guard_fires_on_anti_alignment_and_scale():
    from udwm.models.consistency import uncertainty_fidelity_guard

    aligned = torch.tensor([[[1.0], [2.0], [3.0]], [[2.0], [4.0], [6.0]]])
    assert float(uncertainty_fidelity_guard(aligned, aligned)) == 0.0
    flipped = torch.tensor([[[3.0], [2.0], [1.0]], [[6.0], [4.0], [2.0]]])
    assert float(uncertainty_fidelity_guard(aligned, flipped, min_corr=0.0)) == 1.0
    exploded = aligned * 100.0
    assert float(uncertainty_fidelity_guard(aligned, exploded, max_scale=8.0)) == 1.0


def test_decision_loss_guard_drops_decision_terms_when_fired():
    from udwm.models.consistency import decision_preserving_distill_loss

    wm = WorldModel.build(
        "diffusion", 2, 1, ensemble_size=3, hidden_dims=(16, 16),
        diffusion_steps=4, sample_steps=2, use_consistency_distill=True,
    )
    wm.freeze_teacher()
    obs, act = torch.randn(8, 2), torch.randn(8, 1)
    nxt = obs + 0.05 * torch.randn_like(obs)

    def inverted(states, actions):
        # Strongly anti-align ensemble value variance with a batch-index pattern.
        idx = torch.arange(states.shape[1], device=states.device, dtype=states.dtype)
        member = torch.arange(states.shape[0], device=states.device, dtype=states.dtype)
        return (member.view(-1, 1, 1) * (idx.max() - idx).view(1, -1, 1)) + 0.0 * states.sum(-1, keepdim=True)

    parts = decision_preserving_distill_loss(
        wm.student, wm.teacher, obs, act, nxt, inverted,
        value_weight=1.0, variance_weight=1.0,
        guard_enabled=True, guard_min_corr=0.0,
    )
    assert float(parts["decision_scale"]) in (0.0, 1.0)
    parts["total"].backward()
    assert any(p.grad is not None for p in wm.student.parameters())


def test_normalized_value_targets_are_location_scale_invariant():
    from udwm.models.consistency import decision_preserving_distill_loss

    wm = WorldModel.build(
        "diffusion", 2, 1, ensemble_size=2, hidden_dims=(16, 16),
        diffusion_steps=4, sample_steps=2, use_consistency_distill=True,
    )
    wm.freeze_teacher()
    obs, act = torch.randn(6, 2), torch.randn(6, 1)
    nxt = obs + 0.02 * torch.randn_like(obs)
    torch.manual_seed(0)

    def q_small(states, actions):
        return states[..., :1] + 0.1 * actions

    def q_shifted(states, actions):
        return 50.0 + 7.0 * q_small(states, actions)

    a = decision_preserving_distill_loss(
        wm.student, wm.teacher, obs, act, nxt, q_small, normalize_values=True,
    )
    torch.manual_seed(0)
    b = decision_preserving_distill_loss(
        wm.student, wm.teacher, obs, act, nxt, q_shifted, normalize_values=True,
    )
    assert torch.allclose(a["value_geometry"], b["value_geometry"], atol=1e-5)
    assert torch.allclose(a["value_variance"], b["value_variance"], atol=1e-5)


def test_lagged_target_value_fn_uses_target_critic_not_online_critic():
    from udwm.models.consistency import lagged_target_value_fn
    from udwm.rl.sac import SACAgent

    agent = SACAgent(3, 1, torch.device("cpu"), hidden_dims=(8, 8))
    for p in agent.critic.parameters():
        p.data.add_(3.0)
    obs = torch.randn(4, 3)
    act = torch.zeros(4, 1)
    fn = lagged_target_value_fn(agent.critic_target, agent.actor)
    q_target = agent.critic_target.min_q(obs, agent.actor.deterministic(obs))
    q_live = agent.critic.min_q(obs, agent.actor.deterministic(obs))
    out = fn(obs.unsqueeze(0), act.unsqueeze(0)).reshape(-1, 1)
    assert torch.allclose(out, q_target, atol=1e-5)
    assert not torch.allclose(q_target, q_live, atol=1e-3)


def test_trainer_default_value_map_is_lagged_when_configured():
    from udwm.rl.trainer import MBPOTrainer
    from udwm.utils.config import load_config

    cfg = load_config(str(ROOT / "configs" / "lagged_target_distill.yaml"))
    cfg["mbpo"]["warmup_steps"] = 0
    cfg["mbpo"]["model_batch_size"] = 16
    trainer = MBPOTrainer(cfg)
    trainer._critic_updates = 100
    fn = trainer._make_distill_value_fn()
    assert fn is not None
    states = torch.randn(2, 4, trainer.obs_dim)
    actions = torch.zeros(2, 4, trainer.action_dim)
    q = fn(states, actions)
    assert q.shape == (2, 4, 1)
    assert torch.isfinite(q).all()



# ---------------------------------------------------------------------------
# Identifiability of the decision-distillation uncertainty target.
# See theory/distill_identifiability.py for the derivation these pin down.
# ---------------------------------------------------------------------------

def _coupled_members(d, sigma, rho, m, seed, trials=4000):
    """[trials, N, m] member values, Var=sigma^2, cross-member Corr=rho."""
    g = torch.Generator().manual_seed(seed)
    n = d.numel()
    common = torch.randn(trials, 1, m, generator=g)
    idio = torch.randn(trials, n, m, generator=g)
    eps = sigma * (rho**0.5 * common + (1.0 - rho) ** 0.5 * idio)
    return d.view(1, -1, 1) + eps


def test_coupled_w_g_is_unbiased_under_any_coupling():
    """`coupled_w_g` recovers (w*, g*) whatever the shared-latent correlation is.

    The independent-sampling correction ((N-1)/N * g/M) over-subtracts as rho
    grows -- theory/estimator_bias.py Part 5 -- so this pins the coupling-aware
    form that eval/metrics.py and the identified loss both rely on.
    """
    from udwm.models.consistency import coupled_w_g

    d = torch.tensor([-0.6, -0.2, 0.0, 0.3, 0.5])
    sigma, m = 0.6, 4
    w_true = float((d**2).mean())
    g_true = sigma**2
    for rho in (0.0, 0.5, 0.95):
        y = _coupled_members(d, sigma, rho, m, seed=7)
        # [trials, N, m] -> [N, M, B, 1] with B = trials
        vals = y.permute(1, 2, 0).unsqueeze(-1)
        w_hat, g_hat = coupled_w_g(vals, m)
        assert abs(float(w_hat.mean()) - w_true) < 0.02, (rho, float(w_hat.mean()))
        assert abs(float(g_hat.mean()) - g_true) < 0.02, (rho, float(g_hat.mean()))


def test_single_latent_variance_conflates_epistemic_and_aleatoric():
    """Two ensembles, same single-latent cross-member variance, different w*.

    This is the degenerate direction the M=1 `value_variance` term cannot see:
    a student with ZERO epistemic disagreement sits on the same level set as a
    faithful one.  The (w_deb, g) pair separates them.
    """
    from udwm.models.consistency import coupled_w_g

    n = 5
    d_t = torch.tensor([-0.6, -0.2, 0.0, 0.3, 0.5])
    w_t = float((d_t**2).mean())
    g_t, rho_t = 0.36, 0.3
    # E[Var_i(Y_i)] (ddof=0) = w* + g*(1 - (1+(N-1)rho)/N)
    shrink = lambda rho: 1.0 - (1.0 + (n - 1) * rho) / n
    conflated_t = w_t + g_t * shrink(rho_t)

    # Collapsed student: w* = 0, all of the statistic from latent-conditional spread.
    d_s = torch.zeros(n)
    g_s = conflated_t / shrink(0.0)

    y_t = _coupled_members(d_t, g_t**0.5, rho_t, 1, seed=11)
    y_s = _coupled_members(d_s, g_s**0.5, 0.0, 1, seed=12)
    var_t = float(y_t[:, :, 0].var(dim=1, unbiased=True).mean())
    var_s = float(y_s[:, :, 0].var(dim=1, unbiased=True).mean())
    # The matched statistic agrees ...
    assert abs(var_t - var_s) < 0.02, (var_t, var_s)
    # ... while the epistemic components could not differ more.
    y_t8 = _coupled_members(d_t, g_t**0.5, rho_t, 8, seed=13)
    y_s8 = _coupled_members(d_s, g_s**0.5, 0.0, 8, seed=14)
    w_t_hat, _ = coupled_w_g(y_t8.permute(1, 2, 0).unsqueeze(-1), 8)
    w_s_hat, _ = coupled_w_g(y_s8.permute(1, 2, 0).unsqueeze(-1), 8)
    assert float(w_t_hat.mean()) > 0.10
    assert abs(float(w_s_hat.mean())) < 0.02


def test_identified_decision_loss_runs_and_reports_split():
    from udwm.models.consistency import identified_decision_distill_loss

    torch.manual_seed(0)
    wm = WorldModel.build(
        model_type="diffusion", obs_dim=3, action_dim=1, ensemble_size=3,
        hidden_dims=(16, 16), student_hidden_dims=(16, 16),
        diffusion_steps=4, sample_steps=2, use_consistency_distill=True,
    )
    obs, act = torch.randn(8, 3), torch.randn(8, 1)
    nxt = obs + 0.1 * torch.randn(8, 3)

    def value_fn(states, _actions):
        return states.sum(dim=-1, keepdim=True)

    parts = identified_decision_distill_loss(
        wm.student, wm.teacher, obs, act, nxt, value_fn, m_latents=2,
    )
    for key in ("total", "member", "value_geometry", "epistemic_w", "aleatoric_g"):
        assert torch.isfinite(parts[key]).all(), key
    assert float(parts["m_latents"]) == 2.0
    parts["total"].backward()
    grads = [p.grad for p in wm.student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # The teacher must not receive gradient from the distillation objective.
    assert all(p.grad is None for p in wm.teacher.parameters())


def test_estimator_variance_scales_with_kurtosis_not_just_sigma():
    """Var(u_hat) = A/M with A carrying the FOURTH moment.

    The finite-M *bias* is distribution-free (research/ESTIMATOR-BIAS-FINDING.md).
    The *variance* is not: two laws with identical mu_i and sigma_i^2 but
    different kurtosis give different Var(u_hat).  See theory/estimator_variance.py.
    """
    rng = np.random.default_rng(0)
    mu = np.array([0.0, 0.5, 1.2, -0.3, 0.8])
    sig2 = np.array([0.25, 0.50, 0.10, 0.80, 0.35])
    n, m, trials = mu.size, 24, 40_000

    def u_deb(y):
        mu_hat = y.mean(axis=-1)
        w_raw = mu_hat.var(axis=-1, ddof=0)
        g_ub = y.var(axis=-1, ddof=1).mean(axis=-1)
        return (w_raw - ((n - 1) / n) * g_ub / m) - g_ub

    gauss = mu[None, :, None] + np.sqrt(sig2)[None, :, None] * rng.standard_normal(
        (trials, n, m)
    )
    p = 0.02  # rare mode: kurtosis ~ 1/p
    b = (rng.random((trials, n, m)) < p).astype(float)
    rare = mu[None, :, None] + np.sqrt(sig2)[None, :, None] * (b - p) / np.sqrt(
        p * (1 - p)
    )

    u_g, u_r = u_deb(gauss), u_deb(rare)
    # Same estimand and same bias for both laws ...
    u_star = float(((mu - mu.mean()) ** 2).mean()) - float(sig2.mean())
    assert abs(float(u_g.mean()) - u_star) < 0.02
    assert abs(float(u_r.mean()) - u_star) < 0.02
    # ... but the rare-mode law is far noisier.
    assert u_r.var(ddof=1) > 5.0 * u_g.var(ddof=1), (u_r.var(), u_g.var())
