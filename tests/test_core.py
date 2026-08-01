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

