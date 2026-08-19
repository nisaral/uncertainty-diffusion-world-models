"""Fast smoke test of the shared core (no long training)."""

from __future__ import annotations

import sys
from pathlib import Path

# allow running without install
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from udwm.models.world_model import WorldModel
from udwm.rl.sac import SACAgent
from udwm.uncertainty.mc_ube import MCUBELocalRewards, UNetwork, ube_loss
from udwm.utils.config import set_seed


def main() -> None:
    set_seed(0)
    device = torch.device("cpu")
    obs_dim, action_dim = 3, 1
    batch = 32

    print("== Gaussian world model ==")
    gwm = WorldModel.build(
        "gaussian", obs_dim, action_dim, ensemble_size=3, hidden_dims=(64, 64)
    ).to(device)
    obs = torch.randn(batch, obs_dim)
    act = torch.randn(batch, action_dim)
    next_obs = obs + 0.1 * torch.randn_like(obs)
    rew = torch.randn(batch, 1)
    done = torch.zeros(batch, 1)
    loss = gwm.train_loss(obs, act, next_obs, rew, done)["total"]
    loss.backward()
    print(f"  gaussian train loss: {float(loss.detach()):.4f}")

    print("== Diffusion world model ==")
    dwm = WorldModel.build(
        "diffusion",
        obs_dim,
        action_dim,
        ensemble_size=3,
        hidden_dims=(64, 64),
        diffusion_steps=5,
        sample_steps=2,
    ).to(device)
    loss_d = dwm.train_loss(obs, act, next_obs, rew, done)["total"]
    loss_d.backward()
    print(f"  diffusion train loss: {float(loss_d.detach()):.4f}")

    with torch.no_grad():
        o2 = dwm.dynamics.sample_next(obs[:4], act[:4])
        r2, d2 = dwm.reward_model.predict(obs[:4], act[:4])
        print(f"  sample next shape={tuple(o2.shape)} reward shape={tuple(r2.shape)}")

    print("== SAC + MC-UBE ==")
    agent = SACAgent(
        obs_dim,
        action_dim,
        device,
        hidden_dims=(64, 64),
        action_low=np.array([-2.0]),
        action_high=np.array([2.0]),
        optimism_lambda=0.1,
        use_ube=True,
    )
    u_net = UNetwork(obs_dim, action_dim, (64, 64)).to(device)
    agent.attach_u_net(u_net)
    mc = MCUBELocalRewards(m_samples=4)

    batch_d = {
        "obs": obs,
        "actions": act,
        "rewards": rew,
        "next_obs": next_obs,
        "dones": done,
    }
    info = agent.update(batch_d)
    print(f"  sac update: {info}")

    def q_fn(o, a):
        return agent.q_min(o, a)

    def policy_fn(o):
        return agent.policy_tensor(o)

    local = mc.estimate(dwm, q_fn, policy_fn, obs[:8], act[:8])
    print(
        f"  ube local u/w/g means: "
        f"{float(local['u'].mean()):.4f} / {float(local['w'].mean()):.4f} / {float(local['g'].mean()):.4f}"
    )
    next_a = policy_fn(next_obs[:8])
    next_u = u_net(next_obs[:8], next_a)
    targets = mc.ube_targets(local["u"], next_u, agent.gamma, done[:8])
    uloss = ube_loss(u_net, obs[:8], act[:8], targets)
    uloss.backward()
    print(f"  ube loss: {float(uloss.detach()):.4f}")

    print("== Rollout ==")
    with torch.no_grad():
        roll = dwm.rollout(obs[:4], policy_fn, horizon=2)
    print(f"  rollout obs shape={tuple(roll['obs'].shape)}")

    print("== Distillation (uncertainty-preserving student) ==")
    cwm = WorldModel.build(
        "diffusion",
        obs_dim,
        action_dim,
        ensemble_size=2,
        hidden_dims=(64, 64),
        diffusion_steps=5,
        sample_steps=2,
        use_consistency_distill=True,
        preserve_distilled_uncertainty=True,
    ).to(device)
    cwm.freeze_teacher()
    parts = cwm.train_loss(obs, act, next_obs, rew, done)
    parts["total"].backward()
    print(
        "  distill member/mean/geometry/pairwise: "
        f"{float(parts['distill_member']):.4f} / {float(parts['distill_mean']):.4f} / "
        f"{float(parts['distill_geometry']):.4f} / {float(parts['distill_pairwise']):.4f}"
    )

    print("== Joint reward diffusion (optional path) ==")
    jwm = WorldModel.build(
        "diffusion",
        obs_dim,
        action_dim,
        ensemble_size=2,
        hidden_dims=(64, 64),
        diffusion_steps=5,
        sample_steps=2,
        joint_with_diffusion=True,
    ).to(device)
    assert jwm.joint_reward
    loss_j = jwm.train_loss(obs, act, next_obs, rew, done)["total"]
    loss_j.backward()
    with torch.no_grad():
        o2, r2 = jwm.dynamics.sample_next_with_reward(obs[:4], act[:4])
        roll_j = jwm.rollout(obs[:4], policy_fn, horizon=2)
    print(f"  joint train loss: {float(loss_j.detach()):.4f}")
    print(f"  joint sample next/reward shapes: {tuple(o2.shape)} {tuple(r2.shape)}")
    print(f"  joint rollout rewards shape: {tuple(roll_j['rewards'].shape)}")

    print("== Consistency distill wrapper ==")
    from udwm.models.consistency import ConsistencyStudent, DistilledWorldModel, distill_loss

    teacher = WorldModel.build(
        "diffusion",
        obs_dim,
        action_dim,
        ensemble_size=2,
        hidden_dims=(64, 64),
        diffusion_steps=5,
        sample_steps=2,
        joint_with_diffusion=False,
    ).dynamics
    student = ConsistencyStudent(obs_dim, obs_dim, action_dim, ensemble_size=2, hidden_dims=(64, 64))
    dloss = distill_loss(student, teacher, obs, act, next_obs, rew)
    dloss.backward()
    dwm = DistilledWorldModel(teacher, student)
    with torch.no_grad():
        roll_d = dwm.rollout(obs[:4], policy_fn, horizon=2)
    print(f"  distill loss: {float(dloss.detach()):.4f}")
    print(f"  distilled rollout shape: {tuple(roll_d['obs'].shape)}")

    print("\nSMOKE OK — core + joint reward + consistency distill runnable.")



if __name__ == "__main__":
    main()
