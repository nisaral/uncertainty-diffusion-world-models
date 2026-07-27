from __future__ import annotations

import time
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from udwm.uncertainty.calibration import reliability_summary


@torch.no_grad()
def evaluate_policy_return(
    agent,
    env: gym.Env,
    n_episodes: int = 10,
    seed: int = 0,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    returns = []
    lengths = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            action = agent.act(obs, deterministic=True)
            if action_low is not None:
                action = np.clip(action, action_low, action_high)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            steps += 1
            done = term or trunc
        returns.append(ep_ret)
        lengths.append(steps)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
    }


@torch.no_grad()
def evaluate_world_model_accuracy(
    world_model,
    buffer,
    batch_size: int = 512,
    n_batches: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """One-step next-state / reward prediction error on real buffer data."""
    if len(buffer) < batch_size:
        batch_size = max(1, len(buffer))
    device = device or torch.device("cpu")
    ms_list, rs_list = [], []
    for _ in range(n_batches):
        batch = buffer.sample(batch_size)
        obs, act = batch["obs"], batch["actions"]
        nxt, rew = batch["next_obs"], batch["rewards"]
        if getattr(world_model, "model_type", "") == "gaussian":
            pred_n, pred_r = world_model.dynamics.sample_next(obs, act, deterministic=True)
        elif getattr(world_model, "joint_reward", False) and hasattr(
            world_model.dynamics, "sample_next_with_reward"
        ):
            pred_n, pred_r = world_model.dynamics.sample_next_with_reward(
                obs, act, deterministic=True
            )
        else:
            pred_n = world_model.dynamics.sample_next(obs, act, deterministic=True)
            if world_model.reward_model is not None:
                pred_r, _ = world_model.reward_model.predict(obs, act)
            else:
                pred_r = torch.zeros_like(rew)
        ms_list.append(float(((pred_n - nxt) ** 2).mean().item()))
        rs_list.append(float(((pred_r - rew) ** 2).mean().item()))
    return {
        "next_state_mse": float(np.mean(ms_list)),
        "reward_mse": float(np.mean(rs_list)),
    }


@torch.no_grad()
def evaluate_uncertainty_calibration(
    agent,
    u_net,
    buffer,
    batch_size: int = 512,
    n_batches: int = 3,
) -> Dict[str, float]:
    """Compare √U(s,a) to |TD residual| magnitude (rough calibration proxy)."""
    if len(buffer) < 32:
        return {"corr_std_abserr": 0.0, "mean_pred_std": 0.0, "mean_abs_err": 0.0}
    batch_size = min(batch_size, len(buffer))
    preds, resids = [], []
    gamma = agent.gamma
    for _ in range(n_batches):
        b = buffer.sample(batch_size)
        obs, act = b["obs"], b["actions"]
        rew, nxt, done = b["rewards"], b["next_obs"], b["dones"]
        q = agent.q_min(obs, act)
        next_a = agent.policy_tensor(nxt, deterministic=True)
        q_next = agent.critic_target.min_q(nxt, next_a)
        td = rew + (1.0 - done) * gamma * q_next - q
        u = u_net(obs, act)
        preds.append(torch.sqrt(u + 1e-8))
        resids.append(td)
    pred = torch.cat(preds, 0)
    resid = torch.cat(resids, 0)
    return reliability_summary(pred, resid)


@torch.no_grad()
def throughput_benchmark(
    world_model,
    obs_dim: int,
    action_dim: int,
    batch_size: int = 256,
    n_repeats: int = 20,
    sample_steps_list: Optional[list] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Wall-clock next-state samples/sec for different denoising step counts."""
    device = device or next(world_model.parameters()).device
    obs = torch.randn(batch_size, obs_dim, device=device)
    act = torch.randn(batch_size, action_dim, device=device)
    results = {}
    model_type = getattr(world_model, "model_type", "diffusion")

    if model_type == "gaussian":
        # warmup
        for _ in range(3):
            world_model.dynamics.sample_next(obs, act, deterministic=True)
        t0 = time.perf_counter()
        n_tot = 0
        for _ in range(n_repeats):
            world_model.dynamics.sample_next(obs, act, deterministic=True)
            n_tot += batch_size
        dt = time.perf_counter() - t0
        results["gaussian"] = {
            "samples_per_sec": n_tot / max(dt, 1e-9),
            "ms_per_batch": 1000.0 * dt / n_repeats,
        }
        return results

    steps_list = sample_steps_list or [1, 2, 4, 8, 10]
    dyn = world_model.dynamics
    for k in steps_list:
        if hasattr(dyn, "sample_steps"):
            old = dyn.sample_steps
            dyn.sample_steps = k
        for _ in range(2):
            dyn.sample_next(obs, act, deterministic=True, steps=k)
        t0 = time.perf_counter()
        n_tot = 0
        for _ in range(n_repeats):
            dyn.sample_next(obs, act, deterministic=True, steps=k)
            n_tot += batch_size
        dt = time.perf_counter() - t0
        results[f"steps_{k}"] = {
            "samples_per_sec": n_tot / max(dt, 1e-9),
            "ms_per_batch": 1000.0 * dt / n_repeats,
            "sample_steps": k,
        }
        if hasattr(dyn, "sample_steps"):
            dyn.sample_steps = old
    return results
