from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch

from udwm.uncertainty.calibration import reliability_summary
from udwm.uncertainty.mc_ube import MCUBELocalRewards


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


@torch.no_grad()
def evaluate_distillation_uncertainty(
    distilled_world_model,
    q_fn,
    policy_fn,
    buffer,
    batch_size: int = 256,
    n_batches: int = 3,
    m_samples: int = 8,
) -> Dict[str, float]:
    """Compare teacher and student local UBE quantities on identical states.

    This is the primary evaluation for uncertainty-preserving distillation. It
    deliberately uses the same critic/policy and estimator for both models, so
    the reported gap is about the sampler/student rather than a changed UBE
    implementation. ``distilled_world_model`` must be a DistilledWorldModel.
    """
    if not hasattr(distilled_world_model, "teacher"):
        raise TypeError("expected DistilledWorldModel with a teacher attribute")
    if len(buffer) == 0:
        return {"n": 0.0}
    batch_size = min(int(batch_size), len(buffer))
    teacher_proxy = SimpleNamespace(
        dynamics=distilled_world_model.teacher,
        model_type="diffusion",
        ensemble_size=distilled_world_model.teacher.ensemble_size,
    )
    student_proxy = distilled_world_model
    est = MCUBELocalRewards(u_min=-1e9, m_samples=max(2, int(m_samples)), debias=True)
    teacher_u, student_u, teacher_w, student_w = [], [], [], []
    for _ in range(int(n_batches)):
        batch = buffer.sample(batch_size)
        obs, act = batch["obs"], batch["actions"]
        t = est.estimate(teacher_proxy, q_fn, policy_fn, obs, act)
        s = est.estimate(student_proxy, q_fn, policy_fn, obs, act)
        teacher_u.append(t["u"].reshape(-1).cpu().numpy())
        student_u.append(s["u"].reshape(-1).cpu().numpy())
        teacher_w.append(t["w"].reshape(-1).cpu().numpy())
        student_w.append(s["w"].reshape(-1).cpu().numpy())

    tu, su = np.concatenate(teacher_u), np.concatenate(student_u)
    tw, sw = np.concatenate(teacher_w), np.concatenate(student_w)
    return {
        "n": float(tu.size),
        "u_rank_corr": _rank_corr(tu, su),
        "w_rank_corr": _rank_corr(tw, sw),
        "u_mae": float(np.mean(np.abs(tu - su))),
        "w_mae": float(np.mean(np.abs(tw - sw))),
        "u_rmse": float(np.sqrt(np.mean((tu - su) ** 2))),
        "w_rmse": float(np.sqrt(np.mean((tw - sw) ** 2))),
        "teacher_u_mean": float(np.mean(tu)),
        "student_u_mean": float(np.mean(su)),
    }


@torch.no_grad()
def evaluate_policy_return(
    agent,
    env,
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
    mean_samples: int = 16,
) -> Dict[str, float]:
    """One-step next-state / reward prediction error on real buffer data.

    Reports both a MEAN-prediction MSE (comparable across model classes) and a
    single-draw MSE. For a stochastic model the single draw is inflated by the
    model's own aleatoric variance, so comparing a diffusion draw against a
    Gaussian ensemble's analytic mean measures stochasticity, not accuracy.
    """
    if len(buffer) < batch_size:
        batch_size = max(1, len(buffer))
    device = device or torch.device("cpu")
    ms_list, rs_list, draw_list = [], [], []
    dyn = world_model.dynamics
    for _ in range(n_batches):
        batch = buffer.sample(batch_size)
        obs, act = batch["obs"], batch["actions"]
        nxt, rew = batch["next_obs"], batch["rewards"]
        if getattr(world_model, "model_type", "") == "gaussian":
            pred_n, pred_r = dyn.mean_next(obs, act)
            draw_n, _ = dyn.sample_next(obs, act, deterministic=False)
        elif hasattr(dyn, "predict_mean"):
            pred_n, pred_r_joint = dyn.predict_mean(obs, act, m=mean_samples)
            draw_n = dyn.sample_next(obs, act, deterministic=False)
            if getattr(world_model, "joint_reward", False):
                pred_r = pred_r_joint
            elif world_model.reward_model is not None:
                pred_r, _ = world_model.reward_model.predict(obs, act)
            else:
                pred_r = torch.zeros_like(rew)
        else:
            pred_n = dyn.sample_next(obs, act, deterministic=True)
            draw_n = pred_n
            if world_model.reward_model is not None:
                pred_r, _ = world_model.reward_model.predict(obs, act)
            else:
                pred_r = torch.zeros_like(rew)
        ms_list.append(float(((pred_n - nxt) ** 2).mean().item()))
        rs_list.append(float(((pred_r - rew) ** 2).mean().item()))
        draw_list.append(float(((draw_n - nxt) ** 2).mean().item()))
    return {
        "next_state_mse": float(np.mean(ms_list)),
        "next_state_mse_single_draw": float(np.mean(draw_list)),
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
