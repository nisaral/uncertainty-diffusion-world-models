"""U-gated imagination (novelty A).

Uses multi-step epistemic uncertainty U(s,a) (or √U) to control model rollouts:

* **stop** — terminate an imagined trajectory when √U exceeds a threshold
  (absolute or batch percentile).
* **weight** — store transitions with weight ∝ exp(-β √U) so SAC rarely
  trains on highly uncertain dreams.
* **both** — stop and weight.

This is distinct from fixed-horizon MBPO and from one-step MOPO penalties:
uncertainty is the **UBE network** (Bellman-propagated), and it gates the
*imagination process itself*.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@torch.no_grad()
def u_gated_rollout(
    world_model,
    policy_fn,
    u_net,
    obs: torch.Tensor,
    horizon: int,
    mode: str = "both",
    stop_threshold: Optional[float] = None,
    stop_percentile: float = 0.85,
    weight_beta: float = 1.0,
    min_weight: float = 0.05,
    use_sqrt: bool = True,
) -> Dict[str, torch.Tensor]:
    """Roll out up to ``horizon`` steps with U-based stop/weight.

    Returns dict with keys obs, actions, rewards, next_obs, dones, weights
    each shaped [B, H, ...] (weights [B, H, 1]). Truncated steps after stop
    have weight 0 and done=1.
    """
    mode = (mode or "off").lower()
    if mode == "off":
        roll = world_model.rollout(obs, policy_fn, horizon=horizon)
        b, h = roll["obs"].shape[0], roll["obs"].shape[1]
        roll["weights"] = torch.ones(b, h, 1, device=obs.device)
        roll["sqrt_u"] = torch.zeros(b, h, 1, device=obs.device)
        roll["stopped_frac"] = torch.tensor(0.0, device=obs.device)
        return roll

    device = obs.device
    b = obs.shape[0]
    o = obs
    done = torch.zeros(b, 1, device=device)
    active = torch.ones(b, 1, device=device)

    obs_l, act_l, rew_l, next_l, done_l, w_l, su_l = [], [], [], [], [], [], []

    # collect sqrt_u for percentile threshold on first step if needed
    pending_threshold = stop_threshold
    n_u_stops = 0.0
    n_active_seen = 0.0

    for t in range(horizon):
        a = policy_fn(o)
        u = u_net(o, a)
        score = torch.sqrt(u + 1e-8) if use_sqrt else u

        if pending_threshold is None and mode in ("stop", "both"):
            # use this step's batch distribution for threshold (only among active)
            sc = score.detach().reshape(-1)
            # avoid stopping everyone when scores are nearly constant
            if float(sc.std().item()) < 1e-6:
                pending_threshold = float("inf")
            else:
                pending_threshold = float(torch.quantile(sc, stop_percentile).item())

        # weights
        if mode in ("weight", "both"):
            # higher U → lower weight
            w = torch.exp(-weight_beta * score)
            w = torch.clamp(w, min=min_weight, max=1.0)
        else:
            w = torch.ones_like(score)

        # stop mask (only among still-active trajectories)
        stop = torch.zeros_like(active)
        if mode in ("stop", "both") and pending_threshold is not None:
            stop = ((score > pending_threshold).float()) * active
            n_u_stops += float(stop.sum().item())
            n_active_seen += float(active.sum().item())

        # dynamics step
        model_type = getattr(world_model, "model_type", "diffusion")
        joint = getattr(world_model, "joint_reward", False)
        if model_type == "gaussian":
            o2, r = world_model.dynamics.sample_next(o, a)
            d = torch.zeros_like(r)
        elif joint and hasattr(world_model.dynamics, "sample_next_with_reward"):
            o2, r = world_model.dynamics.sample_next_with_reward(o, a)
            d = (
                world_model.dynamics.predict_done(o, a)
                if hasattr(world_model.dynamics, "predict_done")
                else torch.zeros_like(r)
            )
        else:
            o2 = world_model.dynamics.sample_next(o, a)
            if world_model.reward_model is not None:
                r, d = world_model.reward_model.predict(o, a)
            else:
                r = torch.zeros(b, 1, device=device)
                d = torch.zeros_like(r)
                if hasattr(world_model.dynamics, "predict_done"):
                    d = world_model.dynamics.predict_done(o, a)

        # apply active mask from previous stops
        w = w * active
        # new done if stopped or env done
        d_eff = torch.clamp(d + stop * active, 0, 1)
        d_eff = torch.clamp(done + d_eff * (1.0 - done), 0, 1)

        obs_l.append(o)
        act_l.append(a)
        rew_l.append(r * active)
        next_l.append(o2)
        done_l.append(d_eff)
        w_l.append(w)
        su_l.append(score)

        # deactivate for future steps
        active = active * (1.0 - stop) * (1.0 - d_eff)
        done = d_eff
        o = o2

        if float(active.sum().item()) <= 0:
            # pad remaining steps with zeros
            for _ in range(t + 1, horizon):
                obs_l.append(o)
                act_l.append(a)
                rew_l.append(torch.zeros_like(r))
                next_l.append(o)
                done_l.append(torch.ones_like(d_eff))
                w_l.append(torch.zeros_like(w))
                su_l.append(score)
            break

    weights = torch.stack(w_l, 1)
    # fraction of (step, traj) pairs that U-gate stopped (not env done)
    stopped_frac = n_u_stops / max(n_active_seen, 1.0)
    thr = pending_threshold
    if thr is None or thr == float("inf"):
        thr_out = -1.0
    else:
        thr_out = float(thr)
    return {
        "obs": torch.stack(obs_l, 1),
        "actions": torch.stack(act_l, 1),
        "rewards": torch.stack(rew_l, 1),
        "next_obs": torch.stack(next_l, 1),
        "dones": torch.stack(done_l, 1),
        "weights": weights,
        "sqrt_u": torch.stack(su_l, 1),
        "stopped_frac": torch.tensor(stopped_frac, device=device),
        "stop_threshold_used": torch.tensor(thr_out, device=device),
    }


def weights_to_numpy_mask(weights_bh: torch.Tensor, min_keep: float = 1e-6) -> np.ndarray:
    """Boolean mask [B,H] of transitions worth inserting into the buffer."""
    return (weights_bh.squeeze(-1) > min_keep).cpu().numpy()
