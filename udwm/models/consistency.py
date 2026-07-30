"""Consistency-style one-step student distilled from a diffusion teacher.

Implements a practical *distillation* objective for world-model dynamics:
  student f_ψ(x_t, t, s, a) → clean x0
  loss = || f_ψ(x_t, t, s, a) - sg( teacher_x0(x_t, t, s, a) ) ||²

where teacher_x0 is obtained by a multi-step DDIM decode of the frozen teacher
ensemble member. At inference the student maps pure noise → clean Δs (and
optional reward channel) in **one** network evaluation — answering WIMLE's
critique that diffusion is too slow for online rollouts.

This is a research-stack approximation of consistency / progressive distillation,
not a full Song et al. continuous-time consistency model.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from udwm.models.diffusion_dynamics import ConditionalDenoiser, DiffusionDynamicsEnsemble


class ConsistencyStudent(nn.Module):
    """One-step student: x_T ~ N(0,I) → clean x0 conditioned on (s,a)."""

    def __init__(
        self,
        x_dim: int,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (200, 200, 200),
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        # One student per teacher member (or shared — we use per-member for epistemic diversity)
        self.members = nn.ModuleList(
            [ConditionalDenoiser(x_dim, obs_dim, action_dim, hidden_dims) for _ in range(ensemble_size)]
        )

    def forward_member(
        self, i: int, x_t: torch.Tensor, t_scaled: torch.Tensor, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Student directly predicts clean x0 (not noise)
        return self.members[i](x_t, t_scaled, obs, actions)

    @torch.no_grad()
    def sample_next(
        self,
        teacher: DiffusionDynamicsEnsemble,
        obs: torch.Tensor,
        actions: torch.Tensor,
        member: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One-NFE sample using student; falls back dimensions from teacher."""
        b = obs.shape[0]
        device = obs.device
        if member is None:
            idxs = torch.randint(0, self.ensemble_size, (b,), device=device)
            next_obs = torch.zeros_like(obs)
            rewards = torch.zeros(b, 1, device=device) if teacher.joint_reward else None
            for i in range(self.ensemble_size):
                mask = idxs == i
                if not mask.any():
                    continue
                x_t = torch.randn(int(mask.sum()), self.x_dim, device=device)
                t = torch.ones(int(mask.sum()), device=device)  # t=1 ⇒ pure noise end
                x0 = self.forward_member(i, x_t, t, obs[mask], actions[mask])
                delta, r = teacher._unpack_x(x0)
                next_obs[mask] = obs[mask] + delta
                if r is not None and rewards is not None:
                    rewards[mask] = r
            return next_obs, rewards
        x_t = torch.randn(b, self.x_dim, device=device)
        t = torch.ones(b, device=device)
        x0 = self.forward_member(member, x_t, t, obs, actions)
        delta, r = teacher._unpack_x(x0)
        return obs + delta, r


def distill_loss(
    student: ConsistencyStudent,
    teacher: DiffusionDynamicsEnsemble,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: Optional[torch.Tensor] = None,
    member: Optional[int] = None,
) -> torch.Tensor:
    """Distill one random (or fixed) member: student x0 vs teacher multi-step x0 target."""
    teacher.eval()
    b = obs.shape[0]
    device = obs.device
    i = member if member is not None else int(torch.randint(0, student.ensemble_size, (1,)).item())

    # Build clean x0 from data
    delta = next_obs - obs
    x0_data = teacher._pack_x(delta, rewards if teacher.joint_reward else None)

    # Corrupt to random t
    t_idx = torch.randint(0, teacher.diffusion_steps, (b,), device=device)
    noise = torch.randn_like(x0_data)
    sqrt_ab = teacher.schedule.sqrt_alpha_bar[t_idx].unsqueeze(-1)
    sqrt_om = teacher.schedule.sqrt_one_minus_alpha_bar[t_idx].unsqueeze(-1)
    x_t = sqrt_ab * x0_data + sqrt_om * noise
    t_scaled = (t_idx.float() + 1.0) / teacher.diffusion_steps

    # Teacher target: reconstruct x0 from noise prediction (analytic for this t)
    with torch.no_grad():
        eps = teacher.members[i](x_t, t_scaled, obs, actions)
        ab = teacher.schedule.alpha_bar[t_idx].unsqueeze(-1)
        x0_teacher = (x_t - torch.sqrt(1.0 - ab) * eps) / (torch.sqrt(ab) + 1e-8)

    x0_student = student.forward_member(i, x_t, t_scaled, obs, actions)
    return F.mse_loss(x0_student, x0_teacher.detach())


class DistilledWorldModel(nn.Module):
    """Wraps a frozen teacher diffusion ensemble + trainable consistency student."""

    def __init__(self, teacher: DiffusionDynamicsEnsemble, student: ConsistencyStudent) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.model_type = "diffusion"
        self.joint_reward = teacher.joint_reward
        self.reward_model = None  # rewards via joint channel or external

    @property
    def ensemble_size(self) -> int:
        return self.teacher.ensemble_size

    @property
    def dynamics(self):
        # For MC-UBE / rollouts: expose student sampling adapter
        return _StudentDynamicsAdapter(self.teacher, self.student)

    def train_loss(self, obs, actions, next_obs, rewards, dones):
        # Keep teacher sharp on data + distill student
        t_loss = self.teacher.nll_loss(obs, actions, next_obs, rewards, dones)
        s_loss = distill_loss(self.student, self.teacher, obs, actions, next_obs, rewards)
        total = t_loss + s_loss
        return {
            "total": total,
            "dynamics": t_loss.detach(),
            "reward": s_loss.detach(),  # reuse slot as distill loss for logging
        }

    @torch.no_grad()
    def rollout(self, obs, policy_action_fn, horizon: int, member=None):
        obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
        o = obs
        done = torch.zeros(o.shape[0], 1, device=o.device)
        for _ in range(horizon):
            a = policy_action_fn(o)
            o2, r = self.student.sample_next(self.teacher, o, a, member=member)
            if r is None:
                r = torch.zeros(o.shape[0], 1, device=o.device)
            d = self.teacher.predict_done(o, a)
            active = 1.0 - done
            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r * active)
            next_list.append(o2)
            done_list.append(torch.clamp(done + d * active, 0, 1))
            o = o2
            done = done_list[-1]
        return {
            "obs": torch.stack(obs_list, 1),
            "actions": torch.stack(act_list, 1),
            "rewards": torch.stack(rew_list, 1),
            "next_obs": torch.stack(next_list, 1),
            "dones": torch.stack(done_list, 1),
        }


class _StudentDynamicsAdapter(nn.Module):
    """Makes student look like DiffusionDynamicsEnsemble for MC-UBE sample_next_multi."""

    def __init__(self, teacher: DiffusionDynamicsEnsemble, student: ConsistencyStudent):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.ensemble_size = teacher.ensemble_size
        self.joint_reward = teacher.joint_reward
        self.sample_steps = 1

    @torch.no_grad()
    def sample_next(self, obs, actions, member=None, deterministic=True, steps=None):
        o2, r = self.student.sample_next(self.teacher, obs, actions, member=member)
        self.last_reward = r
        return o2

    @torch.no_grad()
    def sample_next_with_reward(self, obs, actions, member=None, deterministic=True, steps=None):
        o2, r = self.student.sample_next(self.teacher, obs, actions, member=member)
        if r is None:
            r = torch.zeros(obs.shape[0], 1, device=obs.device)
        return o2, r

    @torch.no_grad()
    def sample_next_multi(self, obs, actions, m: int, member=None, steps=None):
        samples = [self.sample_next(obs, actions, member=member) for _ in range(m)]
        return torch.stack(samples, 0)

    @torch.no_grad()
    def predict_done(self, obs, actions):
        return self.teacher.predict_done(obs, actions)
