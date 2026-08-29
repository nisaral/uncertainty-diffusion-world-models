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

from typing import Callable, Dict, Optional, Sequence, Tuple

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
        x_T: Optional[torch.Tensor] = None,
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
                x_t = torch.randn(int(mask.sum()), self.x_dim, device=device) if x_T is None else x_T[mask]
                t = torch.ones(int(mask.sum()), device=device)  # t=1 ⇒ pure noise end
                x0 = self.forward_member(i, x_t, t, obs[mask], actions[mask])
                delta, r = teacher._unpack_x(x0)
                next_obs[mask] = obs[mask] + delta
                if r is not None and rewards is not None:
                    rewards[mask] = r
            return next_obs, rewards
        x_t = torch.randn(b, self.x_dim, device=device) if x_T is None else x_T.to(device)
        if x_t.shape != (b, self.x_dim):
            raise ValueError(f"x_T must have shape {(b, self.x_dim)}, got {tuple(x_t.shape)}")
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


def uncertainty_preserving_distill_loss(
    student: ConsistencyStudent,
    teacher: DiffusionDynamicsEnsemble,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: Optional[torch.Tensor] = None,
    mean_weight: float = 1.0,
    geometry_weight: float = 1.0,
    pairwise_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Distil samples while preserving ensemble epistemic geometry.

    All members see the same corrupted data sample. Besides member-wise teacher
    matching, the loss preserves (a) the ensemble mean, (b) centered member
    deviations, and (c) pairwise squared distances. These are the empirical
    objects that determine one-step ensemble variance and, after a Lipschitz
    value map, the local UBE disagreement term.
    """
    teacher.eval()
    b = obs.shape[0]
    device = obs.device
    delta = next_obs - obs
    x0_data = teacher._pack_x(delta, rewards if teacher.joint_reward else None)
    t_idx = torch.randint(0, teacher.diffusion_steps, (b,), device=device)
    noise = torch.randn_like(x0_data)
    sqrt_ab = teacher.schedule.sqrt_alpha_bar[t_idx].unsqueeze(-1)
    sqrt_om = teacher.schedule.sqrt_one_minus_alpha_bar[t_idx].unsqueeze(-1)
    x_t = sqrt_ab * x0_data + sqrt_om * noise
    t_scaled = (t_idx.float() + 1.0) / teacher.diffusion_steps

    teacher_x0, student_x0 = [], []
    for i in range(student.ensemble_size):
        with torch.no_grad():
            eps = teacher.members[i](x_t, t_scaled, obs, actions)
            ab = teacher.schedule.alpha_bar[t_idx].unsqueeze(-1)
            target = (x_t - torch.sqrt(1.0 - ab) * eps) / (torch.sqrt(ab) + 1e-8)
        teacher_x0.append(target.detach())
        student_x0.append(student.forward_member(i, x_t, t_scaled, obs, actions))

    t_all = torch.stack(teacher_x0, dim=0)  # [N,B,D]
    s_all = torch.stack(student_x0, dim=0)
    member = F.mse_loss(s_all, t_all)
    mean = F.mse_loss(s_all.mean(dim=0), t_all.mean(dim=0))
    t_center = t_all - t_all.mean(dim=0, keepdim=True)
    s_center = s_all - s_all.mean(dim=0, keepdim=True)
    geometry = F.mse_loss(s_center, t_center)

    t_diff = t_all[:, None] - t_all[None, :]
    s_diff = s_all[:, None] - s_all[None, :]
    t_pair = t_diff.pow(2).mean(dim=-1)
    s_pair = s_diff.pow(2).mean(dim=-1)
    pairwise = F.mse_loss(s_pair, t_pair)
    total = (
        member
        + float(mean_weight) * mean
        + float(geometry_weight) * geometry
        + float(pairwise_weight) * pairwise
    )
    return {
        "total": total,
        "member": member,
        "mean": mean,
        "geometry": geometry,
        "pairwise": pairwise,
    }


def uncertainty_fidelity_guard(
    teacher_values: torch.Tensor,
    student_values: torch.Tensor,
    min_corr: float = 0.0,
    max_scale: float = 8.0,
) -> torch.Tensor:
    """Skip decision terms when student uncertainty is anti-aligned or exploded.

    ``teacher_values`` / ``student_values`` are ensemble-stacked maps
    ``[N, B, ...]``. The guard is a stop-gradient diagnostic: it does not
    train the student to game the threshold.
    """
    t_var = teacher_values.detach().reshape(teacher_values.shape[0], -1).var(dim=0, unbiased=False)
    s_var = student_values.detach().reshape(student_values.shape[0], -1).var(dim=0, unbiased=False)
    t_c = t_var - t_var.mean()
    s_c = s_var - s_var.mean()
    corr = (s_c * t_c).sum() / (s_c.norm() * t_c.norm() + 1e-8)
    scale = s_var.mean() / (t_var.mean() + 1e-8)
    fire = (corr < float(min_corr)) | (scale > float(max_scale)) | (scale < 1.0 / max(float(max_scale), 1e-6))
    return fire.to(dtype=teacher_values.dtype)


def lagged_target_value_fn(critic_target, actor) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Value map from a lagged critic and a stop-gradient policy.

    Gradients flow through next-states into the student, not into the critic
    or actor. This is the correction for the online-critic collapse: the
    decision map is slow-moving and is not jointly optimized with the student.
    """

    def value_fn(states: torch.Tensor, _actions: torch.Tensor) -> torch.Tensor:
        flat = states.reshape(-1, states.shape[-1])
        with torch.no_grad():
            actions = actor.deterministic(flat)
        q = critic_target.min_q(flat, actions)
        return q.reshape(*states.shape[:-1], 1)

    return value_fn


def decision_preserving_distill_loss(
    student: ConsistencyStudent,
    teacher: DiffusionDynamicsEnsemble,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rewards: Optional[torch.Tensor] = None,
    value_weight: float = 1.0,
    variance_weight: float = 1.0,
    variance_shape_weight: float = 0.0,
    state_geometry_weight: float = 0.0,
    state_pairwise_weight: float = 0.0,
    normalize_values: bool = False,
    guard_enabled: bool = False,
    guard_min_corr: float = 0.0,
    guard_max_scale: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """Match member outputs and their downstream decision-value disagreement.

    Unlike state-space geometry matching, this objective applies the supplied
    value map before matching centered ensemble values and their variance. It
    therefore targets the actual local-UBE object when Euclidean state errors
    are poorly aligned with decision consequences.
    """
    teacher.eval()
    b = obs.shape[0]
    delta = next_obs - obs
    x0_data = teacher._pack_x(delta, rewards if teacher.joint_reward else None)
    t_idx = torch.randint(0, teacher.diffusion_steps, (b,), device=obs.device)
    noise = torch.randn_like(x0_data)
    sqrt_ab = teacher.schedule.sqrt_alpha_bar[t_idx].unsqueeze(-1)
    sqrt_om = teacher.schedule.sqrt_one_minus_alpha_bar[t_idx].unsqueeze(-1)
    x_t = sqrt_ab * x0_data + sqrt_om * noise
    t_scaled = (t_idx.float() + 1.0) / teacher.diffusion_steps

    teacher_x0, student_x0 = [], []
    for i in range(student.ensemble_size):
        with torch.no_grad():
            eps = teacher.members[i](x_t, t_scaled, obs, actions)
            ab = teacher.schedule.alpha_bar[t_idx].unsqueeze(-1)
            target = (x_t - torch.sqrt(1.0 - ab) * eps) / (torch.sqrt(ab) + 1e-8)
        teacher_x0.append(target.detach())
        student_x0.append(student.forward_member(i, x_t, t_scaled, obs, actions))

    t_all = torch.stack(teacher_x0, dim=0)
    s_all = torch.stack(student_x0, dim=0)
    member = F.mse_loss(s_all, t_all)
    t_state_center = t_all - t_all.mean(dim=0, keepdim=True)
    s_state_center = s_all - s_all.mean(dim=0, keepdim=True)
    state_geometry = F.mse_loss(s_state_center, t_state_center)
    t_pair = (t_all[:, None] - t_all[None, :]).pow(2).mean(dim=-1)
    s_pair = (s_all[:, None] - s_all[None, :]).pow(2).mean(dim=-1)
    state_pairwise = F.mse_loss(s_pair, t_pair)
    t_next = obs.unsqueeze(0) + teacher._unpack_x(t_all)[0]
    s_next = obs.unsqueeze(0) + teacher._unpack_x(s_all)[0]
    expanded_actions = actions.unsqueeze(0).expand(student.ensemble_size, -1, -1)
    with torch.no_grad():
        t_value = value_fn(t_next, expanded_actions)
    s_value = value_fn(s_next, expanded_actions)
    if normalize_values:
        loc = t_value.mean().detach()
        scale = t_value.std(unbiased=False).detach().clamp_min(1e-6)
        t_used = (t_value - loc) / scale
        s_used = (s_value - loc) / scale
    else:
        t_used, s_used = t_value, s_value
    t_center = t_used - t_used.mean(dim=0, keepdim=True)
    s_center = s_used - s_used.mean(dim=0, keepdim=True)
    value_geometry = F.mse_loss(s_center, t_center)
    s_variance = s_used.var(dim=0, unbiased=True)
    t_variance = t_used.var(dim=0, unbiased=True)
    value_variance = F.mse_loss(s_variance, t_variance)
    # Standardization removes absolute scale, so this term targets the ordering
    # and spatial shape of uncertainty across the minibatch.
    t_shape = (t_variance - t_variance.mean()) / (t_variance.std(unbiased=False) + 1e-6)
    s_shape = (s_variance - s_variance.mean()) / (s_variance.std(unbiased=False) + 1e-6)
    variance_shape = F.mse_loss(s_shape, t_shape)
    guard_fire = uncertainty_fidelity_guard(
        t_value, s_value, min_corr=guard_min_corr, max_scale=guard_max_scale
    )
    if not guard_enabled:
        guard_fire = guard_fire.new_zeros(())
    decision_scale = 1.0 - guard_fire
    total = (
        member
        + decision_scale * float(value_weight) * value_geometry
        + decision_scale * float(variance_weight) * value_variance
        + decision_scale * float(variance_shape_weight) * variance_shape
        + float(state_geometry_weight) * state_geometry
        + float(state_pairwise_weight) * state_pairwise
    )
    return {
        "total": total,
        "member": member,
        "value_geometry": value_geometry,
        "value_variance": value_variance,
        "variance_shape": variance_shape,
        "state_geometry": state_geometry,
        "state_pairwise": state_pairwise,
        "guard_fired": guard_fire.detach(),
        "decision_scale": decision_scale.detach(),
    }


def coupled_w_g(values: torch.Tensor, m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coupling-aware (w_deb, g) from member values shaped ``[N, M, B, 1]``.

    Mirrors ``MCUBELocalRewards.combine_coupled`` but keeps the graph, so the
    debiased epistemic term is differentiable w.r.t. the student. ``values`` are
    drawn with the SAME M latents for every member (a shared-latent coupling),
    which is why the independent-sampling correction ``(N-1)/N * g/M`` does not
    apply here and ``(g - Sigma_bar)/M`` does.
    """
    n = values.shape[0]
    mu = values.mean(dim=1)                                  # [N,B,1]
    w_raw = mu.var(dim=0, unbiased=False)                    # [B,1]
    z = values - mu.unsqueeze(1)
    denom = float(max(m - 1, 1))
    g = (z.pow(2).sum(dim=1) / denom).mean(dim=0)            # [B,1]
    sigma_bar = z.mean(dim=0).pow(2).sum(dim=0) / denom      # [B,1]
    w_deb = w_raw - (g - sigma_bar) / float(m) if m > 1 else w_raw
    return w_deb, g


def identified_decision_distill_loss(
    student: ConsistencyStudent,
    teacher: DiffusionDynamicsEnsemble,
    obs: torch.Tensor,
    actions: torch.Tensor,
    next_obs: torch.Tensor,
    value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rewards: Optional[torch.Tensor] = None,
    m_latents: int = 2,
    value_weight: float = 1.0,
    variance_weight: float = 1.0,
    aleatoric_weight: float = 1.0,
    state_geometry_weight: float = 0.0,
    state_pairwise_weight: float = 0.0,
    normalize_values: bool = False,
) -> Dict[str, torch.Tensor]:
    """Decision-aware distillation with an *identified* uncertainty target.

    ``decision_preserving_distill_loss`` matches the cross-member value variance
    at a **single** shared diffusion latent. By the coupling identity that
    statistic has expectation ``w* + (g* - Sigma_bar)`` -- one equation in two
    unknowns -- so a student can satisfy it with zero epistemic disagreement by
    inflating its latent-conditional spread instead. See
    ``theory/distill_identifiability.py`` for the exact zero-loss family.

    Drawing ``m_latents >= 2`` and matching the debiased ``w`` and the aleatoric
    ``g`` as two separate terms removes that direction. This is the epistemic /
    ensemble analogue of the variance correction Voelcker et al. (arXiv:2505.22772)
    use to make a value-aware model loss calibrated under a sampled model; there
    the collapsing variance is one model's aleatoric spread, here it is the
    disagreement across members.
    """
    teacher.eval()
    m = max(2, int(m_latents))
    b = obs.shape[0]
    device = obs.device
    delta = next_obs - obs
    x0_data = teacher._pack_x(delta, rewards if teacher.joint_reward else None)

    t_vals, s_vals, member_sq, state_geom_sq, state_pair_sq = [], [], [], [], []
    for _ in range(m):
        t_idx = torch.randint(0, teacher.diffusion_steps, (b,), device=device)
        noise = torch.randn_like(x0_data)
        sqrt_ab = teacher.schedule.sqrt_alpha_bar[t_idx].unsqueeze(-1)
        sqrt_om = teacher.schedule.sqrt_one_minus_alpha_bar[t_idx].unsqueeze(-1)
        x_t = sqrt_ab * x0_data + sqrt_om * noise
        t_scaled = (t_idx.float() + 1.0) / teacher.diffusion_steps
        t_x0, s_x0 = [], []
        for i in range(student.ensemble_size):
            with torch.no_grad():
                eps = teacher.members[i](x_t, t_scaled, obs, actions)
                ab = teacher.schedule.alpha_bar[t_idx].unsqueeze(-1)
                target = (x_t - torch.sqrt(1.0 - ab) * eps) / (torch.sqrt(ab) + 1e-8)
            t_x0.append(target.detach())
            s_x0.append(student.forward_member(i, x_t, t_scaled, obs, actions))
        t_all = torch.stack(t_x0, dim=0)                      # [N,B,D]
        s_all = torch.stack(s_x0, dim=0)
        member_sq.append(F.mse_loss(s_all, t_all))
        state_geom_sq.append(F.mse_loss(
            s_all - s_all.mean(dim=0, keepdim=True),
            t_all - t_all.mean(dim=0, keepdim=True),
        ))
        state_pair_sq.append(F.mse_loss(
            (s_all[:, None] - s_all[None, :]).pow(2).mean(dim=-1),
            (t_all[:, None] - t_all[None, :]).pow(2).mean(dim=-1),
        ))
        expanded = actions.unsqueeze(0).expand(student.ensemble_size, -1, -1)
        t_next = obs.unsqueeze(0) + teacher._unpack_x(t_all)[0]
        s_next = obs.unsqueeze(0) + teacher._unpack_x(s_all)[0]
        with torch.no_grad():
            t_vals.append(value_fn(t_next, expanded))
        s_vals.append(value_fn(s_next, expanded))

    t_value = torch.stack(t_vals, dim=1)                      # [N,M,B,1]
    s_value = torch.stack(s_vals, dim=1)
    if normalize_values:
        loc = t_value.mean().detach()
        scale = t_value.std(unbiased=False).detach().clamp_min(1e-6)
        t_value = (t_value - loc) / scale
        s_value = (s_value - loc) / scale

    t_w, t_g = coupled_w_g(t_value, m)
    s_w, s_g = coupled_w_g(s_value, m)
    # Member means are now M-averaged, so the geometry term is a lower-variance
    # estimate of the same object the single-latent version was targeting.
    t_mu, s_mu = t_value.mean(dim=1), s_value.mean(dim=1)
    t_center = t_mu - t_mu.mean(dim=0, keepdim=True)
    s_center = s_mu - s_mu.mean(dim=0, keepdim=True)

    member = torch.stack(member_sq).mean()
    state_geometry = torch.stack(state_geom_sq).mean()
    state_pairwise = torch.stack(state_pair_sq).mean()
    value_geometry = F.mse_loss(s_center, t_center)
    epistemic = F.mse_loss(s_w, t_w.detach())
    aleatoric = F.mse_loss(s_g, t_g.detach())
    total = (
        member
        + float(value_weight) * value_geometry
        + float(variance_weight) * epistemic
        + float(aleatoric_weight) * aleatoric
        + float(state_geometry_weight) * state_geometry
        + float(state_pairwise_weight) * state_pairwise
    )
    return {
        "total": total,
        "member": member,
        "value_geometry": value_geometry,
        "epistemic_w": epistemic,
        "aleatoric_g": aleatoric,
        "state_geometry": state_geometry,
        "state_pairwise": state_pairwise,
        "teacher_w_mean": t_w.mean().detach(),
        "student_w_mean": s_w.mean().detach(),
        "m_latents": torch.tensor(float(m), device=device),
    }


class DistilledWorldModel(nn.Module):
    """Wraps a frozen teacher diffusion ensemble + trainable consistency student."""

    def __init__(
        self,
        teacher: DiffusionDynamicsEnsemble,
        student: ConsistencyStudent,
        preserve_uncertainty: bool = False,
        mean_weight: float = 1.0,
        geometry_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        decision_weight: float = 0.0,
        value_variance_weight: float = 0.0,
        hybrid_state_weight: float = 0.0,
        hybrid_pairwise_weight: float = 0.0,
        normalize_values: bool = False,
        guard_enabled: bool = False,
        guard_min_corr: float = 0.0,
        guard_max_scale: float = 8.0,
        identified: bool = False,
        m_latents: int = 2,
        aleatoric_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.model_type = "diffusion"
        self.joint_reward = teacher.joint_reward
        self.reward_model = None  # rewards via joint channel or external
        self.preserve_uncertainty = bool(preserve_uncertainty)
        self.mean_weight = float(mean_weight)
        self.geometry_weight = float(geometry_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.decision_weight = float(decision_weight)
        self.value_variance_weight = float(value_variance_weight)
        self.hybrid_state_weight = float(hybrid_state_weight)
        self.hybrid_pairwise_weight = float(hybrid_pairwise_weight)
        self.normalize_values = bool(normalize_values)
        self.guard_enabled = bool(guard_enabled)
        self.guard_min_corr = float(guard_min_corr)
        self.guard_max_scale = float(guard_max_scale)
        # Identified objective: M >= 2 latents, separate (w_deb, g) terms.
        self.identified = bool(identified)
        self.m_latents = max(2, int(m_latents))
        self.aleatoric_weight = float(aleatoric_weight)
        self.teacher_frozen = False

    def freeze_teacher(self) -> None:
        """Freeze teacher parameters for a genuine two-stage distillation run."""
        self.teacher_frozen = True
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @property
    def ensemble_size(self) -> int:
        return self.teacher.ensemble_size

    @property
    def dynamics(self):
        # For MC-UBE / rollouts: expose student sampling adapter
        return _StudentDynamicsAdapter(self.teacher, self.student)

    def train_loss(self, obs, actions, next_obs, rewards, dones, value_fn=None):
        # Pretraining may update the teacher; after freeze_teacher(), only the
        # student is optimized and the teacher is a fixed target distribution.
        if self.teacher_frozen:
            t_loss = torch.zeros((), device=obs.device)
        else:
            t_loss = self.teacher.nll_loss(obs, actions, next_obs, rewards, dones)
        if self.decision_weight > 0.0 and value_fn is not None and self.identified:
            parts = identified_decision_distill_loss(
                self.student, self.teacher, obs, actions, next_obs, value_fn, rewards,
                m_latents=self.m_latents,
                value_weight=self.decision_weight,
                variance_weight=self.value_variance_weight,
                aleatoric_weight=self.aleatoric_weight,
                normalize_values=self.normalize_values,
            )
            s_loss = parts["total"]
        elif self.decision_weight > 0.0 and value_fn is not None:
            parts = decision_preserving_distill_loss(
                self.student, self.teacher, obs, actions, next_obs, value_fn, rewards,
                value_weight=self.decision_weight,
                variance_weight=self.value_variance_weight,
                state_geometry_weight=self.hybrid_state_weight,
                state_pairwise_weight=self.hybrid_pairwise_weight,
                normalize_values=self.normalize_values,
                guard_enabled=self.guard_enabled,
                guard_min_corr=self.guard_min_corr,
                guard_max_scale=self.guard_max_scale,
            )
            s_loss = parts["total"]
        elif self.preserve_uncertainty:
            parts = uncertainty_preserving_distill_loss(
                self.student, self.teacher, obs, actions, next_obs, rewards,
                mean_weight=self.mean_weight,
                geometry_weight=self.geometry_weight,
                pairwise_weight=self.pairwise_weight,
            )
            s_loss = parts["total"]
        else:
            parts = None
            s_loss = distill_loss(self.student, self.teacher, obs, actions, next_obs, rewards)
        total = t_loss + s_loss
        out = {
            "total": total,
            "dynamics": t_loss.detach(),
            "reward": s_loss.detach(),  # reuse slot as distill loss for logging
        }
        if parts is not None:
            out.update({f"distill_{k}": v.detach() for k, v in parts.items() if k != "total"})
        return out

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
    def sample_next(self, obs, actions, member=None, deterministic=True, steps=None, x_T=None):
        o2, r = self.student.sample_next(self.teacher, obs, actions, member=member, x_T=x_T)
        self.last_reward = r
        return o2

    @torch.no_grad()
    def sample_next_with_reward(self, obs, actions, member=None, deterministic=True, steps=None):
        o2, r = self.student.sample_next(self.teacher, obs, actions, member=member)
        if r is None:
            r = torch.zeros(obs.shape[0], 1, device=obs.device)
        return o2, r

    @torch.no_grad()
    def sample_next_multi(self, obs, actions, m: int, member=None, steps=None, latents=None):
        samples = [
            self.sample_next(obs, actions, member=member, x_T=None if latents is None else latents[j])
            for j in range(m)
        ]
        return torch.stack(samples, 0)

    @torch.no_grad()
    def predict_done(self, obs, actions):
        return self.teacher.predict_done(obs, actions)
