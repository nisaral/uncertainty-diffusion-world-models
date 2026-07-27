from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from udwm.utils.torch_utils import mlp


class UNetwork(nn.Module):
    """Predicts multi-step epistemic variance U(s,a) via UBE residual learning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.trunk = mlp(obs_dim + action_dim, 1, hidden_dims)
        # softplus output for non-negativity
        self.softplus = nn.Softplus()

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raw = self.trunk(torch.cat([obs, actions], dim=-1))
        return self.softplus(raw) + 1e-6

    def raw(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([obs, actions], dim=-1))


class MCUBELocalRewards:
    """Sample-based local UBE rewards for Gaussian or diffusion ensembles.

    Estimates:
      w(s,a) = Var_i( μ_i ) where μ_i ≈ E[Q̄(s',a') | θ_i]
      g(s,a) = E_i[ Var_{s'~θ_i} Q̄(s',a') ]
      u(s,a) = w - g   (clipped at u_min)

    For Gaussian ensembles with closed-form means, set m_samples=1 and
    use_member_means=True for Luis App. D.1 style estimation.
    """

    def __init__(
        self,
        u_min: float = 0.0,
        m_samples: int = 8,
    ) -> None:
        self.u_min = u_min
        self.m_samples = m_samples

    @torch.no_grad()
    def estimate(
        self,
        world_model,
        q_fn,
        policy_fn,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        q_fn(obs, actions) -> Q values [B,1]
        policy_fn(obs) -> next actions [B, action_dim]
        """
        model_type = getattr(world_model, "model_type", "diffusion")
        dyn = world_model.dynamics
        n = world_model.ensemble_size
        b = obs.shape[0]
        device = obs.device

        member_means = []  # E[Q | θ_i]
        member_vars = []   # Var_{s'}[Q | θ_i]

        for i in range(n):
            if model_type == "gaussian":
                # closed-form mean next state + optional MC noise for aleatoric
                next_mean, _ = dyn.sample_next(obs, actions, member=i, deterministic=True)
                # aleatoric via few stochastic samples
                qs = []
                for _ in range(max(self.m_samples, 1)):
                    o2, _ = dyn.sample_next(obs, actions, member=i, deterministic=self.m_samples <= 1)
                    a2 = policy_fn(o2)
                    qs.append(q_fn(o2, a2))
                qs_t = torch.stack(qs, 0)  # [M,B,1]
                mu = q_fn(next_mean, policy_fn(next_mean))
                # if m_samples==1 and deterministic, var~0 for mean path; use sample var
                if self.m_samples <= 1:
                    var = torch.zeros_like(mu)
                else:
                    var = qs_t.var(dim=0, unbiased=False)
                member_means.append(mu)
                member_vars.append(var)
            else:
                # diffusion: MC samples from member i
                samples = dyn.sample_next_multi(obs, actions, m=self.m_samples, member=i)
                # samples: [M,B,obs]
                m = samples.shape[0]
                qs = []
                for j in range(m):
                    a2 = policy_fn(samples[j])
                    qs.append(q_fn(samples[j], a2))
                qs_t = torch.stack(qs, 0)
                mu = qs_t.mean(0)
                var = qs_t.var(0, unbiased=False)
                member_means.append(mu)
                member_vars.append(var)

        means = torch.stack(member_means, 0)  # [N,B,1]
        vars_ = torch.stack(member_vars, 0)
        w = means.var(dim=0, unbiased=False)  # epistemic
        g = vars_.mean(dim=0)                 # mean aleatoric
        u = w - g
        u = torch.clamp(u, min=self.u_min)
        return {"u": u, "w": w, "g": g, "member_means": means}

    def ube_targets(
        self,
        local_u: torch.Tensor,
        next_u: torch.Tensor,
        gamma: float,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """z = γ² u + γ² (1-d) U(s',a')"""
        g2 = gamma * gamma
        return g2 * local_u + g2 * (1.0 - dones) * next_u


def ube_loss(
    u_net: UNetwork,
    obs: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    reg_weight: float = 1e-3,
) -> torch.Tensor:
    pred = u_net(obs, actions)
    mse = F.mse_loss(pred, targets.detach())
    # penalize large negative raw pre-softplus collapse
    raw = u_net.raw(obs, actions)
    reg = F.relu(-raw - 0.1).pow(2).mean()
    return mse + reg_weight * reg
