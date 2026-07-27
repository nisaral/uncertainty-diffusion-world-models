from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from udwm.utils.torch_utils import mlp


class GaussianEnsemble(nn.Module):
    """PETS-style ensemble of probabilistic MLPs predicting Δs and r.

    Each member outputs mean and log-variance of next-state *delta* and reward.
    Used as (a) Gap-3 baseline and (b) closed-form UBE local-reward reference.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (200, 200, 200),
        min_logvar: float = -10.0,
        max_logvar: float = 0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        # Predict delta_s (obs_dim) + reward (1)
        out_dim = 2 * (obs_dim + 1)
        self.members = nn.ModuleList(
            [mlp(obs_dim + action_dim, out_dim, hidden_dims) for _ in range(ensemble_size)]
        )

    def _split(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # raw: [B, 2*(obs+1)]
        mean, logvar = torch.chunk(raw, 2, dim=-1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        delta_mean, r_mean = mean[..., : self.obs_dim], mean[..., self.obs_dim :]
        delta_logvar, r_logvar = logvar[..., : self.obs_dim], logvar[..., self.obs_dim :]
        return delta_mean, delta_logvar, r_mean, r_logvar

    def forward_member(
        self, i: int, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, actions], dim=-1)
        return self._split(self.members[i](x))

    def forward_all(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns stacked [N, B, ...] means/logvars for delta and reward."""
        deltas_m, deltas_lv, rs_m, rs_lv = [], [], [], []
        for i in range(self.ensemble_size):
            dm, dlv, rm, rlv = self.forward_member(i, obs, actions)
            deltas_m.append(dm)
            deltas_lv.append(dlv)
            rs_m.append(rm)
            rs_lv.append(rlv)
        return (
            torch.stack(deltas_m, 0),
            torch.stack(deltas_lv, 0),
            torch.stack(rs_m, 0),
            torch.stack(rs_lv, 0),
        )

    def nll_loss(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        delta = next_obs - obs
        losses = []
        for i in range(self.ensemble_size):
            dm, dlv, rm, rlv = self.forward_member(i, obs, actions)
            # Gaussian NLL
            inv_var_d = torch.exp(-dlv)
            inv_var_r = torch.exp(-rlv)
            nll_d = ((dm - delta) ** 2) * inv_var_d + dlv
            nll_r = ((rm - rewards) ** 2) * inv_var_r + rlv
            losses.append(0.5 * (nll_d.mean() + nll_r.mean()))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample_next(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        member: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (next_obs, reward). Random member if member is None."""
        b = obs.shape[0]
        if member is None:
            # per-sample random member (model-randomized rollouts)
            idxs = torch.randint(0, self.ensemble_size, (b,), device=obs.device)
            next_list, r_list = [], []
            for i in range(self.ensemble_size):
                mask = idxs == i
                if not mask.any():
                    continue
                dm, dlv, rm, rlv = self.forward_member(i, obs[mask], actions[mask])
                if deterministic:
                    delta, r = dm, rm
                else:
                    delta = dm + torch.randn_like(dm) * torch.exp(0.5 * dlv)
                    r = rm + torch.randn_like(rm) * torch.exp(0.5 * rlv)
                next_list.append((mask, obs[mask] + delta, r))
            next_obs = torch.zeros_like(obs)
            rewards = torch.zeros(b, 1, device=obs.device)
            for mask, no, r in next_list:
                next_obs[mask] = no
                rewards[mask] = r
            return next_obs, rewards
        dm, dlv, rm, rlv = self.forward_member(member, obs, actions)
        if deterministic:
            return obs + dm, rm
        delta = dm + torch.randn_like(dm) * torch.exp(0.5 * dlv)
        r = rm + torch.randn_like(rm) * torch.exp(0.5 * rlv)
        return obs + delta, r

    @torch.no_grad()
    def mean_next(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble-mean next state/reward (closed-form means for UBE)."""
        dm, _, rm, _ = self.forward_all(obs, actions)
        return obs + dm.mean(0), rm.mean(0)

    @torch.no_grad()
    def member_means(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-member mean next_obs and reward: [N, B, ...]."""
        dm, _, rm, _ = self.forward_all(obs, actions)
        return obs.unsqueeze(0) + dm, rm
