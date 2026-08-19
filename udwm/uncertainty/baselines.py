"""Established uncertainty scores used as baselines for UBE comparisons."""

from __future__ import annotations

import torch


@torch.no_grad()
def one_step_state_disagreement(
    world_model,
    obs: torch.Tensor,
    actions: torch.Tensor,
    m_samples: int = 4,
) -> torch.Tensor:
    """Variance of member conditional next-state means, averaged over dimensions.

    This is the standard one-step ensemble-disagreement object used by many
    uncertainty-aware rollout methods. Gaussian members expose their means
    directly; implicit members approximate each mean with ``m_samples`` draws.
    The returned shape is [B, 1].
    """
    dyn = world_model.dynamics
    n = world_model.ensemble_size
    if n < 2:
        return torch.zeros(obs.shape[0], 1, device=obs.device)

    if getattr(world_model, "model_type", "diffusion") == "gaussian":
        means, _ = dyn.member_means(obs, actions)
    else:
        means = []
        m = max(1, int(m_samples))
        for i in range(n):
            draws = dyn.sample_next_multi(obs, actions, m=m, member=i)
            means.append(draws.mean(dim=0))
        means = torch.stack(means, dim=0)

    return means.var(dim=0, unbiased=False).mean(dim=-1, keepdim=True)
