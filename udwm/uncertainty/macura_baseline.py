# -*- coding: utf-8 -*-
"""MACURA-style rollout gating signal for the repo's ensemble student.

Registration: research/MACURA-BASELINE-PREREGISTRATION-2026-09-09.md.
Field baseline per the master plan: MACURA (Frauenknecht, Eisele, Subhasish,
Solowjow & Trimpe, ICML 2024, PMLR 235:13973-14005; arXiv:2405.19014) adapts
model-rollout length per state from the ensemble's *epistemic uncertainty*:
the member-KL uncertainty of Eq. (4),

    u_KL(s, a) = sum_{e=1}^E D_KL( p~_{theta_e}(. | s, a) || p~_PE(. | s, a) ),
    p~_PE(. | s, a) = (1/E) sum_e p~_{theta_e}(. | s, a),

i.e. each member's predictive distribution measured against the equal-weight
ensemble mixture. MACURA keeps rolling where members agree (u_KL small) and
stops where they disagree. (In the paper the epistemic uncertainty is defined
as u_KL; the algorithm's practical estimator u_GJS, Eqs. (15)-(20), is a
pairwise geometric-JS surrogate that exists only because its ensemble members
are Gaussian PNNs with closed-form densities. This repo's members are implicit
diffusion samplers, so u_KL is estimated directly by the kernel-plugin below.)

This module reuses the repo's existing ensemble outputs as the input
distributions (no new model, no retraining): a NEW gating rule on top of the
student/teacher ensemble the trainer already has. ``ukl_disagreement`` has
the same signature as ``one_step_state_disagreement``
(udwm/uncertainty/baselines.py) so it can be dropped into
``u_gated_rollout``'s ``score_fn`` slot, which is exactly what the
``macura_gate`` arm does (u_gate score = u_KL instead of the learned UBE u).

Divergence estimator (documented for the registration): each member's
predictive next-state distribution is represented by ``m_samples`` Monte
Carlo draws at the conditioning (s, a). For each state, the pooled support is
the union of all members' draws; each member's distribution is estimated on
that support by a Gaussian-kernel density (bandwidth = per-state median
within-member pairwise distance) and normalized to a categorical. u_KL is then
the member-KL sum of Eq. (4) over those categoricals (nats). This is a
tractable plugin estimate for implicit diffusion samplers, where no closed-form
density exists; the estimator's variance is controlled by ``m_samples`` and
the batch is processed vectorised across states.

Note on the plugin identity: for the kernel-plugin categoricals the Eq. (4)
sum equals E times the multi-distribution JSD of the same categoricals
(u_KL = E * [H(p_bar) - mean_e H(p_e)]), so earlier drafts that returned the
unmultiplied JSD were reporting u_KL / E. The value returned here is the
Eq. (4) member-KL sum itself, the quantity MACURA's admissible set
(u < kappa) is defined on. The plugin is directional (each member's density
is evaluated on its own kernel support), matching Eq. (4)'s KL direction;
the paper's symmetric u_GJS surrogate is only defined for closed-form
Gaussian members and is not applicable to implicit diffusion samplers.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def ukl_disagreement(
    world_model,
    obs: torch.Tensor,
    actions: torch.Tensor,
    m_samples: int = 8,
    bandwidth: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-state MACURA member-KL uncertainty u_KL (Eq. 4). [B, 1].

    u_KL(s, a) = sum_e D_KL(p~_e(. | s, a) || p~_PE(. | s, a)) with
    p~_PE = (1/E) sum_e p~_e, estimated by kernel-plugin categoricals over
    the pooled Monte Carlo support of all members (see module docstring).

    Args:
        world_model: object exposing ``dynamics`` (per-member next-state
            sampler via ``sample_next_multi(obs, actions, m, member=i)``) and
            ``ensemble_size`` (>= 2). The dynamics adapter already satisfies
            this interface for the consistency student.
        obs: [B, D] conditioning states.
        actions: [B, A] conditioning actions.
        m_samples: Monte Carlo draws per member per state.
        bandwidth: optional fixed kernel bandwidth; default per-state median
            within-member pairwise distance.
    """
    dyn = world_model.dynamics
    n = int(world_model.ensemble_size)
    if n < 2:
        return torch.zeros(obs.shape[0], 1, device=obs.device)
    m = max(2, int(m_samples))
    b = obs.shape[0]
    device = obs.device

    # Per-member sample clouds: [n, m, B, D] (draws are per-member iid).
    clouds = torch.stack(
        [dyn.sample_next_multi(obs, actions, m=m, member=i) for i in range(n)],
        dim=0,
    )

    # Per-state bandwidth: mean over members of the median within-member
    # pairwise distance (fallback: global median of all pairwise distances).
    h2 = None
    if bandwidth is None or bandwidth <= 0:
        medians = []
        for i in range(n):
            x = clouds[i]  # [m, B, D]
            # pairwise squared distances within the member cloud per state
            d2 = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1)  # [m,m,B]
            tri = torch.triu(torch.ones(m, m, device=device, dtype=torch.bool),
                             diagonal=1)
            pairs = d2[tri]  # [P, B], P = m*(m-1)/2
            if pairs.shape[0] > 0:
                medians.append(pairs.median(dim=0).values)
        if medians:
            h = torch.stack(medians, dim=0).mean(dim=0)  # [B]
        else:
            h = torch.ones(b, device=device)
        h2 = h.clamp_min(eps).pow(2)
    else:
        h2 = torch.full((b,), float(bandwidth) ** 2, device=device)

    # Pooled support per state: [B, n*m, D]
    support = clouds.permute(2, 0, 1, 3).reshape(b, n * m, -1)
    # Member categoricals over the pooled support: p_i[k] = mean_j K(x_k, x_ij)
    cats = []
    for i in range(n):
        xi = clouds[i].permute(1, 0, 2)  # [B, m, D]
        d2 = (support.unsqueeze(2) - xi.unsqueeze(1)).pow(2).sum(dim=-1)  # [B, S, m]
        k = torch.exp(-d2 / (2.0 * h2[:, None, None]))
        p = k.mean(dim=-1)  # [B, S]
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        cats.append(p)
    p = torch.stack(cats, dim=0)  # [n, B, S]
    mean_p = p.mean(dim=0)  # [B, S], plugin analogue of p~_PE
    log_p = torch.log(p.clamp_min(eps))
    log_pbar = torch.log(mean_p.clamp_min(eps))
    # u_KL = sum_e sum_x p_e(x) log(p_e(x) / p_bar(x))
    ukl = (p * (log_p - log_pbar)).sum(dim=-1).sum(dim=0)  # [B]
    return ukl.clamp_min(0.0).reshape(-1, 1)


def macura_score_fn(world_model, ukl_m_samples: int = 8):
    """score_fn(o, a) -> [B, 1] for u_gated_rollout's score_fn slot."""
    return lambda o, a: ukl_disagreement(
        world_model, o, a, m_samples=int(ukl_m_samples)
    )
