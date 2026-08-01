"""Adaptive Monte Carlo budgets for MC-UBE (novelty B).

Strategy
--------
1. **Probe** with ``m_min`` next-state samples per ensemble member (cheap).
2. Compute probe epistemic local score ``w_probe`` (disagreement of member means).
3. **Refine** only high-uncertainty batch indices with extra samples up to ``m_max``.
4. Optionally request more diffusion NFE (``k_max``) for those indices when the
   dynamics API supports per-call ``steps``.

This spends compute where ensemble disagreement is large — the regime where
Monte Carlo error in local UBE rewards hurts the multi-step ``U`` most.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from udwm.uncertainty.mc_ube import MCUBELocalRewards


class AdaptiveMCUBELocalRewards(MCUBELocalRewards):
    def __init__(
        self,
        u_min: float = 0.0,
        m_min: int = 2,
        m_max: int = 12,
        m_probe: Optional[int] = None,
        refine_frac: float = 0.5,
        w_refine_percentile: float = 0.5,
        k_min: Optional[int] = None,
        k_max: Optional[int] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(u_min=u_min, m_samples=m_max)
        self.m_min = max(1, int(m_min))
        self.m_max = max(self.m_min, int(m_max))
        self.m_probe = int(m_probe) if m_probe is not None else self.m_min
        self.refine_frac = float(refine_frac)  # max fraction of batch to refine
        self.w_refine_percentile = float(w_refine_percentile)
        self.k_min = k_min
        self.k_max = k_max
        self.adaptive_enabled = bool(enabled)

    @torch.no_grad()
    def _member_stats(
        self,
        world_model,
        q_fn,
        policy_fn,
        obs: torch.Tensor,
        actions: torch.Tensor,
        m: int,
        steps: Optional[int],
        idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return member means/vars for full batch or subset idx. shapes [N,B',1]."""
        model_type = getattr(world_model, "model_type", "diffusion")
        dyn = world_model.dynamics
        n = world_model.ensemble_size
        if idx is not None:
            obs_i = obs[idx]
            act_i = actions[idx]
        else:
            obs_i, act_i = obs, actions

        member_means = []
        member_vars = []
        for i in range(n):
            if model_type == "gaussian":
                next_mean, _ = dyn.sample_next(obs_i, act_i, member=i, deterministic=True)
                mu = q_fn(next_mean, policy_fn(next_mean))
                if m <= 1:
                    var = torch.zeros_like(mu)
                else:
                    qs = []
                    for _ in range(m):
                        o2, _ = dyn.sample_next(obs_i, act_i, member=i, deterministic=False)
                        qs.append(q_fn(o2, policy_fn(o2)))
                    qs_t = torch.stack(qs, 0)
                    var = qs_t.var(0, unbiased=False)
                    # blend mean of stochastic with deterministic mean for stability
                    mu = 0.5 * mu + 0.5 * qs_t.mean(0)
                member_means.append(mu)
                member_vars.append(var)
            else:
                # diffusion: optional steps for adaptive NFE
                kwargs = {"m": m, "member": i}
                if steps is not None and hasattr(dyn, "sample_next_multi"):
                    # sample_next_multi may accept steps via sample_next
                    samples = []
                    for _ in range(m):
                        if hasattr(dyn, "sample_next"):
                            try:
                                s = dyn.sample_next(obs_i, act_i, member=i, steps=steps)
                            except TypeError:
                                s = dyn.sample_next(obs_i, act_i, member=i)
                        else:
                            s = dyn.sample_next_multi(obs_i, act_i, m=1, member=i)[0]
                        samples.append(s)
                    samples_t = torch.stack(samples, 0)
                else:
                    samples_t = dyn.sample_next_multi(obs_i, act_i, m=m, member=i)
                m_sz, b_sz, o_dim = samples_t.shape
                flat_s = samples_t.reshape(m_sz * b_sz, o_dim)
                flat_q = q_fn(flat_s, policy_fn(flat_s)).reshape(m_sz, b_sz, 1)
                member_means.append(flat_q.mean(0))
                member_vars.append(flat_q.var(0, unbiased=False))
        return torch.stack(member_means, 0), torch.stack(member_vars, 0)

    @staticmethod
    def _wg_from_members(means: torch.Tensor, vars_: torch.Tensor, u_min: float):
        w = means.var(dim=0, unbiased=False)
        g = vars_.mean(dim=0)
        u = torch.clamp(w - g, min=u_min)
        return u, w, g

    @torch.no_grad()
    def estimate(
        self,
        world_model,
        q_fn,
        policy_fn,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.adaptive_enabled or self.m_max <= self.m_min:
            self.m_samples = self.m_max
            out = super().estimate(world_model, q_fn, policy_fn, obs, actions)
            out["m_mean"] = torch.tensor(float(self.m_max), device=obs.device)
            out["refine_frac_used"] = torch.tensor(0.0, device=obs.device)
            return out

        b = obs.shape[0]
        device = obs.device
        k_probe = self.k_min

        means_p, vars_p = self._member_stats(
            world_model, q_fn, policy_fn, obs, actions, self.m_probe, k_probe, idx=None
        )
        u, w, g = self._wg_from_members(means_p, vars_p, self.u_min)

        # who to refine: top uncertainty by w
        w_flat = w.reshape(b)
        # threshold by percentile among batch
        thresh = torch.quantile(w_flat.detach(), self.w_refine_percentile)
        refine_mask = w_flat >= thresh
        # cap number refined
        max_refine = max(1, int(self.refine_frac * b))
        if int(refine_mask.sum().item()) > max_refine:
            topk = torch.topk(w_flat, k=max_refine).indices
            refine_mask = torch.zeros(b, dtype=torch.bool, device=device)
            refine_mask[topk] = True

        refine_idx = refine_mask.nonzero(as_tuple=False).squeeze(-1)
        m_used = torch.full((b,), float(self.m_probe), device=device)

        if refine_idx.numel() > 0 and self.m_max > self.m_probe:
            extra = self.m_max  # total samples for refined set
            k_ref = self.k_max if self.k_max is not None else k_probe
            means_r, vars_r = self._member_stats(
                world_model,
                q_fn,
                policy_fn,
                obs,
                actions,
                extra,
                k_ref,
                idx=refine_idx,
            )
            u_r, w_r, g_r = self._wg_from_members(means_r, vars_r, self.u_min)
            u = u.clone()
            w = w.clone()
            g = g.clone()
            u[refine_idx] = u_r
            w[refine_idx] = w_r
            g[refine_idx] = g_r
            m_used[refine_idx] = float(extra)

        return {
            "u": u,
            "w": w,
            "g": g,
            "member_means": means_p,
            "m_mean": m_used.mean(),
            "m_used": m_used,
            "refine_frac_used": refine_mask.float().mean(),
            "w_thresh": thresh.detach(),
        }
