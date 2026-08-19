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
        gaussian_mean_blend: float = 0.0,
        sample_split: bool = True,
    ) -> None:
        super().__init__(u_min=u_min, m_samples=m_max)
        # Floor of 2, not 1: at M=1 the within-member variance g is unidentifiable,
        # so neither g nor the finite-M debias exists and u = w - g is meaningless.
        self.m_min = max(2, int(m_min))
        self.m_max = max(self.m_min, int(m_max))
        self.m_probe = max(2, int(m_probe)) if m_probe is not None else self.m_min
        # Weight on the MC mean in the Gaussian branch's member mean.
        # 0.0 == pure closed-form plug-in, matching MCUBELocalRewards.estimate so
        # that adaptive-vs-uniform differs only in M.  A nonzero value changes the
        # estimand (Q(E[s']) -> E[Q(s')]) and is fed to the debias as its noise scale.
        self.gaussian_mean_blend = float(gaussian_mean_blend)
        self.refine_frac = float(refine_frac)  # max fraction of batch to refine
        self.w_refine_percentile = float(w_refine_percentile)
        self.k_min = k_min
        self.k_max = k_max
        self.adaptive_enabled = bool(enabled)
        # Selection and reporting use disjoint samples by default. Reusing the
        # probe makes the reported score conditionally biased after top-k
        # selection (winner's curse), even when the finite-M debias is exact.
        self.sample_split = bool(sample_split)

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
                    var = qs_t.var(0, unbiased=True)
                    beta = self.gaussian_mean_blend
                    if beta > 0.0:
                        mu = (1.0 - beta) * mu + beta * qs_t.mean(0)
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
                member_vars.append(flat_q.var(0, unbiased=m > 1))
        return torch.stack(member_means, 0), torch.stack(member_vars, 0)

    def _wg_at_m(
        self,
        means: torch.Tensor,
        vars_: torch.Tensor,
        m_eff: int,
        noise_scale: float,
    ):
        """(u, w, g) with the finite-M debias applied at this state's own budget.

        Critical for the adaptive scheme: the naive bias scales as 1/M, so mixing
        m_probe and m_max across a batch injects a state-dependent inflation that
        anti-correlates with the refinement decision and compresses the very
        uncertainty ranking refinement is meant to sharpen. Debiasing per-state
        with the M actually spent removes that differential.
        """
        out = self.combine(means, vars_, m_eff=m_eff, mean_noise_scale=noise_scale)
        return out["u"], out["w"], out["g"]

    def _mean_noise_scale(self, world_model) -> float:
        """Coefficient c in Var(member mean) = c^2 sigma^2/M, per model class."""
        if getattr(world_model, "model_type", "diffusion") == "gaussian":
            return self.gaussian_mean_blend
        return 1.0

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

        noise_scale = self._mean_noise_scale(world_model)

        means_p, vars_p = self._member_stats(
            world_model, q_fn, policy_fn, obs, actions, self.m_probe, k_probe, idx=None
        )
        u, w, g = self._wg_at_m(means_p, vars_p, self.m_probe, noise_scale)
        reported_means = means_p.clone()

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
            u_r, w_r, g_r = self._wg_at_m(means_r, vars_r, extra, noise_scale)
            u = u.clone()
            w = w.clone()
            g = g.clone()
            reported_means = reported_means.clone()
            u[refine_idx] = u_r
            w[refine_idx] = w_r
            g[refine_idx] = g_r
            reported_means[:, refine_idx] = means_r
            m_used[refine_idx] = float(extra)

        # Report an independent probe estimate for states that were not
        # refined. The probe above is reserved for selection. This split is
        # essential when adaptive scores are used for calibration or theory;
        # callers can disable it only for a deliberately cheap, biased mode.
        if self.sample_split:
            report_mask = ~refine_mask
            report_idx = report_mask.nonzero(as_tuple=False).squeeze(-1)
            if report_idx.numel() > 0:
                means_s, vars_s = self._member_stats(
                    world_model, q_fn, policy_fn, obs, actions,
                    self.m_probe, k_probe, idx=report_idx,
                )
                u_s, w_s, g_s = self._wg_at_m(
                    means_s, vars_s, self.m_probe, noise_scale
                )
                u[report_idx] = u_s
                w[report_idx] = w_s
                g[report_idx] = g_s
                reported_means[:, report_idx] = means_s
                m_used[report_idx] = float(self.m_probe)

        return {
            "u": u,
            "w": w,
            "g": g,
            "member_means": reported_means,
            "m_mean": m_used.mean(),
            "m_used": m_used,
            "refine_frac_used": refine_mask.float().mean(),
            "w_thresh": thresh.detach(),
            "sample_split": torch.tensor(float(self.sample_split), device=device),
        }
