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

    Finite-M bias
    -------------
    ``w`` is a variance over *noisy* member means, so with the naive ddof=0
    convention each mean contributes its own MC noise::

        E[w_hat] = w* + ((N-1)/N) * g*/M
        E[g_hat] = g* * (M-1)/M
        E[u_hat] = u* + g* * (2N-1)/(N*M)

    i.e. an O(1/M) upward bias proportional to the aleatoric term ``g`` -- the
    very quantity ``u = w - g`` exists to subtract.  ``debias=True`` applies the
    closed-form correction, which is exactly unbiased for ``u*`` at any M >= 2.
    See ``theory/estimator_bias.py`` for the derivation (its numerical check has
    not been run yet) and ``research/ESTIMATOR-BIAS-FINDING.md`` for the writeup.
    """

    def __init__(
        self,
        u_min: float = 0.0,
        m_samples: int = 8,
        debias: bool = True,
    ) -> None:
        self.u_min = u_min
        self.m_samples = m_samples
        self.debias = debias

    def combine(
        self,
        means: torch.Tensor,
        vars_unbiased: torch.Tensor,
        m_eff: int,
        mean_noise_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Combine per-member stats into (u, w, g).

        means:         [N,B,1] estimates of E[Q | theta_i]
        vars_unbiased: [N,B,1] ddof=1 estimates of Var[Q | theta_i]
        m_eff:         samples per member used for ``means``
        mean_noise_scale: coefficient ``c`` with ``Var(means[i]) = c^2 sigma_i^2/m_eff``.
                       1.0 for a pure M-sample MC mean (diffusion), 0.0 for an exact
                       closed-form plug-in (Gaussian, Luis App. D.1), and ``beta`` for
                       a ``(1-beta)*plug_in + beta*mc_mean`` blend.  The debias must
                       track this: correcting a plug-in mean would over-subtract, and
                       skipping the correction on a blend under-subtracts by c^2.
        """
        n = means.shape[0]
        w_raw = means.var(dim=0, unbiased=False)
        g = vars_unbiased.mean(dim=0)
        w = w_raw
        c2 = float(mean_noise_scale) ** 2
        if self.debias and c2 > 0.0 and m_eff > 1:
            w = w_raw - ((n - 1) / n) * c2 * g / float(m_eff)
        u = torch.clamp(w - g, min=self.u_min)
        return {"u": u, "w": w, "w_raw": w_raw, "g": g, "member_means": means}

    def combine_coupled(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Coupled-sample estimator from per-sample Q values ``q`` of shape [N,M,B,1].

        Generalises ``combine`` to samples that may be *dependent across members*
        (e.g. drawn with a shared DDIM latent, see
        ``DiffusionDynamicsEnsemble.sample_next_coupled``). Allowing dependence
        changes the finite-M bias from a constant into a coupling-dependent term::

            E[w_hat] = w* + (g* - mean(Sigma))/M,
            mean(Sigma) = Var(mean_i Y_i) = (1/N^2) sum_ij Cov(Y_i, Y_j)

        and by the identity ``g* - mean(Sigma) = (1/2N^2) sum_ij Var(Y_i - Y_j)``
        the bias is exactly the mean pairwise *disagreement* variance over M.
        Independent sampling maximises it; ``g* - mean(Sigma) >= 0`` always, so a
        coupling can only help. Both terms are estimable from the same samples,
        so the correction needs no knowledge of the coupling and stays exactly
        unbiased for ``u*`` at any M >= 2.

        Setting the off-diagonal covariances to zero recovers ``combine``'s
        ``(N-1)/N * g/M`` correction, which is this estimator's
        independent-sampling special case.

        ``g - mean(Sigma)`` is left unclamped: it is non-negative in population
        but its estimate can dip below zero on noise, and clamping would trade
        the exact unbiasedness that is the point of the correction.
        """
        n, m = q.shape[0], q.shape[1]
        mu = q.mean(dim=1)                                   # [N,B,1]
        w_raw = mu.var(dim=0, unbiased=False)                # [B,1]
        z = q - mu.unsqueeze(1)                              # centred, [N,M,B,1]
        denom = float(max(m - 1, 1))
        g = (z.pow(2).sum(dim=1) / denom).mean(dim=0)        # [B,1]
        sigma_bar = z.mean(dim=0).pow(2).sum(dim=0) / denom  # [B,1] = Var(mean_i Y_i)

        w = w_raw
        if self.debias and m > 1:
            w = w_raw - (g - sigma_bar) / float(m)
        u = torch.clamp(w - g, min=self.u_min)
        return {
            "u": u, "w": w, "w_raw": w_raw, "g": g,
            "sigma_bar": sigma_bar, "member_means": mu,
            # coupling diagnostic: 0 => independent, 1 => members move together.
            # This is the bias/variance reduction factor and the single number
            # that decides whether coupling is worth anything on real models.
            "coupling": 1.0 - (g - sigma_bar) / (g * (1.0 - 1.0 / n) + 1e-12),
        }

    @torch.no_grad()
    def estimate_coupled(
        self,
        world_model,
        q_fn,
        policy_fn,
        obs: torch.Tensor,
        actions: torch.Tensor,
        coupled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Coupled-latent variant of ``estimate`` (diffusion ensembles only).

        ``coupled=False`` draws independent latents per member, so the two modes
        differ in exactly one factor and can be compared directly.
        """
        dyn = world_model.dynamics
        if not hasattr(dyn, "sample_next_coupled"):
            raise TypeError(
                "estimate_coupled requires a sampler with shared-latent support "
                f"(got {type(dyn).__name__}); use estimate() instead"
            )
        m = max(self.m_samples, 2)
        samples, _ = dyn.sample_next_coupled(obs, actions, m=m, coupled=coupled)
        n, m_sz, b_sz, o_dim = samples.shape
        flat_s = samples.reshape(n * m_sz * b_sz, o_dim)
        flat_q = q_fn(flat_s, policy_fn(flat_s)).reshape(n, m_sz, b_sz, 1)
        return self.combine_coupled(flat_q)

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
                    var = qs_t.var(dim=0, unbiased=True)
                member_means.append(mu)
                member_vars.append(var)
            else:
                # diffusion / student: batch M samples then one Q pass on flat batch
                samples = dyn.sample_next_multi(obs, actions, m=self.m_samples, member=i)
                # samples: [M, B, obs_dim]
                m_sz, b_sz, o_dim = samples.shape
                flat_s = samples.reshape(m_sz * b_sz, o_dim)
                flat_a = policy_fn(flat_s)
                flat_q = q_fn(flat_s, flat_a).reshape(m_sz, b_sz, 1)
                mu = flat_q.mean(0)
                var = flat_q.var(0, unbiased=self.m_samples > 1)
                member_means.append(mu)
                member_vars.append(var)

        means = torch.stack(member_means, 0)  # [N,B,1]
        vars_ = torch.stack(member_vars, 0)
        # Gaussian branch uses exact member means (Luis App. D.1 plug-in), so they
        # carry no MC noise; the diffusion branch estimates them from M samples.
        return self.combine(
            means,
            vars_,
            m_eff=self.m_samples,
            mean_noise_scale=0.0 if model_type == "gaussian" else 1.0,
        )

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
