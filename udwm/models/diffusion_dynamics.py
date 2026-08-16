from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from udwm.utils.torch_utils import mlp


class NoiseSchedule:
    """Linear beta schedule for discrete-time DDPM."""

    def __init__(
        self,
        T: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self


class ConditionalDenoiser(nn.Module):
    """MLP denoiser: predict noise ε given (x_t, t, s, a)."""

    def __init__(
        self,
        x_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (200, 200, 200),
        t_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
        )
        in_dim = x_dim + obs_dim + action_dim + t_embed_dim
        self.net = mlp(in_dim, x_dim, hidden_dims)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t.float().unsqueeze(-1)
        else:
            t = t.float()
        t_feat = self.t_embed(t)
        h = torch.cat([x_t, obs, actions, t_feat], dim=-1)
        return self.net(h)


class DiffusionDynamicsEnsemble(nn.Module):
    """Ensemble of conditional diffusion models for p(s'|s,a) via Δs.

    If joint_reward=True (Gap 1): denoise x = [Δs_norm, r_norm] jointly so
    reward is part of the generative process — not a bolted-on head.
    Termination remains a light auxiliary head (binary) for stability.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (200, 200, 200),
        diffusion_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        sample_steps: int = 4,
        joint_reward: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.joint_reward = joint_reward
        self.x_dim = obs_dim + (1 if joint_reward else 0)
        self.schedule = NoiseSchedule(diffusion_steps, beta_start, beta_end)
        self.members = nn.ModuleList(
            [
                ConditionalDenoiser(self.x_dim, obs_dim, action_dim, hidden_dims)
                for _ in range(ensemble_size)
            ]
        )
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))
        self.register_buffer("reward_mean", torch.zeros(1))
        self.register_buffer("reward_std", torch.ones(1))
        self.register_buffer("count", torch.tensor(0.0))
        # Aux termination (even in joint mode — binary is awkward in pure Gaussian noise space)
        self.term_head = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.schedule.to(device)
        return self

    @torch.no_grad()
    def update_stats(
        self, obs: torch.Tensor, next_obs: torch.Tensor, rewards: Optional[torch.Tensor] = None
    ) -> None:
        delta = next_obs - obs
        b = delta.shape[0]
        batch_mean = delta.mean(0)
        batch_var = delta.var(0, unbiased=False)
        n0 = float(self.count.item())
        n1 = n0 + b
        if n0 == 0:
            self.delta_mean.copy_(batch_mean)
            self.delta_std.copy_(torch.sqrt(batch_var + 1e-6))
            if rewards is not None and self.joint_reward:
                self.reward_mean.copy_(rewards.mean(0))
                self.reward_std.copy_(torch.sqrt(rewards.var(0, unbiased=False) + 1e-6))
        else:
            self.delta_mean.copy_((self.delta_mean * n0 + batch_mean * b) / n1)
            self.delta_std.copy_(
                torch.sqrt((self.delta_std**2 * n0 + batch_var * b) / n1 + 1e-6)
            )
            if rewards is not None and self.joint_reward:
                self.reward_mean.copy_((self.reward_mean * n0 + rewards.mean(0) * b) / n1)
                self.reward_std.copy_(
                    torch.sqrt(
                        (self.reward_std**2 * n0 + rewards.var(0, unbiased=False) * b) / n1 + 1e-6
                    )
                )
        self.count.fill_(n1)

    def _pack_x(
        self, delta: torch.Tensor, rewards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        delta_n = (delta - self.delta_mean) / (self.delta_std + 1e-6)
        if not self.joint_reward:
            return delta_n
        assert rewards is not None
        r_n = (rewards - self.reward_mean) / (self.reward_std + 1e-6)
        return torch.cat([delta_n, r_n], dim=-1)

    def _unpack_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta_n = x[..., : self.obs_dim]
        delta = delta_n * (self.delta_std + 1e-6) + self.delta_mean
        if not self.joint_reward:
            return delta, None
        r_n = x[..., self.obs_dim : self.obs_dim + 1]
        r = r_n * (self.reward_std + 1e-6) + self.reward_mean
        return delta, r

    def diffusion_loss_member(
        self,
        i: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = next_obs - obs
        x0 = self._pack_x(delta, rewards)
        b = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.diffusion_steps, (b,), device=device)
        noise = torch.randn_like(x0)
        sqrt_ab = self.schedule.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_om = self.schedule.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_om * noise
        t_scaled = (t.float() + 1.0) / self.diffusion_steps
        pred = self.members[i](x_t, t_scaled, obs, actions)
        return F.mse_loss(pred, noise)

    def nll_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.update_stats(obs, next_obs, rewards)
        losses = [
            self.diffusion_loss_member(i, obs, actions, next_obs, rewards)
            for i in range(self.ensemble_size)
        ]
        loss = torch.stack(losses).mean()
        if dones is not None:
            logit = self.term_head(torch.cat([obs, actions], dim=-1))
            loss = loss + F.binary_cross_entropy_with_logits(logit, dones.clamp(0, 1))
        return loss

    @torch.no_grad()
    def _ddim_sample_member(
        self,
        i: int,
        obs: torch.Tensor,
        actions: torch.Tensor,
        steps: Optional[int] = None,
        deterministic: bool = True,
        mc_mean_samples: int = 0,
        x_T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x0 for member i.

        The DDIM update below is deterministic *given* x_T, so all sampling
        randomness enters through the x_T ~ N(0,I) draw. ``deterministic=True``
        therefore does NOT yield a point prediction on its own: use
        ``mc_mean_samples > 0`` to average over x_T and approximate E[x0], which
        is the right quantity to compare against a Gaussian ensemble's analytic
        mean (see ``member_mean_next``).

        ``x_T`` lets the caller supply the latent instead of drawing it. Passing
        the *same* x_T to several members couples them: each member's marginal
        law is unchanged, but their sampling errors become positively correlated,
        which is what shrinks the finite-M bias and variance of the ensemble
        variance ``w``. See ``research/COUPLED-MC-UBE-PROPOSAL.md``.
        """
        if deterministic and mc_mean_samples > 0:
            xs = [
                self._ddim_sample_member(i, obs, actions, steps, deterministic=False)
                for _ in range(mc_mean_samples)
            ]
            return torch.stack(xs, 0).mean(0)
        steps = steps or self.sample_steps
        b = obs.shape[0]
        device = obs.device
        if x_T is None:
            x = torch.randn(b, self.x_dim, device=device)
        else:
            if x_T.shape != (b, self.x_dim):
                raise ValueError(
                    f"x_T must have shape {(b, self.x_dim)}, got {tuple(x_T.shape)}"
                )
            x = x_T.to(device)
        ts = torch.linspace(self.diffusion_steps - 1, 0, steps, device=device).long()
        for j, t in enumerate(ts):
            t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
            t_scaled = (t_batch.float() + 1.0) / self.diffusion_steps
            eps = self.members[i](x, t_scaled, obs, actions)
            ab = self.schedule.alpha_bar[t]
            sqrt_ab = torch.sqrt(ab)
            sqrt_om = torch.sqrt(1.0 - ab)
            x0 = (x - sqrt_om * eps) / (sqrt_ab + 1e-8)
            if j == steps - 1:
                x = x0
                break
            t_prev = ts[j + 1]
            ab_prev = self.schedule.alpha_bar[t_prev]
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps
        return x

    @torch.no_grad()
    def sample_next(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        member: Optional[int] = None,
        deterministic: bool = True,
        steps: Optional[int] = None,
        mc_mean_samples: int = 0,
    ) -> torch.Tensor:
        """Sample next_obs. If joint_reward, also caches last_reward on self."""
        next_obs, reward = self.sample_next_with_reward(
            obs,
            actions,
            member=member,
            deterministic=deterministic,
            steps=steps,
            mc_mean_samples=mc_mean_samples,
        )
        self.last_reward = reward
        return next_obs

    @torch.no_grad()
    def sample_next_with_reward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        member: Optional[int] = None,
        deterministic: bool = True,
        steps: Optional[int] = None,
        mc_mean_samples: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = obs.shape[0]
        device = obs.device
        if member is None:
            idxs = torch.randint(0, self.ensemble_size, (b,), device=device)
            next_obs = torch.zeros_like(obs)
            rewards = torch.zeros(b, 1, device=device)
            for i in range(self.ensemble_size):
                mask = idxs == i
                if not mask.any():
                    continue
                x = self._ddim_sample_member(
                    i, obs[mask], actions[mask], steps, deterministic, mc_mean_samples
                )
                delta, r = self._unpack_x(x)
                next_obs[mask] = obs[mask] + delta
                if r is not None:
                    rewards[mask] = r
            if not self.joint_reward:
                rewards = torch.zeros(b, 1, device=device)
            return next_obs, rewards
        x = self._ddim_sample_member(
            member, obs, actions, steps, deterministic, mc_mean_samples
        )
        delta, r = self._unpack_x(x)
        if r is None:
            r = torch.zeros(b, 1, device=device)
        return obs + delta, r

    @torch.no_grad()
    def predict_mean(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        m: int = 16,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble-mean point prediction E[s'], E[r] via MC over x_T.

        This is the counterpart of ``GaussianEnsemble.mean_next``. Scoring a
        single stochastic draw against the ground truth (as a bare
        ``sample_next`` call does) inflates MSE by the model's own aleatoric
        variance, which penalises the diffusion model for being stochastic
        rather than for being wrong.
        """
        nxt, rew = [], []
        for i in range(self.ensemble_size):
            xs = [
                self._ddim_sample_member(i, obs, actions, steps, deterministic=False)
                for _ in range(m)
            ]
            x = torch.stack(xs, 0).mean(0)
            delta, r = self._unpack_x(x)
            nxt.append(obs + delta)
            rew.append(r if r is not None else torch.zeros(obs.shape[0], 1, device=obs.device))
        return torch.stack(nxt, 0).mean(0), torch.stack(rew, 0).mean(0)

    @torch.no_grad()
    def predict_done(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logit = self.term_head(torch.cat([obs, actions], dim=-1))
        return (torch.sigmoid(logit) > 0.5).float()

    @torch.no_grad()
    def sample_next_multi(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        m: int,
        member: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        samples = [
            self.sample_next(obs, actions, member=member, steps=steps) for _ in range(m)
        ]
        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def sample_next_coupled(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        m: int,
        steps: Optional[int] = None,
        coupled: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw m samples from EVERY member, sharing latents across members.

        Returns ``(next_obs, rewards)`` shaped ``[N, m, B, obs_dim]`` and
        ``[N, m, B, 1]``.

        With ``coupled=True`` the m latents are drawn once and reused for all N
        members, so member i and member j see the same x_T. Because the DDIM map
        is deterministic given x_T, this is a genuine coupling: each member's
        marginal is untouched (so w*, g*, u* are unchanged as estimands) while
        the members' MC errors become positively correlated. The finite-M bias of
        the ensemble variance is exactly ``(g* - mean(Cov))/M``, i.e. the mean
        pairwise *disagreement* variance over M, so correlating the members
        shrinks it. Independent sampling is the worst case.

        ``coupled=False`` draws fresh latents per member, reproducing the
        independent-sampling behaviour, and exists so the two can be compared as
        the single varied factor.
        """
        b = obs.shape[0]
        device = obs.device
        shared = (
            [torch.randn(b, self.x_dim, device=device) for _ in range(m)]
            if coupled
            else None
        )
        nxt, rew = [], []
        for i in range(self.ensemble_size):
            n_i, r_i = [], []
            for j in range(m):
                x_T = shared[j] if shared is not None else None
                x = self._ddim_sample_member(
                    i, obs, actions, steps, deterministic=False, x_T=x_T
                )
                delta, r = self._unpack_x(x)
                n_i.append(obs + delta)
                r_i.append(r if r is not None else torch.zeros(b, 1, device=device))
            nxt.append(torch.stack(n_i, 0))
            rew.append(torch.stack(r_i, 0))
        return torch.stack(nxt, 0), torch.stack(rew, 0)

    @torch.no_grad()
    def member_mean_next(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        m: int = 8,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        means = []
        for i in range(self.ensemble_size):
            samples = self.sample_next_multi(obs, actions, m=m, member=i, steps=steps)
            means.append(samples.mean(0))
        return torch.stack(means, 0)
