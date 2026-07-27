from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from udwm.utils.torch_utils import mlp, soft_update


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, 2 * action_dim, hidden_dims)
        self.action_dim = action_dim
        if action_low is None:
            action_low = -np.ones(action_dim)
        if action_high is None:
            action_high = np.ones(action_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        y = torch.tanh(mean)
        return y * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, 1, hidden_dims)
        self.q2 = mlp(obs_dim + action_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, actions], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, actions)
        return torch.min(q1, q2)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        hidden_dims: Sequence[int] = (256, 256),
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        optimism_lambda: float = 0.0,
        use_ube: bool = True,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.optimism_lambda = optimism_lambda
        self.use_ube = use_ube

        self.actor = SquashedGaussianActor(
            obs_dim, action_dim, hidden_dims, action_low, action_high
        ).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dims).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.u_net: Optional[nn.Module] = None
        self.u_opt: Optional[torch.optim.Optimizer] = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def attach_u_net(self, u_net: nn.Module, lr: float = 3e-4) -> None:
        self.u_net = u_net.to(self.device)
        self.u_opt = torch.optim.Adam(self.u_net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(o)
            else:
                a, _ = self.actor.sample(o)
        return a.cpu().numpy()[0]

    def policy_tensor(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.actor.deterministic(obs)
        a, _ = self.actor.sample(obs)
        return a

    def q_min(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic.min_q(obs, actions)

    def update_critics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs, actions = batch["obs"], batch["actions"]
        rewards, next_obs, dones = batch["rewards"], batch["next_obs"], batch["dones"]

        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_logp
            # optional UBE bonus/penalty on target
            if self.use_ube and self.u_net is not None and abs(self.optimism_lambda) > 0:
                u = self.u_net(next_obs, next_a)
                q_t = q_t + self.optimism_lambda * torch.sqrt(u + 1e-8)
            target = rewards + (1.0 - dones) * self.gamma * q_t

        q1, q2 = self.critic(obs, actions)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return {"critic_loss": float(loss.item())}

    def update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions, logp = self.actor.sample(obs)
        q = self.critic.min_q(obs, actions)
        if self.use_ube and self.u_net is not None and abs(self.optimism_lambda) > 0:
            u = self.u_net(obs, actions)
            q = q + self.optimism_lambda * torch.sqrt(u.detach() + 1e-8)
        actor_loss = (self.alpha.detach() * logp - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        info = {"actor_loss": float(actor_loss.item())}
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            info["alpha"] = float(self.alpha.item())
            info["alpha_loss"] = float(alpha_loss.item())
        return info

    def soft_update_targets(self) -> None:
        soft_update(self.critic_target, self.critic, self.tau)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        info = {}
        info.update(self.update_critics(batch))
        info.update(self.update_actor(batch))
        self.soft_update_targets()
        return info
