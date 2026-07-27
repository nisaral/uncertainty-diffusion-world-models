from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from udwm.utils.torch_utils import mlp


class RewardTerminationModel(nn.Module):
    """Separate reward + termination heads (DIAMOND-style baseline for Gap 1).

    Gap 1 extension point: replace this with joint diffusion of (s', r, d)
    while keeping the same predict() interface.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        predict_termination: bool = True,
    ) -> None:
        super().__init__()
        self.predict_termination = predict_termination
        out_dim = 2 if predict_termination else 1  # reward, done_logit
        self.net = mlp(obs_dim + action_dim, out_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([obs, actions], dim=-1))
        if self.predict_termination:
            reward = out[..., :1]
            done_logit = out[..., 1:]
        else:
            reward = out
            done_logit = torch.zeros_like(reward)
        return reward, done_logit

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pred_r, done_logit = self.forward(obs, actions)
        loss_r = F.mse_loss(pred_r, rewards)
        if self.predict_termination:
            loss_d = F.binary_cross_entropy_with_logits(done_logit, dones.clamp(0, 1))
            return loss_r + loss_d
        return loss_r

    @torch.no_grad()
    def predict(
        self, obs: torch.Tensor, actions: torch.Tensor, sample_done: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reward, done_logit = self.forward(obs, actions)
        if sample_done:
            done = torch.bernoulli(torch.sigmoid(done_logit))
        else:
            done = (torch.sigmoid(done_logit) > 0.5).float()
        return reward, done
