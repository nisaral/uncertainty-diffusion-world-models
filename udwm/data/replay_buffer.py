from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Circular replay for (s, a, r, s', done) with optional prioritization weights."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.weights = np.ones((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
        weight: float = 1.0,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.weights[self.ptr] = max(float(weight), 1e-8)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        n = obs.shape[0]
        for i in range(n):
            w = float(weights[i]) if weights is not None else 1.0
            self.add(obs[i], actions[i], float(rewards[i]), next_obs[i], float(dones[i]), w)

    def sample(self, batch_size: int, prioritized: bool = False) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        if prioritized and self.size > 0:
            w = self.weights[: self.size].astype(np.float64)
            w = w / (w.sum() + 1e-12)
            idx = np.random.choice(self.size, size=batch_size, replace=True, p=w)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "actions": torch.as_tensor(self.actions[idx], device=self.device),
            "rewards": torch.as_tensor(self.rewards[idx], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "dones": torch.as_tensor(self.dones[idx], device=self.device),
            "weights": torch.as_tensor(self.weights[idx], device=self.device).unsqueeze(-1),
        }

    def sample_numpy(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size
