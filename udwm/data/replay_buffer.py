from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Simple circular replay for (s, a, r, s', done, terminated)."""

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
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        for i in range(n):
            self.add(obs[i], actions[i], float(rewards[i]), next_obs[i], float(dones[i]))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "actions": torch.as_tensor(self.actions[idx], device=self.device),
            "rewards": torch.as_tensor(self.rewards[idx], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "dones": torch.as_tensor(self.dones[idx], device=self.device),
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
