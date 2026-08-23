"""Stochastic continuous-control task with delayed branch consequences."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DelayedBimodalEnv(gym.Env):
    """A small Markov task designed to stress decision-relevant uncertainty.

    Actions influence a latent left/right branch. Its consequence is delivered
    after a short delay, so one-step state error is not sufficient to identify
    the risky transitions. The full latent countdown and pending impulse are in
    the observation, keeping the environment Markov.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int = 120, delay: int = 4):
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self.delay = int(delay)
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.0, -3.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([4.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng()
        self.state = np.zeros(5, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        position = self._rng.uniform(-0.4, 0.4)
        velocity = self._rng.normal(0.0, 0.05)
        context = self._rng.choice([-1.0, 1.0])
        self.state = np.array([position, velocity, context, 0.0, 0.0], dtype=np.float32)
        self._pending = 0.0
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        position, velocity, context, countdown, pending = map(float, self.state)
        self._pending = pending
        if countdown <= 0.0:
            probability_right = 1.0 / (1.0 + np.exp(-(1.8 * position + 1.6 * a + 0.9 * context)))
            branch = 1.0 if self._rng.random() < probability_right else -1.0
            self._pending = branch * (0.45 + 0.35 * abs(a))
            countdown_steps = self.delay
        else:
            countdown_steps = max(0, int(round(countdown * self.delay)) - 1)

        impulse = self._pending if countdown_steps == 0 else 0.0
        if impulse != 0.0:
            self._pending = 0.0
        noise = self._rng.normal(0.0, 0.025)
        velocity = 0.82 * velocity + 0.22 * a + impulse + noise
        position = np.clip(position + velocity, -4.0, 4.0)
        # Context changes slowly and makes identical local states have different
        # downstream desirability unless the model represents it correctly.
        if self._rng.random() < 0.025:
            context *= -1.0
        target = 1.7 * context
        reward = -((position - target) ** 2) - 0.04 * a * a - 0.02 * velocity * velocity
        self.steps += 1
        terminated = bool(abs(position) >= 3.95)
        truncated = self.steps >= self.max_episode_steps
        self.state = np.array(
            [position, velocity, context, countdown_steps / self.delay, self._pending],
            dtype=np.float32,
        )
        return self.state.copy(), float(reward), terminated, truncated, {}
