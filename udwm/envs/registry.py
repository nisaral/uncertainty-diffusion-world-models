"""Multi-env helpers for continuous-control experiments (Gap 3).

Supports any Gymnasium env with Box observation/action spaces.
Optional DeepMind Control via ``shimmy`` if installed:

    pip install "shimmy[dm-control]"

Example ids:
  - Pendulum-v1
  - MountainCarContinuous-v0
  - dm_control/cartpole-balance-v0   (requires shimmy + dm_control)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


KNOWN_VECTOR_ENVS = {
    "Pendulum-v1": {"notes": "Classic continuous control; default for CPU demos."},
    "MountainCarContinuous-v0": {"notes": "Sparse-ish continuous; good exploration stress."},
    "LunarLanderContinuous-v3": {"notes": "Higher-dim continuous; needs gymnasium[box2d]."},
    "DelayedBimodal-v0": {"notes": "Stochastic delayed-consequence uncertainty stress task."},
}


def make_env(env_id: str, seed: int | None = None, **kwargs) -> gym.Env:
    """Create a Gymnasium env; tries dm_control shim if id starts with dm_control/."""
    if env_id == "DelayedBimodal-v0":
        from udwm.envs.delayed_bimodal import DelayedBimodalEnv

        env = DelayedBimodalEnv(**kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return env
    if env_id.startswith("dm_control/") or env_id.startswith("dm-control/"):
        try:
            # Shimmy registers dm_control/* ids when imported
            import shimmy  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DeepMind Control envs require: pip install 'shimmy[dm-control]'"
            ) from e
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def space_info(env: gym.Env) -> Dict[str, Any]:
    obs_space = env.observation_space
    act_space = env.action_space
    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
        raise ValueError(
            f"UDWM shared core expects 1D Box observations; got {obs_space}. "
            "Use a state-based env or flatten observations."
        )
    if not isinstance(act_space, gym.spaces.Box) or len(act_space.shape) != 1:
        raise ValueError(f"Expect continuous Box actions; got {act_space}")
    return {
        "obs_dim": int(obs_space.shape[0]),
        "action_dim": int(act_space.shape[0]),
        "action_low": act_space.low.astype(np.float32),
        "action_high": act_space.high.astype(np.float32),
    }


def list_recommended() -> str:
    lines = ["Recommended continuous-control envs:"]
    for k, v in KNOWN_VECTOR_ENVS.items():
        lines.append(f"  - {k}: {v['notes']}")
    lines.append("  - dm_control/* via shimmy (optional install)")
    return "\n".join(lines)
