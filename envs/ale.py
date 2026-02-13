from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from envs.base import StepResult

# Register ALE environments with gymnasium
gym.register_envs(ale_py)


@dataclass
class ALEEnv:
    env_id: str
    obs_type: str = "pixels"
    render_mode: str | None = None
    frameskip: int = 4
    repeat_action_probability: float = 0.0  # "sticky actions" off for determinism

    def __post_init__(self) -> None:
        # Determinism notes:
        # - repeat_action_probability=0.0 reduces stochasticity from sticky actions
        # - We seed reset() and action_space
        self._env = gym.make(
            self.env_id,
            render_mode=self.render_mode,
            frameskip=self.frameskip,
            repeat_action_probability=self.repeat_action_probability,
        )

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        # seed action space too (important if you sample random actions)
        if seed is not None:
            try:
                self._env.action_space.seed(seed)
            except Exception:
                pass
        return np.asarray(obs), dict(info)

    def step(self, action: int) -> StepResult:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return StepResult(
            obs=np.asarray(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def close(self) -> None:
        self._env.close()

    @property
    def action_space_n(self) -> int:
        space = self._env.action_space
        if isinstance(space, Discrete):
            return int(space.n)
        raise TypeError(f"Expected Discrete action space, got {type(space)}: {space}")

    @property
    def action_meanings(self) -> list[str]:
        # Gymnasium ALE exposes action meanings via the unwrapped env
        try:
            return list(self._env.unwrapped.get_action_meanings())
        except Exception:
            # Fallback if unavailable
            return [str(i) for i in range(self.action_space_n)]
