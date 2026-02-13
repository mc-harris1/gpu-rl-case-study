from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np


class Policy(Protocol):
    name: str

    def reset(self, *, seed: int, action_meanings: list[str], action_space_n: int) -> None: ...
    def act(self, step: int, obs: np.ndarray, reward: float, done: bool) -> int: ...


@dataclass
class RandomPolicy:
    name: str = "random"

    def reset(self, *, seed: int, action_meanings: list[str], action_space_n: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.n = action_space_n

    def act(self, step: int, obs: np.ndarray, reward: float, done: bool) -> int:
        return int(self.rng.integers(0, self.n))


@dataclass
class StickyDirectionalPolicy:
    """
    Tiny scripted policy intended to make telemetry "less boring" than random:

    - Prefer a directional action (UP/DOWN/LEFT/RIGHT) when available.
    - Stick with current direction.
    - If reward hasn't improved for `stuck_window` steps, rotate direction.
    - Occasional random shake-up to avoid infinite loops.

    This does NOT "solve" Pacman. It's just more structured than random.
    """

    name: str = "sticky_dir"
    stuck_window: int = 30
    jitter_prob: float = 0.02

    def reset(self, *, seed: int, action_meanings: list[str], action_space_n: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.n = action_space_n
        self.meanings = action_meanings

        # Identify indices for common directional actions if present.
        # ALE typically has these meanings.
        dirs = ["UP", "RIGHT", "DOWN", "LEFT"]
        self.dir_actions: list[int] = []
        for d in dirs:
            if d in self.meanings:
                self.dir_actions.append(self.meanings.index(d))

        # Fallback: if meanings don't contain directions, just use first 4 actions (best effort).
        if not self.dir_actions:
            self.dir_actions = list(range(min(4, self.n)))

        self.cur_idx = 0
        self.cur_action = self.dir_actions[self.cur_idx]

        self._since_progress = 0

    def act(self, step: int, obs: np.ndarray, reward: float, done: bool) -> int:
        if done:
            self._since_progress = 0
            return self.cur_action

        # Consider any positive reward "progress" for this toy script.
        if reward > 0:
            self._since_progress = 0
        else:
            self._since_progress += 1

        # Occasional jitter
        if self.rng.random() < self.jitter_prob:
            self.cur_idx = int(self.rng.integers(0, len(self.dir_actions)))
            self.cur_action = self.dir_actions[self.cur_idx]
            self._since_progress = 0
            return self.cur_action

        # If stuck, rotate direction
        if self._since_progress >= self.stuck_window:
            self.cur_idx = (self.cur_idx + 1) % len(self.dir_actions)
            self.cur_action = self.dir_actions[self.cur_idx]
            self._since_progress = 0

        return self.cur_action


_POLICIES: Dict[str, Policy] = {
    "random": RandomPolicy(),
    "sticky_dir": StickyDirectionalPolicy(),
}


def list_policies() -> list[str]:
    return sorted(_POLICIES.keys())


def make_policy(name: str) -> Policy:
    if name not in _POLICIES:
        valid = ", ".join(list_policies())
        raise ValueError(f"Unknown policy '{name}'. Valid: {valid}")
    # Return a fresh instance so runs don't share state
    p = _POLICIES[name]
    return type(p)(**getattr(p, "__dict__", {}))  # re-instantiate with same params
