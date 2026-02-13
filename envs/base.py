from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class EnvAdapter(Protocol):
    """A minimal, stable interface so we can swap environments later."""

    env_id: str
    obs_type: str  # "pixels" or "state"

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]: ...
    def step(self, action: int) -> StepResult: ...
    def close(self) -> None: ...

    @property
    def action_space_n(self) -> int: ...


def obs_hash(obs: np.ndarray) -> str:
    """Stable hash for determinism checks. Avoids storing huge arrays."""
    # Ensure contiguous bytes
    b = np.ascontiguousarray(obs).tobytes()
    # Cheap, stable hash
    import hashlib

    # Include both the shape and the flattened array data in the hash
    data = (obs.shape, b)
    return hashlib.sha256(str(data).encode()).hexdigest()
