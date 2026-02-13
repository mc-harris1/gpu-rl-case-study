from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from envs.ale import ALEEnv


@dataclass(frozen=True)
class EnvSpec:
    key: str
    env_id: str
    obs_type: str  # "pixels" or "state" (use "state" for RAM-like)
    description: str


_ENV_SPECS: Dict[str, EnvSpec] = {
    "pacman": EnvSpec(
        key="pacman",
        env_id="ALE/Pacman-v5",
        obs_type="pixels",
        description="Atari Pacman (pixels)",
    ),
    "pacman-ram": EnvSpec(
        key="pacman-ram",
        env_id="ALE/Pacman-ram-v5",
        obs_type="state",
        description="Atari Pacman (RAM/state)",
    ),
}


def list_envs() -> List[EnvSpec]:
    return list(_ENV_SPECS.values())


def get_env_spec(key: str) -> EnvSpec:
    if key not in _ENV_SPECS:
        valid = ", ".join(sorted(_ENV_SPECS.keys()))
        raise ValueError(f"Unknown env key '{key}'. Valid: {valid}")
    return _ENV_SPECS[key]


def make_env(
    env_key: str,
    *,
    render_mode: str | None,
    frameskip: int,
    repeat_action_probability: float,
) -> tuple[EnvSpec, ALEEnv]:
    spec = get_env_spec(env_key)
    env = ALEEnv(
        env_id=spec.env_id,
        render_mode=render_mode,
        frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
    )
    return spec, env
