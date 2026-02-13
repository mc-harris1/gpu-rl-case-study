from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
from envs import ALEEnv
from envs.base import obs_hash


@dataclass
class RunSpec:
    env_id: str
    obs_type: str
    seed: int
    steps: int
    policy: str  # "random" until we add more policies!
    frameskip: int
    repeat_action_probability: float


@dataclass
class RunArtifact:
    run_id: str
    created_unix_s: float
    spec: RunSpec
    actions: List[int]
    total_reward: float
    final_obs_hash: str


def make_run_dir(base: Path, run_id: str) -> Path:
    d = base / run_id
    d.mkdir(parents=True, exist_ok=False)
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a deterministic action trace + telemetry.")
    p.add_argument("--env", dest="env_id", default="ALE/Pacman-v5", help="Gymnasium env id.")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for reset + action sampling.")
    p.add_argument("--steps", type=int, default=5000, help="Number of steps to record.")
    p.add_argument("--frameskip", type=int, default=4, help="ALE frameskip.")
    p.add_argument(
        "--sticky", type=float, default=0.0, help="repeat_action_probability (0.0 for determinism)."
    )
    p.add_argument("--runs-dir", type=str, default="runs", help="Output directory for runs.")
    p.add_argument("--render", action="store_true", help="Enable rendering (slower).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = make_run_dir(runs_dir, run_id)

    env = ALEEnv(
        env_id=args.env_id,
        render_mode="human" if args.render else None,
        frameskip=args.frameskip,
        repeat_action_probability=args.sticky,
    )

    spec = RunSpec(
        env_id=args.env_id,
        obs_type="pixels",
        seed=args.seed,
        steps=args.steps,
        policy="random",
        frameskip=args.frameskip,
        repeat_action_probability=args.sticky,
    )

    obs, info = env.reset(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    actions: List[int] = []
    total_reward = 0.0

    telemetry_path = run_dir / "telemetry.csv"
    with telemetry_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "action",
                "reward",
                "terminated",
                "truncated",
                "done",
                "episode_return",
                "obs_hash",
                "wall_ms",
            ]
        )

        episode_return = 0.0

        for step in range(args.steps):
            a = int(rng.integers(low=0, high=env.action_space_n))
            actions.append(a)

            t0 = time.perf_counter()
            sr = env.step(a)
            t1 = time.perf_counter()

            episode_return += sr.reward
            total_reward += sr.reward
            done = sr.terminated or sr.truncated

            w.writerow(
                [
                    step,
                    a,
                    f"{sr.reward:.6f}",
                    int(sr.terminated),
                    int(sr.truncated),
                    int(done),
                    f"{episode_return:.6f}",
                    obs_hash(sr.obs),
                    f"{(t1 - t0) * 1000.0:.3f}",
                ]
            )

            if done:
                # Start a new episode deterministically? Keep it simple:
                # reset with the same seed + step offset so it's reproducible but not identical episode each time.
                # If you want single-episode only, set steps low or break here.
                new_seed = args.seed + step + 1
                obs, info = env.reset(seed=new_seed)
                episode_return = 0.0
            else:
                obs = sr.obs

    final_hash = obs_hash(obs)

    artifact = RunArtifact(
        run_id=run_id,
        created_unix_s=time.time(),
        spec=spec,
        actions=actions,
        total_reward=float(total_reward),
        final_obs_hash=final_hash,
    )

    with (run_dir / "run.json").open("w") as f:
        json.dump(
            {
                "run_id": artifact.run_id,
                "created_unix_s": artifact.created_unix_s,
                "spec": asdict(artifact.spec),
                "actions": artifact.actions,
                "total_reward": artifact.total_reward,
                "final_obs_hash": artifact.final_obs_hash,
            },
            f,
            indent=2,
        )

    env.close()

    print(f"Recorded run: {run_id}")
    print(f"  dir: {run_dir}")
    print(f"  steps: {len(actions)}")
    print(f"  total_reward: {total_reward:.6f}")
    print(f"  final_obs_hash: {final_hash}")


if __name__ == "__main__":
    main()
