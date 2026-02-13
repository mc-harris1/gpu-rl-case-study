from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from envs.base import obs_hash
from envs.registry import list_envs, make_env

from replay.policies import list_policies, make_policy


@dataclass
class RunSpec:
    env_key: str
    env_id: str
    obs_type: str
    seed: int
    steps: int
    policy: str
    frameskip: int
    repeat_action_probability: float
    single_episode: bool


@dataclass
class RunArtifact:
    run_id: str
    created_unix_s: float
    spec: RunSpec
    actions: List[int]
    total_reward: float
    final_obs_hash: str


def _make_run_dir(base: Path, run_id: str) -> Path:
    d = base / run_id
    d.mkdir(parents=True, exist_ok=False)
    return d


def parse_args() -> argparse.Namespace:
    env_keys = [e.key for e in list_envs()]
    p = argparse.ArgumentParser(description="Record a deterministic action trace + telemetry.")
    p.add_argument(
        "--env",
        dest="env_key",
        default="pacman",
        choices=env_keys,
        help="Environment key (registry).",
    )
    p.add_argument("--list-envs", action="store_true", help="List available env keys and exit.")
    p.add_argument(
        "--policy",
        type=str,
        default="sticky_dir",
        choices=list_policies(),
        help="Policy used to generate actions.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--sticky", type=float, default=0.0)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--single-episode",
        action="store_true",
        help="Stop recording after the first episode terminates.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_envs:
        for s in list_envs():
            print(f"{s.key:12s}  {s.env_id:20s}  obs_type={s.obs_type:6s}  {s.description}")
        return

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = _make_run_dir(runs_dir, run_id)

    env_spec, env = make_env(
        args.env_key,
        render_mode="human" if args.render else None,
        frameskip=args.frameskip,
        repeat_action_probability=args.sticky,
    )

    spec = RunSpec(
        env_key=env_spec.key,
        env_id=env_spec.env_id,
        obs_type=env_spec.obs_type,
        seed=args.seed,
        steps=args.steps,
        policy=args.policy,
        frameskip=args.frameskip,
        repeat_action_probability=args.sticky,
        single_episode=args.single_episode,
    )

    policy = make_policy(args.policy)
    policy.reset(
        seed=args.seed,
        action_meanings=getattr(
            env, "action_meanings", [str(i) for i in range(env.action_space_n)]
        ),
        action_space_n=env.action_space_n,
    )

    obs, _info = env.reset(seed=args.seed)

    actions: List[int] = []
    total_reward = 0.0
    episode_return = 0.0

    last_reward = 0.0
    last_done = False

    telemetry_path = run_dir / "telemetry.csv"
    with telemetry_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode_id",
                "episode_step",
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

        episode_id = 0
        episode_step = 0

        for step in range(args.steps):
            action = int(policy.act(step=step, obs=obs, reward=last_reward, done=last_done))
            actions.append(action)

            t0 = time.perf_counter()
            sr = env.step(action)
            t1 = time.perf_counter()

            obs = sr.obs
            total_reward += sr.reward
            episode_return += sr.reward

            done = sr.terminated or sr.truncated
            last_reward = sr.reward
            last_done = done

            w.writerow(
                [
                    episode_id,
                    episode_step,
                    step,
                    action,
                    f"{sr.reward:.6f}",
                    int(sr.terminated),
                    int(sr.truncated),
                    int(done),
                    f"{episode_return:.6f}",
                    obs_hash(obs),
                    f"{(t1 - t0) * 1000.0:.3f}",
                ]
            )

            episode_step += 1

            if done:
                if args.single_episode:
                    break

                new_seed = args.seed + step + 1
                obs, _info = env.reset(seed=new_seed)

                episode_id += 1
                episode_step = 0
                episode_return = 0.0
                last_reward = 0.0
                last_done = False

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
    print(f"  env: {spec.env_key} ({spec.env_id}) obs_type={spec.obs_type}")
    print(f"  policy: {spec.policy}")
    print(f"  single_episode: {spec.single_episode}")
    print(f"  steps_recorded: {len(actions)}")
    print(f"  total_reward: {total_reward:.6f}")
    print(f"  final_obs_hash: {final_hash}")


if __name__ == "__main__":
    main()
