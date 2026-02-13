# replay/replay.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from envs.ale import ALEEnv
from envs.base import obs_hash


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a recorded run and verify determinism.")
    p.add_argument("--run", type=str, required=True, help="Path to runs/<id>/run.json")
    p.add_argument("--render", action="store_true", help="Enable rendering (slower).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_path = Path(args.run)

    data = json.loads(run_path.read_text())
    spec = data["spec"]
    actions: List[int] = data["actions"]

    expected_total_reward = float(data["total_reward"])
    expected_final_hash = str(data["final_obs_hash"])

    env = ALEEnv(
        env_id=spec["env_id"],
        render_mode="human" if args.render else None,
        frameskip=int(spec.get("frameskip", 4)),
        repeat_action_probability=float(spec.get("repeat_action_probability", 0.0)),
    )

    seed = int(spec["seed"])
    obs, _info = env.reset(seed=seed)

    total_reward = 0.0
    last_hash = obs_hash(obs)

    # IMPORTANT: record.py resets mid-run when done=True, using:
    #   new_seed = seed + step + 1
    # We must mimic that behavior exactly for deterministic replay.
    for step, action in enumerate(actions):
        sr = env.step(int(action))
        obs = sr.obs
        total_reward += sr.reward
        last_hash = obs_hash(obs)

        done = sr.terminated or sr.truncated
        if done:
            new_seed = seed + step + 1
            obs, _info = env.reset(seed=new_seed)
            last_hash = obs_hash(obs)

    env.close()

    ok_reward = abs(total_reward - expected_total_reward) < 1e-6
    ok_hash = last_hash == expected_final_hash

    print("Replay complete")
    print(f"  env_key:              {spec.get('env_key', 'n/a')}")
    print(f"  env_id:               {spec['env_id']}")
    print(f"  obs_type:             {spec.get('obs_type', 'n/a')}")
    print(f"  policy:               {spec.get('policy', 'n/a')}")
    print(f"  steps:                {len(actions)}")
    print(f"  expected_total_reward: {expected_total_reward:.6f}")
    print(f"  actual_total_reward:   {total_reward:.6f}")
    print(f"  expected_final_hash:   {expected_final_hash}")
    print(f"  actual_final_hash:     {last_hash}")
    print(f"  deterministic_reward:  {ok_reward}")
    print(f"  deterministic_hash:    {ok_hash}")

    if not (ok_reward and ok_hash):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
