from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import imageio.v3 as iio
from envs.ale import ALEEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export replay to MP4 by capturing rgb_array frames.")
    p.add_argument("--run", type=str, required=True, help="Path to runs/<id>/run.json")
    p.add_argument(
        "--out", type=str, default=None, help="Output mp4 path (default: runs/<id>/replay.mp4)"
    )
    p.add_argument("--fps", type=int, default=60, help="Video FPS")
    p.add_argument(
        "--max-frames", type=int, default=None, help="Stop after this many frames (optional)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_path = Path(args.run)
    run_dir = run_path.parent
    out_path = Path(args.out) if args.out else (run_dir / "replay.mp4")

    data = json.loads(run_path.read_text())
    spec = data["spec"]
    actions: List[int] = data["actions"]

    env = ALEEnv(
        env_id=spec["env_id"],
        render_mode="rgb_array",
        frameskip=int(spec.get("frameskip", 4)),
        repeat_action_probability=float(spec.get("repeat_action_probability", 0.0)),
    )

    seed = int(spec["seed"])
    obs, _info = env.reset(seed=seed)

    frames = []
    # Capture initial frame
    frame = env._env.render()  # gymnasium returns rgb array in this mode
    if frame is not None:
        frames.append(frame)

    for step, action in enumerate(actions):
        sr = env.step(int(action))

        frame = env._env.render()
        if frame is not None:
            frames.append(frame)

        done = sr.terminated or sr.truncated
        if done:
            # mimic record.py reseed behavior (only relevant if run recorded multi-episode)
            new_seed = seed + step + 1
            obs, _info = env.reset(seed=new_seed)
            frame = env._env.render()
            if frame is not None:
                frames.append(frame)

        if args.max_frames is not None and len(frames) >= args.max_frames:
            break

    env.close()

    # Write MP4
    iio.imwrite(out_path, frames, fps=args.fps)
    print(f"Wrote video: {out_path}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
