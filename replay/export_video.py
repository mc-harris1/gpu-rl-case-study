# replay/export_video.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import imageio.v3 as iio
from envs.ale import ALEEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a recorded replay to MP4 via rgb_array frames.")
    p.add_argument("--run", type=str, required=True, help="Path to runs/<id>/run.json")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mp4 path (default: runs/<id>/replay.mp4)",
    )
    p.add_argument("--fps", type=int, default=60, help="Video FPS")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames written",
    )
    p.add_argument(
        "--capture-every",
        type=int,
        default=1,
        help="Capture every Nth frame to reduce file size (default 1 = every frame).",
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

    def maybe_add_frame(step_idx: int) -> None:
        if step_idx % args.capture_every != 0:
            return
        frame = env.render_rgb()
        if frame is not None:
            frames.append(frame)

    # Capture initial frame
    maybe_add_frame(0)

    for step, action in enumerate(actions):
        sr = env.step(int(action))
        maybe_add_frame(step + 1)

        done = sr.terminated or sr.truncated
        if done:
            # Match record.py multi-episode reseed behavior (harmless if single-episode was used)
            new_seed = seed + step + 1
            obs, _info = env.reset(seed=new_seed)
            maybe_add_frame(step + 2)

        if args.max_frames is not None and len(frames) >= args.max_frames:
            break

    env.close()

    if not frames:
        raise RuntimeError("No frames captured. Ensure env supports render_mode='rgb_array'.")

    iio.imwrite(out_path, frames, fps=args.fps)
    print(f"Wrote video: {out_path}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
