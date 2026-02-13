from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot telemetry with episode segmentation.")
    p.add_argument("--run-dir", type=str, required=True, help="Path to runs/<id>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    csv_path = run_dir / "telemetry.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing telemetry.csv at {csv_path}")

    df = pd.read_csv(csv_path)

    # Coerce numeric columns
    for col in ["reward", "wall_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # If older runs don't have episode_id yet, create a default
    if "episode_id" not in df.columns:
        df["episode_id"] = 0
    if "episode_step" not in df.columns:
        df["episode_step"] = df.groupby("episode_id").cumcount()

    df["cum_reward"] = df["reward"].cumsum()

    # Plot 1: global cumulative reward
    plt.figure()
    plt.plot(df["step"], df["cum_reward"])
    plt.xlabel("Global step")
    plt.ylabel("Cumulative reward")
    plt.title(f"Cumulative reward vs step ({run_dir.name})")
    out1 = run_dir / "cum_reward.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    print(f"Saved: {out1}")

    # Episode return = sum of rewards per episode
    ep_returns = df.groupby("episode_id")["reward"].sum().reset_index(name="episode_return")

    # Plot 2: episode returns
    plt.figure()
    plt.bar(ep_returns["episode_id"], ep_returns["episode_return"])
    plt.xlabel("Episode id")
    plt.ylabel("Episode return")
    plt.title(f"Episode returns ({run_dir.name})")
    out2 = run_dir / "episode_returns.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
