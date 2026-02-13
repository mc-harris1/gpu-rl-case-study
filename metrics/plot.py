from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot basic telemetry from a run.")
    p.add_argument("--run-dir", type=str, required=True, help="Path to runs/<id>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    csv_path = run_dir / "telemetry.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing telemetry.csv at {csv_path}")

    df = pd.read_csv(csv_path)
    # reward column is string if formatted; coerce
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)
    df["cum_reward"] = df["reward"].cumsum()

    plt.figure()
    plt.plot(df["step"], df["cum_reward"])
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.title(f"Cumulative reward vs step ({run_dir.name})")
    out_path = run_dir / "cum_reward.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
