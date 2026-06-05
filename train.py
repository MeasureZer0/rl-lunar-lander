from __future__ import annotations

import argparse
import sys
from statistics import mean

from training import Trainer, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL agents for LunarLander.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML experiment config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_experiment_config(args.config)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    trainer = Trainer(config)
    metrics = trainer.run()

    rewards = [item.total_reward for item in metrics]
    steps = [item.steps for item in metrics]

    print()
    print("training_complete=true")
    print(f"episodes={len(metrics)}")
    print(f"avg_reward={mean(rewards):.2f}")
    print(f"best_reward={max(rewards):.2f}")
    print(f"avg_steps={mean(steps):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
