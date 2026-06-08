from __future__ import annotations

import argparse
import sys

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback  # type: ignore[no-redef]

from training import load_experiment_config
from training.optuna_search import run_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize RL agents using Optuna.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the base YAML experiment config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_config = load_experiment_config(args.config)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    study_name = (
        base_config.optimize.study_name or f"optuna-study-{base_config.agent.name}"
    )
    n_trials = base_config.optimize.n_trials
    direction = base_config.optimize.direction

    callbacks = []

    # Configure unified W&B logging for the whole Optuna study
    if base_config.wandb.enabled:
        wandb_kwargs = {
            "project": base_config.wandb.project,
            "name": study_name,
            "tags": ["optuna"] + base_config.wandb.tags,
            "config": {"base_config": args.config},
        }
        if base_config.wandb.entity:
            wandb_kwargs["entity"] = base_config.wandb.entity

        wandb_callback = WeightsAndBiasesCallback(
            metric_name="avg_reward",
            wandb_kwargs=wandb_kwargs,
        )
        callbacks.append(wandb_callback)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
    )

    print(f"Starting Optimization: {n_trials} trials ({direction})...")

    try:
        study.optimize(
            lambda trial: run_trial(trial, base_config),
            n_trials=n_trials,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Returning best results so far.")

    if not study.trials:
        print("No trials completed.")
        return 1

    print("\noptimization_complete=true")
    print(f"trials_completed={len(study.trials)}")
    print(f"best_value={study.best_value:.2f}")
    print("best_params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
