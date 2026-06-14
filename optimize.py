from __future__ import annotations

import argparse
import sys

import optuna
import wandb
from training import load_experiment_config
from training.optuna_search import (
    log_architecture_plots,
    run_trial,
    save_best_checkpoint,
)


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

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
    )

    print(f"Starting Optimization: {n_trials} trials ({direction})...")

    try:
        study.optimize(
            lambda trial: run_trial(trial, base_config),
            n_trials=n_trials,
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Returning best results so far.")

    if not study.trials:
        print("No trials completed.")
        if base_config.wandb.enabled:
            wandb.finish()
        return 1

    if base_config.wandb.enabled:
        wandb.init(
            project=base_config.wandb.project,
            entity=base_config.wandb.entity,
            name=f"{study_name}-summary",
            group="hpo",
            tags=["optuna", "summary", base_config.optimize.mode],
            config={"base_config": args.config, "n_trials": n_trials},
        )
        wandb.log(
            {
                "best_value": study.best_value,
                **{f"best_param/{k}": v for k, v in study.best_params.items()},
            }
        )
        if base_config.optimize.mode == "architecture":
            log_architecture_plots(study)
        wandb.finish()

    checkpoint_filename = (
        base_config.optimize.baseline_v3_filename
        if base_config.optimize.mode == "architecture"
        else base_config.optimize.baseline_v2_filename
    )
    checkpoint_path = save_best_checkpoint(
        study,
        base_config,
        filename=checkpoint_filename,
    )

    print("\noptimization_complete=true")
    print(f"trials_completed={len(study.trials)}")
    print(f"best_value={study.best_value:.2f}")
    print(f"best_checkpoint={checkpoint_path}")
    print("best_params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
