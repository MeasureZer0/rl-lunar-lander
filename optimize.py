from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import wandb
from training import load_experiment_config
from training.optuna_search import (
    log_architecture_plots,
    log_optuna_summary,
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
    shaping_group = parser.add_mutually_exclusive_group()
    shaping_group.add_argument(
        "--reward-shaping",
        action="store_true",
        dest="reward_shaping",
        help="Enable reward shaping during optimization.",
    )
    shaping_group.add_argument(
        "--no-reward-shaping",
        action="store_false",
        dest="reward_shaping",
        help="Disable reward shaping during optimization.",
    )
    parser.set_defaults(reward_shaping=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_config = load_experiment_config(args.config)
        if args.reward_shaping is not None:
            base_config.reward_shaping.enabled = args.reward_shaping
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    study_name = (
        base_config.optimize.study_name or f"optuna-study-{base_config.agent.name}"
    )
    n_trials = base_config.optimize.n_trials
    direction = base_config.optimize.direction
    storage = base_config.optimize.storage

    # Create storage directory if needed
    if storage and storage.startswith("sqlite:///"):
        db_path = storage.replace("sqlite:///", "")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=storage is not None,
    )

    print(f"Starting Optimization: {n_trials} trials ({direction})...")
    if storage is not None:
        print(f"Study persisted to: {storage}")

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
        log_optuna_summary(study)
        wandb.finish()
    else:
        log_optuna_summary(study)

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
