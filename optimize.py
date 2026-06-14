from __future__ import annotations

import argparse
import sys

import optuna
import wandb
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
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        default=False,
        help="Enable reward shaping during optimization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_config = load_experiment_config(args.config)
        base_config.reward_shaping.enabled = args.reward_shaping
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    study_name = (
        base_config.optimize.study_name or f"optuna-study-{base_config.agent.name}"
    )
    n_trials = base_config.optimize.n_trials
    direction = base_config.optimize.direction

    if base_config.wandb.enabled:
        wandb.init(
            project=base_config.wandb.project,
            entity=base_config.wandb.entity
            if hasattr(base_config.wandb, "entity")
            else None,
            name=study_name,
            group="hpo",
            tags=["optuna"] + base_config.wandb.tags,
            config={"base_config": args.config, "n_trials": n_trials},
        )

    callbacks = []

    if base_config.wandb.enabled:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="avg_reward",
            wandb_kwargs={"reinit": False},
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
        if base_config.wandb.enabled:
            wandb.finish()
        return 1

    if base_config.wandb.enabled and wandb.run is not None:
        wandb.log(
            {
                "best_value": study.best_value,
                **{f"best_param/{k}": v for k, v in study.best_params.items()},
            }
        )
        wandb.finish()

    print("\noptimization_complete=true")
    print(f"trials_completed={len(study.trials)}")
    print(f"best_value={study.best_value:.2f}")
    print("best_params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
