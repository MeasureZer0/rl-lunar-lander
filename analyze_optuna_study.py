from __future__ import annotations

import argparse

import optuna
from training.optuna_search import log_optuna_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline analysis of a persisted Optuna study: hyperparameter "
        "importance, contour/parallel-coordinate/slice plots, and a CSV export "
        "of all trials."
    )
    parser.add_argument("--study-name", required=True, help="Name of the Optuna study.")
    parser.add_argument(
        "--storage",
        required=True,
        help="Optuna storage URL, e.g. sqlite:///optuna_studies/lunar-lander-dqn-hyperparameters.db",
    )
    parser.add_argument(
        "--output-dir",
        default="visualisation",
        help="Directory for CSV/plots output (default: visualisation/).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    print(
        f"study={args.study_name} trials={len(study.trials)} best_value={study.best_value:.2f}"
    )
    print("best_params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    log_optuna_summary(study, output_dir=args.output_dir)

    print("\nanalysis_complete=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
