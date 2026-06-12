from __future__ import annotations

import copy
from statistics import mean

import optuna

from training.config import ExperimentConfig
from training.trainer import Trainer


def run_trial(trial: optuna.Trial, base_config: ExperimentConfig) -> float:
    config = copy.deepcopy(base_config)

    if config.agent.name == "reinforce":
        config.agent.reinforce.learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        config.agent.reinforce.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        config.agent.reinforce.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [64, 128, 256]
        )

    config.wandb.enabled = False
    config.checkpoint.enabled = False
    config.env.render_mode = None
    config.evaluation.render_mode = None

    trainer = Trainer(config)
    metrics = trainer.run()

    if not metrics:
        return float("-inf")

    recent_metrics = metrics[-10:]
    rewards = [item.total_reward for item in recent_metrics]

    return mean(rewards)
