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

    elif config.agent.name == "dqn":
        config.agent.dqn.learning_rate = trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        )
        config.agent.dqn.gamma = trial.suggest_float("gamma", 0.90, 0.999)
        config.agent.dqn.hidden_dim = trial.suggest_categorical(
            "hidden_dim", [64, 128, 256]
        )
        config.agent.dqn.batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        )
        config.agent.dqn.epsilon_decay = trial.suggest_float(
            "epsilon_decay", 0.990, 0.9999
        )
        config.agent.dqn.target_update_frequency = trial.suggest_categorical(
            "target_update_frequency", [50, 100, 200]
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
