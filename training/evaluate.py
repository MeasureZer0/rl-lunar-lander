from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import gymnasium as gym
import numpy as np
import wandb

from agents.protocol import AgentProtocol
from training.config import EnvConfig, EvaluationConfig, TrainingConfig
from training.rollout import EpisodeMetrics


@dataclass(slots=True)
class EvaluationSummary:
    avg_reward: float
    avg_steps: float
    episodes: int


def evaluate_agent(
    agent: AgentProtocol,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    evaluation_config: EvaluationConfig,
    *,
    seed_offset: int,
) -> EvaluationSummary:
    rewards: list[float] = []
    steps: list[int] = []
    env = gym.make(
        env_config.id,
        render_mode=evaluation_config.render_mode,
    )

    try:
        for episode in range(evaluation_config.episodes):
            metrics = _run_eval_episode(
                env=env,
                agent=agent,
                max_steps=training_config.max_steps_per_episode,
                seed=training_config.seed + seed_offset + episode,
            )
            rewards.append(metrics.total_reward)
            steps.append(metrics.steps)
    finally:
        env.close()

    return EvaluationSummary(
        avg_reward=mean(rewards),
        avg_steps=mean(steps),
        episodes=evaluation_config.episodes,
    )


def log_training_episode(
    metrics: EpisodeMetrics,
    update_metrics: dict[str, float],
) -> None:
    suffix = _format_metrics(update_metrics)
    print(
        "episode="
        f"{metrics.episode} reward={metrics.total_reward:.2f} "
        f"steps={metrics.steps} terminated={metrics.terminated} "
        f"truncated={metrics.truncated}{suffix}"
    )
    if wandb.run is not None:
        log_data = {
            "train/reward": metrics.total_reward,
            "train/steps": metrics.steps,
            **{f"train/{k}": v for k, v in update_metrics.items()},
        }
        wandb.log(log_data, step=metrics.episode)


def log_evaluation_summary(
    episode: int,
    summary: EvaluationSummary,
) -> None:
    print(
        "evaluation="
        f"{episode} avg_reward={summary.avg_reward:.2f} "
        f"avg_steps={summary.avg_steps:.2f} episodes={summary.episodes}"
    )
    if wandb.run is not None:
        wandb.log(
            {
                "eval/avg_reward": summary.avg_reward,
                "eval/avg_steps": summary.avg_steps,
            },
            step=episode,
        )


def _run_eval_episode(
    env: gym.Env[np.ndarray, int],
    agent: AgentProtocol,
    *,
    max_steps: int,
    seed: int,
) -> EpisodeMetrics:
    observation, _info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    for step in range(1, max_steps + 1):
        action = agent.act(np.asarray(observation), explore=False)
        observation, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        steps = step
        if terminated or truncated:
            break

    return EpisodeMetrics(
        episode=0,
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return ""
    parts = [f"{key}={value:.4f}" for key, value in sorted(metrics.items())]
    return " " + " ".join(parts)
