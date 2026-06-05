from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.protocol import AgentProtocol
from training.config import CheckpointConfig
from training.rollout import EpisodeMetrics


@dataclass(slots=True)
class CheckpointState:
    best_reward: float | None = None


def maybe_save_checkpoint(
    agent: AgentProtocol,
    checkpoint_config: CheckpointConfig,
    episode_metrics: EpisodeMetrics,
    state: CheckpointState,
) -> None:
    if not checkpoint_config.enabled:
        return

    checkpoint_dir = Path(checkpoint_config.directory)
    if episode_metrics.episode % checkpoint_config.frequency == 0:
        path = checkpoint_dir / f"episode_{episode_metrics.episode:05d}.pt"
        agent.save(path)

    if checkpoint_config.save_best and _is_best_reward(episode_metrics, state):
        path = checkpoint_dir / "best.pt"
        agent.save(path)
        state.best_reward = episode_metrics.total_reward


def _is_best_reward(
    episode_metrics: EpisodeMetrics,
    state: CheckpointState,
) -> bool:
    if state.best_reward is None:
        return True
    return episode_metrics.total_reward > state.best_reward
