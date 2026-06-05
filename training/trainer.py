from __future__ import annotations

import gymnasium as gym

from agents import AgentProtocol, build_agent
from training.checkpoint import CheckpointState, maybe_save_checkpoint
from training.config import ExperimentConfig
from training.evaluate import (
    evaluate_agent,
    log_evaluation_summary,
    log_training_episode,
)
from training.rollout import EpisodeMetrics, collect_episode


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> list[EpisodeMetrics]:
        env = gym.make(
            self.config.env.id,
            render_mode=self.config.env.render_mode,
        )

        try:
            agent = build_agent(
                config=self.config,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            metrics_history: list[EpisodeMetrics] = []
            checkpoint_state = CheckpointState()

            for episode in range(1, self.config.training.episodes + 1):
                episode_metrics = collect_episode(
                    env=env,
                    agent=agent,
                    episode=episode,
                    seed=self.config.training.seed + episode - 1,
                    max_steps=self.config.training.max_steps_per_episode,
                )
                metrics_history.append(episode_metrics)

                update_metrics = self._run_update_loop(agent, episode)
                if episode % self.config.training.log_every == 0 or episode == 1:
                    log_training_episode(episode_metrics, update_metrics)

                self._run_evaluation(agent, episode)
                maybe_save_checkpoint(
                    agent=agent,
                    checkpoint_config=self.config.checkpoint,
                    episode_metrics=episode_metrics,
                    state=checkpoint_state,
                )

            return metrics_history
        finally:
            env.close()

    def _run_update_loop(
        self,
        agent: AgentProtocol,
        episode: int,
    ) -> dict[str, float]:
        if episode % self.config.training.update_every != 0:
            return {}
        return agent.update()

    def _run_evaluation(self, agent: AgentProtocol, episode: int) -> None:
        if not self.config.evaluation.enabled:
            return
        if episode % self.config.evaluation.frequency != 0:
            return

        summary = evaluate_agent(
            agent=agent,
            env_config=self.config.env,
            training_config=self.config.training,
            evaluation_config=self.config.evaluation,
            seed_offset=10_000 + episode,
        )
        log_evaluation_summary(episode, summary)
