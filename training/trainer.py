from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np

from agents import RandomAgent
from training.config import ExperimentConfig


@dataclass(slots=True)
class EpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool


class Trainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> list[EpisodeMetrics]:
        env = gym.make(
            self.config.env.id,
            render_mode=self.config.env.render_mode,
        )

        try:
            agent = self._build_agent(env)
            metrics: list[EpisodeMetrics] = []

            for episode in range(1, self.config.training.episodes + 1):
                episode_seed = self.config.training.seed + episode - 1
                observation, _info = env.reset(seed=episode_seed)
                agent.reset()

                total_reward = 0.0
                steps = 0
                terminated = False
                truncated = False

                for step in range(1, self.config.training.max_steps_per_episode + 1):
                    action = agent.act(np.asarray(observation))
                    observation, reward, terminated, truncated, _info = env.step(action)
                    total_reward += float(reward)
                    steps = step

                    if terminated or truncated:
                        break

                episode_metrics = EpisodeMetrics(
                    episode=episode,
                    total_reward=total_reward,
                    steps=steps,
                    terminated=terminated,
                    truncated=truncated,
                )
                metrics.append(episode_metrics)

                if episode % self.config.training.log_every == 0 or episode == 1:
                    print(
                        "episode="
                        f"{episode_metrics.episode} reward={episode_metrics.total_reward:.2f} "
                        f"steps={episode_metrics.steps} terminated={episode_metrics.terminated} "
                        f"truncated={episode_metrics.truncated}"
                    )

            return metrics
        finally:
            env.close()

    def _build_agent(self, env: gym.Env[np.ndarray, int]) -> RandomAgent:
        if self.config.agent.name != "random":
            msg = f"Unsupported agent '{self.config.agent.name}'."
            raise ValueError(msg)

        return RandomAgent(
            action_space=cast(gym.spaces.Discrete, env.action_space),
            seed=self.config.agent.seed,
        )
