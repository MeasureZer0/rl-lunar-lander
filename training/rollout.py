from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from agents.protocol import AgentProtocol

from training.buffers import Transition


@dataclass(slots=True)
class EpisodeMetrics:
    episode: int
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool


def collect_episode(
    env: gym.Env[np.ndarray, int],
    agent: AgentProtocol,
    *,
    episode: int,
    seed: int,
    max_steps: int,
) -> EpisodeMetrics:
    observation, _info = env.reset(seed=seed)
    agent.reset()

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    for step in range(1, max_steps + 1):
        action = agent.act(np.asarray(observation), explore=True)
        next_observation, reward, terminated, truncated, _info = env.step(action)
        transition = Transition(
            observation=np.asarray(observation),
            action=action,
            reward=float(reward),
            next_observation=np.asarray(next_observation),
            terminated=terminated,
            truncated=truncated,
        )
        agent.observe(transition)

        observation = next_observation
        total_reward += float(reward)
        steps = step

        if terminated or truncated:
            break

    return EpisodeMetrics(
        episode=episode,
        total_reward=total_reward,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
    )
