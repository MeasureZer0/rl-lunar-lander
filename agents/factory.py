from __future__ import annotations

from typing import cast

import gymnasium as gym
import numpy as np

from agents.baseline_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from agents.protocol import AgentProtocol
from agents.reinforce_agent import ReinforceAgent
from training.config import ExperimentConfig


def build_agent(
    config: ExperimentConfig,
    observation_space: gym.Space[np.ndarray],
    action_space: gym.Space[int],
) -> AgentProtocol:
    if not isinstance(action_space, gym.spaces.Discrete):
        msg = "Only discrete action spaces are currently supported."
        raise TypeError(msg)

    if not isinstance(observation_space, gym.spaces.Box):
        msg = "Only box observation spaces are currently supported."
        raise TypeError(msg)

    discrete_action_space = cast(gym.spaces.Discrete, action_space)
    box_observation_space = cast(gym.spaces.Box, observation_space)
    observation_dim = int(np.prod(box_observation_space.shape))

    if config.agent.name == "random":
        return RandomAgent(
            action_space=discrete_action_space,
            seed=config.agent.seed,
        )

    if config.agent.name == "reinforce":
        return ReinforceAgent(
            action_space=discrete_action_space,
            observation_dim=observation_dim,
            seed=config.agent.seed,
            config=config.agent.reinforce,
        )

<<<<<<< HEAD
    if config.agent.name == "dqn":
        return DQNAgent(
            action_space=discrete_action_space,
            observation_dim=observation_dim,
            seed=config.agent.seed,
            config=config.agent.dqn,
        )

=======
>>>>>>> main
    msg = f"Unsupported agent '{config.agent.name}'."
    raise ValueError(msg)
