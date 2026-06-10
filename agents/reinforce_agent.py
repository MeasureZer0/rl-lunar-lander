from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from models.policy_network import PolicyNetwork
from training.buffers import TrajectoryBuffer, Transition
from training.config import ReinforceAgentConfig
from utils.checkpointing import load_checkpoint, save_checkpoint


@dataclass(slots=True)
class ReinforceAgent:
    action_space: gym.spaces.Discrete
    observation_dim: int
    config: ReinforceAgentConfig
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _buffer: TrajectoryBuffer = field(init=False, repr=False)
    _policy: PolicyNetwork = field(init=False, repr=False)
    _optimizer: torch.optim.Optimizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._buffer = TrajectoryBuffer()
        self._policy = PolicyNetwork(
            input_dim=self.observation_dim,
            output_dim=int(self.action_space.n),
            hidden_dim=self.config.hidden_dim,
        )
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=self.config.learning_rate,
        )

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:

        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self._policy(observation_tensor)

        distribution = torch.distributions.Categorical(logits=logits)

        if explore:
            action = distribution.sample()
        else:
            action = torch.argmax(logits, dim=-1)

        return int(action.item())

    def observe(self, transition: Transition) -> None:
        self._buffer.add(transition)

    def _compute_returns(self) -> torch.Tensor:
        transitions = self._buffer.get_items()

        returns = []
        running_return = 0.0

        for transition in reversed(transitions):
            if transition.terminated or transition.truncated:
                running_return = 0.0

            running_return = transition.reward + self.config.gamma * running_return
            returns.append(running_return)

        returns.reverse()

        return torch.tensor(returns, dtype=torch.float32)

    def update(self) -> dict[str, float]:
        transitions = self._buffer.get_items()

        if not transitions:
            return {"loss": 0.0, "avg_return": 0.0, "trajectories_collected": 0.0}

        observations = torch.as_tensor(
            np.stack([transition.observation for transition in transitions]),
            dtype=torch.float32,
        )

        actions = torch.as_tensor(
            [transition.action for transition in transitions], dtype=torch.int64
        )

        returns = self._compute_returns()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        logits = self._policy(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        loss = -(log_probs * returns).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        avg_return = float(returns.mean().item())

        transition_count = len(transitions)
        self._buffer.clear()
        return {
            "loss": float(loss.item()),
            "avg_return": avg_return,
            "trajectories_collected": float(transition_count),
        }

    def reset(self) -> None:
        return

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        state = {
            "epoch": 0,
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "agent_name": "reinforce",
            "seed": self.seed,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
        }
        save_checkpoint(
            state=state,
            checkpoint_dir=checkpoint_path.parent,
            config_name="reinforce",
            filename=checkpoint_path.name,
        )

    def load(self, path: str | Path) -> None:
        load_checkpoint(
            checkpoint_path=Path(path),
            model=self._policy,
            optimizer=self._optimizer,
        )

    def export_metadata(self, path: str | Path) -> None:
        payload = {
            "agent": "reinforce",
            "seed": self.seed,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
