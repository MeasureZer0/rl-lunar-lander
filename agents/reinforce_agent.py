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
        del observation
        del explore
        return int(self._rng.integers(self.action_space.n))

    def observe(self, transition: Transition) -> None:
        self._buffer.add(transition)

    def update(self) -> dict[str, float]:
        trajectory_count = len(self._buffer)
        self._buffer.clear()
        return {
            "placeholder_loss": 0.0,
            "trajectories_collected": float(trajectory_count),
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
