from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from models.policy_network import PolicyNetwork
from training.buffers import ReplayBuffer, Transition
from training.config import DQNAgentConfig
from utils.checkpointing import load_checkpoint, save_checkpoint


@dataclass(slots=True)
class DQNAgent:
    action_space: gym.spaces.Discrete
    observation_dim: int
    config: DQNAgentConfig
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _buffer: ReplayBuffer = field(init=False, repr=False)
    _policy_net: PolicyNetwork = field(init=False, repr=False)
    _target_net: PolicyNetwork = field(init=False, repr=False)
    _optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    _epsilon: float = field(init=False, repr=False)
    _steps: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._buffer = ReplayBuffer(capacity=self.config.buffer_capacity)
        self._epsilon = self.config.epsilon_start
        self._steps = 0

        n_actions = int(self.action_space.n)
        self._policy_net = PolicyNetwork(
            input_dim=self.observation_dim,
            output_dim=n_actions,
            hidden_dim=self.config.hidden_dim,
        )
        self._target_net = PolicyNetwork(
            input_dim=self.observation_dim,
            output_dim=n_actions,
            hidden_dim=self.config.hidden_dim,
        )
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = torch.optim.Adam(
            self._policy_net.parameters(),
            lr=self.config.learning_rate,
        )

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        if explore and self._rng.random() < self._epsilon:
            return int(self._rng.integers(self.action_space.n))

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self._policy_net(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    def observe(self, transition: Transition) -> None:
        self._buffer.add(transition)

    def update(self) -> dict[str, float]:
        if len(self._buffer) < self.config.min_buffer_size:
            return {}

        batch = self._sample_batch()
        loss = self._compute_loss(batch)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._steps += 1
        self._epsilon = max(
            self.config.epsilon_end,
            self._epsilon * self.config.epsilon_decay,
        )

        if self._steps % self.config.target_update_frequency == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

        return {
            "loss": loss.item(),
            "epsilon": self._epsilon,
            "buffer_size": float(len(self._buffer)),
        }

    def reset(self) -> None:
        return

    def _sample_batch(self) -> list[Transition]:
        indices = self._rng.choice(
            len(self._buffer), size=self.config.batch_size, replace=False
        )
        items = list(self._buffer._items)
        return [items[i] for i in indices]

    def _compute_loss(self, batch: list[Transition]) -> torch.Tensor:
        obs = torch.tensor(
            np.array([t.observation for t in batch]), dtype=torch.float32
        )
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_obs = torch.tensor(
            np.array([t.next_observation for t in batch]), dtype=torch.float32
        )
        dones = torch.tensor(
            [t.terminated or t.truncated for t in batch], dtype=torch.float32
        )

        current_q = self._policy_net(obs).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self._target_net(next_obs).max(dim=1).values
            target_q = rewards + self.config.gamma * next_q * (1.0 - dones)

        return F.mse_loss(current_q, target_q)

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        state = {
            "epoch": self._steps,
            "model_state_dict": self._policy_net.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "agent_name": "dqn",
            "seed": self.seed,
            "epsilon": self._epsilon,
            "steps": self._steps,
        }
        save_checkpoint(
            state=state,
            checkpoint_dir=checkpoint_path.parent,
            config_name="dqn",
            filename=checkpoint_path.name,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(
            checkpoint_path=Path(path),
            model=self._policy_net,
            optimizer=self._optimizer,
        )
        self._target_net.load_state_dict(self._policy_net.state_dict())
        if checkpoint and "epsilon" in checkpoint:
            self._epsilon = checkpoint["epsilon"]
        if checkpoint and "steps" in checkpoint:
            self._steps = checkpoint["steps"]

    def export_metadata(self, path: str | Path) -> None:
        payload = {
            "agent": "dqn",
            "seed": self.seed,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "epsilon_end": self.config.epsilon_end,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
