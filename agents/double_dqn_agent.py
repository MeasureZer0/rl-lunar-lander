from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from models.policy_network import PolicyNetwork
from torch import nn
from training.buffers import ReplayBuffer, Transition
from training.config import DoubleDQNAgentConfig
from utils.checkpointing import load_checkpoint, save_checkpoint


@dataclass(slots=True)
class DoubleDQNAgent:
    action_space: gym.spaces.Discrete
    observation_dim: int
    config: DoubleDQNAgentConfig
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _buffer: ReplayBuffer = field(init=False, repr=False)
    _policy_net: PolicyNetwork = field(init=False, repr=False)
    _target_net: PolicyNetwork = field(init=False, repr=False)
    _optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    _scheduler: (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ) = field(
        init=False,
        repr=False,
    )
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
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            weight_init=self.config.weight_init,
            normalization=self.config.normalization,
            normalization_position=self.config.normalization_position,
            dropout=self.config.dropout,
        )
        self._target_net = PolicyNetwork(
            input_dim=self.observation_dim,
            output_dim=n_actions,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            weight_init=self.config.weight_init,
            normalization=self.config.normalization,
            normalization_position=self.config.normalization_position,
            dropout=self.config.dropout,
        )
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = torch.optim.Adam(
            self._policy_net.parameters(),
            lr=self.config.learning_rate,
        )
        self._scheduler = self._build_scheduler()

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        if explore and self._rng.random() < self._epsilon:
            return int(self._rng.integers(self.action_space.n))

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        was_training = self._policy_net.training
        self._policy_net.eval()
        with torch.no_grad():
            q_values = self._policy_net(obs_tensor)
        if was_training:
            self._policy_net.train()
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
        grad_norm = self._clip_gradients()
        self._optimizer.step()
        self._step_scheduler(loss)

        self._steps += 1
        self._epsilon = self._compute_epsilon()

        if self.config.target_update_type == "soft":
            self._soft_update_target()
        elif self._steps % self.config.target_update_frequency == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

        return {
            "loss": loss.item(),
            "epsilon": self._epsilon,
            "buffer_size": float(len(self._buffer)),
            "learning_rate": self._optimizer.param_groups[0]["lr"],
            **({"grad_norm": grad_norm} if grad_norm is not None else {}),
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
        obs = torch.as_tensor(
            np.array([t.observation for t in batch]), dtype=torch.float32
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.long
        ).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        next_obs = torch.as_tensor(
            np.array([t.next_observation for t in batch]), dtype=torch.float32
        )
        dones = torch.as_tensor(
            [t.terminated or t.truncated for t in batch], dtype=torch.float32
        )

        current_q = self._policy_net(obs).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self._policy_net(next_obs).argmax(dim=1, keepdim=True)
            next_q_target = (
                self._target_net(next_obs).gather(1, next_actions).squeeze(1)
            )
            target_q = rewards + self.config.gamma * next_q_target * (1.0 - dones)

        return F.mse_loss(current_q, target_q)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._policy_net.parameters())

    def hidden_activation_snapshot(self, observations: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        was_training = self._policy_net.training
        self._policy_net.eval()
        with torch.no_grad():
            activations = self._policy_net.hidden_activations(obs_tensor)
        if was_training:
            self._policy_net.train()
        return activations.cpu().numpy()

    def _build_scheduler(
        self,
    ) -> (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | None
    ):
        if self.config.lr_scheduler is None:
            return None
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=self.config.step_lr_step_size,
                gamma=self.config.step_lr_gamma,
            )
        if self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._optimizer,
                patience=self.config.reduce_lr_patience,
            )
        msg = f"Unsupported LR scheduler '{self.config.lr_scheduler}'."
        raise ValueError(msg)

    def _step_scheduler(self, loss: torch.Tensor) -> None:
        if self._scheduler is None:
            return
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self._scheduler.step(float(loss.item()))
        else:
            self._scheduler.step()

    def _clip_gradients(self) -> float | None:
        max_norm = self.config.gradient_clip_max_norm
        if max_norm is None:
            return None
        norm = nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_norm)
        return float(norm.item())

    def _compute_epsilon(self) -> float:
        progress = min(1.0, self._steps / max(1, self.config.epsilon_decay_episodes))
        if self.config.epsilon_schedule == "linear":
            value = self.config.epsilon_start + progress * (
                self.config.epsilon_end - self.config.epsilon_start
            )
            return max(self.config.epsilon_end, value)
        if self.config.epsilon_schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (
                self.config.epsilon_end
                + (self.config.epsilon_start - self.config.epsilon_end) * cosine
            )
        if self.config.epsilon_schedule == "exponential":
            return max(
                self.config.epsilon_end,
                self._epsilon * self.config.epsilon_decay,
            )
        msg = f"Unsupported epsilon schedule '{self.config.epsilon_schedule}'."
        raise ValueError(msg)

    def _soft_update_target(self) -> None:
        tau = self.config.tau
        for target_parameter, policy_parameter in zip(
            self._target_net.parameters(),
            self._policy_net.parameters(),
            strict=True,
        ):
            target_parameter.data.copy_(
                tau * policy_parameter.data + (1.0 - tau) * target_parameter.data
            )

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
            "config": {
                "hidden_layers": self.config.hidden_layers,
                "activation": self.config.activation,
                "weight_init": self.config.weight_init,
                "normalization": self.config.normalization,
                "normalization_position": self.config.normalization_position,
                "dropout": self.config.dropout,
            },
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
