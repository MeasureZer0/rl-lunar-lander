from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from models.policy_network import PolicyNetwork
from torch import nn
from training.buffers import TrajectoryBuffer, Transition
from training.config import PpoAgentConfig
from utils.checkpointing import load_checkpoint, save_checkpoint


@dataclass(slots=True)
class PpoAgent:
    action_space: gym.spaces.Discrete
    observation_dim: int
    config: PpoAgentConfig
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
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            weight_init=self.config.weight_init,
            normalization=self.config.normalization,
            normalization_position=self.config.normalization_position,
            dropout=self.config.dropout,
        )
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr=self.config.learning_rate,
        )

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)

        was_training = self._policy.training
        self._policy.eval()
        with torch.no_grad():
            logits = self._policy(observation_tensor)
        if was_training:
            self._policy.train()

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

    def _compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For now, we use Monte Carlo returns as advantages (no separate value function).
        """
        returns = self._compute_returns()

        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns, advantages

    def update(self) -> dict[str, float]:
        transitions = self._buffer.get_items()

        if not transitions:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "avg_return": 0.0,
                "trajectories_collected": 0.0,
            }

        observations = torch.as_tensor(
            np.stack([transition.observation for transition in transitions]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor(
            [transition.action for transition in transitions], dtype=torch.int64
        )

        returns, advantages = self._compute_advantages()
        avg_return = float(returns.mean().item())

        with torch.no_grad():
            logits_old = self._policy(observations)
            dist_old = torch.distributions.Categorical(logits=logits_old)
            log_probs_old = dist_old.log_prob(actions)

        clip_eps = self.config.clip_epsilon
        entropy_coef = self.config.entropy_coef
        ppo_epochs = self.config.ppo_epochs
        minibatch_size = self.config.minibatch_size

        num_samples = observations.shape[0]
        indices = np.arange(num_samples)

        total_loss_acc = 0.0
        total_policy_loss_acc = 0.0
        total_entropy_acc = 0.0
        total_updates = 0
        grad_norm: float | None = None

        for _ in range(ppo_epochs):
            self._rng.shuffle(indices)

            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                obs_mb = observations[mb_idx]
                actions_mb = actions[mb_idx]
                adv_mb = advantages[mb_idx]
                log_probs_old_mb = log_probs_old[mb_idx]

                logits = self._policy(obs_mb)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - log_probs_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - entropy_coef * entropy

                self._optimizer.zero_grad()
                loss.backward()
                grad_norm = self._clip_gradients()
                self._optimizer.step()

                total_loss_acc += float(loss.item())
                total_policy_loss_acc += float(policy_loss.item())
                total_entropy_acc += float(entropy.item())
                total_updates += 1

        transition_count = len(transitions)
        self._buffer.clear()

        avg_loss = total_loss_acc / max(total_updates, 1)
        avg_policy_loss = total_policy_loss_acc / max(total_updates, 1)
        avg_entropy = total_entropy_acc / max(total_updates, 1)

        metrics: dict[str, float] = {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "entropy": avg_entropy,
            "avg_return": avg_return,
            "trajectories_collected": float(transition_count),
        }
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        return metrics

    def reset(self) -> None:
        return

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._policy.parameters())

    def hidden_activation_snapshot(self, observations: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        was_training = self._policy.training
        self._policy.eval()
        with torch.no_grad():
            activations = self._policy.hidden_activations(obs_tensor)
        if was_training:
            self._policy.train()
        return activations.cpu().numpy()

    def _clip_gradients(self) -> float | None:
        max_norm = self.config.gradient_clip_max_norm
        if max_norm is None:
            return None
        norm = nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm)
        return float(norm.item())

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        state = {
            "epoch": 0,
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "agent_name": "ppo",
            "seed": self.seed,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "config": {
                "hidden_layers": self.config.hidden_layers,
                "activation": self.config.activation,
                "weight_init": self.config.weight_init,
                "normalization": self.config.normalization,
                "normalization_position": self.config.normalization_position,
                "dropout": self.config.dropout,
                "clip_epsilon": self.config.clip_epsilon,
                "entropy_coef": self.config.entropy_coef,
                "ppo_epochs": self.config.ppo_epochs,
                "minibatch_size": self.config.minibatch_size,
            },
        }
        save_checkpoint(
            state=state,
            checkpoint_dir=checkpoint_path.parent,
            config_name="ppo",
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
            "agent": "ppo",
            "seed": self.seed,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "clip_epsilon": self.config.clip_epsilon,
            "entropy_coef": self.config.entropy_coef,
            "ppo_epochs": self.config.ppo_epochs,
            "minibatch_size": self.config.minibatch_size,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
