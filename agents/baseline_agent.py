from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from training.buffers import Transition


@dataclass(slots=True)
class RandomAgent:
    action_space: gym.spaces.Discrete
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action_space, gym.spaces.Discrete):
            msg = "RandomAgent currently supports only discrete action spaces."
            raise TypeError(msg)

        self.action_space = cast(gym.spaces.Discrete, self.action_space)
        self._rng = np.random.default_rng(self.seed)

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        del observation
        del explore
        return int(self._rng.integers(self.action_space.n))

    def observe(self, transition: Transition) -> None:
        del transition

    def update(self) -> dict[str, float]:
        return {}

    def reset(self) -> None:
        return

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path).with_suffix(".json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": "random",
            "seed": self.seed,
            "action_space_n": int(self.action_space.n),
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        checkpoint_path = Path(path).with_suffix(".json")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.seed = payload.get("seed")
        self._rng = np.random.default_rng(self.seed)
