from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import gymnasium as gym
import numpy as np


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

    def act(self, _observation: np.ndarray) -> int:
        return int(self._rng.integers(self.action_space.n))

    def reset(self) -> None:
        return
