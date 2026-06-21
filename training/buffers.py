from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, rng: np.random.Generator, batch_size: int) -> list[Transition]:
        """Uniformly sample `batch_size` transitions without replacement.

        This is the public entry point agents should use instead of
        reaching into `_items` directly (which previously broke
        encapsulation in DQNAgent/DoubleDQNAgent).
        """
        if batch_size > len(self._items):
            msg = (
                f"Cannot sample batch_size={batch_size} from a buffer with "
                f"only {len(self._items)} items. Check that "
                "min_buffer_size >= batch_size in your agent config."
            )
            raise ValueError(msg)
        indices = rng.choice(len(self._items), size=batch_size, replace=False)
        items = list(self._items)
        return [items[i] for i in indices]


class TrajectoryBuffer:
    def __init__(self) -> None:
        self._items: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def get_items(self) -> list[Transition]:
        return self._items
