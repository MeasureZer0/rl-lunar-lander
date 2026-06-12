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
