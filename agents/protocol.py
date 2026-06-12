from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from training.buffers import Transition


class AgentProtocol(Protocol):
    def act(self, observation: np.ndarray, *, explore: bool = True) -> int: ...

    def observe(self, transition: Transition) -> None: ...

    def update(self) -> dict[str, float]: ...

    def reset(self) -> None: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...
