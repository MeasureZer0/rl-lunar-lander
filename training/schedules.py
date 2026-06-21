from __future__ import annotations

import math


def compute_epsilon(
    *,
    schedule: str,
    progress_steps: int,
    decay_steps: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    current_epsilon: float,
) -> float:
    progress = min(1.0, progress_steps / max(1, decay_steps))

    if schedule == "linear":
        value = epsilon_start + progress * (epsilon_end - epsilon_start)
        return max(epsilon_end, value)

    if schedule == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return epsilon_end + (epsilon_start - epsilon_end) * cosine

    if schedule == "exponential":
        return max(epsilon_end, current_epsilon * epsilon_decay)

    msg = f"Unsupported epsilon schedule '{schedule}'."
    raise ValueError(msg)
