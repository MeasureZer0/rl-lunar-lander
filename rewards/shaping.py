"""
w_align - reward being directly above the pad (exp decay based on horizontal distance from center)
w_tilt - penalise excessive tilt before touchdown (above 15 degrees/ 0.26 radians)
w_soft - reward near zero vertical speed close to ground (active when y < 0.5, exp decay over vy)
w_hover - penalise idling far from the pad after episode start (penalty = |x| + |vx|, active when y > 0.5)
w_leg_sym - reward both legs making contact at the same time (+0.1 per step)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_align": 0.3,
    "w_tilt": 0.2,
    "w_soft": 0.5,
    "w_hover": 0.05,
    "w_leg_sym": 0.1,
}

def shape(
    obs: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    action: int,
    weights: dict[str, float] | None = None,
) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    x, y, vx, vy, angle, ang_vel, leg_l, leg_r = obs

    shaping = 0.0

    align_bonus = np.exp(-8.0 * x**2)
    shaping += w["w_align"] * align_bonus

    tilt_penalty = np.clip(abs(angle) - 0.26, 0.0, None)
    shaping -= w["w_tilt"] * tilt_penalty

    if y < 0.5:
        descent_speed = max(0.0, -vy)
        soft_bonus = np.exp(-4.0 * descent_speed)
        shaping += w["w_soft"] * soft_bonus

    if y > 0.5:
        drift = abs(x) + abs(vx)
        shaping -= w["w_hover"] * drift

    if leg_l and leg_r:
        shaping += w["w_leg_sym"]

    return reward + shaping

class ShapedLunarLander(gym.Wrapper):
    def __init__(self, env: gym.Env, weights: dict[str, float] | None = None) -> None:
        super().__init__(env)
        self.weights = weights
        self._last_action = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = shape(obs=obs, reward=float(reward), terminated=terminated,
                              truncated=truncated, action=action, weights=self.weights)
        info["original_reward"] = reward
        return obs, shaped_reward, terminated, truncated, info
