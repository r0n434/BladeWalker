from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics


def make_env(env_id: str, render_mode: str | None = None, seed: int | None = None) -> Any:
    env = gym.make(env_id, render_mode=render_mode)
    env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
