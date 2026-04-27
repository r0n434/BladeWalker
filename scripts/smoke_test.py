#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bladewalker.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick environment smoke test")
    parser.add_argument("--env-id", default="BipedalWalker-v3", help="Gymnasium environment id")
    parser.add_argument("--steps", type=int, default=300, help="Max random policy steps")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(args.env_id, render_mode=None, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0

    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print(f"Smoke test OK on {args.env_id}. Reward sum (random): {total_reward:.2f}")


if __name__ == "__main__":
    main()

