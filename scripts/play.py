#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from bladewalker.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained agent in human render mode")
    parser.add_argument("--env-id", default="BipedalWalker-v3", help="Gymnasium environment id")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/latest.zip"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = make_env(args.env_id, render_mode="human", seed=args.seed)
    model = PPO.load(str(args.model_path))

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)

        print(f"Episode {episode + 1}: reward={episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()

