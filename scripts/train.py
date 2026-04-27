#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from bladewalker.config import TrainConfig
from bladewalker.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a test RL agent for BladeWalker stack")
    parser.add_argument("--env-id", default="BipedalWalker-v3", help="Gymnasium environment id")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=Path, default=None, help="Resume from an existing SB3 model zip")
    parser.add_argument("--save-name", default="latest", help="Output model filename without extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(env_id=args.env_id, total_timesteps=args.timesteps, seed=args.seed)
    config.ensure_dirs()

    env = make_env(config.env_id, render_mode=None, seed=config.seed)
    env = Monitor(env)

    if args.model_path:
        model = PPO.load(str(args.model_path), env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=config.seed,
            tensorboard_log="artifacts/tb",
        )

    model.learn(total_timesteps=config.total_timesteps, progress_bar=True)

    save_path = config.model_dir / f"{args.save_name}.zip"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()

