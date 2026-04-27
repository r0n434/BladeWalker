#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from bladewalker.config import TrainConfig
from bladewalker.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MP4 rollouts from a trained model")
    parser.add_argument("--env-id", default="BipedalWalker-v3", help="Gymnasium environment id")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/latest.zip"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-dir", type=Path, default=Path("artifacts/videos"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(video_dir=args.video_dir)
    cfg.ensure_dirs()

    env = make_env(args.env_id, render_mode="rgb_array", seed=args.seed)
    env = RecordVideo(
        env,
        video_folder=str(cfg.video_dir),
        name_prefix="bladewalker-test",
        episode_trigger=lambda episode_id: episode_id < args.episodes,
    )

    model = PPO.load(str(args.model_path))

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

    env.close()
    print(f"Video(s) exported in {cfg.video_dir}")


if __name__ == "__main__":
    main()

