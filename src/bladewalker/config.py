from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    env_id: str = "BipedalWalker-v3"
    total_timesteps: int = 50_000
    seed: int = 42
    model_dir: Path = Path("artifacts/models")
    video_dir: Path = Path("artifacts/videos")

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

