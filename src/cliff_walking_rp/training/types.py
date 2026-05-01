from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    variant: str
    episodes: int = 100
    gamma: float = 0.1
    epsilon: float = 0.08
    epsilon_decay: float = 0.01
    alpha: float = 1.0
    seed: int = 123
    render: str = "none"  # "none" | "human"
    render_sleep: float = 0.0
    out_dir: Path = Path("outputs")

