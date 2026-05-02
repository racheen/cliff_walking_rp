from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import gymnasium_env  # registers envs

from cliff_walking_rp.artifacts.export import export_run_bundle
from cliff_walking_rp.training.common import q_learning_train
from cliff_walking_rp.training.types import TrainConfig


def train_positive_only(cfg: TrainConfig) -> Path:
    run_id = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H%M%SZ") + f"_seed{cfg.seed}"
    run_dir = cfg.out_dir / "positive_only" / run_id

    render_mode = "human" if cfg.render == "human" else None
    env = gymnasium.make(
        "gymnasium_env/CliffWalkerPositive",
        render_mode=render_mode,
        size=(cfg.width, cfg.height),
        cliff_layout=cfg.cliff_layout,
    )

    result = q_learning_train(
        env=env,
        episodes=cfg.episodes,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        epsilon_decay=cfg.epsilon_decay,
        alpha=cfg.alpha,
        seed=cfg.seed,
        render=cfg.render == "human",
        render_sleep=cfg.render_sleep,
    )
    env.close()

    cfg_dict = asdict(cfg)
    cfg_dict["out_dir"] = str(cfg.out_dir)

    return export_run_bundle(
        run_dir=run_dir,
        variant="positive_only",
        run_id=run_id,
        cfg=cfg_dict | {"run_id": run_id},
        result=result,
    )
