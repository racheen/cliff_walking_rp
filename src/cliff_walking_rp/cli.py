from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from cliff_walking_rp.training.positive_only import train_positive_only
from cliff_walking_rp.training.traditional import train_traditional
from cliff_walking_rp.training.types import TrainConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cliff-walking-rp")
    sub = p.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train an agent and export a run bundle")
    train.add_argument(
        "--variant",
        choices=["positive_only", "traditional"],
        required=True,
        help="Which reward strategy/environment to train.",
    )
    train.add_argument("--episodes", type=int, default=100)
    train.add_argument("--gamma", type=float, default=0.1)
    train.add_argument("--epsilon", type=float, default=0.08)
    train.add_argument("--epsilon-decay", type=float, default=0.01)
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument(
        "--render",
        choices=["none", "human"],
        default="none",
        help="Render mode (slower). Use 'none' for export runs.",
    )
    train.add_argument(
        "--render-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between rendered steps (human render only).",
    )
    train.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Base output directory (run bundle will be created inside).",
    )

    showcfg = sub.add_parser("print-config", help="Print resolved training config JSON")
    showcfg.add_argument(
        "--variant",
        choices=["positive_only", "traditional"],
        required=True,
    )
    showcfg.add_argument("--episodes", type=int, default=100)
    showcfg.add_argument("--gamma", type=float, default=0.1)
    showcfg.add_argument("--epsilon", type=float, default=0.08)
    showcfg.add_argument("--epsilon-decay", type=float, default=0.01)
    showcfg.add_argument("--alpha", type=float, default=1.0)
    showcfg.add_argument("--seed", type=int, default=123)
    showcfg.add_argument("--render", choices=["none", "human"], default="none")
    showcfg.add_argument("--render-sleep", type=float, default=0.0)
    showcfg.add_argument("--out", type=str, default="outputs")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        variant=args.variant,
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        alpha=args.alpha,
        seed=args.seed,
        render=args.render,
        render_sleep=args.render_sleep,
        out_dir=Path(args.out),
    )

    if args.cmd == "print-config":
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return

    if args.cmd == "train":
        # Make deterministic-ish behavior for the run bundle creation (paths, etc.)
        os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))

        if cfg.variant == "positive_only":
            run_dir = train_positive_only(cfg)
        else:
            run_dir = train_traditional(cfg)

        print(str(run_dir))
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")

