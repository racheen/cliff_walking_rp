from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from cliff_walking_rp.training.positive_only import train_positive_only
from cliff_walking_rp.training.traditional import train_traditional
from cliff_walking_rp.training.types import TrainConfig
from gymnasium_env.envs.layouts import cliff_positions_for_layout

TRAP_LAYOUT_CHOICES = ["bottom", "gap", "middle", "double", "scattered", "maze", "islands", "mixed"]


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
    train.add_argument("--width", type=int, default=12, help="Grid width.")
    train.add_argument("--height", type=int, default=4, help="Grid height.")
    train.add_argument(
        "--trap-layout",
        choices=TRAP_LAYOUT_CHOICES,
        default="scattered",
        dest="cliff_layout",
        help="Trap layout to train on.",
    )
    train.add_argument(
        "--cliff-layout",
        choices=TRAP_LAYOUT_CHOICES,
        dest="cliff_layout",
        help="Deprecated alias for --trap-layout.",
    )

    train_site = sub.add_parser(
        "train-site",
        help="Train both variants, copy their bundles into the static site, and update the run index.",
    )
    train_site.add_argument("--episodes", type=int, default=100)
    train_site.add_argument("--gamma", type=float, default=0.1)
    train_site.add_argument("--epsilon", type=float, default=0.08)
    train_site.add_argument("--epsilon-decay", type=float, default=0.01)
    train_site.add_argument("--alpha", type=float, default=1.0)
    train_site.add_argument("--seed", type=int, default=123)
    train_site.add_argument(
        "--render",
        choices=["none", "human"],
        default="none",
        help="Render mode (slower). Use 'none' for export runs.",
    )
    train_site.add_argument(
        "--render-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between rendered steps (human render only).",
    )
    train_site.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Base output directory (run bundles will be created inside).",
    )
    train_site.add_argument(
        "--site-runs",
        type=str,
        default="site/public/runs",
        help="Static site runs directory to sync into.",
    )
    train_site.add_argument("--width", type=int, default=12, help="Grid width.")
    train_site.add_argument("--height", type=int, default=4, help="Grid height.")
    train_site.add_argument(
        "--trap-layout",
        choices=TRAP_LAYOUT_CHOICES,
        default="scattered",
        dest="cliff_layout",
        help="Trap layout to train on.",
    )
    train_site.add_argument(
        "--cliff-layout",
        choices=TRAP_LAYOUT_CHOICES,
        dest="cliff_layout",
        help="Deprecated alias for --trap-layout.",
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
    showcfg.add_argument("--width", type=int, default=12)
    showcfg.add_argument("--height", type=int, default=4)
    showcfg.add_argument(
        "--trap-layout",
        choices=TRAP_LAYOUT_CHOICES,
        default="scattered",
        dest="cliff_layout",
    )
    showcfg.add_argument(
        "--cliff-layout",
        choices=TRAP_LAYOUT_CHOICES,
        dest="cliff_layout",
    )

    return p


def _cfg_from_args(args: argparse.Namespace, variant: str) -> TrainConfig:
    return TrainConfig(
        variant=variant,
        episodes=args.episodes,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        alpha=args.alpha,
        seed=args.seed,
        render=args.render,
        render_sleep=args.render_sleep,
        out_dir=Path(args.out),
        width=args.width,
        height=args.height,
        cliff_layout=args.cliff_layout,
    )


def _format_grid_preview(width: int, height: int, trap_layout: str) -> str:
    traps = cliff_positions_for_layout(width, height, trap_layout)
    lines = [f"Grid preview: {width} x {height}, trap layout: {trap_layout}"]
    lines.append("Legend: S=start, G=goal, X=trap/mine, .=open")
    for y in range(height):
        row = []
        for x in range(width):
            if x == 0 and y == height - 1:
                row.append("S")
            elif x == width - 1 and y == height - 1:
                row.append("G")
            elif (x, y) in traps:
                row.append("X")
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)


def _print_grid_preview(cfg: TrainConfig) -> None:
    print(_format_grid_preview(cfg.width, cfg.height, cfg.cliff_layout))
    print()


def _load_site_index(site_runs_dir: Path) -> dict:
    index_path = site_runs_dir / "index.json"
    if not index_path.exists():
        return {
            "schemaVersion": "1",
            "generatedAt": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "runs": [],
        }

    return json.loads(index_path.read_text(encoding="utf-8"))


def _unique_site_run_id(site_runs_dir: Path, base_run_id: str, variant: str, reserved: set[str]) -> str:
    if base_run_id not in reserved and not (site_runs_dir / base_run_id).exists():
        return base_run_id

    candidate = f"{base_run_id}_{variant}"
    if candidate not in reserved and not (site_runs_dir / candidate).exists():
        return candidate

    i = 2
    while True:
        candidate = f"{base_run_id}_{variant}_{i}"
        if candidate not in reserved and not (site_runs_dir / candidate).exists():
            return candidate
        i += 1


def _copy_run_to_site(run_dir: Path, site_runs_dir: Path, variant: str, reserved: set[str]) -> str:
    site_runs_dir.mkdir(parents=True, exist_ok=True)
    site_run_id = _unique_site_run_id(site_runs_dir, run_dir.name, variant, reserved)
    dest_dir = site_runs_dir / site_run_id

    shutil.copytree(run_dir, dest_dir)

    if site_run_id != run_dir.name:
        summary_path = dest_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_id"] = site_run_id
        if isinstance(summary.get("hyperparameters"), dict):
            summary["hyperparameters"]["run_id"] = site_run_id
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return site_run_id


def _write_site_index(site_runs_dir: Path, new_entries: list[dict]) -> None:
    site_runs_dir.mkdir(parents=True, exist_ok=True)
    index = _load_site_index(site_runs_dir)
    existing_runs = index.get("runs", [])
    new_ids = {entry["runId"] for entry in new_entries}
    kept_runs = [entry for entry in existing_runs if entry.get("runId") not in new_ids]
    index["schemaVersion"] = "1"
    index["generatedAt"] = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    index["runs"] = new_entries + kept_runs
    (site_runs_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def _train_variant(cfg: TrainConfig) -> Path:
    if cfg.variant == "positive_only":
        return train_positive_only(cfg)
    if cfg.variant == "traditional":
        return train_traditional(cfg)
    raise RuntimeError(f"Unknown variant: {cfg.variant}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "print-config":
        cfg = _cfg_from_args(args, args.variant)
        print(json.dumps(asdict(cfg), indent=2, default=str))
        print()
        _print_grid_preview(cfg)
        return

    if args.cmd == "train":
        cfg = _cfg_from_args(args, args.variant)
        # Make deterministic-ish behavior for the run bundle creation (paths, etc.)
        os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))

        _print_grid_preview(cfg)
        run_dir = _train_variant(cfg)

        print(str(run_dir))
        return

    if args.cmd == "train-site":
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
        site_runs_dir = Path(args.site_runs)
        reserved = {
            entry.get("runId")
            for entry in _load_site_index(site_runs_dir).get("runs", [])
            if isinstance(entry.get("runId"), str)
        }
        synced_entries = []
        preview_cfg = _cfg_from_args(args, "positive_only")
        _print_grid_preview(preview_cfg)

        for variant in ("positive_only", "traditional"):
            cfg = _cfg_from_args(args, variant)
            run_dir = _train_variant(cfg)
            site_run_id = _copy_run_to_site(run_dir, site_runs_dir, variant, reserved)
            reserved.add(site_run_id)
            synced_entries.append(
                {
                    "runId": site_run_id,
                    "variant": variant,
                    "summaryPath": f"{site_run_id}/summary.json",
                }
            )
            print(f"{variant}: {run_dir} -> {site_runs_dir / site_run_id}")

        _write_site_index(site_runs_dir, synced_entries)
        print(f"Updated {site_runs_dir / 'index.json'}")
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")
