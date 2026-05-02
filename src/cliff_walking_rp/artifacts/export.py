from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from cliff_walking_rp.artifacts.schema import RunMediaPaths, RunSummary
from cliff_walking_rp.training.common import TrainResult


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_series(path: Path, series: list[float] | list[int], title: str, xlabel: str, ylabel: str) -> None:
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(list(range(len(series))), series)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _maxq_heatmaps(qtable_snapshots: np.ndarray, xsize: int, ysize: int) -> np.ndarray:
    # (episodes, states, actions) -> (episodes, y, x)
    if qtable_snapshots.size == 0:
        return np.zeros((0, ysize, xsize), dtype=np.float32)

    episodes = qtable_snapshots.shape[0]
    out = np.zeros((episodes, ysize, xsize), dtype=np.float32)
    for ep in range(episodes):
        qt = qtable_snapshots[ep]
        maxq = np.max(qt, axis=1)  # (states,)
        for y in range(ysize):
            for x in range(xsize):
                s = x + y * xsize
                out[ep, y, x] = maxq[s]
    return out


def _save_heatmap_gif(path: Path, heatmaps: np.ndarray) -> None:
    # Uses PillowWriter (no ffmpeg dependency).
    if heatmaps.shape[0] == 0:
        # Create an empty placeholder figure
        fig = plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "No heatmaps", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path.with_suffix(".png"), dpi=160)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(heatmaps[0], cmap="viridis")
    ax.set_title("Max Q-value heatmap (per episode)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i: int):
        im.set_data(heatmaps[i])
        ax.set_xlabel(f"Episode {i}/{heatmaps.shape[0] - 1}")
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=heatmaps.shape[0], interval=120, blit=True)
    writer = animation.PillowWriter(fps=8)
    ani.save(path, writer=writer)
    plt.close(fig)

def _qvalues_frames(
    qtable_snapshots: np.ndarray, xsize: int, ysize: int, *, max_frames: int = 220
) -> tuple[list[int], np.ndarray]:
    """
    Convert snapshots (episode 0 baseline + episodes, states, actions) to (frames, y, x, actions),
    downsampling episodes to keep the JSON payload reasonable.
    """
    if qtable_snapshots.size == 0:
        return ([], np.zeros((0, ysize, xsize, 4), dtype=np.float32))

    total_frames = int(qtable_snapshots.shape[0])
    stride = max(1, int(np.ceil(total_frames / max_frames)))
    frame_eps = list(range(0, total_frames, stride))

    frames = []
    for ep_idx in range(0, total_frames, stride):
        qt = qtable_snapshots[ep_idx]  # (states, actions)
        grid = np.zeros((ysize, xsize, int(qt.shape[1])), dtype=np.float32)
        for y in range(ysize):
            for x in range(xsize):
                s = x + y * xsize
                grid[y, x, :] = qt[s, :]
        frames.append(grid)

    return (frame_eps, np.stack(frames, axis=0))


def _write_qvalues_json(path: Path, qtable_snapshots: np.ndarray, xsize: int, ysize: int) -> None:
    """
    Writes a compact JSON bundle used by the website to render a 4-wedge per-cell heatmap.
    """
    frame_eps, frames = _qvalues_frames(qtable_snapshots, xsize, ysize)
    if frames.shape[0] == 0:
        payload = {
            "schema_version": "1",
            "xsize": xsize,
            "ysize": ysize,
            "actions": ["right", "up", "left", "down"],
            "frame_episodes": [],
            "min": 0.0,
            "max": 0.0,
            "qvalues": [],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return

    vmin = float(np.min(frames))
    vmax = float(np.max(frames))

    # Round values to keep file smaller without affecting visuals too much.
    rounded = np.round(frames.astype(np.float32), 4)

    payload = {
        "schema_version": "1",
        "xsize": xsize,
        "ysize": ysize,
        "actions": ["right", "up", "left", "down"],
        "frame_episodes": frame_eps,  # 1-indexed episode numbers
        "min": vmin,
        "max": vmax,
        "qvalues": rounded.tolist(),  # [frame][y][x][a]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def export_run_bundle(
    *,
    run_dir: Path,
    variant: str,
    run_id: str,
    cfg: dict,
    result: TrainResult,
) -> Path:
    _ensure_dir(run_dir)
    plots_dir = run_dir / "plots"
    media_dir = run_dir / "media"
    _ensure_dir(plots_dir)
    _ensure_dir(media_dir)

    returns_png = plots_dir / "returns.png"
    steps_png = plots_dir / "steps.png"
    heatmaps_gif = media_dir / "maxq_heatmaps.gif"
    qvalues_json = media_dir / "qvalues_snapshots.json"

    _plot_series(
        returns_png,
        result.episode_returns,
        title="Episode returns",
        xlabel="Episode",
        ylabel="Return",
    )
    _plot_series(
        steps_png,
        result.steps_per_episode,
        title="Steps per episode",
        xlabel="Episode",
        ylabel="Steps",
    )

    heatmaps = _maxq_heatmaps(result.qtable_snapshots, result.xsize, result.ysize)
    _save_heatmap_gif(heatmaps_gif, heatmaps)
    _write_qvalues_json(qvalues_json, result.qtable_snapshots, result.xsize, result.ysize)

    # Keep trajectories as a separate file to keep summary.json small-ish.
    trajectories_path = run_dir / "trajectories.json"
    trajectories_path.write_text(json.dumps(result.agent_positions_per_episode), encoding="utf-8")

    summary = RunSummary(
        schema_version="1",
        variant=variant,
        run_id=run_id,
        xsize=result.xsize,
        ysize=result.ysize,
        cliff_positions=result.cliff_positions,
        hyperparameters=cfg,
        metrics={
            "episode_returns": result.episode_returns,
            "steps_per_episode": result.steps_per_episode,
            "trajectories_path": "trajectories.json",
        },
        media=RunMediaPaths(
            returns_png=str(returns_png.relative_to(run_dir)).replace("\\", "/"),
            steps_png=str(steps_png.relative_to(run_dir)).replace("\\", "/"),
            maxq_heatmaps_gif=str(heatmaps_gif.relative_to(run_dir)).replace("\\", "/"),
            qvalues_snapshots_json=str(qvalues_json.relative_to(run_dir)).replace("\\", "/"),
        ),
    )

    (run_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    # Also store numpy arrays for offline debugging (not required by the website).
    np.save(run_dir / "qtable.npy", result.qtable)
    np.save(run_dir / "qtable_snapshots.npy", result.qtable_snapshots)

    return run_dir
