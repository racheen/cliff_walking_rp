from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMediaPaths:
    returns_png: str
    steps_png: str
    maxq_heatmaps_gif: str


@dataclass(frozen=True)
class RunSummary:
    schema_version: str
    variant: str
    run_id: str

    xsize: int
    ysize: int

    hyperparameters: dict
    metrics: dict
    media: RunMediaPaths

