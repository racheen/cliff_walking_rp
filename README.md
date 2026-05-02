# Trap Grid Walking with Positive-Only Reinforcement

## Overview
This project explores whether a reinforcement learning agent can learn a safe path without punishment, using positive-only reinforcement instead of traditional penalties.

The project began with the classic Cliff Walking environment, then expanded into configurable grid worlds with traps/mines that can appear anywhere on the board. I compare two agents:

- A **traditional Q-learning agent** trained with step penalties and a harsh trap penalty

- A **positive-only agent** trained by rewarding progress toward the goal, with no negative rewards

The results show that an agent can successfully learn safe and effective behavior using encouragement alone—raising interesting parallels between **machine learning and modern animal training practices**.

## Key Questions
- Can an agent learn an optimal path without negative rewards?
- How does reward shaping affect learning stability and exploration?
- What can reinforcement learning teach us about humane learning systems?

## Core Concepts
- Algorithm: Q-learning (from scratch)
- Environment: Custom Gymnasium grid-world implementation
- Reward Strategy:
    - Traditional: step penalty + trap penalty
    - Positive-only: reward for moving closer to the goal, no punishment for mistakes
- Comparison: Learning dynamics, convergence, and learned policies

## Environment Summary
- Grid Size: configurable with `--width` and `--height`
- Start: Bottom-left
- Goal: Bottom-right
- Traps/mines: configurable with `--trap-layout`
- Actions: Up, Down, Left, Right

In the traditional setup, stepping into a trap produces a large negative reward.

In the positive-only setup, the agent is simply reset—no penalty applied.

The CLI prints an ASCII preview before training:

```text
Grid preview: 12 x 6, trap layout: scattered
Legend: S=start, G=goal, X=trap/mine, .=open
. . . . . . . . . . . .
. . . . . . . . . . . .
. . . . X . . . X . . .
. . . . . . X . . . . .
. . . X . . . . . X . .
S . . . . . . . . . . G
```

Supported trap layouts:

| Layout | Idea |
|--------|------|
| `scattered` | Individual traps placed around the board |
| `maze` | Trap walls with openings, forcing navigation around hazards |
| `islands` | Clusters of traps in the interior |
| `mixed` | Classic bottom cliff plus scattered traps, barriers, and islands |
| `bottom` | Classic bottom-row Cliff Walking layout |
| `gap` | Bottom-row cliff with one safe opening |
| `middle` | Horizontal trap row above the bottom |
| `double` | Bottom cliff plus extra upper traps |

## Results at a Glance
| Aspect          | Positive-Only     | Traditional |
|-----------------|-------------------|-------------|
| Trap Penalty    | None              | -100        |
| Step Reward     | Progress-based    | -1          |
| Exploration     | Higher            | Lower       |
| Convergence     | Slower            | Faster      |
| Policy Quality  | Comparable        | Optimal     |

### Key Observations

- The positive-only agent learned a safe path without aversive signals

- Q-values formed a smooth gradient toward the goal rather than sharp penalty zones

- Learning was more stable but required more episodes

## Why Positive-Only Reinforcement?

This experiment is inspired by reward shaping theory in reinforcement learning and positive reinforcement (R+) training in animal behavior.

Rather than teaching an agent what not to do, this approach focuses on reinforcing meaningful progress, mirroring how modern dog trainers teach behavior without fear or punishment.

The web app now uses the full reflection as the home page and places the actual run graphs directly in the write-up.

Repository Structure (High-Level)
```bash
cliff_walking_rp/
├── gymnasium_env/        # Custom environments (positive-only & traditional)
├── qlearning.py          # Main training script
├── qlearning.ipynb      # Interactive notebook with visualizations
├── visualize.py         # Q-table and policy visualization
├── Data Files/          # Q-tables, trajectories, snapshots
├── Results/             # Plots, GIFs, videos
└── documents/           # Extended documentation & reflection
```
Detailed file-level documentation is available in documents/.


## How to Run
### Install (recommended)
This repo is a Python project with a small CLI for training + exporting run bundles.

If you can install editable packages:

```bash
python3 -m pip install -e .
```

If you **cannot** install editable packages, you can still run everything by setting `PYTHONPATH`:

```bash
PYTHONPATH=src python3 -m cliff_walking_rp --help
```

### Train + export a run bundle (headless-safe default)
Positive-only:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train --variant positive_only --episodes 200 --seed 123 --out outputs
```

Traditional:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train --variant traditional --episodes 200 --seed 123 --out outputs
```

Different trap grid:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train \
  --variant positive_only \
  --episodes 200 \
  --width 12 \
  --height 6 \
  --trap-layout scattered \
  --seed 123 \
  --out outputs
```

Full CLI (all flags available):

```bash
PYTHONPATH=src python3 -m cliff_walking_rp train \
  --variant positive_only \
  --episodes 200 \
  --gamma 0.1 \
  --epsilon 0.08 \
  --epsilon-decay 0.01 \
  --alpha 1.0 \
  --seed 123 \
  --render none \
  --width 12 \
  --height 6 \
  --trap-layout scattered \
  --out outputs
```

```bash
PYTHONPATH=src python3 -m cliff_walking_rp train \
  --variant traditional \
  --episodes 200 \
  --gamma 0.1 \
  --epsilon 0.08 \
  --epsilon-decay 0.01 \
  --alpha 1.0 \
  --seed 123 \
  --render none \
  --width 12 \
  --height 6 \
  --trap-layout scattered \
  --out outputs
```

This produces a folder like:

```bash
outputs/<variant>/<runId>/
  summary.json
  plots/returns.png
  plots/steps.png
  media/maxq_heatmaps.gif
  media/qvalues_snapshots.json
  trajectories.json
  qtable.npy
  qtable_snapshots.npy
```

Notes:
- `media/qvalues_snapshots.json` is the **per-action Q-value “mental state”** data used by the website to render a 4-wedge heatmap (up/right/down/left) with a time slider.
- The PNGs/GIFs are still exported for quick previews and offline use.

### (Optional) Render during training
Rendering is slower and requires `pygame` + a display.

```bash
PYTHONPATH=src python3 -m cliff_walking_rp train --variant positive_only --episodes 50 --render human --render-sleep 0.05
```

## Static viewer site (Netlify)
The static viewer lives in `site/` (Next.js + TypeScript). It reads run bundles from:

```bash
site/public/runs/<runId>/
```

### Train both variants and sync the site
Run one command from the repo root:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train-site \
  --episodes 200 \
  --width 12 \
  --height 6 \
  --trap-layout scattered \
  --seed 123
```

This trains both `positive_only` and `traditional`, copies each exported bundle into
`site/public/runs/<runId>/`, and updates `site/public/runs/index.json` for the static
viewer. The command also prints the grid preview before training starts.

You can pass the same hyperparameters as `train`:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train-site \
  --episodes 500 \
  --gamma 0.1 \
  --epsilon 0.08 \
  --epsilon-decay 0.01 \
  --alpha 1.0 \
  --width 14 \
  --height 7 \
  --trap-layout mixed \
  --seed 123 \
  --out outputs \
  --site-runs site/public/runs
```

If both variants finish with the same timestamped `runId`, the site copy gets a
variant suffix such as `_traditional` so the static folders do not collide.

Open the run in the site:
- Reflection home page: `/`
- Training environments browser: `/environments`

Old routes are kept as redirects:
- `/reflection` redirects to `/`
- `/compare` and `/runs` redirect to `/environments`

### Run the site locally (dev)
From `site/`:

```bash
npm install
npm run dev
```

Then open `http://127.0.0.1:3000`.

If you see `EADDRINUSE`, something is already using port 3000. Stop the other process or run Next on a different port.

### Build the site (static export)
From `site/`:

```bash
npm install
npm run build
```

The exported static site will be in `site/out/` (ready for Netlify).

## Tech Stack

- Python 3.9+
- Gymnasium
- NumPy
- Matplotlib
- Seaborn
- Pygame

## References
- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." Proceedings of the Sixteenth International Conference on Machine Learning (ICML).
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

## Final Note

This project demonstrates that learning systems—artificial or biological—do not require punishment to succeed. Clear feedback, consistency, and reinforcement of progress can be enough to produce robust, adaptable behavior.
