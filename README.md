# Cliff Walking with Positive-Only Reinforcement

## Overview
This project explores whether a reinforcement learning agent can learn an optimal policy without punishment, using positive-only reinforcement instead of traditional penalties.

Using the classic Cliff Walking environment, I compare two agents:

- A **traditional Q-learning agent** trained with step penalties and a harsh cliff penalty

- A **positive-only agent** trained by rewarding progress toward the goal, with no negative rewards

The results show that an agent can successfully learn safe and effective behavior using encouragement alone—raising interesting parallels between **machine learning and modern animal training practices**.

## Key Questions
- Can an agent learn an optimal path without negative rewards?
- How does reward shaping affect learning stability and exploration?
- What can reinforcement learning teach us about humane learning systems?

## Core Concepts
- Algorithm: Q-learning (from scratch)
- Environment: Custom Gymnasium Cliff Walking implementation
- Reward Strategy:
    - Traditional: step penalty + cliff penalty
    - Positive-only: reward for moving closer to the goal, no punishment for mistakes
- Comparison: Learning dynamics, convergence, and learned policies

## Environment Summary
- Grid Size: 12 × 4
- Start: Bottom-left
- Goal: Bottom-right
- Cliff: Bottom row between start and goal
- Actions: Up, Down, Left, Right

In the traditional setup, stepping into the cliff produces a large negative reward.

In the positive-only setup, the agent is simply reset—no penalty applied.

## Results at a Glance
| Aspect          | Positive-Only     | Traditional |
|-----------------|-------------------|-------------|
| Cliff Penalty   | None              | -100        |
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

Full reflection and cross-disciplinary analysis
A detailed discussion connecting reinforcement learning, operant conditioning, and humane dog training is available in:

```bash
documents/reflection.pdf
```

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
  --out outputs
```

This produces a folder like:

```bash
outputs/<variant>/<runId>/
  summary.json
  plots/returns.png
  plots/steps.png
  media/maxq_heatmaps.gif
  trajectories.json
  qtable.npy
  qtable_snapshots.npy
```

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

### Add a run to the site (manual sync)
1) Train and export:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train --variant positive_only --episodes 200 --seed 123 --out outputs
```

2) Copy the run folder into the site:

```bash
mkdir -p site/public/runs/<runId>
cp -R outputs/positive_only/<runId>/* site/public/runs/<runId>/
```

3) Update `site/public/runs/index.json` with an entry:

```json
{
  "runId": "<runId>",
  "variant": "positive_only",
  "summaryPath": "<runId>/summary.json"
}
```

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