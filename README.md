# Trap Grid Walking with Positive-Only Reinforcement

This project explores whether a reinforcement learning agent can learn a safe path without punishment. It began with the classic Cliff Walking problem and now uses configurable grid worlds where traps can appear in different layouts across the board.

The project compares two Q-learning agents:

- **Positive-only**: rewards progress toward the goal and resets after unsafe choices without applying a negative reward.
- **Traditional**: uses a step penalty and a large trap penalty.

The result is a small research app: a Python trainer exports run bundles, and a static Next.js viewer turns those runs into reflection pages, environment comparisons, charts, and Q-value heatmaps.

## Key Questions

- Can an agent learn a useful policy without negative rewards?
- How does reward shaping change convergence and exploration?
- What can reinforcement learning reveal about humane learning systems?

## Environment

- Grid size is configurable with `--width` and `--height`.
- Start is bottom-left.
- Goal is bottom-right.
- Actions are up, down, left, and right.
- Traps are configurable with `--trap-layout`.

Supported trap layouts:

| Layout | Description |
| --- | --- |
| `bottom` | Classic bottom-row Cliff Walking layout |
| `gap` | Bottom-row traps with one safe opening |
| `middle` | Horizontal trap row above the bottom |
| `double` | Bottom traps plus extra upper traps |
| `scattered` | Individual traps placed around the board |
| `maze` | Trap walls with openings |
| `islands` | Interior trap clusters |
| `mixed` | Bottom traps, scattered traps, barriers, and islands |

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

## Project Layout

```text
.
├── src/cliff_walking_rp/      # Packaged CLI, training code, and artifact export
├── gymnasium_env/             # Custom Gymnasium environments and layouts
├── site/                      # Static Next.js viewer
├── documents/                 # Reflection PDF
├── qlearning*.py              # Earlier exploratory scripts
├── visualize.py               # Legacy visualization helper
└── pyproject.toml             # Python package metadata
```

Generated training outputs belong in `outputs/` or `site/public/runs/`. Local caches, build output, and ad hoc top-level Q-table files are ignored.

## Install

Use Python 3.9 or newer.

```bash
python3 -m pip install -e .
```

If editable installs are not available, run the package with `PYTHONPATH`:

```bash
PYTHONPATH=src python3 -m cliff_walking_rp --help
```

For headless machines, set `MPLCONFIGDIR` to a writable local directory:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp --help
```

## Train One Run

Positive-only:

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

Traditional:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train \
  --variant traditional \
  --episodes 200 \
  --width 12 \
  --height 6 \
  --trap-layout scattered \
  --seed 123 \
  --out outputs
```

Each run creates a bundle like:

```text
outputs/<variant>/<runId>/
├── summary.json
├── trajectories.json
├── qtable.npy
├── qtable_snapshots.npy
├── plots/
│   ├── returns.png
│   └── steps.png
└── media/
    ├── maxq_heatmaps.gif
    └── qvalues_snapshots.json
```

`media/qvalues_snapshots.json` powers the website's per-action Q-value heatmap.

## Sync Runs Into The Site

Train both variants and copy their exported bundles into the static viewer:

```bash
PYTHONPATH=src MPLCONFIGDIR=.mplcache python3 -m cliff_walking_rp train-site \
  --episodes 200 \
  --width 12 \
  --height 6 \
  --trap-layout scattered \
  --seed 123 \
  --out outputs \
  --site-runs site/public/runs
```

This updates `site/public/runs/index.json`, which the viewer uses to discover available runs. If two variants finish with the same timestamped run ID, the copied site folder gets a variant suffix so paths do not collide.

## Run The Viewer

From `site/`:

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

Viewer routes:

- `/` shows the reflection page with live run graphs.
- `/environments` groups exported runs by grid and trap layout.
- `/reflection`, `/compare`, and `/runs` are kept as redirects for old links.

Build the static export:

```bash
npm run build
```

The exported site is written to `site/out/`.

## Results At A Glance

| Aspect | Positive-only | Traditional |
| --- | --- | --- |
| Trap penalty | None | `-100` |
| Step reward | Progress-based | `-1` |
| Exploration | Higher | Lower |
| Convergence | Slower | Faster |
| Policy quality | Comparable | Often optimal |

Key observations:

- The positive-only agent can learn safe behavior without aversive reward signals.
- Positive-only Q-values tend to form a smoother gradient toward the goal.
- Traditional training converges quickly, but the penalty zones dominate the learned values.

## References

- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." Proceedings of the Sixteenth International Conference on Machine Learning.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
