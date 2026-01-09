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
### Train the Agent
```
python qlearning.py
```

### Visualize Learned Policy
```
python visualize.py
```

### Interactive Exploration
```
jupyter notebook qlearning.ipynb
```

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