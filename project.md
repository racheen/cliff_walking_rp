# Cliff Walking R+

## Overview

This repository explores reward shaping in reinforcement learning by comparing a positive-only Q-learning agent with a traditional penalty-based agent.

The Python side trains agents in configurable grid worlds and exports run bundles. The `site/` app is a static Next.js viewer for reflection content, environment comparisons, charts, and Q-value heatmaps.

## Main Areas

- `src/`: packaged trainer, CLI, and artifact export code
- `gymnasium_env/`: custom environments and wrappers
- `site/`: Next.js artifact viewer
- `site/public/runs/`: exported training bundles consumed by the viewer

## Default Visual Direction

- Neo-minimal
- Grayscale-first hierarchy with pastel green accents
- Soft radius and subtle elevation
- Clean, accessible contrast

## Common Workflows

- Train runs with the CLI
- Export artifacts for the static viewer
- Review run comparisons in the web app
- Keep viewer styling centralized in `site/src/app/globals.css`

