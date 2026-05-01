from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrainResult:
    qtable: np.ndarray
    qtable_snapshots: np.ndarray
    episode_returns: list[float]
    steps_per_episode: list[int]
    agent_positions_per_episode: list[list[list[int]]]
    xsize: int
    ysize: int


def state_to_index(agent_xy: Any, xsize: int) -> int:
    # agent_xy is typically a numpy array like [x, y]
    x = int(agent_xy[0])
    y = int(agent_xy[1])
    return x + y * xsize


def q_learning_train(
    *,
    env: Any,
    episodes: int,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    alpha: float,
    seed: int,
    render: bool,
    render_sleep: float,
) -> TrainResult:
    # Seed numpy and env for reproducibility
    rng = np.random.default_rng(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    obs, _info = env.reset(seed=seed)
    xsize = int(env.observation_space["agent"].high[0]) + 1
    ysize = int(env.observation_space["agent"].high[1]) + 1
    num_states = xsize * ysize
    num_actions = int(env.action_space.n)

    qtable = np.zeros((num_states, num_actions), dtype=np.float32)

    episode_returns: list[float] = []
    steps_per_episode: list[int] = []
    agent_positions_per_episode: list[list[list[int]]] = []
    snapshots: list[np.ndarray] = []

    for _ep in range(episodes):
        positions_this_episode: list[list[int]] = []
        obs, _info = env.reset()
        state = state_to_index(obs["agent"], xsize)

        steps = 0
        total_reward = 0.0
        done = False

        while not done:
            pos_x = int(state % xsize)
            pos_y = int(state // xsize)
            positions_this_episode.append([pos_x, pos_y])

            if render:
                env.render()
                if render_sleep > 0:
                    time.sleep(render_sleep)

            steps += 1

            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(qtable[state]))

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            next_state = state_to_index(next_obs["agent"], xsize)

            total_reward += float(reward)

            # Sutton's Q-learning update
            qtable[state, action] = qtable[state, action] + alpha * (
                float(reward) + gamma * float(np.max(qtable[next_state])) - qtable[state, action]
            )
            state = next_state

        agent_positions_per_episode.append(positions_this_episode)
        episode_returns.append(total_reward)
        steps_per_episode.append(steps)
        snapshots.append(qtable.copy())

        epsilon = epsilon - epsilon_decay * epsilon

    qtable_snapshots = np.stack(snapshots, axis=0) if snapshots else np.zeros((0,) + qtable.shape)

    return TrainResult(
        qtable=qtable,
        qtable_snapshots=qtable_snapshots,
        episode_returns=episode_returns,
        steps_per_episode=steps_per_episode,
        agent_positions_per_episode=agent_positions_per_episode,
        xsize=xsize,
        ysize=ysize,
    )

