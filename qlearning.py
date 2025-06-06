import gymnasium
import gymnasium_env
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

env = gymnasium.make("gymnasium_env/CliffWalkerPositive", render_mode="human")

observation, info = env.reset()

# Calculate number of states (ignoring target state)
num_states = (env.observation_space['agent'].high[0] + 1) * (env.observation_space['agent'].high[1] + 1)
num_actions = env.action_space.n  # Number of possible actions

xsize = env.observation_space['agent'].high[0] + 1
ysize = env.observation_space['agent'].high[1] + 1

# QTable : contains the Q-Values for every (state,action) pair
qtable = np.zeros((num_states, num_actions))

# hyperparameters
episodes = 100
gamma = 0.1
epsilon = 0.08
decay = 0.01
alpha = 1

# Tracking performance
episode_rewards = []
steps_per_episode = []
agent_positions_per_episode = []


# List to hold the snapshots of the Q-table
q_table_snapshots = []

# do a random action 1000 times
for i in range(episodes):
    positions_this_episode = []
    state_dict, info = env.reset()
    state = state_dict['agent'][0] + state_dict['agent'][1] * xsize  # Extract agent position
    print(f"State: {state}, NumStates: {num_states}")
    
    if state >= num_states:
        print(f"Agent X: {(state_dict['agent'][0] + 1) }, Agent Y: {state_dict['agent'][1] + 1}")
        raise ValueError(f"State {state} exceeds numstates {num_states}")

    steps = 0
    total_reward = 0
    done = False

    while not done:

        # Save current position before taking action
        pos_x = state % xsize
        pos_y = state // xsize
        positions_this_episode.append([pos_x, pos_y])

        os.system('clear')
        print(f"Episode # {i + 1} / {episodes}")
        print(f"Current Reward: {total_reward}")
        env.render()
        time.sleep(0.05)

        # count steps to finish game
        steps += 1

         # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        # if not select max action in Qtable (act greedy)
        else:
            action = np.argmax(qtable[state])


        # take action
        next_state_dict, reward, done, truncated, info = env.step(action)
        next_state = next_state_dict['agent'][0] + next_state_dict['agent'][1] * xsize  # Extract agent state
        done = done or truncated

        # Update total reward
        total_reward += reward

         # Q-learning update using Sutton's rule
        qtable[state][action] += alpha * (reward + gamma * max(qtable[next_state]) - qtable[state][action])
         # qtable[state][action] = reward + gamma * max(qtable[next_state])
         # update state
        state = next_state
    
    # At end of episode
    agent_positions_per_episode.append(positions_this_episode)

    # Decay epsilon to reduce exploration over time
    epsilon -= decay * epsilon

    # Store episode results
    episode_rewards.append(total_reward)
    steps_per_episode.append(steps)

    # Save a snapshot of the Q-table after each episode (or any other frequency you prefer)
    q_table_snapshots.append(qtable.copy())  # Save a copy of the current Q-table
    
    print(f"Accumulated Reward: {total_reward}")
    print(f"Done in {steps} steps")
    time.sleep(1)

np.save("qtable.npy", qtable)

env.close()

# Save snapshots of the Q-table for animation
np.save("q_table_snapshots_r+.npy", q_table_snapshots)

# Replace Q with your actual Q-table variable name if needed
Q = qtable  

# Load q_table_snapshots (you likely have it already in memory)
q_table_snapshots = np.load("q_table_snapshots_r+.npy")  # or use the `q_table_snapshots` list directly

xsize = 12  # or env.observation_space['agent'].high[0] + 1
ysize = 4   # or env.observation_space['agent'].high[1] + 1

heatmaps = []

for qtable in q_table_snapshots:
    heatmap = np.zeros((ysize, xsize))
    for y in range(ysize):
        for x in range(xsize):
            state = x + y * xsize
            heatmap[y][x] = np.max(qtable[state])
    heatmaps.append(heatmap.tolist())

# Save to JSON
import json
with open("qtable_heatmaps_r+.json", "w") as f:
    json.dump(heatmaps, f)

import json
with open("agent_positions_per_episode_r+.json", "w") as f:
    json.dump(agent_positions_per_episode, f)
