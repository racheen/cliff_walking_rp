import gymnasium
import gymnasium_env
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# --- Environment Setup ---
env = gymnasium.make("gymnasium_env/CliffWalker", render_mode="human")
observation, info = env.reset()

xsize = env.observation_space['agent'].high[0] + 1
ysize = env.observation_space['agent'].high[1] + 1
num_states = xsize * ysize
num_actions = env.action_space.n

# --- Q-Learning Parameters ---
qtable = np.zeros((num_states, num_actions))
episodes = 100
gamma = 0.1
epsilon = 0.08
decay = 0.01
alpha = 1

episode_rewards = []
steps_per_episode = []
q_table_snapshots = []

# --- Heatmap Setup ---
triangle_patches = []
text_labels = []

# --- Create the figure for heatmap (only) ---
fig1, ax1 = plt.subplots(figsize=(10, 6))

# --- Create the figure for reward and steps plots (after training) ---
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))

# --- Top Left: Q-Table Heatmap (Live) ---
initial_norm = Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap("viridis")

# Create triangles and labels
for y in range(ysize):
    for x in range(xsize):
        grid_state = y * xsize + x
        for action in range(num_actions):
            # Corrected triangle positioning, visual orientation for the action
            cx, cy = x, ysize - y - 1  # Flip y for visual positioning

            if action == 0:  # Right
                points = [(cx + 1, cy), (cx + 1, cy + 1), (cx + 0.5, cy + 0.5)]
            elif action == 1:  # Up
                points = [(cx, cy), (cx + 1, cy), (cx + 0.5, cy + 0.5)]  # Bottom triangle (points up)
            elif action == 2:  # Left
                points = [(cx, cy), (cx, cy + 1), (cx + 0.5, cy + 0.5)]
            elif action == 3:  # Down
                points = [(cx, cy + 1), (cx + 1, cy + 1), (cx + 0.5, cy + 0.5)]  # Top triangle (points down)

            triangle = patches.Polygon(points, color='white')
            ax1.add_patch(triangle)
            triangle_patches.append(triangle)

            # Label positioning offset for better alignment
            offset = {
                0: (0.3, 0),    # Right
                1: (0, -0.3),    # Up
                2: (-0.3, 0),   # Left
                3: (0, 0.3),   # Down
            }[action]

            # Adjust label position to align correctly with action direction
            label_x = cx + 0.5 + offset[0]
            label_y = cy + 0.5 + offset[1]

            # Place the label for Q-value inside the triangle
            text = ax1.text(
                label_x,
                label_y,
                f"{qtable[grid_state][action]:.2f}",
                ha='center',
                va='center',
                fontsize=5,
                color="white"
            )
            text_labels.append(text)

ax1.set_xlim(0, xsize)
ax1.set_ylim(0, ysize)
ax1.set_aspect('equal')
ax1.set_xticks(range(xsize))
ax1.set_yticks(range(ysize))
ax1.set_title("Live Q-Table Heatmap")
plt.colorbar(cm.ScalarMappable(norm=initial_norm, cmap=cmap), ax=ax1)

assert len(triangle_patches) == num_states * num_actions
assert len(text_labels) == num_states * num_actions

def update_q_heatmap():
    flat_q = qtable.flatten()
    norm = Normalize(vmin=np.min(flat_q), vmax=np.max(flat_q))
    for y in range(ysize):
        for x in range(xsize):
            vis_state = y * xsize + x
            for action in range(num_actions):
                idx = vis_state * num_actions + action
                q = qtable[vis_state][action]
                # Set triangle color based on the Q-value
                triangle_patches[idx].set_facecolor(cmap(norm(q)))
                # Update the label for Q-value
                text_labels[idx].set_text(f"{q:.2f}")
    fig1.canvas.draw()
    plt.pause(0.001)

# --- Training Loop ---
for i in range(episodes):
    state_dict, info = env.reset()
    env.unwrapped.set_agent_epsilon(epsilon)
    state = state_dict['agent'][0] + state_dict['agent'][1] * xsize

    if state >= num_states:
        raise ValueError(f"State {state} out of bounds.")

    steps = 0
    total_reward = 0
    done = False

    while not done:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Episode # {i + 1} / {episodes}")
        print(f"Steps: {steps}")
        print(f"Reward: {total_reward}")
        env.render()
        time.sleep(0.05)

        steps += 1
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])

        next_state_dict, reward, done, truncated, info = env.step(action)
        next_state = next_state_dict['agent'][0] + next_state_dict['agent'][1] * xsize
        done = done or truncated
        total_reward += reward

        qtable[state][action] += alpha * (
            reward + gamma * np.max(qtable[next_state]) - qtable[state][action]
        )
        state = next_state

        update_q_heatmap()

    epsilon -= decay * epsilon
    env.unwrapped.set_agent_epsilon(epsilon)

    episode_rewards.append(total_reward)
    steps_per_episode.append(steps)
    q_table_snapshots.append(qtable.copy())

    print(f"Episode Done - Reward: {total_reward}, Steps: {steps}")
    time.sleep(1)

# --- Bottom Left: Reward Plot ---
ax2.plot(range(episodes), episode_rewards, marker='o', linestyle='-')
ax2.set_xlabel("Episode")
ax2.set_ylabel("Total Reward")
ax2.set_title("Episode Returns")

# --- Bottom Right: Steps Plot ---
ax3.plot(range(episodes), steps_per_episode, marker='o', linestyle='-')
ax3.set_xlabel("Episode")
ax3.set_ylabel("Steps Taken")
ax3.set_title("Steps per Episode")

plt.tight_layout()

# Display both the heatmap and the reward/steps plots in separate windows
plt.figure(fig1.number)  # This will bring up the first figure (heatmap)
plt.show()  # Display heatmap first
plt.figure(fig2.number)  # This will bring up the second figure (reward and steps plots)
plt.show()  # Display reward and steps graphs