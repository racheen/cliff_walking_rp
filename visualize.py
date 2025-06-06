import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, cm, colors

# Load Q-table
Q = np.load("qtable.npy")

# Set grid dimensions
xsize = 12
ysize = 4

# Define position for value text
arrow_pos = {
    0: (0.75, 0.5),  # right
    1: (0.5, 0.75),  # up
    2: (0.25, 0.5),  # left
    3: (0.5, 0.25),  # down
}

# Create figure
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)

ax.set_xlim(0, xsize)
ax.set_ylim(0, ysize)
ax.set_xticks(np.arange(xsize))
ax.set_yticks(np.arange(ysize))
ax.set_xticklabels(np.arange(xsize))
ax.set_yticklabels(np.arange(ysize - 1, -1, -1))
ax.grid(True)
ax.set_title("Q-Values per Action in Grid")
ax.set_xlabel("X position")
ax.set_ylabel("Y position (top = 0)")

# Normalize colors
all_q = Q.flatten()
norm = colors.Normalize(vmin=np.min(all_q), vmax=np.max(all_q))
cmap = plt.get_cmap('YlGnBu')


for y in range(ysize):
    for x in range(xsize):
        state = y * xsize + x
        q_values = Q[state]
        colors_per_action = [cmap(norm(q)) for q in q_values]
        grid_y = ysize - y - 1  # Flip Y for display

        cx, cy = x, grid_y

        # Draw triangles
        ax.add_patch(patches.Polygon([(cx, cy + 1), (cx + 1, cy + 1), (cx + 0.5, cy + 0.5)],
                                     color=colors_per_action[1]))  # up
        ax.add_patch(patches.Polygon([(cx, cy), (cx + 1, cy), (cx + 0.5, cy + 0.5)],
                                     color=colors_per_action[3]))  # down
        ax.add_patch(patches.Polygon([(cx, cy), (cx, cy + 1), (cx + 0.5, cy + 0.5)],
                                     color=colors_per_action[2]))  # left
        ax.add_patch(patches.Polygon([(cx + 1, cy), (cx + 1, cy + 1), (cx + 0.5, cy + 0.5)],
                                     color=colors_per_action[0]))  # right

        # Draw Q-values as text
        for i, (tx, ty) in arrow_pos.items():
            ax.text(cx + tx, cy + ty, f"{q_values[i]:.2f}", fontsize=6,
                    ha='center', va='center', color='black')

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, orientation='vertical', label='Q-value')

plt.tight_layout()
plt.show()
