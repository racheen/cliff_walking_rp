import numpy as np
import json

# # Load q_table_snapshots (you likely have it already in memory)
# q_table_snapshots = np.load("q_table_snapshots.npy")  # or use the `q_table_snapshots` list directly

# xsize = 12  # or env.observation_space['agent'].high[0] + 1
# ysize = 4   # or env.observation_space['agent'].high[1] + 1

# heatmaps = []

# for qtable in q_table_snapshots:
#     heatmap = np.zeros((ysize, xsize))
#     for y in range(ysize):
#         for x in range(xsize):
#             state = x + y * xsize
#             heatmap[y][x] = np.max(qtable[state])
#     heatmaps.append(heatmap.tolist())

# # Save to JSON
# import json
# with open("qtable_heatmaps.json", "w") as f:
#     json.dump(heatmaps, f)

# Load qtable_snapshot to list

qtable = np.load("qtable.npy").tolist()
with open("q_table_only.json", "w") as f:
    json.dump(qtable, f)