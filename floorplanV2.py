import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Cluster configurations
clusters = {
    "Cluster 1": {"count": 28, "pd": 8, "area": 9},  # 3x3 chiplets
    "Cluster 2": {"count": 10, "pd": 1, "area": 4},  # 2x2 chiplets
    "Cluster 3": {"count": 15, "pd": 4, "area": 4},  # 2x2 chiplets
    "Cluster 4": {"count": 25, "pd": 28, "area": 4},  # 2x2 chiplets
}

# Grid dimensions
grid_dims = (20, 24)  # Initial grid dimensions
grid = np.zeros((grid_dims[0]+1, grid_dims[1]+1), dtype=int)

# Visualization function with legends
def visualize_grid_with_legends_updated(grid, cluster_positions, grid_dims, clusters, title="Grid Visualization"):
    rows, cols = grid_dims
    plt.figure(figsize=(12, 8))
    plt.imshow(np.ones_like(grid), cmap="gray", origin="lower", alpha=0.1)
    plt.title(title)

    # Define colors for clusters based on power density
    cluster_colors = {
        "Cluster 4": "red",       # Cluster 4 (highest PD)
        "Cluster 1": "orange",    # Cluster 1 (medium PD)
        "Cluster 3": "blue",      # Cluster 3 (lower PD)
        "Cluster 2": "green",     # Cluster 2 (lowest PD)
    }

    for cluster_key, positions in cluster_positions.items():
        chiplet_size = int(np.sqrt(clusters[cluster_key]["area"]))
        color = cluster_colors[cluster_key]
        for idx, (x, y) in enumerate(positions, start=1):
            plt.gca().add_patch(
                plt.Rectangle(
                    (y, x), chiplet_size, chiplet_size, color=color, alpha=0.8, edgecolor="black", linewidth=1.5
                )
            )
            plt.text(
                y + chiplet_size / 2 - 0.2,
                x + chiplet_size / 2,
                f"C{idx}",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
            )

    # Add a legend for the clusters
    handles = [
        plt.Line2D([0], [0], color=cluster_colors[key], lw=4, label=f"{key}: PD={clusters[key]['pd']} W/chiplet")
        for key in clusters.keys()
    ]
    plt.legend(handles=handles, loc="upper right", title="Clusters")

    plt.xticks(ticks=np.arange(0, cols + 1, 1))
    plt.yticks(ticks=np.arange(0, rows + 1, 1))
    plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis (mm)")
    plt.ylabel("Y-axis (mm)")
    plt.show()

# Function to validate if a chiplet can be placed without overlap
def is_valid_position(grid, x, y, chiplet_size):
    rows, cols = grid.shape
    if x + chiplet_size > rows or y + chiplet_size > cols:
        return False
    for dx in range(chiplet_size):
        for dy in range(chiplet_size):
            if grid[x + dx, y + dy] != 0:
                return False
    return True

# Step 1: Place Cluster 4 (20 chiplets on boundary)
cluster_4_positions = []
chiplets_remaining = clusters["Cluster 4"]["count"]
chiplet_size = int(np.sqrt(clusters["Cluster 4"]["area"]))

for y in range(0, grid_dims[1] - 1, chiplet_size):  # Bottom boundary
    if chiplets_remaining > 0:
        cluster_4_positions.append((0, y))
        for dx in range(chiplet_size):
            for dy in range(chiplet_size):
                grid[0 + dx, y + dy] = 4000
        chiplets_remaining -= 1

for x in range(chiplet_size, grid_dims[0], chiplet_size):  # Right boundary
    if chiplets_remaining > 0:
        cluster_4_positions.append((x, grid_dims[1] - chiplet_size))
        for dx in range(chiplet_size):
            for dy in range(chiplet_size):
                grid[x + dx, grid_dims[1] - chiplet_size + dy] = 4000
        chiplets_remaining -= 1

for y in range(grid_dims[1] - chiplet_size, -1, -chiplet_size):  # Top boundary
    if chiplets_remaining > 0:
        cluster_4_positions.append((grid_dims[0] - chiplet_size, y))
        for dx in range(chiplet_size):
            for dy in range(chiplet_size):
                grid[grid_dims[0] - chiplet_size + dx, y + dy] = 4000
        chiplets_remaining -= 1

for x in range(grid_dims[0] - chiplet_size * 2, 0, -chiplet_size):  # Left boundary
    if chiplets_remaining > 0:
        cluster_4_positions.append((x, 0))
        for dx in range(chiplet_size):
            for dy in range(chiplet_size):
                grid[x + dx, 0 + dy] = 4000
        chiplets_remaining -= 1

# Step 2: Place Cluster 2
cluster_2_positions = []
chiplets_remaining = clusters["Cluster 2"]["count"]
chiplet_size = int(np.sqrt(clusters["Cluster 2"]["area"]))

for x in range(grid_dims[0]):
    for y in range(grid_dims[1]):
        if chiplets_remaining > 0 and grid[x, y] == 0:
            if is_valid_position(grid, x, y, chiplet_size):
                cluster_2_positions.append((x, y))
                for dx in range(chiplet_size):
                    for dy in range(chiplet_size):
                        grid[x + dx, y + dy] = 2000
                chiplets_remaining -= 1

# Step 3: Place Cluster 3
cluster_3_positions = []
chiplets_remaining = clusters["Cluster 3"]["count"]
chiplet_size = int(np.sqrt(clusters["Cluster 3"]["area"]))

for x in range(grid_dims[0]):
    for y in range(grid_dims[1]):
        if chiplets_remaining > 0 and grid[x, y] == 0:
            if is_valid_position(grid, x, y, chiplet_size):
                cluster_3_positions.append((x, y))
                for dx in range(chiplet_size):
                    for dy in range(chiplet_size):
                        grid[x + dx, y + dy] = 2500
                chiplets_remaining -= 1

# Step 4: Place Cluster 1
cluster_1_positions = []
chiplets_remaining = clusters["Cluster 1"]["count"]
chiplet_size = int(np.sqrt(clusters["Cluster 1"]["area"]))

for x in range(grid_dims[0]):
    for y in range(grid_dims[1]):
        if chiplets_remaining > 0 and grid[x, y] == 0:
            if is_valid_position(grid, x, y, chiplet_size):
                cluster_1_positions.append((x, y))
                for dx in range(chiplet_size):
                    for dy in range(chiplet_size):
                        grid[x + dx, y + dy] = 1000
                chiplets_remaining -= 1

# Final Visualization
visualize_grid_with_legends_updated(
    grid,
    {
        "Cluster 4": cluster_4_positions,
        "Cluster 2": cluster_2_positions,
        "Cluster 3": cluster_3_positions,
        "Cluster 1": cluster_1_positions,
    },
    grid_dims,
    clusters,
    title="Final Placement of All Clusters"
)