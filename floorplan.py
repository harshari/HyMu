import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Function to place chiplets with collision checking
def place_chiplets_on_entire_boundary(grid_dims, clusters):
    rows, cols = grid_dims
    grid = np.zeros((rows+1, cols+1), dtype=int)

    # Identify the cluster with the highest power density
    highest_pd_cluster = max(clusters.items(), key=lambda x: x[1]["pd"])
    cluster_id = highest_pd_cluster[1]["count"] * 100
    chiplet_area = highest_pd_cluster[1]["area"]
    chiplet_size = int(np.sqrt(chiplet_area))
    chiplet_count = highest_pd_cluster[1]["count"]

    placed_positions = []

    def is_position_valid(x, y):
        if x + chiplet_size > rows or y + chiplet_size > cols:
            return False
        for dx in range(chiplet_size):
            for dy in range(chiplet_size):
                if grid[x + dx, y + dy] != 0:
                    return False
        return True

    def place_chiplet(x, y):
        for dx in range(chiplet_size):
            for dy in range(chiplet_size):
                grid[x + dx, y + dy] = cluster_id
        placed_positions.append((x, y))

    chiplets_remaining = chiplet_count
    for y in range(0, cols, chiplet_size):  # Bottom boundary
        if chiplets_remaining > 0 and is_position_valid(0, y):
            place_chiplet(0, y)
            chiplets_remaining -= 1
    for x in range(chiplet_size, rows, chiplet_size):  # Right boundary
        if chiplets_remaining > 0 and is_position_valid(x, cols - chiplet_size):
            place_chiplet(x, cols - chiplet_size)
            chiplets_remaining -= 1
    for y in range(cols - chiplet_size, -1, -chiplet_size):  # Top boundary
        if chiplets_remaining > 0 and is_position_valid(rows - chiplet_size, y):
            place_chiplet(rows - chiplet_size, y)
            chiplets_remaining -= 1
    for x in range(rows - chiplet_size, -1, -chiplet_size):  # Left boundary
        if chiplets_remaining > 0 and is_position_valid(x, 0):
            place_chiplet(x, 0)
            chiplets_remaining -= 1

    return grid, placed_positions

# Function to find k farthest points from chiplets
def find_k_farthest_points_from_internal_edge(grid, chiplet_positions, grid_dims, k):
    rows, cols = grid_dims

    # Get all chiplet edge positions
    chiplet_array = np.array(chiplet_positions)

    # Generate all grid points
    all_positions = np.array([(x, y) for x in range(rows) for y in range(cols)])

    # Calculate Manhattan distances from all points to the chiplet cluster
    distances = cdist(all_positions, chiplet_array, metric="cityblock")
    sorted_indices = np.argsort(-distances.min(axis=1))  # Sort farthest first
    farthest_points = all_positions[sorted_indices[:k]]

    return farthest_points

# Visualization function
def visualize_grid_with_farthest_points(grid, chiplet_positions, grid_dims, farthest_points, title="Grid Visualization"):
    rows, cols = grid_dims
    plt.figure(figsize=(14, 8))
    plt.imshow(np.ones_like(grid), cmap="gray", origin="lower", alpha=0.1)
    plt.title(title)

    chiplet_size = int(np.sqrt(clusters["Cluster 4"]["area"]))
    for idx, (x, y) in enumerate(chiplet_positions, start=1):
        plt.gca().add_patch(
            plt.Rectangle(
                (y, x), chiplet_size, chiplet_size, color="orange", alpha=0.8, edgecolor="black", linewidth=1.5
            )
        )
        plt.text(
            y + chiplet_size / 2 - 0.2,
            x + chiplet_size / 2,
            f"C{idx}",
            color="black",
            fontsize=10,
            ha="center",
            va="center",
        )

    for x, y in farthest_points:
        plt.gca().add_patch(
            plt.Rectangle(
                (y, x), 1, 1, color="green", alpha=0.8, edgecolor="black", linewidth=1.5
            )
        )

    plt.xticks(ticks=np.arange(0, cols+1, 1))
    plt.yticks(ticks=np.arange(0, rows+1, 1))
    plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis (mm)")
    plt.ylabel("Y-axis (mm)")
    plt.xlim(-0.5, 26 - 0.5)
    plt.ylim(-0.5, 26 - 0.5)
    plt.show()

# Example usage
clusters = {
    "Cluster 4": {"count": 20, "pd": 28, "area": 4},
    
}
grid_dims = (24, 20)

grid, chiplet_positions = place_chiplets_on_entire_boundary(grid_dims, clusters)
farthest_points = find_k_farthest_points_from_internal_edge(grid, chiplet_positions, grid_dims, 135)
visualize_grid_with_farthest_points(grid, chiplet_positions, grid_dims, farthest_points, title="Farthest Points from Internal Edge")