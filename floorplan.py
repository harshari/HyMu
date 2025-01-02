from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Cluster configurations
clusters = {
    "Cluster 1": {"count": 28, "pd": 8, "area": 8},  # 2x4 or 4x2 chiplets
    "Cluster 2": {"count": 12, "pd": 1, "area": 4},  # 2x2 chiplets
    "Cluster 3": {"count": 18, "pd": 4, "area": 4},  # 2x2 chiplets
    "Cluster 4": {"count": 24, "pd": 28, "area": 4},  # 2x2 chiplets
}

# Grid dimensions
grid_dims = (20, 22)
grid = np.zeros((grid_dims[0] + 1, grid_dims[1] + 1), dtype=int)

# Visualization function with legends
def visualize_grid_with_legends_updated(grid, cluster_positions, grid_dims, clusters, title="Grid Visualization"):
    rows, cols = grid_dims
    plt.figure(figsize=(12, 8))
    plt.imshow(np.ones_like(grid), cmap="gray", origin="lower", alpha=0.1)
    plt.title(title)

    cluster_colors = {
        "Cluster 4": "red",
        "Cluster 1": "orange",
        "Cluster 3": "blue",
        "Cluster 2": "green",
    }

    for cluster_key, positions in cluster_positions.items():
        chiplet_area = clusters[cluster_key]["area"]
        chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
        color = cluster_colors[cluster_key]
        for idx, (x, y) in enumerate(positions, start=1):
            plt.gca().add_patch(
                plt.Rectangle(
                    (y, x), chiplet_size[1], chiplet_size[0], color=color, alpha=0.8, edgecolor="black", linewidth=1.5
                )
            )
            plt.text(
                y + chiplet_size[1] / 2 - 0.2,
                x + chiplet_size[0] / 2,
                f"C{idx}",
                color="black",
                fontsize=8,
                ha="center",
                va="center",
            )

    handles = [
        plt.Line2D([0], [0], color=cluster_colors[key], lw=4, label=f"{key}: PD={clusters[key]['pd']} W/chiplet")
        for key in clusters.keys()
    ]
    plt.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), title="Clusters")
    plt.xticks(ticks=np.arange(0, cols + 1, 1))
    plt.yticks(ticks=np.arange(0, rows + 1, 1))
    plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis (mm)")
    plt.ylabel("Y-axis (mm)")
    plt.show()

# Function to validate chiplet placement
def is_valid_position(grid, x, y, chiplet_size):
    rows, cols = grid.shape
    if x + chiplet_size[0] > rows or y + chiplet_size[1] > cols:
        return False
    for dx in range(chiplet_size[0]):
        for dy in range(chiplet_size[1]):
            if grid[x + dx, y + dy] != 0:
                return False
    return True

# Function to place chiplets
def place_chiplets(grid, cluster_key, clusters, boundary_only=False):
    chiplet_area = clusters[cluster_key]["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]

    # Boundary placement
    if boundary_only:
        for y in range(0, grid_dims[1], chiplet_size[1]):  # Bottom boundary
            if chiplets_remaining > 0 and is_valid_position(grid, 0, y, chiplet_size):
                positions.append((0, y))
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[0 + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
                chiplets_remaining -= 1

        for x in range(chiplet_size[0], grid_dims[0], chiplet_size[0]):  # Right boundary
            if chiplets_remaining > 0 and is_valid_position(grid, x, grid_dims[1] - chiplet_size[1], chiplet_size):
                positions.append((x, grid_dims[1] - chiplet_size[1]))
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[x + dx, grid_dims[1] - chiplet_size[1] + dy] = int(cluster_key.split()[-1]) * 1000
                chiplets_remaining -= 1

    # Inside placement (towards center)
    for x in range(grid_dims[0]):
        for y in range(grid_dims[1]):
            if chiplets_remaining > 0 and is_valid_position(grid, x, y, chiplet_size):
                positions.append((x, y))
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
                chiplets_remaining -= 1

    return positions

# Function to calculate average hop count
def calculate_average_hop_count(grid, cluster_positions):
    total_distance = 0
    total_pairs = 0

    for cluster_a, positions_a in cluster_positions.items():
        for cluster_b, positions_b in cluster_positions.items():
            if cluster_a != cluster_b:
                for pos_a in positions_a:
                    for pos_b in positions_b:
                        center_a = (pos_a[0] + 1, pos_a[1] + 1)
                        center_b = (pos_b[0] + 1, pos_b[1] + 1)
                        total_distance += euclidean(center_a, center_b)
                        total_pairs += 1

    return total_distance / total_pairs if total_pairs > 0 else float("inf")

# Function to check for empty spaces
def has_empty_spaces(grid, clusters):
    expected_filled = sum(clusters[key]["count"] for key in clusters)
    actual_filled = np.count_nonzero(grid)
    return actual_filled < expected_filled

# Function to generate all orderings and minimize hop count
def find_min_hop_design(grid, clusters, grid_dims):
    min_hop_count = float("inf")
    best_ordering = None
    best_positions = None
    best_grid = None

    for ordering in permutations(clusters.keys()):
        temp_grid = np.zeros((grid_dims[0] + 1, grid_dims[1] + 1), dtype=int)
        cluster_positions = {}

        for idx, cluster in enumerate(ordering):
            if idx == 0:
                cluster_positions[cluster] = place_chiplets(temp_grid, cluster, clusters, boundary_only=True)
            else:
                cluster_positions[cluster] = place_chiplets(temp_grid, cluster, clusters, boundary_only=False)

        if not has_empty_spaces(temp_grid, clusters):
            avg_hop_count = calculate_average_hop_count(temp_grid, cluster_positions)
            if avg_hop_count < min_hop_count:
                min_hop_count = avg_hop_count
                best_ordering = ordering
                best_positions = cluster_positions
                best_grid = temp_grid

    return best_ordering, best_positions, best_grid, min_hop_count

# Generate the best design
best_ordering, best_positions, best_grid, min_hop_count = find_min_hop_design(grid, clusters, grid_dims)

# Plot the best design
if best_grid is not None:
    visualize_grid_with_legends_updated(
        best_grid,
        best_positions,
        grid_dims,
        clusters,
        title=f"Best Design with Minimized Hop Count ({min_hop_count:.2f})"
    )
    print(f"Best Ordering: {best_ordering}")
else:
    print("No valid design found.")