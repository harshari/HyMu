import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
import pandas as pd
from itertools import permutations
from generate_mfit_floorplan import generate_power_config_file
import os

# Cluster configurations
clusters = {
    "Cluster 1": {"count": 24, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": 0.87e-12},
    "Cluster 2": {"count": 28, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": 0.3e-12},
    "Cluster 3": {"count": 0, "pd": 2, "area": 4, "memory": 108, "tops": 11e12, "energy_per_mac": 0.18e-12},
    "Cluster 4": {"count": 18, "pd": 8, "area": 4, "memory": 2400, "tops": 35e12, "energy_per_mac": 0.22e-12},
    "Cluster 5": {"count": 12, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": 0.27e-12},
}

# Grid dimensions
grid_dims = (20, 22)

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

def find_max_distance_position(grid, chiplet_size):
    rows, cols = grid.shape
    center_x, center_y = rows // 2, cols // 2
    max_distance = -1
    best_pos = None
    for x in range(rows):
        for y in range(cols):
            if is_valid_position(grid, x, y, chiplet_size):
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if dist > max_distance:
                    max_distance = dist
                    best_pos = (x, y)
    return best_pos

def place_chiplets_fixed(grid, cluster_key, clusters):
    chiplet_area = clusters[cluster_key]["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]
    while chiplets_remaining > 0:
        next_pos = find_max_distance_position(grid, chiplet_size)
        if next_pos is None:
            print(f"Unable to place all chiplets for {cluster_key}. Remaining: {chiplets_remaining}")
            return positions
        x, y = next_pos
        positions.append((x, y))
        for dx in range(chiplet_size[0]):
            for dy in range(chiplet_size[1]):
                grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
        chiplets_remaining -= 1
    return positions

def generate_floorplan_data(cluster_positions, clusters):
    floorplan_data = []
    for cluster_key, positions in cluster_positions.items():
        chiplet_area = clusters[cluster_key]["area"]
        chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
        for idx, (x, y) in enumerate(positions, start=1):
            floorplan_data.append({
                "Chiplet": f"{cluster_key}-{idx}".replace("Cluster ", "C"),
                "Lower_Left_Corner": (x, y),
                "Length": chiplet_size[0],
                "Breadth": chiplet_size[1],
            })
    return floorplan_data

def adjust_chiplets_with_spacing(floorplan_data, spacing=0.25):
    adjusted_floorplan = []
    half_spacing = spacing / 2
    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        adjusted_x = x + half_spacing * x
        adjusted_y = y + half_spacing * y
        center_x = adjusted_x + length / 2
        center_y = adjusted_y + breadth / 2
        adjusted_floorplan.append({
            "Chiplet": chiplet["Chiplet"],
            "Lower_Left_Corner": (adjusted_x, adjusted_y),
            "Length": length,
            "Breadth": breadth,
            "Center": (center_x, center_y),
        })
    return adjusted_floorplan

def evaluate_power_variance(floorplan_data, grid_dims, region_divisions=2):
    grid_rows, grid_cols = grid_dims
    power_density_grid = np.zeros((grid_rows, grid_cols))

    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        cluster_key = f"Cluster {chiplet['Chiplet'].split('-')[0][1:]}"  # Map C1 to Cluster 1
        power = clusters[cluster_key]["pd"]

        for dx in range(length):
            for dy in range(breadth):
                grid_x = int(x) + dx
                grid_y = int(y) + dy
                # Check if indices are within bounds
                if 0 <= grid_x < grid_rows and 0 <= grid_y < grid_cols:
                    power_density_grid[grid_x, grid_y] += power / (length * breadth)

    score = 0
    region_size_rows = grid_rows // region_divisions
    region_size_cols = grid_cols // region_divisions

    for region_row in range(0, grid_rows, region_size_rows):
        for region_col in range(0, grid_cols, region_size_cols):
            region = power_density_grid[region_row:region_row + region_size_rows,
                                        region_col:region_col + region_size_cols]
            score += np.var(region)

    return score

def score_design(floorplan_data, grid_dims):
    power_variance_score = evaluate_power_variance(floorplan_data, grid_dims, region_divisions=2)
    return power_variance_score

def visualize_chiplet_centers(i, adjusted_floorplan_with_spacing, spacing=0.25, title="Chiplet Centers Visualization"):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    
    # Determine max bounds for visualization
    max_x = max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in adjusted_floorplan_with_spacing)
    max_y = max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in adjusted_floorplan_with_spacing)
    plt.xlim(0, max_x + spacing)
    plt.ylim(0, max_y + spacing)
    
    # Ensure equal scaling for axes
    plt.gca().set_aspect('equal', adjustable='box')

    for chiplet in adjusted_floorplan_with_spacing:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        center_x, center_y = chiplet["Center"]
        
        # Plot the center as a dot
        plt.plot(center_y, center_x, 'ro')
        plt.text(
            center_y,
            center_x,
            chiplet["Chiplet"],
            color="black",
            fontsize=8,
            ha="center",
            va="center",
        )

    # Add tick marks at every unit
    plt.xticks(ticks=np.arange(0, max_x + 2, 1))
    plt.yticks(ticks=np.arange(0, max_y + 2, 1))
    plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis (mm)")
    plt.ylabel("Y-axis (mm)")
    plt.title(f"exp_{i}/chiplet-centers")
    plt.savefig(f"exp_{i}/chiplet-centers.png")
    plt.close()


def visualize_chiplet_floorplan(i, floorplan_data, clusters, spacing=0.25, title="Chiplet Placement with Tick Marks and Labels"):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    max_x = max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in floorplan_data)
    max_y = max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in floorplan_data)
    plt.xlim(0, max_x + spacing)
    plt.ylim(0, max_y + spacing)

    # Ensure equal scaling for axes
    plt.gca().set_aspect('equal', adjustable='box')

    # Cluster-specific colors
    cluster_colors = {
        "C1": "red",
        "C2": "orange",
        "C3": "purple",
        "C4": "blue",
        "C5": "green"
    }

    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        cluster_key = chiplet["Chiplet"].split("-")[0]
        color = cluster_colors.get(cluster_key, "gray")

        plt.gca().add_patch(
            plt.Rectangle(
                (y, x), breadth, length, facecolor=color, alpha=0.8, edgecolor="black", linewidth=1.5
            )
        )
        plt.text(
            y + breadth / 2,
            x + length / 2,
            chiplet["Chiplet"],
            color="black",
            fontsize=8,
            ha="center",
            va="center",
        )

    # Add tick marks at every unit
    plt.xticks(ticks=np.arange(0, max_x + 2, 1))
    plt.yticks(ticks=np.arange(0, max_y + 2, 1))
    plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis (mm)")
    plt.ylabel("Y-axis (mm)")
    plt.savefig(f"exp_{i}/chiplet-placement.png")
    plt.close()

def main():
    grid = np.zeros(grid_dims, dtype=int)
    cluster_positions = {}
    permutations_list = list(permutations(clusters.keys()))
    top_designs = []
    i = 0  # Initialize visualization index

    # Iterate over permutations
    for perm in permutations_list:
        grid.fill(0)
        cluster_positions.clear()

        # Place chiplets
        for cluster in perm:
            cluster_positions[cluster] = place_chiplets_fixed(grid, cluster, clusters)

        # Generate floorplan and adjust spacing
        floorplan_data = generate_floorplan_data(cluster_positions, clusters)
        adjusted_floorplan = adjust_chiplets_with_spacing(floorplan_data)

        # Calculate score and append results
        score = score_design(adjusted_floorplan, grid_dims)
        top_designs.append((perm, score, adjusted_floorplan))

    # Sort and pick top 3 designs
    top_designs = sorted(top_designs, key=lambda x: x[1])[:3]

    # Visualize top designs
    for idx, (perm, score, adjusted_floorplan) in enumerate(top_designs, start=1):
        print(f"Design {idx}: Score = {score:.2f}, Permutation = {perm}")

        # Create directory for the current experiment
        exp_dir = f"exp_{idx}"
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Visualize chiplet floorplan and centers
        visualize_chiplet_floorplan(idx, adjusted_floorplan, clusters, spacing=0.25, title="Heterogenous Chiplet Placement")
        visualize_chiplet_centers(idx, adjusted_floorplan, spacing=0.25, title="Chiplet Centers Visualization")

if __name__ == "__main__":
    main()
