import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import permutations
from scipy.spatial.distance import cityblock

# Define chiplet clusters and tier configurations
clusters_surface = {
    "Cluster 1": {"count": 24, "area": 4, "pd": 8},
    "Cluster 2": {"count": 28, "area": 8, "pd": 8},
    "Cluster 3": {"count": 0, "area": 4, "pd": 2},
    "Cluster 4": {"count": 18, "area": 4, "pd": 8},
    "Cluster 5": {"count": 12, "area": 4, "pd": 1}
}

clusters_embedded = {
    "Cluster 1": {"count": 0, "area": 4, "pd": 8},
    "Cluster 2": {"count": 12, "area": 8, "pd": 8},
    "Cluster 3": {"count": 0, "area": 4, "pd": 2},
    "Cluster 4": {"count": 12, "area": 4, "pd": 8},
    "Cluster 5": {"count": 12, "area": 4, "pd": 1}
}

tiers = {
    "Tier_0": {"grid_dims": (20, 22), "spacing": 0.25, "clusters": clusters_surface},
    "Tier_1": {"grid_dims": (20, 20), "spacing": 0.25, "clusters": clusters_embedded}
}

# Helper Functions
def get_chiplet_size(cluster_metadata):
    return (4, 2) if cluster_metadata["area"] == 8 else (2, 2)

def get_random_position(grid, chiplet_size):
    rows, cols = grid.shape
    for _ in range(100):  # Try 100 random positions
        x, y = np.random.randint(0, rows), np.random.randint(0, cols)
        if is_valid_position(grid, x, y, chiplet_size):
            return x, y
    return None

def place_chiplets_randomly(grid, cluster_key, cluster_config, max_attempts=100, debug=False):
    """
    Randomly place chiplets for a given cluster within the grid, with debugging output.

    Parameters:
    - grid: 2D numpy array representing the grid.
    - cluster_key: Key representing the cluster (e.g., "Cluster 1").
    - cluster_config: Dictionary with cluster configuration (e.g., area, count, etc.).
    - max_attempts: Maximum number of retries for placement.
    - debug: If True, outputs the grid and placement attempts for debugging.

    Returns:
    - positions: List of placed chiplet positions.
    """
    chiplet_area = cluster_config["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)  # Adjust based on area
    positions = []
    chiplets_remaining = cluster_config["count"]
    rows, cols = grid.shape

    attempts = 0

    while chiplets_remaining > 0:
        for _ in range(max_attempts):
            # Generate a random position within grid bounds
            x = np.random.randint(0, rows - chiplet_size[0] + 1)
            y = np.random.randint(0, cols - chiplet_size[1] + 1)

            # Check if the position is valid
            if is_valid_position(grid, x, y, chiplet_size):
                # Place the chiplet
                positions.append((x, y))
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
                chiplets_remaining -= 1
                break
            attempts += 1
        else:
            # If no valid position is found after max_attempts
            if debug:
                print(f"Debug: Unable to place chiplet after {max_attempts} attempts for {cluster_key}.")
                print(f"Remaining chiplets: {chiplets_remaining}")
                visualize_grid(grid, f"{cluster_key}_debug")
            raise ValueError(f"Unable to place chiplets for {cluster_key}. Remaining: {chiplets_remaining}")

    if debug:
        print(f"Chiplets placed for {cluster_key} after {attempts} attempts.")
        visualize_grid(grid, f"{cluster_key}_final")

    return positions


def visualize_grid(grid, title):
    """
    Visualize the current state of the grid.

    Parameters:
    - grid: 2D numpy array representing the grid.
    - title: Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="viridis", interpolation="none")
    plt.colorbar(label="Cluster ID")
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()



def is_valid_position(grid, x, y, chiplet_size):
    rows, cols = grid.shape
    if x + chiplet_size[0] > rows or y + chiplet_size[1] > cols:
        return False
    for dx in range(chiplet_size[0]):
        for dy in range(chiplet_size[1]):
            if grid[x + dx, y + dy] != 0:
                return False
    return True

def place_chiplet(grid, placements, cluster, position, chiplet_size):
    x, y = position
    placements.append({
        "Chiplet": cluster,
        "Lower_Left_Corner": (x, y),
        "Length": chiplet_size[0],
        "Breadth": chiplet_size[1]
    })
    for dx in range(chiplet_size[0]):
        for dy in range(chiplet_size[1]):
            grid[x + dx, y + dy] = 1

# Random Placement Generation
def generate_random_placement(clusters, grid_dims, max_attempts=1000):
    """
    Generate a random placement for chiplets with retries.

    Parameters:
    - clusters: Dictionary of cluster configurations.
    - grid_dims: Tuple (rows, cols) for the grid dimensions.
    - max_attempts: Maximum number of attempts to generate a valid placement.

    Returns:
    - floorplan_data: List of placed chiplets.
    """
    for attempt in range(max_attempts):
        try:
            grid = np.zeros(grid_dims, dtype=int)  # Reset the grid for each attempt
            cluster_positions = {}

            # Place each cluster's chiplets randomly
            for cluster_key, cluster_config in clusters.items():
                cluster_positions[cluster_key] = place_chiplets_randomly(grid, cluster_key, cluster_config)

            # Generate floorplan data from placed positions
            return generate_floorplan_data(cluster_positions, clusters)

        except ValueError as e:
            # Log the failed attempt for debugging (optional)
            print(f"Attempt {attempt + 1} failed: {e}")

    # If all attempts fail, raise an exception
    raise ValueError(f"Unable to place all chiplets after {max_attempts} attempts.")

def generate_robust_random_placement(clusters, grid_dims, max_iterations=10000):
    """
    Robustly generate a random placement for chiplets in the given grid dimensions.
    
    Parameters:
    - clusters: Dictionary of clusters with chiplet metadata.
    - grid_dims: Tuple of (rows, cols) for grid dimensions.
    - max_iterations: Maximum attempts to place all chiplets.

    Returns:
    - floorplan_data: List of placed chiplets with positions and metadata.
    """
    grid = np.zeros(grid_dims, dtype=int)
    cluster_positions = {}

    for cluster_key, cluster_config in clusters.items():
        chiplet_size = (4, 2) if cluster_config["area"] == 8 else (2, 2)
        cluster_positions[cluster_key] = []
        chiplets_remaining = cluster_config["count"]

        for _ in range(max_iterations):
            x = np.random.randint(0, grid_dims[0] - chiplet_size[0] + 1)
            y = np.random.randint(0, grid_dims[1] - chiplet_size[1] + 1)
            
            if is_valid_position(grid, x, y, chiplet_size):
                # Place chiplet on grid
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[x + dx, y + dy] = 1  # Mark as occupied

                cluster_positions[cluster_key].append((x, y))
                chiplets_remaining -= 1

                if chiplets_remaining == 0:
                    break

        if chiplets_remaining > 0:
            raise ValueError(f"Unable to place all chiplets for cluster {cluster_key}.")

    floorplan_data = generate_floorplan_data(cluster_positions, clusters, embed=False)
    return loorplan_data, grid

# DRC Evaluation
def evaluate_drcs(floorplan_data, spacing_threshold):
    overlaps = []
    spacing_violations = []

    for i, chip1 in enumerate(floorplan_data):
        for j, chip2 in enumerate(floorplan_data):
            if i >= j:
                continue

            if check_overlap(chip1, chip2):
                overlaps.append((chip1["Chiplet"], chip2["Chiplet"]))

            if check_spacing(chip1, chip2, spacing_threshold):
                spacing_violations.append((chip1["Chiplet"], chip2["Chiplet"]))

    return {"Overlaps": overlaps, "Spacing Violations": spacing_violations}

def check_overlap(chip1, chip2):
    x1_min, y1_min = chip1["Lower_Left_Corner"]
    x1_max = x1_min + chip1["Length"]
    y1_max = y1_min + chip1["Breadth"]

    x2_min, y2_min = chip2["Lower_Left_Corner"]
    x2_max = x2_min + chip2["Length"]
    y2_max = y2_min + chip2["Breadth"]

    return not (x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min)

def check_spacing(chip1, chip2, threshold):
    x1_min, y1_min = chip1["Lower_Left_Corner"]
    x1_max = x1_min + chip1["Length"]
    y1_max = y1_min + chip1["Breadth"]

    x2_min, y2_min = chip2["Lower_Left_Corner"]
    x2_max = x2_min + chip2["Length"]
    y2_max = y2_min + chip2["Breadth"]

    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)

    return (dx < threshold and dy < threshold)

# Visualization
def visualize_tiers(floorplan_data, iteration):
    for tier_name, chiplets in floorplan_data.items():
        plt.figure(figsize=(8, 8))
        plt.title(f"{tier_name} Placement - Iteration {iteration}")

        for chiplet in chiplets:
            x, y = chiplet["Lower_Left_Corner"]
            length = chiplet["Length"]
            breadth = chiplet["Breadth"]

            plt.gca().add_patch(plt.Rectangle((y, x), breadth, length, edgecolor='black', facecolor='blue', alpha=0.5))
            plt.text(y + breadth / 2, x + length / 2, chiplet["Chiplet"], ha='center', va='center')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, tiers[tier_name]["grid_dims"][1])
        plt.ylim(0, tiers[tier_name]["grid_dims"][0])
        plt.gca().invert_yaxis()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(f"{tier_name}_placement_{iteration}.png")
        plt.close()

def visualize_debug_placement(grid, grid_dims, filename):
    """
    Visualize the grid to debug placement issues.
    
    Parameters:
    - grid: The grid with chiplet placements.
    - grid_dims: Tuple of (rows, cols) for grid dimensions.
    - filename: Output file for the plot.
    """
    if grid is None:
        print("Debug visualization skipped: No grid data available.")
        return

    plt.figure(figsize=(12, 8))
    plt.title("Debug Placement Visualization")
    plt.imshow(grid, cmap="viridis", origin="upper")
    plt.colorbar(label="Occupancy")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



# Main Execution
def main():
    for iteration in range(10):
        floorplan_data = {}
        grid = None
        for tier_name, tier_config in tiers.items():
            try:
                floorplan_data[tier_name], grid = generate_robust_random_placement(
                    tier_config["clusters"],
                    tier_config["grid_dims"]
                )
                if floorplan_data is None:
                    raise ValueError(f"Unable to place all chiplets for tier {tier_name}.")
            except ValueError as e:
                print(f"Placement failed for {tier_name}: {e}")
                visualize_debug_placement(grid, tier_config["grid_dims"], f"debug_{tier_name}.png")
                continue
        drc_results = evaluate_drcs(floorplan_data["Tier_0"], spacing_threshold=0.25)
        print(f"Iteration {iteration} DRC Results: {drc_results}")

        visualize_tiers(floorplan_data, iteration)

if __name__ == "__main__":
    main()
