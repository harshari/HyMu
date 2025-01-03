import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
import pandas as pd

# Cluster configurations
clusters = {
    "Cluster 1": {"count": 28, "pd": 8, "area": 8},  # 2x4 or 4x2 chiplets
    "Cluster 2": {"count": 12, "pd": 1, "area": 4},  # 2x2 chiplets
    "Cluster 3": {"count": 18, "pd": 4, "area": 4},  # 2x2 chiplets
    "Cluster 4": {"count": 24, "pd": 28, "area": 4},  # 2x2 chiplets
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

# Function to place chiplets in fixed positions for initial layout
def place_chiplets_fixed(grid, cluster_key, clusters):
    chiplet_area = clusters[cluster_key]["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]

    # Start placement from the top-left corner
    for x in range(0, grid_dims[0], chiplet_size[0]):
        for y in range(0, grid_dims[1], chiplet_size[1]):
            if chiplets_remaining > 0 and is_valid_position(grid, x, y, chiplet_size):
                positions.append((x, y))
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
                chiplets_remaining -= 1

    return positions

# Function to generate floorplan data for chiplets
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
                "Breadth": chiplet_size[1]
            })
    return floorplan_data

# Function to adjust positions and sizes with spacing
def adjust_chiplets_with_spacing(floorplan_data, spacing=0.25):
    adjusted_floorplan = []
    half_spacing = spacing / 2  # Distribute spacing equally on all sides

    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        
        # Adjust position and dimensions for spacing
        adjusted_x = x + half_spacing * x
        adjusted_y = y + half_spacing * y
        adjusted_length = length
        adjusted_breadth = breadth

        adjusted_floorplan.append({
            "Chiplet": chiplet["Chiplet"],
            "Lower_Left_Corner": (adjusted_x, adjusted_y),
            "Length": adjusted_length,
            "Breadth": adjusted_breadth
        })
    return adjusted_floorplan

# Function to calculate average hop count using Manhattan distance
def calculate_average_hop_count(cluster_positions):
    total_distance = 0
    total_pairs = 0

    for cluster_a, positions_a in cluster_positions.items():
        for cluster_b, positions_b in cluster_positions.items():
            if cluster_a != cluster_b:  # Calculate only between different clusters
                for pos_a in positions_a:
                    for pos_b in positions_b:
                        total_distance += cityblock(pos_a, pos_b)
                        total_pairs += 1

    return total_distance / total_pairs if total_pairs > 0 else None

# Visualization function with tick marks and equal axes scaling
def visualize_chiplets_with_ticks_and_labels(floorplan_data, clusters, spacing=0.25, title="Chiplet Placement with Tick Marks and Labels"):
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
        "C1": "orange",
        "C2": "green",
        "C3": "blue",
        "C4": "red",
    }

    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        cluster_key = chiplet["Chiplet"].split("-")[0]
        color = cluster_colors[cluster_key]

        plt.gca().add_patch(
            plt.Rectangle(
                (y, x), breadth, length, color=color, alpha=0.8, edgecolor="black", linewidth=1.5
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
    plt.show()

# Initialize the grid and cluster positions
grid = np.zeros((grid_dims[0] + 1, grid_dims[1] + 1), dtype=int)
cluster_positions = {}

# Place clusters based on ordering
ordering = ["Cluster 1", "Cluster 3", "Cluster 4", "Cluster 2"]
for cluster in ordering:
    cluster_positions[cluster] = place_chiplets_fixed(grid, cluster, clusters)

# Generate floorplan data
floorplan_data = generate_floorplan_data(cluster_positions, clusters)

# Adjust floorplan data with spacing
adjusted_floorplan_with_spacing = adjust_chiplets_with_spacing(floorplan_data)

# Calculate average hop count
average_hop_count = calculate_average_hop_count(cluster_positions)
print(f"Average Hop Count: {average_hop_count:.2f}")

# Convert to DataFrame for output
adjusted_floorplan_spacing_df = pd.DataFrame(adjusted_floorplan_with_spacing)

# Uncomment the following line to save the floorplan data to a CSV file
adjusted_floorplan_spacing_df.to_csv("chiplet-position_flp.csv", index=False)

# Visualize chiplet placement with tick marks and labeled clusters
visualize_chiplets_with_ticks_and_labels(adjusted_floorplan_with_spacing, clusters, spacing=0.25, title="Heterogenous Chiplet Placement")