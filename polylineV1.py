import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
import pandas as pd
from itertools import permutations 
from generate_mfit_floorplan import generate_power_config_file
import os

# Cluster configurations
clusters = {
    "Cluster 1": {"count": 28, "pd": 8, "area": 8},  # 2x4 or 4x2 chiplets
    "Cluster 2": {"count": 12, "pd": 1, "area": 4},  # 2x2 chiplets
    "Cluster 3": {"count": 18, "pd": 4, "area": 4},  # 2x2 chiplets
    "Cluster 4": {"count": 24, "pd": 8, "area": 4},  # 2x2 chiplets
}

# Grid dimensions
grid_dims = (20, 22)

# Function to validate chiplet placement
def is_valid_position(grid, x, y, chiplet_size):
    rows, cols = grid.shape
    if x < 0 or y < 0 or x + chiplet_size[0] > rows or y + chiplet_size[1] > cols:
        # Log invalid position for debugging
        # print(f"Invalid position: x={x}, y={y}, size={chiplet_size}, grid={rows}x{cols}")
        return False
    for dx in range(chiplet_size[0]):
        for dy in range(chiplet_size[1]):
            if grid[x + dx, y + dy] != 0:
                return False
    return True

def find_max_distance_position(grid, chiplet_size, current_positions=None):
    rows, cols = grid.shape
    center_x, center_y = rows // 2, cols // 2
    max_distance = -1
    best_pos = None

    for x in range(rows):
        for y in range(cols):
            if is_valid_position(grid, x, y, chiplet_size):
                dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                if dist > max_distance:
                    max_distance = dist
                    best_pos = (x, y)

    if best_pos is None:
        print("No valid position found. Check chiplet size and grid dimensions.")
    return best_pos

def place_chiplets_fixed(grid, cluster_key, clusters):
    chiplet_area = clusters[cluster_key]["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]

    while chiplets_remaining > 0:
        next_pos = find_max_distance_position(grid, chiplet_size, positions)
        
        if next_pos is None:
            print(f"Unable to place all chiplets for {cluster_key}. Remaining: {chiplets_remaining}")
            return positions  # Return the positions already placed
            
        x, y = next_pos
        positions.append((x, y))

        for dx in range(chiplet_size[0]):
            for dy in range(chiplet_size[1]):
                grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
                
        chiplets_remaining -= 1

    return positions

# # Function to place chiplets in fixed positions for initial layout
# def place_chiplets_fixed(grid, cluster_key, clusters):
#     chiplet_area = clusters[cluster_key]["area"]
#     chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
#     positions = []
#     chiplets_remaining = clusters[cluster_key]["count"]

#     # Start placement from the top-left corner
#     for x in range(0, grid_dims[0], chiplet_size[0]):
#         for y in range(0, grid_dims[1], chiplet_size[1]):
#             if chiplets_remaining > 0 and is_valid_position(grid, x, y, chiplet_size):
#                 positions.append((x, y))
#                 for dx in range(chiplet_size[0]):
#                     for dy in range(chiplet_size[1]):
#                         grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000
#                 chiplets_remaining -= 1
#     return positions

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

def initialize_core_and_link_placement(adjusted_floorplan_with_spacing):
    # Core placement ordering
    core_placement_ordering = np.zeros((len(adjusted_floorplan_with_spacing), len(adjusted_floorplan_with_spacing)))
    
    # Link placement ordering
    link_placement_ordering = np.zeros((len(adjusted_floorplan_with_spacing), len(adjusted_floorplan_with_spacing)))
    
    return core_placement_ordering, link_placement_ordering

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
        # Adjust for mirroring logic
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        center_x, center_y = chiplet["Center"]
        
        # Plot the center as a dot
        plt.plot(center_y, center_x, 'ro')  # Adjusted coordinates to match mirroring
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
    plt.show()

# Visualization function with tick marks and equal axes scaling
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
    plt.savefig(f"exp_{i}/chiplet-placement.png")

# Initialize the grid and cluster positions
grid = np.zeros((grid_dims[0], grid_dims[1]), dtype=int)
cluster_positions = {}

# Place clusters based on ordering
ordering = ["Cluster 4", "Cluster 3", "Cluster 2", "Cluster 1"]
# ordering = ["Cluster 3", "Cluster 1", "Cluster 2", "Cluster 4"]
#ordering = ["Cluster 1", "Cluster 3", "Cluster 2", "Cluster 4"]


permutations = list(permutations(ordering))

i = 0
for iterate in permutations:
    i += 1
    grid = np.zeros((grid_dims[0], grid_dims[1]), dtype=int)
    cluster_positions = {}
    for cluster in iterate:
        cluster_positions[cluster] = place_chiplets_fixed(grid, cluster, clusters)
        
    if np.all(grid):
        if os.path.exists(f"exp_{i}"):
            os.system(f"rm -rf exp_{i}")
        
        os.system(f"mkdir exp_{i}")

        # Generate floorplan data
        floorplan_data = generate_floorplan_data(cluster_positions, clusters)
        # Adjust floorplan data with spacing
        adjusted_floorplan_with_spacing = adjust_chiplets_with_spacing(floorplan_data)
        ### Create NoI here. From the center of each chiplet
        ### First for the given floorplan, create grid visualization
        ### then initialize 2d matrix of links and 1d matrix for core ordering
        # Visualize chiplet floorplan

        visualize_chiplet_floorplan(i, adjusted_floorplan_with_spacing, clusters, spacing=0.25, title="Heterogenous Chiplet Placement")
        visualize_chiplet_centers(i, adjusted_floorplan_with_spacing=adjusted_floorplan_with_spacing)
        # Initialize core and link placement ordering
        core_placement_ordering, link_placement_ordering = initialize_core_and_link_placement(adjusted_floorplan_with_spacing)

        # Print shapes
        print("Core Placement Ordering Shape:", core_placement_ordering.shape)
        print("Link Placement Ordering Shape:", link_placement_ordering.shape)
            
        # generate_power_config_file(adjusted_floorplan_with_spacing, clusters, i)
        # Calculate average hop count
        average_hop_count = calculate_average_hop_count(cluster_positions)
        print(f"Average Hop Count: {average_hop_count:.2f}")

        # Convert to DataFrame for output
        #adjusted_floorplan_spacing_df = pd.DataFrame(adjusted_floorplan_with_spacing)

        # Uncomment the following line to save the floorplan data to a CSV file
        #adjusted_floorplan_spacing_df.to_csv(f"exp_{i}/chiplet-position_flp.csv", index=False)

        # Visualize chiplet placement with tick marks and labeled clusters
        print(f"Linking to MFIT here for exp_{i} folder")
    else:
        print("Next permutation")