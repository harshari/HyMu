import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
import pandas as pd
from itertools import permutations 
from generate_mfit_floorplan import generate_power_config_file
import os

# Network details:
network_data = {
    "ResNet18": {
        "Storage": [9.19, 36.0, 36.0, 36.0, 36.0, 72.0, 144.0, 8.0, 144.0, 144.0, 288.0, 576.0, 32.0, 576.0, 576.0, 1152.0, 2304.0, 128.0],
        "Activations": [784.0, 196.0, 196.0, 196.0, 196.0, 98.0, 98.0, 98.0, 98.0, 98.0, 49.0, 49.0, 49.0, 49.0, 49.0, 24.5, 24.5, 24.5],
        "Compute": [118.01, 115.61, 115.61, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61, 57.8, 115.61, 6.42],
        "Sensitivity": [68.22, 0.86, 0.08, 12.62, 0.45, 0.33, 4.95, 0.30, 0.06, 4.31, 0.19, 0.15, 2.21, 0.16, 0.03, 4.93, 0.09, 0.06]
    },
    "ResNet34": {
        "Storage": [9.19, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 72.0, 144.0, 8.0, 144.0, 144.0, 144.0, 144.0, 144.0, 144.0, 288.0, 576.0, 32.0, 576.0, 576.0, 576.0, 576.0, 576.0, 576.0, 576.0, 576.0, 576.0, 576.0, 1152.0, 2304.0, 128.0, 2304.0, 2304.0],
        "Activations": [784.0, 196.0, 196.0, 196.0, 196.0, 196.0, 196.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 98.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 49.0, 24.5, 24.5, 24.5, 24.5, 24.5],
        "Compute": [118.01, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61],
        "Sensitivity": [1.15, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 9.00, 18.00, 1.00, 18.00, 18.00, 18.00, 18.00, 18.00, 18.00, 36.00, 72.00, 4.00, 72.00, 72.00, 72.00, 72.00, 72.00, 72.00, 72.00, 72.00, 72.00, 72.00, 144.00, 288.00, 16.00, 288.00, 288.00]
    },
    "VGG16": {
        "Storage": [1.75, 36.06, 72.12, 144.12, 288.25, 576.25, 576.25, 1152.50, 2304.50, 2304.50, 2304.50, 2304.50, 2304.50, 100356.00, 16388.00, 4000.98],
        "Activations": [3136.00, 3136.00, 1568.00, 1568.00, 784.00, 784.00, 784.00, 392.00, 392.00, 392.00, 98.00, 98.00, 98.00, 4.00, 4.00, 0.98],
        "Compute": [86.70, 1849.69, 924.84, 1849.69, 924.84, 1849.69, 1849.69, 924.84, 1849.69, 1849.69, 462.42, 462.42, 462.42, 102.76, 16.78, 4.10],
        "Sensitivity": [0.17, 0.14, 1.35, 0.15, 10.34, 0.17, 9.15, 0.14, 19.93, 0.15, 15.68, 0.12, 16.16, 0.10, 26.12, 0.14]
    },
    "VGG19": {
        "Storage": [1.75, 36.06, 72.12, 144.12, 288.25, 576.25, 576.25, 576.25, 1152.50, 2304.50, 2304.50, 2304.50, 2304.50, 2304.50, 2304.50, 2304.50, 100356.00, 16388.00, 4000.98],
        "Activations": [3136.00, 3136.00, 1568.00, 1568.00, 784.00, 784.00, 784.00, 784.00, 392.00, 392.00, 392.00, 392.00, 98.00, 98.00, 98.00, 98.00, 4.00, 4.00, 0.98],
        "Compute": [86.70, 1849.69, 924.84, 1849.69, 924.84, 1849.69, 1849.69, 1849.69, 924.84, 1849.69, 1849.69, 1849.69, 462.42, 462.42, 462.42, 462.42, 102.76, 16.78, 4.10],
        "Sensitivity": [0.09, 0.05, 0.62, 0.07, 3.74, 0.07, 5.47, 0.07, 13.70, 0.09, 15.69, 0.08, 12.66, 0.07, 10.98, 0.06, 20.71, 0.08, 15.71]
    },
    "MobileNetV2": {
        "Storage": [0.84, 0.28, 0.50, 1.50, 0.84, 2.25, 3.38, 1.27, 3.38, 3.38, 1.27, 4.50, 6.00, 1.69, 6.00, 6.00, 1.69, 6.00, 6.00, 1.69, 12.00, 24.00, 3.38, 24.00, 24.00, 3.38, 24.00, 24.00, 3.38, 24.00, 24.00, 3.38, 36.00, 54.00, 5.06, 54.00, 54.00, 5.06, 54.00, 54.00, 5.06, 90.00, 150.00, 8.44, 150.00, 150.00, 8.44, 150.00, 150.00, 8.44, 300.00, 400.00],
        "Activations": [392.00, 392.00, 196.00, 1176.00, 294.00, 73.50, 441.00, 441.00, 73.50, 441.00, 110.25, 24.50, 147.00, 147.00, 24.50, 147.00, 147.00, 24.50, 147.00, 36.75, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 18.38, 110.25, 110.25, 18.38, 110.25, 110.25, 18.38, 110.25, 27.56, 7.66, 45.94, 45.94, 7.66, 45.94, 45.94, 7.66, 45.94, 45.94, 15.31, 61.25],
        "Compute": [10.84, 115.61, 6.42, 19.27, 260.11, 7.23, 10.84, 585.25, 10.84, 10.84, 146.31, 3.61, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 65.03, 2.41, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 260.11, 7.23, 10.84, 585.25, 10.84, 10.84, 585.25, 10.84, 10.84, 146.31, 4.52, 7.53, 406.43, 7.53, 7.53, 406.43, 7.53, 7.53, 406.43, 15.05, 20.07],
        "Sensitivity": [1.81, 0.00, 0.16, 47.19, 0.02, 0.11, 0.17, 0.01, 0.00, 2.27, 0.00, 0.02, 21.64, 0.04, 0.01, 0.38, 0.06, 0.00, 1.15, 0.00, 0.01, 6.65, 0.01, 0.01, 0.11, 0.03, 0.00, 1.07, 0.00, 0.00, 9.69, 0.01, 0.00, 0.13, 0.02, 0.00, 0.50, 0.00, 0.00, 4.63, 0.00, 0.00, 0.02, 0.01, 0.00, 0.17, 0.00, 0.00, 1.89, 0.00, 0.00, 0.02]
    }
}

# Cluster configurations
clusters = {
    "Cluster 1": {"count": 4, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},  # Standard - 80mm2
    "Cluster 2": {"count": 10, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},  # Shared_ADC - 80mm2
    "Cluster 3": {"count": 19, "pd": 2, "area": 4, "memory": 108, "tops": 11e12, "energy_per_mac": .18e-12},  # Adder - 80mm2
    "Cluster 4": {"count": 13, "pd": 8, "area": 4, "memory": 2400, "tops": 35e12, "energy_per_mac": .22e-12},  # Accumulator
    "Cluster 5": {"count": 8, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},  # ADC_Less - 96
}
# Grid dimensions
grid_dims = (16, 16)

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

def find_max_distance_position(grid, chiplet_size, current_positions=None):
    '''
    Use: Finds a position farthest from the grid center for chiplet placement.
    For the heuristic placement to boundary
    '''
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

# Function to place chiplets in fixed positions for initial layout
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
def adjust_chiplets_with_spacing(floorplan_data, spacing=.25):
    adjusted_floorplan = []
    half_spacing = spacing / 2  # Distribute spacing equally on all sides

    for chiplet in floorplan_data:
        #type_chiplet = chiplet[""]
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        
        # Adjust position and dimensions for spacing
        adjusted_x = x + half_spacing * x
        adjusted_y = y + half_spacing * y
        adjusted_length = length
        adjusted_breadth = breadth
        centre_x = adjusted_x + adjusted_length/2
        centre_y = adjusted_y + adjusted_breadth/2

        adjusted_floorplan.append({
            "Chiplet": chiplet["Chiplet"],
            "Lower_Left_Corner": (adjusted_x, adjusted_y),
            "Length": adjusted_length,
            "Breadth": adjusted_breadth,
            "Center": (centre_x, centre_y),
            "neighbor_count": chiplet["num_neighbors"]
        })
    return adjusted_floorplan




def detect_grid_dimensions(core_ordering, clusters):
    """
    Automatically detect the grid dimensions (rows x columns) based on chiplet sizes and counts.
    """
    total_area = 0
    for chiplet_name in core_ordering:
        # Resolve cluster key from chiplet name (e.g., 'C4-1' -> 'Cluster 4')
        cluster_key = f"Cluster {chiplet_name.split('-')[0][1]}"
        if cluster_key in clusters:
            total_area += clusters[cluster_key]["area"]
        else:
            raise ValueError(f"Unknown cluster key: {cluster_key} for chiplet {chiplet_name}")

    # Assume each router represents an area of 4 (smallest chiplet area)
    total_routers = total_area // 4

    # Calculate dimensions (favor a slightly rectangular grid if possible)
    rows = int(total_routers ** 0.5)  # Start with a square approximation
    cols = (total_routers + rows - 1) // rows  # Ensure all chiplets fit in

    return rows, cols
def calculate_neighbors(floorplan_data):
    """
    Appends the number of neighbors for each chiplet to the floorplan data.
    Returns:
        pd.DataFrame: Updated DataFrame with an additional column 'num_neighbors'.
    Usage:         updated_floorplan = calculate_neighbors(floorplan_data)

    """
    def is_neighbor(chip1, chip2):
        # Check if chip1 and chip2 are adjacent
        adjacent_x = (chip1['Lower_Left_Corner'][0] + chip1['Length'] == chip2['Lower_Left_Corner'][0]) or (chip2['Lower_Left_Corner'][0] + chip2['Length'] == chip1['Lower_Left_Corner'][0])
        overlapping_y = not (chip1['Lower_Left_Corner'][1] + chip1['Breadth'] <= chip2['Lower_Left_Corner'][1] or chip2['Lower_Left_Corner'][1] + chip2['Breadth'] <= chip1['Lower_Left_Corner'][1])
        
        adjacent_y = (chip1['Lower_Left_Corner'][1] + chip1['Breadth'] == chip2['Lower_Left_Corner'][1]) or (chip2['Lower_Left_Corner'][1] + chip2['Breadth'] == chip1['Lower_Left_Corner'][1])
        overlapping_x = not (chip1['Lower_Left_Corner'][0] + chip1['Length'] <= chip2['Lower_Left_Corner'][0] or chip2['Lower_Left_Corner'][0] + chip2['Length'] <= chip1['Lower_Left_Corner'][0])
        
        return (adjacent_x and overlapping_y) or (adjacent_y and overlapping_x)
    
    for i, chip1 in enumerate(floorplan_data):
        count = 0
        for j, chip2 in enumerate(floorplan_data):
            if i != j and is_neighbor(chip1, chip2):
                count += 1
        floorplan_data[i]['num_neighbors'] = count
        
    return floorplan_data

def generate_core_ordering_with_grid(adjusted_floorplan_with_spacing, clusters):
    """
    Generate core ordering (1D array) and a 2D grid representing chiplet placement.
    Larger chiplets repeat in the grid to reflect their occupancy across multiple routers.
    """
    # Extract centers and chiplet names
    chiplet_positions = [
        (chiplet["Chiplet"], chiplet["Center"][0], chiplet["Center"][1])  # (Name, X, Y)
        for chiplet in adjusted_floorplan_with_spacing
    ]

    # Sort by top-left to bottom-right (row first, then column)
    chiplet_positions.sort(key=lambda x: (x[1], x[2]))  # Sort by Y, then X

    # Generate core ordering (1D array)
    core_ordering = [chiplet[0] for chiplet in chiplet_positions]

    # Detect grid dimensions dynamically
    rows, cols = detect_grid_dimensions(core_ordering, clusters)
    grid = np.zeros((rows, cols), dtype=object)  # Initialize 2D grid

    used_positions = set()  # Track used positions in the grid

    # Place chiplets in the grid
    for core_number, chiplet_name in enumerate(core_ordering, start=1):
        # Get chiplet info
        chiplet = next(c for c in adjusted_floorplan_with_spacing if c["Chiplet"] == chiplet_name)
        cluster_key = f"Cluster {chiplet_name.split('-')[0][1]}"
        area = clusters[cluster_key]["area"]

        # Find next available position in the grid
        for row in range(rows):
            for col in range(cols):
                if (row, col) in used_positions:
                    continue

                if area == 8:  # Large chiplet occupies 2 routers
                    if col + 1 < cols and (row, col + 1) not in used_positions:
                        # Place chiplet in two adjacent routers
                        grid[row, col] = f"{chiplet_name}-{core_number}"
                        grid[row, col + 1] = f"{chiplet_name}-{core_number}"
                        used_positions.add((row, col))
                        used_positions.add((row, col + 1))
                        break
                elif area == 4:  # Small chiplet occupies 1 router
                    # Place chiplet in a single router
                    grid[row, col] = f"{chiplet_name}-{core_number}"
                    used_positions.add((row, col))
                    break
            else:
                # Continue searching in the next row
                continue
            # Break outer loop if placement is done
            break

    return core_ordering, grid

def visualize_chiplet_centers(i, adjusted_floorplan_with_spacing, spacing=.25, title="Chiplet Centers Visualization"):
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
    plt.close()



# Visualization function with tick marks and equal axes scaling
def visualize_chiplet_floorplan(i, floorplan_data, clusters, spacing=0.25, title="Chiplet Placement with Tick Marks and Labels"):
    plt.figure(figsize=(12, 8))
    #plt.title(title, fontsize=14)
    
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
        color = cluster_colors.get(cluster_key, "gray")  # Default to gray if no match

        plt.gca().add_patch(
            plt.Rectangle(
                (y, x), breadth, length, facecolor=color, alpha=0.8, edgecolor="black", linewidth=1.5
            )
        )

    # Show only ticks at multiples of 10
    plt.xticks(ticks=np.arange(0, max_x + 2, 5), fontsize=18)
    plt.yticks(ticks=np.arange(0, max_y + 2, 5), fontsize=18)
    # plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)

    # Labels
    plt.xlabel("X dimension (mm)", fontsize=18)
    plt.ylabel("Y dimension (mm)", fontsize=18)

    plt.savefig(f"exp_{i}/chiplet-placement.png")
    plt.close()



# def visualize_chiplet_floorplan(i, floorplan_data, clusters, spacing=.25, title="Chiplet Placement with Tick Marks and Labels"):
#     plt.figure(figsize=(12, 8))
#     plt.title(title)
#     max_x = max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in floorplan_data)
#     max_y = max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in floorplan_data)
#     plt.xlim(0, max_x + spacing)
#     plt.ylim(0, max_y + spacing)

#     # Ensure equal scaling for axes
#     plt.gca().set_aspect('equal', adjustable='box')

#     # Cluster-specific colors
#     cluster_colors = {
#         "C1": "red",
#         "C2": "orange",
#         "C3": "purple",
#         "C4": "blue",
#         "C5": "green"
#     }

#     for chiplet in floorplan_data:
#         x, y = chiplet["Lower_Left_Corner"]
#         length = chiplet["Length"]
#         breadth = chiplet["Breadth"]
#         cluster_key = chiplet["Chiplet"].split("-")[0]
#         color = cluster_colors[cluster_key]

#         plt.gca().add_patch(
#             plt.Rectangle(
#                 (y, x), breadth, length, facecolor=color, alpha=0.8, edgecolor="black", linewidth=1.5
#             )
#         )
#         plt.text(
#             y + breadth / 2,
#             x + length / 2,
#             chiplet["Chiplet"],
#             color="black",
#             fontsize=8,
#             ha="center",
#             va="center",
#         )

#     # Add tick marks at every unit
#     plt.xticks(ticks=np.arange(0, max_x + 2, 1))
#     plt.yticks(ticks=np.arange(0, max_y + 2, 1))
#     plt.grid(visible=True, which="both", color="gray", linestyle="--", linewidth=0.5)
#     plt.xlabel("X-axis (mm)")
#     plt.ylabel("Y-axis (mm)")
#     plt.savefig(f"exp_{i}/chiplet-placement.png")
#     plt.close()


# Initialize the grid and cluster positions
grid = np.zeros((grid_dims[0], grid_dims[1]), dtype=int)
cluster_positions = {}

# Place clusters based on ordering
ordering = ["Cluster 4", "Cluster 3", "Cluster 2", "Cluster 1", "Cluster 5"]
# ordering = ["Cluster 3", "Cluster 1", "Cluster 2", "Cluster 4"]
#ordering = ["Cluster 1", "Cluster 3", "Cluster 2", "Cluster 4"]


permutations = list(permutations(ordering))
exp_number = 0
peak_temp = 100000 
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

        print("Linking to MFIT here")

        # Generate floorplan data
        floorplan_data = generate_floorplan_data(cluster_positions, clusters)
        updated_floorplan = calculate_neighbors(floorplan_data)
        # Adjust floorplan data with spacing
        adjusted_floorplan_with_spacing = adjust_chiplets_with_spacing(floorplan_data)

        # # Generate core ordering and grid
        core_ordering, core_grid = generate_core_ordering_with_grid(adjusted_floorplan_with_spacing, clusters)

        # # Save 1D core ordering to CSV
        core_ordering_df = pd.DataFrame({"Core Ordering": core_ordering})
        core_ordering_df.to_csv(f"exp_{i}/core_ordering.csv", index=False)

        # # Save 2D grid to CSV
        core_grid_df = pd.DataFrame(core_grid)
        core_grid_df.to_csv(f"exp_{i}/core_grid.csv", index=False)


        # # Optionally save to a CSV file
        # # router_array_df = pd.DataFrame(router_array)
        # # router_array_df.to_csv(f"exp_{i}/router_array.csv", index=False)
        
        # ### Create NoI here. From the center of each chiplet
        # ### First for the given floorplan, create grid visualization
        # ### then initialize 2d matrix of links and 1d matrix for core ordering
        # # Visualize chiplet floorplan

        visualize_chiplet_floorplan(i, adjusted_floorplan_with_spacing, clusters, spacing=.25, title="Heterogenous Chiplet Placement")
        visualize_chiplet_centers(i, adjusted_floorplan_with_spacing=adjusted_floorplan_with_spacing)
        # print(f"Average Hop Count: {average_hop_count:.2f}")

        # ## Send to MFIT
        # T_peak = generate_power_config_file(adjusted_floorplan_with_spacing, clusters, i)
        # # # Calculate average hop count
        # if (T_peak < peak_temp):
        #     peak_temp = T_peak
        #     exp_number = i
        #     print(f"New minimum temp = {(T_peak - 300):.2f} at exp {i}")
        # # # Convert to DataFrame for output
        # adjusted_floorplan_spacing_df = pd.DataFrame(adjusted_floorplan_with_spacing)

        # # # Uncomment the following line to save the floorplan data to a CSV file
        # adjusted_floorplan_spacing_df.to_csv(f"exp_{i}/chiplet-position_flp.csv", index=False)

    else:
        print("Next permutation")
print(f"Lowest T_peak for {exp_number} at T_peak = {T_peak}")