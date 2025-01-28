import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
import pandas as pd
from itertools import permutations 
from generate_mfit_floorplan import generate_power_config_file
import os
from heapq import heappush, heappop
import copy

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
    "Cluster 1": {"count": 24, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},  # Standard - 80mm2
    "Cluster 2": {"count": 28, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},  # Shared_ADC - 80mm2
    "Cluster 3": {"count": 0, "pd": 2, "area": 4, "memory": 108, "tops": 11e12, "energy_per_mac": .18e-12},  # Adder - 80mm2
    "Cluster 4": {"count": 18, "pd": 8, "area": 4, "memory": 2400, "tops": 35e12, "energy_per_mac": .22e-12},  # Accumulator
    "Cluster 5": {"count": 12, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},  # ADC_Less - 96
}
# Grid dimensions
grid_dims = (20, 22)

# Communication matrix defining inter-cluster communication weights
workload_dict = {
    ("C1", "C2"): 5,
    ("C1", "C3"): 3,
    ("C1", "C4"): 1,
    ("C1", "C5"): 0,
    ("C2", "C1"): 5,
    ("C2", "C3"): 0,
    ("C2", "C4"): 5,
    ("C2", "C5"): 1,
    ("C3", "C1"): 3,
    ("C3", "C2"): 0,
    ("C3", "C4"): 2,
    ("C3", "C5"): 4,
    ("C4", "C1"): 1,
    ("C4", "C2"): 5,
    ("C4", "C3"): 2,
    ("C4", "C5"): 3,
    ("C5", "C1"): 0,
    ("C5", "C2"): 1,
    ("C5", "C3"): 4,
    ("C5", "C4"): 3,
}

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
def place_chiplets_fixed(grid, cluster_key, clusters, random_seed=None):
    chiplet_area = clusters[cluster_key]["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]

    # Use a random seed for reproducibility (optional)
    if random_seed is not None:
        np.random.seed(random_seed)

    while chiplets_remaining > 0:
        # Add some randomness to tie-breaking in position selection
        next_pos = find_max_distance_position(grid, chiplet_size)
        if next_pos is None:
            print(f"Unable to place all chiplets for {cluster_key}. Remaining: {chiplets_remaining}")
            return positions

        x, y = next_pos
        positions.append((x, y))

        # Mark the grid as occupied
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

# Chip warpage scoring
def evaluate_chip_warpage(floorplan_data, grid_dims):
    """
    Calculate the normalized warpage score based on the distance of large chiplets (8mm²)
    from the center of the grid. Normalized score ranges from 0 (best) to 1 (worst).

    Args:
        floorplan_data (list): List of chiplet details (dicts).
        grid_dims (tuple): Dimensions of the grid (rows, columns).

    Returns:
        float: Normalized warpage score.
    """
    # Define the grid center
    grid_center = (grid_dims[0] / 2, grid_dims[1] / 2)

    # Calculate the maximum possible distance to the center (diagonal distance to corner)
    max_distance = ((grid_dims[0] / 2)**2 + (grid_dims[1] / 2)**2)**0.5

    total_distance = 0
    large_chiplet_count = 0

    # Iterate through chiplets in the floorplan
    for chiplet in floorplan_data:
        # Check if the chiplet is large (8mm²)
        if chiplet["Length"] == 4 and chiplet["Breadth"] == 2:
            large_chiplet_count += 1
            # Compute the chiplet's center coordinates
            chiplet_center = (
                chiplet["Lower_Left_Corner"][0] + chiplet["Length"] / 2,
                chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] / 2
            )
            # Compute the Euclidean distance to the grid center
            distance = ((chiplet_center[0] - grid_center[0])**2 +
                        (chiplet_center[1] - grid_center[1])**2)**0.5
            total_distance += distance

    # If no large chiplets are placed, return a penalty score of 1 (worst-case)
    if large_chiplet_count == 0:
        return 1.0

    # Compute the average distance
    avg_distance = total_distance / large_chiplet_count

    # Normalize the average distance by the maximum possible distance
    normalized_score = avg_distance / max_distance

    # Ensure the normalized score is in range [0, 1]
    normalized_score = min(max(normalized_score, 0), 1)
    
    return normalized_score

# Intra-cluster continuity scoring
def evaluate_intra_cluster_continuity(floorplan_data, grid_dims, penalty_per_component=10, lonely_penalty=20, normalize=True):
    """
    Evaluates intra-cluster continuity using a graph-based approach.
    - If chiplets are disconnected into multiple sub-clusters, apply a penalty per disconnected component.
    - Lonely chiplets (with no neighbors) receive an additional penalty.
    - Supports normalization of the score between 0 (ideal) and 1 (worst case).
    
    Parameters:
        floorplan_data (list): List of chiplets with their properties.
        grid_dims (tuple): Dimensions of the grid (rows, columns).
        penalty_per_component (int): Penalty for each disconnected component.
        lonely_penalty (int): Penalty for lonely chiplets with no neighbors.
        normalize (bool): Whether to normalize the score between 0 and 1.

    Returns:
        total_score (float): The overall intra-cluster continuity score (raw or normalized).
    """

    # Initialize total score
    total_score = 0

    # Group chiplets by cluster
    clusters = {}
    for chiplet in floorplan_data:
        cluster = chiplet["Chiplet"].split("-")[0]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(chiplet["Lower_Left_Corner"])

    # Evaluate each cluster's connectivity
    max_score = 0
    for cluster_key, positions in clusters.items():
        # Create a graph representation of the cluster
        graph = {}
        for i, pos in enumerate(positions):
            graph[i] = []  # Initialize neighbors for chiplet `i`
            for j, other_pos in enumerate(positions):
                if i != j and is_adjacent(pos, other_pos):  # Check adjacency
                    graph[i].append(j)

        # Find connected components using a DFS
        visited = set()
        connected_components = []

        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in graph.keys():
            if node not in visited:
                component = []
                dfs(node, component)
                connected_components.append(component)

        # Calculate penalties for the cluster
        num_components = len(connected_components)
        cluster_penalty = (num_components - 1) * penalty_per_component

        # Additional penalty for lonely chiplets (size-1 components)
        lonely_penalty_count = sum(1 for component in connected_components if len(component) == 1)
        cluster_penalty += lonely_penalty_count * lonely_penalty

        # Add the cluster's penalty to the total score
        total_score += cluster_penalty

        # Calculate the maximum possible penalty for this cluster
        num_chiplets = len(positions)
        max_cluster_penalty = ((num_chiplets - 1) * penalty_per_component) + (num_chiplets * lonely_penalty)
        max_score += max_cluster_penalty

    # Normalize the score if required
    if normalize:
        normalized_score = total_score / max_score if max_score > 0 else 0.0
        return normalized_score

    return total_score

def is_adjacent(pos1, pos2):
    """
    Determines if two chiplets are adjacent (share an edge).
    """
    adjacent_x = (pos1[0] == pos2[0] and abs(pos1[1] - pos2[1]) == 1)
    adjacent_y = (pos1[1] == pos2[1] and abs(pos1[0] - pos2[0]) == 1)
    return adjacent_x or adjacent_y

# Inter-cluster hop count scoring
def evaluate_inter_cluster_hop_count(floorplan_data, workload_dict, grid_dims):
    """
    Evaluate the inter-cluster communication cost (hop count).

    Parameters:
    - floorplan_data (list of dict): A list of chiplet dictionaries containing:
        - "Chiplet": The name of the chiplet (e.g., "C1-1").
        - "Lower_Left_Corner": The (x, y) position of the chiplet in the grid.
    - workload_dict (dict): A dictionary mapping cluster pairs (e.g., ("C1", "C2")) 
      to their communication weight (e.g., {("C1", "C2"): 5}).
      Higher weights indicate more frequent communication between clusters.
    - grid_dims (tuple): Dimensions of the grid (rows, columns).
    - normalize (bool): Whether to normalize the score between 0 and 1. Default is False.

    Returns:
    - float: If `normalize` is False, returns the raw inter-cluster communication score.
             If `normalize` is True, returns the normalized score between 0 and 1.

    Explanation of Functionality:
    - This function evaluates the inter-cluster communication penalty based on the Manhattan 
      distance (cityblock distance) between chiplets of different clusters.
    - Communication weights (`workload_dict`) are used to assign higher penalties for clusters
      that communicate more frequently.
    - Normalization ensures the score is scaled between 0 (ideal placement) and 1 (worst-case 
      placement based on the maximum possible penalty for the given grid and workload).

    Usage Example:
    --------------
    floorplan_data = [
        {"Chiplet": "C1-1", "Lower_Left_Corner": (0, 0)},
        {"Chiplet": "C1-2", "Lower_Left_Corner": (0, 1)},
        {"Chiplet": "C2-1", "Lower_Left_Corner": (9, 9)},
    ]

    workload_dict = {
        ("C1", "C2"): 5,
    }

    grid_dims = (10, 10)

    # Raw score
    raw_score = evaluate_inter_cluster_hop_count(floorplan_data, workload_dict, grid_dims, normalize=False)

    # Normalized score
    normalized_score = evaluate_inter_cluster_hop_count(floorplan_data, workload_dict, grid_dims, normalize=True)
    """

    score = 0
    cluster_positions = {}

    # Group chiplets by cluster
    for chiplet in floorplan_data:
        cluster = chiplet["Chiplet"].split("-")[0]  # Extract cluster (e.g., "C1")
        if cluster not in cluster_positions:
            cluster_positions[cluster] = []
        cluster_positions[cluster].append(chiplet["Lower_Left_Corner"])

    # Calculate the actual score
    for (cluster_a, cluster_b), weight in workload_dict.items():
        if cluster_a in cluster_positions and cluster_b in cluster_positions:
            for pos_a in cluster_positions[cluster_a]:
                for pos_b in cluster_positions[cluster_b]:
                    # Use Manhattan distance (cityblock) for penalty
                    score += weight * cityblock(pos_a, pos_b)

    # Calculate the maximum possible score
    grid_rows, grid_cols = grid_dims
    max_distance = (grid_rows - 1) + (grid_cols - 1)  # Max Manhattan distance
    max_score = 0

    for (cluster_a, cluster_b), weight in workload_dict.items():
        if cluster_a in cluster_positions and cluster_b in cluster_positions:
            num_pairs = len(cluster_positions[cluster_a]) * len(cluster_positions[cluster_b])
            max_score += weight * max_distance * num_pairs

    # Normalize the score
    if max_score > 0:
        normalized_score = score / max_score
    else:
        normalized_score = 0  # Handle edge case where no inter-cluster communication exists

    return normalized_score

def evaluate_power_variance(floorplan_data, grid_dims, clusters, region_divisions=4):
    """
    Evaluates the power variance of the chiplet floorplan by calculating the variance
    of power density across regions in the grid. Normalizes the score to a range of [0, 1].

    Args:
        floorplan_data (list): List of chiplet details (dicts) with positions and sizes.
        grid_dims (tuple): Dimensions of the grid (rows, cols).
        clusters (dict): Dictionary containing cluster details, including power density (pd).
        region_divisions (int): Number of divisions for the grid into smaller regions.

    Returns:
        float: Normalized power variance score in the range [0, 1].
    """
    # Unpack grid dimensions
    grid_rows, grid_cols = grid_dims
    power_density_grid = np.zeros((grid_rows, grid_cols))  # Initialize power density grid

    # Fill power density grid based on chiplets
    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]  # Chiplet's starting position
        length = chiplet["Length"]          # Length of the chiplet
        breadth = chiplet["Breadth"]        # Breadth of the chiplet

        # Map chiplet to its cluster and retrieve power density
        cluster_key = f"Cluster {chiplet['Chiplet'].split('-')[0][1:]}"  # Extract cluster key
        if cluster_key in clusters:
            power = clusters[cluster_key]["pd"]  # Retrieve power density
        else:
            raise KeyError(f"Unknown cluster key: {cluster_key}")

        # Distribute the chiplet's power density across its area
        for dx in range(length):
            for dy in range(breadth):
                grid_x = int(x) + dx
                grid_y = int(y) + dy
                # Check if the grid indices are within bounds
                if 0 <= grid_x < grid_rows and 0 <= grid_y < grid_cols:
                    power_density_grid[grid_x, grid_y] += power / (length * breadth)

    # Divide the grid into regions and calculate variance in each region
    total_variance = 0
    max_possible_variance = 0
    region_size_rows = grid_rows // region_divisions  # Height of each region
    region_size_cols = grid_cols // region_divisions  # Width of each region

    for region_row in range(0, grid_rows, region_size_rows):
        for region_col in range(0, grid_cols, region_size_cols):
            # Extract the region's power density values
            region = power_density_grid[region_row:region_row + region_size_rows,
                                        region_col:region_col + region_size_cols]

            # Calculate variance for the region
            region_variance = np.var(region)
            total_variance += region_variance

            # Calculate maximum possible variance for this region
            total_power = np.sum(region)  # Total power in the region
            num_cells = region.size  # Number of cells in the region
            if num_cells > 0:
                max_possible_variance += (total_power ** 2) / num_cells

    # Normalize the total variance by the maximum possible variance
    if max_possible_variance == 0:
        return 0  # Perfect balance if there's no possible variance

    normalized_score = total_variance / max_possible_variance
    return normalized_score


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
    plt.title(f"Nexp_{i}/chiplet-centers")
    plt.savefig(f"Nexp_{i}/chiplet-centers.png")
    plt.close()

# Visualization function with tick marks and equal axes scaling
def visualize_chiplet_floorplan(i, floorplan_data, clusters, spacing, title="Chiplet Placement with Tick Marks and Labels"):
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
        color = cluster_colors[cluster_key]

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
    if spacing == 0:
        plt.savefig(f"Nexp_{i}/chiplet-placement.png")
    else: 
        plt.savefig(f"Nexp_{i}/chiplet-placemen_w_spacing.png")

    plt.close()

def is_grid_contiguous(grid):
    """
    Checks if the grid has no empty spaces (whitespace) between chiplets.
    Ensures that all occupied cells form a single connected component.

    Args:
        grid (np.array): A 2D numpy array representing the grid.

    Returns:
        bool: True if the grid is contiguous (no whitespace), False otherwise.
    """
    rows, cols = grid.shape
    visited = set()
    
    # Find the first occupied cell
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:  # Occupied cell
                start = (i, j)
                break
        else:
            continue
        break
    else:
        # No occupied cells in the grid
        return False

    # Perform a flood-fill (DFS) to visit all connected occupied cells
    def dfs(x, y):
        if (x, y) in visited or x < 0 or y < 0 or x >= rows or y >= cols or grid[x, y] == 0:
            return
        visited.add((x, y))
        dfs(x - 1, y)  # Up
        dfs(x + 1, y)  # Down
        dfs(x, y - 1)  # Left
        dfs(x, y + 1)  # Right

    # Start the DFS
    dfs(*start)

    # Count the total number of occupied cells
    total_occupied = np.sum(grid != 0)

    # If the number of visited cells matches the total occupied cells, it's contiguous
    return len(visited) == total_occupied

# Initialize grid, cluster positions, and parameters
grid = np.zeros(grid_dims, dtype=int)
cluster_positions = {}
permutations_list = list(permutations(clusters.keys()))  # All permutations of clusters
top_k = []  # Heap for top-k designs
k = 3  # Number of top designs to keep
max_top_k = 5  # Limit to top 5 designs

# Experiment tracking
exp_number = 0
score_minima = 10000
exp_minima = 0
top_5_minima = []  # This will store tuples of (total_score, exp_number)
unique_floorplans = set()  # Set to track unique floorplans by hash

# Loop through each permutation of clusters
for perm in permutations_list:
    exp_number += 1  # Increment experiment number
    grid.fill(0)  # Reset the grid for each permutation
    cluster_positions.clear()  # Reset cluster positions

    # Place chiplets based on the current permutation
    for cluster in perm:
        cluster_positions[cluster] = place_chiplets_fixed(grid, cluster, clusters)

    # Ensure the grid has been successfully populated
    # if not np.all(grid == 0):
    if np.all(grid):
        if not is_grid_contiguous(grid):
            print(f"Skipping Experiment {exp_number}: Grid is not contiguous.")
            continue
        # Generate floorplan data
        floorplan_data = generate_floorplan_data(cluster_positions, clusters)
        updated_floorplan = calculate_neighbors(floorplan_data)
        floorplan_hash = hash(
            tuple(sorted((chiplet["Chiplet"], chiplet["Lower_Left_Corner"]) for chiplet in updated_floorplan))
        )

        # Skip if the floorplan already exists
        if floorplan_hash in unique_floorplans:
            continue
        # Evaluate placement
        warpage_score = evaluate_chip_warpage(updated_floorplan, grid_dims)
        intra_score = evaluate_intra_cluster_continuity(updated_floorplan, grid_dims)
        inter_score = evaluate_inter_cluster_hop_count(updated_floorplan, workload_dict, grid_dims)
        power_score = evaluate_power_variance(updated_floorplan, grid_dims, clusters)

        # Total score (lower is better)
        a1 = 2
        a2 = 1/2
        a3 = 1
        a4 = 50
        total_score = a1*warpage_score + a2*intra_score + a3*inter_score + a4*power_score
        if (total_score < score_minima):
            score_minima = total_score
            exp_minima = exp_number
        # Add the current design to the list
        unique_floorplans.add(floorplan_hash)
        top_5_minima.append((total_score, exp_number, floorplan_data))

        # Sort the list based on total_score (ascending order)
        top_5_minima.sort(key=lambda x: x[0])

        # If the list exceeds 5 designs, remove the worst (last element)
        if len(top_5_minima) > max_top_k:
            removed_design = top_5_minima.pop()
            # Remove the hash of the removed design from the set
            removed_hash = hash(
                tuple(sorted((chiplet["Chiplet"], chiplet["Lower_Left_Corner"]) for chiplet in removed_design[2]))
            )
            unique_floorplans.discard(removed_hash)
        # print(f"For {exp_number}, Warpage = {warpage_score:.2f}, Intra-cluster = {intra_score:.3f}, Inter-cluster = {inter_score:.2f}, Power variance = {power_score:.3f}")
        # print(f"For {exp_number}, Scaled Warpage = {a1*warpage_score:.2f}, Intra-cluster = {a2*intra_score:.3f}, Inter-cluster = {a3*inter_score:.2f}, Power variance = {a4*power_score:.3f}")
        # print("Minimum score rn = ", score_minima, "at Exp = ", exp_minima)
        # Maintain top-k designs
        # heappush(top_k, (total_score, exp_number, copy.deepcopy(floorplan_data)))
        # if len(top_k) > k:
        #     heappop(top_k)
        print("Top 5 Designs (Lowest Scores):")
        for score, exp_id, _ in top_5_minima:
            print(f"  Experiment {exp_id}: Score = {score:.2f}")
        # Create a folder for the current experiment
        
        exp_dir = f"Nexp_{exp_number}"
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Visualize and save floorplan and centers
        adjusted_floorplan = adjust_chiplets_with_spacing(floorplan_data)

        visualize_chiplet_floorplan(exp_number, floorplan_data, clusters, spacing=0,
                                    title=f"Chiplet Placement no spacing; Score = {total_score}")
        visualize_chiplet_floorplan(exp_number, adjusted_floorplan, clusters, spacing=0.25,
                                    title="Chiplet Placement with spacing")            
        # visualize_chiplet_centers(exp_number, adjusted_floorplan, spacing=0.25,
        #                           title="Chiplet Centers Visualization")
print("\nFinal Top 5 Designs:")
for score, exp_id, _ in top_5_minima:
    print(f"Experiment {exp_id}: Score = {score:.2f}")

# Output top designs
for score, exp_id, floorplan_data in sorted(top_5_minima):
    print(f"Design {exp_id}: Total Score = {score:.2f}")
    exp_dir = f"Nexp_{exp_id}"
    adjusted_floorplan = adjust_chiplets_with_spacing(floorplan_data)

    # Ensure the Archive directory exists
    archive_dir = "Archive"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    # Save visualization of the chiplet floorplan
    archive_exp_dir = os.path.join(archive_dir, f"Design_{exp_id}")
    if not os.path.exists(archive_exp_dir):
        os.makedirs(archive_exp_dir)

    visualize_chiplet_floorplan(
        exp_id,
        adjusted_floorplan,
        clusters,
        spacing=0,
        title=f"Chiplet Floorplan for Design {exp_id}"
    )

    # Move visualization to Archive folder
    os.rename(f"Nexp_{exp_id}/chiplet-placement.png", f"{archive_exp_dir}/chiplet-placement.png")

    # Call the MFIT evaluation
    T_peak = generate_power_config_file(adjusted_floorplan, clusters, exp_id)
    print(f"Thermal Result for {exp_id}: Peak Temp = {T_peak - 300:.2f}°C")
