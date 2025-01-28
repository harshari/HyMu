import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
from itertools import permutations
import pandas as pd
import os
from generate_mfit_floorplan import generate_power_config_file


# source HyMu_env/bin/activate
# Define chiplet cluster dictionary for surface tier
clusters_surface = {
    "Cluster 1": {"count": 24, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},  # Standard - 80mm2
    "Cluster 2": {"count": 28, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},  # Shared_ADC - 80mm2
    "Cluster 3": {"count": 0, "pd": 2, "area": 4, "memory": 108, "tops": 11e12, "energy_per_mac": .18e-12},  # Adder - 80mm2
    "Cluster 4": {"count": 18, "pd": 8, "area": 4, "memory": 2400, "tops": 35e12, "energy_per_mac": .22e-12},  # Accumulator
    "Cluster 5": {"count": 12, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},  # ADC_Less - 96
}

# Define chiplet cluster dictionary for embedded tier
clusters_embedded = {
    "Cluster 1": {"count": 0, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},  # Standard - 80mm2
    "Cluster 2": {"count": 12, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},  # Shared_ADC - 80mm2
    "Cluster 3": {"count": 0, "pd": 2, "area": 4, "memory": 108, "tops": 11e12, "energy_per_mac": .18e-12},  # Adder - 80mm2
    "Cluster 4": {"count": 12, "pd": 8, "area": 4, "memory": 2400, "tops": 35e12, "energy_per_mac": .22e-12},  # Accumulator
    "Cluster 5": {"count": 12, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},  # ADC_Less - 96
}


# Define grid dimensions for both tiers
grid_dims_surface = (20, 22)  # Rows x Columns for surface tier
grid_dims_embedded = (12, 16)  # Rows x Columns for embedded tier
max_rows = max(grid_dims_surface[0], grid_dims_embedded[0])
max_cols = max(grid_dims_surface[1], grid_dims_embedded[1])
max_dims = (max_rows, max_cols)

# Grid dimensions
peak_temp = 100000 
logic_limit_spacing = .25
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

# Function to find the best position for a chiplet
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

    if best_pos is None:
        print("No valid position found. Check chiplet size and grid dimensions.")
    return best_pos

# Function to place chiplets
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

# Generate floorplan data
def generate_floorplan_data(cluster_positions, clusters, embed):
    floorplan_data = []
    if embed:
        for cluster_key, positions in cluster_positions.items():
            chiplet_area = clusters[cluster_key]["area"]
            chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
            for idx, (x, y) in enumerate(positions, start=1):
                floorplan_data.append({
                    "Chiplet": f"{cluster_key}-{idx}".replace("Cluster ", "C"),
                    "Lower_Left_Corner": (x, y),
                    "Length": chiplet_size[0],
                    "Breadth": chiplet_size[1],
                    "location": "embedded"
                })
    else:
        for cluster_key, positions in cluster_positions.items():
            chiplet_area = clusters[cluster_key]["area"]
            chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
            for idx, (x, y) in enumerate(positions, start=1):
                floorplan_data.append({
                    "Chiplet": f"{cluster_key}-{idx}".replace("Cluster ", "C"),
                    "Lower_Left_Corner": (x, y),
                    "Length": chiplet_size[0],
                    "Breadth": chiplet_size[1],
                    "location": "surface"
                })
    return floorplan_data

# Calculate neighbors
def calculate_neighbors(floorplan_data):
    def is_neighbor(chip1, chip2):
        adjacent_x = (chip1['Lower_Left_Corner'][0] + chip1['Length'] == chip2['Lower_Left_Corner'][0]) or (
                chip2['Lower_Left_Corner'][0] + chip2['Length'] == chip1['Lower_Left_Corner'][0])
        overlapping_y = not (chip1['Lower_Left_Corner'][1] + chip1['Breadth'] <= chip2['Lower_Left_Corner'][1] or
                             chip2['Lower_Left_Corner'][1] + chip2['Breadth'] <= chip1['Lower_Left_Corner'][1])

        adjacent_y = (chip1['Lower_Left_Corner'][1] + chip1['Breadth'] == chip2['Lower_Left_Corner'][1]) or (
                chip2['Lower_Left_Corner'][1] + chip2['Breadth'] == chip1['Lower_Left_Corner'][1])
        overlapping_x = not (chip1['Lower_Left_Corner'][0] + chip1['Length'] <= chip2['Lower_Left_Corner'][0] or
                             chip2['Lower_Left_Corner'][0] + chip2['Length'] <= chip1['Lower_Left_Corner'][0])

        return (adjacent_x and overlapping_y) or (adjacent_y and overlapping_x)

    for i, chip1 in enumerate(floorplan_data):
        count = 0
        for j, chip2 in enumerate(floorplan_data):
            if i != j and is_neighbor(chip1, chip2):
                count += 1
        floorplan_data[i]['num_neighbors'] = count

    return floorplan_data

# Adjust chiplet spacing
def adjust_chiplets_with_spacing(floorplan_data, spacing=0.25):
    adjusted_floorplan = []
    half_spacing = spacing / 2  # Spacing is split evenly on all sides

    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]

        # Adjust position by adding spacing to the coordinates
        adjusted_x = x + half_spacing*x
        adjusted_y = y + half_spacing*y

        # Adjust the dimensions to account for spacing around the chiplet
        adjusted_length = length # if we want to scale dimension for cleaner layout, do length - spacing
        adjusted_breadth = breadth # if we want to scale dimension for cleaner layout, do breadth - spacing

        # Calculate the center of the chiplet with adjusted spacing
        center_x = adjusted_x + adjusted_length / 2
        center_y = adjusted_y + adjusted_breadth / 2

        # Add adjusted chiplet to the floorplan
        adjusted_floorplan.append({
            "Chiplet": chiplet["Chiplet"],
            "Lower_Left_Corner": (adjusted_x, adjusted_y),
            "Length": adjusted_length,
            "Breadth": adjusted_breadth,
            "Center": (center_x, center_y),
            "neighbor_count": chiplet["num_neighbors"],
            "location": chiplet["location"]  # Include the location field
        })

    return adjusted_floorplan

# Visualization functions
def visualize_chiplet_centers(exp_dir, floorplan, title, grid_dims, spacing=0.25):
    """
    Visualizes chiplet centers with spacing applied, showing their actual positions,
    including those extending outside the grid.

    Parameters:
    - exp_dir: Directory to save the plot.
    - floorplan: List of chiplets with their centers and metadata.
    - title: Title for the plot.
    - grid_dims: Tuple of (rows, cols) for the original grid dimensions.
    - spacing: Spacing applied between chiplets.
    """
    rows, cols = grid_dims  # Unpack the grid dimensions

    # Calculate dynamic bounds for visualization
    max_x = max(chiplet["Center"][0] + spacing / 2 for chiplet in floorplan)
    max_y = max(chiplet["Center"][1] + spacing / 2 for chiplet in floorplan)

    plt.figure(figsize=(12, 8))
    plt.title(title)

    for chiplet in floorplan:
        center_x, center_y = chiplet["Center"]  # Real center after spacing

        # Plot chiplet center as a red dot
        plt.plot(center_y, center_x, 'ro')

        # Add chiplet label
        plt.text(
            center_y,
            center_x,
            chiplet["Chiplet"],
            fontsize=8,
            ha="center",
            va="center"
        )

    # Set axis limits based on dynamic bounds (including spacing adjustments)
    plt.xlim(0, max(cols, max_y))
    plt.ylim(0, max(rows, max_x))

    # Ensure equal aspect ratio for x and y
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Y-axis (columns)")
    plt.ylabel("X-axis (rows)")
    plt.savefig(f"{exp_dir}.png", dpi=300, bbox_inches="tight")
    plt.close()

def visualize_embedded_chiplet_overlap(exp_dir, surface_floorplan, embedded_floorplan, title, grid_dims, spacing=0.25):
    """
    Visualizes embedded chiplet centers and highlights overlaps with surface chiplets.

    Parameters:
    - exp_dir: Directory to save the plot.
    - surface_floorplan: List of chiplets from the surface tier.
    - embedded_floorplan: List of chiplets from the embedded tier.
    - title: Title for the plot.
    - grid_dims: Tuple of (rows, cols) for the grid dimensions.
    - spacing: Spacing applied between chiplets.
    """
    rows, cols = grid_dims  # Unpack grid dimensions

    plt.figure(figsize=(12, 8))

    # Helper function to check overlap
    def is_overlapping(chip1, chip2):
        x1_start, y1_start = chip1["Lower_Left_Corner"]
        x1_end = x1_start + chip1["Length"]
        y1_end = y1_start + chip1["Breadth"]

        x2_start, y2_start = chip2["Lower_Left_Corner"]
        x2_end = x2_start + chip2["Length"]
        y2_end = y2_start + chip2["Breadth"]

        # Check if bounding boxes overlap
        return not (x1_end <= x2_start or x2_end <= x1_start or
                    y1_end <= y2_start or y2_end <= y1_start)

    # Determine overlaps
    overlapping_points = set()
    for embedded_chiplet in embedded_floorplan:
        for surface_chiplet in surface_floorplan:
            if is_overlapping(embedded_chiplet, surface_chiplet):
                overlapping_points.add(embedded_chiplet["Center"])
                break  # Stop checking further if overlap is found

    # Plot embedded chiplets
    for chiplet in embedded_floorplan:
        center_x, center_y = chiplet["Center"]

        # Color green for overlap, red otherwise
        if chiplet["Center"] in overlapping_points:
            plt.plot(center_y, center_x, 'go')  # Green for overlapping
        else:
            plt.plot(center_y, center_x, 'ro')  # Red for non-overlapping

        # Add chiplet label
        plt.text(
            center_y,
            center_x,
            chiplet["Chiplet"],
            fontsize=8,
            ha="center",
            va="center"
        )

    # Set axis limits
    max_x = max(rows, max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in embedded_floorplan))
    max_y = max(cols, max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in embedded_floorplan))
    plt.xlim(0, max_y)
    plt.ylim(0, max_x)

    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Update title with overlap summary
    total_embedded = len(embedded_floorplan)
    total_overlapping = len(overlapping_points)
    plt.title(f"{title} (Total green points = {total_overlapping} out of {total_embedded})")

    # Add labels
    plt.xlabel("Y-axis (columns)")
    plt.ylabel("X-axis (rows)")

    # Save the plot
    plt.savefig(f"{exp_dir}.png", dpi=300, bbox_inches="tight")
    plt.close()

def calculate_custom_spacing_from_floorplan(surface_floorplan_data, embedded_floorplan_data, surface_grid_dims, embedded_grid_dims, logic_limit_spacing):
    """
    Calculates custom spacing for the embedded grid to match the dimensions of the surface grid.

    Parameters:
    - surface_floorplan_data: List of dictionaries with surface chiplet data.
    - embedded_floorplan_data: List of dictionaries with embedded chiplet data.
    - surface_grid_dims: Tuple (rows, cols) for the surface grid dimensions.
    - embedded_grid_dims: Tuple (rows, cols) for the embedded grid dimensions.
    - logic_limit_spacing: Initial logic spacing for the embedded grid.

    Returns:
    - custom_spacing: The calculated spacing for the embedded grid to match the surface grid dimensions.
    """
    # Calculate total surface grid dimensions based on floorplan data
    surface_total_length = sum(chiplet["Length"] for chiplet in surface_floorplan_data)
    surface_total_breadth = sum(chiplet["Breadth"] for chiplet in surface_floorplan_data)

    # Calculate total embedded grid dimensions based on floorplan data
    embedded_total_length = sum(chiplet["Length"] for chiplet in embedded_floorplan_data)
    embedded_total_breadth = sum(chiplet["Breadth"] for chiplet in embedded_floorplan_data)

    # Adjust embedded spacing to match surface dimensions
    surface_rows, surface_cols = surface_grid_dims
    embedded_rows, embedded_cols = embedded_grid_dims

    # Calculate spacing adjustments for both dimensions
    custom_spacing_length = ((surface_total_length / surface_rows) - (embedded_total_length / embedded_rows))
    custom_spacing_breadth = ((surface_total_breadth / surface_cols) - (embedded_total_breadth / embedded_cols))

    # Take the smaller of the two to ensure alignment
    custom_spacing = max(min(custom_spacing_length, custom_spacing_breadth, logic_limit_spacing), 0)

    return custom_spacing

def visualize_chiplet_floorplan(exp_dir, floorplan, clusters, spacing, title):
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=18)  # Increase title font size

    # Determine the maximum bounds for the grid
    max_x = max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in floorplan)
    max_y = max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in floorplan)

    # Set axis limits based on chiplet positions and spacing
    plt.xlim(0, max_y + spacing)
    plt.ylim(0, max_x + spacing)
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal scaling

    # Cluster-specific colors for better visualization
    cluster_colors = {"C1": "red", "C2": "yellow", "C3": "purple", "C4": "blue", "C5": "green"}

    for chiplet in floorplan:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        cluster_key = chiplet["Chiplet"].split("-")[0]
        color = cluster_colors.get(cluster_key, "gray")

        # Plot the chiplet as a rectangle
        plt.gca().add_patch(
            plt.Rectangle(
                (y, x),  # Rectangle starts at (y, x) (remember matplotlib uses (x, y))
                breadth,  # Width along y-axis
                length,   # Height along x-axis
                facecolor=color,
                alpha=0.6,
                edgecolor="black"
            )
        )
        # Add text label in the center of the chiplet
        plt.text(
            y + breadth / 2,  # Center X position
            x + length / 2,   # Center Y position
            chiplet["Chiplet"],
            fontsize=12,  # Chiplet label font size is smaller
            ha="center",
            va="center"
        )

    # Add gridlines for better visualization
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Y-axis (mm)", fontsize=16)  # Increase axis label font size
    plt.ylabel("X-axis (mm)", fontsize=16)  # Increase axis label font size
    plt.xticks(fontsize=14)  # Increase tick font size
    plt.yticks(fontsize=14)  # Increase tick font size

    # Save the plot to a file
    plt.savefig(f"{exp_dir}.png", dpi=300, bbox_inches="tight")
    plt.close()



# Main Code
tier1_permutations = list(permutations(clusters_surface.keys()))
tier2_permutations = list(permutations(clusters_embedded.keys()))
i = 0
for tier1 in tier1_permutations:
    for tier2 in tier2_permutations:
        i += 1
        # Initialize grid for surface tier
        # Initialize grids for surface and embedded tiers
        grid_surface = np.zeros(grid_dims_surface, dtype=int)
        
        cluster_positions_surface = {}
        for cluster in tier1:
            cluster_positions_surface[cluster] = place_chiplets_fixed(grid_surface, cluster, clusters_surface)
        # Check if surface tier placement was successful

        if not np.all(grid_surface):
            continue
        
        # Initialize a fresh grid for the embedded tier
        grid_embedded = np.zeros(grid_dims_embedded, dtype=int)

        cluster_positions_embedded = {}
        for cluster in tier2:
            cluster_positions_embedded[cluster] = place_chiplets_fixed(grid_embedded, cluster, clusters_embedded)

        if not np.all(grid_embedded):
            continue

        # if any(len(positions) < clusters_surface[cluster]["count"] for cluster, positions in cluster_positions_surface.items()):
        #     print("Surface tier placement failed.")
        #     continue

        # if any(len(positions) < clusters_embedded[cluster]["count"] for cluster, positions in cluster_positions_embedded.items()):
        #     print("Embedded tier placement failed.")
        #     continue

        exp_dir = f"exp_{i}"
        os.makedirs(exp_dir, exist_ok=True)

        # Generate floorplan data
        floorplan_data_surface = generate_floorplan_data(cluster_positions_surface, clusters_surface, embed=False)
        floorplan_data_embedded = generate_floorplan_data(cluster_positions_embedded, clusters_embedded, embed=True)
        # print(floorplan_data_embedded)
        # for chiplet in floorplan_data:
        #     chiplet["location"] = "surface" if chiplet["Chiplet"].split("-")[0] in tier1 else "embedded"

        adjusted_floorplan_surface = adjust_chiplets_with_spacing(calculate_neighbors(floorplan_data_surface),spacing = logic_limit_spacing)
        # custom_spacing = calculate_custom_spacing_from_floorplan(floorplan_data_surface, floorplan_data_embedded, grid_dims_surface,
        #     grid_dims_embedded,
        #     logic_limit_spacing
        # )
        adjusted_floorplan_embedded = adjust_chiplets_with_spacing(calculate_neighbors(floorplan_data_embedded), spacing = 1.25)
        surface_chiplets = [c for c in adjusted_floorplan_surface if c["location"] == "surface"]
        embedded_chiplets = [c for c in adjusted_floorplan_embedded if c["location"] == "embedded"]

        # check if being populated
        # for chiplet in adjusted_floorplan_embedded:
        #     print(f"{chiplet['Chiplet']} - Length: {chiplet['Length']}, Breadth: {chiplet['Breadth']}, Lower_Left_Corner: {chiplet['Lower_Left_Corner']}")

        # Visualize separately
        visualize_chiplet_centers(f"{exp_dir}/surface_centers", adjusted_floorplan_surface, "Surface Chiplet Centers", max_dims)
        visualize_chiplet_centers(f"{exp_dir}/embedded_centers", adjusted_floorplan_embedded, "Embedded Chiplet Centers", max_dims)
        visualize_chiplet_floorplan(f"{exp_dir}/surface_floorplan", adjusted_floorplan_surface, clusters_surface, 0.5, "Surface Floorplan")
        visualize_chiplet_floorplan(f"{exp_dir}/embedded_floorplan", adjusted_floorplan_embedded, clusters_embedded, 0.5, "Embedded Floorplan")
        
        # Visualize embedded chiplet overlaps
        visualize_embedded_chiplet_overlap(f"{exp_dir}/embedded_overlap",adjusted_floorplan_surface, adjusted_floorplan_embedded, "Embedded Chiplet Overlaps With Surface",
            max_dims,
            spacing=0.25
        )
        # Combine surface and embedded floorplans
        combined_floorplan = adjusted_floorplan_surface + adjusted_floorplan_embedded

        # Call the generate_power_config_file function once
        T_peak = generate_power_config_file(combined_floorplan, adjusted_floorplan_embedded, clusters_surface, clusters_embedded, i, grid_dims_surface, grid_dims_embedded)

        if (T_peak < peak_temp):
            peak_temp = T_peak
            exp_number = i
            print(f"New minimum temp = {(T_peak - 300):.2f} at exp {i}")
print(f"Lowest T_peak for {exp_number} at T_peak = {T_peak}")

