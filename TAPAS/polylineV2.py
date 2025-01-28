import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
from itertools import permutations
import pandas as pd
import os
from generate_mfit_floorplan import generate_power_config_file


# source HyMu_env/bin/activate
# Cluster configurations
clusters = {
    "Cluster 1": {"count": 24, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": 0.87e-12},
    "Cluster 2": {"count": 28, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": 0.3e-12},
    "Cluster 3": {"count": 18, "pd": 4, "area": 4, "memory": 4800, "tops": 70e12, "energy_per_mac": 0.11e-12},
    "Cluster 4": {"count": 12, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": 0.27e-12},
}

# Grid dimensions
grid_dims = (20, 22)
peak_temp = 100000 

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
        adjusted_x = x + half_spacing
        adjusted_y = y + half_spacing

        # Adjust the dimensions to account for spacing around the chiplet
        adjusted_length = length - spacing
        adjusted_breadth = breadth - spacing

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
def visualize_chiplet_centers(exp_dir, floorplan, title):
    plt.figure(figsize=(12, 8))
    plt.title(title)
    for chiplet in floorplan:
        center_x, center_y = chiplet["Center"]
        plt.plot(center_y, center_x, 'ro')
        plt.text(center_y, center_x, chiplet["Chiplet"], fontsize=8, ha="center", va="center")
    plt.savefig(f"{exp_dir}.png")
    plt.close()

def visualize_chiplet_floorplan(exp_dir, floorplan, clusters, spacing, title):
    plt.figure(figsize=(12, 8))
    plt.title(title)

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
            fontsize=8,
            ha="center",
            va="center"
        )

    # Add gridlines for better visualization
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Y-axis (mm)")
    plt.ylabel("X-axis (mm)")

    # Save the plot to a file
    plt.savefig(f"{exp_dir}.png")
    plt.close()


# Main Code
tier1_permutations = list(permutations(clusters.keys()))
tier2_permutations = list(permutations(clusters.keys()))
i = 0
for tier1 in tier1_permutations:
    for tier2 in tier2_permutations:
        i += 1
        # Initialize grid for surface tier
        grid_surface = np.zeros((grid_dims[0], grid_dims[1]), dtype=int)
        cluster_positions_surface = {}

        for cluster in tier1:
            cluster_positions_surface[cluster] = place_chiplets_fixed(grid_surface, cluster, clusters)

        # Check if surface tier placement was successful
        if not np.all(grid_surface):
            continue

        # Initialize a fresh grid for the embedded tier
        grid_embedded = np.zeros((grid_dims[0], grid_dims[1]), dtype=int)
        cluster_positions_embedded = {}

        for cluster in tier2:
            cluster_positions_embedded[cluster] = place_chiplets_fixed(grid_embedded, cluster, clusters)

        # Check if embedded tier placement was successful
        if not np.all(grid_embedded):
            continue

        exp_dir = f"exp_{i}"
        os.makedirs(exp_dir, exist_ok=True)

        floorplan_data_surface = generate_floorplan_data(cluster_positions_surface, clusters, embed = False)
        floorplan_data_embedded = generate_floorplan_data(cluster_positions_embedded, clusters, embed = True)
        # print(floorplan_data_embedded)
        # for chiplet in floorplan_data:
        #     chiplet["location"] = "surface" if chiplet["Chiplet"].split("-")[0] in tier1 else "embedded"

        adjusted_floorplan_surface = adjust_chiplets_with_spacing(calculate_neighbors(floorplan_data_surface),spacing = .25)
        adjusted_floorplan_embedded = adjust_chiplets_with_spacing(calculate_neighbors(floorplan_data_embedded), spacing = .25)
        surface_chiplets = [c for c in adjusted_floorplan_surface if c["location"] == "surface"]
        embedded_chiplets = [c for c in adjusted_floorplan_embedded if c["location"] == "embedded"]

        # check if being populated
        # for chiplet in adjusted_floorplan_embedded:
        #     print(f"{chiplet['Chiplet']} - Length: {chiplet['Length']}, Breadth: {chiplet['Breadth']}, Lower_Left_Corner: {chiplet['Lower_Left_Corner']}")

        # Visualize separately
        visualize_chiplet_centers(f"{exp_dir}/surface_centers", adjusted_floorplan_surface, "Surface Chiplet Centers")
        visualize_chiplet_centers(f"{exp_dir}/embedded_centers", adjusted_floorplan_embedded, "Embedded Chiplet Centers")
        visualize_chiplet_floorplan(f"{exp_dir}/surface_floorplan", adjusted_floorplan_surface, clusters, 0.5, "Surface Floorplan")
        visualize_chiplet_floorplan(f"{exp_dir}/embedded_floorplan", adjusted_floorplan_embedded, clusters, 0.5, "Embedded Floorplan")
        # Combine surface and embedded floorplans
        combined_floorplan = adjusted_floorplan_surface + adjusted_floorplan_embedded

        # Call the generate_power_config_file function once
        T_peak = generate_power_config_file(combined_floorplan, adjusted_floorplan_embedded, clusters, i, grid_dims)

        if (T_peak < peak_temp):
            peak_temp = T_peak
            exp_number = i
            print(f"New minimum temp = {(T_peak - 300):.2f} at exp {i}")
print(f"Lowest T_peak for {exp_number} at T_peak = {T_peak}")

