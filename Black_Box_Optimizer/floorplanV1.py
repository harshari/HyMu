import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock
import pandas as pd
from itertools import permutations 
from generate_mfit_floorplan import generate_power_config_file
import os

# Network details:
# ResNet18: 11.7M parameters
# ResNet34: 21.8M parameters
# ResNet50: 25.6M parameters
# VGG16: 138M parameters
# VGG19: 144M parameters
# DenseNet121: 8M parameters
# MobileNetV2: 3.5M parameters

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
    "ResNet50": {
        "Storage": [9.19, 4.0, 36.0, 16.0, 16.0, 16.0, 36.0, 16.0, 16.0, 36.0, 16.0, 32.0, 144.0, 64.0, 128.0, 64.0, 144.0, 64.0, 64.0, 144.0, 64.0, 64.0, 144.0, 64.0, 128.0, 576.0, 256.0, 512.0, 256.0, 576.0, 256.0, 256.0, 576.0, 256.0, 256.0, 576.0, 256.0, 256.0, 576.0, 256.0, 256.0, 576.0, 256.0, 512.0, 2304.0, 1024.0, 2048.0, 1024.0, 2304.0, 1024.0],
        "Activations": [784.0, 196.0, 196.0, 784.0, 784.0, 196.0, 196.0, 784.0, 196.0, 196.0, 784.0, 392.0, 98.0, 392.0, 392.0, 98.0, 98.0, 392.0, 98.0, 98.0, 392.0, 98.0, 98.0, 392.0, 196.0, 49.0, 196.0, 196.0, 49.0, 49.0, 196.0, 49.0, 49.0, 196.0, 49.0, 49.0, 196.0, 49.0, 49.0, 196.0, 49.0, 49.0, 196.0, 98.0, 24.5, 98.0, 98.0, 24.5, 24.5, 98.0],
        "Compute": [118.01, 12.85, 115.61, 51.38, 51.38, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 102.76, 115.61, 51.38, 102.76, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 102.76, 115.61, 51.38, 102.76, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 51.38, 115.61, 51.38, 102.76, 115.61, 51.38, 102.76, 51.38, 115.61, 51.38],
        #"Sensitivity": [0.17, 0.14, 1.35, 0.15, 10.34, 0.17, 9.15, 0.14, 19.93, 0.15, 15.68, 0.12, 16.16, 0.10, 26.12, 0.14]
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
    "DenseNet121": {
        "Storage": [9.19, 8.00, 36.00, 12.00, 36.00, 16.00, 36.00, 20.00, 36.00, 24.00, 36.00, 28.00, 36.00, 32.00, 16.00, 36.00, 20.00, 36.00, 24.00, 36.00, 28.00, 36.00, 32.00, 36.00, 36.00, 36.00, 40.00, 36.00, 44.00, 36.00, 48.00, 36.00, 52.00, 36.00, 56.00, 36.00, 60.00, 36.00, 128.00, 32.00, 36.00, 36.00, 36.00, 40.00, 36.00, 44.00, 36.00, 48.00, 36.00, 52.00, 36.00, 56.00, 36.00, 60.00, 36.00, 64.00, 36.00, 68.00, 36.00, 72.00, 36.00, 76.00, 36.00, 80.00],
        "Activations": [784.00, 392.00, 98.00, 392.00, 98.00, 392.00, 98.00, 392.00, 98.00, 392.00, 98.00, 392.00, 98.00, 392.00, 98.00, 24.50, 98.00, 24.50, 98.00, 24.50, 98.00, 24.50, 98.00, 24.50],
        "Compute": [118.01, 25.69, 115.61, 38.54, 115.61, 51.38, 115.61, 64.23, 115.61, 77.07, 115.61, 89.92, 115.61, 102.76, 12.85, 28.90],
        #"Sensitivity": [0.10, 0.12, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35]
    },
    "MobileNetV2": {
        "Storage": [0.84, 0.28, 0.50, 1.50, 0.84, 2.25, 3.38, 1.27, 3.38, 3.38, 1.27, 4.50, 6.00, 1.69, 6.00, 6.00, 1.69, 6.00, 6.00, 1.69, 12.00, 24.00, 3.38, 24.00, 24.00, 3.38, 24.00, 24.00, 3.38, 24.00, 24.00, 3.38, 36.00, 54.00, 5.06, 54.00, 54.00, 5.06, 54.00, 54.00, 5.06, 90.00, 150.00, 8.44, 150.00, 150.00, 8.44, 150.00, 150.00, 8.44, 300.00, 400.00],
        "Activations": [392.00, 392.00, 196.00, 1176.00, 294.00, 73.50, 441.00, 441.00, 73.50, 441.00, 110.25, 24.50, 147.00, 147.00, 24.50, 147.00, 147.00, 24.50, 147.00, 36.75, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 18.38, 110.25, 110.25, 18.38, 110.25, 110.25, 18.38, 110.25, 27.56, 7.66, 45.94, 45.94, 7.66, 45.94, 45.94, 7.66, 45.94, 45.94, 15.31, 61.25],
        "Compute": [10.84, 115.61, 6.42, 19.27, 260.11, 7.23, 10.84, 585.25, 10.84, 10.84, 146.31, 3.61, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 65.03, 2.41, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 260.11, 7.23, 10.84, 585.25, 10.84, 10.84, 585.25, 10.84, 10.84, 146.31, 4.52, 7.53, 406.43, 7.53, 7.53, 406.43, 7.53, 7.53, 406.43, 15.05, 20.07],
        "Sensitivity": [1.81, 0.00, 0.16, 47.19, 0.02, 0.11, 0.17, 0.01, 0.00, 2.27, 0.00, 0.02, 21.64, 0.04, 0.01, 0.38, 0.06, 0.00, 1.15, 0.00, 0.01, 6.65, 0.01, 0.01, 0.11, 0.03, 0.00, 1.07, 0.00, 0.00, 9.69, 0.01, 0.00, 0.13, 0.02, 0.00, 0.50, 0.00, 0.00, 4.63, 0.00, 0.00, 0.02, 0.01, 0.00, 0.17, 0.00, 0.00, 1.89, 0.00, 0.00, 0.02]
    }
}


# chiplet configurations


# Cluster configurations
clusters = {
    ### Original 
    # "Cluster 1": {"count": 28, "pd": 8, "area": 8, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},  # 2x4 or 4x2 chiplets
    # "Cluster 2": {"count": 12, "pd": 1, "area": 4, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},  # 2x2 chiplets
    # "Cluster 3": {"count": 18, "pd": 4, "area": 4, "memory": 4800, "tops": 70e12, "energy_per_mac": .11e-12},  # 2x2 chiplets
    # "Cluster 4": {"count": 24, "pd": 8, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},  # 2x2 chiplets
    "Cluster 1": {"count": 24, "pd": 8, "area": 4, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},  # 2x4 or 4x2 chiplets
    "Cluster 2": {"count": 28, "pd": 8, "area": 8, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},  # 2x2 chiplets
    "Cluster 3": {"count": 18, "pd": 4, "area": 4, "memory": 4800, "tops": 70e12, "energy_per_mac": .11e-12},  # 2x2 chiplets
    "Cluster 4": {"count": 12, "pd": 1, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},  # 2x2 chiplets
    # "Cluster 4": {"count": 0, "pd": 2, "area": 4, "memory": 108, "tops": 11e12, "energy_per_mac": .18e-12},  # 2x2 chiplets

}

grid_dims = (20, 22)  # Dimensions of the grid

# Network details and clusters (same as provided earlier)
network_data = {...}  # Replace with your network data
clusters = {...}  # Replace with your cluster data
grid_dims = (20, 22)

# Define helper functions for chiplet placement and visualization
def is_valid_position(grid, x, y, chiplet_size):
    rows, cols = grid.shape
    if x + chiplet_size[0] > rows or y + chiplet_size[1] > cols:
        return False
    for dx in range(chiplet_size[0]):
        for dy in range(chiplet_size[1]):
            if grid[x + dx, y + dy] != 0:
                return False
    return True

def place_chiplets_fixed(grid, cluster_key, clusters):
    chiplet_area = clusters[cluster_key]["area"]
    chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]

    while chiplets_remaining > 0:
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if is_valid_position(grid, x, y, chiplet_size):
                    positions.append((x, y))
                    for dx in range(chiplet_size[0]):
                        for dy in range(chiplet_size[1]):
                            grid[x + dx, y + dy] = int(cluster_key.split()[-1])
                    chiplets_remaining -= 1
                    break
            if chiplets_remaining == 0:
                break

    return positions

def generate_floorplan_data(cluster_positions, clusters):
    floorplan_data = []
    for cluster_key, positions in cluster_positions.items():
        chiplet_area = clusters[cluster_key]["area"]
        chiplet_size = (4, 2) if chiplet_area == 8 else (2, 2)
        for idx, (x, y) in enumerate(positions, start=1):
            floorplan_data.append({
                "Chiplet": f"{cluster_key}-{idx}",
                "Lower_Left_Corner": (x, y),
                "Length": chiplet_size[0],
                "Breadth": chiplet_size[1]
            })
    return floorplan_data

def visualize_chiplet_floorplan(exp_index, floorplan_data, clusters, spacing=0.25):
    plt.figure(figsize=(12, 8))
    max_x = max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in floorplan_data)
    max_y = max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in floorplan_data)

    plt.xlim(0, max_x + spacing)
    plt.ylim(0, max_y + spacing)
    plt.gca().set_aspect('equal', adjustable='box')

    cluster_colors = {"Cluster 1": "orange", "Cluster 2": "green", "Cluster 3": "blue", "Cluster 4": "red"}
    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        color = cluster_colors.get(chiplet["Chiplet"].split('-')[0], "gray")

        plt.gca().add_patch(plt.Rectangle((y, x), breadth, length, color=color, alpha=0.8, edgecolor="black"))
        plt.text(y + breadth / 2, x + length / 2, chiplet["Chiplet"], ha="center", va="center", fontsize=8)

    plt.grid(visible=True, color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Chiplet Placement - Experiment {exp_index}")
    plt.savefig(f"exp_{exp_index}/chiplet_placement.png")
    plt.close()

# Main function to iterate over permutations and visualize
def main():
    ordering = list(permutations(["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]))
    i = 0

    for perm in ordering:
        i += 1
        grid = np.zeros(grid_dims, dtype=int)
        cluster_positions = {}

        for cluster in perm:
            cluster_positions[cluster] = place_chiplets_fixed(grid, cluster, clusters)

        if not np.any(grid):
            continue

        os.makedirs(f"exp_{i}", exist_ok=True)

        # Generate floorplan data and visualize
        floorplan_data = generate_floorplan_data(cluster_positions, clusters)
        visualize_chiplet_floorplan(i, floorplan_data, clusters)

        # Save floorplan data to CSV
        floorplan_df = pd.DataFrame(floorplan_data)
        floorplan_df.to_csv(f"exp_{i}/chiplet_floorplan.csv", index=False)

        # Link to external functions if needed
        generate_power_config_file(floorplan_data, clusters, i)

if __name__ == "__main__":
    main()