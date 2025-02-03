import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import os
import hashlib
import random

# Define chiplet clusters for surface and embedded tiers
clusters_surface = {
    "Cluster 1": {"count": 4, "length": 2, "breadth": 2},
    "Cluster 2": {"count": 1, "length": 4, "breadth": 2},
    "Cluster 3": {"count": 2, "length": 2, "breadth": 2},
    "Cluster 4": {"count": 8, "length": 2, "breadth": 1},
    "Cluster 5": {"count": 0, "length": 1, "breadth": 1}
}

clusters_embedded = {
    "Cluster 6": {"count": 8, "length": 3, "breadth": 1},
    "Cluster 7": {"count": 2, "length": 4, "breadth": 1}
}

# Grid dimensions
grid_dims_surface = (10, 10)
grid_dims_embedded = (10, 10)
E_s = 0.5  # Edge spacing threshold
random.seed(42)  # For reproducibility

# Function to generate hash for a floorplan
def hash_floorplan(floorplan_data):
    sorted_data = sorted(
        [(chiplet["Chiplet"], chiplet["Lower_Left_Corner"], chiplet["Length"], chiplet["Breadth"])]
        for chiplet in floorplan_data
    )
    return hashlib.sha256(str(sorted_data).encode('utf-8')).hexdigest()

# Function to calculate Hamming Distance between two designs
def hamming_distance(fp1, fp2):
    count = 0
    for c1, c2 in zip(fp1, fp2):
        if c1["Lower_Left_Corner"] != c2["Lower_Left_Corner"]:
            count += 1
    return count

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

# Function to find max distance position
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

def all_chiplets_placed(cluster_positions, clusters):
    """ Ensures all chiplets for each cluster have been successfully placed. """
    return all(len(positions) >= clusters[cluster]["count"] for cluster, positions in cluster_positions.items())

# Function to place chiplets with rotation fallback
def place_chiplets_fixed(grid, cluster_key, clusters):
    chiplet_length = clusters[cluster_key].get("length", 1)  
    chiplet_breadth = clusters[cluster_key].get("breadth", 1)
    chiplet_size = (chiplet_length, chiplet_breadth)

    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]

    while chiplets_remaining > 0:
        next_pos = find_max_distance_position(grid, chiplet_size)
        if next_pos is None:
            rotated_chiplet_size = (chiplet_size[1], chiplet_size[0])
            next_pos = find_max_distance_position(grid, rotated_chiplet_size)
            if next_pos is None:
                return positions  # Skip if no valid placement found
            chiplet_size = rotated_chiplet_size

        x, y = next_pos
        positions.append((x, y))

        for dx in range(chiplet_size[0]):
            for dy in range(chiplet_size[1]):
                grid[x + dx, y + dy] = int(cluster_key.split()[-1]) * 1000

        chiplets_remaining -= 1

    return positions

def visualize_bounding_box(exp_dir, surface_floorplan, embedded_floorplan, title, E_s):
    """
    Visualize bounding boxes for surface and embedded tiers side by side with proper legends.

    Parameters:
    - exp_dir: Directory to save the plot.
    - surface_floorplan: Floorplan data for the surface tier.
    - embedded_floorplan: Floorplan data for the embedded tier.
    - title: Title for the plot.
    - E_s: Edge spacing threshold.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Cluster-specific colors for better visualization
    cluster_colors = {"C1": "red", "C2": "yellow", "C3": "purple", "C4": "blue", "C5": "grey", "C6": "green", "C7": "orange"}

    # Cluster type mapping
    Cluster_type = {
        "C1": "CCD", 
        "C2": "AID",
        "C3": "XCD",
        "C4": "IPD",
        "C5": "HBM",
        "C6": "EMIB",
        "C7": "EMIB2"
    }

    def plot_floorplan(ax, floorplan, tier_title):
        ax.set_title(tier_title, fontsize=16)

        # Compute max bounds including edge spacing
        max_x = max(chiplet["Lower_Left_Corner"][0] + chiplet["Length"] for chiplet in floorplan) + E_s
        max_y = max(chiplet["Lower_Left_Corner"][1] + chiplet["Breadth"] for chiplet in floorplan) + E_s

        # Draw bounding box with edge spacing
        ax.plot(
            [-E_s, max_y, max_y, -E_s, -E_s], [-E_s, -E_s, max_x, max_x, -E_s],
            color="black", linestyle="--", linewidth=2,
            label=f"Bounding Box: {max_x:.2f} x {max_y:.2f} mm (E_s Applied)"
        )

        added_labels = set()  # Track unique labels for legend

        # Draw chiplets with color differentiation
        for chiplet in floorplan:
            x, y = chiplet["Lower_Left_Corner"]
            length = chiplet["Length"]
            breadth = chiplet["Breadth"]
            cluster_key = chiplet["Chiplet"].split("-")[0]
            color = cluster_colors.get(cluster_key, "gray")  # Default to gray if cluster is unknown
            
            ax.add_patch(
                plt.Rectangle(
                    (y, x), breadth, length,
                    facecolor=color, alpha=0.6, edgecolor="black",
                    label=Cluster_type.get(cluster_key, cluster_key) if cluster_key not in added_labels else None
                )
            )
            ax.text(
                y + breadth / 2, x + length / 2,
                chiplet["Chiplet"], fontsize=8, ha="center", va="center"
            )
            added_labels.add(cluster_key)

        # Add legend with real cluster names
        legend_handles = [
            plt.Line2D([0], [0], color=color, lw=4, label=Cluster_type[cluster])
            for cluster, color in cluster_colors.items()
        ]
        
        ax.legend(
            handles=[
            plt.Line2D([0], [0], color=color, lw=4, label=f"{Cluster_type[cluster]}")
            for cluster, color in cluster_colors.items()
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=5,
            fontsize=12,
            title="Cluster Legend",
            title_fontsize=14
        )
        
        ax.set_xlim(-E_s, max_y + E_s)
        ax.set_ylim(-E_s, max_x + E_s)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Y-axis (mm)", fontsize=12)
        ax.set_ylabel("X-axis (mm)", fontsize=12)
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Plot surface and embedded tiers side by side
    plot_floorplan(axs[0], surface_floorplan, "Tier 1: Surface")
    plot_floorplan(axs[1], embedded_floorplan, "Tier 0: Embedded")

    # Add title
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    plt.savefig(f"{exp_dir}/bounding_box_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

# Generate floorplan data
def generate_floorplan_data(cluster_positions, clusters, embed):
    floorplan_data = []
    for cluster_key, positions in cluster_positions.items():
        for idx, (x, y) in enumerate(positions, start=1):
            floorplan_data.append({
                "Chiplet": f"{cluster_key}-{idx}".replace("Cluster ", "C"),
                "Lower_Left_Corner": (x, y),
                "Length": clusters[cluster_key]["length"],
                "Breadth": clusters[cluster_key]["breadth"],
                "location": "embedded" if embed else "surface"
            })
    return floorplan_data

# Main Code
tier1_permutations = list(permutations(clusters_surface.keys()))
tier2_permutations = list(permutations(clusters_embedded.keys()))
i = 0
seen_floorplans = set()
previous_designs = []

for tier1 in tier1_permutations:
    for tier2 in tier2_permutations:
        i += 1
        grid_surface = np.zeros(grid_dims_surface, dtype=int)
        cluster_positions_surface = {cluster: place_chiplets_fixed(grid_surface, cluster, clusters_surface) for cluster in tier1}
        grid_embedded = np.zeros(grid_dims_embedded, dtype=int)
        cluster_positions_embedded = {cluster: place_chiplets_fixed(grid_embedded, cluster, clusters_embedded) for cluster in tier2}
        floorplan_surface = generate_floorplan_data(cluster_positions_surface, clusters_surface, False)
        floorplan_embedded = generate_floorplan_data(cluster_positions_embedded, clusters_embedded, True)
        
        if not all_chiplets_placed(cluster_positions_surface, clusters_surface) or not all_chiplets_placed(cluster_positions_embedded, clusters_embedded):
            continue  # Skip this design
        
        combined_hash = hash_floorplan(floorplan_surface + floorplan_embedded)
        
        # Check uniqueness via hashing and Hamming distance
        if combined_hash in seen_floorplans or any(hamming_distance(floorplan_surface + floorplan_embedded, prev) < 2 for prev in previous_designs):
            continue
        
        seen_floorplans.add(combined_hash)
        previous_designs.append(floorplan_surface + floorplan_embedded)

        exp_dir = f"exp_{i}"
        os.makedirs(exp_dir, exist_ok=True)
        
        visualize_bounding_box(exp_dir, floorplan_surface, floorplan_embedded, "Bounding Box Visualization", E_s)

        # Archive design
        with open(f"{exp_dir}/design.txt", "w") as f:
            f.write(str(floorplan_surface + floorplan_embedded))

        print(f"Archived unique design at iteration {i}")

