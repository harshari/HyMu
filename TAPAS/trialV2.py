import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import random
import pandas as pd
from itertools import permutations

# Define chiplet clusters
clusters_surface = {
    "Cluster 1": {"count": 4, "length": 2, "breadth": 2},
    "Cluster 2": {"count": 0, "length": 4, "breadth": 2},
    "Cluster 3": {"count": 2, "length": 11, "breadth": 8.2},
    "Cluster 4": {"count": 8, "length": 2, "breadth": 1},
}

clusters_embedded = {
    "Cluster 6": {"count": 8, "length": 3, "breadth": 1},
    "Cluster 7": {"count": 2, "length": 4, "breadth": 1}
}

Cluster_type = {
    "C1": "CCD", 
    "C2": "AID",
    "C3": "XCD",
    "C4": "IPD",
    "C5": "HBM",
    "C6": "EMIB",
    "C7": "EMIB2"
}

E_s = 0.5
random.seed(42)

def hash_floorplan(floorplan_data):
    sorted_data = sorted(
        [(chiplet["Chiplet_Name"], chiplet["X_Position"], chiplet["Y_Position"], chiplet["Length"], chiplet["Breadth"])]
        for chiplet in floorplan_data
    )
    return hashlib.sha256(str(sorted_data).encode('utf-8')).hexdigest()

def place_chiplets_fixed(cluster_key, clusters, existing_chiplets, max_width, max_height, spacing):
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]
    chiplet_size = (clusters[cluster_key]["breadth"], clusters[cluster_key]["length"])

    while chiplets_remaining > 0:
        attempts = 0
        placed = False

        while attempts < 5:  # Try up to 50 times to avoid overlap
            x = random.uniform(0, max_width - chiplet_size[0])
            y = random.uniform(0, max_height - chiplet_size[1])

            # ✅ Check for overlap within the same tier (strict condition)
            if all(
                (x + chiplet_size[0] <= ex or x >= ex + ew) and
                (y + chiplet_size[1] <= ey or y >= ey + eh)
                for ex, ey, ew, eh in existing_chiplets
            ):
                positions.append((x, y, chiplet_size[0], chiplet_size[1]))
                existing_chiplets.append((x, y, chiplet_size[0], chiplet_size[1]))
                placed = True
                break

            attempts += 1

        if not placed:  
            # If failed after 50 attempts, just place it anyway and log overlap (for debugging)
            print(f"⚠ Overlap warning: Placing {cluster_key} chiplet with possible overlap")
            positions.append((x, y, chiplet_size[0], chiplet_size[1]))
            existing_chiplets.append((x, y, chiplet_size[0], chiplet_size[1]))

        chiplets_remaining -= 1

    return positions

def generate_floorplan_data(cluster_positions, clusters, z_position):
    floorplan_data = []

    for cluster_key, positions in cluster_positions.items():
        for idx, (x, y, width, height) in enumerate(positions, start=1):
            chiplet_type = Cluster_type.get(cluster_key.replace("Cluster ", "C"), cluster_key)
            floorplan_data.append({
                "Chiplet_Type": chiplet_type,
                "Chiplet_Name": f"{chiplet_type}{idx}",
                "X_Position": x,
                "Y_Position": y,
                "Z_Position": z_position,
                "Length": width,
                "Breadth": height
            })
    return floorplan_data

def compute_bounding_box(floorplan):
    max_x = max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan)
    max_y = max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan)
    return max_x, max_y

def add_interposer_to_floorplan(floorplan_data, bounding_box_size):
    floorplan_data.append({
        "Chiplet_Type": "Interposer",
        "Chiplet_Name": "Interposer",
        "X_Position": 0,
        "Y_Position": 0,
        "Z_Position": 0,
        "Length": bounding_box_size[0],
        "Breadth": bounding_box_size[1]
    })
    return floorplan_data

def visualize_bounding_box(exp_dir, floorplan_surface, floorplan_embedded, floorplan_interposer, title):
    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    cluster_colors = {"CCD": "red", "AID": "yellow", "XCD": "purple", "IPD": "blue", "HBM": "grey", "EMIB": "green", "EMIB2": "orange", "Interposer": "black"}

    def plot_floorplan(ax, floorplan, tier_title):
        ax.set_title(tier_title, fontsize=16)
        max_x = max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan) + E_s
        max_y = max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan) + E_s
        ax.plot([-E_s, max_y, max_y, -E_s, -E_s], [-E_s, -E_s, max_x, max_x, -E_s], color="black", linestyle="--", linewidth=2)
        
        for chiplet in floorplan:
            x, y = chiplet["X_Position"], chiplet["Y_Position"]
            length, breadth = chiplet["Length"], chiplet["Breadth"]
            color = cluster_colors.get(chiplet["Chiplet_Type"], "gray")
            ax.add_patch(plt.Rectangle((y, x), breadth, length, facecolor=color, alpha=0.6, edgecolor="black"))
            ax.text(y + breadth / 2, x + length / 2, chiplet["Chiplet_Name"], fontsize=8, ha="center", va="center")

        ax.set_xlim(-E_s, max_y + E_s)
        ax.set_ylim(-E_s, max_x + E_s)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Y-axis (mm)")
        ax.set_ylabel("X-axis (mm)")
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    plot_floorplan(axs[0], floorplan_surface, "Tier 2: Surface")
    plot_floorplan(axs[1], floorplan_embedded, "Tier 1: Embedded")
    plot_floorplan(axs[2], floorplan_interposer, "Tier 0: Interposer")

    fig.suptitle(title)
    plt.savefig(f"{exp_dir}/bounding_box_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

def archive_design(exp_dir, floorplan):
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(floorplan)
    df.to_csv(f"{exp_dir}/design.csv", index=False)

def main():
    seen_floorplans = set()
    max_width, max_height = 30, 30

    i = 1
    while i <= 5:  
        print(f"Iteration {i}: Generating floorplan")
        cluster_positions_surface, cluster_positions_embedded = {}, {}
        existing_chiplets_surface, existing_chiplets_embedded = [], []

        for cluster in clusters_surface:
            cluster_positions_surface[cluster] = place_chiplets_fixed(cluster, clusters_surface, existing_chiplets_surface, max_width, max_height, E_s)

        for cluster in clusters_embedded:
            cluster_positions_embedded[cluster] = place_chiplets_fixed(cluster, clusters_embedded, existing_chiplets_embedded, max_width, max_height, E_s)

        floorplan_surface = generate_floorplan_data(cluster_positions_surface, clusters_surface, 2)
        floorplan_embedded = generate_floorplan_data(cluster_positions_embedded, clusters_embedded, 1)

        bounding_box_size = compute_bounding_box(floorplan_surface + floorplan_embedded)

        floorplan_final = floorplan_surface + floorplan_embedded
        floorplan_final = add_interposer_to_floorplan(floorplan_final, bounding_box_size)

        exp_dir = f"exp_1"
        archive_design(exp_dir, floorplan_final)
        visualize_bounding_box(exp_dir, floorplan_surface, floorplan_embedded, floorplan_final, "Bounding Box Visualization")

        print(f"Archived unique design at {exp_dir}")

if __name__ == "__main__":
    main()
