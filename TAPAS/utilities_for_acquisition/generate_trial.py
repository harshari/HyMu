import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import random
import pandas as pd

clusters_surface = {
    "Cluster 1": {"count": 4, "length": 2.5, "breadth": 2},
    "Cluster 2": {"count": 0, "length": 4, "breadth": 2},
    "Cluster 3": {"count": 2, "length": 2, "breadth": 2},
    "Cluster 4": {"count": 8, "length": 2, "breadth": 1},
    "Cluster 5": {"count": 0, "length": 1, "breadth": 1}
}

clusters_embedded = {
    "Cluster 6": {"count": 8, "length": 3, "breadth": 1},
    "Cluster 7": {"count": 2, "length": 4, "breadth": 1}
}

max_width, max_height = 10, 10  
E_s = 0.1
random.seed(42)  

def hash_floorplan(floorplan_data):
    sorted_data = sorted(
        [(chiplet["Chiplet_Type"], chiplet["X_Position"], chiplet["Y_Position"], chiplet["Length"], chiplet["Breadth"])]
        for chiplet in floorplan_data
    )
    return hashlib.sha256(str(sorted_data).encode('utf-8')).hexdigest()

def is_valid_position(existing_chiplets, x, y, chiplet_size, spacing):
    for ex_x, ex_y, ex_width, ex_height in existing_chiplets:
        if (x < ex_x + ex_width + spacing and x + chiplet_size[0] > ex_x - spacing and
            y < ex_y + ex_height + spacing and y + chiplet_size[1] > ex_y - spacing):
            return False
    return True

def find_farthest_position(existing_chiplets, chiplet_size, max_width, max_height, spacing):
    best_pos = None
    max_distance = -1
    for _ in range(100):  
        x = random.uniform(0, max_width - chiplet_size[0])
        y = random.uniform(0, max_height - chiplet_size[1])
        if is_valid_position(existing_chiplets, x, y, chiplet_size, spacing):
            distance = np.sqrt(x**2 + y**2)  
            if distance > max_distance:
                max_distance = distance
                best_pos = (x, y)
    return best_pos

def place_chiplets_fixed(cluster_key, clusters, existing_chiplets, max_width, max_height, spacing):
    positions = []
    chiplets_remaining = clusters[cluster_key]["count"]
    chiplet_size = (clusters[cluster_key]["breadth"], clusters[cluster_key]["length"])

    while chiplets_remaining > 0:
        next_pos = find_farthest_position(existing_chiplets, chiplet_size, max_width, max_height, spacing)
        if next_pos is None:
            return positions  

        x, y = next_pos
        positions.append((x, y, chiplet_size[0], chiplet_size[1]))
        existing_chiplets.append((x, y, chiplet_size[0], chiplet_size[1]))
        chiplets_remaining -= 1

    return positions

def generate_floorplan_data(cluster_positions, clusters, embed):
    floorplan_data = []
    z_position = 1 if embed else 2  

    for cluster_key, positions in cluster_positions.items():
        for idx, (x, y, breadth, length) in enumerate(positions, start=1):
            floorplan_data.append({
                "Chiplet_Type": cluster_key.replace("Cluster ", "C"),
                "Index": f"{cluster_key.replace('Cluster ', 'C')}{idx}",
                "X_Position": x,
                "Y_Position": y,
                "Z_Position": z_position,  
                "Length": length,
                "Breadth": breadth
            })
    return floorplan_data

def visualize_bounding_box(exp_dir, surface_floorplan, embedded_floorplan, title, E_s):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    cluster_colors = {"C1": "red", "C2": "yellow", "C3": "purple", "C4": "blue", "C5": "grey", "C6": "green", "C7": "orange"}

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

        max_x = max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan) + E_s
        max_y = max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan) + E_s

        ax.plot([-E_s, max_y, max_y, -E_s, -E_s], [-E_s, -E_s, max_x, max_x, -E_s], color="black", linestyle="--", linewidth=2)

        added_labels = set()
        for chiplet in floorplan:
            x, y = chiplet["X_Position"], chiplet["Y_Position"]
            length, breadth = chiplet["Length"], chiplet["Breadth"]
            cluster_key = chiplet["Chiplet_Type"]
            color = cluster_colors.get(cluster_key, "gray")
            
            ax.add_patch(plt.Rectangle((y, x), breadth, length, facecolor=color, alpha=0.6, edgecolor="black",
                label=Cluster_type.get(cluster_key, cluster_key) if cluster_key not in added_labels else None))
            ax.text(y + breadth / 2, x + length / 2, chiplet["Index"], fontsize=8, ha="center", va="center")
            added_labels.add(cluster_key)

        ax.set_xlim(-E_s, max_y + E_s)
        ax.set_ylim(-E_s, max_x + E_s)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Y-axis (mm)", fontsize=12)
        ax.set_ylabel("X-axis (mm)", fontsize=12)
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    plot_floorplan(axs[0], surface_floorplan, "Tier 1: Surface")
    plot_floorplan(axs[1], embedded_floorplan, "Tier 0: Embedded")

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"{exp_dir}/bounding_box_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    i = 0
    seen_floorplans = set()

    for _ in range(10):  
        i += 1
        existing_chiplets_surface = []
        existing_chiplets_embedded = []

        cluster_positions_surface = {cluster: place_chiplets_fixed(cluster, clusters_surface, existing_chiplets_surface, max_width, max_height, E_s) for cluster in clusters_surface}
        cluster_positions_embedded = {cluster: place_chiplets_fixed(cluster, clusters_embedded, existing_chiplets_embedded, max_width, max_height, E_s) for cluster in clusters_embedded}

        floorplan_surface = generate_floorplan_data(cluster_positions_surface, clusters_surface, False)
        floorplan_embedded = generate_floorplan_data(cluster_positions_embedded, clusters_embedded, True)

        combined_hash = hash_floorplan(floorplan_surface + floorplan_embedded)
        if combined_hash in seen_floorplans:
            continue

        seen_floorplans.add(combined_hash)
        exp_dir = f"exp_{i}"
        os.makedirs(exp_dir, exist_ok=True)
        
        visualize_bounding_box(exp_dir, floorplan_surface, floorplan_embedded, "Bounding Box Visualization", E_s)

        df = pd.DataFrame(floorplan_surface + floorplan_embedded)
        df.to_csv(f"{exp_dir}/floorplan.csv", index=False)

        print(f"Archived design {i} in {exp_dir}")

if __name__ == "__main__":
    main()
