import random
import os
import pandas as pd
import matplotlib.pyplot as plt

def calculate_spacing_cost(chiplet_a, chiplet_b, required_spacing):
    xa, ya, wa, ha = chiplet_a["X_Position"], chiplet_a["Y_Position"], chiplet_a["Length"], chiplet_a["Breadth"]
    xb, yb, wb, hb = chiplet_b["X_Position"], chiplet_b["Y_Position"], chiplet_b["Length"], chiplet_b["Breadth"]
    
    dx = xa + wa / 2 - (xb + wb / 2)
    dy = ya + ha / 2 - (yb + hb / 2)
    distance = (dx ** 2 + dy ** 2) ** 0.5
    
    perimeter_a = 2 * (wa + ha)
    perimeter_b = 2 * (wb + hb)
    
    if distance > 0.25 * (perimeter_a + perimeter_b):
        return 0  # Too far, no nudge required
    
    return abs(distance - required_spacing)

def nudge_chiplet(chiplet, delta_x=0.05, delta_y=0.05):
    chiplet["X_Position"] += random.choice([-1, 1]) * delta_x
    chiplet["Y_Position"] += random.choice([-1, 1]) * delta_y

def get_neighbors(chiplet, floorplan, spacing, E_s=0.05):
    neighbors = []
    for other_chiplet in floorplan:
        if other_chiplet == chiplet:
            continue
        required_spacing = spacing.get((chiplet["Chiplet_Type"], other_chiplet["Chiplet_Type"]), E_s)
        cost = calculate_spacing_cost(chiplet, other_chiplet, required_spacing)
        if cost > 0:
            neighbors.append((other_chiplet, required_spacing, cost))
    return neighbors

def visualize_floorplan(floorplan, output_dir, iteration):
    fig, ax = plt.subplots(figsize=(10, 10))
    cluster_colors = {"CCD": "red", "XCD": "purple", "IPD": "blue", 
                      "EMIB": "green", "EMIB2": "orange", "Interposer": "black"}
    
    ax.set_title(f"Floorplan Iteration {iteration}")
    
    for chiplet in floorplan:
        x, y = chiplet["X_Position"], chiplet["Y_Position"]
        length, breadth = chiplet["Length"], chiplet["Breadth"]
        color = cluster_colors.get(chiplet["Chiplet_Type"], "gray")
        ax.add_patch(plt.Rectangle((y, x), breadth, length, facecolor=color, alpha=0.6, edgecolor="black"))
        ax.text(y + breadth / 2, x + length / 2, chiplet["Chiplet_Name"], fontsize=8, ha="center", va="center")
    
    ax.set_xlim(-1, max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan) + 1)
    ax.set_ylim(-1, max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan) + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Y-axis (mm)")
    ax.set_ylabel("X-axis (mm)")
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    
    plt.savefig(os.path.join(output_dir, f"floorplan_iteration_{iteration}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

Cluster_type = {
    "C1": "CCD", "C3": "XCD", "C4": "IPD",
    "C6": "EMIB", "C7": "EMIB2"
}

def DRC_annealing(floorplan, spacing, iterations=100, E_s=0.05, output_dir="dr_c_annealing"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for iteration in range(iterations):
        total_cost = 0
        for chiplet in floorplan:
            neighbors = get_neighbors(chiplet, floorplan, spacing, E_s)
            for neighbor, required_spacing, cost in neighbors:
                total_cost += cost
                if cost > 0:
                    nudge_chiplet(chiplet)
        
        print(f"Iteration {iteration}: Total Cost = {total_cost:.4f}")
        
        df = pd.DataFrame(floorplan)
        df.to_csv(os.path.join(output_dir, f"floorplan_iteration_{iteration}.csv"), index=False)
        
        visualize_floorplan(floorplan, output_dir, iteration)

    return floorplan

def generate_floorplan_data(cluster_positions, clusters, z_position):
    floorplan_data = []
    for cluster_key, positions in cluster_positions.items():
        for idx, (x, y, width, height, rotated) in enumerate(positions, start=1):
            chiplet_type = Cluster_type.get(cluster_key.replace("Cluster ", "C"), cluster_key)
            floorplan_data.append({
                "Chiplet_Type": chiplet_type,
                "Chiplet_Name": f"{chiplet_type}{idx}",
                "X_Position": x,
                "Y_Position": y,
                "Z_Position": z_position,
                "Length": width,
                "Breadth": height,
                "Rotated": rotated  # Store whether it was rotated
            })
    return floorplan_data

def main():
    spacing = {
        ("XCD", "IPD"): 0.1,
        ("IPD", "XCD"): 0.1,
        ("XCD", "XCD"): 0.3,
        ("CCD", "IPD"): 0.2,
        ("IPD", "CCD"): 0.2,
        ("EMIB", "XCD"): 0.1,
        ("EMIB", "EMIB"): 0.2
    }
    
    clusters_surface = {
        "Cluster 1": {"count": 4, "length": 2, "breadth": 2},
        "Cluster 3": {"count": 2, "length": 11, "breadth": 8.2},
        "Cluster 4": {"count": 8, "length": 2, "breadth": 1}
    }
    
    clusters_embedded = {
        "Cluster 6": {"count": 8, "length": 3, "breadth": 1},
        "Cluster 7": {"count": 2, "length": 4, "breadth": 1}
    }
    
    Cluster_type = {
        "C1": "CCD", "C3": "XCD", "C4": "IPD",
        "C6": "EMIB", "C7": "EMIB2"
    }
    
    def place_chiplets_fixed(cluster_key, clusters, existing_chiplets, max_width, max_height, spacing):
        positions = []
        chiplets_remaining = clusters[cluster_key]["count"]
        rotate = random.choice([True, False])
        chiplet_size = (clusters[cluster_key]["length"], clusters[cluster_key]["breadth"]) if rotate else (clusters[cluster_key]["breadth"], clusters[cluster_key]["length"])
        max_attempts = 7
        while chiplets_remaining > 0:
            attempts = 0
            best_attempt = None
            lowest_overlap = float("inf")
            while attempts < max_attempts:
                x = random.uniform(0, max_width - chiplet_size[0])
                y = random.uniform(0, max_height - chiplet_size[1])
                total_overlap = sum(
                    not ((x + chiplet_size[0] < ex) or (x > ex + ew) or (y + chiplet_size[1] < ey) or (y > ey + eh))
                    for ex, ey, ew, eh in existing_chiplets
                )
                if total_overlap == 0:
                    positions.append((x, y, chiplet_size[0], chiplet_size[1], rotate))
                    existing_chiplets.append((x, y, chiplet_size[0], chiplet_size[1]))
                    break
                else:
                    if total_overlap < lowest_overlap:
                        lowest_overlap = total_overlap
                        best_attempt = (x, y, chiplet_size[0], chiplet_size[1], rotate)
                attempts += 1
            if attempts == max_attempts and best_attempt:
                positions.append(best_attempt)
                existing_chiplets.append(best_attempt[:4])
            chiplets_remaining -= 1
        return positions
    
    max_width, max_height = 20, 20
    existing_chiplets_surface, existing_chiplets_embedded = [], []
    cluster_positions_surface = {}
    cluster_positions_embedded = {}
    
    for cluster in clusters_surface.keys():
        cluster_positions_surface[cluster] = place_chiplets_fixed(cluster, clusters_surface, existing_chiplets_surface, max_width, max_height, spacing)
    
    for cluster in clusters_embedded.keys():
        cluster_positions_embedded[cluster] = place_chiplets_fixed(cluster, clusters_embedded, existing_chiplets_embedded, max_width, max_height, spacing)

    floorplan_surface = generate_floorplan_data(cluster_positions_surface, clusters_surface, 2)
    floorplan_embedded = generate_floorplan_data(cluster_positions_embedded, clusters_embedded, 1)
    combined_floorplan = floorplan_surface + floorplan_embedded

    optimized_floorplan = DRC_annealing(combined_floorplan, spacing, iterations=100, E_s=0.05, output_dir="dr_c_annealing")
    
if __name__ == "__main__":
    main()
