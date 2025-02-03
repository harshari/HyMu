import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import random
import pandas as pd
import random
import shutil

# Define chiplet clusters
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

associations = {
    ("XCD", "IPD"): 0.8,  # XCD should be close to IPD
    ("IPD", "XCD"): 0.8,
    ("XCD", "XCD"): 1,  # XCDs should be close together
    ("CCD", "IPD"): 0.7,  # CCD should be close to IPD
    ("IPD", "CCD"): 0.7,
    ("EMIB", "XCD"): 1,  # EMIB should be near XCD (inter-tier)
    ("EMIB", "EMIB"): 1  # EMIBs should cluster together
}


E_s = 0.2
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
    
    rotate = random.choice([True, False])
    chiplet_size = (clusters[cluster_key]["length"], clusters[cluster_key]["breadth"]) if rotate else (clusters[cluster_key]["breadth"], clusters[cluster_key]["length"])

    max_attempts = 7  # Limit retries per chiplet
    best_positions = []
    min_total_overlap = float("inf")

    while chiplets_remaining > 0:
        attempts = 0
        best_attempt = None
        lowest_overlap = float("inf")

        while attempts < max_attempts:
            x = random.uniform(0, max_width - chiplet_size[0])
            y = random.uniform(0, max_height - chiplet_size[1])
            
            # Calculate total overlap count for this position
            total_overlap = sum(
                not ((x + chiplet_size[0] < ex) or (x > ex + ew) or (y + chiplet_size[1] < ey) or (y > ey + eh))
                for ex, ey, ew, eh in existing_chiplets
            )

            if total_overlap == 0:  # Perfect placement, no overlap
                positions.append((x, y, chiplet_size[0], chiplet_size[1], rotate))
                existing_chiplets.append((x, y, chiplet_size[0], chiplet_size[1]))
                break  # Move to next chiplet
            else:
                # Keep track of the best attempt (least overlap)
                if total_overlap < lowest_overlap:
                    lowest_overlap = total_overlap
                    best_attempt = (x, y, chiplet_size[0], chiplet_size[1], rotate)

            attempts += 1

        if attempts == max_attempts and best_attempt:
            # If we exhausted attempts, use the best attempt with minimum overlap
            print(f"WARNING: Could not place {cluster_key} perfectly. Using best attempt with {lowest_overlap} overlaps.")
            positions.append(best_attempt)
            existing_chiplets.append(best_attempt[:4])  # Store without rotation info
        
        chiplets_remaining -= 1

    return positions

def compute_cost(floorplan, associations):
    """Compute cost based on chiplet distances, defaulting to 0 association if not specified."""
    cost = 0
    for c1 in floorplan:
        for c2 in floorplan:
            if c1 == c2:
                continue  # Skip self-comparison
            key = (c1["Chiplet_Type"], c2["Chiplet_Type"])
            closeness = associations.get(key, 0)  # Default to 0 if not specified
            
            # Compute Euclidean distance
            distance = np.sqrt(
                (c1["X_Position"] - c2["X_Position"])**2 + 
                (c1["Y_Position"] - c2["Y_Position"])**2 +
                (c1["Z_Position"] - c2["Z_Position"])**2
            )
            
            cost += (1 - closeness) * distance  # Closer chiplets reduce cost if closeness is high

    return cost


def swap_or_nudge(floorplan):
    """Randomly swap or nudge a chiplet slightly to explore new layouts."""
    new_floorplan = floorplan.copy()
    
    if random.random() < 0.5:  # 50% chance to swap
        c1, c2 = random.sample(new_floorplan, 2)
        c1["X_Position"], c2["X_Position"] = c2["X_Position"], c1["X_Position"]
        c1["Y_Position"], c2["Y_Position"] = c2["Y_Position"], c1["Y_Position"]
    else:  # 50% chance to nudge
        c = random.choice(new_floorplan)
        c["X_Position"] += random.uniform(-0.5, 0.5)
        c["Y_Position"] += random.uniform(-0.5, 0.5)
    
    return new_floorplan

def simulated_annealing(floorplan, associations, exp_dir, iterations=50, temperature=100, cooling_rate=0.95):
    """Optimizes the floorplan using simulated annealing while avoiding duplicate designs."""
    current_cost = compute_cost(floorplan, associations)
    best_floorplan = floorplan.copy()
    best_cost = current_cost

    optimizer_dir = os.path.join(exp_dir, "optimizer")
    os.makedirs(optimizer_dir, exist_ok=True)

    log_file = os.path.join(optimizer_dir, "log.csv")
    log_data = []

    seen_hashes = set()  # Store unique hashes

    for i in range(iterations):
        new_floorplan = swap_or_nudge(floorplan)
        new_cost = compute_cost(new_floorplan, associations)
        new_hash = hash_floorplan(new_floorplan)

        # Skip duplicate designs
        if new_hash in seen_hashes:
            continue  # Try another variation

        seen_hashes.add(new_hash)

        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            floorplan = new_floorplan
            current_cost = new_cost
            if new_cost < best_cost:
                best_floorplan = new_floorplan
                best_cost = new_cost

        temperature *= cooling_rate

        iter_dir = os.path.join(optimizer_dir, f"iter_{i}")
        archive_design(iter_dir, floorplan)
        visualize_floorplan(iter_dir, [floorplan], f"Iteration {i} Cost: {current_cost:.2f}")

        log_data.append([i, current_cost])
        print(f"Iteration {i}: Cost = {current_cost:.2f}")

    pd.DataFrame(log_data, columns=["Iteration", "Cost"]).to_csv(log_file, index=False)

    # **Save Final Design**
    final_hash = hash_floorplan(best_floorplan)
    final_file = os.path.join(optimizer_dir, "final_design.csv")
    
    if final_hash not in seen_hashes:  # Ensure final design is unique
        seen_hashes.add(final_hash)
        archive_design(optimizer_dir, best_floorplan)  
        pd.DataFrame(best_floorplan).to_csv(final_file, index=False)

    return best_floorplan

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

def visualize_floorplan(exp_dir, tiers, title):
    fig, axs = plt.subplots(1, 4, figsize=(20, 8))
    cluster_colors = {"CCD": "red", "XCD": "purple", "IPD": "blue", 
                      "EMIB": "green", "EMIB2": "orange", "Interposer": "black"}

    titles = ["Tier 0: Interposer", "Tier 1: Embedded", "Tier 2: Surface", "Combined"]

    for ax, floorplan, tier_title in zip(axs, tiers, titles):
        ax.set_title(tier_title, fontsize=12)
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

    fig.suptitle(title, fontsize=14)
    plt.savefig(f"{exp_dir}/floorplan_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

def archive_design(exp_dir, floorplan):
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(floorplan)
    df.to_csv(f"{exp_dir}/design.csv", index=False)

def main():
    seen_floorplans = set()
    max_width, max_height = 20, 20
    i = 1

    while i <= 5:  
        print(f"Iteration {i}: Generating floorplan")
        cluster_positions_surface, cluster_positions_embedded = {}, {}
        existing_chiplets_surface, existing_chiplets_embedded = [], []

        for cluster in clusters_surface.keys():
            cluster_positions_surface[cluster] = place_chiplets_fixed(cluster, clusters_surface, existing_chiplets_surface, max_width, max_height, E_s)

        for cluster in clusters_embedded.keys():
            cluster_positions_embedded[cluster] = place_chiplets_fixed(cluster, clusters_embedded, existing_chiplets_embedded, max_width, max_height, E_s)

        floorplan_surface = generate_floorplan_data(cluster_positions_surface, clusters_surface, 2)
        floorplan_embedded = generate_floorplan_data(cluster_positions_embedded, clusters_embedded, 1)

        combined_floorplan = floorplan_surface + floorplan_embedded
        bounding_box_size = compute_bounding_box(combined_floorplan)
        final_floorplan = add_interposer_to_floorplan(combined_floorplan, bounding_box_size)

        combined_hash = hash_floorplan(final_floorplan)
        if combined_hash in seen_floorplans:
            continue

        seen_floorplans.add(combined_hash)

        exp_dir = f"exp_{i}"
        archive_design(exp_dir, final_floorplan)
        visualize_floorplan(exp_dir, [final_floorplan[-1:], floorplan_embedded, floorplan_surface, final_floorplan], "Floorplan Visualization")

        # Run simulated annealing optimization
        optimized_floorplan = simulated_annealing(final_floorplan, associations, exp_dir)

        # Archive post-optimization
        archive_design(exp_dir, optimized_floorplan)
        visualize_floorplan(exp_dir, [optimized_floorplan[-1:], floorplan_embedded, floorplan_surface, optimized_floorplan], "Post-Optimization Floorplan")


        print(f"Archived unique design at iteration {i}")
        i += 1

if __name__ == "__main__":
    main()
