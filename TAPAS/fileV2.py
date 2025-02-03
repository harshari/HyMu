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

# Define associations (closeness between different chiplets)
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

def adaptive_cooling(temperature, prev_cost, current_cost):
    """Dynamically adjust cooling rate based on cost improvement."""
    if current_cost < prev_cost:  
        cooling_factor = 0.99  # Small decrease → slow cooling
    else:
        cooling_factor = 0.95  # Stagnant cost → faster cooling

    return temperature * cooling_factor


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
            positions.append(best_attempt)
            existing_chiplets.append(best_attempt[:4])  # Store without rotation info
        
        chiplets_remaining -= 1

    return positions

def compute_cost(floorplan, associations, overlap_weight=.2, closeness_weight=.5, area_weight=.3):
    """
    Computes the cost of the current floorplan based on:
    - Overlap within the same tier (high penalty)
    - Closeness constraint violations (association score)
    - Bounding box area (minimization objective)
    """
    cost1 = 0
    cost2 = 0
    cost3 = 0
    # **1. Overlap Penalty (Hard Constraint)**
    for i, chiplet_a in enumerate(floorplan):
        for j, chiplet_b in enumerate(floorplan):
            if i >= j:
                continue

            xa, ya, wa, ha = chiplet_a["X_Position"], chiplet_a["Y_Position"], chiplet_a["Length"], chiplet_a["Breadth"]
            xb, yb, wb, hb = chiplet_b["X_Position"], chiplet_b["Y_Position"], chiplet_b["Length"], chiplet_b["Breadth"]

            # Check for overlap
            if not ((xa + wa <= xb) or (xa >= xb + wb) or (ya + ha <= yb) or (ya >= yb + hb)):
                overlap_area = (min(xa + wa, xb + wb) - max(xa, xb)) * (min(ya + ha, yb + hb) - max(ya, yb))
                cost1 += overlap_weight * overlap_area  # Heavy penalty for overlap

    # **2. Closeness Cost (Encourages Grouping)**
    for (chiplet_type1, chiplet_type2), desired_closeness in associations.items():
        chiplets_1 = [c for c in floorplan if c["Chiplet_Type"] == chiplet_type1]
        chiplets_2 = [c for c in floorplan if c["Chiplet_Type"] == chiplet_type2]

        for c1 in chiplets_1:
            for c2 in chiplets_2:
                xa, ya, wa, ha = c1["X_Position"], c1["Y_Position"], c1["Length"], c1["Breadth"]
                xb, yb, wb, hb = c2["X_Position"], c2["Y_Position"], c2["Length"], c2["Breadth"]

                # **New Distance Calculation**
                dx = abs(xa - xb)
                dy = abs(ya - yb)

                # Manhattan distance (better for alignment)
                distance = dx + dy  

                # **If association is 1, force side-by-side alignment**
                if desired_closeness == 1:
                    # If chiplets are NOT perfectly aligned, add a heavy penalty
                    if not ((dx == 0 and (ya + ha == yb or yb + hb == ya)) or  # Perfect vertical stacking
                            (dy == 0 and (xa + wa == xb or xb + wb == xa))):  # Perfect horizontal stacking
                        cost2 += closeness_weight * 10  # Huge penalty for misalignment
                else:
                    # Normal closeness penalty for other associations
                    closeness_penalty = closeness_weight * (distance - (1 - desired_closeness) * 10) ** 2
                    cost2 += closeness_penalty

    # **3. Bounding Box Area Penalty (Encourages Compact Design)**
    max_x = max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan)
    max_y = max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan)
    total_area = max_x * max_y
    cost3 += area_weight * total_area  # Encourage compact designs
    print(f"{cost1} + {cost3} = {cost1+cost3}")
    return cost1+cost3


def make_associations(associations, cluster_types):
    """ Ensure all chiplet type pairs have an association, defaulting to 0 if missing.
        Also programmatically ensure high association to the interposer. """
    
    filled_associations = {}

    # Convert provided associations into a dictionary
    for (chiplet_type1, chiplet_type2), closeness in associations.items():
        filled_associations[(chiplet_type1, chiplet_type2)] = closeness
        filled_associations[(chiplet_type2, chiplet_type1)] = closeness  # Symmetric

    # Ensure all pairs exist in the dictionary
    for type1 in cluster_types:
        for type2 in cluster_types:
            if (type1, type2) not in filled_associations:
                filled_associations[(type1, type2)] = 0  # Default to 0 closeness
    
    # Add high closeness to interposer for all chiplet types
    for type1 in cluster_types:
        filled_associations[(type1, "Interposer")] = 1.0  # High closeness for interposer
        filled_associations[("Interposer", type1)] = 1.0  # High closeness for interposer

    return filled_associations


import random

def swap_or_nudge(floorplan, temperature, iteration, total_iterations):
    """Nudges chiplets to resolve overlap. Swaps & Rotations only in early iterations."""
    new_floorplan = floorplan.copy()
    max_movement = min(temperature * 2, 1.0)
    overlaps_exist = False

    def check_overlap(chiplet_a, chiplet_b):
        xa, ya, wa, ha = chiplet_a["X_Position"], chiplet_a["Y_Position"], chiplet_a["Length"], chiplet_a["Breadth"]
        xb, yb, wb, hb = chiplet_b["X_Position"], chiplet_b["Y_Position"], chiplet_b["Length"], chiplet_b["Breadth"]
        return not ((xa + wa <= xb) or (xa >= xb + wb) or (ya + ha <= yb) or (ya >= yb + hb))

    def calculate_overlap_area(chiplet_a, chiplet_b):
        xa, ya, wa, ha = chiplet_a["X_Position"], chiplet_a["Y_Position"], chiplet_a["Length"], chiplet_a["Breadth"]
        xb, yb, wb, hb = chiplet_b["X_Position"], chiplet_b["Y_Position"], chiplet_b["Length"], chiplet_b["Breadth"]

        if not check_overlap(chiplet_a, chiplet_b):
            return 0  # No overlap

        overlap_width = min(xa + wa, xb + wb) - max(xa, xb)
        overlap_height = min(ya + ha, yb + hb) - max(ya, yb)
        
        return overlap_width * overlap_height

    def nudge_chiplets_based_on_overlap(chiplet_a, chiplet_b, overlap_area):
        area_a = chiplet_a["Length"] * chiplet_a["Breadth"]
        area_b = chiplet_b["Length"] * chiplet_b["Breadth"]
        
        overlap_ratio_a = overlap_area / area_a
        overlap_ratio_b = overlap_area / area_b
        
        movement_a = random.choice([0, 1]) * max_movement * overlap_ratio_a
        movement_b = random.choice([0, 1]) * max_movement * overlap_ratio_b
        
        chiplet_a["X_Position"] += movement_a
        chiplet_a["Y_Position"] += movement_a
        chiplet_b["X_Position"] -= movement_b
        chiplet_b["Y_Position"] -= movement_b

    # **Resolve Overlaps First**
    for i, chiplet_a in enumerate(new_floorplan):
        for j, chiplet_b in enumerate(new_floorplan):
            if i >= j:
                continue
            overlap_area = calculate_overlap_area(chiplet_a, chiplet_b)
            
            if overlap_area > 0:
                overlaps_exist = True
                # Nudge both chiplets based on overlap area and size
                nudge_chiplets_based_on_overlap(chiplet_a, chiplet_b, overlap_area)

    # **Only Swap and Rotate in the First 10% Iterations**
    if iteration < 0.01 * total_iterations:
        tier_chiplets = {}
        for chiplet in new_floorplan:
            if chiplet["Z_Position"] not in tier_chiplets:
                tier_chiplets[chiplet["Z_Position"]] = []
            tier_chiplets[chiplet["Z_Position"]].append(chiplet)

        for tier, chiplets in tier_chiplets.items():
            if len(chiplets) > 1:
                c1, c2 = random.sample(chiplets, 2)
                c1["X_Position"], c2["X_Position"] = c2["X_Position"], c1["X_Position"]
                c1["Y_Position"], c2["Y_Position"] = c2["Y_Position"], c1["Y_Position"]

                if random.random() < 0.1:
                    c1["Rotated"] = not c1["Rotated"]
                    c2["Rotated"] = not c2["Rotated"]

    return new_floorplan


def simulated_annealing_V2(floorplan, associations, exp_dir, iterations=500, temperature=100, cooling_factor=0.95, log_interval=10):
    """Optimizes the floorplan using simulated annealing with a better perturbation and cooling schedule."""
    
    # Initialize variables
    total_iterations = iterations
    current_cost = compute_cost(floorplan, associations)
    best_floorplan = floorplan.copy()
    best_cost = current_cost
    print("Initial Cost of the system is", best_cost)

    optimizer_dir = os.path.join(exp_dir, "optimizer")
    os.makedirs(optimizer_dir, exist_ok=True)

    log_file = os.path.join(optimizer_dir, "log.csv")
    log_data = []
    seen_hashes = set()

    # Iterate through the number of iterations
    for i in range(iterations):
        # **Apply perturbation (swap or nudge)**: Introduce better perturbations
        new_floorplan = swap_or_nudge(floorplan, temperature, i, total_iterations)
        new_cost = compute_cost(new_floorplan, associations)
        new_hash = hash_floorplan(new_floorplan)

        # **Avoid duplicate designs**: Ensure that we don't revisit the same design
        if new_hash in seen_hashes:
            continue
        seen_hashes.add(new_hash)

        # **Acceptance criterion (Simulated Annealing): Accept better or worse solutions with probability**
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            floorplan = new_floorplan
            current_cost = new_cost
            if new_cost < best_cost:
                best_floorplan = new_floorplan
                best_cost = new_cost

        # **Cooling schedule**: Reduce the temperature over time
        temperature *= cooling_factor

        # **Archiving & Visualization**: Save current state periodically
        if i % log_interval == 0:
            iter_dir = os.path.join(optimizer_dir, f"iter_{i}")
            os.makedirs(iter_dir, exist_ok=True)

            archive_design(iter_dir, floorplan)
            floorplan = add_interposer_to_floorplan(floorplan)

            # Visualize floorplan (with cost displayed)
            visualize_floorplan(iter_dir, [best_floorplan], f"Iteration {i} Cost: {current_cost:.2f}")

            # **Logging Cost at Each Iteration**
            log_data.append([i, current_cost])
            print(f"Iteration {i}: Cost = {current_cost:.2f}")

    # **Save Optimization Log**
    pd.DataFrame(log_data, columns=["Iteration", "Cost"]).to_csv(log_file, index=False)

    # **Save Final Best Design**
    final_hash = hash_floorplan(best_floorplan)
    final_file = os.path.join(optimizer_dir, "final_design.csv")
    
    if final_hash not in seen_hashes:  # Ensure final design is unique
        seen_hashes.add(final_hash)
        archive_design(optimizer_dir, best_floorplan)
        pd.DataFrame(best_floorplan).to_csv(final_file, index=False)

    return best_floorplan

def simulated_annealing(floorplan, associations, exp_dir, iterations=500, temperature=100):
    """Optimizes the floorplan using simulated annealing while archiving and visualizing every step."""
    global total_iterations
    total_iterations = iterations  # For reference inside swap_or_nudge

    current_cost = compute_cost(floorplan, associations)
    best_floorplan = floorplan.copy()
    best_cost = current_cost
    print("Current Cost of the system is ", best_cost)
    optimizer_dir = os.path.join(exp_dir, "optimizer")
    os.makedirs(optimizer_dir, exist_ok=True)

    log_file = os.path.join(optimizer_dir, "log.csv")
    log_data = []
    seen_hashes = set()

    for i in range(iterations):
        # Apply perturbation (swap or nudge)
        new_floorplan = swap_or_nudge(floorplan, temperature, i, total_iterations)
        new_cost = compute_cost(new_floorplan, associations)
        new_hash = hash_floorplan(new_floorplan)

        # **Avoid duplicate designs**
        if new_hash in seen_hashes:
            continue
        seen_hashes.add(new_hash)

        # **Accept new design if better OR with probability**
        if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
            floorplan = new_floorplan
            current_cost = new_cost
            # slow cooling now to perturb this design
            temperature = adaptive_cooling(temperature, best_cost, current_cost)
            if new_cost < best_cost:
                best_floorplan = new_floorplan
                best_cost = new_cost

        # **Archiving & Visualization**
        iter_dir = os.path.join(optimizer_dir, f"iter_{i}")
        os.makedirs(iter_dir, exist_ok=True)

        archive_design(iter_dir, floorplan)
        floorplan = add_interposer_to_floorplan(floorplan)

        # Update interposer in designs
        new_floorplan = add_interposer_to_floorplan(new_floorplan)
        best_floorplan = add_interposer_to_floorplan(best_floorplan)

        visualize_floorplan(iter_dir, [best_floorplan], f"Iteration {i} Cost: {new_cost:.2f}")

        # **Logging Cost at Each Iteration**
        log_data.append([i, new_cost])
        print(f"Iteration {i}: Cost = {new_cost:.2f}")

    # **Save Optimization Log**
    pd.DataFrame(log_data, columns=["Iteration", "Cost"]).to_csv(log_file, index=False)

    # **Save Final Best Design**
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

def add_interposer_to_floorplan(floorplan_data, spacing=E_s):
    # Ensure no previous interposer exists in the floorplan
    floorplan_data = [chiplet for chiplet in floorplan_data if chiplet["Chiplet_Type"] != "Interposer"]

    # Compute the bounding box for all chiplets to define the interposer size
    min_x = min(chiplet["X_Position"] for chiplet in floorplan_data)
    min_y = min(chiplet["Y_Position"] for chiplet in floorplan_data)
    max_x = max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan_data)
    max_y = max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan_data)

    # Add spacing around the edges of the bounding box
    interposer_x_position = min_x - spacing
    interposer_y_position = min_y - spacing
    interposer_length = (max_x - min_x) + 2 * spacing  # Add spacing on both sides
    interposer_breadth = (max_y - min_y) + 2 * spacing  # Add spacing on both sides

    # Add the interposer with its size based on the bounding box + spacing
    floorplan_data.append({
        "Chiplet_Type": "Interposer",
        "Chiplet_Name": "Interposer",
        "X_Position": interposer_x_position,
        "Y_Position": interposer_y_position,
        "Z_Position": 0,  # Interposer at the base layer (Z = 0)
        "Length": interposer_length,  # Width of the bounding box + spacing
        "Breadth": interposer_breadth  # Height of the bounding box + spacing
    })
    
    return floorplan_data



def visualize_floorplan(exp_dir, tiers, title, E_s=0.05):
    os.makedirs(exp_dir, exist_ok=True)  # Ensure directory exists before saving

    fig, axs = plt.subplots(1, 4, figsize=(20, 8))
    cluster_colors = {"CCD": "red", "XCD": "purple", "IPD": "blue", 
                      "EMIB": "green", "EMIB2": "orange", "Interposer": "black"}

    titles = ["Tier 0: Interposer", "Tier 1: Embedded", "Tier 2: Surface", "Combined"]

    # Unpack the floorplan into its separate tiers
    interposer_floorplan = [chiplet for chiplet in tiers[0] if chiplet["Z_Position"] == 0]
    embedded_floorplan = [chiplet for chiplet in tiers[0] if chiplet["Z_Position"] == 1]
    surface_floorplan = [chiplet for chiplet in tiers[0] if chiplet["Z_Position"] == 2]

    floorplans = [interposer_floorplan, embedded_floorplan, surface_floorplan, tiers[0]]

    for ax, floorplan, tier_title in zip(axs, floorplans, titles):
        ax.set_title(tier_title, fontsize=12)

        # Calculate the bounding box based on min and max coordinates (x and y)
        min_x = min(chiplet["X_Position"] for chiplet in floorplan)
        min_y = min(chiplet["Y_Position"] for chiplet in floorplan)
        max_x = max(chiplet["X_Position"] + chiplet["Length"] for chiplet in floorplan)
        max_y = max(chiplet["Y_Position"] + chiplet["Breadth"] for chiplet in floorplan)

        # Plot bounding box with a small margin (E_s)
        ax.plot([min_y - E_s, max_y + E_s, max_y + E_s, min_y - E_s, min_y - E_s],
                [min_x - E_s, min_x - E_s, max_x + E_s, max_x + E_s, min_x - E_s],
                color="black", linestyle="--", linewidth=2)

        # Plot chiplets
        for chiplet in floorplan:
            x, y = chiplet["X_Position"], chiplet["Y_Position"]
            length, breadth = chiplet["Length"], chiplet["Breadth"]
            color = cluster_colors.get(chiplet["Chiplet_Type"], "gray")
            if chiplet["Chiplet_Type"] == "Interposer":
                ax.add_patch(plt.Rectangle((y, x), breadth, length, facecolor="black", alpha=0.2, edgecolor="black"))
            else:
                ax.add_patch(plt.Rectangle((y, x), breadth, length, facecolor=color, alpha=0.6, edgecolor="black"))
            ax.text(y + breadth / 2, x + length / 2, chiplet["Chiplet_Name"], fontsize=8, ha="center", va="center")

        ax.set_xlim(min_y - E_s, max_y + E_s)
        ax.set_ylim(min_x - E_s, max_x + E_s)
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
    max_width, max_height = 20, 20 # starting package size
    i = 1

    while i <= 3:  
        exp_dir = f"exp_{i}"  # Create unique directory for each experiment
        os.makedirs(exp_dir, exist_ok=True)  # Ensure directory exists
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
        final_floorplan = add_interposer_to_floorplan(combined_floorplan)

        # Visualize the layout before optimization
        visualize_floorplan(f"{exp_dir}", [final_floorplan], "Initial Floorplan")

        # Run simulated annealing optimization
        total_associations = make_associations(associations, Cluster_type.values())
        optimized_floorplan = simulated_annealing(final_floorplan, total_associations, exp_dir)

        # Archive post-optimization
        archive_design("exp_1", optimized_floorplan)
        visualize_floorplan("exp_1", [optimized_floorplan], "Post-Optimization Floorplan")

        print(f"Archived unique design at iteration {i}")
        i += 1


if __name__ == "__main__":
    main()
