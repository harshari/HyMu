import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
import copy
import random

###Helper
# Calculate module area
def calculate_module_area(chiplets):
    min_x = min(c["x"] for c in chiplets)
    max_x = max(c["x"] + c["w"] for c in chiplets)
    min_y = min(c["y"] for c in chiplets)
    max_y = max(c["y"] + c["h"] for c in chiplets)
    return (max_x - min_x) * (max_y - min_y)

# Calculate package utilization
def calculate_package_utilization(chiplets, module_area):
    total_chiplet_area = sum(c["w"] * c["h"] for c in chiplets)
    return total_chiplet_area / module_area

# Calculate wirelength
def calculate_wirelength(chiplets, connections):
    wirelength_details = []
    total_wirelength = 0

    for c1_id, c2_id in connections:
        c1 = next(c for c in chiplets if c["id"] == c1_id)
        c2 = next(c for c in chiplets if c["id"] == c2_id)

        # Manhattan distance
        wirelength = abs(c1["x"] + c1["w"] / 2 - (c2["x"] + c2["w"] / 2)) + \
                     abs(c1["y"] + c1["h"] / 2 - (c2["y"] + c2["h"] / 2))
        wirelength_details.append(((c1_id, c2_id), wirelength))
        total_wirelength += wirelength

    return total_wirelength, wirelength_details

# Detect neighbors based on edge-to-edge distance
def find_neighbors_distance_based(chiplets, max_distance=0.3):
    neighbors = []

    def chiplet_edges(chiplet):
        x1, y1 = chiplet["x"], chiplet["y"]
        x2, y2 = x1 + chiplet["w"], y1 + chiplet["h"]
        return [
            LineString([(x1, y1), (x2, y1)]),  # Bottom edge
            LineString([(x2, y1), (x2, y2)]),  # Right edge
            LineString([(x2, y2), (x1, y2)]),  # Top edge
            LineString([(x1, y2), (x1, y1)]),  # Left edge
        ]

    for i, c1 in enumerate(chiplets):
        for j, c2 in enumerate(chiplets):
            if i < j:
                edges_c1 = chiplet_edges(c1)
                edges_c2 = chiplet_edges(c2)

                min_distance = min(edge1.distance(edge2) for edge1 in edges_c1 for edge2 in edges_c2)
                if min_distance <= max_distance:
                    neighbors.append((c1["id"], c2["id"]))

    return neighbors

###Perturbation
# Scaling perturbation with local logic
def scaling_perturb_local(chiplets, target_id, scale_factor, min_spacing=0.14):
    updated_chiplets = copy.deepcopy(chiplets)
    target_chiplet = next(c for c in updated_chiplets if c["id"] == target_id)

    # Scale the target chiplet
    original_w, original_h = target_chiplet["w"], target_chiplet["h"]
    target_chiplet["w"] *= scale_factor
    target_chiplet["h"] *= scale_factor

    delta_w = (target_chiplet["w"] - original_w) / 2
    delta_h = (target_chiplet["h"] - original_h) / 2

    neighbors = find_neighbors_distance_based(updated_chiplets)

    for c1_id, c2_id in neighbors:
        if c1_id == target_id or c2_id == target_id:
            neighbor_id = c1_id if c2_id == target_id else c2_id
            neighbor_chiplet = next(c for c in updated_chiplets if c["id"] == neighbor_id)

            if neighbor_chiplet["x"] < target_chiplet["x"]:  # Left
                neighbor_chiplet["x"] -= delta_w + min_spacing
            elif neighbor_chiplet["x"] > target_chiplet["x"] + target_chiplet["w"]:  # Right
                neighbor_chiplet["x"] += delta_w + min_spacing

            if neighbor_chiplet["y"] < target_chiplet["y"]:  # Below
                neighbor_chiplet["y"] -= delta_h + min_spacing
            elif neighbor_chiplet["y"] > target_chiplet["y"] + target_chiplet["h"]:  # Above
                neighbor_chiplet["y"] += delta_h + min_spacing

    return updated_chiplets

# Enforce spacing with module edge constraints
def enforce_spacing_with_module_edge(chiplets, neighbors, min_spacing=0.14, edge_spacing=0.28):
    spaced_chiplets = copy.deepcopy(chiplets)

    def chiplet_polygon(chiplet):
        return Polygon([
            (chiplet["x"], chiplet["y"]),
            (chiplet["x"] + chiplet["w"], chiplet["y"]),
            (chiplet["x"] + chiplet["w"], chiplet["y"] + chiplet["h"]),
            (chiplet["x"], chiplet["y"] + chiplet["h"]),
        ])

    for c1_id, c2_id in neighbors:
        c1 = next(c for c in spaced_chiplets if c["id"] == c1_id)
        c2 = next(c for c in spaced_chiplets if c["id"] == c2_id)

        while chiplet_polygon(c1).distance(chiplet_polygon(c2)) < min_spacing:
            c2["x"] += min_spacing / 2
            c2["y"] += min_spacing / 2

    min_x = min(c["x"] for c in spaced_chiplets)
    min_y = min(c["y"] for c in spaced_chiplets)
    max_x = max(c["x"] + c["w"] for c in spaced_chiplets)
    max_y = max(c["y"] + c["h"] for c in spaced_chiplets)

    for chiplet in spaced_chiplets:
        if chiplet["x"] - min_x < edge_spacing:
            chiplet["x"] += edge_spacing - (chiplet["x"] - min_x)
        if chiplet["y"] - min_y < edge_spacing:
            chiplet["y"] += edge_spacing - (chiplet["y"] - min_y)
        if max_x - (chiplet["x"] + chiplet["w"]) < edge_spacing:
            chiplet["x"] -= edge_spacing - (max_x - (chiplet["x"] + chiplet["w"]))
        if max_y - (chiplet["y"] + chiplet["h"]) < edge_spacing:
            chiplet["y"] -= edge_spacing - (max_y - (chiplet["y"] + chiplet["h"]))

    return spaced_chiplets

### Visualize
def plot_chiplets_with_utilization(chiplets, module_area, utilization):
    fig, ax = plt.subplots(figsize=(8, 6))

    for c in chiplets:
        ax.add_patch(plt.Rectangle((c["x"], c["y"]), c["w"], c["h"], edgecolor="blue", facecolor="lightblue"))
        ax.text(c["x"] + c["w"] / 2, c["y"] + c["h"] / 2, f"ID {c['id']}", ha="center", va="center", fontsize=10)

    min_x = min(c["x"] for c in chiplets)
    max_x = max(c["x"] + c["w"] for c in chiplets)
    min_y = min(c["y"] for c in chiplets)
    max_y = max(c["y"] + c["h"] for c in chiplets)
    ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], color="green", linestyle="-", linewidth=2)

    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(0, max_y + 1)
    ax.set_xlabel("X-axis (Module Length)")
    ax.set_ylabel("Y-axis (Module Width)")
    ax.set_title(f"Chiplet Placement with Utilization = {utilization * 100:.2f}% (Module Area = {module_area:.2f})")
    ax.grid(True)
    plt.show()

###Input 
chiplets_with_types = [
    {"id": 1, "x": 3, "y": 0, "w": 3, "h": 3, "type": "DRAM"},
    {"id": 2, "x": 6, "y": 0, "w": 3, "h": 3, "type": "DRAM"},
    {"id": 3, "x": 3, "y": 3, "w": 6, "h": 3, "type": "SRAM"},
    {"id": 4, "x": 3, "y": 6, "w": 3, "h": 3, "type": "DRAM"},
    {"id": 5, "x": 6, "y": 6, "w": 3, "h": 3, "type": "DRAM"}
]

def main_with_module_edge_spacing():
    print("Step 1: Initial Parameters with Spacing and Module Edge Constraints")
    initial_neighbors = find_neighbors_distance_based(chiplets_with_types)
    spaced_chiplets = enforce_spacing_with_module_edge(chiplets_with_types, initial_neighbors)
    initial_module_area = calculate_module_area(spaced_chiplets)
    initial_utilization = calculate_package_utilization(spaced_chiplets, initial_module_area)
    plot_chiplets_with_utilization(spaced_chiplets, initial_module_area, initial_utilization)

    print("\nStep 2: Scaling Perturbation with Local Logic and Module Edge Spacing")
    scaled_chiplets = scaling_perturb_local(spaced_chiplets, target_id=1, scale_factor=2)
    scaled_neighbors = find_neighbors_distance_based(scaled_chiplets)
    spaced_scaled_chiplets = enforce_spacing_with_module_edge(scaled_chiplets, scaled_neighbors)
    scaled_module_area = calculate_module_area(spaced_scaled_chiplets)
    scaled_utilization = calculate_package_utilization(spaced_scaled_chiplets, scaled_module_area)
    plot_chiplets_with_utilization(spaced_scaled_chiplets, scaled_module_area, scaled_utilization)

    print("\nStep 3: Wirelength Calculation")
    connections = [(1, 2), (4, 5), (1, 3), (3, 5)]
    total_wirelength, wirelength_details = calculate_wirelength(spaced_scaled_chiplets, connections)
    print(f"Total Wirelength: {total_wirelength:.2f}")
    for pair, length in wirelength_details:
        print(f"Connection {pair}: {length:.2f} units")

# main_with_module_edge_spacing()
def main_with_detailed_steps():
    print("Step 1: Initial Parameters with Spacing and Module Edge Constraints")
    # Step 1: Initial Placement
    initial_neighbors = find_neighbors_distance_based(chiplets_with_types)
    print(f"Initial Neighbors: {initial_neighbors}")
    spaced_chiplets = enforce_spacing_with_module_edge(chiplets_with_types, initial_neighbors)
    initial_module_area = calculate_module_area(spaced_chiplets)
    initial_utilization = calculate_package_utilization(spaced_chiplets, initial_module_area)
    print(f"Initial Module Area: {initial_module_area:.2f}")
    print(f"Initial Utilization: {initial_utilization * 100:.2f}%")
    plot_chiplets_with_utilization(spaced_chiplets, initial_module_area, initial_utilization)

    print("\nStep 2: Scaling Perturbation with Local Logic and Module Edge Spacing")
    # Step 2: Scaling Perturbation
    scaled_chiplets = scaling_perturb_local(spaced_chiplets, target_id=1, scale_factor=2)
    scaled_neighbors = find_neighbors_distance_based(scaled_chiplets)
    print(f"Neighbors After Scaling: {scaled_neighbors}")
    spaced_scaled_chiplets = enforce_spacing_with_module_edge(scaled_chiplets, scaled_neighbors)
    scaled_module_area = calculate_module_area(spaced_scaled_chiplets)
    scaled_utilization = calculate_package_utilization(spaced_scaled_chiplets, scaled_module_area)
    print(f"Scaled Module Area: {scaled_module_area:.2f}")
    print(f"Scaled Utilization: {scaled_utilization * 100:.2f}%")
    plot_chiplets_with_utilization(spaced_scaled_chiplets, scaled_module_area, scaled_utilization)

    print("\nStep 3: Positional Perturbation")
    # Step 3: Positional Perturbation
    perturbed_chiplets = positional_perturb(spaced_scaled_chiplets)
    perturbed_neighbors = find_neighbors_distance_based(perturbed_chiplets)
    print(f"Neighbors After Positional Perturbation: {perturbed_neighbors}")
    spaced_perturbed_chiplets = enforce_spacing_with_module_edge(perturbed_chiplets, perturbed_neighbors)
    perturbed_module_area = calculate_module_area(spaced_perturbed_chiplets)
    perturbed_utilization = calculate_package_utilization(spaced_perturbed_chiplets, perturbed_module_area)
    print(f"Perturbed Module Area: {perturbed_module_area:.2f}")
    print(f"Perturbed Utilization: {perturbed_utilization * 100:.2f}%")
    plot_chiplets_with_utilization(spaced_perturbed_chiplets, perturbed_module_area, perturbed_utilization)

    print("\nStep 4: Wirelength Calculation")
    # Step 4: Wirelength Calculation
    connections = [(1, 2), (4, 5), (1, 3), (3, 5)]
    total_wirelength, wirelength_details = calculate_wirelength(spaced_perturbed_chiplets, connections)
    print(f"Total Wirelength: {total_wirelength:.2f}")
    for pair, length in wirelength_details:
        print(f"Connection {pair}: {length:.2f} units")

    print("\nStep 5: Optimization Metrics")
    # Additional Metrics for Optimization
    symmetry_score = calculate_symmetry(spaced_perturbed_chiplets)
    print(f"Symmetry Score: {symmetry_score:.2f}")
    plot_pareto_front(total_wirelength, perturbed_utilization, symmetry_score, perturbed_module_area)

def positional_perturb(chiplets, max_move=2):
    """
    Randomly perturb the position of a single chiplet while maintaining constraints.
    
    Args:
        chiplets (list): List of chiplets with 'x', 'y', 'w', 'h', 'id'.
        max_move (float): Maximum distance a chiplet can move in x or y direction.

    Returns:
        updated_chiplets (list): Updated list of chiplets with one perturbed.
    """
    import random

    updated_chiplets = copy.deepcopy(chiplets)

    # Select a random chiplet to perturb
    chiplet_to_move = random.choice(updated_chiplets)

    # Generate a new random position within the bounds of max_move
    new_x = chiplet_to_move["x"] + random.uniform(-max_move, max_move)
    new_y = chiplet_to_move["y"] + random.uniform(-max_move, max_move)

    # Ensure the new position does not cause overlap or violate edge spacing
    min_spacing = 0.14
    edge_spacing = 0.28

    # Adjust for module edges
    new_x = max(new_x, edge_spacing)
    new_y = max(new_y, edge_spacing)

    # Calculate module boundaries
    module_max_x = max(c["x"] + c["w"] for c in updated_chiplets)
    module_max_y = max(c["y"] + c["h"] for c in updated_chiplets)

    new_x = min(new_x, module_max_x - chiplet_to_move["w"] - edge_spacing)
    new_y = min(new_y, module_max_y - chiplet_to_move["h"] - edge_spacing)

    # Update position if no overlap
    chiplet_to_move["x"] = new_x
    chiplet_to_move["y"] = new_y

    return updated_chiplets

# Symmetry Calculation for Optimization
def calculate_symmetry(chiplets):
    """
    Calculate a symmetry score based on alignment with the center of the module.
    """
    module_center_x = (max(c["x"] + c["w"] for c in chiplets) + min(c["x"] for c in chiplets)) / 2
    module_center_y = (max(c["y"] + c["h"] for c in chiplets) + min(c["y"] for c in chiplets)) / 2

    # Sum absolute differences from center for all chiplets
    symmetry_deviation = sum(
        abs((c["x"] + c["w"] / 2) - module_center_x) + abs((c["y"] + c["h"] / 2) - module_center_y)
        for c in chiplets
    )
    # Normalize: lower symmetry_deviation => higher symmetry score
    max_deviation = len(chiplets) * (module_center_x + module_center_y)
    return 1 - (symmetry_deviation / max_deviation)

# Plotting Pareto Front
def plot_pareto_front(wirelength, utilization, symmetry_score, module_area):
    """
    Visualize trade-offs between key objectives using a 3D Pareto front.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Axes for Pareto Front
    ax.scatter(wirelength, utilization, symmetry_score, color="blue", s=100, label="Design Point")
    ax.set_xlabel("Wirelength (Minimize)")
    ax.set_ylabel("Utilization (Maximize)")
    ax.set_zlabel("Symmetry (Maximize)")
    ax.set_title(f"Pareto Optimization (Module Area: {module_area:.2f})")
    ax.legend()
    plt.show()


# Run the step-by-step main function
main_with_detailed_steps()