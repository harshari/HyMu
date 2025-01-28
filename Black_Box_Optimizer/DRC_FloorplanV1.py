import numpy as np
import os
from itertools import permutations
from scipy.spatial.distance import cityblock
import matplotlib.pyplot as plt
import random

def evaluate_drcs(floorplan_data, spacing_threshold, association_weights, grid_dims):
    """
    Evaluate DRC violations for a given floorplan, including grid boundary checks.

    Parameters:
    - floorplan_data: List of dictionaries with chiplet metadata.
    - spacing_threshold: Minimum spacing allowed between chiplets.
    - association_weights: Dictionary of (chiplet1, chiplet2) -> weight.
    - grid_dims: Tuple (rows, cols) representing the grid dimensions.

    Returns:
    - Summary of DRC violations.
    - Detailed logs.
    """
    overlaps = []
    spacing_violations = []
    wirelength_issues = []
    boundary_violations = []

    rows, cols = grid_dims  # Unpack grid dimensions

    for i, chip1 in enumerate(floorplan_data):
        # Check grid boundary violations
        if not is_within_grid(chip1, grid_dims):
            boundary_violations.append(chip1["Chiplet"])
        
        for j, chip2 in enumerate(floorplan_data):
            if i >= j:  # Avoid redundant checks
                continue
            
            # Check overlap
            if check_overlap(chip1, chip2):
                overlaps.append((chip1["Chiplet"], chip2["Chiplet"]))

            # Check spacing
            if check_spacing(chip1, chip2, spacing_threshold):
                spacing_violations.append((chip1["Chiplet"], chip2["Chiplet"]))

            # Calculate wirelength
            wirelength = calculate_wirelength(chip1, chip2)
            pair = (chip1["Chiplet"], chip2["Chiplet"])
            if pair in association_weights and wirelength > association_weights[pair]:
                wirelength_issues.append((pair, wirelength, association_weights[pair]))

    # Summarize results
    summary = {
        "Total Overlaps": len(overlaps),
        "Total Spacing Violations": len(spacing_violations),
        "Total Wirelength Issues": len(wirelength_issues),
        "Total Boundary Violations": len(boundary_violations),
    }

    detailed_logs = {
        "Overlaps": overlaps,
        "Spacing Violations": spacing_violations,
        "Wirelength Issues": wirelength_issues,
        "Boundary Violations": boundary_violations,
    }

    return summary, detailed_logs

def is_within_grid(chiplet, grid_dims):
    """
    Checks if a chiplet is within the grid dimensions.

    Parameters:
    - chiplet: Dictionary with chiplet metadata.
    - grid_dims: Tuple (rows, cols) representing the grid dimensions.

    Returns:
    - True if the chiplet is within the grid, False otherwise.
    """
    x_min, y_min = chiplet["Lower_Left_Corner"]
    x_max = x_min + chiplet["Length"]
    y_max = y_min + chiplet["Breadth"]

    rows, cols = grid_dims
    return 0 <= x_min < rows and 0 <= y_min < cols and x_max <= rows and y_max <= cols

def update_chiplet_geometry(chiplet, orientation):
    """
    Updates chiplet geometry based on its orientation (0°, 90°, 180°, 270°).

    Parameters:
    - chiplet: Dictionary with chiplet metadata.
    - orientation: Orientation angle of the chiplet.

    Returns:
    - Updated chiplet dictionary.
    """
    length, breadth = chiplet["Length"], chiplet["Breadth"]

    if orientation in (90, 270):
        chiplet["Length"], chiplet["Breadth"] = breadth, length

    chiplet["Orientation"] = orientation
    return chiplet

def is_within_bounds(chiplet, grid_dims):
    """
    Checks if a chiplet is within the grid boundaries.

    Parameters:
    - chiplet: Dictionary with chiplet metadata.
    - grid_dims: Tuple (rows, cols) for the grid dimensions.

    Returns:
    - True if the chiplet is within bounds, else False.
    """
    x, y = chiplet["Lower_Left_Corner"]
    length, breadth = chiplet["Length"], chiplet["Breadth"]
    rows, cols = grid_dims

    return 0 <= x < rows and 0 <= y < cols and x + length <= rows and y + breadth <= cols

def check_overlap(chip1, chip2):
    x1_min, y1_min = chip1["Lower_Left_Corner"]
    x1_max = x1_min + chip1["Length"]
    y1_max = y1_min + chip1["Breadth"]

    x2_min, y2_min = chip2["Lower_Left_Corner"]
    x2_max = x2_min + chip2["Length"]
    y2_max = y2_min + chip2["Breadth"]

    return not (x1_max <= x2_min or x2_max <= x1_min or
                y1_max <= y2_min or y2_max <= y1_min)

def check_spacing(chip1, chip2, threshold):
    x1_min, y1_min = chip1["Lower_Left_Corner"]
    x1_max = x1_min + chip1["Length"]
    y1_max = y1_min + chip1["Breadth"]

    x2_min, y2_min = chip2["Lower_Left_Corner"]
    x2_max = x2_min + chip2["Length"]
    y2_max = y2_min + chip2["Breadth"]

    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)

    return (dx < threshold and dy < threshold)

def calculate_wirelength(chip1, chip2):
    x1_min, y1_min = chip1["Lower_Left_Corner"]
    x1_max = x1_min + chip1["Length"]
    y1_max = y1_min + chip1["Breadth"]

    x2_min, y2_min = chip2["Lower_Left_Corner"]
    x2_max = x2_min + chip2["Length"]
    y2_max = y2_min + chip2["Breadth"]

    d1 = min(abs(x1_min - x2_min), abs(x1_max - x2_max))
    d2 = min(abs(y1_min - y2_min), abs(y1_max - y2_max))

    return d1 + d2

def log_drc_violations(exp_dir, drc_violations, iteration):
    """
    Logs detailed DRC violations to a file for analysis.

    Parameters:
    - exp_dir: Directory to save the log file.
    - drc_violations: Dictionary of DRC violations.
    - iteration: Current iteration number.
    """
    log_file = os.path.join(exp_dir, f"drc_log_iteration_{iteration}.txt")
    with open(log_file, "w") as file:
        file.write("DRC Violations Summary\n")
        file.write("=========================\n\n")

        for key, value in drc_violations.items():
            if key != "Details":
                file.write(f"{key}: {value}\n")
        
        file.write("\nDetails:\n")
        for detail_key, detail_value in drc_violations["Details"].items():
            file.write(f"{detail_key}:\n")
            for item in detail_value:
                file.write(f"  - {item}\n")
            file.write("\n")

def find_closest_edge_distance(chip1, chip2):
    """
    Finds the closest edge-to-edge distance between two chiplets.

    Parameters:
    - chip1, chip2: Dictionaries containing chiplet metadata.

    Returns:
    - Minimum edge-to-edge distance.
    """
    x1_min, y1_min = chip1["Lower_Left_Corner"]
    x1_max = x1_min + chip1["Length"]
    y1_max = y1_min + chip1["Breadth"]

    x2_min, y2_min = chip2["Lower_Left_Corner"]
    x2_max = x2_min + chip2["Length"]
    y2_max = y2_min + chip2["Breadth"]

    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)

    return dx + dy

def visualize_drc_violations(exp_dir, floorplan_data, drc_violations, grid_dims):
    """
    Visualizes DRC violations in the floorplan.

    Parameters:
    - exp_dir: Directory to save the plot.
    - floorplan_data: List of dictionaries with chiplet metadata.
    - drc_violations: Detailed logs of DRC violations.
    - grid_dims: Tuple (rows, cols) for the grid dimensions.
    """
    plt.figure(figsize=(12, 8))
    plt.title("DRC Violations", fontsize=18)

    # Plot chiplets
    for chiplet in floorplan_data:
        x, y = chiplet["Lower_Left_Corner"]
        length = chiplet["Length"]
        breadth = chiplet["Breadth"]
        plt.gca().add_patch(
            plt.Rectangle(
                (y, x),
                breadth,
                length,
                facecolor="blue",
                alpha=0.6,
                edgecolor="black"
            )
        )

    # Highlight violations
    for violation_type, violations in drc_violations.items():
        color = {"Overlaps": "red", "Spacing Violations": "orange", "Wirelength Issues": "green"}.get(violation_type, "gray")
        for violation in violations:
            chip1, chip2 = violation[:2] if isinstance(violation, tuple) else violation
            x1, y1 = chip1["Lower_Left_Corner"]
            x2, y2 = chip2["Lower_Left_Corner"]

            # Draw lines between violating chiplets
            plt.plot([y1 + chip1["Breadth"] / 2, y2 + chip2["Breadth"] / 2],
                     [x1 + chip1["Length"] / 2, x2 + chip2["Length"] / 2],
                     color=color, linestyle="--", linewidth=1)

    # Set grid and axis limits
    plt.xlim(0, grid_dims[1])
    plt.ylim(0, grid_dims[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    # Save visualization
    plt.xlabel("Y-axis (columns)")
    plt.ylabel("X-axis (rows)")
    plt.savefig(f"{exp_dir}/drc_violations.png", dpi=300, bbox_inches="tight")
    plt.close()

def ensure_equal_grid_dimensions(tiers, grid_dims):
    """
    Ensure all tiers have the same grid dimensions.

    Parameters:
    - tiers: Dictionary where keys are tier names and values are chiplet lists.
    - grid_dims: Target grid dimensions (rows, cols).

    Returns:
    - adjusted_tiers: Tiers with validated dimensions.
    """
    adjusted_tiers = {}
    for tier_name, chiplets in tiers.items():
        adjusted_tiers[tier_name] = {
            "chiplets": chiplets,
            "grid_dims": grid_dims  # Apply the same grid dimensions to all tiers
        }
    return adjusted_tiers

def randomize_chiplet_placement(tier, grid_dims, weights=None):
    """
    Randomly place chiplets within the given grid dimensions.

    Parameters:
    - tier: Dictionary with chiplet metadata.
    - grid_dims: Tuple (rows, cols) for grid dimensions.
    - weights: Optional dictionary of (chiplet1, chiplet2) -> weight for proximity preference.

    Returns:
    - randomized_positions: List of chiplets with updated positions.
    """
    rows, cols = grid_dims
    grid = np.zeros((rows, cols), dtype=int)  # Empty grid
    randomized_positions = []

    for chiplet in tier["chiplets"]:
        placed = False
        chiplet_size = (chiplet["Length"], chiplet["Breadth"])

        while not placed:
            # Randomly select a position within the grid
            x = random.randint(0, rows - chiplet_size[0])
            y = random.randint(0, cols - chiplet_size[1])

            # Check if the position is valid (no overlap)
            if is_valid_position(grid, x, y, chiplet_size):
                # Place chiplet on grid
                for dx in range(chiplet_size[0]):
                    for dy in range(chiplet_size[1]):
                        grid[x + dx, y + dy] = 1  # Mark grid as occupied

                # Update chiplet with its new position
                chiplet["Lower_Left_Corner"] = (x, y)
                randomized_positions.append(chiplet)
                placed = True

    return randomized_positions

def weight_aware_random_placement(tier, grid_dims, weights):
    """
    Randomize placement while prioritizing chiplets based on weights.

    Parameters:
    - tier: Dictionary with chiplet metadata.
    - grid_dims: Tuple (rows, cols) for grid dimensions.
    - weights: Dictionary of (chiplet1, chiplet2) -> weight for proximity preference.

    Returns:
    - placement: List of chiplets with positions updated.
    """
    # Call the basic randomization first
    randomized_positions = randomize_chiplet_placement(tier, grid_dims)

    # Adjust positions based on weights
    for chiplet in randomized_positions:
        for associated_chiplet, weight in weights.get(chiplet["Chiplet"], {}).items():
            # Find the associated chiplet and adjust its placement
            associated_chip = next((c for c in randomized_positions if c["Chiplet"] == associated_chiplet), None)
            if associated_chip:
                # Move associated chiplet closer based on the weight
                chiplet_center = np.array(chiplet["Lower_Left_Corner"]) + np.array([chiplet["Length"] / 2, chiplet["Breadth"] / 2])
                associated_center = np.array(associated_chip["Lower_Left_Corner"]) + np.array([associated_chip["Length"] / 2, associated_chip["Breadth"] / 2])

                distance = np.linalg.norm(chiplet_center - associated_center)
                if distance > weight:
                    # Adjust position (basic heuristic for now)
                    new_x = int((chiplet_center[0] + associated_center[0]) / 2)
                    new_y = int((chiplet_center[1] + associated_center[1]) / 2)
                    associated_chip["Lower_Left_Corner"] = (new_x, new_y)

    return randomized_positions

def visualize_tiers_side_by_side(tiers, exp_dir, title):
    """
    Visualize chiplet placements for each tier in separate subplots.

    Parameters:
    - tiers: Dictionary where keys are tier names and values are chiplet lists with metadata.
    - exp_dir: Directory to save the visualizations.
    - title: Title for the overall visualization.

    Returns:
    - Saves the plot with side-by-side tier visualizations.
    """
    num_tiers = len(tiers)
    fig, axes = plt.subplots(1, num_tiers, figsize=(6 * num_tiers, 8), sharey=True)
    fig.suptitle(title, fontsize=16)

    for ax, (tier_name, tier_data) in zip(axes, tiers.items()):
        ax.set_title(tier_name, fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        grid_dims = tier_data["grid_dims"]

        # Set axis limits based on the grid dimensions
        ax.set_xlim(0, grid_dims[1])  # Columns
        ax.set_ylim(0, grid_dims[0])  # Rows
        ax.invert_yaxis()  # Invert Y-axis for intuitive visualization

        # Plot each chiplet in the tier
        for chiplet in tier_data["chiplets"]:
            x, y = chiplet["Lower_Left_Corner"]
            length = chiplet["Length"]
            breadth = chiplet["Breadth"]

            # Add rectangle for chiplet
            ax.add_patch(
                plt.Rectangle(
                    (y, x),  # Rectangle starts at (y, x)
                    breadth,  # Width
                    length,  # Height
                    facecolor="skyblue",
                    edgecolor="black",
                    alpha=0.6
                )
            )

            # Add chiplet label
            ax.text(
                y + breadth / 2,
                x + length / 2,
                chiplet["Chiplet"],
                fontsize=10,
                ha="center",
                va="center"
            )

        # Add gridlines for better reference
        ax.set_xticks(range(grid_dims[1]))
        ax.set_yticks(range(grid_dims[0]))
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Columns", fontsize=12)

    axes[0].set_ylabel("Rows", fontsize=12)  # Add Y-axis label to the first subplot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the title
    plt.savefig(f"{exp_dir}/{title.replace(' ', '_')}_side_by_side.png", dpi=300)
    plt.close()

def check_bounding_box_overlap_optimized(upper_chiplet, lower_tier_chiplets, required_overlap=0.5):
    """
    Checks bounding box overlap for an upper chiplet against a filtered list of lower-tier chiplets.

    Parameters:
    - upper_chiplet: Upper-tier chiplet dictionary.
    - lower_tier_chiplets: List of lower-tier chiplets.
    - required_overlap: Fraction of the upper chiplet's area that must overlap with the lower chiplet.

    Returns:
    - TSV violations: List of violations with details of overlap ratios.
    """
    x1_min, y1_min = upper_chiplet["Lower_Left_Corner"]
    x1_max = x1_min + upper_chiplet["Length"]
    y1_max = y1_min + upper_chiplet["Breadth"]

    violations = []

    # Iterate over filtered lower-tier chiplets within potential bounding region
    for lower_chiplet in lower_tier_chiplets:
        x2_min, y2_min = lower_chiplet["Lower_Left_Corner"]
        x2_max = x2_min + lower_chiplet["Length"]
        y2_max = y2_min + lower_chiplet["Breadth"]

        # Filter based on potential overlap
        if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
            # Compute overlap area
            overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            overlap_area = overlap_x * overlap_y

            # Calculate overlap ratio
            upper_area = upper_chiplet["Length"] * upper_chiplet["Breadth"]
            overlap_ratio = overlap_area / upper_area if upper_area > 0 else 0

            # Check if it violates the required overlap
            if overlap_ratio < required_overlap:
                violations.append({
                    "upper_chiplet": upper_chiplet["Chiplet"],
                    "lower_chiplet": lower_chiplet["Chiplet"],
                    "overlap_ratio": overlap_ratio,
                })

    return violations
