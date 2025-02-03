import yaml 
import numpy as np
import csv
import subprocess
import re

router_power_scaling = {
    1: 0,
    2: 1.4,
    3: 2.2,
    4: 3.4,
    5: 3.8,
    6: 4.2 
}

router_power_scaling_floret = {
    2: 1.1,
    3: 2,
    4: 2.5,
    5: 2.8,
    6: 3 
}


def run_mfit(iter, timeout=60):
    script = './MFIT/thermal_RC.py'

    args = [
        '--power_config_file', f'./exp_{iter}/power_dist_config.yaml',
        '--power_seq_file', f'./exp_{iter}/power_seq.csv',
        '--output_dir', f'././exp_{iter}/'
    ]

    # Start the subprocess with a timeout
    process = subprocess.Popen(
        ['python3', script] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Wait for the process to complete and get output with a timeout
        stdout, stderr = process.communicate(timeout=timeout)
        
        # If there's an error in execution, print stderr and raise an error
        if process.returncode != 0:
            print("STDERR:", stderr)
            raise RuntimeError(f"thermal_RC.py failed with error: {stderr}")

        # Strip any surrounding whitespace from the output
        T_peak_str = stdout.strip()
        
        # Use regex to extract the first floating-point number from the output
        match = re.search(r"\d+\.\d+", T_peak_str)
        
        if match:
            T_peak = float(match.group())  # Convert the matched number to a float
            return T_peak
        else:
            raise ValueError(f"Invalid T_peak value received: {T_peak_str}")
        
    except subprocess.TimeoutExpired:
        # Handle timeout error (if process takes too long)
        process.kill()  # Kill the process if it exceeds the timeout
        stdout, stderr = process.communicate()
        print(f"Process timed out. STDOUT: {stdout}, STDERR: {stderr}")
        raise TimeoutError(f"thermal_RC.py exceeded the timeout of {timeout} seconds.")
        
    except Exception as e:
        # Handle any other errors
        print(f"Error occurred: {str(e)}")
        raise e
        
    finally:
        # Ensure subprocess is terminated if it has not already been killed
        process.terminate()

import numpy as np

def generate_glass_blocks(floorplan_d, microblock_size=0.25):
    """
    Generates glass blocks to fill gaps between chiplets in the 2D floorplan.

    Args:
        floorplan_d (list): List of chiplets with positions and dimensions.
        microblock_size (float): The size of the glass blocks to use for filling gaps.

    Returns:
        dict: Dictionary containing glass blocks to fill the gaps.
    """
    if not isinstance(microblock_size, (int, float)):
        raise TypeError(f"Expected microblock_size to be a float or int, got {type(microblock_size)}")

    glass_blocks = {}
    occupied = []  # List to track occupied areas

    # Iterate through all chiplets and mark their occupied regions
    for chiplet in floorplan_d:
        x_start = chiplet['Lower_Left_Corner'][0]
        y_start = chiplet['Lower_Left_Corner'][1]
        x_end = x_start + chiplet['Length']
        y_end = y_start + chiplet['Breadth']
        occupied.append((x_start, x_end, y_start, y_end))

    # Define the bounds of the floorplan
    x_min = min(oc[0] for oc in occupied)
    x_max = max(oc[1] for oc in occupied)
    y_min = min(oc[2] for oc in occupied)
    y_max = max(oc[3] for oc in occupied)

    # Create a grid of microblocks and iterate over it
    gap_idx = 0
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            # Check if the microblock intersects with any occupied area
            is_occupied = False
            for oc in occupied:
                if oc[0] <= x < oc[1] and oc[2] <= y < oc[3]:
                    is_occupied = True
                    break

            # If not occupied, create a glass block
            if not is_occupied:
                glass_block = generate_new_power_dist(
                    cluster='Glass',
                    idx=gap_idx,
                    start_x=float(x),  # Ensure start_x is a float
                    start_y=float(y),  # Ensure start_y is a float
                    len_x=microblock_size,
                    len_y=microblock_size,
                    nodes_x=1,
                    nodes_y=1,
                    power=0,
                    glass=True
                )
                glass_blocks.update(glass_block)
                gap_idx += 1

            # Move to the next microblock along the y-axis
            y = float(y) + float(microblock_size)  # Ensure `y` remains a float

        # Move to the next microblock along the x-axis
        x = float(x) + float(microblock_size)  # Ensure `x` remains a float

    return glass_blocks

def generate_new_power_dist(cluster, idx, start_x, start_y, len_x, len_y, nodes_x, nodes_y, power, glass = False):
    chiplet = {'start_chiplet_x': start_x+2.5, 'start_chiplet_y': start_y+2.5, 'length_chiplet_x': len_x, 'length_chiplet_y': len_y}
    # layout_blocks = {'start_chiplet_x': start_x, 'start_chiplet_y': start_y, 'layout_blocks': layout_blocks}

    if power>0:
        layout_blocks = {'layout_blocks': {'chiplet': {'length_x': len_x, 'length_y': len_y, 'max_power': power, 'start_point_x': 0, 'start_point_y': 0}}}
        chiplet.update(layout_blocks)
        chiplet.update({'nodes_x': nodes_x, 'nodes_y': nodes_y})
    else:
        chiplet.update({'nodes_x': 1, 'nodes_y': 1})

    if glass:
        chiplet.update({'material': 'glass'})

    chiplet_name = cluster+ '_chiplet_' + str(idx)

    chiplet = {chiplet_name:  chiplet}

    return chiplet

mapping = {f"C{i}": f"Cluster {i}" for i in range(1, 6)} # Change manually if cluster count increases

def generate_power_config_file(floorplan_data, embedded_floorplan_data, clusters_surface, clusters_embedded, iter, grid_dims_surface, grid_dims_embedded):
    embedded_chiplet_dict = {}
    chiplet_dict = {}
    u_ubump_dict = {}

    power_list = []

    for chiplet_element in floorplan_data:

        neighbors = chiplet_element["neighbor_count"]
        cluster_key = chiplet_element["Chiplet"].split("-")[0]
        cluster_idx = int(chiplet_element["Chiplet"].split("-")[1])
        chiplet_x = chiplet_element["Lower_Left_Corner"][1]
        chiplet_y = chiplet_element["Lower_Left_Corner"][0]
        chiplet_len_y = chiplet_element['Length']
        chiplet_len_x = chiplet_element['Breadth']

        # Select the appropriate cluster dictionary based on location
        if chiplet_element["location"] == "surface":
            clusters = clusters_surface
            power_string_name = f'chiplet_{cluster_key}_chiplet_{cluster_idx}_chiplet'
        elif chiplet_element["location"] == "embedded":
            clusters = clusters_embedded
            power_string_name = f'embedded_chiplet_{cluster_key}_chiplet_{cluster_idx}_chiplet'
        else:
            raise ValueError(f"Unknown location: {chiplet_element['location']}")

        power_seq = [100] * 50
        power_seq.insert(0, power_string_name)
        power_list.append(power_seq)

        if chiplet_len_y == 4:
            nodes_x, nodes_y = 2, 4
        elif chiplet_len_x == 4:
            nodes_x, nodes_y = 4, 2
        else:
            nodes_x, nodes_y = 2, 2

        # Calculate power using the selected cluster dictionary
        power = clusters[mapping[cluster_key]]['pd'] + router_power_scaling[neighbors]

        # Generate chiplet and uBump power distributions
        chiplet = generate_new_power_dist(cluster=cluster_key,
                                          idx=cluster_idx,
                                          start_x=chiplet_x,
                                          start_y=chiplet_y,
                                          len_x=chiplet_len_x,
                                          len_y=chiplet_len_y,
                                          nodes_x=nodes_x,
                                          nodes_y=nodes_y,
                                          power=power)

        u_ubump = generate_new_power_dist(cluster=cluster_key,
                                          idx=cluster_idx,
                                          start_x=chiplet_x,
                                          start_y=chiplet_y,
                                          len_x=chiplet_len_x,
                                          len_y=chiplet_len_y,
                                          nodes_x=nodes_x,
                                          nodes_y=nodes_y,
                                          power=0)

        if chiplet_element["location"] == "surface":
            chiplet_dict.update(chiplet)
            u_ubump_dict.update(u_ubump)
        elif chiplet_element["location"] == "embedded":
            embedded_chiplet_dict.update(chiplet)

    # Uncomment to add glass material block in between
    # glass_blocks = generate_glass_blocks(embedded_floorplan_data, grid_dims_surface, grid_dims_embedded)
    # embedded_chiplet_dict.update(glass_blocks)

    power_dist_dict = {
        'chiplet': chiplet_dict,
        'embedded_chiplet': embedded_chiplet_dict,
        'ubump': u_ubump_dict
    }

    # Write power distribution YAML file
    with open(f'./exp_{iter}/power_dist_config.yaml', 'w') as file:
        yaml.dump(power_dist_dict, file)

    # Write power sequence CSV file
    with open(f'./exp_{iter}/power_seq.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(power_list)

    # Run MFIT to calculate T_peak
    T_peak = run_mfit(iter)
    return T_peak

    


