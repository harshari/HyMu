import os
import pandas as pd
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
        "Storage": [
            0.84, 0.28, 0.50, 1.50, 0.84, 2.25, 3.38, 1.27, 3.38, 3.38, 1.27, 4.50, 6.00, 
            1.69, 6.00, 6.00, 1.69, 6.00, 6.00, 1.69, 12.00, 24.00, 3.38, 24.00, 24.00, 
            3.38, 24.00, 24.00, 3.38, 24.00, 24.00, 3.38, 36.00, 54.00, 5.06, 54.00, 54.00, 
            5.06, 54.00, 54.00, 5.06, 90.00, 150.00, 8.44, 150.00, 150.00, 8.44, 150.00, 
            150.00, 8.44, 300.00, 400.00
        ],
        "Activations": [
            392.00, 392.00, 196.00, 1176.00, 294.00, 73.50, 441.00, 441.00, 73.50, 441.00, 
            110.25, 24.50, 147.00, 147.00, 24.50, 147.00, 147.00, 24.50, 147.00, 36.75, 
            12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 12.25, 73.50, 73.50, 12.25, 73.50, 
            73.50, 18.38, 110.25, 110.25, 18.38, 110.25, 110.25, 18.38, 110.25, 27.56, 
            7.66, 45.94, 45.94, 7.66, 45.94, 45.94, 7.66, 45.94, 45.94, 15.31, 61.25
        ],
        "Compute": [
            10.84, 115.61, 6.42, 19.27, 260.11, 7.23, 10.84, 585.25, 10.84, 10.84, 
            146.31, 3.61, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 65.03, 2.41, 
            4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 260.11, 4.82, 4.82, 260.11, 
            7.23, 10.84, 585.25, 10.84, 10.84, 585.25, 10.84, 10.84, 146.31, 4.52, 
            7.53, 406.43, 7.53, 7.53, 406.43, 7.53, 7.53, 406.43, 15.05, 20.07
        ],
        "Sensitivity": [
            1.81, 0.00, 0.16, 47.19, 0.02, 0.11, 0.17, 0.01, 0.00, 2.27, 0.00, 
            0.02, 21.64, 0.04, 0.01, 0.38, 0.06, 0.00, 1.15, 0.00, 0.01, 6.65, 
            0.01, 0.01, 0.11, 0.03, 0.00, 1.07, 0.00, 0.00, 9.69, 0.01, 0.00, 
            0.13, 0.02, 0.00, 0.50, 0.00, 0.00, 4.63, 0.00, 0.00, 0.02, 0.01, 
            0.00, 0.17, 0.00, 0.00, 1.89, 0.00, 0.00, 0.02
        ]
    }

}

# Cluster Configurations
cluster_config = {
    "Cluster 1": {"count": 28, "pd": 8, "area": 8, "memory": 1196, "tops": 30e12, "energy_per_mac": .87e-12},
    "Cluster 2": {"count": 12, "pd": 1, "area": 4, "memory": 1080, "tops": 27e12, "energy_per_mac": .3e-12},
    "Cluster 3": {"count": 18, "pd": 4, "area": 4, "memory": 4800, "tops": 70e12, "energy_per_mac": .11e-12},
    "Cluster 4": {"count": 24, "pd": 8, "area": 4, "memory": 300, "tops": 3.8e12, "energy_per_mac": .27e-12},
}


def parse_temperature_files(base_path, experiment_prefix="exp_", temperature_file="output/temperature_chiplet_50.0.csv"):
    """
    Parses temperature files from available experiment directories.

    Args:
        base_path (str): Path to the base directory containing experiments.
        experiment_prefix (str): Prefix of experiment directories (e.g., "exp_").
        temperature_file (str): Relative path to the temperature file inside each experiment directory.

    Returns:
        dict: A dictionary with experiment names as keys and temperature DataFrames as values.
    """
    temperature_data = {}
    for experiment in sorted(os.listdir(base_path)):
        if experiment.startswith(experiment_prefix):
            experiment_path = os.path.join(base_path, experiment, temperature_file)
            if os.path.exists(experiment_path):
                try:
                    df = pd.read_csv(
                        experiment_path,
                        header=None,
                        names=["Chiplet", "Start_Temperature", "Peak_Temperature"],
                    )
                    # Extract Cluster and Chiplet numbers
                    df[['Cluster', 'Chiplet']] = df['Chiplet'].str.extract(
                        r'(C\d+)_chiplet_(\d+)', expand=True
                    )

                    # Convert Cluster values to uppercase (if needed)
                    df['Cluster'] = df['Cluster'].str.upper()

                    # Drop rows where extraction failed
                    df = df.dropna(subset=['Cluster', 'Chiplet'])

                    # Ensure Chiplet is treated as an integer
                    df['Chiplet'] = df['Chiplet'].astype(int)

                    # Sort by Cluster and Peak_Temperature
                    temperature_data[experiment] = df.sort_values(
                        by=['Cluster', 'Peak_Temperature']
                    )
                except Exception as e:
                    print(f"Error reading {experiment_path}: {e}")
    return temperature_data


def map_layers(network_data, cluster_config, temp_data, order, avg_hop_count=1.2, interconnect_bandwidth=1e9):
    """
    Maps neural network layers to chiplets based on temperature and memory availability.

    Args:
        network_data (dict): Information about neural networks.
        cluster_config (dict): Configuration details for clusters.
        temp_data (dict): Temperature data for experiments.
        order (list): Execution order of networks.
        avg_hop_count (float): Average hop count for inter-chiplet communication.
        interconnect_bandwidth (float): Bandwidth of the interconnect.

    Returns:
        dict: Results of mapping for each experiment.
    """
    results = {}

    for experiment, temp_df in temp_data.items():
        exp_results = {}

        # Validate that all clusters in the temperature data exist in cluster_config
        invalid_clusters = [cluster for cluster in temp_df['Cluster'].unique() if cluster not in cluster_config]
        if invalid_clusters:
            raise ValueError(f"Invalid clusters found in {experiment}: {invalid_clusters}. Check cluster_config.")

        # Initialize chiplet availability for this experiment
        chiplet_availability = temp_df.groupby('Cluster').apply(
            lambda x: {chiplet: cluster_config[x['Cluster'].iloc[0]]['memory'] for chiplet in x['Chiplet']}
        ).to_dict()

        for network_name in order:
            network = network_data[network_name]
            layers = len(network["Compute"])

            total_compute_time = 0
            total_communication_time = 0
            current_chiplet = None
            current_cluster = None

            for layer_idx in range(layers):
                sensitivity = network["Sensitivity"][layer_idx]
                storage = network["Storage"][layer_idx]
                activations = network["Activations"][layer_idx]
                compute = network["Compute"][layer_idx]

                # Update available memory in the DataFrame
                temp_df['Available_Memory'] = temp_df.apply(
                    lambda row: chiplet_availability[row['Cluster']].get(row['Chiplet'], 0), axis=1
                )

                # Sort chiplets with sufficient memory by temperature
                temp_data_sorted = temp_df[temp_df['Available_Memory'] >= storage].sort_values(by=['Peak_Temperature'])

                if temp_data_sorted.empty:
                    raise ValueError(f"Layer {layer_idx} of {network_name} cannot fit in any chiplet in {experiment}.")

                # Select the coolest chiplet with enough memory
                chiplet = temp_data_sorted.iloc[0]
                chiplet_id = f"{chiplet['Cluster']}_chiplet_{chiplet['Chiplet']}"

                # Compute time for this layer
                cluster = cluster_config[chiplet['Cluster']]
                compute_time = compute / cluster["tops"]
                total_compute_time += compute_time

                # Compute communication time if switching chiplets or clusters
                if current_chiplet and chiplet_id != current_chiplet:
                    comm_cost = activations * avg_hop_count
                    comm_time = comm_cost / interconnect_bandwidth
                    total_communication_time += comm_time

                # Update current chiplet and cluster
                current_chiplet = chiplet_id
                current_cluster = chiplet['Cluster']

                # Reduce the available memory of the selected chiplet
                chiplet_availability[chiplet['Cluster']][chiplet['Chiplet']] -= storage

            # Store results for this network in the current experiment
            exp_results[network_name] = {
                "Compute Time (s)": total_compute_time,
                "Communication Time (s)": total_communication_time,
                "Total Time (s)": total_compute_time + total_communication_time,
            }

        results[experiment] = exp_results

    return results


# Example Input
base_path = "/Users/harsh/Documents/Github/HyMu-Paper"

order = ["ResNet34", "ResNet50"]

# Parse temperature files
temp_data = parse_temperature_files(base_path)

# Execute mapping for all available experiments
results = map_layers(network_data, cluster_config, temp_data, order)

# Display Results
for experiment, result in results.items():
    print(f"\nResults for {experiment}:")
    print(pd.DataFrame(result).T)
