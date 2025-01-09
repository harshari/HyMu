import pandas as pd
import os
import csv
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling

i = 4
temp_file_path = f"/Users/harsh/Documents/Github/HyMu-Paper/exp_{i}/output/temperature_chiplet_50.0.csv"
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

@dataclass
class Layer:
    storage: float
    activations: float
    compute: float  # in TOPS
    sensitivity: float

@dataclass
class Chiplet:
    cluster_id: str
    chiplet_id: int
    pd: int
    area: float
    memory: float
    tops: float  # in TOPS
    energy_per_mac: float
    temperature: float
    is_reram: bool

class ChipletSystem:
    def __init__(self, cluster_config: Dict, temp_file_path: str):
        self.chiplets: List[Chiplet] = []
        self.temp_data = pd.read_csv(temp_file_path, header=None)
        self.initialize_chiplets(cluster_config)
        
    def initialize_chiplets(self, cluster_config: Dict):
        reram_clusters = {'Cluster 1', 'Cluster 3'}
        
        for cluster_name, config in cluster_config.items():
            for i in range(config['count']):
                chiplet_id = i + 1
                chiplet_key = f"C{cluster_name[-1]}_chiplet_{chiplet_id}"
                
                # Get temperature from dataframe
                temp_row = self.temp_data[self.temp_data[0] == chiplet_key]
                temp = temp_row.iloc[0, 2] if not temp_row.empty else 300.0
                
                self.chiplets.append(Chiplet(
                    cluster_id=cluster_name,
                    chiplet_id=chiplet_id,
                    pd=config['pd'],
                    area=config['area'],
                    memory=config['memory'],
                    tops=config['tops'] / 1e12,  # Convert to TOPS
                    energy_per_mac=config['energy_per_mac'],
                    temperature=temp,
                    is_reram=cluster_name in reram_clusters
                ))

class ChipletMappingProblem(Problem):
    def __init__(self, system: ChipletSystem, layers: List[Layer]):
        self.system = system
        self.layers = layers
        n_var = len(layers)  # Each variable represents which chiplet to use for each layer
        n_obj = 3  # Performance, Energy, Sensitivity
        n_constr = 0
        xl = np.zeros(n_var)  # Chiplet indices start from 0
        xu = np.array([len(system.chiplets) - 1] * n_var)  # Max chiplet index
        
        super().__init__(n_var=n_var, 
                        n_obj=n_obj, 
                        n_constr=n_constr, 
                        xl=xl, 
                        xu=xu, 
                        vtype=int)

def calculate_accuracy_loss(sensitivity: float, temperature: float, frequency: float = 100e6) -> float:
    """Calculate accuracy loss based on ReRAM variation at temperature
    
    Args:
        sensitivity: Layer's sensitivity to weight variation
        temperature: Temperature in Kelvin
        frequency: Operating frequency (default 100MHz)
    
    Returns:
        Accuracy loss percentage
    """
    KB = 1.380649e-23  # Boltzmann constant
    V = 0.5  # Assuming 0.5V operating voltage for ReRAM
    # Calculate thermal noise variation
    variation = .3e-3 * np.sqrt(temperature) * 128 # 128 rows of crossbar # 1.29e-5 *
    # Calculate accuracy loss
    return sensitivity * variation  # Convert to percentage

class ChipletMappingProblem(Problem):
    def __init__(self, system: ChipletSystem, networks: List[str], network_data: Dict):
        self.system = system
        self.networks = networks
        self.network_data = network_data
        
        # First check if total storage fits
        total_storage = 0
        self.layers = []
        
        for network in networks:
            network_storage = sum(self.network_data[network]['Storage'])
            total_storage += network_storage
            
            # Collect all layers from all networks
            for i in range(len(self.network_data[network]['Storage'])):
                self.layers.append(Layer(
                    storage=self.network_data[network]['Storage'][i],
                    activations=self.network_data[network]['Activations'][i],
                    compute=self.network_data[network]['Compute'][i],
                    sensitivity=self.network_data[network]['Sensitivity'][i]
                ))
        
        # Calculate total system storage
        system_storage = sum(chiplet.memory for chiplet in system.chiplets)
        
        if total_storage > system_storage:
            raise ValueError(f"Total network storage ({total_storage}) exceeds system capacity ({system_storage})")
        
        n_var = len(self.layers)
        n_obj = 3  # Performance, Energy, Accuracy Loss
        n_constr = 0
        xl = np.zeros(n_var)
        xu = np.array([len(system.chiplets) - 1] * n_var)
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, vtype=int)
    
    def _evaluate(self, x, out, *args, **kwargs):
        n_solutions = x.shape[0]
        F = np.zeros((n_solutions, self.n_obj))
        
        for i in range(n_solutions):
            mapping = [(self.layers[j], self.system.chiplets[int(x[i, j])]) 
                      for j in range(len(self.layers))]
            
            compute_time = 0
            energy = 0
            accuracy_loss = 0
            
            for layer, chiplet in mapping:
                # Compute time
                compute_time += (layer.compute * 1e-3 / chiplet.tops)
                # Energy
                energy += layer.compute * 1e6 * chiplet.energy_per_mac
                
                # Accuracy loss (only for ReRAM)
                if chiplet.is_reram:
                    acc_loss = calculate_accuracy_loss(
                        layer.sensitivity,
                        chiplet.temperature
                    )
                    accuracy_loss = max(accuracy_loss, acc_loss)  # Take worst case accuracy loss
            
            F[i, 0] = compute_time  # Minimize compute time
            F[i, 1] = energy        # Minimize energy
            F[i, 2] = accuracy_loss # Minimize accuracy loss
            
        out["F"] = F

def run_moo_simulation(cluster_config: Dict, network_list: List[str], network_data: Dict, temp_file_path: str):
    """
    Run MOO simulation for multiple networks
    
    Args:
        cluster_config: Configuration of chiplet clusters
        network_list: List of networks to map (e.g., ["ResNet18", "VGG16"])
        network_data: Dictionary containing network specifications
        temp_file_path: Path to temperature data file
    """
    # Initialize system
    system = ChipletSystem(cluster_config, temp_file_path)
    
    try:
        # Setup and run MOO (this will check storage constraints)
        problem = ChipletMappingProblem(system, network_list, network_data)
        
        algorithm = NSGA2(
            pop_size=100,
            n_offsprings=100,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(problem,
                      algorithm,
                      ('n_gen', 100),
                      seed=1,
                      verbose=True)
        
        return res
        
    except ValueError as e:
        print(f"Error: {e}")
        return None

def visualize_results(results):
    # Create 3D scatter plot for Pareto front
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Pareto front
    F = results.F
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='b', marker='o')
    
    ax.set_xlabel('Compute Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_zlabel('Accuracy loss (in %)')
    ax.set_title('Pareto Front of Chiplet Mapping Solutions')
    # plt.show()
    plt.savefig(f"exp_{i}/MOO_Output_three_axis.png")
    
    # Create 2D plots for each pair of objectives
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.scatter(F[:, 0], F[:, 1])
    ax1.set_xlabel('Compute Time (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Compute vs Energy')
    
    ax2.scatter(F[:, 0], F[:, 2])
    ax2.set_xlabel('Compute Time (s)')
    ax2.set_ylabel('Accuracy loss (in %)')
    ax2.set_title('Compute vs Accuracy loss')
    
    ax3.scatter(F[:, 1], F[:, 2])
    ax3.set_xlabel('Energy (J)')
    ax3.set_ylabel('Accuracy loss (in %)')
    ax3.set_title('Energy vs Accuracy loss')
    
    plt.tight_layout()
    plt.savefig(f"exp_{i}/MOO_Output_Bi-Axis.png")


# Example usage:
if __name__ == "__main__":
    networks_to_map = ["MobileNetV2", "ResNet18"]  # Example network combination


# Define the file path for the final results CSV
final_results_file = "/Users/harsh/Documents/Github/HyMu-Paper/final_results.csv"

# Initialize the CSV file with headers
if not os.path.exists(final_results_file):
    with open(final_results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "Metric", "Compute Time (s)", "Energy (J)", "Accuracy Loss (%)"])

# Main experiment loop
for i in range(0, 30, 1):
    try:
        temp_file_path = f"/Users/harsh/Documents/Github/HyMu-Paper/exp_{i}/output/temperature_chiplet_50.0.csv"
        print(f"Attempting to map networks: {networks_to_map}")
        results = run_moo_simulation(cluster_config, networks_to_map, network_data, temp_file_path)
        if results is not None:
            # Find best solutions
            F = results.F
            best_performance_idx = np.argmin(F[:, 0])
            best_energy_idx = np.argmin(F[:, 1])
            best_accuracy_idx = np.argmin(F[:, 2])
            
            best_performance = F[best_performance_idx]
            best_energy = F[best_energy_idx]
            best_accuracy = F[best_accuracy_idx]
            
            print("\nBest solutions found:")
            print(f"Best Performance Solution:")
            print(f"Compute Time: {best_performance[0]:.2f} s")
            print(f"Energy: {best_performance[1]:.2e} J")
            print(f"Accuracy Loss: {best_performance[2]:.2f}%")
            
            print(f"\nBest Energy Solution:")
            print(f"Compute Time: {best_energy[0]:.2f} s")
            print(f"Energy: {best_energy[1]:.2e} J")
            print(f"Accuracy Loss: {best_energy[2]:.2f}%")
            
            print(f"\nBest Accuracy Solution:")
            print(f"Compute Time: {best_accuracy[0]:.2f} s")
            print(f"Energy: {best_accuracy[1]:.2e} J")
            print(f"Accuracy Loss: {best_accuracy[2]:.2f}%")
            
            # Save the best solutions to the CSV file
            with open(final_results_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, "Performance", best_performance[0], best_performance[1], best_performance[2]])
                writer.writerow([i, "Energy", best_energy[0], best_energy[1], best_energy[2]])
                writer.writerow([i, "Accuracy", best_accuracy[0], best_accuracy[1], best_accuracy[2]])
            
            # Visualize results
            visualize_results(results)
    
    except FileNotFoundError:
        print(f"File not found: {temp_file_path}. Skipping this iteration.")
    except Exception as e:
        print(f"An error occurred during iteration {i}: {e}")

# Load the final_results.csv and determine the global best solutions
try:
    with open(final_results_file, mode='r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        
        # Parse the rows to find the best overall solutions
        best_latency = min(rows, key=lambda x: float(x["Compute Time (s)"]))
        best_energy = min(rows, key=lambda x: float(x["Energy (J)"]))
        best_accuracy = min(rows, key=lambda x: float(x["Accuracy Loss (%)"]))
        
        print("\nOverall Best Designs Across All Experiments:")
        print(f"Best Latency: Experiment {best_latency['Experiment']}, Compute Time = {best_latency['Compute Time (s)']} s")
        print(f"Best Energy: Experiment {best_energy['Experiment']}, Energy = {best_energy['Energy (J)']} J")
        print(f"Best Accuracy: Experiment {best_accuracy['Experiment']}, Accuracy Loss = {best_accuracy['Accuracy Loss (%)']} %")
except Exception as e:
    print(f"An error occurred while processing the final results: {e}")



    
    # for i in range(0, 30, 1):
    #     try:
    #         temp_file_path = f"/Users/harsh/Documents/Github/HyMu-Paper/exp_{i}/output/temperature_chiplet_50.0.csv"
    #         print(f"Attempting to map networks: {networks_to_map}")
    #         results = run_moo_simulation(cluster_config, networks_to_map, network_data, temp_file_path)
            
    #         if results is not None:
    #             # Find best solutions
    #             F = results.F
    #             best_performance_idx = np.argmin(F[:, 0])
    #             best_energy_idx = np.argmin(F[:, 1])
    #             best_accuracy_idx = np.argmin(F[:, 2])
                
    #             print("\nBest solutions found:")
    #             print(f"Best Performance Solution:")
    #             print(f"Compute Time: {F[best_performance_idx, 0]:.2f} s")
    #             print(f"Energy: {F[best_performance_idx, 1]:.2e} J")
    #             print(f"Accuracy Loss: {F[best_performance_idx, 2]:.2f}%")
                
    #             print(f"\nBest Energy Solution:")
    #             print(f"Compute Time: {F[best_energy_idx, 0]:.2f} s")
    #             print(f"Energy: {F[best_energy_idx, 1]:.2e} J")
    #             print(f"Accuracy Loss: {F[best_energy_idx, 2]:.2f}%")
                
    #             print(f"\nBest Accuracy Solution:")
    #             print(f"Compute Time: {F[best_accuracy_idx, 0]:.2f} s")
    #             print(f"Energy: {F[best_accuracy_idx, 1]:.2e} J")
    #             print(f"Accuracy Loss: {F[best_accuracy_idx, 2]:.2f}%")
                
    #             # Visualize results
    #             visualize_results(results)
        
    #     except FileNotFoundError:
    #         print(f"File not found: {temp_file_path}. Skipping this iteration.")
    #     except Exception as e:
    #         print(f"An error occurred during iteration {i}: {e}")
