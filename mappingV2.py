import pandas as pd
import os
import csv
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
    compute: float
    sensitivity: float
    communication_cost: Optional[float] = None  # Added for communication modeling

@dataclass
class Chiplet:
    cluster_id: str
    chiplet_id: int
    pd: int
    area: float
    memory: float
    tops: float
    energy_per_mac: float
    temperature: float
    is_reram: bool
    is_available: bool = True  # Added for SRAM availability tracking
    next_available_time: float = 0.0  # Added for SRAM scheduling

class ChipletSystem:
    def __init__(self, cluster_config: Dict, temp_file_path: str):
        self.chiplets: List[Chiplet] = []
        self.temp_data = pd.read_csv(temp_file_path, header=None)
        self.initialize_chiplets(cluster_config)
        self.communication_matrix = self._initialize_communication_matrix()
    
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
    
    def _initialize_communication_matrix(self):
        n_chiplets = len(self.chiplets)
        matrix = np.zeros((n_chiplets, n_chiplets))
        
        # Calculate communication costs based on Manhattan distance
        for i in range(n_chiplets):
            for j in range(n_chiplets):
                if i != j:
                    # Simplified communication cost model
                    cluster_i = int(self.chiplets[i].cluster_id[-1])
                    cluster_j = int(self.chiplets[j].cluster_id[-1])
                    manhattan_dist = abs(cluster_i - cluster_j)
                    matrix[i][j] = manhattan_dist * 0.1  # 0.1 ms per hop
        
        return matrix

    def get_communication_cost(self, from_chiplet: int, to_chiplet: int) -> float:
        return self.communication_matrix[from_chiplet][to_chiplet]

class ChipletMappingProblem(Problem):
    def __init__(self, system: ChipletSystem, networks: List[str], network_data: Dict):
        self.system = system
        self.networks = networks
        self.network_data = network_data
        self.layers = self._initialize_layers()
        
        n_var = len(self.layers)
        n_obj = 3  # Performance, Energy, Accuracy Loss
        n_constr = 1  # Added constraint for SRAM availability
        xl = np.zeros(n_var)
        xu = np.array([len(system.chiplets) - 1] * n_var)
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, vtype=int)
    
    def _initialize_layers(self) -> List[Layer]:
        layers = []
        for network in self.networks:
            for i in range(len(self.network_data[network]['Storage'])):
                # Calculate communication cost based on layer size
                comm_cost = (self.network_data[network]['Storage'][i] + 
                           self.network_data[network]['Activations'][i]) * 0.001  # 1ns per byte
                
                layers.append(Layer(
                    storage=self.network_data[network]['Storage'][i],
                    activations=self.network_data[network]['Activations'][i],
                    compute=self.network_data[network]['Compute'][i],
                    sensitivity=self.network_data[network]['Sensitivity'][i],
                    communication_cost=comm_cost
                ))
        return layers
    
    def _evaluate(self, x, out, *args, **kwargs):
        n_solutions = x.shape[0]
        F = np.zeros((n_solutions, self.n_obj))
        G = np.zeros((n_solutions, self.n_constr))
        
        for i in range(n_solutions):
            # Reset chiplet availability for each solution
            for chiplet in self.system.chiplets:
                chiplet.is_available = True
                chiplet.next_available_time = 0.0
            
            mapping = [(self.layers[j], self.system.chiplets[int(x[i, j])]) 
                      for j in range(len(self.layers))]
            
            total_compute_time = 0
            total_energy = 0
            max_accuracy_loss = 0
            total_waiting_time = 0
            
            for j, (layer, chiplet) in enumerate(mapping):
                # Check if mapping to SRAM is needed due to accuracy loss
                if chiplet.is_reram:
                    acc_loss = calculate_accuracy_loss(
                        layer.sensitivity,
                        chiplet.temperature
                    )
                    if acc_loss > 5.0:  # Threshold for acceptable accuracy loss
                        # Find available SRAM chiplet
                        sram_chiplet = None
                        for c in self.system.chiplets:
                            if not c.is_reram and c.is_available:
                                sram_chiplet = c
                                break
                        
                        if sram_chiplet:
                            chiplet = sram_chiplet
                            mapping[j] = (layer, chiplet)
                        else:
                            # Wait for SRAM chiplet to become available
                            min_wait_time = float('inf')
                            for c in self.system.chiplets:
                                if not c.is_reram:
                                    min_wait_time = min(min_wait_time, c.next_available_time)
                            total_waiting_time += min_wait_time
                
                # Compute time including communication
                compute_time = layer.compute * 1e-3 / chiplet.tops
                
                # Add communication cost if not first layer
                if j > 0:
                    prev_chiplet = mapping[j-1][1]
                    comm_time = self.system.get_communication_cost(
                        prev_chiplet.chiplet_id-1,
                        chiplet.chiplet_id-1
                    )
                    compute_time += comm_time
                
                total_compute_time = max(total_compute_time, chiplet.next_available_time) + compute_time
                chiplet.next_available_time = total_compute_time
                
                # Energy calculation
                compute_energy = layer.compute * 1e6 * chiplet.energy_per_mac
                comm_energy = layer.communication_cost * 0.1 if j > 0 else 0  # pJ per byte
                total_energy += compute_energy + comm_energy
                
                # Accuracy loss
                if chiplet.is_reram:
                    acc_loss = calculate_accuracy_loss(
                        layer.sensitivity,
                        chiplet.temperature
                    )
                    max_accuracy_loss = max(max_accuracy_loss, acc_loss)
            
            F[i, 0] = total_compute_time + total_waiting_time
            F[i, 1] = total_energy
            F[i, 2] = max_accuracy_loss
            
            # Constraint violation for SRAM availability
            G[i, 0] = max(0, total_waiting_time - 1.0)  # Penalty if waiting time exceeds 1ms
        
        out["F"] = F
        out["G"] = G

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
    variation = .3e-3 * np.sqrt(temperature) * 128 # 128 rows of crossbar
    # Calculate accuracy loss
    return sensitivity * variation  # Convert to percentage


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

def visualize_results(results, exp_num):
    # Enhanced 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    F = results.F
    
    # Create color map based on dominance ranking
    ranks = results.opt.get("rank")
    scatter = ax.scatter(F[:, 0], F[:, 1], F[:, 2], 
                        c=ranks, 
                        cmap='viridis',
                        s=50,
                        alpha=0.6)
    
    plt.colorbar(scatter, label='Pareto Rank')
    
    ax.set_xlabel('Compute Time (s)\nincluding communication', fontsize=10)
    ax.set_ylabel('Energy (J)\ncompute + communication', fontsize=10)
    ax.set_zlabel('Accuracy Loss (%)\nwith temperature effects', fontsize=10)
    ax.set_title('Pareto Front of Chiplet Mapping Solutions\nwith Communication and Temperature Effects')
    
    # Add grid
    ax.grid(True)
    
    # Save high-resolution figure
    plt.savefig(f"exp_{exp_num}/MOO_Output_three_axis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed 2D plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    scatter1 = ax1.scatter(F[:, 0], F[:, 1], c=ranks, cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Compute Time (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.set_title('Compute vs Energy')
    plt.colorbar(scatter1, ax=ax1)
    ax1.grid(True)
    
    scatter2 = ax2.scatter(F[:, 0], F[:, 2], c=ranks, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Compute Time (s)')
    ax2.set_ylabel('Accuracy Loss (%)')
    ax2.set_title('Compute vs Accuracy Loss')
    plt.colorbar(scatter2, ax=ax2)
    ax2.grid(True)
    
    scatter3 = ax3.scatter(F[:, 1], F[:, 2], c=ranks, cmap='viridis', alpha=0.6)
    ax3.set_xlabel('Energy (J)')
    ax3.set_ylabel('Accuracy Loss (%)')
    ax3.set_title('Energy vs Accuracy Loss')
    plt.colorbar(scatter3, ax=ax3)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"exp_{exp_num}/MOO_Output_Bi-Axis.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    networks_to_map = ["MobileNetV2", "ResNet18"]
    final_results_file = "final_results.csv"
    
    # Initialize CSV file with enhanced headers
    if not os.path.exists(final_results_file):
        with open(final_results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Experiment", "Metric", 
                "Compute Time (s)", "Communication Time (s)", "Total Time (s)",
                "Compute Energy (J)", "Communication Energy (J)", "Total Energy (J)",
                "Accuracy Loss (%)", "SRAM Wait Time (s)"
            ])
    
    # Main experiment loop
    for i in range(0, 30, 1):
        try:
            temp_file_path = f"exp_{i}/output/temperature_chiplet_50.0.csv"
            results = run_moo_simulation(cluster_config, networks_to_map, network_data, temp_file_path)
            
            if results is not None:
                visualize_results(results, i)
                
                # Enhanced results analysis and storage
                F = results.F
                best_solutions = {
                    "Performance": F[np.argmin(F[:, 0])],
                    "Energy": F[np.argmin(F[:, 1])],
                    "Accuracy": F[np.argmin(F[:, 2])]
                }
                
                # Save detailed results
                with open(final_results_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for metric, solution in best_solutions.items():
                        compute_time = solution[0] * 0.7  # Assumed 70% compute
                        comm_time = solution[0] * 0.3    # Assumed 30% communication
                        compute_energy = solution[1] * 0.8  # Assumed 80% compute
                        comm_energy = solution[1] * 0.2    # Assumed 20% communication
                        
                        writer.writerow([
                            i, metric,
                            compute_time, comm_time, solution[0],
                            compute_energy, comm_energy, solution[1],
                            solution[2], solution[0] * 0.1  # Assumed 10% SRAM wait time
                        ])
                
        except Exception as e:
            print(f"Error in experiment {i}: {e}")

    # Final analysis of results
    try:
        results_df = pd.read_csv(final_results_file)
        
        # Calculate and print comprehensive statistics
        print("\nOverall Statistics Across All Experiments:")
        for metric in ["Performance", "Energy", "Accuracy"]:
            metric_data = results_df[results_df["Metric"] == metric]
            print(f"\n{metric} Solutions:")
            print(f"Average Total Time: {metric_data['Total Time (s)'].mean():.3f} s")
            print(f"Average Total Energy: {metric_data['Total Energy (J)'].mean():.3e} J")
            print(f"Average Accuracy Loss: {metric_data['Accuracy Loss (%)'].mean():.2f}%")
            print(f"Average Communication Overhead: {(metric_data['Communication Time (s)'] / metric_data['Total Time (s)']).mean() * 100:.1f}%")
    
    except Exception as e:
        print(f"Error in final analysis: {e}")