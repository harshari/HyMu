from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import copy
from dataclasses import dataclass
import random
from scipy.optimize import minimize
from typing import Callable

# Import helper functions from your existing implementation
from floorplan_simulator import (
    calculate_module_area,
    calculate_package_utilization,
    calculate_wirelength,
    plot_chiplets_with_utilization,
    chiplets_with_types
)

@dataclass
class MacroPlacement:
    sequence_pair: Tuple[List[int], List[int]]
    cost: float
    hpwl: float
    area: float

class SequencePairKernel:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale
    
    def lcs_length(self, seq1: List[int], seq2: List[int]) -> int:
        """O(N log N) implementation of Longest Common Subsequence."""
        def patience_sort(seq):
            n = len(seq)
            if n == 0:
                return 0
            tails = [0] * n
            tails[0] = seq[0]
            length = 1
            
            for i in range(1, n):
                if seq[i] < tails[0]:
                    tails[0] = seq[i]
                elif seq[i] > tails[length-1]:
                    tails[length] = seq[i]
                    length += 1
                else:
                    tails[self._binary_search(tails, -1, length-1, seq[i])] = seq[i]
            return length
            
        return patience_sort([seq2[i] for i in seq1])
    
    def _binary_search(self, arr: List[int], l: int, r: int, key: int) -> int:
        while r - l > 1:
            m = l + (r - l) // 2
            if arr[m] >= key:
                r = m
            else:
                l = m
        return r
    
    def __call__(self, X1: List[Tuple[List[int], List[int]]], 
                 X2: List[Tuple[List[int], List[int]]]) -> np.ndarray:
        """Compute kernel matrix between sequence pairs."""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                sim1 = self.lcs_length(X1[i][0], X2[j][0]) / len(X1[i][0])
                sim2 = self.lcs_length(X1[i][1], X2[j][1]) / len(X1[i][1])
                K[i, j] = np.exp(-0.5 * ((1-sim1)**2 + (1-sim2)**2) / self.length_scale**2)
        
        return K

class MacroPlacementOptimizer:
    def __init__(self, 
                 chiplets: List[Dict],
                 connections: List[Tuple[int, int]],
                 batch_size: int = 2,
                 n_iter: int = 100):
        self.chiplets = chiplets
        self.connections = connections
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.n_macros = len(chiplets)
        
        # Create mapping between 0-based indices and chiplet IDs
        self.id_to_index = {c["id"]: i for i, c in enumerate(chiplets)}
        self.index_to_id = {i: c["id"] for i, c in enumerate(chiplets)}
        
        # Initialize GP model
        kernel = SequencePairKernel(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        self.X_train: List[Tuple[List[int], List[int]]] = []
        self.y_train: List[float] = []
    
    def generate_random_sequence_pair(self) -> Tuple[List[int], List[int]]:
        """Generate a random sequence pair using chiplet IDs."""
        chiplet_ids = [c["id"] for c in self.chiplets]
        seq1 = chiplet_ids.copy()
        seq2 = chiplet_ids.copy()
        random.shuffle(seq1)
        random.shuffle(seq2)
        return (seq1, seq2)
    
    def sequence_pair_to_placement(self, 
                                 sequence_pair: Tuple[List[int], List[int]]) -> List[Dict]:
        """Convert sequence pair to actual placement coordinates."""
        horizontal_constraints = []
        vertical_constraints = []
        
        # Extract constraints from sequence pair using chiplet IDs
        seq1, seq2 = sequence_pair
        for i in seq1:
            for j in seq1:
                if i != j:
                    if seq1.index(i) < seq1.index(j) and seq2.index(i) < seq2.index(j):
                        horizontal_constraints.append((i, j))
                    if seq1.index(i) < seq1.index(j) and seq2.index(i) > seq2.index(j):
                        vertical_constraints.append((i, j))
        
        # Create initial placement
        placement = copy.deepcopy(self.chiplets)
        
        # Reset initial positions to origin to avoid overlaps
        for chiplet in placement:
            chiplet["x"] = 0
            chiplet["y"] = 0
        
        # Apply constraints iteratively with minimum spacing
        min_spacing = 0.14
        for i, j in horizontal_constraints:
            c1 = next(c for c in placement if c["id"] == i)
            c2 = next(c for c in placement if c["id"] == j)
            c2["x"] = c1["x"] + c1["w"] + min_spacing
        
        for i, j in vertical_constraints:
            c1 = next(c for c in placement if c["id"] == i)
            c2 = next(c for c in placement if c["id"] == j)
            c2["y"] = c1["y"] + c1["h"] + min_spacing
        
        return placement
    
    def evaluate_placement(self, sequence_pair: Tuple[List[int], List[int]]) -> float:
        """Evaluate the placement cost."""
        try:
            # Convert sequence pair to actual placement
            placement = self.sequence_pair_to_placement(sequence_pair)
            
            # Calculate HPWL
            total_hpwl, _ = calculate_wirelength(placement, self.connections)
            
            # Calculate area and utilization
            module_area = calculate_module_area(placement)
            utilization = calculate_package_utilization(placement, module_area)
            
            # Composite cost function
            cost = total_hpwl + (1 - utilization) * module_area
            
            # Add penalty if placement is invalid
            if not self.is_valid_placement(placement):
                cost *= 2
                
            return cost
        except Exception as e:
            print(f"Error in evaluate_placement: {e}")
            return float('inf')
    
    def is_valid_placement(self, placement: List[Dict]) -> bool:
        """Check if placement is valid (no overlaps)."""
        for i, c1 in enumerate(placement):
            for j, c2 in enumerate(placement):
                if i < j:
                    # Check for overlap
                    if not (c1["x"] + c1["w"] + 0.14 <= c2["x"] or
                            c2["x"] + c2["w"] + 0.14 <= c1["x"] or
                            c1["y"] + c1["h"] + 0.14 <= c2["y"] or
                            c2["y"] + c2["h"] + 0.14 <= c1["y"]):
                        return False
        return True
    
    def optimize(self) -> Tuple[MacroPlacement, List[MacroPlacement]]:
        """Main optimization loop."""
        try:
            # Initialize with random points
            while len(self.X_train) < 5:  # Initial population
                seq_pair = self.generate_random_sequence_pair()
                cost = self.evaluate_placement(seq_pair)
                if cost != float('inf'):
                    self.X_train.append(seq_pair)
                    self.y_train.append(cost)
            
            history = []
            best_placement = None
            best_cost = float('inf')
            
            for iteration in range(self.n_iter):
                try:
                    # Fit GP model
                    self.gp.fit(self.X_train, self.y_train)
                    
                    # Get next batch of points
                    next_points = self.optimize_acquisition()
                    
                    # Evaluate new points
                    for seq_pair in next_points:
                        cost = self.evaluate_placement(seq_pair)
                        if cost != float('inf'):
                            self.X_train.append(seq_pair)
                            self.y_train.append(cost)
                            
                            placement = self.sequence_pair_to_placement(seq_pair)
                            current_placement = MacroPlacement(
                                sequence_pair=seq_pair,
                                cost=cost,
                                hpwl=calculate_wirelength(placement, self.connections)[0],
                                area=calculate_module_area(placement)
                            )
                            history.append(current_placement)
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_placement = current_placement
                    
                    print(f"Iteration {iteration}: Best cost = {best_cost}")
                
                except Exception as e:
                    print(f"Error in iteration {iteration}: {e}")
                    continue
            
            return best_placement, history
        
        except Exception as e:
            print(f"Error in optimize: {e}")
            raise

# Example usage remains the same
if __name__ == "__main__":
    connections = [(1, 2), (4, 5), (1, 3), (3, 5)]
    
    optimizer = MacroPlacementOptimizer(
        chiplets=chiplets_with_types,
        connections=connections,
        batch_size=2,
        n_iter=50
    )
    
    best_placement, history = optimizer.optimize()
    
    if best_placement:
        print("\nBest placement found:")
        print(f"Cost: {best_placement.cost}")
        print(f"HPWL: {best_placement.hpwl}")
        print(f"Area: {best_placement.area}")
        
        final_placement = optimizer.sequence_pair_to_placement(best_placement.sequence_pair)
        plot_chiplets_with_utilization(
            final_placement,
            best_placement.area,
            calculate_package_utilization(final_placement, best_placement.area)
        )
    else:
        print("Optimization failed to find a valid placement")