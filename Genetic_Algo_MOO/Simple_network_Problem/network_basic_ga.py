import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

class NetworkProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, xl=0, xu=1, vtype=int)
        
    def _evaluate(self, x, out, *args, **kwargs):
        processors = [
            {"speed": 1.0, "energy": 2.0},  # Processor 0
            {"speed": 2.0, "energy": 4.0}   # Processor 1
        ]
        
        layers = [
            {"compute": 100, "size": 50},  # Layer 0
            {"compute": 200, "size": 100}, # Layer 1
            {"compute": 150, "size": 75}   # Layer 2
        ]
        
        total_time = []
        total_energy = []
        
        # Convert x to integer indices
        x_int = x.astype(int)
        
        for solution in x_int:
            time = 0
            energy = 0
            
            for layer_idx, proc_idx in enumerate(solution):
                proc = processors[proc_idx]
                layer = layers[layer_idx]
                
                time += layer["compute"] / proc["speed"]
                energy += layer["compute"] * proc["energy"]
            
            total_time.append(time)
            total_energy.append(energy)
        
        out["F"] = np.column_stack([total_time, total_energy])

def run_network_simple_example():
    problem = NetworkProblem()
    algorithm = NSGA2(pop_size=50)
    result = minimize(problem, algorithm, ('n_gen', 20))

    plt.figure(figsize=(10, 6))
    plt.scatter(result.F[:, 0], result.F[:, 1], c='red', label='Solutions')
    plt.xlabel('Compute Time')
    plt.ylabel('Energy Consumption')
    plt.title('Network Mapping Pareto Front')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nBest solutions found:")
    print("Minimum Time Solution:", result.F[np.argmin(result.F[:, 0])])
    print("Minimum Energy Solution:", result.F[np.argmin(result.F[:, 1])])

if __name__ == "__main__":
    run_network_simple_example()