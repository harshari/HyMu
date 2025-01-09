import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NetworkProblem3Obj(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=3, xl=0, xu=1, vtype=int)
        
    def _evaluate(self, x, out, *args, **kwargs):
        processors = [
            {"speed": 1.0, "energy": 2.0, "accuracy": 0.99},
            {"speed": 2.0, "energy": 4.0, "accuracy": 0.95}
        ]
        
        layers = [
            {"compute": 100, "size": 50, "sensitivity": 0.1},
            {"compute": 200, "size": 100, "sensitivity": 0.2},
            {"compute": 150, "size": 75, "sensitivity": 0.3}
        ]
        
        total_time = []
        total_energy = []
        accuracy_loss = []
        
        # Convert x to integer indices
        x_int = x.astype(int)
        
        for solution in x_int:
            time = 0
            energy = 0
            acc_loss = 0
            
            for layer_idx, proc_idx in enumerate(solution):
                proc = processors[proc_idx]
                layer = layers[layer_idx]
                
                time += layer["compute"] / proc["speed"]
                energy += layer["compute"] * proc["energy"]
                acc_loss = max(acc_loss, 
                             layer["sensitivity"] * (1 - proc["accuracy"]))
            
            total_time.append(time)
            total_energy.append(energy)
            accuracy_loss.append(acc_loss)
        
        out["F"] = np.column_stack([total_time, total_energy, accuracy_loss])

def run_network_three_obj_example():
    problem = NetworkProblem3Obj()
    algorithm = NSGA2(pop_size=100)
    result = minimize(problem, algorithm, ('n_gen', 50))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(result.F[:, 0], result.F[:, 1], result.F[:, 2],
                        c='green', marker='o', label='Solutions')
    ax.set_xlabel('Compute Time')
    ax.set_ylabel('Energy')
    ax.set_zlabel('Accuracy Loss')
    plt.title('3D Pareto Front')
    plt.legend()
    plt.show()

    print("\nBest solutions found:")
    print("Minimum Time Solution:", result.F[np.argmin(result.F[:, 0])])
    print("Minimum Energy Solution:", result.F[np.argmin(result.F[:, 1])])
    print("Minimum Accuracy Loss Solution:", result.F[np.argmin(result.F[:, 2])])

if __name__ == "__main__":
    run_network_three_obj_example()