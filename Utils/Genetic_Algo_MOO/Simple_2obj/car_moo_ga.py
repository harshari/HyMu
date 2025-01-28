import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

def run_basic_car_example():
    class CarProblem(Problem):
        def __init__(self):
            super().__init__(n_var=1, n_obj=2, xl=0, xu=4, vtype=int)
            
        def _evaluate(self, x, out, *args, **kwargs):
            cars = [
                (20000, 70),  # (cost, performance)
                (25000, 75),
                (30000, 80),
                (35000, 85),
                (40000, 90)
            ]
            
            f1 = [cars[int(i)][0] for i in x]  # Cost
            f2 = [-cars[int(i)][1] for i in x]  # -Performance
            
            out["F"] = np.column_stack([f1, f2])

    # Setup and run optimization
    problem = CarProblem()
    algorithm = NSGA2(pop_size=20)
    result = minimize(problem, algorithm, ('n_gen', 10))

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(result.F[:, 0], -result.F[:, 1], c='blue', label='Solutions')
    plt.xlabel('Cost ($)')
    plt.ylabel('Performance Score')
    plt.title('Car Selection Pareto Front')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print best solutions
    print("\nBest solutions found:")
    for i in range(len(result.F)):
        print(f"Solution {i+1}: Cost=${result.F[i,0]:.0f}, Performance={-result.F[i,1]:.1f}")

if __name__ == "__main__":
    run_basic_car_example()