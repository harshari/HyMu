import os
import sys

# Add parent directory to path to find mappingV2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simple_2obj.car_moo_ga import run_basic_car_example
from Simple_network_Problem.network_basic_ga import run_network_simple_example
from Simple_3obj_network.network_ga_optimize import run_network_three_obj_example
from mappingV2 import run_moo_simulation, cluster_config, network_data, visualize_results

def run_all_examples():
    print("Running Basic Car Example...")
    run_basic_car_example()
    
    print("\nRunning Simple Network Example...")
    run_network_simple_example()
    
    print("\nRunning Three Objective Network Example...")
    run_network_three_obj_example()
    
    print("\nRunning Full Chiplet Mapping...")
    networks_to_map = ["MobileNetV2", "ResNet18"]
    temp_file_path = "exp_0/output/temperature_chiplet_50.0.csv"
    results = run_moo_simulation(cluster_config, networks_to_map, network_data, temp_file_path)
    visualize_results(results, 0)

if __name__ == "__main__":
    run_all_examples()