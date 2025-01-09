import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Helper function for second-order polynomial
def second_order_poly(x, a, b, c):
    return a * x**2 + b * x + c

# Function to plot curves with manual points
def plot_accuracy_vs_edp(data_points):
    plt.figure(figsize=(10, 6))
    for label, points in data_points.items():
        # Uncomment and provide manual points like [(x1, y1), (x2, y2), ...]
        # points = [(10, 0.85), (20, 0.83), (30, 0.80), (40, 0.78), (50, 0.75)]

        # Extract x and y values
        x_values, y_values = zip(*points)

        # Scatter points
        plt.scatter(x_values, y_values, label=label, alpha=0.8)

        # Curve fitting
        popt, _ = curve_fit(second_order_poly, x_values, y_values)
        x_curve = np.linspace(min(x_values), max(x_values), 100)
        y_curve = second_order_poly(x_curve, *popt)

        # Plot the fitted curve
        plt.plot(x_curve, y_curve, linestyle="--")
        
        # Add shaded region for variation
        std_dev = np.std(y_values)  # Calculate standard deviation from points
        plt.fill_between(x_curve, y_curve - std_dev/5, y_curve + std_dev/5, alpha=0.1)

    # Customize plot
    plt.xlabel("Energy Delay Product (EDP) (in mJ-ms)")
    plt.ylabel("Accuracy Loss (%)")
    plt.title("Accuracy vs EDP for different NoI architectures")
    plt.legend(title="Different NoI architectures")
    plt.grid(True)
    plt.show()

# Example data input
data_points = {
    "HexaMesh": [(300, 16), (330, 14), (360, 10), (440, 2)]
    # "Kite": [(295, 21), (306, 19.3), (416, 5), (427, 4), (450, 1.8)],
    # "Mesh": [(300, 20), (309, 17), (346, 14), (420, 1.4), (430, .9)],
    # "Degenerated_NoI": [(200, 3.2), (240, 1.3), (251, .60), (268, .4), ]
}
# "Floret": [
#     (200, 90.8), (205, 91.3), (211, 91.6), (217, 91.4), (222, 91.8),
#     (228, 91.7), (234, 92.0), (240, 92.7), (245, 92.4), (251, 93.0),
#     (257, 93.4), (262, 93.0), (268, 93.6), (274, 93.5), (280, 94.5)
# ]
# "Mesh": [
#     (300, 73.5), (309, 76.7), (318, 78.8), (327, 78.1), (337, 81.4),
#     (346, 80.1), (355, 84.0), (365, 83.6), (374, 84.1), (383, 86.5),
#     (392, 87.1), (402, 88.5), (411, 89.9), (420, 92.6), (430, 93.1)
# ]

# "Kite": [
#     (295, 73.1), (306, 74.3), (317, 73.8), (328, 76.4), (339, 77.9),
#     (350, 79.4), (361, 81.5), (372, 82.6), (383, 84.3), (394, 86.3),
#     (405, 87.8), (416, 88.8), (427, 91.1), (438, 88.7), (450, 90.8)
# ]
# "HexaMesh": [
#     (300, 78), (280, 69), (330, 80), (400, 84), (440, 92),
#     (470, 89), (418, 93), (432, 85), (495, 85), (434, 90),
#     (489, 87), (471, 94), (465, 93), (402, 89), (425, 92)
# ],
# Call the function with the data
plot_accuracy_vs_edp(data_points)