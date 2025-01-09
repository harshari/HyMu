import matplotlib.pyplot as plt
import numpy as np

# Data
layers = list(range(1, 19))  # Layer numbers
sensitivity = [68.22, 0.86, 0.08, 12.62, 0.45, 0.33, 4.95, 0.30, 0.06, 4.31, 0.19, 0.15, 2.21, 0.16, 0.03, 4.93, 0.09, 0.06]
compute = [118.01, 115.61, 115.61, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61, 57.8, 115.61, 6.42, 115.61, 115.61, 57.8, 115.61, 6.42]

# Create a dual-axis plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Line plot for sensitivity
ax1.plot(layers, sensitivity, color='tab:red', marker='o', label='Sensitivity')
ax1.set_xlabel('Layer Number')
ax1.set_ylabel('Sensitivity', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_xticks(layers)
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# Bar plot for compute on a secondary y-axis
ax2 = ax1.twinx()
ax2.bar(layers, compute, color='tab:blue', alpha=0.6, label='Compute', width=0.5)
ax2.set_ylabel('Compute', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add legends
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))

# Title
plt.title("Layer-wise Sensitivity and Compute for ResNet18")

# Show plot
plt.tight_layout()
plt.savefig("resnet18_sensitivity.png")
plt.show()
