# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:37:07 2025

@author: yifan
"""

import numpy as np
import matplotlib.pyplot as plt

# Data: Preferred phase and 95% Confidence Interval (CI) for 6 animals
animals = np.arange(1, 7)  # Animal IDs
preferred_phases = np.array([-146.38, -150.27, -112.69, -138.46, -171.97, -98.43])  # Degrees
ci_lower = np.array([-159.70, -176.15, -137.52, -156.70, -193.94, -128.70])  # Lower bound
ci_upper = np.array([-133.06, -124.38, -87.86, -120.22, -150.00, -68.15])  # Upper bound

# Calculate error bars
lower_errors = preferred_phases - ci_lower
upper_errors = ci_upper - preferred_phases
errors = [lower_errors, upper_errors]

# Create figure
plt.figure(figsize=(6, 4))

# Plot each animal separately to control colours
for i, animal in enumerate(animals):
    color = 'blue' if animal in [3, 6] else 'red'
    plt.errorbar(animal, preferred_phases[i], yerr=[[lower_errors[i]], [upper_errors[i]]],
                 fmt='o', color=color, capsize=5, label=f"Animal {animal}" if animal in [3, 6] else "")

# Formatting
plt.axhline(0, color='grey', linestyle='--', alpha=0.7)
plt.yticks(np.arange(-200, 50, 25))  # Adjusted for better visibility
plt.xticks(animals, [f"Animal {i}" for i in animals])
plt.ylabel("Theta Phase (Degrees)")
plt.title("Preferred Theta Phase Across Animals with 95% CI")
plt.grid(axis='y', linestyle='--', alpha=0.5)
# Show plot
plt.show()