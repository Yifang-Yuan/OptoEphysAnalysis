# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 17:00:17 2025

@author: yifan
"""

import numpy as np
import matplotlib.pyplot as plt

# Define sensor types and their features
categories = [
    "Quantum Efficiency", "Time Resolution", "Spatial Resolution", 
    "Low Noise", "Sensitivity", "Suitability for Fibre Photometry", 
    "Ease for Multi-ROI Imaging", "Affordability"
]

# Data for each sensor type (normalized 0-1 scale)
sensor_data = {
    "SPAD": [0.7, 0.9, 0.6, 0.9, 0.9, 0.9, 0.9, 0.7],
    "PMT": [0.3, 0.7, 0.2, 0.7, 0.7, 0.8, 0.3, 0.3],
    "ICCD": [0.5, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
    "EMCCD": [0.9, 0.3, 0.7, 0.7, 0.7, 0.3, 0.3, 0.5],
    "sCMOS": [0.9, 0.5, 0.9, 0.5, 0.5, 0.7, 0.7, 0.7]
}

# Compute the angles for the radar chart
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Close the shape

# Create the radar chart
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# Plot each sensor's data
for sensor, values in sensor_data.items():
    values += values[:1]  # Close the shape
    ax.plot(angles, values, label=sensor, linewidth=2)
    ax.fill(angles, values, alpha=0.2)

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)

# Set title and legend
plt.title("Comparison of Imaging Sensors", fontsize=12)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.show()