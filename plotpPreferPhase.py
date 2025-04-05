# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:37:07 2025

@author: yifan
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# New dataset (converted to 0-360°)
preferred_phases = np.array([-158.91, -152.2, -112.69, -121.19, -154.8]) % 360
ci_lower = np.array([-175.65, -172.53, -137.52, -147.21, -182.28]) % 360
ci_upper = np.array([-142.17, -131.86, -87.86, -95.18, -127.32]) % 360

# Convert degrees to radians
preferred_phases_rad = np.radians(preferred_phases)
ci_lower_rad = np.radians(ci_lower)
ci_upper_rad = np.radians(ci_upper)

# **High-contrast color palette**
colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#984EA3"]  # Red, Blue, Green, Orange, Purple

# Create polar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

# Plot each animal with its CI as a fan
for i in range(len(preferred_phases_rad)):
    # CI fan area
    theta_range = np.linspace(ci_lower_rad[i], ci_upper_rad[i], 50)  # Smooth sector
    r = np.ones_like(theta_range)  # Radial range from centre to outer circle
    ax.fill_between(theta_range, 0, r, color=colors[i], alpha=0.06)  # CI as a fan

    # Preferred phase as a radial line
    ax.plot([preferred_phases_rad[i], preferred_phases_rad[i]], [0, 1], color=colors[i], linewidth=3, label=f"{i+1}")

# Formatting
ax.set_theta_zero_location("E")  # 0° at right
ax.set_theta_direction(1)  # ✅ Set anticlockwise direction
ax.set_xticks(np.linspace(0, 2 * np.pi, 9))  # 0° to 360° ticks
ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "0°"], fontsize=18)

ax.spines['polar'].set_linewidth(4)
ax.spines['polar'].set_alpha(0.5)
ax.grid(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), frameon=False)
ax.set_yticklabels([])
ax.set_title("Preferred Theta Phase", fontsize=14, fontweight="bold")

# Show plot
plt.show()

fig_path = os.path.join('C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/2_Writing/1_SPADphotometry/Data_Figure3/ThetaPhaseMultipleAnimals','prefered_phase_multi_Animal.png')
fig.savefig(fig_path, transparent=True)

#%%
# Data (converted to 0-360°)
preferred_phases = np.array([-141.01, -160.15, -146.38, -152.45, -146.93, -120.9, 
                             -158.91, -100.48, -151.03, -142.79, -134.58, -138.12]) % 360
ci_lower = np.array([-158.10, -179.36, -159.70, -166.57, -163.76, -140.05, 
                     -175.65, -125.69, -174.34, -161.20, -155.15, -155.67]) % 360
ci_upper = np.array([-123.93, -138.93, -133.06, -138.32, -130.11, -101.75, 
                     -142.17, -75.26, -127.72, -124.37, -114.02, -120.57]) % 360

# Convert degrees to radians
preferred_phases_rad = np.radians(preferred_phases)
ci_lower_rad = np.radians(ci_lower)
ci_upper_rad = np.radians(ci_upper)

# Define groups (days)
groups = {
    "Day 1": [0, 1],     # Trial 1, 2
    "Day 2": [2, 3, 4],  # Trial 3, 4, 5
    "Day 3": [5, 6],     # Trial 6, 7
    "Day 4": [7, 8, 9],  # Trial 8, 9, 10
    "Day 5": [10, 11]    # Trial 11, 12
}

# Assign distinct grayscale colors for each day
nature_colors = ["#303030", "#FC8D62", "#A6D854", "#E78AC3", "#8DA0CB"] # Light to dark gray
day_colors = {day: nature_colors[i] for i, day in enumerate(groups.keys())}

# Create polar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

# Plot each group with a different grayscale shade
for day, trials in groups.items():
    for i in trials:
        # CI fan area
        theta_range = np.linspace(ci_lower_rad[i], ci_upper_rad[i], 50)
        r = np.ones_like(theta_range)
        #ax.fill_between(theta_range, 0, r, color=day_colors[day], alpha=0.05)  # CI as a fan

        # Preferred phase as a radial line
        ax.plot([preferred_phases_rad[i], preferred_phases_rad[i]], [0, 1], 
                color=day_colors[day], linewidth=3, label=day if i == trials[0] else "")

# Formatting
ax.set_theta_zero_location("E")  # 0° at right
ax.set_theta_direction(1)  # Anticlockwise
ax.set_xticks(np.linspace(0, 2 * np.pi, 9))  # 0° to 360° ticks
ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "0°"], fontsize=18)

# **Thicker circular border**
ax.spines['polar'].set_linewidth(4)
ax.spines['polar'].set_alpha(0.5)

ax.grid(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), frameon=False, title="Days")  # No frame for legend
ax.set_yticklabels([])  # Remove radial axis labels
ax.set_title("Preferred Theta Phase by Day", fontsize=14, fontweight="bold")

# Show plot
plt.show()
fig_path = os.path.join('C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/2_Writing/1_SPADphotometry/Data_Figure3/ThetaPhaseExampleAnimal','prefered_phase_multi_Animal.png')
fig.savefig(fig_path, transparent=True)
#%%
# Data (converted to 0-360°)
# When the animal was in the home cage
preferred_phases = np.array([-148.8, 157.18, -167.44, -131.32, -116.3, -145.15, 
                             -114.92, -119.47, -140.3, -110.47, -142.31, -169.74, 
                             -146.44, -128.95]) % 360

ci_lower = np.array([-168.34, 136.54, -190.25, -148.73, -137.15, -162.67, 
                     -136.79, -138.58, -158.48, -129.50, -160.33, -190.81, 
                     -167.86, -148.54]) % 360

ci_upper = np.array([-129.25, 177.81, -144.63, -113.90, -95.44, -127.63, 
                     -93.06, -100.36, -122.12, -91.44, -124.29, -148.66, 
                     -125.02, -109.35]) % 360

# Convert degrees to radians
preferred_phases_rad = np.radians(preferred_phases)
ci_lower_rad = np.radians(ci_lower)
ci_upper_rad = np.radians(ci_upper)

# Define groups (days)
groups = {
    "Day 1": [0, 1,2],     # Trial 1, 2
    "Day 2": [3,4, 5],  # Trial 3, 4, 5
    "Day 3": [6, 7],     # Trial 6, 7
    "Day 4": [8, 9, 10],  # Trial 8, 9, 10
    "Day 5": [11, 12,13]    # Trial 11, 12
}

# Assign distinct grayscale colors for each day
nature_colors = ["#303030", "#FC8D62", "#A6D854", "#E78AC3", "#8DA0CB"] # Light to dark gray
day_colors = {day: nature_colors[i] for i, day in enumerate(groups.keys())}

# Create polar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

# Plot each group with a different grayscale shade
for day, trials in groups.items():
    for i in trials:
        # CI fan area
        theta_range = np.linspace(ci_lower_rad[i], ci_upper_rad[i], 50)
        r = np.ones_like(theta_range)
        #ax.fill_between(theta_range, 0, r, color=day_colors[day], alpha=0.05)  # CI as a fan

        # Preferred phase as a radial line
        ax.plot([preferred_phases_rad[i], preferred_phases_rad[i]], [0, 1], 
                color=day_colors[day], linewidth=3, label=day if i == trials[0] else "")

# Formatting
ax.set_theta_zero_location("E")  # 0° at right
ax.set_theta_direction(1)  # Anticlockwise
ax.set_xticks(np.linspace(0, 2 * np.pi, 9))  # 0° to 360° ticks
ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "0°"], fontsize=18)

# **Thicker circular border**
ax.spines['polar'].set_linewidth(4)
ax.spines['polar'].set_alpha(0.5)

ax.grid(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), frameon=False, title="Days")  # No frame for legend
ax.set_yticklabels([])  # Remove radial axis labels
ax.set_title("Preferred Theta Phase by Day", fontsize=14, fontweight="bold")
