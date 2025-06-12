# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:37:07 2025

@author: yifan
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# === 1. Load from Excel ===
dpath = "F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/ThetaPhaseExampleAnimal/"  
file_name= 'Prefered_theta_phase.xlsx' #Update this to your actual file path
file_path=os.path.join(dpath,file_name)
df = pd.read_excel(file_path)

# === 2. Extract data columns ===
preferred_phases = df["prefered phase"].values % 360
days = df["Day"].values

# Parse the CI string into two numbers
def parse_ci(ci_string):
    ci_string = ci_string.strip("[]")
    lower_str, upper_str = ci_string.split(",")
    return float(lower_str), float(upper_str)

ci_bounds = df["CI"].apply(parse_ci)
ci_lower = np.array([b[0] for b in ci_bounds]) % 360
ci_upper = np.array([b[1] for b in ci_bounds]) % 360

# === 3. Convert degrees to radians ===
preferred_phases_rad = np.radians(preferred_phases)
ci_lower_rad = np.radians(ci_lower)
ci_upper_rad = np.radians(ci_upper)

# === 4. Group indices by day ===
groups = {}
for i, day in enumerate(days):
    groups.setdefault(f"Day {day}", []).append(i)

# === 5. Colour map for days ===
nature_colors = ["#303030", "#FC8D62", "#A6D854", "#E78AC3", "#8DA0CB"]
day_colors = {day: nature_colors[i] for i, day in enumerate(sorted(groups.keys()))}

# === 6. Plot ===
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

for day, indices in groups.items():
    for i in indices:
        # Plot confidence interval (CI) as fan (optional)
        # theta_range = np.linspace(ci_lower_rad[i], ci_upper_rad[i], 50)
        # r = np.ones_like(theta_range)
        # ax.fill_between(theta_range, 0, r, color=day_colors[day], alpha=0.1)

        # Plot preferred phase line
        ax.plot([preferred_phases_rad[i], preferred_phases_rad[i]], [0, 1], 
                color=day_colors[day], linewidth=3, label=day if i == indices[0] else "")

# === 7. Format polar plot ===
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "0°"], fontsize=18)
ax.spines['polar'].set_linewidth(4)
ax.spines['polar'].set_alpha(0.5)
ax.grid(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), frameon=False, title="Days")
ax.set_yticklabels([])  # remove radial labels
ax.set_title("Preferred Phase by Day", fontsize=14, fontweight="bold")

plt.show()

fig_path = os.path.join(dpath,'prefered_phase_multi_Animal.png')
fig.savefig(fig_path, transparent=True)

#%%


# === 1. Load from Excel ===
dpath = "F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhaseMultipleAnimals/"  
file_name= 'Prefered_gamma_phase.xlsx' #Update this to your actual file path
file_path=os.path.join(dpath,file_name)
df = pd.read_excel(file_path)

# === 2. Extract data columns ===
preferred_phases = df["prefered phase"].values % 360
days = df["Animal"].values

# Parse the CI string into two numbers
def parse_ci(ci_string):
    ci_string = ci_string.strip("[]")
    lower_str, upper_str = ci_string.split(",")
    return float(lower_str), float(upper_str)

ci_bounds = df["CI"].apply(parse_ci)
ci_lower = np.array([b[0] for b in ci_bounds]) % 360
ci_upper = np.array([b[1] for b in ci_bounds]) % 360

# === 3. Convert degrees to radians ===
preferred_phases_rad = np.radians(preferred_phases)
ci_lower_rad = np.radians(ci_lower)
ci_upper_rad = np.radians(ci_upper)

# === 4. Group indices by day ===
groups = {}
for i, day in enumerate(days):
    groups.setdefault(f"Animal {day}", []).append(i)

# === 5. Colour map for days ===
nature_colors = ["#303030", "#FC8D62", "#A6D854", "#E78AC3", "#8DA0CB"]
day_colors = {day: nature_colors[i] for i, day in enumerate(sorted(groups.keys()))}

# === 6. Plot ===
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

for day, indices in groups.items():
    for i in indices:
        # Plot confidence interval (CI) as fan (optional)
        theta_range = np.linspace(ci_lower_rad[i], ci_upper_rad[i], 50)
        r = np.ones_like(theta_range)
        ax.fill_between(theta_range, 0, r, color=day_colors[day], alpha=0.1)

        # Plot preferred phase line
        ax.plot([preferred_phases_rad[i], preferred_phases_rad[i]], [0, 1], 
                color=day_colors[day], linewidth=3, label=day if i == indices[0] else "")

# === 7. Format polar plot ===
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "0°"], fontsize=18)
ax.spines['polar'].set_linewidth(4)
ax.spines['polar'].set_alpha(0.5)
ax.grid(False)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), frameon=False, title="Days")
ax.set_yticklabels([])  # remove radial labels
ax.set_title("Preferred Phase by Animal", fontsize=14, fontweight="bold")

plt.show()

fig_path = os.path.join(dpath,'prefered_phase_multi_Animal.png')
fig.savefig(fig_path, transparent=True)