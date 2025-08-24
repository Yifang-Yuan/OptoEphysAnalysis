# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:37:07 2025

@author: yifan
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shift_phase_deg(deg_array):
    return (deg_array + 180) % 360  # shift peak→trough, wrap to [0–360)

def parse_ci(ci_string):
    lower_str, upper_str = ci_string.strip("[]").split(",")
    return float(lower_str), float(upper_str)

def plot_polar_phases(file_path, colour_map, group_col, title, save_path):
    df = pd.read_excel(file_path)

    # --- Parse phase and CI ---
    def parse_ci(ci_string):
        lower_str, upper_str = ci_string.strip("[]").split(",")
        return float(lower_str), float(upper_str)

    preferred_raw = df["prefered phase"].values
    ci_bounds = df["CI"].apply(parse_ci)
    ci_lower_raw = np.array([b[0] for b in ci_bounds])
    ci_upper_raw = np.array([b[1] for b in ci_bounds])

    # === 1. Shift all angles (theta peak → trough = +180°), without mod 360
    preferred_shifted = preferred_raw + 180
    ci_lower_shifted = ci_lower_raw + 180
    ci_upper_shifted = ci_upper_raw + 180

    # === 2. Convert all to radians
    preferred_rad = np.radians(preferred_shifted % 360)
    ci_lower_rad = np.radians(ci_lower_shifted)
    ci_upper_rad = np.radians(ci_upper_shifted)

    # === 3. Group by group_col (e.g. Day or Animal)
    groups = {}
    for i, val in enumerate(df[group_col].values):
        key = f"{group_col} {val}"
        groups.setdefault(key, []).append(i)

    # === 4. Define colours
    
    group_colours = {g: colour_map[i % len(colour_map)] for i, g in enumerate(sorted(groups.keys()))}

    # === 5. Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    for group, indices in groups.items():
        for i in indices:
            # --- Handle circular CI range properly ---
            lower = ci_lower_rad[i]
            upper = ci_upper_rad[i]

            # Handle CI that wraps around 2π
            if upper < lower:
                theta_range1 = np.linspace(lower, 2 * np.pi, 25)
                theta_range2 = np.linspace(0, upper, 25)
                theta_range = np.concatenate([theta_range1, theta_range2])
            else:
                theta_range = np.linspace(lower, upper, 50)

            r = np.ones_like(theta_range)
            ax.fill_between(theta_range, 0, r, color=group_colours[group], alpha=0.1)

            # --- Preferred phase line ---
            ax.plot([preferred_rad[i], preferred_rad[i]], [0, 1],
                    color=group_colours[group], linewidth=3,
                    label=group if i == indices[0] else "")

    # === 6. Format polar plot
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°", "0°"], fontsize=14)
    ax.set_yticklabels([])
    ax.spines['polar'].set_linewidth(4)
    ax.spines['polar'].set_alpha(0.5)
    ax.grid(False)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), frameon=False, title=group_col)

    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, transparent=True)

    return -1

colour_map_A = [
    
    "#ffd92f",  # Yellow
    "#66c2a5",  # Teal
    "#8da0cb",  # Blue-violet
    "#e78ac3",  # Pink
    "#a6d854",  # Lime green
    "#e5c494",  # Tan
    "#b3b3b3",   # Light grey
    "#fc8d62",  # Orange
]

colour_map_B = [
    "#1f78b4",  # Deep Blue
    "#33a02c",  # Green
    "#e31a1c",  # Red
    "#ff7f00",  # Orange
    "#6a3d9a",  # Purple
    "#a6cee3",  # Light Blue
    "#fb9a99",   # Salmon pink
    "#b15928",  # Brown
]
    

#=== Run on both datasets ===

# file_path="F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/ThetaPhaseExampleAnimal/Prefered_theta_phase_5day.xlsx"
# group_col="Day"
# title="Preferred Phase by Day"
# save_path="F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/ThetaPhaseExampleAnimal/prefered_phase_single_Animal.png"
# colour_map=colour_map_A
# plot_polar_phases(file_path, colour_map, group_col, title, save_path)


file_path="F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/ThetaPhaseMultipleAnimals/Prefered_theta_phase.xlsx"
group_col="Animal"
title="Preferred Phase by Animal"
save_path="F:/2025_ATLAS_SPAD/Figure2_Pyr_theta/ThetaPhaseMultipleAnimals/prefered_phase_multi_Animal.png"
colour_map=colour_map_B
plot_polar_phases(file_path, colour_map, group_col, title, save_path)


# file_path="F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhaseExampleAnimal/Prefered_gamma_phase_5days.xlsx"
# group_col="Day"
# title="Preferred Phase by Day"
# save_path="F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhaseExampleAnimal/prefered_phase_single_Animal.png"
# colour_map=colour_map_A
# plot_polar_phases(file_path, colour_map, group_col, title, save_path)


file_path="F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhaseMultipleAnimals/Prefered_gamma_phase.xlsx"
group_col="Animal"
title="Preferred Phase by Animal"
save_path="F:/2025_ATLAS_SPAD/Figure3_Pyr_gamma/GammaPhaseMultipleAnimals/prefered_phase_multi_Animal.png"
colour_map=colour_map_B
plot_polar_phases(file_path, colour_map, group_col, title, save_path)