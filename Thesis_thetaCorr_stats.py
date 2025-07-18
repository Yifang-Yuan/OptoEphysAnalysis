# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:04:40 2024

@author: Yifang
"""
import matplotlib.pyplot as plt
import numpy as np

groups = ['GECI-Pyr', 'GECI-PVIN', 'GEVI-Pyr', 'iGluSnFR', 'mCherry']
means = [0.475, 0.255, 0.444,  0.299, 0.067]
ci_lows = [0.453, 0.226, 0.418,  0.262, 0.038]
ci_highs = [0.498, 0.284, 0.471,  0.335, 0.095]

# Compute CI error
lower_err = np.array(means) - np.array(ci_lows)
upper_err = np.array(ci_highs) - np.array(means)
errors = [lower_err, upper_err]

# Compute CI error
lower_err = np.array(means) - np.array(ci_lows)
upper_err = np.array(ci_highs) - np.array(means)
errors = [lower_err, upper_err]

plt.figure(figsize=(5, 4))

# Plot mean ± CI as points with error bars
plt.errorbar(
    x=groups,
    y=means,
    yerr=errors,
    fmt='o',
    capsize=4,
    color='black',
    ecolor='gray',
    elinewidth=1.5,
    markerfacecolor='black',
    markersize=6
)

# Aesthetic formatting
plt.xticks(fontsize=12, rotation=30)
plt.yticks(fontsize=12)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Max correlation', fontsize=14)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.tick_params(axis='both', which='both', direction='out', length=4, width=0.8)

plt.tight_layout()
plt.show()

#%%
import pandas as pd
# List of Excel file paths and group labels
file_paths = [
    'C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/0_ForThesis/0YifangThesis/ThetaMaxCorr_CI_1732333_pyramidal_G8f_Atlas.xlsx',
    'C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/0_ForThesis/0YifangThesis/ThetaMaxCorr_CI_1765010_PVGCaMP8f_Atlas.xlsx',
    'C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/0_ForThesis/0YifangThesis/ThetaMaxCorr_CI_1765508_Jedi2p_Atlas.xlsx',
    'C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/0_ForThesis/0YifangThesis/ThetaMaxCorr_CI_1765507_iGlu_Atlas.xlsx',
    'C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/0_ForThesis/0YifangThesis/ThetaMaxCorr_CI_1825507_mCherry.xlsx'
]
group_labels = ['GECI-PYR', 'GECI-PVIN', 'iGluSnFR', 'GEVI-PYR', 'mCherry']
colours = ['black', 'orange', 'green', 'blue', 'red']
markers = ['o', 's', '^', 'D', 'x']  # circle, square, triangle, diamond, x

plt.figure(figsize=(6, 4))

# Plot each group
for i, (file, label, color, marker) in enumerate(zip(file_paths, group_labels, colours, markers)):
    df = pd.read_excel(file)
    
    x = df['Trial length']
    y = df['Max_Corr']
    yerr = [y - df['CI_low'], df['CI_high'] - y]

    plt.errorbar(
        x, y, yerr=yerr,
        fmt=marker + '-',  # line with marker
        capsize=4,
        label=label,
        color=color,
        alpha=0.6,
        elinewidth=1.5,
        markersize=8,
        markeredgewidth=1.2
    )
# Custom x-ticks
trial_lengths = [30, 60, 90, 120]
plt.xticks(trial_lengths, labels=[str(t) for t in trial_lengths], fontsize=12)
# Aesthetics
plt.xlabel('Trial length (s)', fontsize=14)
plt.ylabel('Max correlation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.tick_params(axis='both', which='both', direction='out', length=4, width=0.8)

# Legend outside top-right corner
plt.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    fontsize=10,
    frameon=False
)

plt.tight_layout()
plt.show()