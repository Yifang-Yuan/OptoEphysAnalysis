# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:35:15 2025

@author: yifan
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Data from table
animals = np.array([1, 2, 3, 4, 5])
peak_corr = np.array([0.364863826, 0.192974722, 0.101697242, 0.113687694, 0.111070327])
ci_lower = np.array([0.317571893, 0.124756018, 0.049131131, 0.072511719, 0.072768789])
ci_upper = np.array([0.412155759, 0.261193426, 0.154263352, 0.154863669, 0.149371864])

# Compute error bars
lower_errors = peak_corr - ci_lower
upper_errors = ci_upper - peak_corr
errors = [lower_errors, upper_errors]

# Create figure
fig, ax = plt.subplots(figsize=(5.5, 4))

ax.errorbar(animals, peak_corr, yerr=errors, fmt='o', color='#377eb8', 
            capsize=4, capthick=1.5, markersize=10, markeredgewidth=1.2, 
            markerfacecolor='#377eb8',  elinewidth=2, alpha=0.9)

# Aesthetics
ax.set_xticks(animals)
ax.set_xticklabels([f'Animal {i}' for i in animals], fontsize=12, fontname='Arial')

# Format y-axis tick labels to 3 decimal places
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f'{ytick:.2f}' for ytick in ax.get_yticks()], fontsize=12, fontname='Arial')

ax.set_xlabel("Animals", fontsize=14, fontname='Arial')
ax.set_ylabel("Peak Correlation", fontsize=14, fontname='Arial')
#ax.set_title("LFP-Optical Cross-Correlation", fontsize=14, fontname='Arial')

# Remove top and right spines for a clean look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)


fig_path = os.path.join('C:/Users/yifan/OneDrive - University of Edinburgh/1_SPAD/2_Writing/1_SPADphotometry/Data_Figure3/Ripple_MultipleMiceCorrs','Ripple_corr_multi_Animal.png')
# Save the figure
os.makedirs(os.path.dirname(fig_path), exist_ok=True)  # Ensure directory exists
fig.savefig(fig_path, transparent=True, dpi=300)
