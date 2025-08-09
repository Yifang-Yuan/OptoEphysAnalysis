# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 21:49:36 2025

@author: yifan
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# Data
# ripple_freqs = np.array([
#     1.4792, 1.5778, 1.4767, 1.5358,
#     1.9165, 1.5626, 1.3647, 1.2762,
#     1.4440, 1.5204, 1.0289, 1.7811
# ])

# mean_freq = np.mean(ripple_freqs)
# sem_freq = sem(ripple_freqs)

# # Plot style
# sns.set_context("notebook", font_scale=1.4)
# sns.set_style("white")

# fig, ax = plt.subplots(figsize=(5, 5))

# # Plot individual data points (jittered)
# x_jitter = 0.08 * (np.random.rand(len(ripple_freqs)) - 0.5)
# ax.scatter(np.zeros_like(ripple_freqs) + x_jitter, ripple_freqs,
#            color='black', s=30, zorder=10, label='Individual trials')

# # Mean line
# ax.hlines(mean_freq, -0.2, 0.2, color='tomato', linewidth=3, label='Mean')

# # SEM lines
# ax.hlines([mean_freq - sem_freq, mean_freq + sem_freq],
#           -0.1, 0.1, color='tomato', linewidth=1, label='Mean ± SEM')

# # Axis settings
# ax.set_ylabel('Ripple Frequency (Hz)')
# ax.set_xticks([])
# ax.set_xlim(-0.4, 0.4)
# ax.set_ylim(0.8, 2.0)

# # Keep only left and bottom spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# # Legend outside the plot
# ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# plt.tight_layout()
# # Optional save
# # plt.savefig("ripple_frequency_final_clean.pdf", dpi=300, bbox_inches='tight')
# plt.show()

#%%
'REM ratio'
REM_ratio = np.array([
    0.1424, 0.1207, 0.1347, 0.1428,
    0.1407, 0.1720, 0.1147, 0.1243,
    0.1435, 0.1205, 0.1208, 0.1175,
    0.1814, 0.1231, 0.1334
])

mean_freq = np.mean(REM_ratio)
sem_freq = sem(REM_ratio)

# Plot style
sns.set_context("notebook", font_scale=1.4)
sns.set_style("white")

fig, ax = plt.subplots(figsize=(5, 5))

# Plot individual data points (jittered)
x_jitter = 0.08 * (np.random.rand(len(REM_ratio)) - 0.5)
ax.scatter(np.zeros_like(REM_ratio) + x_jitter, REM_ratio,
           color='black', s=30, zorder=10, label='Individual trials')

# Mean line
ax.hlines(mean_freq, -0.2, 0.2, color='blue', linewidth=3, label='Mean')

# SEM lines
ax.hlines([mean_freq - sem_freq, mean_freq + sem_freq],
          -0.1, 0.1, color='blue', linewidth=1, label='Mean ± SEM')

# Axis settings
ax.set_ylabel('REM/Total-Sleep Ratio')
ax.set_xticks([])
# ax.set_xlim(-0.4, 0.4)
# ax.set_ylim(0.8, 2.0)

# Keep only left and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Legend outside the plot
ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
# Optional save
# plt.savefig("ripple_frequency_final_clean.pdf", dpi=300, bbox_inches='tight')
plt.show()