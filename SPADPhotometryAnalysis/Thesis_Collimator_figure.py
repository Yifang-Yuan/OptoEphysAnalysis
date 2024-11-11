# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:17:47 2024

@author: Yifang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns

# Load the data
data = pd.DataFrame({
    'Test ID': [1, 2, 3, 4, 5, 6],
    'Collimator': [105485, 211942, 220194, 444308, 448382, 583790],
    'NoCollimation': [98884, 218026, 260467, 3471361, 5113089, 6658394],
    'percentage_loss': [-0.066754986, 0.027904929, 0.154618435, 0.87200755, 0.912307022, 0.912322701]
})

# Calculate means and standard error
collimator_mean = np.mean(data['Collimator'])
no_collimation_mean = np.mean(data['NoCollimation'])
collimator_sem = sem(data['Collimator'])
no_collimation_sem = sem(data['NoCollimation'])

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

# Left subplot: Box plot with horizontal line for mean
ax1.boxplot([data['Collimator'], data['NoCollimation']], positions=[1, 2], widths=0.5,
            showfliers=False, boxprops=dict(color='white'), medianprops=dict(color='white'))

# Plot the mean as a horizontal line with standard error
#ax1.errorbar(1, collimator_mean, yerr=collimator_sem, fmt='_', color='skyblue', linewidth=2, capsize=5, label='Collimator Mean ± SEM')
#ax1.errorbar(2, no_collimation_mean, yerr=no_collimation_sem, fmt='_', color='salmon', linewidth=2, capsize=5, label='No Collimation Mean ± SEM')

# Connecting lines for each test ID with larger data points
for i, row in data.iterrows():
    ax1.plot([1, 2], [row['Collimator'], row['NoCollimation']], color='gray', linestyle='--', marker='o', markersize=5)

# Adjust x-ticks and labels with larger font size
ax1.set_xticks([1, 2])
ax1.set_xticklabels(['Collimator', 'No Collimator'], fontsize=18)
ax1.set_ylabel('Total photon Count', fontsize=18)
ax1.tick_params(axis='y', labelsize=18)
#ax1.set_yticklabels(fontsize=16)
#ax1.set_title('Photon Count Comparison with Collimation vs. No Collimation', fontsize=16)
# ax1.legend(fontsize=12)
# ax1.legend(frameon=False)

# Remove top and right frame for ax1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right subplot: Violin plot for percentage loss with more transparency
sns.violinplot(data=data, y='percentage_loss', ax=ax2, inner=None, color='lightgrey', alpha=0.3)
ax2.axhline(np.mean(data['percentage_loss']), color='purple', linestyle='--', linewidth=2, label='Mean Percentage Loss')

# Set labels and title for percentage differences with larger font size
ax2.set_xticks([])
ax2.set_ylabel('Percentage Loss', fontsize=18)
ax1.tick_params(axis='y', labelsize=18)
#ax2.set_title('Violin Plot of Percentage Loss', fontsize=16)
# Position the legend outside the plot area
#ax2.legend(fontsize=12, frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
# ax2.legend(fontsize=12)
# ax2.legend(frameon=False)# Remove top and right frame for ax2
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()