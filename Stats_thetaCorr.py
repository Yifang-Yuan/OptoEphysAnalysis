# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 23:04:40 2024

@author: Yifang
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
groups = ['GECI-Pyr', 'GECI-PVIN', 'GEVI-Pyr', 'GEVI-PVIN', 'iGluSnFR', 'mCherry']
means = [0.164893797, 0.206634306, 0.2012971, 0.089030256, 0.147787202, 0.026471225]
CI_low = [0.124609423, 0.170298114, 0.181186543, 0.054234837, 0.110359521, 0.010682862]
CI_high = [0.205178172, 0.242970499, 0.221407658, 0.123825674, 0.185214883, 0.042259588]

# Calculate error (distance from mean to each CI bound)
errors = [np.array(means) - np.array(CI_low), np.array(CI_high) - np.array(means)]

# Plotting
plt.figure(figsize=(8, 5))
plt.errorbar(groups, means, yerr=errors, fmt='o', capsize=5, color='dodgerblue', ecolor='gray', elinewidth=2, markerfacecolor='dodgerblue', markersize=8)

# Aesthetics
plt.xlabel('Groups', fontsize=12)
plt.ylabel('Max Corr', fontsize=12)
plt.title('Mean Corr with Confidence Intervals', fontsize=14)
plt.xticks(rotation=45)
plt.tick_params(axis='both', which='both', length=0)  # Remove tick marks
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

plt.show()