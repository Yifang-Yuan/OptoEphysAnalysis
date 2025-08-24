# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 23:59:12 2025

@author: yifan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------
# New input data
# -------------------------
data = {
    "mCherry": [0.047, 0.067, 0.050, 0.073, 0.087, 0.101, 0.070, 0.066],
    "Animal1 (JEDI-2P)": [0.101, 0.120, 0.154, 0.084, 0.112, 0.094, 0.083, 0.057],
    "Animal2": [0.097, 0.085, 0.083, 0.071, 0.091, 0.111, 0.112, 0.079],
    "Animal3": [0.065, 0.076, 0.065, 0.068, 0.081, 0.067, 0.065, 0.052],
    "Animal4": [0.105, 0.082, 0.069, 0.076, 0.156, 0.068, 0.091, 0.109],
    "Animal5": [0.085, 0.092, 0.141, 0.075, 0.103, 0.081, 0.086, 0.098],
}
df = pd.DataFrame(data)

# -------------------------
# Scatter with mean ± SEM for each group
# -------------------------
groups = list(df.columns)       # includes mCherry first
xpos = np.arange(1, len(groups)+1)

plt.figure(figsize=(9, 6))
means, sems = [], []

for i, col in enumerate(groups, start=1):
    y = df[col].values
    x = np.random.normal(i, 0.05, size=len(y))  # jitter
    
    # points
    if col == "mCherry":
        plt.scatter(x, y, alpha=0.8, color='gray')
    else:
        plt.scatter(x, y, alpha=0.7, color='gray')
    
    # store summary
    means.append(np.mean(y))
    sems.append(stats.sem(y))

# mean ± SEM: red for mCherry, blue for animals
for i, (m, s) in enumerate(zip(means, sems), start=1):
    col = groups[i-1]
    if col == "mCherry":
        plt.errorbar(i, m, yerr=s, fmt='o', color='red', capsize=6,
                     markersize=10, label='mCherry mean ± SEM')
    else:
        plt.errorbar(i, m, yerr=s, fmt='o', color='blue', capsize=6,
                     markersize=10, label='Animal mean ± SEM' if i == 2 else None)

plt.xticks(xpos, groups, rotation=30, fontsize=16)
plt.ylabel('Correlation Value', fontsize=16)
plt.title('mCherry vs PV Animals: Scatter with Mean ± SEM', fontsize=18)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# -------------------------
# Pairwise Welch t-tests vs mCherry
# -------------------------
mCherry_vals = df["mCherry"].values
for col in groups[1:]:
    t, p = stats.ttest_ind(df[col].values, mCherry_vals, equal_var=False)
    print(f"{col:18s}  t = {t: .3f},  p = {p: .4f}")
