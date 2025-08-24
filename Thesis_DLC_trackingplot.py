# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 19:39:06 2025

@author: yifan
"""
'Plot tracking from DeepLabCut'
'CONVERT pixel to cm'
'''
Pixel span (x, y): 1134.7 px, 769.8 px

Chosen real size (width×height): 60 × 40 cm (best aspect match)

Scale: 0.0529 cm/px (x), 0.0520 cm/px (y)
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Config ---------
csv_path = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\Day1\BehaviourDLC/AnimalVideo_4DLC_Resnet50_OpenfieldAug9shuffle1_detector_150_snapshot_100_filtered.csv"
bodypart = "shoulder"  # case-insensitive
bodypartname = "neck" 
# --------------

# Load DLC CSV with MultiIndex header
df = pd.read_csv(csv_path, header=[0, 1, 2])

# Find x and y columns under the given bodypart
x_col = None
y_col = None
for col in df.columns:
    scorer, bp, coord = col
    if bp.strip().lower() == bodypart.lower():
        if coord.strip().lower() == "x":
            x_col = col
        elif coord.strip().lower() == "y":
            y_col = col

if x_col is None or y_col is None:
    raise ValueError(f"No x/y columns found for bodypart '{bodypart}'.")

# Extract numeric data
x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
t = np.arange(len(df))

# Keep only valid rows
valid = np.isfinite(x) & np.isfinite(y)
xv, yv, tv = x[valid], y[valid], t[valid]

# Plot trajectory (colour = time in frames)
plt.figure(figsize=(6, 6))
sc = plt.scatter(xv, yv, c=tv, s=6)
cb = plt.colorbar(sc)
cb.set_label("Time (frames)")
plt.title(f"{bodypartname.capitalize()} trajectory ({len(xv)} points)")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.tight_layout()
#plt.savefig("shoulder_xy_trajectory.png", dpi=200)
plt.show()

#%%
