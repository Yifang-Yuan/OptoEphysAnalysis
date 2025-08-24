# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 19:39:43 2025

@author: yifan
"""

'Plot tracking from Bonsai'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
csv_path = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\Day1\Behaviour/AnimalTracking_4.csv"  # change if needed
point_size = 6
invert_y = False  # set True if your image coords have y downward
# --------------

def read_x_y(path: str):
    """
    Reads a CSV where the first column is x and the second is y.
    Works whether the file has a header row or not.
    """
    # Try reading with no header first
    df = pd.read_csv(path, header=None)
    # If the first row looks like text headers, re-read with header
    first_is_text = isinstance(df.iloc[0, 0], str) or isinstance(df.iloc[0, 1], str)
    if first_is_text:
        df = pd.read_csv(path)  # read with header row
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    else:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    return x, y

def plot_trajectory_pixels(x, y, point_size=6, invert_y=False, outfile="animal_xy_trajectory_pixels.png"):
    t = np.arange(len(x))
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv, tv = x[valid], y[valid], t[valid]

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(xv, yv, c=tv, s=point_size)
    cb = plt.colorbar(sc); cb.set_label("Time (frames)")
    plt.title(f"Animal trajectory ({len(xv)} points, pixels)")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    x, y = read_x_y(csv_path)
    plot_trajectory_pixels(x, y, point_size=point_size, invert_y=invert_y)