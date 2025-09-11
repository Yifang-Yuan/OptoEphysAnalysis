# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Fri Sep  5 15:40:42 2025

@author: yifan
"""
"""
Cheeseboard sync + tracking visualiser

- Shows (doesn't save) the cam sync plot with the first NaN/NaN marked.
- Computes and prints FPS (median & mean) from Timestamp deltas.
- Draws arena landmarks (start box, bridge, cheeseboard circle, reward point).
- Plots head trajectory from DLC filtered CSV (3-row header format).
- Finds first approach: head in reward circle (r=5 px) while bottom is out.

Edit the file/folder paths and the landmark coordinates in main().
"""
import os

import math

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Cam sync utilities ----------

def read_cam_sync(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"Timestamp", "Value.X", "Value.Y"}.issubset(df.columns):
        raise ValueError("Cam sync CSV needs columns: Timestamp, Value.X, Value.Y")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def find_first_nan_both(df: pd.DataFrame) -> int:
    nan_both = df[["Value.X", "Value.Y"]].isna().all(axis=1).to_numpy()
    idx = np.flatnonzero(nan_both)
    if idx.size == 0:
        raise ValueError("No row where both Value.X and Value.Y are NaN.")
    return int(idx[0])


def estimate_fps_from_timestamps(ts: pd.Series) -> Tuple[float, float]:
    dt = ts.diff().dt.total_seconds()
    fps_median = 1.0 / dt[1:].median()
    fps_mean = 1.0 / dt[1:].mean()
    return float(fps_median), float(fps_mean)


def show_cam_sync_plot(df: pd.DataFrame, start_idx: int, window_before: int = 200, window_after: int = 50) -> None:
    lo = max(0, start_idx - window_before)
    hi = min(len(df) - 1, start_idx + window_after)

    plt.figure(figsize=(10, 4))
    x = np.arange(lo, hi)
    plt.plot(x, df["Value.X"].iloc[lo:hi].to_numpy(), label="Value.X")
    plt.plot(x, df["Value.Y"].iloc[lo:hi].to_numpy(), label="Value.Y")
    plt.axvline(start_idx, linestyle="--", label=f"Start sync (idx={start_idx})")
    plt.title("Cam sync around start pulse (first NaN, NaN)")
    plt.xlabel("Frame index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- DLC tracking utilities ----------

def read_dlc_filtered_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads DLC filtered CSV with 3 header rows: [scorer, bodyparts, coords].
    Returns a DataFrame with MultiIndex columns.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    return df


def get_top_scorer_column(df: pd.DataFrame) -> str:
    # The first level contains 'scorer' and one long scorer name; we want the latter
    lvl0 = list(df.columns.get_level_values(0).unique())
    if "scorer" in lvl0 and len(lvl0) >= 2:
        return [x for x in lvl0 if x != "scorer"][0]
    # Fallback: just take the first non-'scorer' name
    return lvl0[0]


def extract_xy(df: pd.DataFrame, part: str, scorer0: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if scorer0 is None:
        scorer0 = get_top_scorer_column(df)
    x = df[(scorer0, part, "x")].to_numpy(dtype=float)
    y = df[(scorer0, part, "y")].to_numpy(dtype=float)
    return x, y


# ---------- Geometry helpers ----------

def polygon(points: List[Tuple[float, float]], close: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(points, dtype=float)
    if close:
        arr = np.vstack([arr, arr[0]])
    return arr[:, 0], arr[:, 1]


def mean_radius_from_center(center: Tuple[float, float], points: List[Tuple[float, float]]) -> float:
    cx, cy = center
    rs = [math.hypot(px - cx, py - cy) for px, py in points]
    return float(np.mean(rs))


# ---------- Analysis: first approach ----------

def first_approach_frame(
    head_xy: Tuple[np.ndarray, np.ndarray],
    bottom_xy: Tuple[np.ndarray, np.ndarray],
    target: Tuple[float, float],
    rad: float = 5.0,
) -> Optional[int]:
    hx, hy = head_xy
    bx, by = bottom_xy
    cx, cy = target

    d_head = np.hypot(hx - cx, hy - cy)
    d_bottom = np.hypot(bx - cx, by - cy)

    inside_head = d_head <= rad
    outside_bottom = d_bottom > rad

    hits = np.flatnonzero(inside_head & outside_bottom)
    if hits.size == 0:
        return None
    return int(hits[0])


# ---------- Plot arena + trajectory ----------

def show_arena_and_trajectory(
    head_xy: Tuple[np.ndarray, np.ndarray],
    start_box_pts: List[Tuple[int, int]],
    bridge_pts: List[Tuple[int, int]],
    cheeseboard_center: Tuple[int, int],
    cheeseboard_ends: List[Tuple[int, int]],
    reward_pt: Tuple[int, int],
    reward_circle_rad: float = 5.0,
) -> None:

    # Cheeseboard circle radius from centre and four “ends”
    R = mean_radius_from_center(cheeseboard_center, cheeseboard_ends)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Start box (black)
    sx, sy = polygon(start_box_pts)
    ax.plot(sx, sy, color="black", linewidth=2, label="Start box")

    # Bridge (grey)
    bx, by = polygon(bridge_pts)
    ax.plot(bx, by, color="grey", linewidth=2, label="Bridge")

    # Cheeseboard (circle)
    theta = np.linspace(0, 2 * np.pi, 400)
    cx, cy = cheeseboard_center
    ax.plot(cx + R * np.cos(theta), cy + R * np.sin(theta), linewidth=2, label="Cheeseboard")

    # Reward point + dashed 10 px diameter circle (r=5 px)
    ax.scatter([reward_pt[0]], [reward_pt[1]], marker="x", s=60, label="Reward")
    ax.plot(
        reward_pt[0] + reward_circle_rad * np.cos(theta),
        reward_pt[1] + reward_circle_rad * np.sin(theta),
        linestyle="--",
        label="Reward zone",
    )

    # Trajectory (head)
    hx, hy = head_xy
    ax.plot(hx, hy, linewidth=1, alpha=0.8, label="Head trajectory")

    # Make it look like image coordinates
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Arena landmarks and animal trajectory")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


# ---------- Main: edit paths & landmarks here ----------

def main():
    # --- Paths (EDIT THESE) ---
    cam_sync_csv = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour\Cam_sync_1.csv"
    dlc_csv = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour\AnimalTracking_1DLC_HrnetW32_CBtestSep27shuffle1_detector_200_snapshot_200_filtered.csv"

    # --- Landmarks (from your message) ---
    # Start box vertices (assumed that "(0.831)" meant (0, 831))
    start_box = [(41, 215), (41, 297), (0,210), (0, 297)]
    # Bridge rectangle vertices
    bridge = [(44, 237), (44, 275), (102, 237), (102, 275)]
    # Cheeseboard centre and “four ends”
    cheeseboard_center = (310, 231)
    cheeseboard_ends = [(251, 60), (516,215), (289, 23), (324, 435)]
    # Reward well
    reward_pt = (409, 112)
    reward_zone_radius = 10.0  # 10 px diameter

    # --- 1) Cam sync: start index + FPS and plot (show only) ---
    cam = read_cam_sync(cam_sync_csv)
    start_idx = find_first_nan_both(cam)
    fps_med, fps_mean = estimate_fps_from_timestamps(cam["Timestamp"])

    print("=== Cam sync ===")
    print(f"Start sync frame index: {start_idx} (0-based) | {start_idx+1} (1-based)")
    print(f"Start Timestamp: {cam['Timestamp'].iloc[start_idx] if 0 <= start_idx < len(cam) else 'N/A'}")
    print(f"FPS (median): {fps_med:.4f} | FPS (mean): {fps_mean:.4f}")
    print("================")

    show_cam_sync_plot(cam, start_idx)

    # --- 2) DLC: read and plot arena + trajectory ---
    dlc = read_dlc_filtered_csv(dlc_csv)
    scorer0 = get_top_scorer_column(dlc)

    head_xy = extract_xy(dlc, "head", scorer0)
    bottom_xy = extract_xy(dlc, "bottom", scorer0)

    show_arena_and_trajectory(
        head_xy=head_xy,
        start_box_pts=start_box,
        bridge_pts=bridge,
        cheeseboard_center=cheeseboard_center,
        cheeseboard_ends=cheeseboard_ends,
        reward_pt=reward_pt,
        reward_circle_rad=reward_zone_radius,
    )

    # --- 3) First approach detection ---
    idx_approach = first_approach_frame(head_xy, bottom_xy, reward_pt, rad=reward_zone_radius)
    if idx_approach is None:
        print("No approach event found (head in reward zone while bottom out) in this file.")
    else:
        # Optional timestamp lookup from cam sync if lengths roughly match
        ts = None
        if idx_approach < len(cam):
            ts = cam["Timestamp"].iloc[idx_approach]
        print("=== First approach ===")
        print(f"Frame index: {idx_approach} (0-based) | {idx_approach+1} (1-based)")
        if ts is not None and pd.notna(ts):
            print(f"Approx. time: {ts}")
        else:
            print("Approx. time: (no timestamp available at that index)")
        print("======================")


if __name__ == "__main__":
    main()
