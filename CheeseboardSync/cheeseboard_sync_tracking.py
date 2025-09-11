#!/usr/bin/env python3
"""
Cheeseboard sync + tracking visualiser (folded version)

- Shows (doesn't save) the cam sync plot with the first NaN/NaN marked.
- Computes and prints FPS (median & mean) from Timestamp deltas.
- Draws arena landmarks (rectangles for start box & bridge, circle for cheeseboard, reward point).
- Plots head trajectory from DLC filtered CSV (3-row header format).
- Filters out head points that are outside BOTH the cheeseboard and the bridge (likely bad tracking).
- Colours the trajectory by time (frame index) and shows a colour bar.
- Finds first approach: head in reward circle while bottom is out.

Edit the file/folder paths and the landmark coordinates in main().
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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
    return lvl0[0]


def extract_xy(df: pd.DataFrame, part: str, scorer0: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    if scorer0 is None:
        scorer0 = get_top_scorer_column(df)
    x = df[(scorer0, part, "x")].to_numpy(dtype=float)
    y = df[(scorer0, part, "y")].to_numpy(dtype=float)
    return x, y


# ---------- Geometry helpers ----------

def rect_from_four_points(pts: List[Tuple[float, float]]):
    """Return the axis-aligned rectangle defined by the min/max of the four points,
    along with the closed outline for plotting."""
    arr = np.array(pts, dtype=float)
    x0, y0 = arr[:, 0].min(), arr[:, 1].min()
    x1, y1 = arr[:, 0].max(), arr[:, 1].max()
    xs = np.array([x0, x1, x1, x0, x0])
    ys = np.array([y0, y0, y1, y1, y0])
    return xs, ys, (x0, y0, x1, y1)


def inside_rect(x: np.ndarray, y: np.ndarray, rect_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = rect_bounds
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def inside_circle(x: np.ndarray, y: np.ndarray, center: Tuple[float, float], r: float) -> np.ndarray:
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def mean_radius_from_center(center: Tuple[float, float], points: List[Tuple[float, float]]) -> float:
    cx, cy = center
    rs = [math.hypot(px - cx, py - cy) for px, py in points]
    return float(np.mean(rs))


# ---------- Analysis: first approach ----------

def first_approach_frame(
    head_xy: Tuple[np.ndarray, np.ndarray],
    bottom_xy: Tuple[np.ndarray, np.ndarray],
    target: Tuple[float, float],
    rad: float,
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
    reward_circle_rad: float,
) -> None:

    # Rect outlines and bounds
    sx, sy, sbounds = rect_from_four_points(start_box_pts)
    gx, gy, gbounds = rect_from_four_points(bridge_pts)

    # Cheeseboard circle radius from centre and four “ends”
    R = mean_radius_from_center(cheeseboard_center, cheeseboard_ends)

    # Filtering: keep head points that are inside the cheeseboard OR inside the bridge
    hx, hy = head_xy
    mask_inside_board = inside_circle(hx, hy, cheeseboard_center, R)
    mask_inside_bridge = inside_rect(hx, hy, gbounds)
    mask_keep = mask_inside_board | mask_inside_bridge

    hx_f = hx.copy(); hy_f = hy.copy()
    hx_f[~mask_keep] = np.nan
    hy_f[~mask_keep] = np.nan

    # Build time-coloured line segments from filtered head positions
    points = np.column_stack((hx_f, hy_f))
    valid_seg = np.isfinite(points[:-1]).all(axis=1) & np.isfinite(points[1:]).all(axis=1)
    segs = np.stack([points[:-1][valid_seg], points[1:][valid_seg]], axis=1)
    t = np.arange(len(hx_f) - 1)[valid_seg]  # colour by time index

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Rectangles (edges only)
    ax.plot(sx, sy, linewidth=2, label="Start box")
    ax.plot(gx, gy, linewidth=2, label="Bridge")

    # Cheeseboard (circle)
    theta = np.linspace(0, 2 * np.pi, 400)
    cx, cy = cheeseboard_center
    ax.plot(cx + R * np.cos(theta), cy + R * np.sin(theta), linewidth=2, label="Cheeseboard")

    # Reward + dashed circle
    ax.scatter([reward_pt[0]], [reward_pt[1]], marker="x", s=60, label="Reward")
    ax.plot(
        reward_pt[0] + reward_circle_rad * np.cos(theta),
        reward_pt[1] + reward_circle_rad * np.sin(theta),
        linestyle="--",
        label=f"Reward zone (r={reward_circle_rad:g} px)",
    )

    # Time-coloured trajectory
    lc = LineCollection(segs, linewidths=3)
    lc.set_array(t)
    ax.add_collection(lc)
    # make colour bar same height as axes
    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Frame (time index)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Aesthetics
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)", fontsize=14)
    ax.set_ylabel("y (px)", fontsize=14)
    ax.set_title("Arena and time-coloured head trajectory", fontsize=16)
    ax.tick_params(labelsize=12)

    # Legend bottom left, outside plot
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.05), fontsize=12)

    plt.tight_layout()
    plt.show()


# ---------- Main: edit paths & landmarks here ----------

def main():
    # --- Paths (EDIT THESE) ---
    cam_sync_csv = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour\Cam_sync_0.csv"
    dlc_csv      = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour\AnimalTracking_0DLC_HrnetW32_CBtestSep27shuffle1_detector_200_snapshot_200_filtered.csv"

    # --- Landmarks (YOUR UPDATED VALUES) ---
    start_box = [(39, 207), (39, 289), (0, 207), (0, 289)]
    bridge = [(40, 237), (40, 273), (101, 231), (101, 269)]
    cheeseboard_center = (306, 230)
    cheeseboard_ends = [(99, 246), (515, 214), (291, 26), (232, 436)]
    reward_pt = (410, 111)
    reward_zone_radius = 12.0  # px radius

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

    # --- 2) DLC: read and plot arena + trajectory (filtered) ---
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

    # --- 3) First approach detection: head in reward zone while bottom is out ---
    idx_approach = first_approach_frame(head_xy, bottom_xy, reward_pt, rad=reward_zone_radius)
    if idx_approach is None:
        print("No approach event found (head in reward zone while bottom out).")
    else:
        ts = cam["Timestamp"].iloc[idx_approach] if idx_approach < len(cam) else None
        print("=== First approach ===")
        print(f"Frame index: {idx_approach} (0-based) | {idx_approach+1} (1-based)")
        if ts is not None and pd.notna(ts):
            print(f"Approx. time: {ts}")
        else:
            print("Approx. time: (no timestamp available at that index)")
        print("======================")


if __name__ == "__main__":
    main()
