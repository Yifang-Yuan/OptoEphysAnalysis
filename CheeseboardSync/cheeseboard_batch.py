#!/usr/bin/env python3
"""
Cheeseboard sync + tracking: batch processing (patched patterns)

This version fixes pairing by using two explicit regexes:

- CAM: Cam_sync_(\d+)\.csv
- DLC: AnimalTracking_(\d+)DLC_.*_filtered\.csv

It also prints a debug table of what files were found and the indices parsed.
"""

from __future__ import annotations
import os, re, math, glob
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ----------------- Utilities from previous script (simplified where possible) -----------------

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

def cam_sync_plot(df: pd.DataFrame, start_idx: int, out_png: Optional[str] = None, window_before: int = 200, window_after: int = 50, show: bool = True) -> None:
    lo = max(0, start_idx - window_before)
    hi = min(len(df) - 1, start_idx + window_after)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(lo, hi)
    ax.plot(x, df["Value.X"].iloc[lo:hi].to_numpy(), label="Value.X")
    ax.plot(x, df["Value.Y"].iloc[lo:hi].to_numpy(), label="Value.Y")
    ax.axvline(start_idx, linestyle="--", label=f"Start sync (idx={start_idx})")
    ax.set_title("Cam sync around start pulse (first NaN, NaN)")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def read_dlc_filtered_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=[0, 1, 2])

def get_top_scorer_column(df: pd.DataFrame) -> str:
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

def rect_from_four_points(pts: List[Tuple[float, float]]):
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
    return (x - cx)**2 + (y - cy)**2 <= r**2

def mean_radius_from_center(center: Tuple[float, float], points: List[Tuple[float, float]]) -> float:
    cx, cy = center
    rs = [math.hypot(px - cx, py - cy) for px, py in points]
    return float(np.mean(rs))

def first_approach_frame(head_xy, bottom_xy, target, rad: float):
    hx, hy = head_xy
    bx, by = bottom_xy
    cx, cy = target
    d_head = np.hypot(hx - cx, hy - cy)
    d_bottom = np.hypot(bx - cx, by - cy)
    inside_head = d_head <= rad
    outside_bottom = d_bottom > rad
    hits = np.flatnonzero(inside_head & outside_bottom)
    return int(hits[0]) if hits.size else None

def plot_arena_and_trajectory(
    head_xy, start_box_pts, bridge_pts, cheeseboard_center, cheeseboard_ends, reward_pt, reward_circle_rad,
    out_png: Optional[str] = None, show: bool = True,
):
    # Geometry
    sx, sy, sbounds = rect_from_four_points(start_box_pts)
    gx, gy, gbounds = rect_from_four_points(bridge_pts)
    R = mean_radius_from_center(cheeseboard_center, cheeseboard_ends)

    # Head coords + filtering
    hx, hy = head_xy
    mask_inside_board = inside_circle(hx, hy, cheeseboard_center, R)
    mask_inside_bridge = inside_rect(hx, hy, gbounds)
    mask_keep = mask_inside_board | mask_inside_bridge
    hx_f = hx.copy(); hy_f = hy.copy()
    hx_f[~mask_keep] = np.nan
    hy_f[~mask_keep] = np.nan

    # Time index for colouring
    t = np.arange(len(hx_f))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))

    # Arena
    ax.plot(sx, sy, linewidth=2.5, label="Start box")
    ax.plot(gx, gy, linewidth=2.5, label="Bridge")
    theta = np.linspace(0, 2*np.pi, 400)
    cx, cy = cheeseboard_center
    ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), linewidth=2.5, label="Cheeseboard")
    ax.scatter([reward_pt[0]], [reward_pt[1]], marker="x", s=100, label="Reward")
    ax.plot(reward_pt[0] + reward_circle_rad*np.cos(theta),
            reward_pt[1] + reward_circle_rad*np.sin(theta),
            linestyle="--", linewidth=2,
            label=f"Reward zone (r={reward_circle_rad:g} px)")

    # Trajectory as dots
    sc = ax.scatter(hx_f, hy_f, c=t, cmap="viridis", s=12, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time (frame index)", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    # Style
    ax.set_aspect("equal", adjustable="box"); ax.invert_yaxis()
    ax.set_xlabel("x (pixel)", fontsize=16); ax.set_ylabel("y (pixel)", fontsize=16)
    ax.set_title("Arena and time-coloured head trajectory", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.05), fontsize=16)

    fig.tight_layout()
    if out_png: fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

def process_pair(cam_sync_path, dlc_path, landmarks, save_plots=False, show_plots=True, out_dir=None):
    cam = read_cam_sync(cam_sync_path)
    start_idx = find_first_nan_both(cam)
    fps_med, fps_mean = estimate_fps_from_timestamps(cam["Timestamp"])
    cam_png = None
    if save_plots and out_dir:
        base = os.path.splitext(os.path.basename(cam_sync_path))[0]
        cam_png = os.path.join(out_dir, f"{base}_start.png")
    cam_sync_plot(cam, start_idx, out_png=cam_png, show=show_plots)
    dlc = read_dlc_filtered_csv(dlc_path)
    scorer0 = get_top_scorer_column(dlc)
    head_xy = extract_xy(dlc, "head", scorer0)
    bottom_xy = extract_xy(dlc, "bottom", scorer0)
    arena_png = None
    if save_plots and out_dir:
        base = os.path.splitext(os.path.basename(dlc_path))[0]
        arena_png = os.path.join(out_dir, f"{base}_arena.png")
    plot_arena_and_trajectory(
        head_xy, landmarks["start_box"], landmarks["bridge"], landmarks["cheeseboard_center"],
        landmarks["cheeseboard_ends"], landmarks["reward_pt"], landmarks["reward_zone_radius"],
        out_png=arena_png, show=show_plots
    )
    idx_approach = first_approach_frame(head_xy, bottom_xy, landmarks["reward_pt"], landmarks["reward_zone_radius"])
    ts_approach = cam["Timestamp"].iloc[idx_approach] if (idx_approach is not None and idx_approach < len(cam)) else None
    return {
        "cam_sync": cam_sync_path,
        "dlc": dlc_path,
        "start_idx_0based": start_idx,
        "start_idx_1based": start_idx + 1,
        "start_timestamp": cam["Timestamp"].iloc[start_idx] if 0 <= start_idx < len(cam) else None,
        "fps_median": fps_med,
        "fps_mean": fps_mean,
        "approach_idx_0based": idx_approach,
        "approach_idx_1based": (None if idx_approach is None else idx_approach + 1),
        "approach_timestamp": ts_approach,
        "cam_plot": cam_png,
        "arena_plot": arena_png,
    }

# ----------------- NEW: explicit patterns & pairing -----------------
CAM_IDX_RE = re.compile(r'Cam_sync_(\d+)\.csv$', re.IGNORECASE)
DLC_IDX_RE = re.compile(r'AnimalTracking_(\d+)DLC_.*_filtered\.csv$', re.IGNORECASE)

def index_cam(path: str) -> Optional[int]:
    m = CAM_IDX_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def index_dlc(path: str) -> Optional[int]:
    m = DLC_IDX_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def find_pairs(folder: str) -> List[Tuple[int, str, str]]:
    cam_files = sorted(glob.glob(os.path.join(folder, "Cam_sync_*.csv")))
    dlc_files = sorted(glob.glob(os.path.join(folder, "AnimalTracking_*filtered.csv")))

    cams = {index_cam(p): p for p in cam_files if index_cam(p) is not None}
    dlcs = {index_dlc(p): p for p in dlc_files if index_dlc(p) is not None}

    # Debug printout
    print("Found CAM files (index -> path):")
    for k in sorted(cams):
        print(f"  {k}: {cams[k]}")
    print("Found DLC files (index -> path):")
    for k in sorted(dlcs):
        print(f"  {k}: {dlcs[k]}")

    idxs = sorted(set(cams.keys()) & set(dlcs.keys()))
    print("Paired indices:", idxs)
    return [(i, cams[i], dlcs[i]) for i in idxs]

def batch_process_folder(folder: str, landmarks: Dict, save_plots=True, show_plots=False, out_dir: Optional[str]=None) -> pd.DataFrame:
    pairs = find_pairs(folder)
    if not pairs:
        raise FileNotFoundError("No matching Cam_sync_{i}.csv and AnimalTracking_{i}*filtered.csv pairs found for the explicit patterns.")
    if out_dir is None:
        out_dir = os.path.join(folder, "batch_outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for idx, cam_path, dlc_path in pairs:
        print(f"\nProcessing index {idx}:\n  CAM: {cam_path}\n  DLC: {dlc_path}")
        info = process_pair(cam_path, dlc_path, landmarks, save_plots=save_plots, show_plots=show_plots, out_dir=out_dir)
        info["index"] = idx
        rows.append(info)
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "batch_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved batch summary: {out_csv}")
    return summary

def main():
    folder = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour"
    landmarks = {
        "start_box": [(39, 207), (39, 289), (0, 207), (0, 289)],
        "bridge":  [(40, 237), (40, 273), (101, 231), (101, 269)],
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 12.0,
    }
    batch_process_folder(folder, landmarks, save_plots=True, show_plots=False, out_dir=None)

if __name__ == "__main__":
    main()
