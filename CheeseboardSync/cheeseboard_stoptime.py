# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:54:28 2025

@author: yifan
"""
from __future__ import annotations
#!/usr/bin/env python3
"""
Cheeseboard reward-zone dwell time across animals and days.

Dwell time definition:
  From the first frame at/after the 'approach' where head speed < 3 cm/s
  (and head is still inside the reward zone) to the first frame after approach
  where BOTH head and neck are outside the reward zone. If neck isn't available,
  'leave' falls back to head-outside.

Inputs (per trial):
  - Cam_sync_<N>.csv        : has 'Timestamp', 'Value.X', 'Value.Y'
  - AnimalTracking_<N>DLC_..._filtered.csv : DLC results (multiindex columns)

Outputs:
  - CSV with per-trial dwell time and bookkeeping fields
  - Per-animal per-day scatter + mean±SEM plots
  - Across-animals per-day mean±SEM plot
"""

import os, re, glob, math
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- file discovery ----------------
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
    idxs = sorted(set(cams.keys()) & set(dlcs.keys()))
    return [(i, cams[i], dlcs[i]) for i in idxs]

# ---------------- I/O helpers ----------------
def read_cam_sync(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"Timestamp", "Value.X", "Value.Y"}.issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def estimate_fps(ts: pd.Series) -> Tuple[float, float]:
    dt = ts.diff().dt.total_seconds()
    return float(1.0/dt[1:].median()), float(1.0/dt[1:].mean())

def read_dlc_filtered(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=[0,1,2])

def get_top_scorer(df: pd.DataFrame) -> str:
    lvl0 = list(df.columns.get_level_values(0).unique())
    if "scorer" in lvl0 and len(lvl0) >= 2:
        return [x for x in lvl0 if x != "scorer"][0]
    return lvl0[0]

def xy_of(df: pd.DataFrame, part: str, scorer: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray]:
    if scorer is None:
        scorer = get_top_scorer(df)
    x = df[(scorer, part, "x")].to_numpy(float)
    y = df[(scorer, part, "y")].to_numpy(float)
    return x, y

# ---------------- geometry & kinematics ----------------
def inside_circle(x: np.ndarray, y: np.ndarray, center: Tuple[float,float], r: float) -> np.ndarray:
    cx, cy = center
    return (x - cx)**2 + (y - cy)**2 <= r**2

def mean_radius_from_center(center: Tuple[float,float], points: List[Tuple[float,float]]) -> float:
    cx, cy = center
    return float(np.mean([math.hypot(px-cx, py-cy) for px,py in points]))

def compute_px_to_cm_scale(cheeseboard_center, cheeseboard_ends, real_diameter_cm=100.0) -> float:
    """Return cm per pixel using cheeseboard diameter (100 cm by spec)."""
    cx, cy = cheeseboard_center
    radii = [np.hypot(x - cx, y - cy) for x, y in cheeseboard_ends]
    diameter_px = 2.0 * np.mean(radii)
    return real_diameter_cm / diameter_px if diameter_px > 0 else 1.0

def head_speed_cm_s(cam_ts: pd.Series, head_xy: Tuple[np.ndarray, np.ndarray], px_to_cm: float) -> np.ndarray:
    """Per-frame head speed in cm/s, aligned to DLC frames (same length)."""
    t = cam_ts.view(np.int64) / 1e9  # seconds (numpy float)
    t = t - t[0]
    hx, hy = head_xy
    n = len(hx)
    # guard equal lengths
    n = min(n, len(t))
    t = t[:n]; hx = hx[:n]; hy = hy[:n]
    dt = np.diff(t)
    dxy = np.hypot(np.diff(hx), np.diff(hy))
    with np.errstate(divide='ignore', invalid='ignore'):
        v_px = dxy / dt
    v_cm = v_px * px_to_cm
    # align to frames: set v[0] = v[1] for convenience
    v = np.empty(n, float)
    v[1:] = v_cm
    v[0] = v[1] if n > 1 else np.nan
    return v

# ---------------- event indices ----------------
def first_reward_approach_idx(head_xy, bottom_xy, reward_pt, reward_radius) -> Optional[int]:
    """First frame head enters reward zone while bottom still outside (approach)."""
    hx, hy = head_xy
    bx, by = bottom_xy
    in_head = inside_circle(hx, hy, reward_pt, reward_radius)
    in_bottom = inside_circle(bx, by, reward_pt, reward_radius)
    hits = np.flatnonzero(in_head & (~in_bottom))
    return int(hits[0]) if hits.size else None

def first_leave_idx_after(head_xy, neck_xy, reward_pt, reward_radius, start_idx: int) -> Optional[int]:
    """First frame >= start_idx where both head & neck are outside the reward zone.
       If neck is None, fall back to head-outside."""
    hx, hy = head_xy
    in_head = inside_circle(hx, hy, reward_pt, reward_radius)
    if neck_xy is None:
        out_mask = ~in_head
    else:
        nx, ny = neck_xy
        in_neck = inside_circle(nx, ny, reward_pt, reward_radius)
        out_mask = (~in_head) & (~in_neck)
    idxs = np.flatnonzero(out_mask & (np.arange(len(hx)) >= int(start_idx)))
    return int(idxs[0]) if idxs.size else None

def first_immobile_after(head_xy, cam_ts: pd.Series, px_to_cm: float,
                         start_idx: int, reward_pt, reward_radius,
                         speed_thresh_cm_s: float = 3.0) -> Optional[int]:
    """First index >= start_idx with head in reward zone & speed<thresh."""
    v = head_speed_cm_s(cam_ts, head_xy, px_to_cm)
    hx, hy = head_xy
    in_head = inside_circle(hx, hy, reward_pt, reward_radius)
    n = len(v)
    i0 = max(0, int(start_idx))
    for i in range(i0, n):
        if in_head[i] and np.isfinite(v[i]) and (v[i] < speed_thresh_cm_s):
            return i
    return None

# ---------------- per-trial processing ----------------
def process_trial_dwell(cam_csv: str, dlc_csv: str, landmarks: Dict,
                        speed_thresh_cm_s: float = 3.0) -> Dict:
    cam = read_cam_sync(cam_csv)
    fps_med, fps_mean = estimate_fps(cam["Timestamp"])
    dlc = read_dlc_filtered(dlc_csv)
    scorer = get_top_scorer(dlc)

    head_xy   = xy_of(dlc, "head", scorer)
    bottom_xy = xy_of(dlc, "bottom", scorer)
    neck_xy   = None
    try:
        neck_xy = xy_of(dlc, "neck", scorer)
    except Exception:
        neck_xy = None  # optional

    # px→cm scale from cheeseboard geometry (diameter = 100 cm by spec)
    px_to_cm = compute_px_to_cm_scale(landmarks["cheeseboard_center"],
                                      landmarks["cheeseboard_ends"],
                                      real_diameter_cm=100.0)

    # indices (frames) for approach, start (immobile), and leave
    appr_idx = first_reward_approach_idx(head_xy, bottom_xy,
                                         landmarks["reward_pt"],
                                         landmarks["reward_zone_radius"])
    if appr_idx is None:
        return {
            "cam_file": cam_csv, "dlc_file": dlc_csv,
            "approach_idx": None, "immobile_start_idx": None, "leave_idx": None,
            "approach_time_s": None, "immobile_start_time_s": None, "leave_time_s": None,
            "dwell_time_s": None, "fps_median": fps_med, "fps_mean": fps_mean
        }

    imm_start_idx = first_immobile_after(
        head_xy=head_xy,
        cam_ts=cam["Timestamp"],
        px_to_cm=px_to_cm,
        start_idx=appr_idx,
        reward_pt=landmarks["reward_pt"],
        reward_radius=landmarks["reward_zone_radius"],
        speed_thresh_cm_s=speed_thresh_cm_s
    )

    leave_idx = first_leave_idx_after(
        head_xy=head_xy,
        neck_xy=neck_xy,
        reward_pt=landmarks["reward_pt"],
        reward_radius=landmarks["reward_zone_radius"],
        start_idx=appr_idx
    )

    def idx_to_time_s(idx: Optional[int]) -> Optional[float]:
        if idx is None or idx >= len(cam): return None
        t0 = cam["Timestamp"].iloc[0]
        return float((cam["Timestamp"].iloc[idx] - t0).total_seconds())

    appr_t = idx_to_time_s(appr_idx)
    imm_t  = idx_to_time_s(imm_start_idx)
    leave_t= idx_to_time_s(leave_idx)

    dwell = None
    if (imm_t is not None) and (leave_t is not None) and (leave_t > imm_t):
        dwell = leave_t - imm_t

    return {
        "cam_file": cam_csv, "dlc_file": dlc_csv,
        "approach_idx": appr_idx, "immobile_start_idx": imm_start_idx, "leave_idx": leave_idx,
        "approach_time_s": appr_t, "immobile_start_time_s": imm_t, "leave_time_s": leave_t,
        "dwell_time_s": dwell, "fps_median": fps_med, "fps_mean": fps_mean
    }

def process_behaviour_folder_dwell(beh_folder: str, landmarks: Dict,
                                   speed_thresh_cm_s: float = 3.0) -> pd.DataFrame:
    pairs = find_pairs(beh_folder)
    rows = []
    for idx, cam_path, dlc_path in pairs:
        info = process_trial_dwell(cam_path, dlc_path, landmarks, speed_thresh_cm_s=speed_thresh_cm_s)
        info["TrialIndex"] = idx
        rows.append(info)
    return pd.DataFrame(rows)

# ---------------- summaries & plots ----------------
def mean_sem(a: np.ndarray) -> Tuple[float, float, int]:
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0: return np.nan, np.nan, 0
    m = float(np.mean(a))
    sem = float(np.std(a, ddof=1)/np.sqrt(n)) if n > 1 else np.nan
    return m, sem, n

def plot_per_animal_dwell(animal_id: str, day_vals: Dict[str, List[float]],
                          ylabel: str = "Dwell time in reward zone (s)",
                          out_png: Optional[str] = None, show: bool = True):
    days = ["Day1","Day2","Day3","Day4"]; x = np.arange(1,5)
    labels = ["Day2","Day3","Day4","Day5"]  # same shift you used before

    plt.figure(figsize=(6,5))
    for i,d in enumerate(days, start=1):
        y = np.array(day_vals.get(d, []), float); y = y[~np.isnan(y)]
        if y.size:
            jitter = (np.random.rand(y.size)-0.5)*0.12
            plt.scatter(np.full_like(y,i,dtype=float)+jitter, y, s=30, alpha=0.8)
        m,s,n = mean_sem(np.array(day_vals.get(d, []), float))
        if not np.isnan(m):
            plt.errorbar(i, m, yerr=s, fmt='o', capsize=5, elinewidth=3, markersize=7, color='k')

    plt.xticks(x, labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    short_id = animal_id.split("_")[0]
    plt.title(f"Dwell per trial by day — ID:{short_id}", fontsize=20)

    plt.grid(alpha=0.2); plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()

def plot_summary_across_animals_dwell(animal_day_means: Dict[str, Dict[str, Tuple[float,float]]],
                                      ylabel: str = "Stop time in reward zone (s)",
                                      out_png: Optional[str] = None, show: bool = True):
    days = ["Day1","Day2","Day3","Day4"]; x = np.arange(1,5)
    labels = ["Day2","Day3","Day4","Day5"]

    plt.figure(figsize=(8,6))
    for animal, dct in animal_day_means.items():
        short_id = animal.split("_")[0]
        means = [dct.get(d, (np.nan, np.nan))[0] for d in days]
        sems  = [dct.get(d, (np.nan, np.nan))[1] for d in days]
        plt.errorbar(x, means, yerr=sems, marker='o', capsize=5, linewidth=3, label=short_id)

    plt.xticks(x, labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title("Stop time by day — Four mice (mean ± SEM)", fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.2); plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()

# ---------------- batch runner (edit ROOTS/LANDMARKS as needed) ----------------
ROOTS = {
    "1881363_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363",
    "1881365_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881365",
    "1907336_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1907336",
    "1910567_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567",
}
DAYS = ["Day1","Day2","Day3","Day4"]
BEHAV_SUB = "Behaviour"
LANDMARKS = {
    "start_box": [(39, 207), (39, 289), (0, 207), (0, 289)],
    "bridge":  [(40, 237), (40, 273), (101, 231), (101, 269)],
    "cheeseboard_center": (306, 230),
    "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
    "reward_pt": (410, 111),
    "reward_zone_radius": 12.0,   # px radius of the reward circle in the video coordinate system
}

def main():
    all_rows = []
    per_animal_day_dwell: Dict[str, Dict[str, List[float]]] = {}
    for animal, root in ROOTS.items():
        print(f"\n=== {animal} ===")
        per_animal_day_dwell[animal] = {}
        for day in DAYS:
            beh = os.path.join(root, day, BEHAV_SUB)
            if not os.path.isdir(beh):
                print(f"  [skip] {beh} not found"); continue
            print(f"  Processing {day} ...")
            df_trials = process_behaviour_folder_dwell(beh, LANDMARKS, speed_thresh_cm_s=3.0)
            if df_trials.empty:
                print(f"  [warn] No pairs found in {beh}"); continue
            df_trials["AnimalID"] = animal; df_trials["Day"] = day
            all_rows.append(df_trials)
            vals = df_trials["dwell_time_s"].to_numpy(float)
            per_animal_day_dwell[animal][day] = [v if np.isfinite(v) else np.nan for v in vals]

    if len(all_rows) == 0:
        print("No data found across all animals/days."); return

    result = pd.concat(all_rows, ignore_index=True)
    out_csv = "cheeseboard_dwell_time_per_trial.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved per-trial dwell-time table: {os.path.abspath(out_csv)}")

    # Per-animal day plots
    for animal, dct in per_animal_day_dwell.items():
        out_png = f"dwell_by_day_{animal}.png"
        plot_per_animal_dwell(animal, dct, out_png=out_png, show=True)

    # Across-animal summary
    animal_day_means: Dict[str, Dict[str, Tuple[float,float]]] = {}
    for animal, dct in per_animal_day_dwell.items():
        animal_day_means[animal] = {}
        for day in DAYS:
            arr = np.array(dct.get(day, []), dtype=float)
            m, s, n = mean_sem(arr)
            if not np.isnan(m): animal_day_means[animal][day] = (m, s)
    plot_summary_across_animals_dwell(animal_day_means,
                                      out_png="dwell_summary_four_animals.png",
                                      show=True)

if __name__ == "__main__":
    main()
