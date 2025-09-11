#!/usr/bin/env python3
"""
Cheeseboard performance analysis across animals and days
(see earlier docstring in the conversation for details)
"""
from __future__ import annotations
import os, re, glob, math
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def inside_circle(x: np.ndarray, y: np.ndarray, center: Tuple[float,float], r: float) -> np.ndarray:
    cx, cy = center
    return (x - cx)**2 + (y - cy)**2 <= r**2

def mean_radius_from_center(center: Tuple[float,float], points: List[Tuple[float,float]]) -> float:
    cx, cy = center
    return float(np.mean([math.hypot(px-cx, py-cy) for px,py in points]))

def first_full_body_entry_idx(head_xy, bottom_xy, board_center, board_radius) -> Optional[int]:
    hx, hy = head_xy
    bx, by = bottom_xy
    in_head = inside_circle(hx, hy, board_center, board_radius)
    in_bottom = inside_circle(bx, by, board_center, board_radius)
    both_in = in_head & in_bottom
    if not np.any(both_in): return None
    idxs = np.flatnonzero(both_in)
    for idx in idxs:
        if idx == 0: return int(idx)
        if not both_in[idx-1]: return int(idx)
    return int(idxs[0])

def first_reward_approach_idx(head_xy, bottom_xy, reward_pt, reward_radius) -> Optional[int]:
    hx, hy = head_xy
    bx, by = bottom_xy
    in_head = inside_circle(hx, hy, reward_pt, reward_radius)
    in_bottom = inside_circle(bx, by, reward_pt, reward_radius)
    hits = np.flatnonzero(in_head & (~in_bottom))
    return int(hits[0]) if hits.size else None

def process_trial(cam_csv: str, dlc_csv: str, landmarks: Dict) -> Dict:
    cam = read_cam_sync(cam_csv)
    fps_med, fps_mean = estimate_fps(cam["Timestamp"])
    dlc = read_dlc_filtered(dlc_csv)
    scorer = get_top_scorer(dlc)
    head_xy = xy_of(dlc, "head", scorer)
    bottom_xy = xy_of(dlc, "bottom", scorer)
    R = mean_radius_from_center(landmarks["cheeseboard_center"], landmarks["cheeseboard_ends"])
    entry_idx = first_full_body_entry_idx(head_xy, bottom_xy, landmarks["cheeseboard_center"], R)
    appr_idx  = first_reward_approach_idx(head_xy, bottom_xy, landmarks["reward_pt"], landmarks["reward_zone_radius"])
    def idx_to_time_s(idx: Optional[int]) -> Optional[float]:
        if idx is None or idx >= len(cam): return None
        t0 = cam["Timestamp"].iloc[0]
        return float((cam["Timestamp"].iloc[idx] - t0).total_seconds())
    entry_t = idx_to_time_s(entry_idx)
    appr_t  = idx_to_time_s(appr_idx)
    latency = (appr_t - entry_t) if (entry_t is not None and appr_t is not None and appr_t >= entry_t) else None
    return {
        "cam_file": cam_csv, "dlc_file": dlc_csv,
        "entry_idx": entry_idx, "approach_idx": appr_idx,
        "entry_time_s": entry_t, "approach_time_s": appr_t,
        "latency_s": latency, "fps_median": fps_med, "fps_mean": fps_mean,
    }

def process_behaviour_folder(beh_folder: str, landmarks: Dict) -> pd.DataFrame:
    pairs = find_pairs(beh_folder)
    rows = []
    for idx, cam_path, dlc_path in pairs:
        info = process_trial(cam_path, dlc_path, landmarks)
        info["TrialIndex"] = idx
        rows.append(info)
    return pd.DataFrame(rows)

def mean_sem(a: np.ndarray) -> Tuple[float, float, int]:
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0: return np.nan, np.nan, 0
    m = float(np.mean(a))
    sem = float(np.std(a, ddof=1)/np.sqrt(n)) if n > 1 else np.nan
    return m, sem, n

def plot_per_animal_latencies(animal_id: str, day_latencies: Dict[str, List[float]], out_png: Optional[str] = None, show: bool = True):
    days = ["Day1","Day2","Day3","Day4"]; x = np.arange(1,5)
    labels = ["Day2","Day3","Day4","Day5"]  # shifted labels

    plt.figure(figsize=(6,5))
    for i,d in enumerate(days, start=1):
        y = np.array(day_latencies.get(d, []), float); y = y[~np.isnan(y)]
        if y.size:
            jitter = (np.random.rand(y.size)-0.5)*0.12
            plt.scatter(np.full_like(y,i,dtype=float)+jitter, y, s=30, alpha=0.8)
        m,s,n = mean_sem(np.array(day_latencies.get(d, []), float))
        if not np.isnan(m):
            plt.errorbar(i, m, yerr=s, fmt='o', capsize=5, elinewidth=3, markersize=7, color='k')

    plt.xticks(x, labels, fontsize=20)
    plt.yticks(fontsize=20)   # 🔑 bigger y tick labels
    plt.ylabel("Latency (s)", fontsize=20)

    short_id = animal_id.split("_")[0]
    plt.title(f"Latency per trial by day — ID:{short_id}", fontsize=20)

    plt.grid(alpha=0.2); plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()


def plot_summary_across_animals(animal_day_means: Dict[str, Dict[str, Tuple[float,float]]], out_png: Optional[str] = None, show: bool = True):
    days = ["Day1","Day2","Day3","Day4"]; x = np.arange(1,5)
    labels = ["Day2","Day3","Day4","Day5"]  # shifted labels

    plt.figure(figsize=(8,6))
    for animal, dct in animal_day_means.items():
        short_id = animal.split("_")[0]
        means = [dct.get(d, (np.nan, np.nan))[0] for d in days]
        sems  = [dct.get(d, (np.nan, np.nan))[1] for d in days]
        plt.errorbar(x, means, yerr=sems, marker='o', capsize=5, linewidth=3, label=short_id)

    plt.xticks(x, labels, fontsize=20)
    plt.yticks(fontsize=20)   # 🔑 bigger y tick labels
    plt.ylabel("Latency (s)", fontsize=20)
    plt.title("Latency by day-Four Mice (mean ± SEM)", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(alpha=0.2); plt.tight_layout()
    if out_png: plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()


ROOTS = {
    "1881363_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363",
    "1881365_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881365",
    "1907336_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1907336",
    "1910567_Jedi2p_CB": r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567",
}
DAYS = ["Day1","Day2","Day3","Day4"]; BEHAV_SUB = "Behaviour"
LANDMARKS = {
    "start_box": [(39, 207), (39, 289), (0, 207), (0, 289)],
    "bridge":  [(40, 237), (40, 273), (101, 231), (101, 269)],
    "cheeseboard_center": (306, 230),
    "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
    "reward_pt": (410, 111),
    "reward_zone_radius": 12.0,
}

def main():
    all_rows = []
    per_animal_day_latencies: Dict[str, Dict[str, List[float]]] = {}
    for animal, root in ROOTS.items():
        print(f"\n=== {animal} ===")
        per_animal_day_latencies[animal] = {}
        for day in DAYS:
            beh = os.path.join(root, day, BEHAV_SUB)
            if not os.path.isdir(beh):
                print(f"  [skip] {beh} not found"); continue
            print(f"  Processing {day} ...")
            df_trials = process_behaviour_folder(beh, LANDMARKS)
            if df_trials.empty:
                print(f"  [warn] No pairs found in {beh}"); continue
            df_trials["AnimalID"] = animal; df_trials["Day"] = day
            all_rows.append(df_trials)
            lat = df_trials["latency_s"].to_numpy(float)
            per_animal_day_latencies[animal][day] = [v if np.isfinite(v) else np.nan for v in lat]
    if len(all_rows) == 0:
        print("No data found across all animals/days."); return
    result = pd.concat(all_rows, ignore_index=True)
    out_csv = "cheeseboard_latencies_per_trial.csv"; result.to_csv(out_csv, index=False)
    print(f"\nSaved per-trial latency table: {os.path.abspath(out_csv)}")
    for animal, dct in per_animal_day_latencies.items():
        out_png = f"latency_by_day_{animal}.png"
        plot_per_animal_latencies(animal, dct, out_png=out_png, show=True)
    animal_day_means: Dict[str, Dict[str, Tuple[float,float]]] = {}
    for animal, dct in per_animal_day_latencies.items():
        animal_day_means[animal] = {}
        for day in DAYS:
            arr = np.array(dct.get(day, []), dtype=float)
            m, s, n = mean_sem(arr)
            if not np.isnan(m): animal_day_means[animal][day] = (m, s)
    plot_summary_across_animals(animal_day_means, out_png="latency_summary_four_animals.png", show=True)

if __name__ == "__main__":
    main()
