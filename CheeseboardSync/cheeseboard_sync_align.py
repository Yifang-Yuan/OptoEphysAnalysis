#!/usr/bin/env python3
"""
Cheeseboard: align Behaviour ↔ Ephys ↔ ATLAS Photometry

For each SyncRecordingN folder (e.g., ...\Day1\SyncRecording1):
- Load ephys pkl (open_ephys_read_pd.pkl) containing:
    columns: timestamps (s), LFP_1..LFP_4, SPAD_mask (bool), cam_mask (bool), and optional TTLs
    sampling rate: 30_000 Hz (used to build fine-grained ephys time vectors when needed)
- Load photometry CSVs in the same folder (Green_traceAll.csv, Red_traceAll.csv, Zscore_traceAll.csv)
  with known SPAD sampling rate (e.g., 1682.92 Hz).
- Load Behaviour files from the DayX\Behaviour folder:
    Cam_sync_{k}.csv and AnimalTracking_{k}DLC_..._filtered.csv
  where k = SyncRecordingN - 1.
- Compute Behaviour events from DLC:
    * full-body entry to cheeseboard (first frame both head & bottom inside board, transition)
    * reward approach (head inside reward zone while bottom outside)
- Align timelines using masks:
    * behaviour start time in ephys = first True index in cam_mask
    * photometry start time in ephys = first True index in SPAD_mask
    * camera relative time origin = first NaN/NaN in Cam_sync (start pulse)
- Map behaviour frames to ephys time:
    t_ephys_beh[k] = t_ephys_cam_start + (cam_ts[k] - cam_ts[start_nan_idx])
- Map photometry samples to ephys time:
    t_ephys_phot[n] = t_ephys_spad_start + n / spad_fs
- Cut all streams to the SPAD_mask window (≈ photometry duration)
- Optionally resample Behaviour head/bottom coordinates to the photometry timebase
- Save a per-trial pickle with aligned arrays & event times.

Outputs (pickle dict)
---------------------
{
  'meta': {...},
  'ephys': {'t': t_ephys_win, 'LFP_1':..., 'LFP_2':..., 'LFP_3':..., 'LFP_4':...},
  'phot':  {'t': t_phot_ephys, 'green': g, 'red': r, 'z': z, 'fs': spad_fs},
  'beh':   {'t': t_beh_ephys, 'head': (hx, hy), 'bottom': (bx, by),
            'entry_time': entry_time_ephys, 'approach_time': approach_time_ephys,
            'entry_frame': entry_idx, 'approach_frame': appr_idx,
            'cam_start_frame': cam_start_nan_idx},
}
"""

from __future__ import annotations
import os, re, glob, math, pickle
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

# -------------------- Behaviour helpers --------------------

CAM_IDX_RE = re.compile(r'Cam_sync_(\d+)\.csv$', re.IGNORECASE)
DLC_IDX_RE = re.compile(r'AnimalTracking_(\d+)DLC_.*_filtered\.csv$', re.IGNORECASE)

def index_cam(path: str) -> Optional[int]:
    m = CAM_IDX_RE.search(os.path.basename(path)); return int(m.group(1)) if m else None
def index_dlc(path: str) -> Optional[int]:
    m = DLC_IDX_RE.search(os.path.basename(path)); return int(m.group(1)) if m else None

def find_behaviour_files(beh_folder: str, trial_index_zero_based: int) -> Tuple[str, str]:
    cams = sorted(glob.glob(os.path.join(beh_folder, "Cam_sync_*.csv")))
    dlcs = sorted(glob.glob(os.path.join(beh_folder, "AnimalTracking_*filtered.csv")))
    cams_by_idx = {index_cam(p): p for p in cams if index_cam(p) is not None}
    dlcs_by_idx = {index_dlc(p): p for p in dlcs if index_dlc(p) is not None}
    if trial_index_zero_based not in cams_by_idx or trial_index_zero_based not in dlcs_by_idx:
        raise FileNotFoundError(f"Missing behaviour files for index {trial_index_zero_based} in {beh_folder}")
    return cams_by_idx[trial_index_zero_based], dlcs_by_idx[trial_index_zero_based]

def read_cam_sync(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"Timestamp", "Value.X", "Value.Y"}.issubset(df.columns):
        raise ValueError("Cam sync CSV needs columns: Timestamp, Value.X, Value.Y")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

def cam_first_anchor_index(df: pd.DataFrame) -> int:
    nan_both = df[["Value.X", "Value.Y"]].isna().all(axis=1).to_numpy()
    idx = np.flatnonzero(nan_both)
    if idx.size == 0:
        raise ValueError("No NaN/NaN anchor found in Cam_sync file.")
    return int(idx[0])

def estimate_cam_fps(ts: pd.Series) -> Tuple[float, float]:
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
    if scorer is None: scorer = get_top_scorer(df)
    x = df[(scorer, part, "x")].to_numpy(float)
    y = df[(scorer, part, "y")].to_numpy(float)
    return x, y

def inside_circle(x: np.ndarray, y: np.ndarray, center: Tuple[float,float], r: float) -> np.ndarray:
    cx, cy = center; return (x - cx)**2 + (y - cy)**2 <= r**2

def mean_radius_from_center(center: Tuple[float,float], points: List[Tuple[float,float]]) -> float:
    cx, cy = center; return float(np.mean([math.hypot(px-cx, py-cy) for px,py in points]))

def first_full_body_entry_idx(head_xy, bottom_xy, board_center, board_radius) -> Optional[int]:
    hx, hy = head_xy; bx, by = bottom_xy
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
    hx, hy = head_xy; bx, by = bottom_xy
    in_head = inside_circle(hx, hy, reward_pt, reward_radius)
    in_bottom = inside_circle(bx, by, reward_pt, reward_radius)
    hits = np.flatnonzero(in_head & (~in_bottom))
    return int(hits[0]) if hits.size else None

# -------------------- Ephys & Photometry IO --------------------

def load_ephys_pkl(pkl_path: str) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("open_ephys_read_pd.pkl should contain a pandas DataFrame.")
    required = {"timestamps","LFP_1","LFP_2","LFP_3","LFP_4","SPAD_mask","cam_mask"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in ephys pkl: {required - set(df.columns)}")
    return df

def load_phot_csvs(sync_folder: str) -> Dict[str, np.ndarray]:
    def read_one(name: str) -> np.ndarray:
        p = os.path.join(sync_folder, name)
        if not os.path.isfile(p): return None
        arr = pd.read_csv(p, header=0).iloc[:,0].to_numpy(float)
        return arr
    out = {}
    out["green"] = read_one("Green_traceAll.csv")
    out["red"]   = read_one("Red_traceAll.csv")
    out["z"]     = read_one("Zscore_traceAll.csv")
    return out

# -------------------- Core alignment --------------------

def align_single_trial(sync_folder: str, beh_folder: str, trial_idx_one_based: int,
                       landmarks: Dict, spad_fs: float = 1682.92,
                       save_pickle: bool = True) -> Dict:
    """
    Align one SyncRecordingN with Behaviour index k=N-1.
    Returns a dictionary and optionally saves '<sync_folder>\\aligned_cheeseboard.pkl'.
    """
    trial_k = trial_idx_one_based - 1

    # Load ephys & photometry
    ephys_pkl = os.path.join(sync_folder, "open_ephys_read_pd.pkl")
    E = load_ephys_pkl(ephys_pkl)
    t_ephys = E["timestamps"].to_numpy(float)  # seconds

    # Masks
    cam_mask = E["cam_mask"].to_numpy(bool)
    spad_mask = E["SPAD_mask"].to_numpy(bool)

    if not cam_mask.any():
        raise ValueError("cam_mask has no True samples.")
    if not spad_mask.any():
        raise ValueError("SPAD_mask has no True samples.")

    # Start times in ephys
    i_cam_start = int(np.flatnonzero(cam_mask)[0])
    i_spad_start = int(np.flatnonzero(spad_mask)[0])
    t_cam_start_ephys = float(t_ephys[i_cam_start])
    t_spad_start_ephys = float(t_ephys[i_spad_start])

    # Photometry signals & time in ephys
    phot = load_phot_csvs(sync_folder)
    n_phot = None
    for key in ("z","green","red"):
        if phot.get(key) is not None:
            n_phot = len(phot[key]); break
    if n_phot is None:
        raise FileNotFoundError("No photometry CSVs found in sync folder.")
    t_phot = np.arange(n_phot, dtype=float) / float(spad_fs)
    t_phot_ephys = t_spad_start_ephys + t_phot

    # Behaviour files
    cam_csv, dlc_csv = find_behaviour_files(beh_folder, trial_k)
    cam = read_cam_sync(cam_csv)
    cam_fps_med, cam_fps_mean = estimate_cam_fps(cam["Timestamp"])
    cam_ts = cam["Timestamp"]
    cam_ts_s = (cam_ts - cam_ts.iloc[0]).dt.total_seconds().to_numpy()
    cam_start_nan_idx = cam_first_anchor_index(cam)
    # Behaviour time (ephys) for each camera frame:
    # Align the camera start pulse (first NaN) to cam_mask start in ephys
    cam_origin_s = cam_ts_s[cam_start_nan_idx]
    t_beh_ephys = t_cam_start_ephys + (cam_ts_s - cam_origin_s)

    # DLC events
    dlc = read_dlc_filtered(dlc_csv); scorer = get_top_scorer(dlc)
    head_xy = xy_of(dlc, "head", scorer); bottom_xy = xy_of(dlc, "bottom", scorer)

    R = mean_radius_from_center(landmarks["cheeseboard_center"], landmarks["cheeseboard_ends"])
    entry_idx = first_full_body_entry_idx(head_xy, bottom_xy, landmarks["cheeseboard_center"], R)
    appr_idx  = first_reward_approach_idx(head_xy, bottom_xy, landmarks["reward_pt"], landmarks["reward_zone_radius"])

    entry_time_ephys = float(t_beh_ephys[entry_idx]) if entry_idx is not None else None
    approach_time_ephys = float(t_beh_ephys[appr_idx]) if appr_idx is not None else None

    # Window: cut everything to SPAD_mask True region (photometry duration)
    i_spad_end = int(np.flatnonzero(spad_mask)[-1])
    t_spad_end_ephys = float(t_ephys[i_spad_end])
    # Ephys window mask
    win = (t_ephys >= t_spad_start_ephys) & (t_ephys <= t_spad_end_ephys)
    ephys_win = {
        "t": t_ephys[win],
        "LFP_1": E["LFP_1"].to_numpy(float)[win],
        "LFP_2": E["LFP_2"].to_numpy(float)[win],
        "LFP_3": E["LFP_3"].to_numpy(float)[win],
        "LFP_4": E["LFP_4"].to_numpy(float)[win],
    }

    # Behaviour within window (optional to keep full vectors too)
    beh_win_mask = (t_beh_ephys >= t_spad_start_ephys) & (t_beh_ephys <= t_spad_end_ephys)
    beh = {
        "t": t_beh_ephys[beh_win_mask],
        "head": (head_xy[0][beh_win_mask], head_xy[1][beh_win_mask]),
        "bottom": (bottom_xy[0][beh_win_mask], bottom_xy[1][beh_win_mask]),
        "entry_time": entry_time_ephys,
        "approach_time": approach_time_ephys,
        "entry_frame": entry_idx,
        "approach_frame": appr_idx,
        "cam_start_frame": cam_start_nan_idx,
        "fps_median": cam_fps_med,
        "fps_mean": cam_fps_mean,
    }

    # Package photometry
    phot_pack = {"t": t_phot_ephys, "fs": float(spad_fs)}
    for key in ("green","red","z"):
        if phot.get(key) is not None:
            phot_pack[key] = phot[key]

    out = {
        "meta": {
            "sync_folder": sync_folder,
            "beh_folder": beh_folder,
            "trial_index_one_based": trial_idx_one_based,
            "cam_csv": cam_csv,
            "dlc_csv": dlc_csv,
            "ephys_pkl": ephys_pkl,
        },
        "ephys": ephys_win,
        "phot": phot_pack,
        "beh": beh,
        "anchors": {
            "t_cam_start_ephys": t_cam_start_ephys,
            "t_spad_start_ephys": t_spad_start_ephys,
            "t_spad_end_ephys": t_spad_end_ephys,
        },
    }

    if save_pickle:
        out_path = os.path.join(sync_folder, "aligned_cheeseboard.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"Saved: {out_path}")

    return out

# -------------------- Example call --------------------
if __name__ == "__main__":
    # Example placeholder (fill these before running locally)
    sync_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363\Day1\SyncRecording1"
    beh_folder  = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363\Day1\Behaviour"
    landmarks = {
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 12.0,
    }
    align_single_trial(sync_folder, beh_folder, trial_idx_one_based=1, landmarks=landmarks, spad_fs=1682.92, save_pickle=True)
    pass
