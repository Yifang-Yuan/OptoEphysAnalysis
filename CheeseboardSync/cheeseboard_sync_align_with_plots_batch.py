
#!/usr/bin/env python3
"""
Batch-align Cheeseboard Behaviour ↔ Ephys ↔ Photometry across SyncRecording* per Day,
SHOWING the per-trial QC plots (no plot files saved).

- For each SyncRecordingN:
    - Calls align_single_trial(..., make_plots=True) which displays:
        1) LFP1..4 (full SPAD window)
        2) Z-score (full SPAD window)
        3) LFP1..4 (searching epoch: entry→approach; or entry→end if no approach)
        4) Z-score (searching epoch)
        5) Head speed (px/s) entry→end
    - Writes aligned_cheeseboard.pkl in the SyncRecordingN folder
- Optionally writes a CSV summary (no figures saved).

Usage example:
--------------
from cheeseboard_sync_align_with_plots_batch import batch_align_day

day_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363\Day1"
beh_folder = os.path.join(day_folder, "Behaviour")
landmarks = {...}
batch_align_day(day_folder, beh_folder, landmarks, spad_fs=1682.92,
                make_plots=True, save_summary=True)
"""

from __future__ import annotations
import os, re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse align_single_trial from the plotting script
from cheeseboard_sync_align_with_plots import align_single_trial

SYNC_RE = re.compile(r'^SyncRecording(\d+)$', re.IGNORECASE)

def list_sync_recordings(day_folder: str) -> List[Tuple[int, str]]:
    """Return list of (N, fullpath) for subfolders named SyncRecordingN, sorted by N."""
    items = []
    for name in os.listdir(day_folder):
        full = os.path.join(day_folder, name)
        if os.path.isdir(full):
            m = SYNC_RE.match(name)
            if m:
                n = int(m.group(1))
                items.append((n, full))
    items.sort(key=lambda x: x[0])
    return items

def batch_align_day(day_folder: str, beh_folder: str, landmarks: Dict,
                    spad_fs: float = 1682.92, make_plots: bool = True,
                    save_summary: bool = True) -> pd.DataFrame:
    """
    Align all SyncRecording* in a Day folder against the shared Behaviour folder.
    Returns a pandas DataFrame summary (and writes CSV if save_summary).
    """
    recs = list_sync_recordings(day_folder)
    if not recs:
        raise FileNotFoundError(f"No SyncRecording* folders found in {day_folder}")

    rows = []
    for n, sync_dir in recs:
        trial_idx_one_based = n  # mapping: SyncRecordingN ↔ Behaviour index k=N-1
        print(f"\n--- Aligning SyncRecording{n} ---")
        try:
            out = align_single_trial(sync_dir, beh_folder, trial_idx_one_based,
                                     landmarks=landmarks, spad_fs=spad_fs,
                                     save_pickle=True, make_plots=make_plots)
            beh = out["beh"]; anchors = out["anchors"]
            rows.append({
                "SyncRecording": n,
                "sync_folder": sync_dir,
                "cam_csv": out["meta"]["cam_csv"],
                "dlc_csv": out["meta"]["dlc_csv"],
                "ephys_pkl": out["meta"]["ephys_pkl"],
                "cam_start_ephys_s": anchors["t_cam_start_ephys"],
                "spad_start_ephys_s": anchors["t_spad_start_ephys"],
                "spad_end_ephys_s": anchors["t_spad_end_ephys"],
                "entry_time_s": beh["entry_time"],
                "approach_time_s": beh["approach_time"],
                "latency_s": (None if (beh["entry_time"] is None or beh["approach_time"] is None)
                              else (beh["approach_time"] - beh["entry_time"])),
                "fps_median": beh["fps_median"],
                "fps_mean": beh["fps_mean"],
            })
        except Exception as e:
            print(f"[ERROR] SyncRecording{n}: {e}")
            rows.append({
                "SyncRecording": n,
                "sync_folder": sync_dir,
                "error": str(e)
            })

    summary = pd.DataFrame(rows)
    if save_summary:
        out_csv = os.path.join(day_folder, "day_alignment_summary.csv")
        summary.to_csv(out_csv, index=False)
        print(f"Saved day summary: {out_csv}")
    return summary

# -------------- CLI convenience --------------
if __name__ == "__main__":
    # Example CLI-like usage; edit paths and landmarks before running this file directly

    landmarks = {
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 12.0,
    }
    day_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1907336\Day3"
    beh_folder = os.path.join(day_folder, "Behaviour")
    batch_align_day(day_folder, beh_folder, landmarks, spad_fs=1682.92,
                    make_plots=True, save_summary=True)

    
    pass
