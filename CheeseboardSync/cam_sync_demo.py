#!/usr/bin/env python3
"""
cam_sync_pipeline.py

Reads a camera sync CSV (columns: Timestamp, Value.X, Value.Y),
finds the first index where both Value.X and Value.Y become NaN (start of SPAD),
plots the region around that start pulse, and estimates the camera frame rate
from the timestamps. Can also batch process a folder of Cam_sync_*.csv files.

Usage examples:
    python cam_sync_pipeline.py --file "G:\\2025_ATLAS_SPAD\\CB\\1881363_Jedi2p_CB\\Day1\\Behaviour\\Cam_sync_0.csv"
    python cam_sync_pipeline.py --folder "G:\\2025_ATLAS_SPAD\\CB\\1881363_Jedi2p_CB\\Day1\\Behaviour" --save-plots

Outputs:
    - Printed start frame index (0-based and 1-based) and FPS (median & mean).
    - Optional PNG plot(s) saved next to input CSV(s).
    - Optional batch summary CSV (cam_sync_summary.csv) if using --folder.
"""
import argparse
import sys
import os
import glob
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_cam_sync(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' column not found in {csv_path}")
    if not {"Value.X", "Value.Y"}.issubset(df.columns):
        raise ValueError(f"'Value.X' and 'Value.Y' not both found in {csv_path}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def find_start_index(df: pd.DataFrame) -> int:
    nan_both = df[["Value.X", "Value.Y"]].isna().all(axis=1).to_numpy()
    idx = np.flatnonzero(nan_both)
    if idx.size == 0:
        raise ValueError("No rows found where both Value.X and Value.Y are NaN; cannot determine start.")
    return int(idx[0])


def estimate_fps(df: pd.DataFrame, use_range: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
    if use_range is not None:
        lo, hi = use_range
        ts = df["Timestamp"].iloc[lo:hi].reset_index(drop=True)
    else:
        ts = df["Timestamp"]
    dt = ts.diff().dt.total_seconds()
    # Robust to occasional timestamp jitter or dropped frames:
    fps_median = 1.0 / dt[1:].median()
    fps_mean = 1.0 / dt[1:].mean()
    return float(fps_median), float(fps_mean)


def plot_start_pulse(df: pd.DataFrame, start_idx: int, out_png: Optional[str] = None, window_before: int = 200, window_after: int = 50):
    lo = max(0, start_idx - window_before)
    hi = min(len(df) - 1, start_idx + window_after)

    fig = plt.figure(figsize=(10, 4))
    x = np.arange(lo, hi)
    plt.plot(x, df["Value.X"].iloc[lo:hi].to_numpy(), label="Value.X")
    plt.plot(x, df["Value.Y"].iloc[lo:hi].to_numpy(), label="Value.Y")
    plt.axvline(start_idx, linestyle="--", label=f"Start sync (idx={start_idx})")
    plt.title("Cam sync around start pulse (first NaN NaN)")
    plt.xlabel("Frame index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def process_file(csv_path: str, save_plot: bool = False) -> dict:
    df = read_cam_sync(csv_path)
    start_idx = find_start_index(df)
    fps_median, fps_mean = estimate_fps(df)

    # Duration and frame count for info
    frames = len(df)
    duration_s = (df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds()

    # Optional plot
    out_png = None
    if save_plot:
        out_png = os.path.splitext(csv_path)[0] + "_start_pulse.png"
    plot_start_pulse(df, start_idx, out_png=out_png)

    info = {
        "file": csv_path,
        "start_idx_0based": start_idx,
        "start_idx_1based": start_idx + 1,
        "start_timestamp": df["Timestamp"].iloc[start_idx].isoformat() if pd.notna(df["Timestamp"].iloc[start_idx]) else "",
        "frames": frames,
        "duration_s": duration_s,
        "fps_median": fps_median,
        "fps_mean": fps_mean,
    }

    print("\n=== Cam Sync Summary ===")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("========================\n")

    return info


def process_folder(folder: str, save_plots: bool = False, pattern: str = "Cam_sync_*.csv") -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern} in {folder}")

    rows = []
    for p in paths:
        print(f"Processing {p}")
        try:
            info = process_file(p, save_plot=save_plots)
            rows.append(info)
        except Exception as e:
            print(f"ERROR processing {p}: {e}", file=sys.stderr)

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(folder, "cam_sync_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description="Find start-of-SPAD from Cam_sync CSV and estimate FPS.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Path to a single Cam_sync_X.csv file.")
    g.add_argument("--folder", type=str, help="Path to a folder containing Cam_sync_*.csv files.")
    ap.add_argument("--save-plots", action="store_true", help="Save PNG plot(s) next to input file(s).")
    return ap.parse_args()


def main():
    # === MODIFY PATH HERE ===
    # Example: single file
    # csv_file = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour\Cam_sync_0.csv"
    # process_file(csv_file, save_plot=True)

    # Example: whole folder
    folder = r"G:\2025_ATLAS_SPAD\CB\1881363_Jedi2p_CB\Day1\Behaviour"
    process_folder(folder, save_plots=True)


if __name__ == "__main__":
    main()
