# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 10:11:15 2026

@author: yifan
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
FS_ATLAS = 1682.92
FS_CAM = 10.0

CM_PER_PX_X = 0.0525
CM_PER_PX_Y = 0.0525

SPEED_THRESH_CM_S = 1.0


# -------------------------
# File finding (case-insensitive keyword matching)
# -------------------------
def find_csv_by_keywords(folder: Path, include: list[str], exclude: list[str] | None = None) -> Path | None:
    include = [k.lower() for k in include]
    exclude = [k.lower() for k in (exclude or [])]

    for p in sorted(folder.glob("*.csv")):
        name = p.name.lower()
        if all(k in name for k in include) and not any(k in name for k in exclude):
            return p
    return None


# -------------------------
# Optical loader (handles your 1-column traceAll.csv files)
# -------------------------
def load_optical_trace(csv_path: Path, fs_atlas: float = FS_ATLAS):
    """
    Supports:
      - single-column CSV with no header (your current files)
      - CSV with named columns (fallback)
    Returns (t_s, trace)
    """
    # Try read with header; if the "header" is actually numeric, re-read headerless
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        # If the column name is numeric-looking, it's probably the first data point mis-read as header
        try:
            float(str(df.columns[0]))
            df = pd.read_csv(csv_path, header=None)
        except Exception:
            pass

    if df.shape[1] == 1:
        trace = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
        t = np.arange(len(trace), dtype=float) / fs_atlas
        return t, trace

    # Named/multi-column fallback: use first numeric column
    numeric_cols = []
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().sum() > 0:
            numeric_cols.append(c)
    if not numeric_cols:
        raise ValueError(f"No numeric trace column found in {csv_path.name}")
    trace = pd.to_numeric(df[numeric_cols[0]], errors="coerce").dropna().to_numpy(dtype=float)
    t = np.arange(len(trace), dtype=float) / fs_atlas
    return t, trace


# -------------------------
# DLC loader + robust column extraction
# -------------------------
def read_dlc_filtered(csv_path: Path) -> pd.DataFrame:
    # Most DLC filtered exports are header=[0,1,2], index_col=0
    return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)


def choose_bodypart(dlc_df: pd.DataFrame) -> str:
    """
    Pick a stable point for speed. Preference order:
    shoulder -> bottom -> head -> first available
    """
    if not (isinstance(dlc_df.columns, pd.MultiIndex) and dlc_df.columns.nlevels >= 3):
        raise ValueError("DLC columns are not a MultiIndex; this script expects standard DLC output.")

    bodyparts = list(pd.unique(dlc_df.columns.get_level_values(1)))
    preferred = ["shoulder", "bottom", "head"]
    bp_lower = {str(bp).lower(): str(bp) for bp in bodyparts}
    for p in preferred:
        if p in bp_lower:
            return bp_lower[p]
    return str(bodyparts[0])


def get_dlc_series(dlc_df: pd.DataFrame, bodypart: str, coord: str) -> pd.Series:
    """
    Robustly fetch e.g. (bodypart='head', coord='x') without xs(slice(None),...).
    """
    cols = dlc_df.columns
    mask = (cols.get_level_values(1) == bodypart) & (cols.get_level_values(2) == coord)
    if mask.sum() == 0:
        raise KeyError(f"Missing DLC column for bodypart={bodypart}, coord={coord}")
    return pd.to_numeric(dlc_df.loc[:, mask].iloc[:, 0], errors="coerce")


def compute_speed_cm_s(
    dlc_df: pd.DataFrame,
    bodypart: str = "shoulder",
    fs_cam: float = FS_CAM,
    cm_per_px_x: float = CM_PER_PX_X,
    cm_per_px_y: float = CM_PER_PX_Y,
    likelihood_thresh: float | None = None,   # default: ignore likelihood
    smooth_frames: int = 3,                   # rolling median window at 10 Hz
    max_speed_cm_s: float = 50.0,             # NEW: clamp/remove >50 cm/s
) -> pd.DataFrame:
    """
    Speed (cm/s) at tracking rate (10 Hz), computed from bodypart x/y using anisotropic scaling.

    New behaviour:
      - Any frame with speed > max_speed_cm_s is treated as artefactual:
        set to NaN and linearly interpolated from neighbours.
    """

    x = get_dlc_series(dlc_df, bodypart, "x")
    y = get_dlc_series(dlc_df, bodypart, "y")

    # Optional likelihood gating (safe)
    if likelihood_thresh is not None:
        try:
            like = get_dlc_series(dlc_df, bodypart, "likelihood")
            good_frac = np.mean(like.to_numpy(dtype=float) >= likelihood_thresh)
            if good_frac >= 0.05:
                bad = (like < likelihood_thresh) | (~np.isfinite(like))
                x = x.mask(bad)
                y = y.mask(bad)
        except KeyError:
            pass

    # Fill gaps in positions
    x = pd.to_numeric(x, errors="coerce").interpolate(limit_direction="both")
    y = pd.to_numeric(y, errors="coerce").interpolate(limit_direction="both")

    # Per-frame displacement in cm
    dx_cm = x.diff() * cm_per_px_x
    dy_cm = y.diff() * cm_per_px_y
    ds_cm = np.sqrt(dx_cm**2 + dy_cm**2)

    # Speed (cm/s)
    speed = ds_cm * float(fs_cam)
    speed.iloc[0] = 0.0

    # NEW: remove unrealistic spikes and interpolate
    if max_speed_cm_s is not None:
        spike = speed > float(max_speed_cm_s)
        if spike.any():
            speed = speed.mask(spike)
            speed = speed.interpolate(limit_direction="both")

    # Light smoothing (median is robust)
    speed = speed.rolling(smooth_frames, min_periods=1, center=True).median()
    speed.name = "speed_cm_s"

    t = np.arange(len(speed), dtype=float) / fs_cam
    return pd.DataFrame({
        "t_s": t,
        "x": x.to_numpy(dtype=float),
        "y": y.to_numpy(dtype=float),
        "speed_cm_s": speed.to_numpy(dtype=float),
    })


def interp_to(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src, float)
    y_src = np.asarray(y_src, float)
    t_dst = np.asarray(t_dst, float)
    if len(t_src) < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    t_clip = np.clip(t_dst, t_src[0], t_src[-1])
    return np.interp(t_clip, t_src, y_src)


# -------------------------
# Per-folder processing
# -------------------------
def process_sync_recording_folder(folder: Path):
    folder = Path(folder)

    # Flexible optical patterns
    ca1_l_path = find_csv_by_keywords(folder, include=["green", "traceall"])
    ca1_r_path = find_csv_by_keywords(folder, include=["red", "traceall"])
    ca3_path = (
        find_csv_by_keywords(folder, include=["zscore", "traceall"])
        or find_csv_by_keywords(folder, include=["ca3", "traceall"])
        or find_csv_by_keywords(folder, include=["zscore"])  # last resort
    )

    if ca1_l_path is None:
        raise FileNotFoundError(f"Could not find CA1_L (green) traceAll CSV in {folder}")
    if ca1_r_path is None:
        raise FileNotFoundError(f"Could not find CA1_R (red) traceAll CSV in {folder}")
    if ca3_path is None:
        raise FileNotFoundError(f"Could not find CA3_L (zscore/CA3) traceAll CSV in {folder}")

    t_g, ca1_l = load_optical_trace(ca1_l_path)
    t_r, ca1_r = load_optical_trace(ca1_r_path)
    t_c, ca3_l = load_optical_trace(ca3_path)

    # Use common optical overlap
    t_end = min(t_g[-1], t_r[-1], t_c[-1])
    n_opt = int(np.floor(t_end * FS_ATLAS)) + 1
    t_opt = np.arange(n_opt, dtype=float) / FS_ATLAS

    ca1_l_opt = interp_to(t_g, ca1_l, t_opt)
    ca1_r_opt = interp_to(t_r, ca1_r, t_opt)
    ca3_l_opt = interp_to(t_c, ca3_l, t_opt)

    # DLC file pattern
    dlc_path = (
        find_csv_by_keywords(folder, include=["dlc", "filtered"])
        or find_csv_by_keywords(folder, include=["filtered"])
    )
    if dlc_path is None:
        raise FileNotFoundError(f"Could not find DLC filtered CSV in {folder}")

    dlc_df = read_dlc_filtered(dlc_path)
    bodypart = choose_bodypart(dlc_df)

    beh = compute_speed_cm_s(dlc_df, bodypart="shoulder", likelihood_thresh=None, max_speed_cm_s=50.0)

    # Truncate behaviour to optical duration (camera ran longer)
    beh = beh[beh["t_s"] <= t_opt[-1]].reset_index(drop=True)
    beh["state"] = np.where(beh["speed_cm_s"] > SPEED_THRESH_CM_S, "moving", "not_moving")

    # Interpolate behaviour onto optical timebase
    speed_opt = interp_to(beh["t_s"].to_numpy(), beh["speed_cm_s"].to_numpy(), t_opt)
    state_opt = np.where(speed_opt > SPEED_THRESH_CM_S, "moving", "not_moving")

    df_high = pd.DataFrame({
        "t_s": t_opt,
        "CA1_L": ca1_l_opt,
        "CA1_R": ca1_r_opt,
        "CA3_L": ca3_l_opt,
        "speed_cm_s": speed_opt,
        "state": state_opt,
    })

    df_10hz = pd.DataFrame({
        "t_s": beh["t_s"].to_numpy(),
        "speed_cm_s": beh["speed_cm_s"].to_numpy(),
        "state": beh["state"].to_numpy(),
        "CA1_L": interp_to(t_opt, ca1_l_opt, beh["t_s"].to_numpy()),
        "CA1_R": interp_to(t_opt, ca1_r_opt, beh["t_s"].to_numpy()),
        "CA3_L": interp_to(t_opt, ca3_l_opt, beh["t_s"].to_numpy()),
    })

    # Save
    df_high.to_parquet(folder / "synced_optical_behaviour_highrate.parquet", index=False)
    df_10hz.to_parquet(folder / "synced_optical_behaviour_10Hz.parquet", index=False)

    # QC plot
    decim = max(1, int(round(FS_ATLAS / 200.0)))
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_opt[::decim], ca1_l_opt[::decim], lw=0.8)
    axes[0].set_ylabel("CA1_L")

    axes[1].plot(t_opt[::decim], ca1_r_opt[::decim], lw=0.8)
    axes[1].set_ylabel("CA1_R")

    axes[2].plot(t_opt[::decim], ca3_l_opt[::decim], lw=0.8)
    axes[2].set_ylabel("CA3_L")

    axes[3].plot(beh["t_s"], beh["speed_cm_s"], lw=1.0)
    axes[3].axhline(SPEED_THRESH_CM_S, ls="--", lw=1.0)
    axes[3].set_ylabel("Speed (cm/s)")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle(f"{folder.name} | DLC point: {bodypart} | synced: {t_opt[-1]:.2f} s")
    fig.tight_layout()
    fig.savefig(folder / "synced_optical_behaviour_qc.png", dpi=150)
    plt.close(fig)

    return {
        "folder": folder.name,
        "opt_files": (ca1_l_path.name, ca1_r_path.name, ca3_path.name),
        "dlc_file": dlc_path.name,
        "bodypart": bodypart,
        "duration_s": float(t_opt[-1]),
    }


def batch_process_animal(animal_dir: str | Path):
    animal_dir = Path(animal_dir)
    sync_folders = sorted([p for p in animal_dir.glob("SyncRecording*") if p.is_dir()])
    if not sync_folders:
        raise FileNotFoundError(f"No SyncRecording* folders found under {animal_dir}")

    logs = []
    for f in sync_folders:
        try:
            info = process_sync_recording_folder(f)
            logs.append(info)
            print(f"[OK] {f.name} | bodypart={info['bodypart']} | {info['duration_s']:.2f}s")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")

    if logs:
        pd.DataFrame(logs).to_csv(animal_dir / "sync_optical_behaviour_runlog.csv", index=False)


if __name__ == "__main__":
    # Example:
    batch_process_animal(r"G:\2025_ATLAS_SPAD\MultiFibre2\1966293_Jedi2p_Multi\Day3")
    pass
