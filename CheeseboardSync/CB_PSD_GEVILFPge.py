# -*- coding: utf-8 -*-
"""
avg_psd_dualaxis_ci_fullsweeps_lfp_pkl_gevi_csv.py

ONE plot (twin y-axes) with mean ± 95% CI:
- LFP PSD (black) from aligned_cheeseboard.pkl in each SyncRecording* folder
- GEVI PSD (green) from Green_traceAll.csv in each SyncRecording* folder (Fs=1682.92)

FULL sweeps only (no running/immobile split).

Notes:
- Welch settings are separate for LFP vs GEVI.
- CI is computed in LINEAR units then converted to dB/Hz for plotting.
- Frequency axis is aligned by interpolation onto a reference frequency grid per modality.

Python 3.8+ compatible.
"""

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import t


# =========================
# User settings (EDIT HERE)
# =========================
ROOT_FOLDER = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567"

# If ROOT/DayX/SyncRecording*/..., set DAYS; else set DAYS=None to search recursively.
DAYS = [ "Day2", "Day3", "Day4"]   # or None

# LFP from PKL
PKL_NAME = "aligned_cheeseboard.pkl"
LFP_CHANNEL = "LFP_4"

# GEVI from CSV
GEVI_CSV_NAME = "Green_traceAll.csv"
FS_GEVI = 1682.92

# Plot range
XLIM = (1.0, 40.0)
FIGSIZE = (4, 6)

# Colours
COLOUR_GEVI = "green"
COLOUR_LFP = "black"
ALPHA_GEVI = 0.20
ALPHA_LFP = 0.15

# Optional y-lims
YLIM_GEVI = None
YLIM_LFP = None

# --- Welch parameters (separate) ---
# LFP: more averaging (less noisy)
LFP_WINDOW_S = 2.0
LFP_OVERLAP_FRAC = 0.5
LFP_DETREND = False

# GEVI: choose longer windows for better frequency resolution (less “over-smooth”)
GEVI_WINDOW_S = 4.0
GEVI_OVERLAP_FRAC = 0.5
GEVI_DETREND = "constant"


# =========================
# Helpers
# =========================
def natural_sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return (name.lower(), int(m.group(1)) if m else -1)


def find_sync_dirs(root_folder: str, days=None):
    root_folder = os.path.normpath(root_folder)

    sync_dirs = []
    if days is None:
        pattern = os.path.join(root_folder, "**", "SyncRecording*")
        sync_dirs = [p for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
    else:
        for d in days:
            day_dir = os.path.join(root_folder, d)
            pattern = os.path.join(day_dir, "SyncRecording*")
            sync_dirs.extend([p for p in glob.glob(pattern) if os.path.isdir(p)])

    sync_dirs.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return sync_dirs


def infer_fs_from_t(tvec: np.ndarray, fallback: float) -> float:
    tvec = np.asarray(tvec, float)
    if tvec.size < 2:
        return float(fallback)
    dt = np.nanmedian(np.diff(tvec))
    if not (np.isfinite(dt) and dt > 0):
        return float(fallback)
    return float(1.0 / dt)


def welch_psd_linear_full(x: np.ndarray, fs: float, window_s: float, overlap_frac: float, detrend):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < int(0.5 * fs):
        return None, None

    nperseg = int(round(window_s * fs))
    nperseg = max(8, min(nperseg, x.size))

    if nperseg >= x.size:
        noverlap = 0
    else:
        noverlap = int(round(overlap_frac * nperseg))
        noverlap = min(max(noverlap, 0), nperseg - 1)

    f, Pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
        average="mean",
        return_onesided=True
    )
    return f, Pxx


def mean_ci_t(arr: np.ndarray, alpha: float = 0.05):
    arr = np.asarray(arr, float)
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)

    if n < 2:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)

    sd = np.nanstd(arr, axis=0, ddof=1)
    sem = sd / np.sqrt(n)
    tcrit = t.ppf(1 - alpha / 2, df=n - 1)
    lo = mean - tcrit * sem
    hi = mean + tcrit * sem
    return mean, lo, hi


def to_db(x: np.ndarray, eps: float = 1e-20):
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), eps))


def read_csv_1col(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("CSV trace too short after cleaning.")
    return x


# =========================
# Compute + plot
# =========================
def compute_mean_ci_psd(root_folder: str):
    sync_dirs = find_sync_dirs(root_folder, days=DAYS)
    if not sync_dirs:
        raise FileNotFoundError(f"No SyncRecording* folders found under: {root_folder}")

    P_lfp_list = []
    P_gevi_list = []
    used = []
    skipped = []

    f_lfp_ref = None
    f_gevi_ref = None

    for sd in sync_dirs:
        rec = os.path.basename(sd)

        pkl_path = os.path.join(sd, PKL_NAME)
        csv_path = os.path.join(sd, GEVI_CSV_NAME)

        # Require both for inclusion (keeps n consistent)
        if not (os.path.isfile(pkl_path) and os.path.isfile(csv_path)):
            continue

        try:
            # --- LFP from PKL
            with open(pkl_path, "rb") as f:
                D = pickle.load(f)
            e = D["ephys"]

            t_e = np.asarray(e["t"], float)
            lfp = np.asarray(e[LFP_CHANNEL], float)
            fs_e = infer_fs_from_t(t_e, fallback=30000.0)

            fL, PL = welch_psd_linear_full(
                lfp, fs=fs_e,
                window_s=LFP_WINDOW_S,
                overlap_frac=LFP_OVERLAP_FRAC,
                detrend=LFP_DETREND
            )
            if fL is None:
                raise ValueError("LFP trace too short for PSD.")

            if f_lfp_ref is None:
                f_lfp_ref = fL
            elif fL.shape != f_lfp_ref.shape or not np.allclose(fL, f_lfp_ref):
                PL = np.interp(f_lfp_ref, fL, PL)

            # --- GEVI from CSV
            gevi = read_csv_1col(csv_path)

            fG, PG = welch_psd_linear_full(
                gevi, fs=FS_GEVI,
                window_s=GEVI_WINDOW_S,
                overlap_frac=GEVI_OVERLAP_FRAC,
                detrend=GEVI_DETREND
            )
            if fG is None:
                raise ValueError("GEVI trace too short for PSD.")

            if f_gevi_ref is None:
                f_gevi_ref = fG
            elif fG.shape != f_gevi_ref.shape or not np.allclose(fG, f_gevi_ref):
                PG = np.interp(f_gevi_ref, fG, PG)

            P_lfp_list.append(PL)
            P_gevi_list.append(PG)
            used.append(rec)

        except Exception as ex:
            skipped.append((rec, str(ex)))

    if len(used) == 0:
        msg = "No recordings processed."
        if skipped:
            msg += "\nFirst skipped errors:\n" + "\n".join([f"  - {r}: {m}" for r, m in skipped[:10]])
        raise RuntimeError(msg)

    # Apply xlim to both frequency axes (may differ slightly across modalities)
    idxL = (f_lfp_ref >= XLIM[0]) & (f_lfp_ref <= XLIM[1])
    idxG = (f_gevi_ref >= XLIM[0]) & (f_gevi_ref <= XLIM[1])

    fL = f_lfp_ref[idxL]
    fG = f_gevi_ref[idxG]

    PL = np.vstack(P_lfp_list)[:, idxL]
    PG = np.vstack(P_gevi_list)[:, idxG]

    mL, loL, hiL = mean_ci_t(PL, alpha=0.05)
    mG, loG, hiG = mean_ci_t(PG, alpha=0.05)

    return {
        "n": len(used),
        "used": used,
        "fL": fL, "mL_db": to_db(mL), "loL_db": to_db(loL), "hiL_db": to_db(hiL),
        "fG": fG, "mG_db": to_db(mG), "loG_db": to_db(loG), "hiG_db": to_db(hiG),
        "skipped": skipped
    }


def plot_dualaxis(res):
    fig, ax_gevi = plt.subplots(1, 1, figsize=FIGSIZE)
    ax_lfp = ax_gevi.twinx()

    # GEVI (left)
    ax_gevi.plot(res["fG"], res["mG_db"], color=COLOUR_GEVI, linewidth=2, label="GEVI mean")
    ax_gevi.fill_between(res["fG"], res["loG_db"], res["hiG_db"], color=COLOUR_GEVI, alpha=ALPHA_GEVI, linewidth=0)

    # LFP (right)
    ax_lfp.plot(res["fL"], res["mL_db"], color=COLOUR_LFP, linewidth=2, label="LFP mean")
    ax_lfp.fill_between(res["fL"], res["loL_db"], res["hiL_db"], color=COLOUR_LFP, alpha=ALPHA_LFP, linewidth=0)

    ax_gevi.set_xlim(XLIM)
    ax_gevi.set_xlabel("Frequency [Hz]", fontsize=14)

    ax_gevi.set_ylabel("Optical PSD [dB/Hz]", color=COLOUR_GEVI, fontsize=14)
    ax_gevi.tick_params(axis="y", labelcolor=COLOUR_GEVI, labelsize=14)
    ax_gevi.tick_params(axis="x", labelsize=14)

    ax_lfp.set_ylabel("LFP PSD [dB/Hz]", color=COLOUR_LFP, fontsize=14)
    ax_lfp.tick_params(axis="y", labelcolor=COLOUR_LFP, labelsize=14)

    if YLIM_GEVI is not None:
        ax_gevi.set_ylim(YLIM_GEVI)
    if YLIM_LFP is not None:
        ax_lfp.set_ylim(YLIM_LFP)

    # Combined legend (top right), same convention as before
    h1, l1 = ax_gevi.get_legend_handles_labels()
    h2, l2 = ax_lfp.get_legend_handles_labels()
    ax_lfp.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=12)

    # Layout consistent with your earlier dual-axis figure
    fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)
    ax_gevi.set_title(f"n={res['n']} sweeps", fontsize=14)

    plt.show()
    return fig, ax_gevi, ax_lfp


# =========================
# Run
# =========================
if __name__ == "__main__":
    res = compute_mean_ci_psd(ROOT_FOLDER)
    plot_dualaxis(res)
