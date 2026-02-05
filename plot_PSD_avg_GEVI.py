# -*- coding: utf-8 -*-
"""
avg_psd_multiROI_gevi_ci_pkl_or_csv.py

Multi-ROI GEVI PSD (mean ± 95% CI) across SyncRecording* sweeps, with selectable input source:

INPUT_SOURCE = "csv"
  - Reads 3 ROI traces from CSV files in each sweep folder
  - Uses FS_OPTICAL_CSV, NPERSEG_CSV, NFFT_CSV

INPUT_SOURCE = "pkl"
  - Reads 3 ROI traces from the SAME PKL-aligned dict as LFP:
      session.Ephys_tracking_spad_aligned[channel]
  - Uses FS_PKL = 10000, NPERSEG_PKL = 8192, NFFT_PKL = 8192
  - Optional SEGMENT_MODE = "full" / "theta" / "non_theta" using pynacollada_label_theta
    (same logic as your extract_lfp_segment pattern; no unpacking)

GEVI only (no LFP plotting).

Python 3.8+ compatible.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import t, zscore

from SyncOECPySessionClass import SyncOEpyPhotometrySession


# =========================
# User settings (EDIT HERE)
# =========================
DPATH = r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\Day9_Cont"
SWEEP_GLOB = "*SyncRecording*"

# Choose input source: "csv" or "pkl"
INPUT_SOURCE = "csv"

# ---- PKL settings (required by you)
FS_PKL = 10000
NPERSEG_PKL = 8192
NFFT_PKL = 8192

# If INPUT_SOURCE == "pkl", you can choose segment mode:
#   "full"      -> session.Ephys_tracking_spad_aligned[channel]
#   "theta"     -> session.theta_part[channel] after pynacollada_label_theta(...)
#   "non_theta" -> session.non_theta_part[channel] after pynacollada_label_theta(...)
SEGMENT_MODE = "full"

# Theta labelling parameters (only used if SEGMENT_MODE != "full")
# Note: for GEVI segmentation we still use the same call signature; low/high are global here.
LOW_THRES = 0.2
HIGH_THRES = 8
PLOT_THETA_DIAGNOSTIC = False

# ---- CSV settings (only used if INPUT_SOURCE == "csv")
FS_OPTICAL_CSV = 1682.92
NPERSEG_CSV = 4096
NFFT_CSV = 4096

# Frequency axis
XLIM = (1.5, 40)
YLIM = None

# Whether to z-score each sweep trace before PSD
Z_SCORE_EACH_SWEEP = True

# ROI definitions for PKL and CSV
# (PKL) channels in session.Ephys_tracking_spad_aligned / theta_part / non_theta_part
# ROI_PKL_CHANNELS = [
#     ("sig_raw",    "L-CA1", "#2ca02c"),  # green
#     ("ref_raw",    "R-CA1", "#d62728"),  # red
#     ("zscore_raw", "L-CA3", "#b8860b"),  # more yellow (gold)
# ]

# # (CSV) filenames inside each sweep folder
# ROI_CSV_FILES = [
#     ("Green_traceAll.csv",  "L-CA1", "#2ca02c"),
#     ("Red_traceAll.csv",    "R-CA1", "#d62728"),
#     ("Zscore_traceAll.csv", "L-CA3", "#b8860b"),
# ]
ROI_PKL_CHANNELS = [
    ("sig_raw",    "Sig", "#2ca02c"),  # green
    ("ref_raw",    "Ref", "#d62728"),  # red
    ("zscore_raw", "Zscore", "#b8860b"),  # more yellow (gold)
]

# (CSV) filenames inside each sweep folder
ROI_CSV_FILES = [
    ("Green_traceAll.csv",  "Sig", "#2ca02c"),
    ("Red_traceAll.csv",    "Ref", "#d62728"),
    ("Zscore_traceAll.csv", "Zscore", "Blue"),
]
# Plot style
FIGSIZE = (4, 6)
LINEWIDTH = 2
ALPHA_CI = 0.05  # 95% CI


# =========================
# Helpers
# =========================
def natural_sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return (name.lower(), int(m.group(1)) if m else -1)


def find_sweeps(dpath: str, sweep_glob: str):
    dpath = os.path.normpath(dpath)
    pattern = os.path.join(dpath, sweep_glob)
    paths = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    paths.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return paths


def read_csv_1col(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError(f"Trace too short/invalid after cleaning: {csv_path}")
    return x


def welch_psd_linear(x: np.ndarray, fs: float, nperseg: int, nfft: int):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Trace too short after removing NaNs/Infs.")

    nperseg_eff = min(nperseg, x.size)
    nfft_eff = max(nfft, nperseg_eff)

    f, Pxx = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg_eff,
        noverlap=nperseg_eff // 2,
        nfft=nfft_eff,
        detrend="constant",
        scaling="density",
        average="mean",
    )
    return f, Pxx


def mean_ci_t(arr: np.ndarray, alpha: float = 0.05):
    arr = np.asarray(arr, dtype=float)
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
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), eps))


def _get_from_dict_attr(session, channel: str, attr: str) -> np.ndarray:
    if not hasattr(session, attr):
        raise AttributeError(f"Session has no attribute '{attr}'.")
    d = getattr(session, attr)
    try:
        if channel in d:
            x = np.asarray(d[channel], dtype=float)
            x = x[np.isfinite(x)]
            if x.size < 10:
                raise ValueError(f"Channel '{channel}' in session.{attr} is too short after cleaning.")
            return x
    except Exception:
        pass
    raise KeyError(f"Channel '{channel}' not found in session.{attr}.")


def extract_pkl_segment(session, channel: str, segment_mode: str, low_thres: float, high_thres: float) -> np.ndarray:
    """
    EXACTLY the same logic pattern as your extract_lfp_segment, but for ANY channel
    that lives in the same aligned dicts.

    - full:      session.Ephys_tracking_spad_aligned[channel]
    - theta:     run pynacollada_label_theta(channel, Low_thres=..., High_thres=...)
                then read session.theta_part[channel]
    - non_theta: run pynacollada_label_theta(...)
                then read session.non_theta_part[channel] (or nontheta_part)
    """
    mode = segment_mode.lower()

    if mode == "full":
        return _get_from_dict_attr(session, channel, "Ephys_tracking_spad_aligned")

    # theta/non-theta: call for side effects, DO NOT unpack return value
    session.pynacollada_label_theta(
        channel,
        Low_thres=low_thres,
        High_thres=high_thres,
        save=False,
        plot_theta=PLOT_THETA_DIAGNOSTIC,
    )

    if mode == "theta":
        return _get_from_dict_attr(session, channel, "theta_part")

    if mode in ["non_theta", "non-theta", "nontheta"]:
        if hasattr(session, "non_theta_part"):
            return _get_from_dict_attr(session, channel, "non_theta_part")
        if hasattr(session, "nontheta_part"):
            return _get_from_dict_attr(session, channel, "nontheta_part")
        raise AttributeError("Session has no 'non_theta_part' (or 'nontheta_part') after theta labelling.")

    raise ValueError("SEGMENT_MODE must be 'full', 'theta', or 'non_theta'.")


# =========================
# Main computation
# =========================
def compute_multiROI_psd_ci(dpath: str):
    sweep_paths = find_sweeps(dpath, SWEEP_GLOB)
    if not sweep_paths:
        raise FileNotFoundError(f"No sweep folders found under: {dpath} with pattern {SWEEP_GLOB}")

    if INPUT_SOURCE.lower() == "pkl":
        roi_spec = ROI_PKL_CHANNELS
        fs = FS_PKL
        nperseg = NPERSEG_PKL
        nfft = NFFT_PKL
    elif INPUT_SOURCE.lower() == "csv":
        roi_spec = ROI_CSV_FILES
        fs = FS_OPTICAL_CSV
        nperseg = NPERSEG_CSV
        nfft = NFFT_CSV
    else:
        raise ValueError("INPUT_SOURCE must be 'csv' or 'pkl'.")

    psd_by_roi = {label: [] for (_, label, _) in roi_spec}
    used = []
    skipped = []
    f_ref = None

    for sweep_path in sweep_paths:
        sweep_name = os.path.basename(sweep_path)
        try:
            traces = {}

            if INPUT_SOURCE.lower() == "csv":
                for fname, label, _ in roi_spec:
                    csv_path = os.path.join(sweep_path, fname)
                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"Missing {fname} in {sweep_name}")

                    x = read_csv_1col(csv_path)
                    if Z_SCORE_EACH_SWEEP:
                        x = zscore(x, ddof=0, nan_policy="omit")
                        x = np.asarray(x, dtype=float)
                        x = x[np.isfinite(x)]
                    traces[label] = x

            else:
                # PKL via session loader (same pkl pipeline as LFP)
                sess = SyncOEpyPhotometrySession(
                    os.path.join(os.path.normpath(dpath), ""),
                    sweep_name,
                    IsTracking=False,
                    read_aligned_data_from_file=True,
                    recordingMode="Atlas",
                    indicator="GEVI",
                )

                for ch_name, label, _ in roi_spec:
                    x = extract_pkl_segment(
                        sess,
                        channel=ch_name,
                        segment_mode=SEGMENT_MODE,
                        low_thres=LOW_THRES,
                        high_thres=HIGH_THRES,
                    )
                    if Z_SCORE_EACH_SWEEP:
                        x = zscore(x, ddof=0, nan_policy="omit")
                        x = np.asarray(x, dtype=float)
                        x = x[np.isfinite(x)]
                    traces[label] = x

            # PSD per ROI
            for label, x in traces.items():
                f, Pxx = welch_psd_linear(x, fs=fs, nperseg=nperseg, nfft=nfft)

                if f_ref is None:
                    f_ref = f
                else:
                    if f.shape != f_ref.shape or not np.allclose(f, f_ref):
                        Pxx = np.interp(f_ref, f, Pxx)

                psd_by_roi[label].append(Pxx)

            used.append(sweep_name)

        except Exception as e:
            skipped.append((sweep_name, str(e)))

    print(f"Processed {len(used)} sweeps; skipped {len(skipped)}. | source={INPUT_SOURCE}")
    if skipped:
        print("First skipped errors:")
        for s, m in skipped[:10]:
            print("  -", s, "=>", m)

    if len(used) == 0:
        raise RuntimeError("No sweeps were successfully processed. See skipped errors above.")

    # Filter frequency window
    idx = (f_ref >= XLIM[0]) & (f_ref <= XLIM[1])
    f_plot = f_ref[idx]

    results = {}
    for _, label, _ in roi_spec:
        arr = np.vstack(psd_by_roi[label])  # (n_sweeps, n_freq) linear
        arr = arr[:, idx]

        mean_lin, lo_lin, hi_lin = mean_ci_t(arr, alpha=ALPHA_CI)

        results[label] = dict(
            mean_db=to_db(mean_lin),
            lo_db=to_db(lo_lin),
            hi_db=to_db(hi_lin),
            n=arr.shape[0],
        )

    return f_plot, results, used, roi_spec


def plot_multiROI_psd_ci(f, results, roi_spec):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    for _, label, colour in roi_spec:
        d = results[label]
        ax.plot(f, d["mean_db"], color=colour, linewidth=LINEWIDTH, linestyle="-", label=label)
        ax.fill_between(f, d["lo_db"], d["hi_db"], color=colour, alpha=0.20, linewidth=0)

    ax.set_xlim(XLIM)
    if YLIM is not None:
        ax.set_ylim(YLIM)

    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_ylabel("Optical PSD [dB/Hz]", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.12)
    return fig, ax


# =========================
# Run
# =========================
if __name__ == "__main__":
    f_plot, results, used, roi_spec = compute_multiROI_psd_ci(DPATH)
    fig, ax = plot_multiROI_psd_ci(f_plot, results, roi_spec)

    ax.set_title(f"Multi-ROI mean PSD ±95% CI | n={len(used)}", fontsize=12)
    plt.show()
