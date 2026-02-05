# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 15:51:53 2026

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
avg_psd_two_lfp_ci_single_folder.py

From ONE parent folder containing SyncRecording* subfolders:
- Extract LFP_2 and LFP_3 traces (FULL / theta / non-theta)
- For theta/non-theta: run pynacollada_label_theta() separately for each channel,
  using channel-specific LOW_THRES (because you requested different thresholds)
- Compute Welch PSD per sweep
- Compute mean PSD ± 95% CI across sweeps
- Plot BOTH LFP PSDs (same axis) with shaded 95% CI
- Colours: LFP_2 blue, LFP_3 purple

Python 3.8+ compatible.
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import t

from SyncOECPySessionClass import SyncOEpyPhotometrySession


# =========================
# User settings (EDIT HERE)
# =========================
PARENT_FOLDER = r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\PSD_Move"

FS = 10000  # Hz
# Welch PSD params
NPERSEG = 8192
NFFT = 8192

LFP2_CHANNEL = "LFP_2"
LFP3_CHANNEL = "LFP_3"

# Choose: "full", "theta", or "non_theta"
SEGMENT_MODE = "theta"

# Channel-specific theta thresholds (EDIT THESE)
LOW_THRES_MAP = {
    "LFP_2": -0.05,
    "LFP_3":  -0.3,
}
HIGH_THRES = 8
PLOT_THETA_DIAGNOSTIC = False  # True to show theta labelling plots per sweep (can be slow)

# Plot
XLIM = (2, 45)
FIGSIZE = (4, 6)
YLIM =  (25, 53)  # e.g. (25, 55) or leave None

# Colours
COLOUR_LFP2 = "tab:blue"
COLOUR_LFP3 = "tab:purple"


# =========================
# Helpers
# =========================
def natural_sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return (name.lower(), int(m.group(1)) if m else -1)


def find_syncrecordings(parent_folder: str):
    parent_folder = os.path.normpath(parent_folder)
    pattern = os.path.join(parent_folder, "SyncRecording*")
    paths = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    recs = [os.path.basename(p) for p in paths]
    recs.sort(key=natural_sort_key)
    return recs


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
    """
    arr: (n_sweeps, n_freq) in linear units
    returns mean, lo, hi in linear units (t-based CI)
    """
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


def _get_from_dict_attr(session, channel: str, attr: str):
    if not hasattr(session, attr):
        raise AttributeError(f"Session has no attribute '{attr}'.")
    obj = getattr(session, attr)
    try:
        if channel in obj:
            return np.asarray(obj[channel])
    except Exception:
        pass
    raise KeyError(f"Channel '{channel}' not found in session.{attr}.")


def extract_lfp_segment(session, lfp_channel: str, segment_mode: str, low_thres: float, high_thres: float):
    """
    Returns one 1D LFP trace for this sweep according to segment_mode.

    - full: session.Ephys_tracking_spad_aligned[lfp_channel]
    - theta/non_theta: run pynacollada_label_theta(lfp_channel, Low_thres=..., High_thres=...)
      then read session.theta_part / session.non_theta_part for that channel
    """
    mode = segment_mode.lower()

    if mode == "full":
        return _get_from_dict_attr(session, lfp_channel, "Ephys_tracking_spad_aligned")

    # theta or non-theta: call for side effects, do NOT unpack return value
    session.pynacollada_label_theta(
        lfp_channel,
        Low_thres=low_thres,
        High_thres=high_thres,
        save=False,
        plot_theta=PLOT_THETA_DIAGNOSTIC,
    )

    if mode == "theta":
        return _get_from_dict_attr(session, lfp_channel, "theta_part")

    if mode in ["non_theta", "non-theta", "nontheta"]:
        # naming varies across codebases; try both
        if hasattr(session, "non_theta_part"):
            return _get_from_dict_attr(session, lfp_channel, "non_theta_part")
        if hasattr(session, "nontheta_part"):
            return _get_from_dict_attr(session, lfp_channel, "nontheta_part")
        raise AttributeError("Session has no 'non_theta_part' (or 'nontheta_part') after theta labelling.")

    raise ValueError("SEGMENT_MODE must be 'full', 'theta', or 'non_theta'.")


def compute_mean_ci_psd_two_lfps_one_folder(parent_folder: str):
    recs = find_syncrecordings(parent_folder)
    if not recs:
        raise FileNotFoundError(f"No SyncRecording* subfolders found under: {parent_folder}")

    dpath = os.path.join(os.path.normpath(parent_folder), "")  # trailing separator

    psd_lfp2_list = []
    psd_lfp3_list = []
    used = []
    skipped = []
    f_ref = None

    for rec in recs:
        try:
            sess = SyncOEpyPhotometrySession(
                dpath,
                rec,
                IsTracking=False,
                read_aligned_data_from_file=True,
                recordingMode="Atlas",
                indicator="GEVI",
            )

            lfp2 = extract_lfp_segment(
                sess, LFP2_CHANNEL, SEGMENT_MODE,
                low_thres=LOW_THRES_MAP[LFP2_CHANNEL],
                high_thres=HIGH_THRES
            )
            lfp3 = extract_lfp_segment(
                sess, LFP3_CHANNEL, SEGMENT_MODE,
                low_thres=LOW_THRES_MAP[LFP3_CHANNEL],
                high_thres=HIGH_THRES
            )

            f2, P2 = welch_psd_linear(lfp2, fs=FS, nperseg=NPERSEG, nfft=NFFT)
            f3, P3 = welch_psd_linear(lfp3, fs=FS, nperseg=NPERSEG, nfft=NFFT)

            # establish reference grid from LFP_2 first sweep
            if f_ref is None:
                f_ref = f2
            else:
                if f2.shape != f_ref.shape or not np.allclose(f2, f_ref):
                    P2 = np.interp(f_ref, f2, P2)

            # align LFP_3 onto reference if needed
            if f3.shape != f_ref.shape or not np.allclose(f3, f_ref):
                P3 = np.interp(f_ref, f3, P3)

            psd_lfp2_list.append(P2)
            psd_lfp3_list.append(P3)
            used.append(rec)

        except Exception as e:
            skipped.append((rec, str(e)))

    print(f"[{os.path.basename(parent_folder)}] processed {len(used)} / {len(recs)}; skipped {len(skipped)}")
    if skipped:
        print("First skipped errors:")
        for r, m in skipped[:10]:
            print("  -", r, "=>", m)

    if len(used) == 0:
        raise RuntimeError("No recordings were successfully processed. See skipped messages above.")

    psd2 = np.vstack(psd_lfp2_list)  # (n, n_freq) linear
    psd3 = np.vstack(psd_lfp3_list)

    # Frequency window
    idx = (f_ref >= XLIM[0]) & (f_ref <= XLIM[1])
    f = f_ref[idx]
    psd2 = psd2[:, idx]
    psd3 = psd3[:, idx]

    # Mean + 95% CI in linear -> convert to dB for plotting
    m2, lo2, hi2 = mean_ci_t(psd2, alpha=0.05)
    m3, lo3, hi3 = mean_ci_t(psd3, alpha=0.05)

    return {
        "f": f,
        "lfp2_mean_db": to_db(m2),
        "lfp2_lo_db": to_db(lo2),
        "lfp2_hi_db": to_db(hi2),
        "lfp3_mean_db": to_db(m3),
        "lfp3_lo_db": to_db(lo3),
        "lfp3_hi_db": to_db(hi3),
        "used": used,
    }


def plot_two_lfps_ci(result):
    f = result["f"]

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    # LFP_2 (blue)
    ax.plot(f, result["lfp2_mean_db"], color=COLOUR_LFP2, linewidth=2, label="LFP2 mean")
    ax.fill_between(f, result["lfp2_lo_db"], result["lfp2_hi_db"], color=COLOUR_LFP2, alpha=0.20, linewidth=0)

    # LFP_3 (purple)
    ax.plot(f, result["lfp3_mean_db"], color=COLOUR_LFP3, linewidth=2, label="LFP3 mean")
    ax.fill_between(f, result["lfp3_lo_db"], result["lfp3_hi_db"], color=COLOUR_LFP3, alpha=0.20, linewidth=0)

    ax.set_xlim(XLIM)
    if YLIM is not None:
        ax.set_ylim(YLIM)

    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_ylabel("LFP PSD [dB/Hz]", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(loc="upper right", frameon=False, fontsize=14)

    fig.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.12)

    title = (
        f"{LFP2_CHANNEL} vs {LFP3_CHANNEL} | "
        f"n={len(result['used'])}"
    )
    ax.set_title(title, fontsize=12)

    plt.show()
    return fig, ax


# =========================
# Run
# =========================
if __name__ == "__main__":
    res = compute_mean_ci_psd_two_lfps_one_folder(PARENT_FOLDER)
    plot_two_lfps_ci(res)
