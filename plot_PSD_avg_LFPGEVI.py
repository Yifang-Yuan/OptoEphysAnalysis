# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 17:29:05 2026

@author: yifan
"""

"""
avg_psd_dualaxis_ci_single_folder.py

From ONE parent folder containing SyncRecording* subfolders:
- compute Welch PSD per sweep for LFP (black) and optical z-score (red)
- compute mean PSD ± 95% CI across sweeps
- plot on one figure with twin y-axes (optical left, LFP right)
- xlim: 0.5–100 Hz

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

PARENT_FOLDER = r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\Day9_Cont"

FS = 10000  # Hz (sampling rate for aligned traces used in PSD)

LFP_CHANNEL = "LFP_1"
OPTICAL_CHANNEL = "zscore_raw"  # your optical z-score channel name

# Choose segment mode:
#   "full"     -> no theta labelling; uses aligned traces directly
#   "theta"    -> uses session.theta_part[...] after pynacollada_label_theta
#   "non_theta"-> uses session.non_theta_part[...] after pynacollada_label_theta
SEGMENT_MODE = "full"

# Theta labelling parameters (only used if SEGMENT_MODE is theta/non_theta)
LOW_THRES = -0.2
HIGH_THRES = 8
PLOT_THETA_DIAGNOSTIC = False

# Welch PSD params
NPERSEG = 8192
NFFT = 8192

# Plot
XLIM = (0.5, 40)

FIGSIZE = (4, 6)

# Optional y-lims (set None to auto)
YLIM_OPTICAL = (-30,-20) # e.g. (-40, -15)
YLIM_LFP = (25, 50)   # e.g. (25, 55)

# Colours
#COLOUR_OPT = "#ff4d4d"  # red
COLOUR_OPT = 'green'  
COLOUR_LFP = "black"

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


def _get_from_any_attr(session, channel: str, attrs):
    for attr in attrs:
        if hasattr(session, attr):
            obj = getattr(session, attr)
            try:
                if channel in obj:
                    return np.asarray(obj[channel])
            except Exception:
                pass
    raise KeyError(f"Channel '{channel}' not found in any of: {attrs}")


def extract_trace(session, lfp_channel: str, opt_channel: str, segment_mode: str):
    """
    Returns (lfp_trace, opt_trace) for one sweep according to segment_mode.
    """
    mode = segment_mode.lower()

    if mode == "full":
        # Your example: session.Ephys_tracking_spad_aligned[LFP_channel]
        # Optical may also be stored there in your pipeline; if not, add/adjust attrs below.
        lfp = _get_from_any_attr(session, lfp_channel, attrs=["Ephys_tracking_spad_aligned"])
        opt = _get_from_any_attr(
            session, opt_channel,
            attrs=["Ephys_tracking_spad_aligned", "SPAD_tracking_spad_aligned", "Photometry_tracking_spad_aligned"])
            
        return lfp, opt

    # theta / non-theta: call labelling for side effects, then read from session.theta_part etc.
    session.pynacollada_label_theta(
        lfp_channel,
        Low_thres=LOW_THRES,
        High_thres=HIGH_THRES,
        save=False,
        plot_theta=PLOT_THETA_DIAGNOSTIC,
    )

    if mode == "theta":
        if not hasattr(session, "theta_part"):
            raise AttributeError("Session has no attribute 'theta_part' after theta labelling.")
        lfp = np.asarray(session.theta_part[lfp_channel])
        opt = np.asarray(session.theta_part[opt_channel])
        return lfp, opt

    if mode in ["non_theta", "non-theta", "nontheta"]:
        # Attribute name can vary; try both
        if hasattr(session, "non_theta_part"):
            part = session.non_theta_part
        elif hasattr(session, "nontheta_part"):
            part = session.nontheta_part
        else:
            raise AttributeError("Session has no 'non_theta_part' (or 'nontheta_part') after theta labelling.")
        lfp = np.asarray(part[lfp_channel])
        opt = np.asarray(part[opt_channel])
        return lfp, opt

    raise ValueError("SEGMENT_MODE must be 'full', 'theta', or 'non_theta'.")


def compute_mean_ci_psd_one_folder(parent_folder: str):
    recs = find_syncrecordings(parent_folder)
    if not recs:
        raise FileNotFoundError(f"No SyncRecording* subfolders found under: {parent_folder}")

    dpath = os.path.join(os.path.normpath(parent_folder), "")  # trailing separator

    psd_lfp_list = []
    psd_opt_list = []
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

            lfp, opt = extract_trace(sess, LFP_CHANNEL, OPTICAL_CHANNEL, SEGMENT_MODE)

            f_lfp, Pxx_lfp = welch_psd_linear(lfp, fs=FS, nperseg=NPERSEG, nfft=NFFT)
            f_opt, Pxx_opt = welch_psd_linear(opt, fs=FS, nperseg=NPERSEG, nfft=NFFT)

            if f_ref is None:
                f_ref = f_lfp
            else:
                if f_lfp.shape != f_ref.shape or not np.allclose(f_lfp, f_ref):
                    Pxx_lfp = np.interp(f_ref, f_lfp, Pxx_lfp)

            # Align optical onto f_ref if needed
            if f_opt.shape != f_ref.shape or not np.allclose(f_opt, f_ref):
                Pxx_opt = np.interp(f_ref, f_opt, Pxx_opt)

            psd_lfp_list.append(Pxx_lfp)
            psd_opt_list.append(Pxx_opt)
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

    psd_lfp = np.vstack(psd_lfp_list)  # (n, n_freq)
    psd_opt = np.vstack(psd_opt_list)

    # Limit to x-range now
    idx = (f_ref >= XLIM[0]) & (f_ref <= XLIM[1])
    f = f_ref[idx]
    psd_lfp = psd_lfp[:, idx]
    psd_opt = psd_opt[:, idx]

    # Mean + 95% CI in linear units -> convert to dB for plotting
    m_lfp, lo_lfp, hi_lfp = mean_ci_t(psd_lfp, alpha=0.05)
    m_opt, lo_opt, hi_opt = mean_ci_t(psd_opt, alpha=0.05)

    out = dict(
        f=f,
        lfp_mean_db=to_db(m_lfp),
        lfp_lo_db=to_db(lo_lfp),
        lfp_hi_db=to_db(hi_lfp),
        opt_mean_db=to_db(m_opt),
        opt_lo_db=to_db(lo_opt),
        opt_hi_db=to_db(hi_opt),
        used=used,
    )
    return out


def plot_dualaxis_psd_ci(result):
    f = result["f"]

    fig, ax_opt = plt.subplots(1, 1, figsize=FIGSIZE)
    ax_lfp = ax_opt.twinx()

    # Optical (left axis, red)
    ax_opt.plot(f, result["opt_mean_db"], color=COLOUR_OPT, linewidth=2, label="GEVI mean")
    ax_opt.fill_between(f, result["opt_lo_db"], result["opt_hi_db"], color=COLOUR_OPT, alpha=0.20, linewidth=0)

    # LFP (right axis, black)
    ax_lfp.plot(f, result["lfp_mean_db"], color=COLOUR_LFP, linewidth=2, label="LFP mean")
    ax_lfp.fill_between(f, result["lfp_lo_db"], result["lfp_hi_db"], color=COLOUR_LFP, alpha=0.15, linewidth=0)

    # Axes formatting
    ax_opt.set_xlim(XLIM)

    ax_opt.set_xlabel("Frequency [Hz]", fontsize=14)

    ax_opt.set_ylabel("Optical PSD [dB/Hz]", color=COLOUR_OPT, fontsize=14)
    ax_opt.tick_params(axis="y", labelcolor=COLOUR_OPT, labelsize=14)
    ax_opt.tick_params(axis="x", labelsize=14)

    ax_lfp.set_ylabel("LFP PSD [dB/Hz]", color=COLOUR_LFP, fontsize=14)
    ax_lfp.tick_params(axis="y", labelcolor=COLOUR_LFP, labelsize=14)

    if YLIM_OPTICAL is not None:
        ax_opt.set_ylim(YLIM_OPTICAL)
    if YLIM_LFP is not None:
        ax_lfp.set_ylim(YLIM_LFP)

    # Combined legend (top right)
    h1, l1 = ax_opt.get_legend_handles_labels()
    h2, l2 = ax_lfp.get_legend_handles_labels()
    ax_lfp.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=14)

    # Layout similar to your convention
    fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

    title = f"n={len(result['used'])} sweeps"
    ax_opt.set_title(title, fontsize=14)

    plt.show()
    return fig, ax_opt, ax_lfp


# =========================
# Run
# =========================
if __name__ == "__main__":
    res = compute_mean_ci_psd_one_folder(PARENT_FOLDER)
    plot_dualaxis_psd_ci(res)
