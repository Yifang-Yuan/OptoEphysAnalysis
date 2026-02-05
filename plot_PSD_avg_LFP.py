"""
avg_psd_move_vs_rest_ci.py

Adds SEGMENT_MODE = "full" to analyse the full aligned LFP trace:
    session.Ephys_tracking_spad_aligned[LFP_channel]

Other modes:
    "theta"     -> session.theta_part[LFP_channel] (after pynacollada_label_theta)
    "non_theta" -> session.non_theta_part[LFP_channel] (after pynacollada_label_theta)

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
FS = 10000  # Hz

MOVE_FOLDER = r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\Thesis_Figure3-7\Sleep"
REST_FOLDER = r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\Thesis_Figure3-7\Sleep"  # edit

LFP_CHANNEL = "LFP_4"

# Choose: "theta", "non_theta", or "full"
SEGMENT_MODE = "full"

# Thresholds for pynacollada_label_theta (only used if SEGMENT_MODE is theta/non_theta)
LOW_THRES = 0
HIGH_THRES = 8
PLOT_THETA_DIAGNOSTIC = False  # True to view per-sweep theta labelling

# Welch PSD params
NPERSEG = 8192
NFFT = 8192

# Plot ranges
XLIM_LOW = (0.5, 80)
XLIM_HIGH = (80, 200)
FIGSIZE = (4, 6)

# Optional y-lims (set None to auto)
YLIM_LOW = None   # e.g. (25, 59)
YLIM_HIGH = None  # e.g. (8, 30)
YLIM_LOW = (22, 57)   # e.g. (25, 59)
YLIM_HIGH = (8, 30)  # e.g. (8, 30)

MOVE_COLOUR = "black"
REST_COLOUR = "grey"
MOVE_LABEL = "Move"
REST_LABEL = "Rest"


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


def extract_segment_trace(session, lfp_channel: str, segment_mode: str):
    """
    segment_mode:
      - "full": use session.Ephys_tracking_spad_aligned[lfp_channel]
      - "theta": call pynacollada_label_theta then use session.theta_part[lfp_channel]
      - "non_theta": call pynacollada_label_theta then use session.non_theta_part[lfp_channel]
    """
    mode = segment_mode.lower()

    if mode == "full":
        if not hasattr(session, "Ephys_tracking_spad_aligned"):
            raise AttributeError("Session has no attribute 'Ephys_tracking_spad_aligned'.")
        if lfp_channel not in session.Ephys_tracking_spad_aligned:
            raise KeyError(f"Channel '{lfp_channel}' not found in session.Ephys_tracking_spad_aligned.")
        return np.asarray(session.Ephys_tracking_spad_aligned[lfp_channel])

    # theta / non-theta requires labelling
    session.pynacollada_label_theta(
        lfp_channel,
        Low_thres=LOW_THRES,
        High_thres=HIGH_THRES,
        save=False,
        plot_theta=PLOT_THETA_DIAGNOSTIC,
    )

    if mode == "theta":
        if hasattr(session, "theta_part") and session.theta_part is not None and lfp_channel in session.theta_part:
            return np.asarray(session.theta_part[lfp_channel])
        raise KeyError(f"Could not locate theta segment for '{lfp_channel}' in session.theta_part.")

    if mode in ["non_theta", "non-theta", "nontheta"]:
        if hasattr(session, "non_theta_part") and session.non_theta_part is not None and lfp_channel in session.non_theta_part:
            return np.asarray(session.non_theta_part[lfp_channel])
        
        raise KeyError(f"Could not locate non-theta segment for '{lfp_channel}' in session.non_theta_part.")

    raise ValueError("SEGMENT_MODE must be 'theta', 'non_theta', or 'full'.")


def compute_condition_psds(parent_folder: str, lfp_channel: str, segment_mode: str):
    recs = find_syncrecordings(parent_folder)
    if not recs:
        raise FileNotFoundError(f"No SyncRecording* subfolders found under: {parent_folder}")

    dpath = os.path.join(os.path.normpath(parent_folder), "")  # force trailing separator

    psds = []
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

            x = extract_segment_trace(sess, lfp_channel, segment_mode)
            f, Pxx = welch_psd_linear(x, fs=FS, nperseg=NPERSEG, nfft=NFFT)

            if f_ref is None:
                f_ref = f
            else:
                if f.shape != f_ref.shape or not np.allclose(f, f_ref):
                    Pxx = np.interp(f_ref, f, Pxx)
                    f = f_ref

            psds.append(Pxx)
            used.append(rec)

        except Exception as e:
            skipped.append((rec, str(e)))

    print(f"[{os.path.basename(parent_folder)}] processed {len(used)} / {len(recs)}; skipped {len(skipped)}")
    if skipped:
        for r, m in skipped[:10]:
            print("  -", r, "=>", m)

    if len(used) == 0:
        msg = "\n".join([f"  - {r}: {m}" for r, m in skipped[:10]])
        raise RuntimeError(
            f"No recordings were successfully processed under: {parent_folder}\n"
            f"First skipped errors:\n{msg}"
        )

    return f_ref, np.vstack(psds), used, skipped


def plot_two_conditions_with_ci(f, A_mean_db, A_lo_db, A_hi_db, A_label, A_colour,
                                B_mean_db, B_lo_db, B_hi_db, B_label, B_colour,
                                xlim, ylim, title=""):
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

    ax.plot(f, A_mean_db, color=A_colour, linewidth=2, linestyle="-", label=A_label)
    ax.fill_between(f, A_lo_db, A_hi_db, color=A_colour, alpha=0.2, linewidth=0)

    ax.plot(f, B_mean_db, color=B_colour, linewidth=2, linestyle="-", label=B_label)
    ax.fill_between(f, B_lo_db, B_hi_db, color=B_colour, alpha=0.2, linewidth=0)

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel("PSD [dB/Hz]", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Frequency [Hz]", fontsize=14)
    ax.set_title(title, fontsize=16)

    ax.legend(loc="upper right", frameon=False, fontsize=14)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.12)
    return fig, ax


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=== Move ===")
    f_move, psd_move, used_move, _ = compute_condition_psds(MOVE_FOLDER, LFP_CHANNEL, 'theta')

    print("=== Rest ===")
    f_rest, psd_rest, used_rest, _ = compute_condition_psds(REST_FOLDER, LFP_CHANNEL, 'non_theta')

    # Align grids if needed
    if f_rest.shape != f_move.shape or not np.allclose(f_rest, f_move):
        psd_rest = np.vstack([np.interp(f_move, f_rest, row) for row in psd_rest])
        f = f_move
    else:
        f = f_move

    # Mean and 95% CI in linear units, then convert to dB for plotting
    m_move, lo_move, hi_move = mean_ci_t(psd_move, alpha=0.05)
    m_rest, lo_rest, hi_rest = mean_ci_t(psd_rest, alpha=0.05)

    m_move_db, lo_move_db, hi_move_db = to_db(m_move), to_db(lo_move), to_db(hi_move)
    m_rest_db, lo_rest_db, hi_rest_db = to_db(m_rest), to_db(lo_rest), to_db(hi_rest)

    #title = f"n(Move)={len(used_move)}, n(Rest)={len(used_rest)}"
    title = f"n(REM)={len(used_move)}, n(NonREM)={len(used_rest)}"

    # Figure 1: 0.5–80 Hz
    idx_low = (f >= XLIM_LOW[0]) & (f <= XLIM_LOW[1])
    plot_two_conditions_with_ci(
        f[idx_low],
        m_move_db[idx_low], lo_move_db[idx_low], hi_move_db[idx_low], MOVE_LABEL, MOVE_COLOUR,
        m_rest_db[idx_low], lo_rest_db[idx_low], hi_rest_db[idx_low], REST_LABEL, REST_COLOUR,
        xlim=XLIM_LOW,
        ylim=YLIM_LOW,
        title=title,
    )
    plt.show()

    # Figure 2: 80–200 Hz
    idx_high = (f >= XLIM_HIGH[0]) & (f <= XLIM_HIGH[1])
    plot_two_conditions_with_ci(
        f[idx_high],
        m_move_db[idx_high], lo_move_db[idx_high], hi_move_db[idx_high], MOVE_LABEL, MOVE_COLOUR,
        m_rest_db[idx_high], lo_rest_db[idx_high], hi_rest_db[idx_high], REST_LABEL, REST_COLOUR,
        xlim=XLIM_HIGH,
        ylim=YLIM_HIGH,
        title=title,
    )
    plt.show()
