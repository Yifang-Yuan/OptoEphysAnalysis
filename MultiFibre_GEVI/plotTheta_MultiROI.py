# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 21:51:13 2026

@author: yifan
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy import signal


# -------------------------
# Config
# -------------------------
FS_ATLAS = 1682.92
FS_DS = 200.0                 # downsample for wavelet/theta analysis (fast + enough for 0–20 Hz)
THETA_BAND = (6.0, 10.0)       # adjust if needed
SPEED_THRESH = 3.0             # cm/s

# Example window
EXAMPLE_DUR_S = 3.0
WAVELET_FMAX = 20.0


# -------------------------
# I/O
# -------------------------
from scipy import signal
import numpy as np

def butter_highpass_filtfilt(x: np.ndarray, fs: float, cutoff: float = 2.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="highpass")
    return signal.filtfilt(b, a, x)

def load_synced_highrate(folder: Path) -> pd.DataFrame:
    folder = Path(folder)
    candidates = [
        folder / "synced_optical_behaviour_highrate.parquet",
        folder / "synced_optical_behaviour_highrate.pkl.gz",
        folder / "synced_optical_behaviour_highrate.pkl",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.name.endswith(".pkl.gz"):
                return pd.read_pickle(p, compression="gzip")
            if p.suffix == ".pkl":
                return pd.read_pickle(p)
    raise FileNotFoundError(f"No synced_optical_behaviour_highrate.(parquet|pkl.gz|pkl) in {folder}")


# -------------------------
# Utilities
# -------------------------
def detrend_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 5:
        return x
    x2 = x.copy()
    x2[~m] = np.nanmean(x2[m])
    return signal.detrend(x2, type="constant")


def downsample_poly(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    frac = Fraction(fs_out / fs_in).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    return signal.resample_poly(x, up=up, down=down)


def butter_bandpass_filtfilt(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-12:
        return (x - med)
    return (x - med) / (1.4826 * mad)

def clean_speed_trace(speed: np.ndarray, max_speed: float = 50.0) -> np.ndarray:
    """
    Replace unrealistically high speeds (>max_speed) with NaN, then interpolate.
    Keeps length unchanged.
    """
    if speed is None:
        return None
    s = np.asarray(speed, float).copy()
    bad = ~np.isfinite(s) | (s > max_speed)
    if bad.any():
        s[bad] = np.nan
        # linear interpolation over NaNs
        x = np.arange(len(s))
        good = np.isfinite(s)
        if good.sum() >= 2:
            s[~good] = np.interp(x[~good], x[good], s[good])
        elif good.sum() == 1:
            s[~good] = s[good][0]
        else:
            s[:] = 0.0
    return s

# -------------------------
# 2) Theta-cycle average (phase-binned) using CA1_L phase
# -------------------------

def theta_cycle_average_two_cycles_with_heatmaps(
    folder: Path,
    state: str = "all",                 # "all" | "moving" | "not_moving"
    theta_band: tuple[float, float] = (6.0, 10.0),
    fs_atlas: float = 1682.92,
    fs_ds: float = 200.0,
    n_phase_bins: int = 60,
    speed_thresh: float = 3.0,
    save_name: str | None = None,
    transparent: bool = True,
):
    folder = Path(folder)
    df = load_synced_highrate(folder)

    # --- extract + downsample ---
    ca1l = signal.detrend(df["CA1_L"].to_numpy(float), type="constant")
    ca1r = signal.detrend(df["CA1_R"].to_numpy(float), type="constant")
    ca3l = signal.detrend(df["CA3_L"].to_numpy(float), type="constant")
    spd  = df["speed_cm_s"].to_numpy(float)

    ca1l_ds = downsample_poly(ca1l, fs_atlas, fs_ds)
    ca1r_ds = downsample_poly(ca1r, fs_atlas, fs_ds)
    ca3l_ds = downsample_poly(ca3l, fs_atlas, fs_ds)
    spd_ds  = downsample_poly(spd,  fs_atlas, fs_ds)

    n = min(len(ca1l_ds), len(ca1r_ds), len(ca3l_ds), len(spd_ds))
    ca1l_ds, ca1r_ds, ca3l_ds, spd_ds = ca1l_ds[:n], ca1r_ds[:n], ca3l_ds[:n], spd_ds[:n]

    # --- theta bandpass ---
    ca1l_th = butter_bandpass_filtfilt(ca1l_ds, fs_ds, theta_band)
    ca1r_th = butter_bandpass_filtfilt(ca1r_ds, fs_ds, theta_band)
    ca3l_th = butter_bandpass_filtfilt(ca3l_ds, fs_ds, theta_band)

    # robust normalisation (keeps relative amplitude, avoids outliers)
    ca1l_th = robust_zscore(ca1l_th)
    ca1r_th = robust_zscore(ca1r_th)
    ca3l_th = robust_zscore(ca3l_th)

    # --- define cycles using CA1_L theta phase (Hilbert) ---
    phase = np.unwrap(np.angle(signal.hilbert(ca1l_th)))  # unwrapped radians
    cycle_id = np.floor((phase + np.pi) / (2 * np.pi)).astype(int)
    change = np.where(np.diff(cycle_id) != 0)[0] + 1
    bounds = np.r_[0, change, len(cycle_id)]

    # plausible theta-cycle length bounds (in samples at fs_ds)
    min_len = int(round(fs_ds / theta_band[1] * 0.7))   # ~14 samples @ 200 Hz, 10 Hz theta
    max_len = int(round(fs_ds / theta_band[0] * 1.5))   # ~50 samples @ 200 Hz, 6 Hz theta

    def cycle_ok(i0, i1):
        L = i1 - i0
        return (L >= min_len) and (L <= max_len)

    def state_ok(i0, i1):
        mspd = np.nanmean(spd_ds[i0:i1])
        if state == "moving":
            return mspd > speed_thresh
        if state == "not_moving":
            return mspd <= speed_thresh
        return True

    # --- warp each cycle to n_phase_bins ---
    mats = {"CA1_L": [], "CA1_R": [], "CA3_L": []}
    for k in range(len(bounds) - 1):
        i0, i1 = bounds[k], bounds[k + 1]
        if not cycle_ok(i0, i1):
            continue
        if not state_ok(i0, i1):
            continue

        # phase-normalised coordinate within cycle
        x_old = np.linspace(0, 1, i1 - i0, endpoint=False)
        x_new = np.linspace(0, 1, n_phase_bins, endpoint=False)

        mats["CA1_L"].append(np.interp(x_new, x_old, ca1l_th[i0:i1]))
        mats["CA1_R"].append(np.interp(x_new, x_old, ca1r_th[i0:i1]))
        mats["CA3_L"].append(np.interp(x_new, x_old, ca3l_th[i0:i1]))

    # stack into (n_cycles, n_bins)
    for k in mats:
        mats[k] = np.asarray(mats[k], float)

    n_cycles = mats["CA1_L"].shape[0]
    if n_cycles < 5:
        raise ValueError(f"{folder.name}: too few cycles ({n_cycles}) for {state}")

    # mean ± SEM across cycles
    def mean_sem(M):
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])
        return mu, se

    mu1, se1 = mean_sem(mats["CA1_L"])
    mu2, se2 = mean_sem(mats["CA1_R"])
    mu3, se3 = mean_sem(mats["CA3_L"])

    # two-cycle x-axis (0–720 deg)
    ph = np.linspace(0, 360, n_phase_bins, endpoint=False)
    ph2 = np.concatenate([ph, ph + 360])

    def two_cycle(arr):
        return np.concatenate([arr, arr])

    # heatmaps also repeated to 2 cycles
    H1 = np.concatenate([mats["CA1_L"], mats["CA1_L"]], axis=1)
    H2 = np.concatenate([mats["CA1_R"], mats["CA1_R"]], axis=1)
    H3 = np.concatenate([mats["CA3_L"], mats["CA3_L"]], axis=1)

    # shared colour limits across all heatmaps
    all_vals = np.concatenate([H1.ravel(), H2.ravel(), H3.ravel()])
    vmax = np.nanpercentile(np.abs(all_vals), 98)
    vmin = -vmax

    # --- plot (mean + 3 heatmaps) ---
    fig = plt.figure(figsize=(10, 9))
    fig.patch.set_facecolor("black")

    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.0, 1.0], hspace=0.15)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # mean traces (two cycles)
    ax0.plot(ph2, two_cycle(mu1), lw=2.2, label="CA1_L (ref)")
    ax0.fill_between(ph2, two_cycle(mu1 - se1), two_cycle(mu1 + se1), alpha=0.25)
    ax0.plot(ph2, two_cycle(mu2), lw=2.2, label="CA1_R")
    ax0.fill_between(ph2, two_cycle(mu2 - se2), two_cycle(mu2 + se2), alpha=0.25)
    ax0.plot(ph2, two_cycle(mu3), lw=2.2, label="CA3_L")
    ax0.fill_between(ph2, two_cycle(mu3 - se3), two_cycle(mu3 + se3), alpha=0.25)

    ax0.set_ylabel("Theta-band (robust z)", color="white")
    ax0.legend(loc="upper right")
    ax0.axvline(360, color="white", lw=1.0, alpha=0.5)

    # heatmaps
    im1 = ax1.imshow(H1, aspect="auto", origin="lower", extent=[0, 720, 0, n_cycles],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_ylabel("Cycle #", color="white")
    ax1.set_title("CA1_L", color="white", loc="left")
    ax1.axvline(360, color="white", lw=1.0, alpha=0.5)

    im2 = ax2.imshow(H2, aspect="auto", origin="lower", extent=[0, 720, 0, n_cycles],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_ylabel("Cycle #", color="white")
    ax2.set_title("CA1_R", color="white", loc="left")
    ax2.axvline(360, color="white", lw=1.0, alpha=0.5)

    im3 = ax3.imshow(H3, aspect="auto", origin="lower", extent=[0, 720, 0, n_cycles],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_ylabel("Cycle #", color="white")
    ax3.set_title("CA3_L", color="white", loc="left")
    ax3.axvline(360, color="white", lw=1.0, alpha=0.5)

    ax3.set_xlabel("CA1_L theta phase (deg)", color="white")

    # shared colourbar (bottom)
    cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3], orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.ax.tick_params(colors="white")

    fig.suptitle(f"{folder.name} | theta {theta_band[0]}–{theta_band[1]} Hz | {state} | cycles={n_cycles}",
                 color="white")
    # --- display vs save ---
    if save_name is not None:
        out_png = folder / save_name
        fig.savefig(out_png, dpi=300, transparent=False, bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.close(fig)
    return out_png

def mean_ci95(mat: np.ndarray):
    """
    mat: (n_epochs, n_time)
    returns mean, ci_low, ci_high
    """
    mu = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(min=1))
    ci = 1.96 * sem
    return mu, mu - ci, mu + ci


def extract_two_cycle_epochs_raw(
    ca1l_raw: np.ndarray, ca1r_raw: np.ndarray, ca3l_raw: np.ndarray,
    speed: np.ndarray | None,
    fs: float,
    theta_band: tuple[float, float],
    n_bins_per_cycle: int = 180,
    state: str = "all",           # "all" | "moving" | "not_moving"
    speed_thresh: float = 3.0,
    min_cycles: int = 5,
    normalise_for_display: str = "none",  # "none" | "robust_z"
):
    """
    Returns:
      M1, M2, M3: (n_epochs, 2*n_bins_per_cycle) raw-signal epochs (2 cycles)
      info dict
    Notes:
      - Cycle boundaries come from CA1_L theta-band Hilbert phase.
      - Epoch signals are RAW (not theta-filtered), optionally normalised for display.
    """
    n = min(len(ca1l_raw), len(ca1r_raw), len(ca3l_raw))
    ca1l_raw, ca1r_raw, ca3l_raw = ca1l_raw[:n], ca1r_raw[:n], ca3l_raw[:n]
    if speed is not None:
        speed = speed[:n]

    # Optional display normalisation (keeps “raw” meaning broadband, but improves visibility)
    if normalise_for_display == "robust_z":
        ca1l_raw_n = robust_zscore(ca1l_raw)
        ca1r_raw_n = robust_zscore(ca1r_raw)
        ca3l_raw_n = robust_zscore(ca3l_raw)
    else:
        ca1l_raw_n, ca1r_raw_n, ca3l_raw_n = ca1l_raw, ca1r_raw, ca3l_raw

    # Reference theta phase from CA1_L theta-band
    ca1l_theta = butter_bandpass_filtfilt(ca1l_raw, fs=fs, band=theta_band, order=4)
    phase = np.unwrap(np.angle(signal.hilbert(ca1l_theta)))

    # Define cycle IDs and boundaries
    cycle_id = np.floor((phase + np.pi) / (2 * np.pi)).astype(int)
    change = np.where(np.diff(cycle_id) != 0)[0] + 1
    bounds = np.r_[0, change, len(cycle_id)]

    # Plausible per-cycle length constraints (reject phase glitches)
    min_len = int(round(fs / theta_band[1] * 0.7))
    max_len = int(round(fs / theta_band[0] * 1.5))

    def cycle_ok(i0, i1):
        L = i1 - i0
        return (L >= min_len) and (L <= max_len)

    def state_ok(i0, i2):
        if speed is None:
            return True
        mspd = np.nanmean(speed[i0:i2])
        if state == "moving":
            return mspd > speed_thresh
        if state == "not_moving":
            return mspd <= speed_thresh
        return True

    # Build 2-cycle epochs by pairing consecutive cycles: [i0:i1] + [i1:i2]
    M1, M2, M3 = [], [], []
    kept = 0
    rejected = 0

    for k in range(len(bounds) - 2):
        i0, i1, i2 = bounds[k], bounds[k + 1], bounds[k + 2]

        if not (cycle_ok(i0, i1) and cycle_ok(i1, i2)):
            rejected += 1
            continue
        if not state_ok(i0, i2):
            rejected += 1
            continue

        # Warp each cycle separately to n_bins_per_cycle, then concatenate -> 2 cycles
        def warp_two_cycles(x):
            seg0 = x[i0:i1]
            seg1 = x[i1:i2]
            x_old0 = np.linspace(0, 1, len(seg0), endpoint=False)
            x_old1 = np.linspace(0, 1, len(seg1), endpoint=False)
            x_new = np.linspace(0, 1, n_bins_per_cycle, endpoint=False)
            w0 = np.interp(x_new, x_old0, seg0)
            w1 = np.interp(x_new, x_old1, seg1)
            return np.concatenate([w0, w1])

        M1.append(warp_two_cycles(ca1l_raw_n))
        M2.append(warp_two_cycles(ca1r_raw_n))
        M3.append(warp_two_cycles(ca3l_raw_n))
        kept += 1

    M1 = np.asarray(M1, float)
    M2 = np.asarray(M2, float)
    M3 = np.asarray(M3, float)

    if kept < min_cycles:
        raise ValueError(f"Too few 2-cycle epochs kept: {kept} (rejected {rejected})")

    info = {
        "kept_epochs": kept,
        "rejected_epochs": rejected,
        "n_bins_per_cycle": n_bins_per_cycle,
        "state": state,
    }
    return M1, M2, M3, info


def find_theta_troughs(theta_ref: np.ndarray, fs: float,
                       theta_band: tuple[float, float] = (6.0, 10.0),
                       min_prominence: float | None = None):
    """
    Find trough indices on a theta-band signal using find_peaks on the inverted signal.
    """
    # troughs = peaks of (-theta_ref)
    min_dist = int(round(fs / theta_band[1] * 0.7))  # ~0.7 * shortest theta period
    inv = -theta_ref

    # If prominence not specified, choose a robust default based on MAD
    if min_prominence is None:
        med = np.nanmedian(inv)
        mad = np.nanmedian(np.abs(inv - med)) + 1e-12
        min_prominence = 0.5 * (1.4826 * mad)

    trough_idx, _ = signal.find_peaks(inv, distance=min_dist, prominence=min_prominence)
    return trough_idx

def _despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)

def _style_ticks(ax, tick_fs=14):
    ax.tick_params(axis="both", which="both", labelsize=tick_fs, length=4, width=1)

def plot_theta_trough_triggered_raw_average(
    folder,
    state: str = "all",                 # "all" | "moving" | "not_moving"
    theta_band: tuple[float, float] = (6.0, 10.0),
    fs_atlas: float = FS_ATLAS,
    fs_ds: float = FS_DS,
    t_before: float = 0.2,
    t_after: float = 0.2,
    speed_thresh: float = SPEED_THRESH,
    robust_display_z: bool = True,      # z for display only (keeps “raw broadband” nature)
    min_epochs: int = 20,
    show: bool = True,
    save_path=None,                     # None => do not save
):
    """
    Trough-triggered raw window average (your method).

    Heatmaps:
      - each row = one trough-centred epoch (RAW signal, optionally robust-z for display)
      - epochs include only troughs that have full window inside the recording
      - optional state filter based on mean speed in the window
    """
    folder = Path(folder)
    df = load_synced_highrate(folder)

    # --- load and detrend raw ---
    ca1l = detrend_nan(df["CA1_L"].to_numpy(float))
    ca1r = detrend_nan(df["CA1_R"].to_numpy(float))
    ca3l = detrend_nan(df["CA3_L"].to_numpy(float))
    spd  = df["speed_cm_s"].to_numpy(float) if "speed_cm_s" in df.columns else None

    # --- downsample for speed + trough detection stability ---
    ca1l_ds = downsample_poly(ca1l, fs_atlas, fs_ds)
    ca1r_ds = downsample_poly(ca1r, fs_atlas, fs_ds)
    ca3l_ds = downsample_poly(ca3l, fs_atlas, fs_ds)
    spd_ds  = downsample_poly(spd, fs_atlas, fs_ds) if spd is not None else None

    n = min(len(ca1l_ds), len(ca1r_ds), len(ca3l_ds), len(spd_ds) if spd_ds is not None else 10**12)
    ca1l_ds, ca1r_ds, ca3l_ds = ca1l_ds[:n], ca1r_ds[:n], ca3l_ds[:n]
    if spd_ds is not None:
        spd_ds = clean_speed_trace(spd_ds, max_speed=50.0)

    # --- theta reference (filtered) for trough finding only ---
    ca1l_theta = butter_bandpass_filtfilt(ca1l_ds, fs=fs_ds, band=theta_band, order=4)

    trough_idx = find_theta_troughs(ca1l_theta, fs=fs_ds, theta_band=theta_band)

    # --- fixed window indices around trough ---
    pre  = int(round(t_before * fs_ds))
    post = int(round(t_after * fs_ds))
    L = pre + post + 1

    # keep only troughs with full window in bounds
    good = trough_idx[(trough_idx - pre >= 0) & (trough_idx + post < n)]

    # optional state filter based on mean speed in the window
    if spd_ds is not None and state != "all":
        keep_mask = []
        for ti in good:
            mspd = np.nanmean(spd_ds[ti - pre : ti + post + 1])
            if state == "moving":
                keep_mask.append(mspd > speed_thresh)
            else:  # "not_moving"
                keep_mask.append(mspd <= speed_thresh)
        good = good[np.array(keep_mask, dtype=bool)]

    if good.size < min_epochs:
        raise ValueError(f"{folder.name}: too few trough epochs ({good.size}) for state='{state}'")

    # --- extract RAW epochs ---
    def stack_epochs(x):
        return np.vstack([x[ti - pre : ti + post + 1] for ti in good])

    E1 = stack_epochs(ca1l_ds)
    E2 = stack_epochs(ca1r_ds)
    E3 = stack_epochs(ca3l_ds)

    # display scaling (optional): robust z-score per channel across the whole sweep
    if robust_display_z:
        E1 = robust_zscore(E1)
        E2 = robust_zscore(E2)
        E3 = robust_zscore(E3)

    # mean + CI
    mu1, lo1, hi1 = mean_ci95(E1)
    mu2, lo2, hi2 = mean_ci95(E2)
    mu3, lo3, hi3 = mean_ci95(E3)

    # time axis relative to trough
    t_rel = np.linspace(-t_before, t_after, L)

    # heatmap colour range (shared)
    all_vals = np.concatenate([E1.ravel(), E2.ravel(), E3.ravel()])
    vmax = np.nanpercentile(np.abs(all_vals), 99)
    vmin = -vmax

        # --- plot (journal styling) ---
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.0, 1.0], hspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    # font sizes (tweak once, used everywhere)
    title_fs  = 18
    label_fs  = 16
    tick_fs   = 14
    legend_fs = 14
    cbar_fs   = 14

    # Top mean traces
    ax0.plot(t_rel, mu1, lw=2.5, label="CA1_L (raw)")
    ax0.fill_between(t_rel, lo1, hi1, alpha=0.18)

    ax0.plot(t_rel, mu2, lw=2.5, label="CA1_R (raw)")
    ax0.fill_between(t_rel, lo2, hi2, alpha=0.18)

    ax0.plot(t_rel, mu3, lw=2.5, label="CA3_L (raw)")
    ax0.fill_between(t_rel, lo3, hi3, alpha=0.18)

    ax0.axvline(0.0, lw=1.6, alpha=0.55)
    ax0.set_ylabel("Raw (display scale)", fontsize=label_fs)
    ax0.set_title(f"{folder.name} | trough-triggered raw | {state} | epochs={good.size}", fontsize=title_fs)

    leg = ax0.legend(loc="upper right", fontsize=legend_fs, frameon=False)
    for lh in leg.legendHandles:
        lh.set_linewidth(3.0)

    # Heatmaps
    im1 = ax1.imshow(E1, aspect="auto", origin="lower",
                     extent=[t_rel[0], t_rel[-1], 0, E1.shape[0]],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_ylabel("Epoch #", fontsize=label_fs)
    ax1.set_title("CA1_L raw", fontsize=label_fs, loc="left")
    ax1.axvline(0.0, lw=1.4, alpha=0.55)

    im2 = ax2.imshow(E2, aspect="auto", origin="lower",
                     extent=[t_rel[0], t_rel[-1], 0, E2.shape[0]],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_ylabel("Epoch #", fontsize=label_fs)
    ax2.set_title("CA1_R raw", fontsize=label_fs, loc="left")
    ax2.axvline(0.0, lw=1.4, alpha=0.55)

    im3 = ax3.imshow(E3, aspect="auto", origin="lower",
                     extent=[t_rel[0], t_rel[-1], 0, E3.shape[0]],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_ylabel("Epoch #", fontsize=label_fs)
    ax3.set_title("CA3_L raw", fontsize=label_fs, loc="left")
    ax3.axvline(0.0, lw=1.4, alpha=0.55)
    ax3.set_xlabel("Time relative to CA1_L theta trough (s)", fontsize=label_fs)

    # Remove frames and style ticks
    for ax in (ax0, ax1, ax2, ax3):
        _despine_all(ax)
        _style_ticks(ax, tick_fs=tick_fs)

    # Hide upper x tick labels (cleaner)
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    # Shared colourbar
    cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3], orientation="horizontal",
                        fraction=0.06, pad=0.10)
    cbar.set_label("Raw value (display scale)", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=cbar_fs)

    fig.tight_layout()


    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return save_path

    if show:
        plt.show()
        return fig
    else:
        plt.close(fig)
        return None

def _extract_trough_epochs_from_folder(
    sync_folder: Path,
    state: str,
    theta_band: tuple[float, float],
    fs_atlas: float,
    fs_ds: float,
    t_before: float,
    t_after: float,
    speed_thresh: float,
    robust_display_z: bool,
    min_epochs: int = 10,
):
    """
    Returns E1, E2, E3, plus metadata.
    Each E is (n_epochs, n_time), extracted from RAW signals,
    troughs defined on CA1_L theta-filtered reference.
    """
    df = load_synced_highrate(sync_folder)

    ca1l = detrend_nan(df["CA1_L"].to_numpy(float))
    ca1r = detrend_nan(df["CA1_R"].to_numpy(float))
    ca3l = detrend_nan(df["CA3_L"].to_numpy(float))
    spd  = df["speed_cm_s"].to_numpy(float) if "speed_cm_s" in df.columns else None

    ca1l_ds = downsample_poly(ca1l, fs_atlas, fs_ds)
    ca1r_ds = downsample_poly(ca1r, fs_atlas, fs_ds)
    ca3l_ds = downsample_poly(ca3l, fs_atlas, fs_ds)
    spd_ds  = downsample_poly(spd, fs_atlas, fs_ds) if spd is not None else None

    n = min(len(ca1l_ds), len(ca1r_ds), len(ca3l_ds), len(spd_ds) if spd_ds is not None else 10**12)
    ca1l_ds, ca1r_ds, ca3l_ds = ca1l_ds[:n], ca1r_ds[:n], ca3l_ds[:n]
    if spd_ds is not None:
        spd_ds = clean_speed_trace(spd_ds, max_speed=50.0)

    # Theta reference for trough detection
    ca1l_theta = butter_bandpass_filtfilt(ca1l_ds, fs=fs_ds, band=theta_band, order=4)
    trough_idx = find_theta_troughs(ca1l_theta, fs=fs_ds, theta_band=theta_band)

    pre  = int(round(t_before * fs_ds))
    post = int(round(t_after * fs_ds))
    L = pre + post + 1

    good = trough_idx[(trough_idx - pre >= 0) & (trough_idx + post < n)]

    # state filter on mean speed within window
    if spd_ds is not None and state != "all":
        keep = []
        for ti in good:
            mspd = np.nanmean(spd_ds[ti - pre : ti + post + 1])
            if state == "moving":
                keep.append(mspd > speed_thresh)
            else:
                keep.append(mspd <= speed_thresh)
        good = good[np.array(keep, dtype=bool)]

    if good.size < min_epochs:
        return None  # skip this folder

    def stack_epochs(x):
        return np.vstack([x[ti - pre : ti + post + 1] for ti in good])

    E1 = stack_epochs(ca1l_ds)
    E2 = stack_epochs(ca1r_ds)
    E3 = stack_epochs(ca3l_ds)

    if robust_display_z:
        E1 = robust_zscore(E1)
        E2 = robust_zscore(E2)
        E3 = robust_zscore(E3)

    return E1, E2, E3, {"n_epochs": int(good.size), "L": int(L)}


def plot_day_trough_triggered_raw_summary(
    day_dir: str | Path,
    heatmap_folder: str = "SyncRecording1",
    state: str = "all",
    theta_band: tuple[float, float] = (4.0, 12.0),
    fs_atlas: float = FS_ATLAS,
    fs_ds: float = FS_DS,
    t_before: float = 0.2,
    t_after: float = 0.2,
    speed_thresh: float = SPEED_THRESH,
    robust_display_z: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    - Top traces: pooled epochs from ALL SyncRecording* under day_dir
    - Heatmaps: epochs from ONE chosen folder (heatmap_folder)
    """
    day_dir = Path(day_dir)
    sync_folders = sorted([p for p in day_dir.glob("SyncRecording*") if p.is_dir()])
    if not sync_folders:
        raise FileNotFoundError(f"No SyncRecording* folders found under {day_dir}")

    # Pool epochs across all folders for smoother mean traces
    all_E1, all_E2, all_E3 = [], [], []
    used = 0

    for f in sync_folders:
        out = _extract_trough_epochs_from_folder(
            f, state=state, theta_band=theta_band,
            fs_atlas=fs_atlas, fs_ds=fs_ds,
            t_before=t_before, t_after=t_after,
            speed_thresh=speed_thresh,
            robust_display_z=robust_display_z,
            min_epochs=10,
        )
        if out is None:
            continue
        E1, E2, E3, meta = out
        all_E1.append(E1); all_E2.append(E2); all_E3.append(E3)
        used += 1

    if used == 0:
        raise ValueError(f"{day_dir.name}: no folders had enough trough epochs for state='{state}'")

    E1_pool = np.vstack(all_E1)
    E2_pool = np.vstack(all_E2)
    E3_pool = np.vstack(all_E3)

    # Heatmap epochs from the chosen folder
    hm_folder = day_dir / heatmap_folder
    if not hm_folder.exists():
        raise FileNotFoundError(f"Heatmap folder not found: {hm_folder}")

    hm_out = _extract_trough_epochs_from_folder(
        hm_folder, state=state, theta_band=theta_band,
        fs_atlas=fs_atlas, fs_ds=fs_ds,
        t_before=t_before, t_after=t_after,
        speed_thresh=speed_thresh,
        robust_display_z=robust_display_z,
        min_epochs=20,
    )
    if hm_out is None:
        raise ValueError(f"{heatmap_folder}: too few epochs for heatmap under state='{state}'")
    E1_hm, E2_hm, E3_hm, hm_meta = hm_out

    # Mean + CI from pooled epochs
    mu1, lo1, hi1 = mean_ci95(E1_pool)
    mu2, lo2, hi2 = mean_ci95(E2_pool)
    mu3, lo3, hi3 = mean_ci95(E3_pool)

    L = E1_pool.shape[1]
    t_rel = np.linspace(-t_before, t_after, L)

    # Heatmap colour limits based on the chosen heatmap folder (more consistent visually)
    all_vals = np.concatenate([E1_hm.ravel(), E2_hm.ravel(), E3_hm.ravel()])
    vmax = np.nanpercentile(np.abs(all_vals), 99)
    vmin = -vmax

    # ---- journal plot layout (same as single-folder) ----
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.0, 1.0], hspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    title_fs  = 18
    label_fs  = 16
    tick_fs   = 14
    legend_fs = 14
    cbar_fs   = 14

    # Region colour code (match your PSD figure)
    COL_CA1L = "#2ca02c"  # green
    COL_CA1R = "#d62728"  # red
    COL_CA3L = "#b8860b"  # yellow-brown
    
    # --- Top averaged traces (use your colour code) ---
    ax0.plot(t_rel, mu1, lw=2.5, color=COL_CA1L, label="CA1_L")
    ax0.fill_between(t_rel, lo1, hi1, color=COL_CA1L, alpha=0.18, linewidth=0)
    
    ax0.plot(t_rel, mu2, lw=2.5, color=COL_CA1R, label="CA1_R")
    ax0.fill_between(t_rel, lo2, hi2, color=COL_CA1R, alpha=0.18, linewidth=0)
    
    ax0.plot(t_rel, mu3, lw=2.5, color=COL_CA3L, label="CA3_L")
    ax0.fill_between(t_rel, lo3, hi3, color=COL_CA3L, alpha=0.18, linewidth=0)


    ax0.axvline(0.0, lw=1.6, alpha=0.55)
    ax0.set_ylabel("Raw (display scale)", fontsize=label_fs)
    ax0.set_title(
        f"theta cycle GEVI signals ({used} sweeps; epochs={E1_pool.shape[0]})|{state}",
        fontsize=title_fs
    )
    ax0.legend(loc="lower right", fontsize=legend_fs, frameon=False)

    im1 = ax1.imshow(E1_hm, aspect="auto", origin="lower",
                     extent=[t_rel[0], t_rel[-1], 0, E1_hm.shape[0]],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax1.set_ylabel("Epoch #", fontsize=label_fs)
    ax1.set_title("CA1_L", fontsize=label_fs, loc="left")
    ax1.axvline(0.0, lw=1.4, alpha=0.55)

    im2 = ax2.imshow(E2_hm, aspect="auto", origin="lower",
                     extent=[t_rel[0], t_rel[-1], 0, E2_hm.shape[0]],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax2.set_ylabel("Epoch #", fontsize=label_fs)
    ax2.set_title("CA1_R", fontsize=label_fs, loc="left")
    ax2.axvline(0.0, lw=1.4, alpha=0.55)

    im3 = ax3.imshow(E3_hm, aspect="auto", origin="lower",
                     extent=[t_rel[0], t_rel[-1], 0, E3_hm.shape[0]],
                     vmin=vmin, vmax=vmax, cmap="viridis")
    ax3.set_ylabel("Epoch #", fontsize=label_fs)
    ax3.set_title("CA3_L", fontsize=label_fs, loc="left")
    ax3.axvline(0.0, lw=1.4, alpha=0.55)
    ax3.set_xlabel("Time relative to CA1_L theta trough (s)", fontsize=label_fs)

    for ax in (ax0, ax1, ax2, ax3):
        _despine_all(ax)
        _style_ticks(ax, tick_fs=tick_fs)

    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3], orientation="horizontal",
                        fraction=0.06, pad=0.15)
    cbar.set_label("Raw value (display scale)", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=cbar_fs)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return save_path

    if show:
        plt.show()
        return fig
    plt.close(fig)
    return None

# -------------------------
# Batch helpers
# -------------------------
def run_one_folder(parent_folder: str | Path,folder: str | Path, start_s: float, dur_s: float = 3.0):
    folder = Path(folder)

    'theta cycle (filtered) plot'
    # p2_all = theta_cycle_average_two_cycles_with_heatmaps(folder, state="all",
    # save_name="theta_cycle_average_all.png")
    # p2_mv = theta_cycle_average_two_cycles_with_heatmaps(folder, state="moving",
    # save_name="theta_cycle_average_moving.png")
    # 'theta cycle (raw signals) plot'
    # plot_theta_trough_triggered_raw_average(folder,state="all",
    #                                         t_before=0.2,t_after=0.2,show=True,save_path=None)
    # plot_theta_trough_triggered_raw_average(folder, state="moving", t_before=0.2,t_after=0.2)
    
    plot_day_trough_triggered_raw_summary(
    parent_folder,
    heatmap_folder="SyncRecording1",
    state="all",
    t_before=0.15, t_after=0.15,
    show=True,
    save_path=None
    )

    return -1


if __name__ == "__main__":
    # Example usage:
    'plot theta averages and heatmaps'
    parent_folder=r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1966293_Jedi2p_Multi\ALocomotion"
    folder=r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1966293_Jedi2p_Multi\ALocomotion\SyncRecording1"
    run_one_folder(parent_folder,folder, start_s=12.0, dur_s=3.0)

    pass
