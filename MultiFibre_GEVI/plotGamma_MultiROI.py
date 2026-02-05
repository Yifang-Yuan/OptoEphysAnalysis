# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 16:55:34 2026

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Gamma trough-triggered multi-ROI summary (day-level)

- Aligns to CA1_L GAMMA troughs (slow or fast gamma band).
- Pools epochs across ALL SyncRecording* for top mean trace.
- Heatmaps are shown for ONE chosen SyncRecording folder (heatmap_folder).
- Optional state gating by speed: all / moving / not_moving.
- Uses region colours matching your PSD figure.

Tested design goals:
- Fast gamma (65–100 Hz) requires fs_band >= 2*100 Hz. Default fs_band=500 Hz.
- Shorter windows for gamma: default ~2–4 cycles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from fractions import Fraction
from scipy import signal
from typing import Tuple, Optional, Union, List, Dict


# -------------------------
# Config
# -------------------------
FS_ATLAS = 1682.92

# Default sampling rate used for gamma extraction / trough detection.
# Must satisfy fs_band > 2*highcut (Nyquist). 500 is safe for 100 Hz.
FS_BAND_DEFAULT = 500.0

# Bands
SLOW_GAMMA_BAND = (30.0, 55.0)
FAST_GAMMA_BAND = (65.0, 100.0)

# Speed gating
SPEED_THRESH = 3.0  # cm/s

# Region colour code (match your PSD figure)
COL_CA1L = "#2ca02c"  # green
COL_CA1R = "#d62728"  # red
COL_CA3L = "#b8860b"  # yellow-brown


# -------------------------
# I/O
# -------------------------
def load_synced_highrate(folder: Union[str, Path]) -> pd.DataFrame:
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
    raise FileNotFoundError(
        f"No synced_optical_behaviour_highrate.(parquet|pkl.gz|pkl) in {folder}"
    )


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
    x = np.asarray(x, float)
    if abs(fs_in - fs_out) < 1e-9:
        return x
    frac = Fraction(fs_out / fs_in).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    return signal.resample_poly(x, up=up, down=down)


def butter_bandpass_filtfilt(x: np.ndarray,
                             fs: float,
                             band: Tuple[float, float],
                             order: int = 4) -> np.ndarray:
    lo, hi = float(band[0]), float(band[1])
    if hi <= lo:
        raise ValueError(f"Invalid band: {band}. Need (lo, hi) with hi > lo.")
    nyq = 0.5 * fs
    if hi >= nyq:
        raise ValueError(
            f"Band highcut {hi} Hz exceeds Nyquist {nyq:.1f} Hz. "
            f"Increase fs (current fs={fs}) or lower band."
        )
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return signal.filtfilt(b, a, np.asarray(x, float))


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-12:
        return x - med
    return (x - med) / (1.4826 * mad)


def clean_speed_trace(speed: Optional[np.ndarray],
                      max_speed: float = 50.0) -> Optional[np.ndarray]:
    if speed is None:
        return None
    s = np.asarray(speed, float).copy()
    bad = ~np.isfinite(s) | (s > max_speed)
    if bad.any():
        s[bad] = np.nan
        x = np.arange(len(s))
        good = np.isfinite(s)
        if good.sum() >= 2:
            s[~good] = np.interp(x[~good], x[good], s[good])
        elif good.sum() == 1:
            s[~good] = s[good][0]
        else:
            s[:] = 0.0
    return s


def mean_ci95(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mat: (n_epochs, n_time)
    returns mean, ci_low, ci_high (95% normal approx)
    """
    mat = np.asarray(mat, float)
    mu = np.nanmean(mat, axis=0)
    n_eff = np.sum(np.isfinite(mat), axis=0).astype(float)
    n_eff[n_eff < 1] = 1.0
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n_eff)
    ci = 1.96 * sem
    return mu, mu - ci, mu + ci


def find_band_troughs(x_band: np.ndarray,
                      fs: float,
                      band: Tuple[float, float],
                      min_prominence: Optional[float] = None,
                      min_dist_factor: float = 0.7) -> np.ndarray:
    """
    Find troughs on a band-passed signal using peaks on inverted signal.

    min distance between troughs:
      ~ min_dist_factor * shortest period = min_dist_factor * fs / band_high
    """
    x_band = np.asarray(x_band, float)
    inv = -x_band

    min_dist = int(round((fs / band[1]) * min_dist_factor))
    min_dist = max(min_dist, 1)

    if min_prominence is None:
        med = np.nanmedian(inv)
        mad = np.nanmedian(np.abs(inv - med)) + 1e-12
        min_prominence = 0.5 * (1.4826 * mad)

    trough_idx, _ = signal.find_peaks(inv, distance=min_dist, prominence=min_prominence)
    return trough_idx


# -------------------------
# Plot styling
# -------------------------
def _despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)

def _style_ticks(ax, tick_fs=14):
    ax.tick_params(axis="both", which="both", labelsize=tick_fs, length=4, width=1)


# -------------------------
# Core extraction per SyncRecording folder (gamma trough aligned)
# -------------------------
def _extract_gamma_trough_epochs_from_folder(
    sync_folder: Union[str, Path],
    state: str,
    gamma_band: Tuple[float, float],
    fs_atlas: float,
    fs_band: float,
    t_before: float,
    t_after: float,
    speed_thresh: float,
    robust_display_z: bool,
    min_epochs: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
    """
    Returns (E1, E2, E3, meta) or None if too few epochs.
      - E1/E2/E3 are GAMMA-band epochs aligned to CA1_L gamma troughs
      - state filter uses mean speed within the epoch window (at fs_band)
    """
    sync_folder = Path(sync_folder)
    df = load_synced_highrate(sync_folder)

    # --- load & detrend ---
    ca1l = detrend_nan(df["CA1_L"].to_numpy(float))
    ca1r = detrend_nan(df["CA1_R"].to_numpy(float))
    ca3l = detrend_nan(df["CA3_L"].to_numpy(float))

    # speed name variants
    if "speed_cm_s" in df.columns:
        spd = df["speed_cm_s"].to_numpy(float)
    elif "speed" in df.columns:
        spd = df["speed"].to_numpy(float)
    else:
        spd = None

    # --- downsample to fs_band (for gamma + trough timing) ---
    ca1l_ds = downsample_poly(ca1l, fs_atlas, fs_band)
    ca1r_ds = downsample_poly(ca1r, fs_atlas, fs_band)
    ca3l_ds = downsample_poly(ca3l, fs_atlas, fs_band)
    spd_ds  = downsample_poly(spd,  fs_atlas, fs_band) if spd is not None else None

    n = min(len(ca1l_ds), len(ca1r_ds), len(ca3l_ds),
            len(spd_ds) if spd_ds is not None else 10**12)

    ca1l_ds, ca1r_ds, ca3l_ds = ca1l_ds[:n], ca1r_ds[:n], ca3l_ds[:n]
    if spd_ds is not None:
        spd_ds = spd_ds[:n]
        spd_ds = clean_speed_trace(spd_ds, max_speed=50.0)

    # --- gamma bandpass (for alignment + epochs) ---
    ca1l_g = butter_bandpass_filtfilt(ca1l_ds, fs=fs_band, band=gamma_band, order=4)
    ca1r_g = butter_bandpass_filtfilt(ca1r_ds, fs=fs_band, band=gamma_band, order=4)
    ca3l_g = butter_bandpass_filtfilt(ca3l_ds, fs=fs_band, band=gamma_band, order=4)

    # --- troughs on CA1_L gamma ---
    trough_idx = find_band_troughs(ca1l_g, fs=fs_band, band=gamma_band)

    pre  = int(round(t_before * fs_band))
    post = int(round(t_after  * fs_band))
    L = pre + post + 1

    good = trough_idx[(trough_idx - pre >= 0) & (trough_idx + post < n)]

    # --- state filter on mean speed within window ---
    if spd_ds is not None and state != "all":
        keep = []
        for ti in good:
            mspd = float(np.nanmean(spd_ds[ti - pre: ti + post + 1]))
            if state == "moving":
                keep.append(mspd > speed_thresh)
            else:
                keep.append(mspd <= speed_thresh)
        good = good[np.array(keep, dtype=bool)]

    if good.size < min_epochs:
        return None

    # --- extract epochs (gamma-band) ---
    def stack_epochs(x):
        return np.vstack([x[ti - pre: ti + post + 1] for ti in good])

    E1 = stack_epochs(ca1l_g)
    E2 = stack_epochs(ca1r_g)
    E3 = stack_epochs(ca3l_g)

    # optional display scaling (robust z) on EPOCH matrices
    if robust_display_z:
        E1 = robust_zscore(E1)
        E2 = robust_zscore(E2)
        E3 = robust_zscore(E3)

    return E1, E2, E3, {"n_epochs": int(good.size), "L": int(L), "fs_band": float(fs_band)}


# -------------------------
# Day-level pooled plot (gamma trough aligned)
# -------------------------
def plot_day_gamma_trough_triggered_summary(
    day_dir: Union[str, Path],
    gamma_band: Tuple[float, float],
    heatmap_folder: str = "SyncRecording1",
    state: str = "all",
    fs_atlas: float = FS_ATLAS,
    fs_band: float = FS_BAND_DEFAULT,
    t_before: Optional[float] = None,
    t_after: Optional[float] = None,
    speed_thresh: float = SPEED_THRESH,
    robust_display_z: bool = True,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    - Top traces: pooled GAMMA-band epochs from ALL SyncRecording* under day_dir
    - Heatmaps: epochs from ONE chosen folder (heatmap_folder)
    - Alignment: CA1_L GAMMA troughs (within gamma_band)
    """
    day_dir = Path(day_dir)

    # default shorter windows for gamma cycles
    if t_before is None or t_after is None:
        if gamma_band[1] <= 60.0:
            # slow gamma: 30–55 Hz -> 2–4 cycles in ~120 ms window total
            t_before = 0.04 if t_before is None else t_before
            t_after  = 0.04 if t_after  is None else t_after
        else:
            # fast gamma: 65–100 Hz -> 2–4 cycles in ~80 ms window total
            t_before = 0.02 if t_before is None else t_before
            t_after  = 0.02 if t_after  is None else t_after

    # sanity: fs_band must support gamma_band
    if fs_band <= 2.0 * gamma_band[1]:
        raise ValueError(
            f"fs_band={fs_band} too low for gamma up to {gamma_band[1]} Hz. "
            f"Use fs_band >= {2*gamma_band[1]:.1f} Hz (e.g. 400–600)."
        )

    sync_folders = sorted([p for p in day_dir.glob("SyncRecording*") if p.is_dir()])
    if not sync_folders:
        raise FileNotFoundError(f"No SyncRecording* folders found under {day_dir}")

    # Pool epochs across all folders
    all_E1, all_E2, all_E3 = [], [], []
    used = 0

    for f in sync_folders:
        out = _extract_gamma_trough_epochs_from_folder(
            f, state=state, gamma_band=gamma_band,
            fs_atlas=fs_atlas, fs_band=fs_band,
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
        raise ValueError(
            f"{day_dir.name}: no folders had enough gamma trough epochs for state='{state}' "
            f"(gamma {gamma_band[0]}–{gamma_band[1]} Hz)."
        )

    E1_pool = np.vstack(all_E1)
    E2_pool = np.vstack(all_E2)
    E3_pool = np.vstack(all_E3)

    # Heatmap epochs from chosen folder
    hm_folder = day_dir / heatmap_folder
    if not hm_folder.exists():
        raise FileNotFoundError(f"Heatmap folder not found: {hm_folder}")

    hm_out = _extract_gamma_trough_epochs_from_folder(
        hm_folder, state=state, gamma_band=gamma_band,
        fs_atlas=fs_atlas, fs_band=fs_band,
        t_before=t_before, t_after=t_after,
        speed_thresh=speed_thresh,
        robust_display_z=robust_display_z,
        min_epochs=20,
    )
    if hm_out is None:
        raise ValueError(
            f"{heatmap_folder}: too few epochs for heatmap under state='{state}' "
            f"(gamma {gamma_band[0]}–{gamma_band[1]} Hz)."
        )
    E1_hm, E2_hm, E3_hm, hm_meta = hm_out

    # Mean + CI from pooled epochs
    mu1, lo1, hi1 = mean_ci95(E1_pool)
    mu2, lo2, hi2 = mean_ci95(E2_pool)
    mu3, lo3, hi3 = mean_ci95(E3_pool)

    L = E1_pool.shape[1]
    t_rel = np.linspace(-t_before, t_after, L)

    # Heatmap colour limits based on chosen folder
    all_vals = np.concatenate([E1_hm.ravel(), E2_hm.ravel(), E3_hm.ravel()])
    vmax = np.nanpercentile(np.abs(all_vals), 99)
    vmin = -vmax

    # ---- plot layout ----
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

    # Top mean traces (region colours)
    ax0.plot(t_rel, mu1, lw=2.5, color=COL_CA1L, label="CA1_L")
    ax0.fill_between(t_rel, lo1, hi1, color=COL_CA1L, alpha=0.18, linewidth=0)

    ax0.plot(t_rel, mu2, lw=2.5, color=COL_CA1R, label="CA1_R")
    ax0.fill_between(t_rel, lo2, hi2, color=COL_CA1R, alpha=0.18, linewidth=0)

    ax0.plot(t_rel, mu3, lw=2.5, color=COL_CA3L, label="CA3_L")
    ax0.fill_between(t_rel, lo3, hi3, color=COL_CA3L, alpha=0.18, linewidth=0)

    ax0.axvline(0.0, lw=1.6, alpha=0.55)
    ax0.set_ylabel("zscore (a.u)", fontsize=label_fs)
    ax0.set_title(
        f"Gamma cycles (CA1_L ref) | {gamma_band[0]:.0f}–{gamma_band[1]:.0f} Hz "
        f"| {used} sweeps; epochs={E1_pool.shape[0]}",
        fontsize=title_fs
    )
    ax0.legend(loc="upper right", fontsize=legend_fs, frameon=False)

    # Heatmaps (one folder)
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
    ax3.set_xlabel("Time relative to CA1_L gamma trough (s)", fontsize=label_fs)

    for ax in (ax0, ax1, ax2, ax3):
        _despine_all(ax)
        _style_ticks(ax, tick_fs=tick_fs)

    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)

    cbar = fig.colorbar(im3, ax=[ax1, ax2, ax3], orientation="horizontal",
                        fraction=0.06, pad=0.15)
    cbar.set_label("Gamma-band value (display scale)", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=cbar_fs)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return save_path

    if show:
        plt.show()
        return fig

    plt.close(fig)
    return None


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    day_dir = r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1966293_Jedi2p_Multi\ALocomotion"

    # Slow gamma (30–55 Hz), gamma-trough aligned
    plot_day_gamma_trough_triggered_summary(
        day_dir=day_dir,
        gamma_band=SLOW_GAMMA_BAND,
        heatmap_folder="SyncRecording2",
        state="moving",
        fs_band=500.0,
        # t_before/t_after left None -> auto defaults for slow gamma
        show=True,
        save_path=None,
        # save_path=r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\gamma_slow_trough_summary.png",
    )

    # Fast gamma (65–100 Hz), gamma-trough aligned
    plot_day_gamma_trough_triggered_summary(
        day_dir=day_dir,
        gamma_band=FAST_GAMMA_BAND,
        heatmap_folder="SyncRecording2",
        state="moving",
        fs_band=500.0,
        # t_before/t_after left None -> auto defaults for fast gamma
        show=True,
        save_path=None,
        # save_path=r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\gamma_fast_trough_summary.png",
    )
