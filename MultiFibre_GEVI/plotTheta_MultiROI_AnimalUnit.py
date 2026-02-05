# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:07:55 2026

@author: yifan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from itertools import product
from fractions import Fraction

from typing import Tuple, List, Optional, Union, Dict, Any


# -------------------------
# Config
# -------------------------
FS_ATLAS = 1682.92
FS_DS = 200.0

# Region colour code (match your PSD figure)
COL_CA1L = "#2ca02c" # green
COL_CA1R = "#d62728"  # red
COL_CA3L = "#b8860b"   # yellow browhn

# Pair colours (for xcorr + phase plots)
COL_IPSI  = COL_CA3L   # CA1_L vs CA3_L
COL_BILAT = COL_CA1R   # CA1_L vs CA1_R


# -------------------------
# I/O + preprocessing
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
    raise FileNotFoundError(f"No synced_optical_behaviour_highrate.(parquet|pkl.gz|pkl) in {folder}")


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
    if abs(fs_in - fs_out) < 1e-12:
        return x
    frac = Fraction(fs_out / fs_in).limit_denominator(2000)
    return signal.resample_poly(x, up=frac.numerator, down=frac.denominator)


def butter_bandpass_filtfilt(x: np.ndarray, fs: float, band: Tuple[float, float], order: int = 4) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return signal.filtfilt(b, a, np.asarray(x, float))


def find_theta_troughs(theta_ref: np.ndarray,
                       fs: float,
                       theta_band: Tuple[float, float] = (6.0, 10.0),
                       min_prominence: Optional[float] = None) -> np.ndarray:
    """
    Find trough indices on a theta-band signal using find_peaks on the inverted signal.
    """
    min_dist = int(round(fs / theta_band[1] * 0.7))
    inv = -np.asarray(theta_ref, float)

    if min_prominence is None:
        med = np.nanmedian(inv)
        mad = np.nanmedian(np.abs(inv - med)) + 1e-12
        min_prominence = 0.5 * (1.4826 * mad)

    trough_idx, _ = signal.find_peaks(inv, distance=min_dist, prominence=min_prominence)
    return trough_idx


# -------------------------
# Circular helpers
# -------------------------
def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi]."""
    x = np.asarray(x, float)
    return (x + np.pi) % (2 * np.pi) - np.pi


def circ_mean(alpha: np.ndarray) -> float:
    """Circular mean angle in [-pi, pi]."""
    alpha = np.asarray(alpha, float)
    alpha = alpha[np.isfinite(alpha)]
    if alpha.size == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * alpha))))


def exact_paired_circular_signflip(angles_rad: np.ndarray) -> float:
    """
    Exact paired circular sign-flip test (two-sided) for mean angle != 0.
    angles_rad: per-animal angles (radians).
    Test statistic: |circ_mean(angles)|.
    """
    a = np.asarray(angles_rad, float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return np.nan

    obs = abs(circ_mean(a))
    stats = []
    for signs in product([1.0, -1.0], repeat=n):
        s = np.asarray(signs, float)
        stats.append(abs(circ_mean(s * a)))
    stats = np.asarray(stats)
    return float(np.mean(stats >= (obs - 1e-12)))


# -------------------------
# Plot styling helpers
# -------------------------
def _despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)


def _style_ticks(ax, tick_fs: int = 12):
    ax.tick_params(axis="both", which="both", labelsize=tick_fs, length=4, width=1)


def _set_pi_ticks(ax):
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


# -------------------------
# Core extraction per SyncRecording folder
# -------------------------
def _extract_trough_epochs_and_theta(
    sync_folder: Path,
    theta_band: Tuple[float, float],
    fs_atlas: float,
    fs_ds: float,
    t_before: float,
    t_after: float,
    min_epochs: int = 20,
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with:
      E_raw: raw epochs (n_epochs, n_t) for CA1_L/CA1_R/CA3_L (detrended+downsampled)
      E_th : theta-band epochs (same shape)
      phase_offsets: per-epoch phase offset at trough time (radians), pairs:
          'CA1L_CA3L' and 'CA1L_CA1R'  (phi_other - phi_ref at trough index)
      meta
    """
    df = load_synced_highrate(sync_folder)

    # raw detrend
    ca1l_raw = detrend_nan(df["CA1_L"].to_numpy(float))
    ca1r_raw = detrend_nan(df["CA1_R"].to_numpy(float))
    ca3l_raw = detrend_nan(df["CA3_L"].to_numpy(float))

    # downsample
    ca1l = downsample_poly(ca1l_raw, fs_atlas, fs_ds)
    ca1r = downsample_poly(ca1r_raw, fs_atlas, fs_ds)
    ca3l = downsample_poly(ca3l_raw, fs_atlas, fs_ds)

    n = min(len(ca1l), len(ca1r), len(ca3l))
    ca1l, ca1r, ca3l = ca1l[:n], ca1r[:n], ca3l[:n]

    # theta band
    ca1l_th = butter_bandpass_filtfilt(ca1l, fs=fs_ds, band=theta_band, order=4)
    ca1r_th = butter_bandpass_filtfilt(ca1r, fs=fs_ds, band=theta_band, order=4)
    ca3l_th = butter_bandpass_filtfilt(ca3l, fs=fs_ds, band=theta_band, order=4)

    trough_idx = find_theta_troughs(ca1l_th, fs=fs_ds, theta_band=theta_band)

    pre = int(round(t_before * fs_ds))
    post = int(round(t_after * fs_ds))
    L = pre + post + 1

    good = trough_idx[(trough_idx - pre >= 0) & (trough_idx + post < n)]
    if good.size < min_epochs:
        return None

    def stack_epochs(x):
        return np.vstack([x[ti - pre: ti + post + 1] for ti in good])

    E_raw = {
        "CA1_L": stack_epochs(ca1l),
        "CA1_R": stack_epochs(ca1r),
        "CA3_L": stack_epochs(ca3l),
    }
    E_th = {
        "CA1_L": stack_epochs(ca1l_th),
        "CA1_R": stack_epochs(ca1r_th),
        "CA3_L": stack_epochs(ca3l_th),
    }

    # phase offsets at trough sample (relative to CA1_L)
    phi1 = np.angle(signal.hilbert(ca1l_th))
    phi2 = np.angle(signal.hilbert(ca1r_th))
    phi3 = np.angle(signal.hilbert(ca3l_th))

    off_13 = wrap_pi(phi3[good] - phi1[good])  # CA3L - CA1L
    off_12 = wrap_pi(phi2[good] - phi1[good])  # CA1R - CA1L

    return {
        "E_raw": E_raw,
        "E_th": E_th,
        "phase_offsets": {
            "CA1L_CA3L": off_13,
            "CA1L_CA1R": off_12,
        },
        "meta": {
            "n_epochs": int(good.size),
            "L": int(L),
        }
    }


# -------------------------
# Cross-correlation per epoch (theta-band)
# -------------------------
def _epoch_xcorr_curve(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: float):
    """
    Normalised cross-correlation curve for one epoch.
    correlate(y, x): positive lag => y is shifted later relative to x (y lags x).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    x = x / (np.nanstd(x) + 1e-12)
    y = y / (np.nanstd(y) + 1e-12)

    r_full = signal.correlate(y, x, mode="full") / len(x)

    # robust lag vector (avoid SciPy version issues)
    n = len(x)
    lags = np.arange(-(n - 1), n) / fs

    m = np.abs(lags) <= max_lag_s
    return lags[m], r_full[m]


# -------------------------
# Animal-level pooling
# -------------------------
def _list_syncrecordings(root: Path) -> List[Path]:
    hits = sorted([p for p in root.glob("SyncRecording*") if p.is_dir()])
    if hits:
        return hits
    return sorted([p for p in root.rglob("SyncRecording*") if p.is_dir()])


def _infer_animal_id(aloco_path: Path) -> str:
    if aloco_path.name.lower() == "alocomotion":
        return aloco_path.parent.name
    return aloco_path.name


def compute_animal_theta_summary(
    aloco_path: Union[str, Path],
    theta_band: Tuple[float, float] = (6.0, 10.0),
    fs_atlas: float = FS_ATLAS,
    fs_ds: float = FS_DS,
    t_before: float = 0.2,
    t_after: float = 0.2,
    max_lag_s: float = 0.2,
    min_epochs_per_sweep: int = 20,
) -> Dict[str, Any]:
    """
    For one animal (locomotion only):
      - Pool trough-triggered epochs across all SyncRecording*
      - Animal-level mean raw waveforms (CA1_L, CA1_R, CA3_L)
      - Animal-level mean theta-band xcorr curves for two pairs
      - Phase offset samples + animal-level circular mean offsets
    """
    aloco_path = Path(aloco_path)
    animal_id = _infer_animal_id(aloco_path)

    sync_folders = _list_syncrecordings(aloco_path)
    if not sync_folders:
        raise FileNotFoundError(f"No SyncRecording* folders found under: {aloco_path}")

    raw_pool = {k: [] for k in ["CA1_L", "CA1_R", "CA3_L"]}
    th_pool  = {k: [] for k in ["CA1_L", "CA1_R", "CA3_L"]}

    off_13_all = []
    off_12_all = []

    lags_ref = None
    sum_xcorr_13 = None
    sum_xcorr_12 = None
    n_xcorr = 0

    used_sweeps = 0

    for f in sync_folders:
        out = _extract_trough_epochs_and_theta(
            f, theta_band=theta_band, fs_atlas=fs_atlas, fs_ds=fs_ds,
            t_before=t_before, t_after=t_after, min_epochs=min_epochs_per_sweep
        )
        if out is None:
            continue

        used_sweeps += 1
        n_ep = out["meta"]["n_epochs"]

        Eraw = out["E_raw"]
        Eth  = out["E_th"]

        for k in raw_pool:
            raw_pool[k].append(Eraw[k])
            th_pool[k].append(Eth[k])

        off_13_all.append(out["phase_offsets"]["CA1L_CA3L"])
        off_12_all.append(out["phase_offsets"]["CA1L_CA1R"])

        # xcorr per epoch (pooled within animal)
        for i in range(n_ep):
            x = Eth["CA1_L"][i]
            y13 = Eth["CA3_L"][i]
            y12 = Eth["CA1_R"][i]

            lags, r13 = _epoch_xcorr_curve(x, y13, fs=fs_ds, max_lag_s=max_lag_s)
            _,    r12 = _epoch_xcorr_curve(x, y12, fs=fs_ds, max_lag_s=max_lag_s)

            if lags_ref is None:
                lags_ref = lags
                sum_xcorr_13 = np.zeros_like(r13, float)
                sum_xcorr_12 = np.zeros_like(r12, float)

            if lags.shape != lags_ref.shape or np.max(np.abs(lags - lags_ref)) > 1e-9:
                r13 = np.interp(lags_ref, lags, r13)
                r12 = np.interp(lags_ref, lags, r12)

            sum_xcorr_13 += r13
            sum_xcorr_12 += r12
            n_xcorr += 1

    if used_sweeps == 0:
        raise ValueError(f"{animal_id}: no SyncRecording had >= {min_epochs_per_sweep} trough epochs.")

    Eraw_all = {k: np.vstack(raw_pool[k]) for k in raw_pool}
    Eth_all  = {k: np.vstack(th_pool[k])  for k in th_pool}

    off_13_all = np.concatenate(off_13_all) if off_13_all else np.array([], float)
    off_12_all = np.concatenate(off_12_all) if off_12_all else np.array([], float)

    mean_raw = {k: np.nanmean(Eraw_all[k], axis=0) for k in Eraw_all}

    xcorr_13 = sum_xcorr_13 / max(n_xcorr, 1)
    xcorr_12 = sum_xcorr_12 / max(n_xcorr, 1)

    mu_13 = circ_mean(off_13_all)
    mu_12 = circ_mean(off_12_all)

    def peak_lag(lags_s, r):
        i = int(np.nanargmax(r))
        return float(lags_s[i]), float(r[i])

    lag13, rpk13 = peak_lag(lags_ref, xcorr_13)
    lag12, rpk12 = peak_lag(lags_ref, xcorr_12)

    r0_13 = float(np.interp(0.0, lags_ref, xcorr_13))
    r0_12 = float(np.interp(0.0, lags_ref, xcorr_12))

    return {
        "animal_id": animal_id,
        "used_sweeps": int(used_sweeps),
        "n_epochs_total": int(Eraw_all["CA1_L"].shape[0]),
        "t_rel": np.linspace(-t_before, t_after, Eraw_all["CA1_L"].shape[1]),
        "mean_raw": mean_raw,
        "lags_s": lags_ref,
        "xcorr": {
            "CA1L_CA3L": xcorr_13,
            "CA1L_CA1R": xcorr_12,
        },
        "phase_offsets": {
            "CA1L_CA3L": off_13_all,
            "CA1L_CA1R": off_12_all,
        },
        "phase_mean_rad": {
            "CA1L_CA3L": mu_13,
            "CA1L_CA1R": mu_12,
        },
        "xcorr_metrics": {
            "r0_CA1L_CA3L": r0_13,
            "r0_CA1L_CA1R": r0_12,
            "lagpk_s_CA1L_CA3L": lag13,
            "rpk_CA1L_CA3L": rpk13,
            "lagpk_s_CA1L_CA1R": lag12,
            "rpk_CA1L_CA1R": rpk12,
        }
    }


# -------------------------
# Group-level plot + stats
# -------------------------
def plot_group_theta_relationships(
    aloco_paths: List[str],
    theta_band: Tuple[float, float] = (6.0, 10.0),
    fs_atlas: float = FS_ATLAS,
    fs_ds: float = FS_DS,
    t_before: float = 0.2,
    t_after: float = 0.2,
    max_lag_s: float = 0.2,
    min_epochs_per_sweep: int = 20,
    n_phase_bins: int = 36,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Figure:
      A) group mean trough-triggered RAW waveforms (mean ± SEM across animals)
      B) group mean theta xcorr curves for pairs (mean ± SEM across animals)
      C) group mean phase-shift density for pairs (mean ± SEM across animals)
      D) per-animal circular mean phase offsets (paired) + signflip p-values
    """
    summaries = []
    for p in aloco_paths:
        summaries.append(compute_animal_theta_summary(
            p, theta_band=theta_band, fs_atlas=fs_atlas, fs_ds=fs_ds,
            t_before=t_before, t_after=t_after, max_lag_s=max_lag_s,
            min_epochs_per_sweep=min_epochs_per_sweep
        ))

    # per-animal table
    rows = []
    for s in summaries:
        rows.append({
            "animal_id": s["animal_id"],
            "used_sweeps": s["used_sweeps"],
            "n_epochs_total": s["n_epochs_total"],
            "phase_mean_CA1L_CA3L_rad": s["phase_mean_rad"]["CA1L_CA3L"],
            "phase_mean_CA1L_CA1R_rad": s["phase_mean_rad"]["CA1L_CA1R"],
            "phase_mean_CA1L_CA3L_deg": np.degrees(s["phase_mean_rad"]["CA1L_CA3L"]),
            "phase_mean_CA1L_CA1R_deg": np.degrees(s["phase_mean_rad"]["CA1L_CA1R"]),
            **s["xcorr_metrics"],
        })
    df_anim = pd.DataFrame(rows)

    # group mean raw waveforms
    t_rel = summaries[0]["t_rel"]
    M_ca1l = np.vstack([s["mean_raw"]["CA1_L"] for s in summaries])
    M_ca1r = np.vstack([s["mean_raw"]["CA1_R"] for s in summaries])
    M_ca3l = np.vstack([s["mean_raw"]["CA3_L"] for s in summaries])

    def mean_sem(M):
        mu = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])
        return mu, se

    mu1, se1 = mean_sem(M_ca1l)
    mu2, se2 = mean_sem(M_ca1r)
    mu3, se3 = mean_sem(M_ca3l)

    # group mean xcorr
    lags = summaries[0]["lags_s"]
    X_13 = np.vstack([s["xcorr"]["CA1L_CA3L"] for s in summaries])
    X_12 = np.vstack([s["xcorr"]["CA1L_CA1R"] for s in summaries])
    xmu13, xse13 = mean_sem(X_13)
    xmu12, xse12 = mean_sem(X_12)

    # phase density
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def density_per_animal(offsets):
        h, _ = np.histogram(offsets, bins=edges, density=True)
        return h

    D13 = np.vstack([density_per_animal(s["phase_offsets"]["CA1L_CA3L"]) for s in summaries])
    D12 = np.vstack([density_per_animal(s["phase_offsets"]["CA1L_CA1R"]) for s in summaries])
    dmu13, dse13 = mean_sem(D13)
    dmu12, dse12 = mean_sem(D12)

    # stats on phase shift across animals (animal is unit)
    a13 = wrap_pi(df_anim["phase_mean_CA1L_CA3L_rad"].to_numpy(float))
    a12 = wrap_pi(df_anim["phase_mean_CA1L_CA1R_rad"].to_numpy(float))
    diff = wrap_pi(a13 - a12)  # ipsi - bilat

    p_13_vs0 = exact_paired_circular_signflip(a13)
    p_12_vs0 = exact_paired_circular_signflip(a12)
    p_diff = exact_paired_circular_signflip(diff)

    group_stats = {
        "p_phase_CA1L_CA3L_vs0": p_13_vs0,
        "p_phase_CA1L_CA1R_vs0": p_12_vs0,
        "p_phase_ipsi_minus_bilat": p_diff,
        "mean_phase_CA1L_CA3L_deg": float(np.degrees(circ_mean(a13))),
        "mean_phase_CA1L_CA1R_deg": float(np.degrees(circ_mean(a12))),
        "mean_phase_diff_deg": float(np.degrees(circ_mean(diff))),
        "n_animals": int(len(summaries)),
    }

    # -------------------------
    # Plot
    # -------------------------
    title_fs = 14
    label_fs = 13
    tick_fs = 12
    legend_fs = 11

    fig = plt.figure(figsize=(12.0, 7.8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.0, 1.0],
        hspace=0.40,
        wspace=0.32
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # A) group mean raw waveforms (region colours)
    axA.plot(t_rel, mu1, lw=2.6, color=COL_CA1L, label="CA1_L (ref)")
    axA.fill_between(t_rel, mu1 - se1, mu1 + se1, color=COL_CA1L, alpha=0.18)

    axA.plot(t_rel, mu2, lw=2.6, color=COL_CA1R, label="CA1_R")
    axA.fill_between(t_rel, mu2 - se2, mu2 + se2, color=COL_CA1R, alpha=0.18)

    axA.plot(t_rel, mu3, lw=2.6, color=COL_CA3L, label="CA3_L")
    axA.fill_between(t_rel, mu3 - se3, mu3 + se3, color=COL_CA3L, alpha=0.18)

    axA.axvline(0, lw=1.4, alpha=0.55)
    axA.set_title("Mean θ waveforms (0 = CA1_L trough)", fontsize=title_fs)
    axA.set_xlabel("Time relative to CA1_L trough (s)", fontsize=label_fs)
    axA.set_ylabel("Raw (a.u.; detrended, downsampled)", fontsize=label_fs)
    axA.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    # B) xcorr curves (pair colours)
    axB.plot(lags, xmu13, lw=2.6, color=COL_IPSI, label="CA1_L vs CA3_L (ipsi)")
    axB.fill_between(lags, xmu13 - xse13, xmu13 + xse13, color=COL_IPSI, alpha=0.18)

    axB.plot(lags, xmu12, lw=2.6, color=COL_BILAT, label="CA1_L vs CA1_R (bilat)")
    axB.fill_between(lags, xmu12 - xse12, xmu12 + xse12, color=COL_BILAT, alpha=0.18)

    axB.axvline(0, lw=1.4, alpha=0.55)
    axB.set_title("θ cross-correlation (theta-band)", fontsize=title_fs)
    axB.set_xlabel("Lag (s)", fontsize=label_fs)
    axB.set_ylabel("Correlation (r)", fontsize=label_fs)
    axB.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    # C) phase-shift density (pair colours)
    axC.plot(centres, dmu13, lw=2.6, color=COL_IPSI, label=r"$\phi_{CA3L}-\phi_{CA1L}$ (ipsi)")
    axC.fill_between(centres, dmu13 - dse13, dmu13 + dse13, color=COL_IPSI, alpha=0.18)

    axC.plot(centres, dmu12, lw=2.6, color=COL_BILAT, label=r"$\phi_{CA1R}-\phi_{CA1L}$ (bilat)")
    axC.fill_between(centres, dmu12 - dse12, dmu12 + dse12, color=COL_BILAT, alpha=0.18)

    axC.set_title("Phase-shift distribution (vs CA1_L)", fontsize=title_fs)
    axC.set_xlabel("Phase offset (rad)", fontsize=label_fs)
    axC.set_ylabel("Density", fontsize=label_fs)
    _set_pi_ticks(axC)
    axC.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    # D) per-animal circular means (paired) + p-values
    y1 = np.degrees(a13)  # ipsi
    y2 = np.degrees(a12)  # bilat

    for i in range(len(y1)):
        axD.plot([0, 1], [y1[i], y2[i]], lw=1.8, color="0.6", alpha=0.8)

    axD.scatter(np.zeros_like(y1), y1, s=55, color=COL_IPSI, label="ipsi: CA1L–CA3L")
    axD.scatter(np.ones_like(y2),  y2, s=55, color=COL_BILAT, label="bilat: CA1L–CA1R")
    axD.axhline(0, lw=1.2, alpha=0.5)

    axD.set_xlim(-0.4, 1.4)
    axD.set_xticks([0, 1])
    axD.set_xticklabels(["ipsi", "bilat"], fontsize=tick_fs)
    axD.set_title("Animal-level phase offsets", fontsize=title_fs)
    axD.set_ylabel("Circular mean phase offset (deg)", fontsize=label_fs)
    axD.legend(frameon=False, fontsize=legend_fs, loc="upper left")

    annot = (
        f"N={group_stats['n_animals']} animals\n"
        f"ipsi vs 0: p={group_stats['p_phase_CA1L_CA3L_vs0']:.3g}\n"
        f"bilat vs 0: p={group_stats['p_phase_CA1L_CA1R_vs0']:.3g}\n"
        f"ipsi-bilat: p={group_stats['p_phase_ipsi_minus_bilat']:.3g}"
    )
    #axD.text(0.02, 0.98, annot, transform=axD.transAxes, va="top", ha="left", fontsize=11)

    # styling
    for ax in (axA, axB, axC, axD):
        _despine_all(ax)
        _style_ticks(ax, tick_fs=tick_fs)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, df_anim, group_stats


if __name__ == "__main__":
    alocos = [
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1887932_Jedi2p_Multi\ALocomotion",
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1887933_Jedi2p_multi\ALocomotion",
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1955299_Jedi2p_Multi\ALocomotion",
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1966293_Jedi2p_Multi\ALocomotion",
    ]

    fig, df_anim, stats_out = plot_group_theta_relationships(
        alocos,
        theta_band=(4.0, 12.0),
        t_before=0.2, t_after=0.2,
        max_lag_s=0.2,
        min_epochs_per_sweep=20,
        show=True,
        save_path=r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\group_theta_multifibre.png",
    )

    print(df_anim)
    print(stats_out)

    df_anim.to_csv(
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\group_theta_multifibre_per_animal.csv",
        index=False
    )
