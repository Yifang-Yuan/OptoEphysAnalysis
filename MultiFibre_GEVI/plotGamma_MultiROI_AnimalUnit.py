# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:20:26 2026

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Animal-wise gamma analysis (slow + fast), analogous to theta animal-unit analysis.

Key change vs theta:
- Alignment anchor: CA1_L GAMMA troughs (band-specific)
- Uses FS_GAMMA (default 500 Hz) for bandpass / trough / phase / xcorr,
  so fast gamma 65–100 Hz is valid.

Outputs:
- plot_group_gamma_relationships(...) -> (fig, df_anim, group_stats)
- compute_animal_gamma_summary(...)   -> dict per animal

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

# Must satisfy fs_gamma > 2*highcut (Nyquist). For 100 Hz highcut, fs_gamma must be > 200 Hz.
FS_GAMMA = 500.0

SLOW_GAMMA_BAND = (30.0, 55.0)
FAST_GAMMA_BAND = (65.0, 100.0)

# Region colour code (match your PSD figure)
COL_CA1L = "#2ca02c"  # green
COL_CA1R = "#d62728"  # red
COL_CA3L = "#b8860b"  # yellow-brown

# Pair colours (for xcorr + phase plots)
COL_IPSI = COL_CA3L   # CA1_L vs CA3_L
COL_BILAT = COL_CA1R  # CA1_L vs CA1_R


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
    raise FileNotFoundError(
        "No synced_optical_behaviour_highrate.(parquet|pkl.gz|pkl) in {}".format(folder)
    )


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


def butter_bandpass_filtfilt(x: np.ndarray,
                             fs: float,
                             band: Tuple[float, float],
                             order: int = 4) -> np.ndarray:
    lo, hi = float(band[0]), float(band[1])
    if hi <= lo:
        raise ValueError("Invalid band {} (need hi>lo)".format(band))
    nyq = 0.5 * fs
    if hi >= nyq:
        raise ValueError(
            "Band highcut {:.1f} Hz >= Nyquist {:.1f} Hz. Increase fs or lower band.".format(hi, nyq)
        )
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return signal.filtfilt(b, a, np.asarray(x, float))


def find_band_troughs(x_band: np.ndarray,
                      fs: float,
                      band: Tuple[float, float],
                      min_prominence: Optional[float] = None,
                      min_dist_factor: float = 0.7) -> np.ndarray:
    """
    Find trough indices on a bandpassed signal via peaks of the inverted signal.
    min_dist ~ 0.7 * shortest period = 0.7 * fs / band_high
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
# Circular helpers
# -------------------------
def wrap_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x + np.pi) % (2 * np.pi) - np.pi


def circ_mean(alpha: np.ndarray) -> float:
    alpha = np.asarray(alpha, float)
    alpha = alpha[np.isfinite(alpha)]
    if alpha.size == 0:
        return np.nan
    return float(np.angle(np.mean(np.exp(1j * alpha))))


def exact_paired_circular_signflip(angles_rad: np.ndarray) -> float:
    """
    Exact paired circular sign-flip test (two-sided) for mean angle != 0.
    angles_rad: per-animal angles (radians).
    Test statistic: |circ_mean(angles)|
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
# Core extraction per SyncRecording folder (gamma trough aligned)
# -------------------------
def _extract_trough_epochs_and_gamma(
    sync_folder: Path,
    gamma_band: Tuple[float, float],
    fs_atlas: float,
    fs_gamma: float,
    t_before: float,
    t_after: float,
    min_epochs: int = 40,
) -> Optional[Dict[str, Any]]:
    """
    Returns dict with:
      E_gam: gamma-band epochs (n_epochs, n_t) for CA1_L/CA1_R/CA3_L
      phase_offsets: per-epoch phase offset at trough time (radians), pairs:
          'CA1L_CA3L' and 'CA1L_CA1R'  (phi_other - phi_ref at trough index)
      meta
    """
    df = load_synced_highrate(sync_folder)

    # raw detrend
    ca1l_raw = detrend_nan(df["CA1_L"].to_numpy(float))
    ca1r_raw = detrend_nan(df["CA1_R"].to_numpy(float))
    ca3l_raw = detrend_nan(df["CA3_L"].to_numpy(float))

    # downsample to fs_gamma (must support gamma band)
    ca1l = downsample_poly(ca1l_raw, fs_atlas, fs_gamma)
    ca1r = downsample_poly(ca1r_raw, fs_atlas, fs_gamma)
    ca3l = downsample_poly(ca3l_raw, fs_atlas, fs_gamma)

    n = min(len(ca1l), len(ca1r), len(ca3l))
    ca1l, ca1r, ca3l = ca1l[:n], ca1r[:n], ca3l[:n]

    # gamma band
    ca1l_g = butter_bandpass_filtfilt(ca1l, fs=fs_gamma, band=gamma_band, order=4)
    ca1r_g = butter_bandpass_filtfilt(ca1r, fs=fs_gamma, band=gamma_band, order=4)
    ca3l_g = butter_bandpass_filtfilt(ca3l, fs=fs_gamma, band=gamma_band, order=4)

    # troughs on CA1_L gamma
    trough_idx = find_band_troughs(ca1l_g, fs=fs_gamma, band=gamma_band)

    pre = int(round(t_before * fs_gamma))
    post = int(round(t_after * fs_gamma))
    L = pre + post + 1

    good = trough_idx[(trough_idx - pre >= 0) & (trough_idx + post < n)]
    if good.size < min_epochs:
        return None

    def stack_epochs(x):
        return np.vstack([x[ti - pre: ti + post + 1] for ti in good])

    E_gam = {
        "CA1_L": stack_epochs(ca1l_g),
        "CA1_R": stack_epochs(ca1r_g),
        "CA3_L": stack_epochs(ca3l_g),
    }

    # phase offsets at trough sample (relative to CA1_L)
    phi1 = np.angle(signal.hilbert(ca1l_g))
    phi2 = np.angle(signal.hilbert(ca1r_g))
    phi3 = np.angle(signal.hilbert(ca3l_g))

    off_13 = wrap_pi(phi3[good] - phi1[good])  # CA3L - CA1L
    off_12 = wrap_pi(phi2[good] - phi1[good])  # CA1R - CA1L

    return {
        "E_gam": E_gam,
        "phase_offsets": {
            "CA1L_CA3L": off_13,
            "CA1L_CA1R": off_12,
        },
        "meta": {
            "n_epochs": int(good.size),
            "L": int(L),
            "fs_gamma": float(fs_gamma),
        }
    }


# -------------------------
# Cross-correlation per epoch (gamma-band)
# -------------------------
def _epoch_xcorr_curve(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: float):
    """
    Normalised cross-correlation curve for one epoch.
    correlate(y, x): positive lag => y is shifted later relative to x (y lags x).
    Uses a robust lag vector to avoid SciPy version differences.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    x = x / (np.nanstd(x) + 1e-12)
    y = y / (np.nanstd(y) + 1e-12)

    r_full = signal.correlate(y, x, mode="full") / len(x)

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


def _default_gamma_window(gamma_band: Tuple[float, float]) -> Tuple[float, float]:
    # show ~2–4 cycles by default
    if gamma_band[1] <= 60.0:
        return 0.06, 0.06  # slow gamma
    return 0.04, 0.04      # fast gamma


def compute_animal_gamma_summary(
    aloco_path: Union[str, Path],
    gamma_band: Tuple[float, float],
    fs_atlas: float = FS_ATLAS,
    fs_gamma: float = FS_GAMMA,
    t_before: Optional[float] = None,
    t_after: Optional[float] = None,
    max_lag_s: Optional[float] = None,
    min_epochs_per_sweep: int = 40,
) -> Dict[str, Any]:
    """
    For one animal (ALocomotion folder):
      - Pool gamma-trough-triggered epochs across all SyncRecording*
      - Animal-level mean gamma-band waveforms (CA1_L, CA1_R, CA3_L)
      - Animal-level mean gamma-band xcorr curves for two pairs
      - Phase offset samples + animal-level circular mean offsets
    """
    aloco_path = Path(aloco_path)
    animal_id = _infer_animal_id(aloco_path)

    if fs_gamma <= 2.0 * gamma_band[1]:
        raise ValueError(
            "fs_gamma={} too low for gamma up to {} Hz. Use fs_gamma >= {} Hz (e.g. 400–600).".format(
                fs_gamma, gamma_band[1], int(2 * gamma_band[1] + 1)
            )
        )

    if t_before is None or t_after is None:
        tb, ta = _default_gamma_window(gamma_band)
        if t_before is None:
            t_before = tb
        if t_after is None:
            t_after = ta

    if max_lag_s is None:
        max_lag_s = 0.05 if gamma_band[1] <= 60.0 else 0.03

    sync_folders = _list_syncrecordings(aloco_path)
    if not sync_folders:
        raise FileNotFoundError("No SyncRecording* folders found under: {}".format(aloco_path))

    gam_pool = {k: [] for k in ["CA1_L", "CA1_R", "CA3_L"]}
    off_13_all = []
    off_12_all = []

    lags_ref = None
    sum_xcorr_13 = None
    sum_xcorr_12 = None
    n_xcorr = 0
    used_sweeps = 0

    for f in sync_folders:
        out = _extract_trough_epochs_and_gamma(
            f, gamma_band=gamma_band, fs_atlas=fs_atlas, fs_gamma=fs_gamma,
            t_before=t_before, t_after=t_after, min_epochs=min_epochs_per_sweep
        )
        if out is None:
            continue

        used_sweeps += 1
        n_ep = out["meta"]["n_epochs"]
        Eg = out["E_gam"]

        for k in gam_pool:
            gam_pool[k].append(Eg[k])

        off_13_all.append(out["phase_offsets"]["CA1L_CA3L"])
        off_12_all.append(out["phase_offsets"]["CA1L_CA1R"])

        for i in range(n_ep):
            x = Eg["CA1_L"][i]
            y13 = Eg["CA3_L"][i]
            y12 = Eg["CA1_R"][i]

            lags, r13 = _epoch_xcorr_curve(x, y13, fs=fs_gamma, max_lag_s=max_lag_s)
            _,    r12 = _epoch_xcorr_curve(x, y12, fs=fs_gamma, max_lag_s=max_lag_s)

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
        raise ValueError(
            "{}: no SyncRecording had >= {} gamma trough epochs ({}–{} Hz).".format(
                animal_id, min_epochs_per_sweep, gamma_band[0], gamma_band[1]
            )
        )

    Eg_all = {k: np.vstack(gam_pool[k]) for k in gam_pool}
    off_13_all = np.concatenate(off_13_all) if off_13_all else np.array([], float)
    off_12_all = np.concatenate(off_12_all) if off_12_all else np.array([], float)

    mean_gam = {k: np.nanmean(Eg_all[k], axis=0) for k in Eg_all}

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

    L = Eg_all["CA1_L"].shape[1]
    t_rel = np.linspace(-t_before, t_after, L)

    return {
        "animal_id": animal_id,
        "gamma_band": gamma_band,
        "fs_gamma": float(fs_gamma),
        "used_sweeps": int(used_sweeps),
        "n_epochs_total": int(Eg_all["CA1_L"].shape[0]),
        "t_rel": t_rel,
        "mean_gam": mean_gam,
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
def plot_group_gamma_relationships(
    aloco_paths: List[str],
    gamma_band: Tuple[float, float],
    fs_atlas: float = FS_ATLAS,
    fs_gamma: float = FS_GAMMA,
    t_before: Optional[float] = None,
    t_after: Optional[float] = None,
    max_lag_s: Optional[float] = None,
    min_epochs_per_sweep: int = 40,
    n_phase_bins: int = 36,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Figure (animal is unit):
      A) group mean gamma-band trough-triggered waveforms (mean ± SEM across animals)
      B) group mean gamma-band xcorr curves for pairs (mean ± SEM across animals)
      C) group mean phase-shift density for pairs (mean ± SEM across animals)
      D) per-animal circular mean phase offsets (paired) + signflip p-values
    """
    summaries = []
    for p in aloco_paths:
        summaries.append(compute_animal_gamma_summary(
            p, gamma_band=gamma_band,
            fs_atlas=fs_atlas, fs_gamma=fs_gamma,
            t_before=t_before, t_after=t_after,
            max_lag_s=max_lag_s,
            min_epochs_per_sweep=min_epochs_per_sweep
        ))

    # per-animal table
    rows = []
    for s in summaries:
        rows.append({
            "animal_id": s["animal_id"],
            "used_sweeps": s["used_sweeps"],
            "n_epochs_total": s["n_epochs_total"],
            "gamma_lo_hz": float(s["gamma_band"][0]),
            "gamma_hi_hz": float(s["gamma_band"][1]),
            "fs_gamma": float(s["fs_gamma"]),
            "phase_mean_CA1L_CA3L_rad": s["phase_mean_rad"]["CA1L_CA3L"],
            "phase_mean_CA1L_CA1R_rad": s["phase_mean_rad"]["CA1L_CA1R"],
            "phase_mean_CA1L_CA3L_deg": np.degrees(s["phase_mean_rad"]["CA1L_CA3L"]),
            "phase_mean_CA1L_CA1R_deg": np.degrees(s["phase_mean_rad"]["CA1L_CA1R"]),
            **s["xcorr_metrics"],
        })
    df_anim = pd.DataFrame(rows)

    # group mean waveforms (gamma-band)
    t_rel = summaries[0]["t_rel"]
    M_ca1l = np.vstack([s["mean_gam"]["CA1_L"] for s in summaries])
    M_ca1r = np.vstack([s["mean_gam"]["CA1_R"] for s in summaries])
    M_ca3l = np.vstack([s["mean_gam"]["CA3_L"] for s in summaries])

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

    # phase density (animal-mean densities)
    edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def density_per_animal(offsets):
        h, _ = np.histogram(offsets, bins=edges, density=True)
        return h

    D13 = np.vstack([density_per_animal(s["phase_offsets"]["CA1L_CA3L"]) for s in summaries])
    D12 = np.vstack([density_per_animal(s["phase_offsets"]["CA1L_CA1R"]) for s in summaries])
    dmu13, dse13 = mean_sem(D13)
    dmu12, dse12 = mean_sem(D12)

    # stats across animals (animal is unit)
    a13 = wrap_pi(df_anim["phase_mean_CA1L_CA3L_rad"].to_numpy(float))
    a12 = wrap_pi(df_anim["phase_mean_CA1L_CA1R_rad"].to_numpy(float))
    diff = wrap_pi(a13 - a12)  # ipsi - bilat

    p_13_vs0 = exact_paired_circular_signflip(a13)
    p_12_vs0 = exact_paired_circular_signflip(a12)
    p_diff = exact_paired_circular_signflip(diff)

    group_stats = {
        "gamma_band": gamma_band,
        "fs_gamma": float(fs_gamma),
        "n_animals": int(len(summaries)),
        "p_phase_CA1L_CA3L_vs0": p_13_vs0,
        "p_phase_CA1L_CA1R_vs0": p_12_vs0,
        "p_phase_ipsi_minus_bilat": p_diff,
        "mean_phase_CA1L_CA3L_deg": float(np.degrees(circ_mean(a13))),
        "mean_phase_CA1L_CA1R_deg": float(np.degrees(circ_mean(a12))),
        "mean_phase_diff_deg": float(np.degrees(circ_mean(diff))),
    }

    # -------------------------
    # Plot (same 2×2 layout as theta)
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

    # A) mean gamma waveform (region colours)
    axA.plot(t_rel, mu1, lw=2.6, color=COL_CA1L, label="CA1_L (ref)")
    axA.fill_between(t_rel, mu1 - se1, mu1 + se1, color=COL_CA1L, alpha=0.18, linewidth=0)

    axA.plot(t_rel, mu2, lw=2.6, color=COL_CA1R, label="CA1_R")
    axA.fill_between(t_rel, mu2 - se2, mu2 + se2, color=COL_CA1R, alpha=0.18, linewidth=0)

    axA.plot(t_rel, mu3, lw=2.6, color=COL_CA3L, label="CA3_L")
    axA.fill_between(t_rel, mu3 - se3, mu3 + se3, color=COL_CA3L, alpha=0.18, linewidth=0)

    axA.axvline(0, lw=1.4, alpha=0.55)
    axA.set_title("Mean γ waveforms (0 = CA1_L γ trough)", fontsize=title_fs)
    axA.set_xlabel("Time relative to CA1_L γ trough (s)", fontsize=label_fs)
    axA.set_ylabel("Gamma-band (a.u.; filtered)", fontsize=label_fs)
    axA.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    # B) xcorr curves (pair colours)
    axB.plot(lags, xmu13, lw=2.6, color=COL_IPSI, label="CA1_L vs CA3_L (ipsi)")
    axB.fill_between(lags, xmu13 - xse13, xmu13 + xse13, color=COL_IPSI, alpha=0.18, linewidth=0)

    axB.plot(lags, xmu12, lw=2.6, color=COL_BILAT, label="CA1_L vs CA1_R (bilat)")
    axB.fill_between(lags, xmu12 - xse12, xmu12 + xse12, color=COL_BILAT, alpha=0.18, linewidth=0)

    axB.axvline(0, lw=1.4, alpha=0.55)
    axB.set_title("γ cross-correlation (gamma-band)", fontsize=title_fs)
    axB.set_xlabel("Lag (s)", fontsize=label_fs)
    axB.set_ylabel("Correlation (r)", fontsize=label_fs)
    axB.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    # C) phase-shift density
    axC.plot(centres, dmu13, lw=2.6, color=COL_IPSI, label=r"$\phi_{CA3L}-\phi_{CA1L}$ (ipsi)")
    axC.fill_between(centres, dmu13 - dse13, dmu13 + dse13, color=COL_IPSI, alpha=0.18, linewidth=0)

    axC.plot(centres, dmu12, lw=2.6, color=COL_BILAT, label=r"$\phi_{CA1R}-\phi_{CA1L}$ (bilat)")
    axC.fill_between(centres, dmu12 - dse12, dmu12 + dse12, color=COL_BILAT, alpha=0.18, linewidth=0)

    axC.set_title("Phase-shift distribution (vs CA1_L)", fontsize=title_fs)
    axC.set_xlabel("Phase offset (rad)", fontsize=label_fs)
    axC.set_ylabel("Density", fontsize=label_fs)
    _set_pi_ticks(axC)
    axC.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    # D) per-animal circular means (paired)
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
        "N={} animals\n"
        "ipsi vs 0: p={:.3g}\n"
        "bilat vs 0: p={:.3g}\n"
        "ipsi-bilat: p={:.3g}".format(
            group_stats["n_animals"],
            group_stats["p_phase_CA1L_CA3L_vs0"],
            group_stats["p_phase_CA1L_CA1R_vs0"],
            group_stats["p_phase_ipsi_minus_bilat"],
        )
    )
    #axD.text(0.02, 0.98, annot, transform=axD.transAxes, va="top", ha="left", fontsize=11)

    for ax in (axA, axB, axC, axD):
        _despine_all(ax)
        _style_ticks(ax, tick_fs=tick_fs)

    fig.suptitle("Gamma {}–{} Hz (aligned to CA1_L gamma troughs)".format(gamma_band[0], gamma_band[1]),
                 fontsize=15, y=0.99)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, df_anim, group_stats


# -------------------------
# Convenience: run both slow + fast gamma
# -------------------------
def run_gamma_bands_for_animals(
    aloco_paths: List[str],
    out_dir: Union[str, Path],
    fs_gamma: float = FS_GAMMA,
    show: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    for name, band in [("slow_gamma", SLOW_GAMMA_BAND), ("fast_gamma", FAST_GAMMA_BAND)]:
        fig_path = out_dir / "group_{}_multifibre.png".format(name)
        csv_path = out_dir / "group_{}_multifibre_per_animal.csv".format(name)

        fig, df_anim, stats_out = plot_group_gamma_relationships(
            aloco_paths,
            gamma_band=band,
            fs_gamma=fs_gamma,
            show=show,
            save_path=fig_path
        )

        df_anim.to_csv(csv_path, index=False)

        outputs[name] = {
            "band": band,
            "fig_path": str(fig_path),
            "csv_path": str(csv_path),
            "stats": stats_out
        }

    return outputs


if __name__ == "__main__":
    # --- your example animals ---
    alocos = [
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1887932_Jedi2p_Multi\ALocomotion",
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1887933_Jedi2p_multi\ALocomotion",
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1955299_Jedi2p_Multi\ALocomotion",
        r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1966293_Jedi2p_Multi\ALocomotion",
    ]

    out_dir = r"G:\2025_ATLAS_SPAD\MultiFibreGEVI"

    outputs = run_gamma_bands_for_animals(
        alocos,
        out_dir=out_dir,
        fs_gamma=FS_GAMMA,
        show=True
    )

    print(outputs)
