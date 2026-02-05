# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:56:51 2026

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
# Config (edit as needed)
# -------------------------
FS_ATLAS = 1682.92

THETA_BAND = (6.0, 10.0)     # Hz
FS_DS = 200.0                # downsample rate for theta/coherence (Hz)

SPEED_THRESH = 3.0           # cm/s (moving if > 3)
MIN_SECONDS_FOR_METRICS = 10.0   # require at least this much data


# -------------------------
# IO helpers
# -------------------------
def load_synced_highrate(folder: Path) -> pd.DataFrame:
    """
    Load synced optical+behaviour table saved previously.
    Supports parquet or gzip-pickle fallback.
    """
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
            if p.suffix in [".pkl", ".gz"] or p.name.endswith(".pkl.gz"):
                return pd.read_pickle(p, compression="gzip" if p.name.endswith(".gz") else None)
    raise FileNotFoundError(f"Cannot find synced_optical_behaviour_highrate.(parquet|pkl.gz) in {folder}")


def safe_to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------
# Signal processing helpers
# -------------------------
def downsample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Anti-aliased downsampling using resample_poly with rational ratio.
    """
    if abs(fs_in - fs_out) < 1e-9:
        return x

    frac = Fraction(fs_out / fs_in).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    return signal.resample_poly(x, up=up, down=down)


def butter_bandpass_filtfilt(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    lo, hi = band
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq
    b, a = signal.butter(order, [lo_n, hi_n], btype="bandpass")
    return signal.filtfilt(b, a, x)


def plv_from_bandpassed(x_theta: np.ndarray, y_theta: np.ndarray) -> float:
    """
    Phase-locking value computed from Hilbert phase of bandpassed signals.
    """
    phx = np.angle(signal.hilbert(x_theta))
    phy = np.angle(signal.hilbert(y_theta))
    return float(np.abs(np.mean(np.exp(1j * (phx - phy)))))


def coherence_theta(x: np.ndarray, y: np.ndarray, fs: float, theta_band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (f, Cxy, mean_theta_coh).
    """
    # Welch params: longer segments improve low-f resolution
    nperseg = int(round(fs * 4.0))        # 4 s
    nperseg = max(256, nperseg)
    noverlap = int(round(nperseg * 0.5))

    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    lo, hi = theta_band
    mask = (f >= lo) & (f <= hi)
    mean_theta = float(np.nanmean(Cxy[mask])) if mask.any() else np.nan
    return f, Cxy, mean_theta


def _clean_finite(*arrs):
    """
    Keep only indices where all arrays are finite.
    """
    m = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return [a[m] for a in arrs], m


# -------------------------
# Per-folder analysis
# -------------------------
def analyse_theta_coupling_one_folder(
    folder: Path,
    fs_atlas: float = FS_ATLAS,
    fs_ds: float = FS_DS,
    theta_band: tuple[float, float] = THETA_BAND,
    speed_thresh: float = SPEED_THRESH,
):
    folder = Path(folder)
    df = load_synced_highrate(folder)

    # Required columns
    for col in ["t_s", "CA1_L", "CA1_R", "CA3_L", "speed_cm_s"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {folder.name}")

    # Extract arrays
    ca1l = df["CA1_L"].to_numpy(dtype=float)
    ca1r = df["CA1_R"].to_numpy(dtype=float)
    ca3l = df["CA3_L"].to_numpy(dtype=float)
    speed = df["speed_cm_s"].to_numpy(dtype=float)

    # Detrend (helps coherence/correlation, keeps band content)
    ca1l = signal.detrend(ca1l, type="constant")
    ca1r = signal.detrend(ca1r, type="constant")
    ca3l = signal.detrend(ca3l, type="constant")

    # Downsample
    ca1l_ds = downsample(ca1l, fs_in=fs_atlas, fs_out=fs_ds)
    ca1r_ds = downsample(ca1r, fs_in=fs_atlas, fs_out=fs_ds)
    ca3l_ds = downsample(ca3l, fs_in=fs_atlas, fs_out=fs_ds)
    speed_ds = downsample(speed, fs_in=fs_atlas, fs_out=fs_ds)

    n = min(len(ca1l_ds), len(ca1r_ds), len(ca3l_ds), len(speed_ds))
    ca1l_ds, ca1r_ds, ca3l_ds, speed_ds = ca1l_ds[:n], ca1r_ds[:n], ca3l_ds[:n], speed_ds[:n]
    t_ds = np.arange(n, dtype=float) / fs_ds

    # Bandpass to theta; CA1_L is your reference theta waveform
    ca1l_theta = butter_bandpass_filtfilt(ca1l_ds, fs=fs_ds, band=theta_band)
    ca1r_theta = butter_bandpass_filtfilt(ca1r_ds, fs=fs_ds, band=theta_band)
    ca3l_theta = butter_bandpass_filtfilt(ca3l_ds, fs=fs_ds, band=theta_band)

    # ---------- Metrics on all samples ----------
    [x_all, y_ca3_all, y_ca1r_all], _ = _clean_finite(ca1l_theta, ca3l_theta, ca1r_theta)

    min_n = int(MIN_SECONDS_FOR_METRICS * fs_ds)
    if len(x_all) < min_n:
        raise ValueError(f"Too little valid data for metrics in {folder.name} ({len(x_all)/fs_ds:.1f}s)")

    # Coherence spectra + theta-mean coherence
    f13, C13, coh13 = coherence_theta(x_all, y_ca3_all, fs=fs_ds, theta_band=theta_band)   # CA1L-CA3L
    f12, C12, coh12 = coherence_theta(x_all, y_ca1r_all, fs=fs_ds, theta_band=theta_band)  # CA1L-CA1R

    # Theta-band Pearson r
    r13 = float(np.corrcoef(x_all, y_ca3_all)[0, 1])
    r12 = float(np.corrcoef(x_all, y_ca1r_all)[0, 1])

    # PLV
    plv13 = plv_from_bandpassed(x_all, y_ca3_all)
    plv12 = plv_from_bandpassed(x_all, y_ca1r_all)

    # ---------- Metrics during movement only ----------
    move_mask = np.isfinite(speed_ds) & (speed_ds > speed_thresh)
    # Concatenate moving samples (note: introduces boundary discontinuities, but practical for per-sweep summaries)
    x_mv = ca1l_theta[move_mask]
    y_ca3_mv = ca3l_theta[move_mask]
    y_ca1r_mv = ca1r_theta[move_mask]
    [x_mv, y_ca3_mv, y_ca1r_mv], _ = _clean_finite(x_mv, y_ca3_mv, y_ca1r_mv)

    if len(x_mv) >= min_n:
        f13m, C13m, coh13_mv = coherence_theta(x_mv, y_ca3_mv, fs=fs_ds, theta_band=theta_band)
        f12m, C12m, coh12_mv = coherence_theta(x_mv, y_ca1r_mv, fs=fs_ds, theta_band=theta_band)
        r13_mv = float(np.corrcoef(x_mv, y_ca3_mv)[0, 1])
        r12_mv = float(np.corrcoef(x_mv, y_ca1r_mv)[0, 1])
        plv13_mv = plv_from_bandpassed(x_mv, y_ca3_mv)
        plv12_mv = plv_from_bandpassed(x_mv, y_ca1r_mv)
    else:
        coh13_mv = coh12_mv = np.nan
        r13_mv = r12_mv = np.nan
        plv13_mv = plv12_mv = np.nan

    # ---------- Save metrics ----------
    metrics = pd.DataFrame([{
        "folder": folder.name,
        "fs_ds": fs_ds,
        "theta_lo": theta_band[0],
        "theta_hi": theta_band[1],
        "theta_coh_CA1L_CA3L_all": coh13,
        "theta_coh_CA1L_CA1R_all": coh12,
        "theta_r_CA1L_CA3L_all": r13,
        "theta_r_CA1L_CA1R_all": r12,
        "theta_plv_CA1L_CA3L_all": plv13,
        "theta_plv_CA1L_CA1R_all": plv12,
        "theta_coh_CA1L_CA3L_moving": coh13_mv,
        "theta_coh_CA1L_CA1R_moving": coh12_mv,
        "theta_r_CA1L_CA3L_moving": r13_mv,
        "theta_r_CA1L_CA1R_moving": r12_mv,
        "theta_plv_CA1L_CA3L_moving": plv13_mv,
        "theta_plv_CA1L_CA1R_moving": plv12_mv,
        "moving_fraction": float(np.mean(move_mask)),
    }])

    safe_to_csv(metrics, folder / "theta_coupling_metrics.csv")

    # ---------- Plot per folder ----------
    # show first 30 seconds for readability (or full if shorter)
    show_s = min(30.0, t_ds[-1])
    idx = t_ds <= show_s

    fig = plt.figure(figsize=(12, 9))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t_ds[idx], ca1l_ds[idx], lw=0.8, label="CA1_L (raw ds)")
    ax1.plot(t_ds[idx], ca3l_ds[idx], lw=0.8, label="CA3_L (raw ds)")
    ax1.plot(t_ds[idx], ca1r_ds[idx], lw=0.8, label="CA1_R (raw ds)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t_ds[idx], ca1l_theta[idx], lw=1.0, label="CA1_L theta")
    ax2.plot(t_ds[idx], ca3l_theta[idx], lw=1.0, label="CA3_L theta")
    ax2.plot(t_ds[idx], ca1r_theta[idx], lw=1.0, label="CA1_R theta")
    ax2.set_ylabel("Theta-band")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(f13, C13, lw=1.2, label=f"CA1_L–CA3_L (theta mean={coh13:.3f})")
    ax3.plot(f12, C12, lw=1.2, label=f"CA1_L–CA1_R (theta mean={coh12:.3f})")
    ax3.axvspan(theta_band[0], theta_band[1], alpha=0.2)
    ax3.set_xlim(0, 30)  # coherence spectrum up to 30 Hz is usually enough here
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Coherence")
    ax3.legend(loc="upper right")

    fig.suptitle(f"{folder.name} | theta {theta_band[0]}–{theta_band[1]} Hz | ds={fs_ds} Hz")
    fig.tight_layout()
    fig.savefig(folder / "theta_coupling_coherence.png", dpi=150)
    plt.close(fig)

    return metrics.iloc[0].to_dict()


def batch_analyse_animal(animal_dir: str | Path):
    animal_dir = Path(animal_dir)
    folders = sorted([p for p in animal_dir.glob("SyncRecording*") if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"No SyncRecording* folders under {animal_dir}")

    rows = []
    for f in folders:
        try:
            row = analyse_theta_coupling_one_folder(f)
            rows.append(row)
            print(f"[OK] {f.name}: coh_CA1L_CA3L={row['theta_coh_CA1L_CA3L_all']:.3f}, coh_CA1L_CA1R={row['theta_coh_CA1L_CA1R_all']:.3f}")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")

    if rows:
        summary = pd.DataFrame(rows)
        safe_to_csv(summary, animal_dir / "theta_coupling_summary.csv")

        # Optional summary plot across sweeps
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(len(summary))
        ax.plot(x, summary["theta_coh_CA1L_CA3L_all"], marker="o", lw=1.2, label="CA1L–CA3L")
        ax.plot(x, summary["theta_coh_CA1L_CA1R_all"], marker="o", lw=1.2, label="CA1L–CA1R")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["folder"], rotation=45, ha="right")
        ax.set_ylabel("Mean theta coherence")
        ax.set_title("Theta coherence per SyncRecording")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(animal_dir / "theta_coupling_summary.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    # Example:
    batch_analyse_animal(r"G:\2025_ATLAS_SPAD\MultiFibreGEVI\1955299_Jedi2p_Multi\Day1")
