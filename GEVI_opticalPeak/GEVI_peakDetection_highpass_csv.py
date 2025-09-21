# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 13:53:52 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
GEVI peak validation with high-pass detection:
- Load CSV ('zscore_raw' or first numeric column)
- (Optional) invert signal
- High-pass at 100 Hz -> detect peaks on HP signal (MAD rule)
- Plot 3-s segments:
    Row1: raw trace + peak dots + MAD threshold (computed on HP signal)
    Row2: (optional) theta-band trace + peak dots
    Row3: high-pass (100 Hz) trace + peak dots
- Also outputs amplitude histogram (of the HP signal) and firing-rate distribution (1/ISI)

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import median_abs_deviation

# -------------------- global plot style --------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.6,
})

# -------------------- user settings --------------------
csv_path = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"
Fs = 841.68
invert_signal = True

# Peak detection (MAD rule) — now computed on the **high-passed** signal
height_factor   = 3.0         # threshold = median + 3×MAD
prominence_mul  = 2.0         # prominence ≥ 2×MAD
min_distance_ms = 2.0         # ≥ 2 ms apart

# Filters
HP_CUTOFF_HZ    = 100.0       # high-pass cutoff for detection
HP_ORDER        = 3
theta_band      = (4.0, 12.0) # set to None to disable theta subplot
theta_order     = 3
# ------------------------------------------------------


def load_signal_csv(csv_file, preferred_col="zscore_raw"):
    df = pd.read_csv(csv_file)
    if preferred_col in df.columns:
        sig = df[preferred_col]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric columns found in the CSV.")
        sig = df[num_cols[0]]
    x = sig.to_numpy(dtype=float)
    if np.any(np.isnan(x)):
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    return np.asarray(x, dtype=float), df.columns.tolist()


def butter_highpass(x, fs, cutoff_hz, order=3):
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    if not (0 < wn < 1):
        raise ValueError("High-pass cutoff must be between 0 and Nyquist.")
    b, a = butter(order, wn, btype='high')
    return filtfilt(b, a, x)


def butter_bandpass(x, fs, band, order=3):
    nyq = fs / 2.0
    low, high = band[0] / nyq, band[1] / nyq
    if not (0 < low < high < 1):
        raise ValueError("Band must lie within (0, Nyquist).")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def detect_peaks(signal_for_detection, fs, height_k=3.0, prom_k=2.0, min_dist_ms=2.0):
    """MAD-based detection on the provided signal (here: high-passed)."""
    x = np.asarray(signal_for_detection, float)
    baseline = np.median(x)
    mad = median_abs_deviation(x, scale=1.0)

    height_thresh      = baseline + height_k * mad
    prominence_thresh  = prom_k * mad
    distance_samples   = max(1, int(round((min_dist_ms / 1000.0) * fs)))

    peak_idx, _ = find_peaks(
        x, height=height_thresh, distance=distance_samples, prominence=prominence_thresh
    )
    return peak_idx, dict(
        baseline=baseline,
        mad=mad,
        height_thresh=height_thresh,
        prominence_thresh=prominence_thresh,
        distance_samples=distance_samples
    )


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=6, width=1.4)
    ax.grid(False)


def plot_segments_raw_theta_hp(raw, theta, hp, peak_idx, fs, out_dir,
                               seg_len_s=3.0, thresh_hp=None,
                               colour_raw="green", colour_peaks="black"):
    """
    3 rows per 3-s segment:
      1) Raw (z) + peaks (indices from HP detection), optional threshold line (HP threshold)
      2) Theta band + peaks (if theta is not None)
      3) High-pass (100 Hz) + peaks  [detection was done on this signal]
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    n = raw.size
    seg_len = int(round(seg_len_s * fs))
    n_segments = int(np.ceil(n / seg_len))
    t = np.arange(n) / fs

    for k in range(n_segments):
        i0, i1 = k * seg_len, min((k + 1) * seg_len, n)
        if i0 >= i1: continue

        seg_t  = t[i0:i1]
        seg_raw = raw[i0:i1]
        seg_hp  = hp[i0:i1]
        seg_th  = None if theta is None else theta[i0:i1]

        mask = (peak_idx >= i0) & (peak_idx < i1)
        pseg = peak_idx[mask]
        pseg_t   = t[pseg]
        pseg_y_r = raw[pseg]
        pseg_y_hp= hp[pseg]
        pseg_y_th= None if theta is None else theta[pseg]

        # rows: raw / (theta?) / hp
        nrows = 3 if theta is not None else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 7.2 if nrows==3 else 5.2),
                                 dpi=170, sharex=True)
        if nrows == 2:
            ax1, ax3 = axes
            ax2 = None
        else:
            ax1, ax2, ax3 = axes

        # --- Row 1: Raw
        ax1.plot(seg_t, seg_raw, lw=1.1, color=colour_raw, label="zscore_raw")
        ax1.scatter(pseg_t, pseg_y_r, s=20, color=colour_peaks, zorder=3, label="Detected peaks")
        
        ax1.set_ylabel("Amplitude (z)")
        ax1.set_title(f"Raw + {'theta + ' if theta is not None else ''}high-pass traces with peaks • Segment {k+1}/{n_segments}")
        _clean_axes(ax1); ax1.margins(x=0)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        # --- Row 2: Theta (optional)
        if ax2 is not None:
            ax2.plot(seg_t, seg_th, lw=1.1, color="black", label=f"Theta {theta_band[0]}–{theta_band[1]} Hz")
            ax2.scatter(pseg_t, pseg_y_th, s=20, color=colour_peaks, zorder=3, label="Detected peaks")
            ax2.set_ylabel("Theta (a.u.)")
            _clean_axes(ax2); ax2.margins(x=0)
            ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        # --- Row 3 (or 2): High-pass
        ax3.plot(seg_t, seg_hp, lw=1.0, color="0.1", label=f"High-pass {HP_CUTOFF_HZ:.0f} Hz")
        ax3.scatter(pseg_t, pseg_y_hp, s=16, color=colour_peaks, zorder=3, label="Detected peaks")
        if thresh_hp is not None:
            ax3.axhline(thresh_hp, ls="--", lw=1.6, color="red", alpha=0.85, label="HP MAD threshold")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("HP (a.u.)")
        _clean_axes(ax3); ax3.margins(x=0)
        ax3.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        fig.tight_layout(); fig.subplots_adjust(right=0.82)
        fig.savefig(out_dir / f"trace_theta_hp_peaks_seg{k+1:03d}.png", bbox_inches="tight")
        plt.close(fig)


def plot_amplitude_histogram(signal, thresh, out_path, bins=200):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.hist(signal, bins=bins, alpha=0.85, color="steelblue")
    ax.axvline(thresh, linestyle="--", linewidth=2.2, color="red",
               label="Threshold = median + 3×MAD")

    # Larger fonts
    ax.set_xlabel("Sample amplitude (z)", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_title("Amplitude distribution with MAD threshold", fontsize=20)
    ax.tick_params(labelsize=20)
    ax.legend(frameon=False, fontsize=18)

    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def plot_firing_rate_distribution(peak_idx, fs, out_path, max_percentile=99.5):
    """Instantaneous rate = 1 / ISI."""
    if peak_idx.size < 2: return
    isi_s = np.diff(peak_idx) / fs
    isi_s = isi_s[isi_s > 0]
    rates = 1.0 / isi_s
    xmax = np.percentile(rates, max_percentile)
    bins = np.linspace(0, xmax, 60)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=170)
    ax.hist(rates, bins=bins, alpha=0.9)
    ax.set_xlabel("Instantaneous firing rate (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Firing-rate distribution (1/ISI)")
    _clean_axes(ax)
    ax.text(0.98, 0.95, f"Mean: {np.mean(rates):.2f} Hz\nMedian: {np.median(rates):.2f} Hz\nN ISIs: {len(isi_s)}",
            transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def main():
    # 1) load
    raw, cols = load_signal_csv(csv_path, preferred_col="zscore_raw")
    if invert_signal:
        raw = -raw

    # 2) filters
    hp  = butter_highpass(raw, Fs, HP_CUTOFF_HZ, order=HP_ORDER)
    th  = None if theta_band is None else butter_bandpass(raw, Fs, theta_band, order=theta_order)

    # 3) detect peaks on **high-passed** signal
    peak_idx, stats = detect_peaks(
        hp, Fs,
        height_k=height_factor,
        prom_k=prominence_mul,
        min_dist_ms=min_distance_ms
    )
    print(f"Samples: {raw.size}  Duration: {raw.size/Fs:.2f}s  Peaks: {peak_idx.size}")
    print(f"HP MAD={stats['mad']:.4f}  HP threshold={stats['height_thresh']:.4f}  "
          f"PromThresh={stats['prominence_thresh']:.4f}  MinDist={stats['distance_samples']} samples")

    # 4) outputs
    out_root = Path(csv_path).with_suffix("")
    base_dir = out_root.parent / "optical_event_validation_hp100"
    seg_dir  = base_dir / "segments"
    base_dir.mkdir(parents=True, exist_ok=True); seg_dir.mkdir(parents=True, exist_ok=True)

    # 5) segment plots (raw + theta? + HP)
    plot_segments_raw_theta_hp(
        raw, th, hp, peak_idx, Fs, seg_dir, seg_len_s=3.0, thresh_hp=stats["height_thresh"]
    )

    # 6) amplitude histogram of the HP signal (the one used for detection)
    plot_amplitude_histogram(hp, stats["height_thresh"], base_dir / "amplitude_histogram_hp.png", bins=200)

    # 7) optional ISI-based firing-rate distribution
    plot_firing_rate_distribution(peak_idx, Fs, base_dir / "firing_rate_distribution.png")

    print(f"\nSaved figures to: {base_dir}")


if __name__ == "__main__":
    main()
