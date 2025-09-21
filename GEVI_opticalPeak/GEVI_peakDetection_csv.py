# -*- coding: utf-8 -*-
"""
Optical-event (peak) visual validation from CSV with:
- Segments (raw + theta + peaks)
- Firing-rate distribution (1/ISI)
- Amplitude distribution (with MAD threshold line)

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
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.6,
})

# -------------------- user settings --------------------
csv_path = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"

Fs = 841.68            # Hz
invert_signal = True   # GEVI inverted in your case

# Peak detection parameters
height_factor   = 3.0         # threshold = median + 3×MAD
prominence_mul  = 2.0         # prominence ≥ 2×MAD
min_distance_ms = 2.0         # peaks at least 2 ms apart

# Theta filter settings
theta_band = (4.0, 12.0)      # Hz
theta_filt_order = 3
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


def detect_peaks(signal, fs, height_k=3.0, prom_k=2.0, min_dist_ms=2.0):
    x = signal.copy()
    baseline = np.median(x)
    mad = median_abs_deviation(x, scale=1.0)

    height_thresh = baseline + height_k * mad
    prominence_thresh = prom_k * mad
    distance_samples = max(1, int(round((min_dist_ms / 1000.0) * fs)))

    peak_idx, props = find_peaks(
        x,
        height=height_thresh,
        distance=distance_samples,
        prominence=prominence_thresh
    )
    return peak_idx, props, dict(
        baseline=baseline,
        mad=mad,
        height_thresh=height_thresh,
        prominence_thresh=prominence_thresh,
        distance_samples=distance_samples
    )


def bandpass_theta(x, fs, band=(6.0, 10.0), order=3):
    nyq = fs / 2.0
    low, high = band[0] / nyq, band[1] / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x, method="pad")


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.grid(False)


def plot_segments_with_peaks_and_theta(signal, theta, peak_idx, fs, out_dir,
                                       seg_len_s=3.0,
                                       colour_trace="green",
                                       colour_peaks="black",
                                       thresh=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    n = signal.size
    seg_len = int(round(seg_len_s * fs))
    n_segments = int(np.ceil(n / seg_len))
    t = np.arange(n) / fs

    for k in range(n_segments):
        i0, i1 = k * seg_len, min((k + 1) * seg_len, n)
        if i0 >= i1: 
            continue

        seg_t, seg_x, seg_th = t[i0:i1], signal[i0:i1], theta[i0:i1]
        mask = (peak_idx >= i0) & (peak_idx < i1)
        pseg = peak_idx[mask]
        pseg_t, pseg_y_raw, pseg_y_th = t[pseg], signal[pseg], theta[pseg]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=160, sharex=True)
        ax1, ax2 = axes

        # Raw trace
        ax1.plot(seg_t, seg_x, lw=1.1, color=colour_trace, label="zscore_raw")
        ax1.scatter(pseg_t, pseg_y_raw, s=30, color=colour_peaks, zorder=3, label="Detected peaks")
        if thresh is not None:
            ax1.axhline(thresh, ls="--", lw=1.5, color="red", alpha=0.8, label="MAD threshold")
        ax1.set_ylabel("Amplitude (z)", fontsize=18)
        ax1.set_title(f"Raw + theta traces with peaks • Segment {k+1}/{n_segments}", fontsize=20)
        _clean_axes(ax1); ax1.margins(x=0)
        ax1.tick_params(labelsize=16)
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=14)

        # Theta trace
        ax2.plot(seg_t, seg_th, lw=1.1, color="black", label=f"Theta {theta_band[0]}–{theta_band[1]} Hz")
        ax2.scatter(pseg_t, pseg_y_th, s=30, color=colour_peaks, zorder=3, label="Detected peaks")
        ax2.set_xlabel("Time (s)", fontsize=18)
        ax2.set_ylabel("Theta (a.u.)", fontsize=18)
        _clean_axes(ax2); ax2.margins(x=0)
        ax2.tick_params(labelsize=16)
        ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=14)

        fig.tight_layout(); fig.subplots_adjust(right=0.82)
        fig.savefig(out_dir / f"trace_theta_peaks_seg{k+1:03d}.png", bbox_inches="tight")
        plt.close(fig)


def plot_firing_rate_distribution(peak_idx, fs, out_path, max_percentile=99.5):
    if peak_idx.size < 2: return
    isi_s = np.diff(peak_idx) / fs
    isi_s = isi_s[isi_s > 0]
    rates = 1.0 / isi_s

    xmax = np.percentile(rates, max_percentile)
    bins = np.linspace(0, xmax, 60)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.hist(rates, bins=bins, alpha=0.9)
    ax.set_xlabel("Instantaneous firing rate (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Firing-rate distribution (1/ISI)")
    _clean_axes(ax)

    mean_r, med_r = np.mean(rates), np.median(rates)
    ax.text(0.98, 0.95, f"Mean: {mean_r:.2f} Hz\nMedian: {med_r:.2f} Hz\nN ISIs: {len(isi_s)}",
            transform=ax.transAxes, ha="right", va="top")

    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def plot_amplitude_histogram(signal, thresh, out_path, bins=200):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.hist(signal, bins=bins, alpha=0.85, color="steelblue")
    ax.axvline(thresh, linestyle="--", linewidth=2.2, color="red",
               label="Threshold = median + 3×MAD")

    # Larger fonts
    ax.set_xlabel("Sample amplitude (z)", fontsize=20)
    ax.set_ylabel("Count", fontsize=20)
    ax.set_title("Amplitude distribution", fontsize=20)
    ax.tick_params(labelsize=20)
    ax.legend(frameon=False, fontsize=18)

    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def main():
    x, cols = load_signal_csv(csv_path, preferred_col="zscore_raw")
    if invert_signal: x = -x

    peak_idx, props, stats = detect_peaks(
        x, Fs, height_k=height_factor, prom_k=prominence_mul, min_dist_ms=min_distance_ms
    )

    print(f"Signal length: {x.size} samples ({x.size / Fs:.2f} s)")
    print(f"Detected peaks: {peak_idx.size}")

    theta = bandpass_theta(x, Fs, band=theta_band, order=theta_filt_order)

    out_root = Path(csv_path).with_suffix("")
    base_dir = out_root.parent / "optical_event_validation_csv"
    seg_dir  = base_dir / "segments_theta"
    base_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Segments
    plot_segments_with_peaks_and_theta(
        x, theta, peak_idx, Fs, seg_dir, seg_len_s=3.0, thresh=stats["height_thresh"]
    )

    # Firing-rate distribution
    plot_firing_rate_distribution(
        peak_idx, Fs, base_dir / "firing_rate_distribution.png"
    )

    # Amplitude distribution
    plot_amplitude_histogram(
        x, stats["height_thresh"], base_dir / "amplitude_histogram.png", bins=200
    )

    print(f"\nSaved segment figures to: {seg_dir}")
    print(f"Saved firing-rate histogram and amplitude histogram to: {base_dir}")


if __name__ == "__main__":
    main()
