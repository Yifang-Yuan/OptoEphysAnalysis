# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 13:53:21 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Spike-like peak analysis from GEVI optical signal (CSV):
- Optional 100 Hz high-pass before peak detection
- Detect peaks with MAD rule
- Plot: (1) spike histogram over time, (2) autocorrelogram (large window),
        (3) autocorrelogram (small window)

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt   # <-- added butter, filtfilt
from scipy.stats import median_abs_deviation

# -------------------- global plot style --------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.4,
})

def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.grid(False)


# -------------------- user settings --------------------
csv_path = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"
# If running on the uploaded file in this workspace instead, use:
# csv_path = "/mnt/data/Zscore_traceAll.csv"

Fs = 841.68
invert_signal = True

# --- High-pass for detection ---
APPLY_HIGHPASS = True     # set False to disable
HP_CUTOFF_HZ   = 100.0    # 100 Hz high-pass
HP_ORDER       = 3

# Peak detection (your method)
height_factor   = 3      # threshold = median + height_factor × MAD
prominence_mul  = 2.0      # prominence ≥ 2×MAD
min_distance_ms = 0        # ≥ 0 ms apart (you set this; 2 ms is typical)

# Spike histogram bin (time)
spike_hist_bin_s = 0.5     # seconds

# Autocorrelogram settings
acg_large_window_s = 0.5   # ±0.5 s
acg_large_bin_s    = 0.002 # 2 ms bins
acg_small_window_s = 0.02  # ±20 ms
acg_small_bin_s    = 0.002 # 2 ms bins

# Normalisation for ACG: "count" or "rate"
acg_norm = "rate"

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


def highpass(x, fs, cutoff_hz=100.0, order=3):
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    if not (0 < wn < 1):
        raise ValueError("High-pass cutoff must be between 0 and Nyquist.")
    b, a = butter(order, wn, btype="high")
    return filtfilt(b, a, x)


def detect_peaks(signal, fs, height_k=3.0, prom_k=2.0, min_dist_ms=2.0):
    """
    MAD-based detection on the provided signal (raw or high-passed).
    """
    x = np.asarray(signal, float)
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
        baseline=baseline, mad=mad,
        height_thresh=height_thresh,
        prominence_thresh=prominence_thresh,
        distance_samples=distance_samples
    )


def spike_histogram(spike_times_s, duration_s, bin_s, out_path):
    edges = np.arange(0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.bar(edges[:-1], counts, width=bin_s, align="edge")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of events")
    ax.set_title("Histogram of optical-events")
    _clean_axes(ax)
    ax.set_xlim(0, duration_s)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def autocorrelogram(spike_times_s, window_s, bin_s, normalisation="rate"):
    """
    Return lags (bin centres) and ACG values (excludes self-count at 0 lag).
    """
    st = np.asarray(spike_times_s, dtype=float)
    st.sort()
    n = st.size
    half_w = window_s

    edges = np.arange(-half_w, half_w + bin_s, bin_s)
    centres = (edges[:-1] + edges[1:]) / 2
    acg_counts = np.zeros(centres.size, dtype=float)

    # Two-pointer sweep over positive lags, then mirror
    j_left = 0
    for i in range(n):
        t_i = st[i]
        while j_left < n and st[j_left] < t_i - half_w:
            j_left += 1
        j = i + 1
        while j < n and st[j] <= t_i + half_w:
            dt = st[j] - t_i
            bin_idx = int(np.floor((dt + half_w) / bin_s))
            if 0 <= bin_idx < acg_counts.size:
                acg_counts[bin_idx] += 1
            j += 1

    acg_full = acg_counts[::-1] + acg_counts

    if normalisation == "rate":
        acg_full = acg_full / (n * bin_s)

    return centres, acg_full


def plot_acg(centres, acg, title, out_path, x_unit="s", custom_ticks=None):
    """
    centres: bin centres in seconds
    x_unit: "s" or "ms"
    custom_ticks: list of tick positions (in the chosen unit) to force
    """
    if x_unit == "ms":
        x = centres * 1e3
        bar_w = (centres[1] - centres[0]) * 1e3
        xlabel = "Lag (ms)"
    else:
        x = centres
        bar_w = (centres[1] - centres[0])
        xlabel = "Lag (s)"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.bar(x, acg, width=bar_w, align="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if acg_norm == "rate" else "Count")
    ax.set_title(title)
    _clean_axes(ax)

    # apply custom ticks if provided
    if custom_ticks is not None:
        ax.set_xticks(custom_ticks)

    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    # 1) load and (optionally) invert
    x, cols = load_signal_csv(csv_path, preferred_col="zscore_raw")
    if invert_signal:
        x = -x

    duration_s = x.size / Fs

    # 2) optional high-pass BEFORE detection (and thus before 'firing rate'/ACG/hist)
    x_detect = highpass(x, Fs, HP_CUTOFF_HZ, HP_ORDER) if APPLY_HIGHPASS else x

    # 3) detect peaks on the chosen signal
    peak_idx, props, stats = detect_peaks(
        x_detect, Fs, height_k=height_factor,
        prom_k=prominence_mul, min_dist_ms=min_distance_ms
    )
    spike_times_s = peak_idx / Fs

    print(f"Columns: {cols}")
    print(f"Duration: {duration_s:.2f} s, Samples: {x.size}")
    print(f"Detected peaks: {peak_idx.size}")
    print(
        f"{'HP ' if APPLY_HIGHPASS else ''}baseline={stats['baseline']:.4f}, "
        f"{'HP ' if APPLY_HIGHPASS else ''}MAD={stats['mad']:.4f}, "
        f"thr={stats['height_thresh']:.4f}, prom={stats['prominence_thresh']:.4f}, "
        f"min_dist={stats['distance_samples']} samples"
    )

    # 4) output folder
    out_root = Path(csv_path).with_suffix("")
    tag = "hp100" if APPLY_HIGHPASS else "raw"
    out_dir = out_root.parent / f"spike_stats_from_optical_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) (1) Histogram of events over time
    spike_histogram(spike_times_s, duration_s, spike_hist_bin_s,
                    out_dir / "spike_histogram_over_time.png")

    # 6) (2) ACG large window
    lag_c_large, acg_large = autocorrelogram(
        spike_times_s, window_s=acg_large_window_s,
        bin_s=acg_large_bin_s, normalisation=acg_norm
    )
    plot_acg(lag_c_large, acg_large,
         f"Autocorrelogram (±{acg_large_window_s}s, bin {acg_large_bin_s*1e3:.0f} ms)",
         out_dir / "acg_large_window.png",
         x_unit="s")


    # 7) (3) ACG small window
    lag_c_small, acg_small = autocorrelogram(
        spike_times_s, window_s=acg_small_window_s,
        bin_s=acg_small_bin_s, normalisation=acg_norm
    )
    plot_acg(
    lag_c_small, acg_small,
    f"Autocorrelogram (±{acg_small_window_s}s, bin {acg_small_bin_s*1e3:.0f} ms)",
    out_dir / "acg_small_window.png",
    x_unit="ms",
    custom_ticks=[-20, -10, 0, 10, 20]
    )
    print(f"\nSaved figures in: {out_dir}")


if __name__ == "__main__":
    main()
