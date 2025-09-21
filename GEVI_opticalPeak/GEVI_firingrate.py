# -*- coding: utf-8 -*-
"""
Spike-like peak analysis from GEVI optical signal (CSV):
- Detect peaks with MAD rule
- Plot: (1) spike histogram over time, (2) autocorrelogram (large window), (3) autocorrelogram (small window)

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

# Peak detection (your method)
height_factor   = 3      # threshold = median + 3×MAD
prominence_mul  = 2.0      # prominence ≥ 2×MAD
min_distance_ms = 0      # ≥ 2 ms apart

# Spike histogram bin (time)
spike_hist_bin_s = 0.5     # seconds; change to taste

# Autocorrelogram settings
# "Large window" vs "Small window"
acg_large_window_s = 0.5    # ±0.5 s
acg_large_bin_s    = 0.002  # 2 ms bins
acg_small_window_s = 0.02  # ±50 ms
acg_small_bin_s    = 0.002  # 1 ms bins

# Normalisation for ACG: "count" or "rate"
#   "count" = raw counts per bin
#   "rate"  = counts / (N_spikes * bin_s)  → approximate probability density (Hz-like units)
acg_norm = "rate"

# Optional stub for theta (kept for future use; does nothing if False)
plot_theta = False
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
    Return lags (bin centres) and ACG values.
    Efficient sliding-window method; excludes 0-lag (self) counts.
    """
    st = np.asarray(spike_times_s, dtype=float)
    st.sort()
    n = st.size
    half_w = window_s

    # Bin edges for ±window
    edges = np.arange(-half_w, half_w + bin_s, bin_s)
    centres = (edges[:-1] + edges[1:]) / 2
    acg_counts = np.zeros(centres.size, dtype=float)

    # Two-pointer sweep
    j_left = 0
    for i in range(n):
        t_i = st[i]
        # advance left pointer to maintain st[j_left] >= t_i - half_w
        while j_left < n and st[j_left] < t_i - half_w:
            j_left += 1
        # examine neighbours to the right within window
        j = i + 1
        while j < n and st[j] <= t_i + half_w:
            dt = st[j] - t_i  # positive lag
            bin_idx = int(np.floor((dt + half_w) / bin_s))
            if 0 <= bin_idx < acg_counts.size:
                acg_counts[bin_idx] += 1
            j += 1

    # Mirror to negative lags
    acg_counts_neg = acg_counts[::-1]
    acg_full = acg_counts_neg + acg_counts
    # Remove the centre bin (0 lag) if it exists (it will be split across ± due to mirroring)
    # With our construction, 0-lag sits between bins; no explicit self-count remains.

    if normalisation == "rate":
        # counts per reference spike per second (≈ probability density)
        acg_full = acg_full / (st.size * bin_s)

    return centres, acg_full


# --- replace your current plot_acg() with this ---
# --- tweak plot_acg ---
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


# --- add: helpers to build a firing-rate series and its autocorrelation ----
from scipy.signal import gaussian, convolve

def make_rate_series(spike_idx, fs_signal,
                     bin_ms=1.0,
                     smooth_ms=3.0):
    """
    Build a firing-rate (Hz) time-series from spike indices.
    - Bin width = bin_ms (default 1 ms)
    - Convolve with a Gaussian kernel of std = smooth_ms
    Returns: rate (np.ndarray), fs_rate (Hz)
    """
    if len(spike_idx) == 0:
        return np.zeros(1, dtype=float), 1.0

    bin_s   = bin_ms / 1000.0
    fs_rate = 1.0 / bin_s
    n_bins  = int(np.ceil(spike_idx[-1] / fs_signal / bin_s)) + 1

    counts = np.zeros(n_bins, dtype=float)
    # convert spike indices to rate bins
    bin_idx = (spike_idx / fs_signal / bin_s).astype(int)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]
    np.add.at(counts, bin_idx, 1.0)

    # convert to Hz
    rate = counts / bin_s

    # Gaussian smoothing (optional; set smooth_ms<=0 to disable)
    if smooth_ms and smooth_ms > 0:
        sigma_bins = smooth_ms / bin_ms
        # kernel length ~ 6σ (odd)
        half = int(np.ceil(3 * sigma_bins))
        win  = gaussian(2 * half + 1, std=sigma_bins)
        win /= win.sum()
        rate = convolve(rate, win, mode="same")

    return rate, fs_rate


def autocorr_normalised(x):
    """
    Normalised (Pearson) autocorrelation of a 1D signal.
    Returns the full ACF (lags from -(N-1) to +(N-1)) and index of 0 lag.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:
        acf = np.zeros(1, dtype=float)
        return acf, 0
    acf_full = np.correlate(x, x, mode="full") / var
    zero_idx = len(acf_full) // 2
    return acf_full, zero_idx


def plot_rate_acfs(spike_idx, fs_signal, out_path_small, out_path_large,
                   small_win_ms=10.0, large_win_ms=250.0,
                   bin_ms=1.0, smooth_ms=3.0):
    """
    Build rate(t), compute its normalised ACF, and save two panels:
      - small window (±small_win_ms)
      - large window (±large_win_ms)
    X-axis in ms; Y-axis is autocorrelation (a.u., dimensionless).
    """
    # 1) rate series (Hz)
    rate, fs_rate = make_rate_series(spike_idx, fs_signal, bin_ms=bin_ms, smooth_ms=smooth_ms)

    # 2) autocorrelation
    acf, zero = autocorr_normalised(rate)

    # helper to slice ACF to window in ms
    def plot_window(win_ms, out_path, title):
        win_s  = win_ms / 1000.0
        # convert ACF index offsets to time
        lags_s = np.arange(-zero, len(acf)-zero) / fs_rate
        sel    = np.where((lags_s >= -win_s) & (lags_s <= win_s))[0]
        lags_ms = lags_s[sel] * 1000.0
        vals    = acf[sel]

        fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=170)
        ax.bar(lags_ms, vals, width=(bin_ms), align="center")  # bar width ~ bin size in ms
        ax.set_xlabel("time lag (ms)")
        ax.set_ylabel("autocorr (a.u.)")
        ax.set_title(title)
        _clean_axes(ax)
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    # 3) plots
    plot_window(small_win_ms, out_path_small,
                f"Rate ACF (±{small_win_ms:.0f} ms)")
    plot_window(large_win_ms, out_path_large,
                f"Rate ACF (±{large_win_ms:.0f} ms)")


def main():
    # 1) load and (optionally) invert
    x, cols = load_signal_csv(csv_path, preferred_col="zscore_raw")
    if invert_signal:
        x = -x

    duration_s = x.size / Fs

    # 2) detect peaks (spikes)
    peak_idx, props, stats = detect_peaks(
        x, Fs, height_k=height_factor,
        prom_k=prominence_mul, min_dist_ms=min_distance_ms
    )

    spike_times_s = peak_idx / Fs

    print(f"Columns: {cols}")
    print(f"Duration: {duration_s:.2f} s, Samples: {x.size}")
    print(f"Detected peaks (spikes): {peak_idx.size}")
    print(
        f"baseline={stats['baseline']:.4f}, MAD={stats['mad']:.4f}, "
        f"height_thresh={stats['height_thresh']:.4f}, "
        f"prom_thresh={stats['prominence_thresh']:.4f}, "
        f"min_distance={stats['distance_samples']} samples"
    )

    # 3) output folder
    out_root = Path(csv_path).with_suffix("")
    out_dir = out_root.parent / "spike_stats_from_optical"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) (1) Histogram of spikes over time
    spike_histogram(spike_times_s, duration_s, spike_hist_bin_s,
                    out_dir / "spike_histogram_over_time.png")

    # 5) (2) ACG large window
    lag_c_large, acg_large = autocorrelogram(
        spike_times_s, window_s=acg_large_window_s,
        bin_s=acg_large_bin_s, normalisation=acg_norm
    )
    plot_acg(lag_c_large, acg_large,
         f"Autocorrelogram (±{acg_large_window_s}s, bin {acg_large_bin_s*1e3:.0f} ms)",
         out_dir / "acg_large_window.png",
         x_unit="s")

    # 6) (3) ACG small window
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
    
    plot_rate_acfs(
    peak_idx, Fs,
    out_path_small = out_dir / "rate_acf_small.png",
    out_path_large = out_dir / "rate_acf_large.png",
    small_win_ms=10.0,
    large_win_ms=250.0,
    bin_ms=1.0,        # 1 ms bins for rate
    smooth_ms=3.0      # light smoothing; set to 0 to disable
)


if __name__ == "__main__":
    main()
