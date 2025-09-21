# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 15:26:57 2025

@author: yifan
"""
# -*- coding: utf-8 -*-
"""
Spike-like peak analysis (CSV or synthetic white noise):
- Generate white-noise trace and detect events with MAD rule
- Plot: (1) event histogram over time, (2) ACG large window, (3) ACG small window,
        (4) rate ACF small + large windows

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, gaussian, convolve
from scipy.stats import median_abs_deviation

# -------------------- fonts/style --------------------
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.linewidth": 1.4,
})

def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.grid(False)

# -------------------- switches --------------------
USE_WHITE_NOISE = True   # <- set False to analyse a CSV
csv_path = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"

# -------------------- signal params --------------------
Fs = 841.68                      # Hz
invert_signal = True

# Synthetic white-noise settings
DURATION_S   = 30.0              # seconds
NOISE_STD    = 1.0               # standard deviation
RNG_SEED     = 1                 # reproducible

# Peak detection (MAD rule)
height_factor   = 3.0            # threshold = median + k * MAD
prominence_mul  = 2.0            # prominence ≥ 2×MAD
min_distance_ms = 2.0            # enforce ≥ 2 ms between events

# Spike-time histogram
spike_hist_bin_s = 0.5           # seconds

# ACG windows/bins
acg_large_window_s = 0.5         # ±0.5 s
acg_large_bin_s    = 0.002       # 2 ms bins
acg_small_window_s = 0.02        # ±20 ms
acg_small_bin_s    = 0.002       # 2 ms bins
acg_norm           = "rate"      # "rate" -> counts/(N*bin_s); else "count"

# Rate ACF settings
rate_small_win_ms = 10.0
rate_large_win_ms = 250.0
rate_bin_ms       = 1.0          # 1 ms bins for rate(t)
rate_smooth_ms    = 3.0          # Gaussian σ (ms); set 0 to disable

# -------------------- IO --------------------
OUT_PARENT = Path.cwd() / "white_noise_demo"  # change if desired
OUT_PARENT.mkdir(parents=True, exist_ok=True)

# -------------------- helpers --------------------
def load_signal_csv(csv_file, preferred_col="zscore_raw"):
    df = pd.read_csv(csv_file)
    if preferred_col in df.columns:
        sig = df[preferred_col]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError("No numeric columns in CSV.")
        sig = df[num_cols[0]]
    x = sig.to_numpy(float)
    if np.any(np.isnan(x)):
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    return np.asarray(x, float)

def detect_peaks(signal, fs, height_k=3.0, prom_k=2.0, min_dist_ms=2.0):
    x = np.asarray(signal, float)
    baseline = np.median(x)
    mad = median_abs_deviation(x, scale=1.0)
    height_thresh     = baseline + height_k * mad
    prominence_thresh = prom_k * mad
    distance_samples  = max(1, int(round((min_dist_ms/1000.0)*fs)))
    idx, props = find_peaks(x, height=height_thresh,
                               distance=distance_samples,
                               prominence=prominence_thresh)
    return idx, dict(baseline=baseline, mad=mad,
                     height_thresh=height_thresh,
                     prominence_thresh=prominence_thresh,
                     distance_samples=distance_samples)

def spike_histogram(spike_times_s, duration_s, bin_s, out_path):
    edges = np.arange(0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.bar(edges[:-1], counts, width=bin_s, align="edge",color="grey")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Number of events")
    ax.set_title("Histogram of optical-events")
    _clean_axes(ax); ax.set_xlim(0, duration_s); ax.margins(x=0)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def autocorrelogram(spike_times_s, window_s, bin_s, normalisation="rate"):
    st = np.asarray(spike_times_s, float); st.sort()
    n = st.size; half_w = window_s
    edges   = np.arange(-half_w, half_w + bin_s, bin_s)
    centres = (edges[:-1] + edges[1:]) / 2
    counts  = np.zeros(centres.size, float)
    j_left = 0
    for i in range(n):
        t_i = st[i]
        while j_left < n and st[j_left] < t_i - half_w:
            j_left += 1
        j = i + 1
        while j < n and st[j] <= t_i + half_w:
            dt = st[j] - t_i
            b  = int(np.floor((dt + half_w) / bin_s))
            if 0 <= b < counts.size:
                counts[b] += 1
            j += 1
    acg = counts[::-1] + counts
    if normalisation == "rate":
        acg = acg / (n * bin_s)
    return centres, acg

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
    ax.bar(x, acg, width=bar_w, align="center",color='grey')
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
def make_rate_series(spike_idx, fs_signal, bin_ms=1.0, smooth_ms=3.0):
    if len(spike_idx) == 0:
        return np.zeros(1, float), 1.0
    bin_s   = bin_ms/1000.0
    fs_rate = 1.0/bin_s
    n_bins  = int(np.ceil(spike_idx[-1]/fs_signal/bin_s)) + 1
    counts  = np.zeros(n_bins, float)
    b_idx   = (spike_idx/fs_signal/bin_s).astype(int)
    b_idx   = b_idx[(b_idx>=0) & (b_idx<n_bins)]
    np.add.at(counts, b_idx, 1.0)
    rate = counts / bin_s
    if smooth_ms and smooth_ms > 0:
        sigma_bins = smooth_ms / bin_ms
        half = int(np.ceil(3*sigma_bins))
        win  = gaussian(2*half+1, std=sigma_bins); win /= win.sum()
        rate = convolve(rate, win, mode="same")
    return rate, fs_rate

def autocorr_normalised(x):
    x = np.asarray(x, float); x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:  # flat series
        return np.zeros(1, float), 0
    acf = np.correlate(x, x, mode="full") / var
    return acf, len(acf)//2

def plot_rate_acfs(spike_idx, fs_signal, out_small, out_large,
                   small_win_ms=10.0, large_win_ms=250.0,
                   bin_ms=1.0, smooth_ms=3.0):
    rate, fs_rate = make_rate_series(spike_idx, fs_signal, bin_ms, smooth_ms)
    acf, zero = autocorr_normalised(rate)
    def _plot(win_ms, out_path, title):
        lags_s = np.arange(-zero, len(acf)-zero) / fs_rate
        sel    = (lags_s >= -win_ms/1000.0) & (lags_s <= win_ms/1000.0)
        lags_ms = (lags_s[sel] * 1000.0); vals = acf[sel]
        fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=170)
        ax.bar(lags_ms, vals, width=bin_ms, align="center")
        ax.set_xlabel("time lag (ms)"); ax.set_ylabel("autocorr (a.u.)")
        ax.set_title(title); _clean_axes(ax); ax.margins(x=0)
        fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    _plot(small_win_ms, out_small, f"Rate ACF (±{small_win_ms:.0f} ms)")
    _plot(large_win_ms, out_large, f"Rate ACF (±{large_win_ms:.0f} ms)")
def plot_amplitude_histogram(signal, thresh, out_path, bins=200):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.hist(signal, bins=bins, alpha=0.85, color="grey")
    ax.axvline(thresh, linestyle="--", linewidth=2.2, color="red",
               label="Threshold = median + 3×MAD")
    ax.set_xlabel("Sample amplitude (z)")
    ax.set_ylabel("Count")
    ax.set_title("Amplitude distribution with MAD threshold")
    ax.legend(frameon=False, fontsize=12)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# -------------------- main --------------------
def main():
    # --- build signal ---
    if USE_WHITE_NOISE:
        rng = np.random.default_rng(RNG_SEED)
        n = int(round(DURATION_S * Fs))
        x = rng.normal(loc=0.0, scale=NOISE_STD, size=n).astype(float)
        tag = f"white_noise_{int(DURATION_S)}s"
        print(f"Generated white noise: n={n}, duration={DURATION_S:.2f}s, std={NOISE_STD}")
    else:
        x = load_signal_csv(csv_path)
        tag = "from_csv"
        print(f"Loaded CSV: {csv_path}")
    if invert_signal:
        x = -x

    duration_s = x.size / Fs

    # --- detect peaks ---
    peak_idx, stats = detect_peaks(
        x, Fs, height_k=height_factor, prom_k=prominence_mul, min_dist_ms=min_distance_ms
    )
    spike_times_s = peak_idx / Fs
    print(f"Detected peaks: {peak_idx.size}")
    print(f"baseline={stats['baseline']:.4f}, MAD={stats['mad']:.4f}, "
          f"thr={stats['height_thresh']:.4f}, prom={stats['prominence_thresh']:.4f}, "
          f"min_dist={stats['distance_samples']} samples")

    # --- output dir ---
    out_dir = OUT_PARENT / f"spike_stats_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) histogram over time
    spike_histogram(spike_times_s, duration_s, spike_hist_bin_s,
                    out_dir / "spike_histogram_over_time.png")

    # (2) ACG large
    cL, aL = autocorrelogram(spike_times_s, acg_large_window_s, acg_large_bin_s, acg_norm)
    plot_acg(cL, aL,
         f"Autocorrelogram (±{acg_large_window_s}s, bin {acg_large_bin_s*1e3:.0f} ms)",
         out_dir / "acg_large_window.png",
         x_unit="s")

    # (3) ACG small
    cS, aS = autocorrelogram(spike_times_s, acg_small_window_s, acg_small_bin_s, acg_norm)
    plot_acg(
    cS, aS,
    f"Autocorrelogram (±{acg_small_window_s}s, bin {acg_small_bin_s*1e3:.0f} ms)",
    out_dir / "acg_small_window.png",
    x_unit="ms",
    custom_ticks=[-20, -10, 0, 10, 20]
    )

    # # (4) Rate ACFs
    # plot_rate_acfs(
    #     peak_idx, Fs,
    #     out_small = out_dir / "rate_acf_small.png",
    #     out_large = out_dir / "rate_acf_large.png",
    #     small_win_ms=rate_small_win_ms, large_win_ms=rate_large_win_ms,
    #     bin_ms=rate_bin_ms, smooth_ms=rate_smooth_ms
    # )
    
    # --- amplitude histogram
    plot_amplitude_histogram(
        x, stats["height_thresh"],
        out_dir / "amplitude_distribution.png"
    )


    print(f"\nSaved figures to: {out_dir}")

if __name__ == "__main__":
    main()

