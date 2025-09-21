# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 13:08:11 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Compare z-scored GEVI vs 100 Hz high-pass GEVI:
- Peak detection with MAD rule on each signal
- Fano factor vs bin size curves for event trains
- Average peak waveforms (raw amplitude ± SEM) overlaid
- Average peak waveforms (peak-normalised ± SEM) overlaid

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, peak_widths
from scipy.stats import median_abs_deviation
# -------------------- plotting style --------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.6,
})

def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=5.5, width=1.4)
    ax.grid(False)

# -------------------- user settings --------------------
csv_path      = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"
preferred_col = "zscore_raw"     # fall back to first numeric if missing
Fs            = 841.68           # Hz
invert_signal = True             # your GEVI polarity (invert if spikes are down)

# Peak detection (your method)
height_factor   = 3.0            # threshold = median + 3×MAD
prominence_mul  = 2.0            # prominence ≥ 2×MAD
min_distance_ms = 2.0            # ≥ 2 ms apart

# HP filter
HP_CUTOFF = 100.0
HP_ORDER  = 3

# Waveform snippets for averaging
pre_ms, post_ms = 6.0, 6.0

# Fano factor bins to scan (ms)
ff_bins_ms = [2, 5, 10, 20, 30, 50, 75, 100, 150, 200]
# ------------------------------------------------------


# ================= helpers =================
def load_signal_csv(csv_file, preferred="zscore_raw"):
    df = pd.read_csv(csv_file)
    if preferred in df.columns:
        s = df[preferred]
    else:
        num = df.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            raise ValueError("No numeric columns in the CSV.")
        s = df[num[0]]
    x = s.to_numpy(float)
    if np.any(np.isnan(x)):
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    return x

def highpass(x, fs, cutoff_hz, order=3):
    nyq = fs / 2.0
    wn = cutoff_hz / nyq
    if not (0 < wn < 1):
        raise ValueError("High-pass cutoff must be (0, Nyquist).")
    b, a = butter(order, wn, btype="high")
    return filtfilt(b, a, x)

def detect_peaks(signal, fs, height_k=3.0, prom_k=2.0, min_dist_ms=2.0):
    x = np.asarray(signal, float)
    baseline = np.median(x)
    mad      = median_abs_deviation(x, scale=1.0)
    height_thresh     = baseline + height_k * mad
    prominence_thresh = prom_k * mad
    distance_samples  = max(1, int(round((min_dist_ms / 1000.0) * fs)))
    idx, props = find_peaks(
        x,
        height=height_thresh,
        distance=distance_samples,
        prominence=prominence_thresh
    )
    return idx, dict(baseline=baseline, mad=mad,
                     height_thresh=height_thresh,
                     prominence_thresh=prominence_thresh,
                     distance_samples=distance_samples)

def fano_factor(spike_times_s, bin_s, duration_s=None):
    if duration_s is None:
        duration_s = float(spike_times_s[-1]) if len(spike_times_s) else 0.0
    if duration_s <= 0.0:
        return np.nan
    edges = np.arange(0.0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    if counts.size < 2:
        return np.nan
    mu = counts.mean()
    return np.var(counts, ddof=1) / (mu + 1e-12)

def fano_curve(spike_idx, fs, bins_ms, duration_s):
    st = spike_idx / fs
    vals = []
    for b in bins_ms:
        vals.append(fano_factor(st, b/1000.0, duration_s))
    return np.array(vals, float)

def extract_peak_snippets(x, peak_idx, fs, pre_ms=6.0, post_ms=6.0, normalise=None):
    """
    normalise: None | "peak" (divide each snippet by its own max |abs|)
    """
    x = np.asarray(x, float)
    n = x.size
    pre_n  = int(round(pre_ms/1000.0 * fs))
    post_n = int(round(post_ms/1000.0 * fs))
    win_len = pre_n + post_n + 1
    if win_len < 3:
        raise ValueError("Increase pre_ms/post_ms.")
    keep = (peak_idx - pre_n >= 0) & (peak_idx + post_n < n)
    p = peak_idx[keep]
    if p.size == 0:
        t_ms = np.linspace(-pre_ms, post_ms, win_len)
        return np.empty((0, win_len)), t_ms
    snip = np.empty((p.size, win_len), float)
    for i, k in enumerate(p):
        seg = x[k-pre_n:k+post_n+1]
        if normalise == "peak":
            denom = np.max(np.abs(seg)) if np.max(np.abs(seg)) > 0 else 1.0
            seg = seg / denom
        snip[i, :] = seg
    t_ms = np.linspace(-pre_ms, post_ms, win_len)
    return snip, t_ms

def _sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    sd = np.nanstd(a, axis=axis, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))

# ================= plotting =================
def plot_fano_curves(bins_ms, ff_raw, ff_hp, out_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=170)
    ax.plot(bins_ms, ff_raw, "-o", lw=2.0, ms=5.5, label="z-score")
    ax.plot(bins_ms, ff_hp,  "-o", lw=2.0, ms=5.5, label="highpass")
    ax.axhline(1.0, color="0.6", lw=1.5, ls="--", label="Poisson (FF=1)")
    ax.set_xlabel("Bin size (ms)")
    ax.set_ylabel("Fano factor (variance/mean)")
    ax.set_title("Fano factor vs bin size")
    ax.set_xscale("log")
    ax.set_xticks(bins_ms); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, loc="best")
    _clean_axes(ax); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def plot_peak_waveforms_overlaid(snip_raw, snip_hp, t_ms, out_dir):
    # (A) raw amplitude
    fig, ax = plt.subplots(figsize=(9, 6), dpi=170)
    if snip_raw.size:
        mu = np.nanmean(snip_raw, axis=0); se = _sem(snip_raw, axis=0)
        ax.plot(t_ms, mu, color="C0", lw=2.2, label=f"z-score (n={snip_raw.shape[0]})")
        ax.fill_between(t_ms, mu-se, mu+se, color="C0", alpha=0.25)
    if snip_hp.size:
        mu = np.nanmean(snip_hp, axis=0); se = _sem(snip_hp, axis=0)
        ax.plot(t_ms, mu, color="C2", lw=2.2, label=f"highpass (n={snip_hp.shape[0]})")
        ax.fill_between(t_ms, mu-se, mu+se, color="C2", alpha=0.25)
    ax.axvline(0, color="k", lw=1.0, alpha=0.7)
    ax.set_xlabel("Time from peak (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("Average peak waveform (raw amplitude ± SEM)")
    ax.legend(frameon=False, loc="upper right")
    _clean_axes(ax); fig.tight_layout()
    fig.savefig(Path(out_dir) / "avg_peak_shape_raw_GEVI_vs_HP.png", bbox_inches="tight")
    plt.close(fig)

    # (B) peak-normalised
    fig, ax = plt.subplots(figsize=(9, 6), dpi=170)
    if snip_raw.size:
        sn = snip_raw / np.maximum(np.max(np.abs(snip_raw), axis=1, keepdims=True), 1e-12)
        mu = np.nanmean(sn, axis=0); se = _sem(sn, axis=0)
        ax.plot(t_ms, mu, color="C0", lw=2.2, label=f"z-score (n={snip_raw.shape[0]})")
        ax.fill_between(t_ms, mu-se, mu+se, color="C0", alpha=0.25)
    if snip_hp.size:
        sn = snip_hp / np.maximum(np.max(np.abs(snip_hp), axis=1, keepdims=True), 1e-12)
        mu = np.nanmean(sn, axis=0); se = _sem(sn, axis=0)
        ax.plot(t_ms, mu, color="C2", lw=2.2, label=f"highpass (n={snip_hp.shape[0]})")
        ax.fill_between(t_ms, mu-se, mu+se, color="C2", alpha=0.25)
    ax.axvline(0, color="k", lw=1.0, alpha=0.7)
    ax.set_xlabel("Time from peak (ms)")
    ax.set_ylabel("Normalised amplitude (peak = 1)")
    ax.set_title("Average peak waveform (normalised ± SEM)")
    ax.legend(frameon=False, loc="upper right")
    _clean_axes(ax); fig.tight_layout()
    fig.savefig(Path(out_dir) / "avg_peak_shape_norm_GEVI_vs_HP.png", bbox_inches="tight")
    plt.close(fig)

# ================= main =================
def main():
    out_dir = Path(csv_path).parent / "zscore_vs_HP100_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & prep
    x_raw = load_signal_csv(csv_path, preferred_col)
    if invert_signal:
        x_raw = -x_raw
    x_hp  = highpass(x_raw, Fs, HP_CUTOFF, HP_ORDER)

    duration_s = len(x_raw) / Fs

    # 2) Detect peaks on each signal *independently*
    idx_raw, stats_raw = detect_peaks(
        x_raw, Fs, height_k=height_factor,
        prom_k=prominence_mul, min_dist_ms=min_distance_ms
    )
    idx_hp, stats_hp = detect_peaks(
        x_hp, Fs, height_k=height_factor,
        prom_k=prominence_mul, min_dist_ms=min_distance_ms
    )

    print(f"Samples: {len(x_raw)}  ({duration_s:.2f} s @ {Fs} Hz)")
    print(f"RAW peaks: {len(idx_raw)}   [baseline={stats_raw['baseline']:.3f}, MAD={stats_raw['mad']:.3f}, th={stats_raw['height_thresh']:.3f}]")
    print(f"HP  peaks: {len(idx_hp)}    [baseline={stats_hp['baseline']:.3f}, MAD={stats_hp['mad']:.3f}, th={stats_hp['height_thresh']:.3f}]")

    # 3) Fano factor vs bin size
    ff_raw = fano_curve(idx_raw, Fs, ff_bins_ms, duration_s)
    ff_hp  = fano_curve(idx_hp,  Fs, ff_bins_ms, duration_s)
    plot_fano_curves(ff_bins_ms, ff_raw, ff_hp, out_dir / "fano_vs_bins.png")

    # 4) Average peak waveforms (use the signal used for detection)
    sn_raw, t_ms = extract_peak_snippets(x_raw, idx_raw, Fs, pre_ms, post_ms, normalise=None)
    sn_hp,  _    = extract_peak_snippets(x_hp,  idx_hp,  Fs, pre_ms, post_ms, normalise=None)
    plot_peak_waveforms_overlaid(sn_raw, sn_hp, t_ms, out_dir)

    # 5) (Optional) print quick morphology summaries
    if len(idx_raw):
        w_raw_s = peak_widths(x_raw, idx_raw, rel_height=0.5)[0] / Fs
        print(f"RAW width@50% (median): {np.median(w_raw_s)*1e3:.2f} ms")
    if len(idx_hp):
        w_hp_s  = peak_widths(x_hp,  idx_hp,  rel_height=0.5)[0] / Fs
        print(f"HP  width@50% (median): {np.median(w_hp_s)*1e3:.2f} ms")

    print(f"\nSaved figures in: {out_dir}")

if __name__ == "__main__":
    main()
