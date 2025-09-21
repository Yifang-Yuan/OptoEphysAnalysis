# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:51:52 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
White noise vs 100 Hz high-pass filtered optical signal:
- High-pass (100 Hz) the optical signal before detection/analysis
- White noise surrogate of same length
- Amplitude KS, IEI KS, ACG theta permutation, PSD comparison (0–200 Hz),
  spike histogram, ACGs (large/small), rate ACFs (small/large)

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, butter, filtfilt, gaussian, convolve
from scipy.stats import median_abs_deviation, ks_2samp, kstest

# -------------------- User settings --------------------
csv_path       = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"
preferred_col  = "zscore_raw"
Fs             = 841.68
invert_signal  = True

# Filtering
APPLY_HIGHPASS_REAL  = True   # 100 Hz on optical signal (requested)
APPLY_HIGHPASS_NOISE = False  # set True to also HP the noise surrogate
HP_CUTOFF_HZ         = 100.0
HP_ORDER             = 3

# Peak detection (MAD rule)
height_factor   = 3.0
prominence_mul  = 2.0
min_distance_ms = 2.0

# Spike histogram
spike_hist_bin_s = 0.5

# ACG / rate ACF
acg_large_window_s = 0.5
acg_large_bin_s    = 0.002
acg_small_window_s = 0.02
acg_small_bin_s    = 0.002
rate_small_win_ms  = 10.0
rate_large_win_ms  = 250.0
rate_bin_ms        = 1.0
rate_smooth_ms     = 3.0

# Theta band + permutation
theta_band = (4.0, 12.0)
n_perm     = 500
RNG_SEED   = 0

# Output
out_dir = Path(csv_path).parent / "stats_HP100_vs_whitenoise"
# ------------------------------------------------------


# ============ helpers ============
def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.grid(False)

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
    distance_samples  = max(1, int(round((min_dist_ms/1000.0)*fs)))
    idx, props = find_peaks(x, height=height_thresh,
                               distance=distance_samples,
                               prominence=prominence_thresh)
    return idx, dict(baseline=baseline, mad=mad,
                     height_thresh=height_thresh,
                     prominence_thresh=prominence_thresh,
                     distance_samples=distance_samples)

def spike_times_to_acg(spike_times_s, window_s, bin_s, normalisation="rate"):
    st = np.asarray(spike_times_s, float); st.sort()
    n = st.size; half = window_s
    edges   = np.arange(-half, half + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    counts  = np.zeros_like(centers)
    j_left = 0
    for i in range(n):
        ti = st[i]
        while j_left < n and st[j_left] < ti - half:
            j_left += 1
        j = i + 1
        while j < n and st[j] <= ti + half:
            dt = st[j] - ti
            b  = int(np.floor((dt + half) / bin_s))
            if 0 <= b < counts.size:
                counts[b] += 1
            j += 1
    acg = counts[::-1] + counts
    if normalisation == "rate" and n > 0:
        acg = acg / (n * bin_s)  # Hz-like density
    return centers, acg

def poisson_spike_train(rate_hz, duration_s, rng):
    n_exp = rng.poisson(rate_hz * duration_s)
    times = rng.uniform(0.0, duration_s, size=n_exp)
    times.sort()
    return times

def band_power_from_acg(acg_vals, bin_s, f_band):
    x = np.asarray(acg_vals, float) - np.mean(acg_vals)
    fs_acg = 1.0 / bin_s
    freqs = np.fft.rfftfreq(x.size, d=1/fs_acg)
    spec  = np.abs(np.fft.rfft(x))**2
    mask  = (freqs >= f_band[0]) & (freqs <= f_band[1])
    return np.sum(spec[mask])

def welch_band_ratio(x, fs, band, full_band=(1.0, 100.0)):
    f, Pxx = welch(x, fs=fs, nperseg=min(8192, len(x)))
    band_mask = (f >= band[0]) & (f <= band[1])
    full_mask = (f >= full_band[0]) & (f <= full_band[1])
    num  = np.trapz(Pxx[band_mask], f[band_mask]) if np.any(band_mask) else 0.0
    denom= np.trapz(Pxx[full_mask], f[full_mask]) if np.any(full_mask) else 1e-12
    return num / denom, f, Pxx

# ---- plotting helpers ----
def plot_hist(data, thresh, title, out_path, xlabel="Sample amplitude (a.u.)"):
    fig, ax = plt.subplots(figsize=(8,5), dpi=160)
    ax.hist(data, bins=200, alpha=0.85, color="steelblue")
    if thresh is not None:
        ax.axvline(thresh, ls="--", lw=2, color="red", label="median + 3×MAD")
        ax.legend(frameon=False)
    ax.set_xlabel(xlabel); ax.set_ylabel("Count"); ax.set_title(title)
    _clean_axes(ax); fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def plot_spike_hist(spike_times_s, duration_s, bin_s, out_path, title):
    edges = np.arange(0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    fig, ax = plt.subplots(figsize=(8,5), dpi=160)
    ax.bar(edges[:-1], counts, width=bin_s, align="edge")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Number of events"); ax.set_title(title)
    _clean_axes(ax); ax.set_xlim(0, duration_s); ax.margins(x=0)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def plot_acg_bar(centers, acg, title, out_path, ylabel="Density (Hz)"):
    fig, ax = plt.subplots(figsize=(8,5), dpi=160)
    ax.bar(centers, acg, width=(centers[1]-centers[0]), align="center")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    _clean_axes(ax); ax.margins(x=0)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

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
    x = np.asarray(x, float)
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 0:
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
        lags_ms = lags_s[sel] * 1000.0
        vals    = acf[sel]
        fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=170)
        ax.bar(lags_ms, vals, width=bin_ms, align="center")
        ax.set_xlabel("time lag (ms)"); ax.set_ylabel("autocorr (a.u.)"); ax.set_title(title)
        _clean_axes(ax); ax.margins(x=0)
        fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    _plot(small_win_ms, out_small, f"Rate ACF (±{small_win_ms:.0f} ms)")
    _plot(large_win_ms, out_large, f"Rate ACF (±{large_win_ms:.0f} ms)")

def plot_psd_comparison(f_real, Pxx_real, f_noise, Pxx_noise,
                        title, out_path, theta=None, fmax=200):
    fig, ax = plt.subplots(figsize=(6,5), dpi=160)
    ax.semilogy(f_real,  Pxx_real,  color="C0", label="HP optical")
    ax.semilogy(f_noise, Pxx_noise, color="0.5", label="White noise")
    if theta is not None:
        ax.axvspan(theta[0], theta[1], color="orange", alpha=0.2, label="Theta band")
    ax.set_xlim(100, fmax)
    ax.set_ylim(10e-7, 10e-1)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD"); ax.set_title(title)
    ax.legend(frameon=False); _clean_axes(ax)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
# --- NEW: peak-shape extraction (width in seconds, prominence, height) ---
def peak_shape_stats(signal_hp, idx, fs):
    # Re-run peaks to fetch properties aligned to idx
    # (If you already have props from find_peaks, you can pass them instead.)
    from scipy.signal import peak_widths
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([])
    # half-prominence width
    results = peak_widths(signal_hp, idx, rel_height=0.5)
    width_samples = results[0]        # w at half-prominence (samples)
    width_s = width_samples / fs
    # heights at detected indices
    heights = signal_hp[idx]
    # crude prominence via local window if not available
    # (better: request props= from your original find_peaks call)
    return width_s, heights, None  # None placeholder for external prominence if you have it

# --- NEW: Fano factor over small bins ---
def fano_factor(spike_times_s, bin_s=0.02, duration_s=None):
    if duration_s is None:
        duration_s = float(spike_times_s[-1]) if len(spike_times_s) else 0.0
    edges = np.arange(0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    if counts.size < 2: return np.nan
    mu = counts.mean()
    return np.var(counts, ddof=1) / (mu + 1e-12)

# Poisson surrogate FF distribution
def fano_perm_pvalue(rate_hz, duration_s, n_perm=500, bin_s=0.02, rng=None, ff_obs=None):
    if rng is None: 
        rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_perm):
        n_exp = rng.poisson(rate_hz * duration_s)
        t = np.sort(rng.uniform(0.0, duration_s, size=n_exp))
        vals.append(fano_factor(t, bin_s, duration_s))
    vals = np.array(vals, float)
    if ff_obs is None or np.isnan(ff_obs):
        return np.nan, (np.nan, np.nan)
    p = (np.sum(vals >= ff_obs) + 1) / (n_perm + 1)
    return p, (vals.mean(), vals.std())

# --- NEW: small-lag ACG integral statistic (absolute area 1–10 ms) ---
def acg_smalllag_integral(centers, acg, lo_ms=1.0, hi_ms=10.0):
    lo, hi = lo_ms/1000.0, hi_ms/1000.0
    mask = (np.abs(centers) >= lo) & (np.abs(centers) <= hi)
    if not np.any(mask): return 0.0
    # trapezoid integral of |ACG|
    x = centers[mask]
    y = np.abs(acg[mask])
    return np.trapz(y, x)

def acg_perm_pvalue(rate_hz, duration_s, lo_ms=1.0, hi_ms=10.0, bin_s=0.002, window_s=0.02, n_perm=500, rng=None, obs_stat=None):
    if rng is None:
        rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_perm):
        st = np.sort(rng.uniform(0.0, duration_s, size=rng.poisson(rate_hz * duration_s)))
        c, a = spike_times_to_acg(st, window_s, bin_s)
        vals.append(acg_smalllag_integral(c, a, lo_ms, hi_ms))
    vals = np.array(vals, float)
    if obs_stat is None:
        return np.nan, (np.nan, np.nan)
    p = (np.sum(vals >= obs_stat) + 1) / (n_perm + 1)
    return p, (vals.mean(), vals.std())

# --- NEW: high-frequency PSD slope (log-log) in 100–200 Hz ---
def psd_slope_loglog(x, fs, f_lo=100.0, f_hi=200.0):
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=min(8192, len(x)))
    mask = (f >= f_lo) & (f <= f_hi)
    f_seg, P_seg = f[mask], P[mask]
    if np.sum(mask) < 5:
        return np.nan, (f, P)
    X = np.log10(f_seg + 1e-12)
    Y = np.log10(P_seg + 1e-24)
    # linear fit slope
    slope = np.polyfit(X, Y, deg=1)[0]
    return slope, (f, P)

def extract_peak_waveforms(x, peak_idx, fs, pre_ms=6.0, post_ms=6.0, normalise=None):
    """
    Extract snippets around peaks.
      x          : 1D array (signal used for detection, e.g., HP optical)
      peak_idx   : indices of peaks in x
      fs         : sampling rate (Hz)
      pre_ms     : ms before the peak
      post_ms    : ms after the peak
      normalise  : None | "peak"
                   - None: keep raw amplitudes
                   - "peak": divide each snippet by its own max value

    Returns
    -------
    snippets : (n_peaks, n_samples) array
    t_ms     : time axis for snippets (ms), length n_samples
    keep_idx : boolean mask of peaks kept (within bounds)
    """
    x = np.asarray(x, float)
    n = x.size
    pre_s   = pre_ms / 1000.0
    post_s  = post_ms / 1000.0
    pre_n   = int(np.round(pre_s  * fs))
    post_n  = int(np.round(post_s * fs))
    win_len = pre_n + post_n + 1
    if win_len < 3:
        raise ValueError("Snippet window too small; increase pre_ms/post_ms.")

    keep = (peak_idx - pre_n >= 0) & (peak_idx + post_n < n)
    good_peaks = peak_idx[keep]
    if good_peaks.size == 0:
        return np.empty((0, win_len), float), np.linspace(-pre_ms, post_ms, win_len), keep

    snippets = np.empty((good_peaks.size, win_len), float)
    for i, p in enumerate(good_peaks):
        seg = x[p - pre_n : p + post_n + 1]
        if normalise == "peak":
            denom = np.max(np.abs(seg)) if np.max(np.abs(seg)) > 0 else 1.0
            seg = seg / denom
        snippets[i, :] = seg

    t_ms = np.linspace(-pre_ms, post_ms, win_len)
    return snippets, t_ms, keep


def _sem(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    sd = np.nanstd(a, axis=axis, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))


def plot_mean_peak_shapes(x_real_hp, idx_real, x_noise_proc, idx_noise, fs,
                          pre_ms=6.0, post_ms=6.0, out_dir=Path(".")):
    """
    Make two plots:
      (A) mean raw-amplitude waveforms (±SEM)
      (B) mean peak-normalised waveforms (±SEM)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- (A) RAW amplitude ---
    sn_r_raw, t_ms, keep_r_raw = extract_peak_waveforms(x_real_hp,  idx_real,  fs, pre_ms, post_ms, normalise=None)
    sn_n_raw, _,    keep_n_raw = extract_peak_waveforms(x_noise_proc, idx_noise, fs, pre_ms, post_ms, normalise=None)

    # stats
    mu_r_raw  = np.nanmean(sn_r_raw, axis=0) if sn_r_raw.size else None
    sem_r_raw = _sem(sn_r_raw, axis=0)       if sn_r_raw.size else None
    mu_n_raw  = np.nanmean(sn_n_raw, axis=0) if sn_n_raw.size else None
    sem_n_raw = _sem(sn_n_raw, axis=0)       if sn_n_raw.size else None

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=170)
    if mu_r_raw is not None:
        ax.plot(t_ms, mu_r_raw, lw=2.0, color="C0", label=f"Highpass optical (n={sn_r_raw.shape[0]})")
        ax.fill_between(t_ms, mu_r_raw - sem_r_raw, mu_r_raw + sem_r_raw, alpha=0.25, color="C0")
    if mu_n_raw is not None:
        ax.plot(t_ms, mu_n_raw, lw=2.0, color="0.4", label=f"White noise (n={sn_n_raw.shape[0]})")
        ax.fill_between(t_ms, mu_n_raw - sem_n_raw, mu_n_raw + sem_n_raw, alpha=0.25, color="0.5")
    ax.axvline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Time from peak (ms)", fontsize=16)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=16)
    ax.set_title("Average peak waveform (raw amplitude ± SEM)", fontsize=18)
    _clean_axes(ax); ax.tick_params(labelsize=14)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_peak_shape_raw.png", bbox_inches="tight")
    plt.close(fig)

    # --- (B) Normalised to each-peak maximum ---
    sn_r_nrm, t_ms, keep_r_nrm = extract_peak_waveforms(x_real_hp,  idx_real,  fs, pre_ms, post_ms, normalise="peak")
    sn_n_nrm, _,    keep_n_nrm = extract_peak_waveforms(x_noise_proc, idx_noise, fs, pre_ms, post_ms, normalise="peak")

    mu_r_nrm  = np.nanmean(sn_r_nrm, axis=0) if sn_r_nrm.size else None
    sem_r_nrm = _sem(sn_r_nrm, axis=0)       if sn_r_nrm.size else None
    mu_n_nrm  = np.nanmean(sn_n_nrm, axis=0) if sn_n_nrm.size else None
    sem_n_nrm = _sem(sn_n_nrm, axis=0)       if sn_n_nrm.size else None

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=170)
    if mu_r_nrm is not None:
        ax.plot(t_ms, mu_r_nrm, lw=2.0, color="C0", label=f"Highpass optical (n={sn_r_nrm.shape[0]})")
        ax.fill_between(t_ms, mu_r_nrm - sem_r_nrm, mu_r_nrm + sem_r_nrm, alpha=0.25, color="C0")
    if mu_n_nrm is not None:
        ax.plot(t_ms, mu_n_nrm, lw=2.0, color="0.4", label=f"White noise (n={sn_n_nrm.shape[0]})")
        ax.fill_between(t_ms, mu_n_nrm - sem_n_nrm, mu_n_nrm + sem_n_nrm, alpha=0.25, color="0.5")
    ax.axvline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Time from peak (ms)", fontsize=16)
    ax.set_ylabel("Normalised amplitude (peak = 1)", fontsize=16)
    ax.set_title("Average peak waveform (normalised ± SEM)", fontsize=18)
    _clean_axes(ax); ax.tick_params(labelsize=14)
    ax.legend(frameon=False, fontsize=13, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_peak_shape_normalised.png", bbox_inches="tight")
    plt.close(fig)

    # (optional) return arrays for quantitative summaries if you want to save them
    return {
        "t_ms": t_ms,
        "real_raw_mean": mu_r_raw,  "real_raw_sem": sem_r_raw,  "n_real_raw": sn_r_raw.shape[0] if sn_r_raw.size else 0,
        "noise_raw_mean": mu_n_raw, "noise_raw_sem": sem_n_raw, "n_noise_raw": sn_n_raw.shape[0] if sn_n_raw.size else 0,
        "real_nrm_mean": mu_r_nrm,  "real_nrm_sem": sem_r_nrm,  "n_real_nrm": sn_r_nrm.shape[0] if sn_r_nrm.size else 0,
        "noise_nrm_mean": mu_n_nrm, "noise_nrm_sem": sem_n_nrm, "n_noise_nrm": sn_n_nrm.shape[0] if sn_n_nrm.size else 0,
    }


# ============ main ============
def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load & pre-process optical ---
    x_real = load_signal_csv(csv_path, preferred_col)
    if invert_signal: x_real = -x_real
    x_real_hp = highpass(x_real, Fs, HP_CUTOFF_HZ, HP_ORDER) if APPLY_HIGHPASS_REAL else x_real

    # --- White noise (same length) ---
    rng = np.random.default_rng(RNG_SEED)
    x_noise = rng.normal(0.0, 1.0, size=x_real.size)
    x_noise_proc = highpass(x_noise, Fs, HP_CUTOFF_HZ, HP_ORDER) if APPLY_HIGHPASS_NOISE else x_noise

    duration_s = x_real.size / Fs

    # --- Amplitude KS (HP optical vs noise as processed) ---
    ks_stat, ks_p = ks_2samp(x_real_hp, x_noise_proc)

    # --- Peak detection on processed signals ---
    idx_real,  stats_real  = detect_peaks(x_real_hp,  Fs, height_factor, prominence_mul, min_distance_ms)
    idx_noise, stats_noise = detect_peaks(x_noise_proc, Fs, height_factor, prominence_mul, min_distance_ms)
    t_real, t_noise = idx_real / Fs, idx_noise / Fs

    # --- Amplitude histograms (show thresholds used) ---
    plot_hist(x_real_hp,  stats_real["height_thresh"],  "Amplitude (HP optical)", out_dir/"amp_hist_hp_optical.png", xlabel="HP amplitude (a.u.)")
    plot_hist(x_noise_proc, stats_noise["height_thresh"], "Amplitude (white noise)", out_dir/"amp_hist_noise.png", xlabel="Amplitude (a.u.)")

    # --- Spike/event histograms over time ---
    plot_spike_hist(t_real,  duration_s, spike_hist_bin_s, out_dir/"spike_hist_hp_optical.png", "Event histogram (HP optical)")
    plot_spike_hist(t_noise, duration_s, spike_hist_bin_s, out_dir/"spike_hist_noise.png",       "Event histogram (white noise)")

    # --- ACGs (large/small) ---
    cL_r, aL_r = spike_times_to_acg(t_real,  acg_large_window_s, acg_large_bin_s)
    cS_r, aS_r = spike_times_to_acg(t_real,  acg_small_window_s, acg_small_bin_s)
    cL_n, aL_n = spike_times_to_acg(t_noise, acg_large_window_s, acg_large_bin_s)
    cS_n, aS_n = spike_times_to_acg(t_noise, acg_small_window_s, acg_small_bin_s)

    plot_acg_bar(cL_r, aL_r, f"ACG HP optical (±{acg_large_window_s}s, {acg_large_bin_s*1e3:.0f} ms bins)", out_dir/"acg_large_hp_optical.png")
    plot_acg_bar(cS_r, aS_r, f"ACG HP optical (±{acg_small_window_s}s, {acg_small_bin_s*1e3:.0f} ms bins)", out_dir/"acg_small_hp_optical.png")
    plot_acg_bar(cL_n, aL_n, f"ACG white noise (±{acg_large_window_s}s, {acg_large_bin_s*1e3:.0f} ms bins)", out_dir/"acg_large_noise.png")
    plot_acg_bar(cS_n, aS_n, f"ACG white noise (±{acg_small_window_s}s, {acg_small_bin_s*1e3:.0f} ms bins)", out_dir/"acg_small_noise.png")

    # --- Rate ACFs (small/large) ---
    plot_rate_acfs(idx_real,  Fs, out_dir/"rate_acf_small_hp_optical.png", out_dir/"rate_acf_large_hp_optical.png",
                   small_win_ms=rate_small_win_ms, large_win_ms=rate_large_win_ms,
                   bin_ms=rate_bin_ms, smooth_ms=rate_smooth_ms)
    plot_rate_acfs(idx_noise, Fs, out_dir/"rate_acf_small_noise.png", out_dir/"rate_acf_large_noise.png",
                   small_win_ms=rate_small_win_ms, large_win_ms=rate_large_win_ms,
                   bin_ms=rate_bin_ms, smooth_ms=rate_smooth_ms)

    # --- ACG theta-band permutation test (for HP optical) ---
    rng = np.random.default_rng(RNG_SEED)
    centres_r, acg_r = cL_r, aL_r
    theta_power_real = band_power_from_acg(acg_r, acg_large_bin_s, theta_band)
    rate_est = (len(t_real) / duration_s) if duration_s > 0 else 0.0
    perm_vals = []
    for _ in range(n_perm):
        st = poisson_spike_train(rate_est, duration_s, rng)
        _, acg_p = spike_times_to_acg(st, acg_large_window_s, acg_large_bin_s)
        perm_vals.append(band_power_from_acg(acg_p, acg_large_bin_s, theta_band))
    perm_vals = np.asarray(perm_vals)
    p_perm = (np.sum(perm_vals >= theta_power_real) + 1) / (n_perm + 1)

    # --- IEI ~ exponential KS (for both) ---
    def iei_ks(spike_idx):
        if spike_idx.size < 2: return np.nan, np.nan
        iei = np.diff(spike_idx) / Fs
        lam = 1.0 / np.mean(iei)
        stat, p = kstest(iei, 'expon', args=(0, 1.0/lam))
        return stat, p
    iei_stat_real,  iei_p_real  = iei_ks(idx_real)
    iei_stat_noise, iei_p_noise = iei_ks(idx_noise)

    # --- PSD (0–200 Hz), noise in grey ---
    ratio_real,  f_r, Pxx_r = welch_band_ratio(x_real_hp,  Fs, theta_band)
    ratio_noise, f_n, Pxx_n = welch_band_ratio(x_noise_proc, Fs, theta_band)
    plot_psd_comparison(f_r, Pxx_r, f_n, Pxx_n,
                        "PSD Comparison (HP optical vs white noise)",
                        out_dir / "psd_comparison_0_200Hz.png",
                        theta=theta_band, fmax=200)

    # ---- summary ----
    
    # Peak shape (optical HP vs noise)
    w_real_s, h_real, _ = peak_shape_stats(x_real_hp, idx_real, Fs)
    w_noise_s, h_noise, _ = peak_shape_stats(x_noise_proc, idx_noise, Fs)
    
    ks_w,  p_w  = ks_2samp(w_real_s,  w_noise_s)  if len(w_real_s)  and len(w_noise_s)  else (np.nan, np.nan)
    ks_h,  p_h  = ks_2samp(h_real,    h_noise)    if len(h_real)    and len(h_noise)    else (np.nan, np.nan)
    
    # Fano factor (20 ms bins by default)
    bin_s_ff = 0.02
    ff_real  = fano_factor(t_real,  bin_s_ff, duration_s)
    ff_noise = fano_factor(t_noise, bin_s_ff, duration_s)
    rate_est = (len(t_real) / duration_s) if duration_s > 0 else 0.0
    p_ff, (ff_perm_mu, ff_perm_sd) = fano_perm_pvalue(rate_est, duration_s, n_perm=500,
                                                      bin_s=bin_s_ff, rng=np.random.default_rng(0),
                                                      ff_obs=ff_real)
    
    # Short-lag ACG integral (1–10 ms) and permutation p
    c_small_r, a_small_r = spike_times_to_acg(t_real, window_s=0.02, bin_s=0.002)  # ±20 ms, 2 ms bins
    stat_acg = acg_smalllag_integral(c_small_r, a_small_r, 1.0, 10.0)
    p_acg_small, (acg_mu, acg_sd) = acg_perm_pvalue(rate_est, duration_s, lo_ms=1.0, hi_ms=10.0,
                                                    bin_s=0.002, window_s=0.02, n_perm=500,
                                                    rng=np.random.default_rng(1),
                                                    obs_stat=stat_acg)
    
    # High-frequency PSD slope 100–200 Hz
    slope_real, (f_r, Pxx_r)   = psd_slope_loglog(x_real_hp,  Fs, 100.0, 200.0)
    slope_noise, (f_n, Pxx_n)  = psd_slope_loglog(x_noise_proc, Fs, 100.0, 200.0)
    plot_psd_comparison(f_r, Pxx_r, f_n, Pxx_n,
                        "PSD Comparison (100–400 Hz)", out_dir/"psd_comparison_100_200Hz.png",
                        theta=None, fmax=400)  # theta=None because we’re in HP domain
    
    print("\n=== HP(100 Hz) Optical vs White Noise (HP-appropriate metrics) ===")
    print(f"[Amplitude KS (HP)]              D={ks_stat:.4f}, p={ks_p:.3e}")
    print(f"[Peak width (50% prom) KS]       D={ks_w:.4f},  p={p_w:.3e}   (median w: {np.median(w_real_s)*1e3:.2f} vs {np.median(w_noise_s)*1e3:.2f} ms)")
    print(f"[Peak height KS]                  D={ks_h:.4f},  p={p_h:.3e}   (median h: {np.median(h_real):.3f} vs {np.median(h_noise):.3f})")
    print(f"[Fano factor, {int(bin_s_ff*1e3)} ms bins]  real={ff_real:.2f}, noise={ff_noise:.2f}, perm μ≈{ff_perm_mu:.2f}±{ff_perm_sd:.2f}, p={p_ff:.4f}")
    print(f"[ACG small-lag area 1–10 ms]     obs={stat_acg:.3e}, perm μ≈{acg_mu:.3e}±{acg_sd:.3e}, p={p_acg_small:.4f}")
    print(f"[PSD slope 100–200 Hz]           real={slope_real:.3f}, noise={slope_noise:.3f}  (more negative = steeper drop)")
    print(f"Figures saved to: {out_dir}")
    
    # --- Average peak shapes: raw & normalised ---
    plot_mean_peak_shapes(
        x_real_hp, idx_real,
        x_noise_proc, idx_noise,
        Fs,
        pre_ms=6.0, post_ms=6.0,
        out_dir=out_dir
    )


if __name__ == "__main__":
    main()
