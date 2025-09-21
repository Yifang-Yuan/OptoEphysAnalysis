# -*- coding: utf-8 -*-
"""
Is the GEVI signal different from white noise?
Statistical comparison + plots.

Adds peak-level & dispersion metrics for the NO high-pass case:
- Peak width (50% prominence) KS test
- Peak height KS test
- Fano factor (20 ms bins) with permutation p-value
- ACG small-lag (1–10 ms) area with permutation p-value

Author: yifang + ChatGPT
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, butter, filtfilt, peak_widths
from scipy.stats import median_abs_deviation, ks_2samp, kstest

# ===================== User settings =====================
csv_path       = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\FiringRateExample\SyncRecording7\Zscore_traceAll.csv"
preferred_col  = "zscore_raw"          # fallback to first numeric column if missing
Fs             = 841.68                # Hz
invert_signal  = True                  # your GEVI polarity
APPLY_HIGHPASS = False                 # keep False for this script (no HP)
HP_CUTOFF_HZ   = 100.0
HP_ORDER       = 3

# Peak detection (MAD rule)
height_factor   = 3.0                  # thresh = median + k*MAD
prominence_mul  = 2.0                  # prominence >= 2*MAD
min_distance_ms = 2.0                  # refractory like constraint in ms

# ACG / spectral params
acg_window_s    = 0.5                  # +/- 0.5 s
acg_bin_s       = 0.002                # 2 ms bins
theta_band      = (4.0, 12.0)          # Hz (for ACG spectral power & PSD comparisons)
n_perm          = 500                  # permutations for ACG test

# Fano / small-lag ACG extra params
bin_s_ff        = 0.02                 # 20 ms bins for Fano factor
smalllag_lo_ms  = 1.0
smalllag_hi_ms  = 10.0
small_acg_win_s = 0.02                 # ±20 ms window for small-lag ACG
small_acg_bin_s = 0.002                # 2 ms bins

# White-noise surrogate
RNG_SEED        = 0

# Output
out_dir = Path(csv_path).parent / "signal_vs_whitenoise_stats"
# =========================================================

def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction="out", length=5, width=1.2)
    ax.grid(False)

# ------------------ helpers ------------------
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
    # props includes 'prominences', 'left_bases', 'right_bases' etc.
    return idx, props, dict(baseline=baseline, mad=mad,
                            height_thresh=height_thresh,
                            prominence_thresh=prominence_thresh,
                            distance_samples=distance_samples)

def poisson_spike_train(rate_hz, duration_s, rng):
    """Homogeneous Poisson process, returned as event times (s)."""
    n_exp = rng.poisson(rate_hz * duration_s)
    times = rng.uniform(0.0, duration_s, size=n_exp)
    times.sort()
    return times

def spike_times_to_acg(spike_times_s, window_s, bin_s):
    st = np.asarray(spike_times_s, float); st.sort()
    n = st.size
    half = window_s
    edges   = np.arange(-half, half + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2
    counts  = np.zeros_like(centers)
    # positive lags
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
    # rate normalisation (counts per ref spike per second)
    acg = acg / (n * bin_s) if n > 0 else acg
    return centers, acg

def band_power_from_acg(acg_vals, bin_s, f_band):
    """Theta power from ACG via FFT."""
    x = np.asarray(acg_vals, float)
    x = x - np.mean(x)
    fs_acg = 1.0 / bin_s
    freqs = np.fft.rfftfreq(x.size, d=1/fs_acg)
    spec  = np.abs(np.fft.rfft(x))**2
    band_mask = (freqs >= f_band[0]) & (freqs <= f_band[1])
    return np.sum(spec[band_mask])

def welch_band_ratio(x, fs, band, full_band=(1.0, 100.0)):
    f, Pxx = welch(x, fs=fs, nperseg=min(8192, len(x)))
    band_mask = (f >= band[0]) & (f <= band[1])
    full_mask = (f >= full_band[0]) & (f <= full_band[1])
    num  = np.trapz(Pxx[band_mask], f[band_mask]) if np.any(band_mask) else 0.0
    denom= np.trapz(Pxx[full_mask], f[full_mask]) if np.any(full_mask) else 1e-12
    return num / denom, f, Pxx

def plot_hist(data, thresh, title, out_path):
    fig, ax = plt.subplots(figsize=(8,5), dpi=160)
    ax.hist(data, bins=200, alpha=0.85, color="steelblue")
    if thresh is not None:
        ax.axvline(thresh, ls="--", lw=2, color="red", label="median + 3×MAD")
        ax.legend(frameon=False)
    ax.set_xlabel("Sample amplitude (z)"); ax.set_ylabel("Count")
    ax.set_title(title); _clean_axes(ax)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def plot_acg(centers, acg, title, out_path):
    fig, ax = plt.subplots(figsize=(8,5), dpi=160)
    ax.bar(centers, acg, width=(centers[1]-centers[0]), align="center")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("Density (Hz)")
    ax.set_title(title); _clean_axes(ax)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

def plot_psd_comparison(f_real, Pxx_real, f_noise, Pxx_noise,
                        title, out_path, theta=None, fmax=200):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.semilogy(f_real, Pxx_real, color="C0", label="Real signal")
    ax.semilogy(f_noise, Pxx_noise, color="0.5", alpha=0.6, label="White noise")
    if theta is not None:
        ax.axvspan(theta[0], theta[1], color="orange", alpha=0.2, label="Theta band")
    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title(title); ax.legend(frameon=False); _clean_axes(ax)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

# ---------- NEW: peak-level + dispersion helpers ----------
def peak_shape_stats(signal_1d, peak_idx, fs):
    """Return (width_s at 50% prominence, heights)."""
    if len(peak_idx) == 0:
        return np.array([]), np.array([])
    w_samples, _, _, _ = peak_widths(signal_1d, peak_idx, rel_height=0.5)
    w_s = w_samples / fs
    h   = np.asarray(signal_1d, float)[peak_idx]
    return w_s, h

def fano_factor(spike_times_s, bin_s=0.02, duration_s=None):
    if len(spike_times_s) == 0:
        return np.nan
    if duration_s is None:
        duration_s = float(spike_times_s[-1])
    edges = np.arange(0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    if counts.size < 2:
        return np.nan
    mu = counts.mean()
    return np.var(counts, ddof=1) / (mu + 1e-12)

def fano_perm_pvalue(rate_hz, duration_s, n_perm=500, bin_s=0.02, rng=None, ff_obs=None):
    if rng is None:
        rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_perm):
        n_exp = rng.poisson(rate_hz * duration_s)
        t = np.sort(rng.uniform(0.0, duration_s, size=n_exp))
        vals.append(fano_factor(t, bin_s, duration_s))
    vals = np.asarray(vals, float)
    if ff_obs is None or np.isnan(ff_obs):
        return np.nan, (np.nan, np.nan)
    p = (np.sum(vals >= ff_obs) + 1) / (n_perm + 1)
    return p, (vals.mean(), vals.std())

def acg_smalllag_integral(centers, acg, lo_ms=1.0, hi_ms=10.0):
    lo, hi = lo_ms/1000.0, hi_ms/1000.0
    mask = (np.abs(centers) >= lo) & (np.abs(centers) <= hi)
    if not np.any(mask):
        return 0.0
    x = centers[mask]
    y = np.abs(acg[mask])
    return np.trapz(y, x)

def acg_perm_pvalue(rate_hz, duration_s, lo_ms=1.0, hi_ms=10.0,
                    bin_s=0.002, window_s=0.02, n_perm=500, rng=None, obs_stat=None):
    if rng is None:
        rng = np.random.default_rng(1)
    vals = []
    for _ in range(n_perm):
        st = np.sort(rng.uniform(0.0, duration_s, size=rng.poisson(rate_hz * duration_s)))
        c, a = spike_times_to_acg(st, window_s, bin_s)
        vals.append(acg_smalllag_integral(c, a, lo_ms, hi_ms))
    vals = np.asarray(vals, float)
    if obs_stat is None:
        return np.nan, (np.nan, np.nan)
    p = (np.sum(vals >= obs_stat) + 1) / (n_perm + 1)
    return p, (vals.mean(), vals.std())

# ------------------ main pipeline ------------------
def main():
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real signal
    x_real = load_signal_csv(csv_path, preferred_col)
    if invert_signal:
        x_real = -x_real
    x_for_detect = highpass(x_real, Fs, HP_CUTOFF_HZ, HP_ORDER) if APPLY_HIGHPASS else x_real  # here: no HP

    # White noise surrogate of the SAME length
    rng = np.random.default_rng(RNG_SEED)
    x_noise = rng.normal(0.0, 1.0, size=x_real.size)

    # --- 1) Amplitude KS test (real vs noise) ---
    ks_stat, ks_p = ks_2samp(x_real, x_noise)

    # --- detect peaks for both ---
    idx_real, props_real, stats_real = detect_peaks(x_for_detect, Fs, height_factor, prominence_mul, min_distance_ms)
    idx_noise, props_noise, stats_noise = detect_peaks(x_noise, Fs, height_factor, prominence_mul, min_distance_ms)
    t_real  = idx_real / Fs
    t_noise = idx_noise / Fs
    duration_s = len(x_real) / Fs

    # Histograms (amplitude)
    plot_hist(x_real,  stats_real["height_thresh"], "Amplitude (real signal)", out_dir/"amp_hist_real.png")
    plot_hist(x_noise, stats_noise["height_thresh"], "Amplitude (white noise)", out_dir/"amp_hist_noise.png")

    # --- 2) IEI ~ exponential? (one-sample KS) ---
    def iei_ks(spike_idx):
        if spike_idx.size < 2:
            return np.nan, np.nan
        iei = np.diff(spike_idx) / Fs
        lam = 1.0 / np.mean(iei)
        # kstest against exponential with scale=1/λ, loc=0
        stat, p = kstest(iei, 'expon', args=(0, 1.0/lam))
        return stat, p

    iei_stat_real,  iei_p_real  = iei_ks(idx_real)
    iei_stat_noise, iei_p_noise = iei_ks(idx_noise)

    # --- 3) ACG theta-band structure (permutation test vs Poisson) ---
    centres_r, acg_r = spike_times_to_acg(t_real, acg_window_s, acg_bin_s)
    theta_power_real = band_power_from_acg(acg_r, acg_bin_s, theta_band)

    rate_est = (len(t_real) / duration_s) if duration_s > 0 else 0.0
    perm_vals = []
    for _ in range(n_perm):
        st = poisson_spike_train(rate_est, duration_s, rng)
        _, acg_p = spike_times_to_acg(st, acg_window_s, acg_bin_s)
        perm_vals.append(band_power_from_acg(acg_p, acg_bin_s, theta_band))
    perm_vals = np.asarray(perm_vals)
    p_perm = (np.sum(perm_vals >= theta_power_real) + 1) / (n_perm + 1)

    # Save ACG plots
    plot_acg(centres_r, acg_r, f"ACG (real, ±{acg_window_s}s, bin {acg_bin_s*1e3:.0f} ms)", out_dir/"acg_real.png")
    centres_n, acg_n = spike_times_to_acg(t_noise, acg_window_s, acg_bin_s)
    plot_acg(centres_n, acg_n, f"ACG (white noise, ±{acg_window_s}s, bin {acg_bin_s*1e3:.0f} ms)", out_dir/"acg_noise.png")

    # --- 4) PSD comparison (Welch) ---
    ratio_real, f_r, Pxx_r = welch_band_ratio(x_real, Fs, theta_band)
    ratio_noise, f_n, Pxx_n = welch_band_ratio(x_noise, Fs, theta_band)
    plot_psd_comparison(f_r, Pxx_r, f_n, Pxx_n,
                        "PSD Comparison (Real vs White noise)",
                        out_dir / "psd_comparison.png",
                        theta=theta_band, fmax=200)

    # -------- NEW: Peak-level stats + dispersion + small-lag ACG --------
    # Peak shapes measured on the same signals used for detection:
    w_real_s,  h_real  = peak_shape_stats(x_for_detect, idx_real,  Fs)
    w_noise_s, h_noise = peak_shape_stats(x_noise,     idx_noise, Fs)

    ks_w, p_w = ks_2samp(w_real_s, w_noise_s)   if len(w_real_s) and len(w_noise_s) else (np.nan, np.nan)
    ks_h, p_h = ks_2samp(h_real,   h_noise)     if len(h_real)   and len(h_noise)   else (np.nan, np.nan)

    # Fano factor and permutation against Poisson with same rate/duration
    ff_real  = fano_factor(t_real,  bin_s_ff, duration_s)
    ff_noise = fano_factor(t_noise, bin_s_ff, duration_s)
    p_ff, (ff_perm_mu, ff_perm_sd) = fano_perm_pvalue(rate_est, duration_s, n_perm=n_perm,
                                                      bin_s=bin_s_ff, rng=np.random.default_rng(0),
                                                      ff_obs=ff_real)

    # Small-lag ACG area 1–10 ms and permutation p-value
    c_small_r, a_small_r = spike_times_to_acg(t_real, window_s=small_acg_win_s, bin_s=small_acg_bin_s)
    stat_acg = acg_smalllag_integral(c_small_r, a_small_r,
                                     lo_ms=smalllag_lo_ms, hi_ms=smalllag_hi_ms)
    p_acg_small, (acg_mu, acg_sd) = acg_perm_pvalue(rate_est, duration_s,
                                                    lo_ms=smalllag_lo_ms, hi_ms=smalllag_hi_ms,
                                                    bin_s=small_acg_bin_s, window_s=small_acg_win_s,
                                                    n_perm=n_perm, rng=np.random.default_rng(1),
                                                    obs_stat=stat_acg)

    # ---------------- report ----------------
    print("\n=== STATISTICAL COMPARISON: Real vs White Noise (no HP) ===")
    print(f"[Amplitude KS]           D={ks_stat:.4f}, p={ks_p:.3e}")
    print(f"[IEI ~ Exp]   (real)     KS={iei_stat_real if not np.isnan(iei_stat_real) else np.nan:.4f}, "
          f"p={iei_p_real if not np.isnan(iei_p_real) else np.nan:.3e}, n_spikes={len(idx_real)}")
    print(f"[IEI ~ Exp]   (noise)    KS={iei_stat_noise if not np.isnan(iei_stat_noise) else np.nan:.4f}, "
          f"p={iei_p_noise if not np.isnan(iei_p_noise) else np.nan:.3e}, n_spikes={len(idx_noise)}")
    print(f"[ACG theta power]        real={theta_power_real:.3e}; "
          f"perm mean={perm_vals.mean():.3e} ± {perm_vals.std():.3e}; "
          f"one-sided p={p_perm:.4f} (null ≥ observed)")
    print(f"[PSD theta ratio]        real={ratio_real:.3f}, noise={ratio_noise:.3f}")

    # NEW prints requested
    print(f"[Peak width (50% prom) KS]       D={ks_w:.4f},  p={p_w:.3e}   (median w: {np.median(w_real_s)*1e3:.2f} vs {np.median(w_noise_s)*1e3:.2f} ms)")
    print(f"[Peak height KS]                  D={ks_h:.4f},  p={p_h:.3e}   (median h: {np.median(h_real):.3f} vs {np.median(h_noise):.3f})")
    print(f"[Fano factor, {int(bin_s_ff*1e3)} ms bins]  real={ff_real:.2f}, noise={ff_noise:.2f}, perm μ≈{ff_perm_mu:.2f}±{ff_perm_sd:.2f}, p={p_ff:.4f}")
    print(f"[ACG small-lag area 1–10 ms]     obs={stat_acg:.3e}, perm μ≈{acg_mu:.3e}±{acg_sd:.3e}, p={p_acg_small:.4f}")

    print(f"\nFigures saved to: {out_dir}\n")

    # Optional: average peak shapes (raw amplitude & peak-normalised), using the detection signal
    _ = plot_mean_peak_shapes_for_nohp(x_for_detect, idx_real, x_noise, idx_noise, Fs, out_dir)

# (optional) quick plotting of average peak shapes for the no-HP case
def plot_mean_peak_shapes_for_nohp(x_real_det, idx_real, x_noise, idx_noise, fs, out_dir, pre_ms=6.0, post_ms=6.0):
    def extract_peak_waveforms(x, peak_idx, fs, pre_ms=6.0, post_ms=6.0, normalise=None):
        x = np.asarray(x, float)
        n = x.size
        pre_n = int(round((pre_ms/1000.0)*fs))
        post_n= int(round((post_ms/1000.0)*fs))
        win   = pre_n + post_n + 1
        keep  = (peak_idx - pre_n >= 0) & (peak_idx + post_n < n)
        peaks = peak_idx[keep]
        if peaks.size == 0:
            return np.empty((0, win), float), np.linspace(-pre_ms, post_ms, win)
        snips = np.empty((peaks.size, win), float)
        for i, p in enumerate(peaks):
            seg = x[p-pre_n:p+post_n+1]
            if normalise == "peak":
                denom = np.max(np.abs(seg)) if np.max(np.abs(seg)) > 0 else 1.0
                seg = seg/denom
            snips[i] = seg
        t_ms = np.linspace(-pre_ms, post_ms, win)
        return snips, t_ms

    def _sem(a, axis=0):
        a = np.asarray(a, float)
        n = np.sum(~np.isnan(a), axis=axis)
        sd = np.nanstd(a, axis=axis, ddof=1)
        return sd / np.sqrt(np.maximum(n, 1))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # raw amplitude mean ± SEM
    sn_r_raw, t_ms = extract_peak_waveforms(x_real_det, idx_real, fs, pre_ms, post_ms, normalise=None)
    sn_n_raw, _    = extract_peak_waveforms(x_noise,    idx_noise, fs, pre_ms, post_ms, normalise=None)

    if sn_r_raw.size and sn_n_raw.size:
        mu_r, se_r = np.nanmean(sn_r_raw, 0), _sem(sn_r_raw, 0)
        mu_n, se_n = np.nanmean(sn_n_raw, 0), _sem(sn_n_raw, 0)
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=170)
        ax.plot(t_ms, mu_r, lw=2, color="C0", label=f"Real (n={sn_r_raw.shape[0]})")
        ax.fill_between(t_ms, mu_r-se_r, mu_r+se_r, color="C0", alpha=0.25)
        ax.plot(t_ms, mu_n, lw=2, color="0.4", label=f"Noise (n={sn_n_raw.shape[0]})")
        ax.fill_between(t_ms, mu_n-se_n, mu_n+se_n, color="0.5", alpha=0.25)
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_xlabel("Time from peak (ms)"); ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title("Average peak waveform (raw amplitude ± SEM)"); _clean_axes(ax)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_dir/"avg_peak_shape_raw_noHP.png", bbox_inches="tight"); plt.close(fig)

    # peak-normalised
    sn_r_nrm, t_ms = extract_peak_waveforms(x_real_det, idx_real, fs, pre_ms, post_ms, normalise="peak")
    sn_n_nrm, _    = extract_peak_waveforms(x_noise,    idx_noise, fs, pre_ms, post_ms, normalise="peak")

    if sn_r_nrm.size and sn_n_nrm.size:
        mu_r, se_r = np.nanmean(sn_r_nrm, 0), _sem(sn_r_nrm, 0)
        mu_n, se_n = np.nanmean(sn_n_nrm, 0), _sem(sn_n_nrm, 0)
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=170)
        ax.plot(t_ms, mu_r, lw=2, color="C0", label=f"Real (n={sn_r_nrm.shape[0]})")
        ax.fill_between(t_ms, mu_r-se_r, mu_r+se_r, color="C0", alpha=0.25)
        ax.plot(t_ms, mu_n, lw=2, color="0.4", label=f"Noise (n={sn_n_nrm.shape[0]})")
        ax.fill_between(t_ms, mu_n-se_n, mu_n+se_n, color="0.5", alpha=0.25)
        ax.axvline(0, color="k", lw=1, alpha=0.6)
        ax.set_xlabel("Time from peak (ms)"); ax.set_ylabel("Normalised amplitude (peak = 1)")
        ax.set_title("Average peak waveform (normalised ± SEM)"); _clean_axes(ax)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_dir/"avg_peak_shape_norm_noHP.png", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    main()
