# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:51:50 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 21:50:24 2025

@author: yifan
"""

import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import hilbert
from scipy.stats import sem

import re

def _safe_filename(s: str) -> str:
    """Turn any label into a filesystem-safe token."""
    s = s.replace("π", "pi").replace("−", "minus").replace("–", "minus").replace("—", "minus")
    # replace anything not [a-zA-Z0-9._-] with underscore
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
# ---------------------------
# Filters & basic utilities
# ---------------------------
def bandpass_filter(signal_data, fs, lowcut=5, highcut=11, order=4):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, signal_data)

def _phase(signal_band):
    return np.angle(hilbert(signal_band))

def _envelope(signal_band):
    return np.abs(hilbert(signal_band))

def _circmean(angles):
    # angles in radians
    return np.angle(np.mean(np.exp(1j * angles)))

def _wrap_pi(x):
    # wrap to (-pi, pi]
    return (x + np.pi) % (2*np.pi) - np.pi

def filter_close_peaks(peak_indices, min_distance_samples):
    if len(peak_indices) == 0:
        return np.array([], dtype=int)
    filtered = [int(peak_indices[0])]
    for idx in peak_indices[1:]:
        if int(idx) - filtered[-1] >= min_distance_samples:
            filtered.append(int(idx))
    return np.array(filtered, dtype=int)

# ---------------------------
# Cross-corr / PLV / envelope-r
# ---------------------------
def cross_correlation_theta(lfp_theta, opt_theta, fs, max_lag_s=0.5):
    corr = signal.correlate(lfp_theta, opt_theta, mode='full')
    lags = signal.correlation_lags(len(lfp_theta), len(opt_theta), mode='full')
    corr = corr / (np.std(lfp_theta) * np.std(opt_theta) * len(lfp_theta))
    lags_sec = lags / fs
    mask = (lags_sec >= -max_lag_s) & (lags_sec <= max_lag_s)
    return lags_sec[mask], corr[mask]

def phase_locking_value(lfp_theta, opt_theta):
    phi_l = _phase(lfp_theta)
    phi_o = _phase(opt_theta)
    dphi = _wrap_pi(phi_l - phi_o)
    return np.abs(np.mean(np.exp(1j * dphi)))

def amplitude_correlation(lfp_theta, opt_theta):
    r = np.corrcoef(_envelope(lfp_theta), _envelope(opt_theta))[0, 1]
    return float(r)

# ---------------------------
# Sliding-window coupling
# ---------------------------
def sliding_coupling(lfp_theta, opt_theta, fs, win_s=2.0, step_s=0.2):
    win = int(win_s * fs)
    step = int(step_s * fs)
    n = len(lfp_theta)
    t_centres, plv_list, r_list = [], [], []
    for start in range(0, n - win + 1, step):
        end = start + win
        plv_list.append(phase_locking_value(lfp_theta[start:end], opt_theta[start:end]))
        r_list.append(amplitude_correlation(lfp_theta[start:end], opt_theta[start:end]))
        t_centres.append((start + end) / 2 / fs)
    return np.array(t_centres), np.array(plv_list), np.array(r_list)

# ---------------------------
# Theta cycle segmentation (trough-to-trough on LFP)
# ---------------------------
def detect_lfp_troughs(lfp_theta, fs, min_period_s=0.08):
    # troughs ≈ peaks of -lfp_theta
    distance = int(min_period_s * fs)
    trough_idx, _ = signal.find_peaks(-lfp_theta, distance=distance)
    return trough_idx

def resample_cycle(x, n_samples=200):
    # resample a 1D array to fixed length
    old_idx = np.linspace(0, 1, num=len(x), endpoint=True)
    new_idx = np.linspace(0, 1, num=n_samples, endpoint=True)
    return np.interp(new_idx, old_idx, x)

def cycle_analysis(lfp_theta, opt_theta, fs, n_resample=200, cycle_normalize="zscore"):
    """
    Segment LFP theta into trough→trough cycles and resample each cycle.
    cycle_normalize: "zscore" (per-cycle z), "minmax", or None.
    """
    troughs = detect_lfp_troughs(lfp_theta, fs)
    troughs = troughs[(troughs > 0) & (troughs < len(lfp_theta)-1)]
    if len(troughs) < 3:
        return {
            "cycle_centres_t": np.array([]),
            "cycle_phase_shift": np.array([]),
            "lfp_cycles": None,
            "opt_cycles": None,
            "inst_freq_hz": np.array([]),
        }

    phi_l = _phase(lfp_theta)
    phi_o = _phase(opt_theta)

    cycle_centres_t, cycle_phase_shift, inst_freq_hz = [], [], []
    lfp_stack, opt_stack = [], []

    for i in range(len(troughs)-1):
        a, b = troughs[i], troughs[i+1]
        if b <= a + 2:
            continue

        # instantaneous frequency per cycle
        period_s = (b - a) / fs
        if period_s <= 0:
            continue
        inst_freq_hz.append(1.0 / period_s)
        cycle_centres_t.append((a + b) / 2 / fs)

        # per-cycle phase shift
        dphi = _wrap_pi(phi_l[a:b] - phi_o[a:b])
        cycle_phase_shift.append(_circmean(dphi))

        # resample waveforms to fixed length
        lfp_cycle = resample_cycle(lfp_theta[a:b], n_samples=n_resample)
        opt_cycle = resample_cycle(opt_theta[a:b], n_samples=n_resample)

        # --- normalise per cycle ---
        if cycle_normalize == "zscore":
            # avoid div-by-zero if a cycle is (nearly) flat
            eps = 1e-12
            lfp_std = np.std(lfp_cycle) + eps
            opt_std = np.std(opt_cycle) + eps
            lfp_cycle = (lfp_cycle - np.mean(lfp_cycle)) / lfp_std
            opt_cycle = (opt_cycle - np.mean(opt_cycle)) / opt_std
        elif cycle_normalize == "minmax":
            lfp_min, lfp_ptp = np.min(lfp_cycle), (np.max(lfp_cycle) - np.min(lfp_cycle) + 1e-12)
            opt_min, opt_ptp = np.min(opt_cycle), (np.max(opt_cycle) - np.min(opt_cycle) + 1e-12)
            lfp_cycle = (lfp_cycle - lfp_min) / lfp_ptp
            opt_cycle = (opt_cycle - opt_min) / opt_ptp
        else:
            # raw (no scaling)
            pass

        lfp_stack.append(lfp_cycle)
        opt_stack.append(opt_cycle)

    return {
        "cycle_centres_t": np.array(cycle_centres_t),
        "cycle_phase_shift": np.array(cycle_phase_shift),
        "lfp_cycles": np.vstack(lfp_stack) if lfp_stack else None,
        "opt_cycles": np.vstack(opt_stack) if opt_stack else None,
        "inst_freq_hz": np.array(inst_freq_hz),
    }
def _quadratic_vertex(xm1, x0, xp1, ym1, y0, yp1):
    """
    Three-point quadratic interpolation. Returns (xv, yv) of the parabola's vertex.
    Assumes x are equally spaced: xp1 - x0 == x0 - xm1.
    """
    denom = (ym1 - 2*y0 + yp1)
    if denom == 0:
        return x0, y0
    h = x0 - xm1
    delta = 0.5 * (ym1 - yp1) / denom  # in units of h relative to x0
    xv = x0 + delta * h
    yv = y0 - 0.25 * (ym1 - yp1) * delta
    return float(xv), float(yv)
def estimate_trough_x(x, y):
    """
    Estimate trough (minimum) location on y(x) with quadratic interpolation
    around the discrete minimum. Returns x_trough (float).
    """
    y = np.asarray(y)
    x = np.asarray(x)
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])  # edge case: no neighbors
    xv, _ = _quadratic_vertex(x[i-1], x[i], x[i+1],
                              y[i-1], y[i], y[i+1])
    return xv  # just the x position


def estimate_peak_lag_from_xcorr(lags_s, corr, search_window_s=None):
    """
    Return (lag_at_peak_s, peak_value) using quadratic (parabolic) interpolation
    around the discrete maximum. If search_window_s=(a,b) is given, restrict
    search to a<=lag<=b.
    """
    lags_s = np.asarray(lags_s)
    corr = np.asarray(corr)

    if search_window_s is not None:
        a, b = search_window_s
        mask = (lags_s >= a) & (lags_s <= b)
        lags_s, corr = lags_s[mask], corr[mask]

    # index of discrete max
    i = int(np.argmax(corr))

    # if peak is at an edge, skip interpolation
    if i == 0 or i == len(corr) - 1:
        return float(lags_s[i]), float(corr[i])

    # quadratic interpolation using three points (i-1, i, i+1)
    y1, y2, y3 = corr[i-1], corr[i], corr[i+1]
    x1, x2, x3 = lags_s[i-1], lags_s[i], lags_s[i+1]

    # assume uniform lag step h (it is, from correlate); use vertex formula
    h = x2 - x1
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        x_peak = x2
        y_peak = y2
    else:
        delta = 0.5 * (y1 - y3) / denom  # location in units of h relative to x2
        x_peak = x2 + delta * h
        # evaluate parabola at x_peak
        y_peak = y2 - 0.25 * (y1 - y3) * delta

    return float(x_peak), float(y_peak)

# ---------------------------
# PLV permutation (circular-shift surrogate)
# ---------------------------
def plv_permutation_test(lfp_theta, opt_theta, fs, n_perm=1000, min_shift_s=1.0):
    n = len(lfp_theta)
    min_shift = int(min_shift_s * fs)
    # observed
    obs = phase_locking_value(lfp_theta, opt_theta)

    null = np.zeros(n_perm, dtype=float)
    for k in range(n_perm):
        # circular shift the optical signal by a random offset (>= min_shift)
        shift = np.random.randint(min_shift, n - min_shift)
        opt_shifted = np.roll(opt_theta, shift)
        null[k] = phase_locking_value(lfp_theta, opt_shifted)

    # p-value (right-tailed)
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return float(obs), null, float(p)

# ---------------------------
# Master routine (figures + dict)
# ---------------------------
def run_theta_cycle_analysis(theta_part, LFP_channel, fs=10000,
                             bp_low=5, bp_high=12,
                             win_s=2.0, step_s=0.2,
                             n_perm=1000, min_shift_s=1.0,
                             n_resample=200, cycle_normalize="zscore",
                             make_plots=True):
    # bandpass both to theta
    lfp_theta = bandpass_filter(theta_part[LFP_channel].values, fs, bp_low, bp_high)
    opt_theta = bandpass_filter(theta_part['zscore_raw'].values, fs, bp_low, bp_high)

    # ---------- cross-correlation figure ----------
    lags, corr = cross_correlation_theta(lfp_theta, opt_theta, fs)
    # Estimate peak lag within a sensible window (e.g., ±0.25 s)
    lag_peak_s, corr_peak = estimate_peak_lag_from_xcorr(lags, corr, search_window_s=(-0.25, 0.25))
    lag_peak_ms = 1000.0 * lag_peak_s
    print(f"XCorr peak lag: {lag_peak_ms:+.1f} ms (corr={corr_peak:.3f})")

    if make_plots:
        plt.figure(figsize=(7,4))
        ax = plt.gca()
        ax.plot(lags, corr, lw=2)
        ax.axvline(0, color='k', ls='--')
    
        # labels and title
        ax.set_xlabel('Lag (s)', fontsize=16)
        ax.set_ylabel('Correlation', fontsize=16)
        ax.set_title('Cross-correlation (LFP vs GEVI theta)', fontsize=16)
    
        # enlarge tick labels
        ax.tick_params(axis='both', labelsize=14)
    
        # remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        plt.tight_layout()
        plt.show()

    # ---------- global PLV / envelope-r ----------
    plv_global = phase_locking_value(lfp_theta, opt_theta)
    r_env_global = amplitude_correlation(lfp_theta, opt_theta)
    print(f"Global PLV: {plv_global:.3f}")
    print(f"Global envelope r: {r_env_global:.3f}")

    # ---------- sliding window coupling ----------
    t_c, plv_t, r_t = sliding_coupling(lfp_theta, opt_theta, fs, win_s, step_s)
    if make_plots and len(t_c):
        fig, ax = plt.subplots(2, 1, figsize=(8,5), sharex=True)
        ax[0].plot(t_c, plv_t, lw=1.5)
        ax[0].set_ylabel('PLV (win)')
        ax[0].set_title('Sliding-window coupling')
        ax[1].plot(t_c, r_t, lw=1.5)
        ax[1].set_ylabel('Envelope r (win)')
        ax[1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

     # ---------- cycle-wise analysis ----------
    cyc = cycle_analysis(lfp_theta, opt_theta, fs, n_resample=n_resample,
                         cycle_normalize=cycle_normalize)

    if make_plots and cyc["lfp_cycles"] is not None:
        # ... phase shift plots unchanged ...

        # cycle-averaged waveforms ± SEM (now normalised per cycle)
        from scipy.stats import sem
        lfp_mean = np.nanmean(cyc["lfp_cycles"], axis=0)
        lfp_sem  = sem(cyc["lfp_cycles"], axis=0, nan_policy='omit')
        opt_mean = np.nanmean(cyc["opt_cycles"], axis=0)
        opt_sem  = sem(cyc["opt_cycles"], axis=0, nan_policy='omit')
        # ---- cycle-averaged waveforms with troughs & peaks marked ----
        plt.figure(figsize=(7,4))
        ax = plt.gca()
        
        n = len(lfp_mean)
        x = np.linspace(0, 2*np.pi, n)  # cycle in radians (0 → 2π)
        
        # Plot mean ± SEM
        ax.plot(x, lfp_mean, label='LFP θ (mean)', color='C0')
        ax.fill_between(x, lfp_mean-lfp_sem, lfp_mean+lfp_sem, alpha=0.25, color='C0')
        ax.plot(x, opt_mean, label='GEVI θ (mean)', color='C1')
        ax.fill_between(x, opt_mean-opt_sem, opt_mean+opt_sem, alpha=0.25, color='C1')
        
        # ---- mark troughs ----
        lfp_trough_idx = np.argmin(lfp_mean)
        gevi_trough_idx = np.argmin(opt_mean)
        ax.axvline(x[lfp_trough_idx], color='C0', ls='--', lw=1.5, label='LFP trough')
        ax.axvline(x[gevi_trough_idx], color='C1', ls='--', lw=1.5, label='GEVI trough')
        
        # ---- mark peaks ----
        lfp_peak_idx = np.argmax(lfp_mean)
        gevi_peak_idx = np.argmax(opt_mean)
        # ax.axvline(x[lfp_peak_idx], color='C0', ls=':', lw=1.5, label='LFP peak')
        # ax.axvline(x[gevi_peak_idx], color='C1', ls=':', lw=1.5, label='GEVI peak')
        
        # Labels and style
        ax.set_xlabel('Theta phase (rad)', fontsize=16)
        ax.set_ylabel('Amplitude\n(normalised per cycle)', fontsize=16)
        ax.set_title('Cycle-averaged waveforms with peaks and troughs', fontsize=16)
        ax.legend(fontsize=12, ncol=2)
        
        # Set xticks at canonical radian positions
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=14)
        
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        # ---- numeric offsets ----
        lfp_trough_x = x[lfp_trough_idx]; gevi_trough_x = x[gevi_trough_idx]
        lfp_peak_x   = x[lfp_peak_idx];   gevi_peak_x   = x[gevi_peak_idx]
        
        offset_trough_cycle = gevi_trough_x - lfp_trough_x
        offset_peak_cycle   = gevi_peak_x - lfp_peak_x
        
        offset_trough_deg = 360 * offset_trough_cycle
        offset_peak_deg   = 360 * offset_peak_cycle
        
        f_med = np.median(cyc["inst_freq_hz"]) if len(cyc["inst_freq_hz"]) else np.nan
        T_ms = 1000.0 / f_med if np.isfinite(f_med) else np.nan
        
        print(f"Trough offset (GEVI vs LFP): {offset_trough_cycle:+.3f} cycles "
              f"({offset_trough_deg:+.1f}°), ≈{offset_trough_cycle*T_ms:+.1f} ms at {f_med:.2f} Hz")
        
        print(f"Peak offset   (GEVI vs LFP): {offset_peak_cycle:+.3f} cycles "
              f"({offset_peak_deg:+.1f}°), ≈{offset_peak_cycle*T_ms:+.1f} ms at {f_med:.2f} Hz")

    # ---- histogram of (wrapped) phase shift ----
    plt.figure(figsize=(4.5,3.5))
    ax = plt.gca()
    bins = np.linspace(-np.pi, np.pi, 37)
    ax.hist(_wrap_pi(cyc["cycle_phase_shift"]), bins=bins, density=True, alpha=0.8)
    
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r"-π", r"-π/2", "0", r"π/2", r"π"], fontsize=14)
    
    ax.set_xlabel('Phase shift (rad)', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('Cycle-wise(φ_LFP − φ_GEVI)', fontsize=16)
    
    # bigger ticks and remove top/right spines
    ax.tick_params(axis='both', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # ---------- PLV permutation test ----------
    # obs_plv, null_plv, p_plv = plv_permutation_test(
    #     lfp_theta, opt_theta, fs, n_perm=n_perm, min_shift_s=min_shift_s
    # )
    # print(f"PLV permutation test: observed={obs_plv:.3f}, p={p_plv:.4f}")

    # if make_plots:
    #     plt.figure(figsize=(6,4))
    #     plt.hist(null_plv, bins=40, alpha=0.8, density=True)
    #     plt.axvline(obs_plv, color='r', lw=2, label=f'Observed PLV = {obs_plv:.3f}')
    #     plt.xlabel('PLV (null, circular-shift)')
    #     plt.ylabel('Density')
    #     plt.title(f'PLV significance (n={len(null_plv)}, p={p_plv:.4f})')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # ---------- pack results ----------
    results = {
        "PLV_global": float(plv_global),
        "Envelope_r_global": float(r_env_global),
        "t_sliding": t_c,
        "PLV_sliding": plv_t,
        "Envelope_r_sliding": r_t,
        "lags_s": lags,
        "xcorr": corr,
        "cycle_centres_t": cyc["cycle_centres_t"],
        "cycle_phase_shift_rad": cyc["cycle_phase_shift"],
        "inst_freq_hz": cyc["inst_freq_hz"],
        "n_cycles": 0 if cyc["lfp_cycles"] is None else cyc["lfp_cycles"].shape[0],
        # "plv_perm_p": p_plv,
        # "plv_perm_null": null_plv,
        "xcorr_peak_lag_s": lag_peak_s,
        "xcorr_peak_corr": corr_peak,
    }
    print(f"n_cycles = {results['n_cycles']}")
    
    return results


def calculate_theta_trough_index(df,angle_source, Fs=10000):
    # Detect local minima in theta phase: trough = 0 rad
    troughs = (
        (df[angle_source] < df[angle_source].shift(-1)) &
        (df[angle_source] < df[angle_source].shift(1)) &
        ((df[angle_source] < 0.2) | (df[angle_source] > (2 * np.pi - 0.2)))  # Around 0
    )
    trough_index = df.index[troughs]

    # Detect peaks: now at π radians (≈3.14)
    peaks = (
        (df[angle_source] > (np.pi - 0.1)) &
        (df[angle_source] < (np.pi + 0.1))
    )
    peak_index = df.index[peaks]

    return trough_index, peak_index

def plot_theta_traces(theta_part, LFP_channel, start_time, end_time, fs=10000):
    # Convert time to index
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    time_vector = np.arange(len(theta_part)) / fs

    # Subset and time axis
    segment = theta_part.iloc[start_idx:end_idx].copy()
    t = time_vector[start_idx:end_idx]

    # Apply theta bandpass filter
    filtered_LFP = bandpass_filter(segment[LFP_channel].values, fs)
    filtered_zscore = bandpass_filter(segment['zscore_raw'].values, fs)

    # Detect LFP theta peaks (phase near ±π)
    segment_peak_LFP = segment[
        (segment['LFP_theta_angle'] > 2.9) &
        (segment['LFP_theta_angle'] < 3.2)
    ].index

    # Identify optical theta peaks: same idea
    segment_peak_optical = segment[
        (segment['optical_theta_angle'] > 2.9) &
        (segment['optical_theta_angle'] < 3.2)
    ].index

    # Enforce 80 ms minimum distance between peaks
    min_dist_samples = int(0.08 * fs)  # 80 ms
    peak_idx_LFP = filter_close_peaks(segment_peak_LFP.to_numpy(), min_dist_samples)
    peak_idx_optical = filter_close_peaks(segment_peak_optical.to_numpy(), min_dist_samples)

    # Convert to time
    peak_t_LFP = peak_idx_LFP / fs
    peak_t_optical = peak_idx_optical / fs

    # Start plotting
    fig, axes = plt.subplots(4, 1, figsize=(16, 6), sharex=True)

    # Remove all subplot frames & set tick label size
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.margins(x=0)
        ax.margins(y=0)
        ax.tick_params(axis='both', which='both', labelsize=20)  # enlarge tick labels

    # 1. Raw LFP trace
    axes[0].plot(t, segment[LFP_channel], color='black')
    axes[0].set_ylabel(LFP_channel, fontsize=20)
    #axes[0].set_title('LFP trace', fontsize=20)

    # 2. zscore_raw
    zscore_lowpass = OE.smooth_signal(segment['zscore_raw'], fs, 100, window='flat')
    axes[1].plot(t, zscore_lowpass, color='green')
    axes[1].set_ylabel('zscore', fontsize=20)
    #axes[1].set_title('GEVI Signal', fontsize=20)

    # 3. Filtered LFP + LFP peaks + zscore peaks as dots
    axes[2].plot(t, filtered_LFP, color='black', label='Filtered LFP')
    for pt in peak_t_LFP:
        if start_time <= pt <= end_time:
            axes[2].axvline(x=pt, color='red', linestyle='--', alpha=0.6)
    # Overlay optical theta peaks as dots
    dot_y = np.interp(peak_t_optical, t, filtered_LFP)
    axes[2].scatter(peak_t_optical, dot_y, color='green', marker='o', s=40, label='zscore peaks')
    axes[2].set_ylabel('LFP theta', fontsize=20)
    #axes[2].set_title('LFP Theta band + Optical Peaks', fontsize=20)

    # 4. Filtered zscore + optical peaks
    axes[3].plot(t, filtered_zscore, color='green')
    for pt in peak_t_optical:
        if start_time <= pt <= end_time:
            axes[3].axvline(x=pt, color='red', linestyle='--', alpha=0.6)
    axes[3].set_ylabel('zscore theta', fontsize=20)
    #axes[3].set_title('GEVI theta band', fontsize=20)
    axes[3].set_xlabel('Time (s)', fontsize=20)
    

    plt.tight_layout()
    plt.show()
    
def plot_zscore_peaks_on_LFP_phase(theta_part, fs=10000, wrap_cycles=2):
    """
    Plot a scatter plot of optical theta peaks (zscore) on the corresponding LFP theta phase.
    
    Args:
        theta_part (pd.DataFrame): DataFrame containing 'optical_theta_angle' and 'LFP_theta_angle' columns.
        fs (int): Sampling rate in Hz.
        wrap_cycles (int): How many theta cycles to wrap the LFP phase (1 for 0–360, 2 for 0–720).
    """
    # Detect optical theta peaks (where optical phase ~ π)
    peak_idx_optical = theta_part[
        (theta_part['optical_theta_angle'] > 2.9) & (theta_part['optical_theta_angle'] < 3.2)
    ].index.to_numpy()

    # Enforce 80 ms minimum distance
    min_dist_samples = int(0.08 * fs)
    peak_idx_optical = filter_close_peaks(peak_idx_optical, min_dist_samples)

    # Get LFP theta phase at each optical peak
    lfp_phases = theta_part.loc[peak_idx_optical, 'LFP_theta_angle'].values

    # Convert radians to degrees and wrap across cycles if needed
    lfp_degrees = np.rad2deg(lfp_phases) % 360
    if wrap_cycles == 2:
        # Track which cycle (0 or 1) each peak comes from, and offset accordingly
        cycle_index = np.arange(len(lfp_degrees)) % 2
        lfp_degrees += 360 * cycle_index

    # Plot
    plt.figure(figsize=(4, 6))
    plt.scatter(np.arange(len(lfp_degrees)), lfp_degrees, color='purple', alpha=0.7)
    plt.ylim(0, 360 * wrap_cycles)
    plt.xlabel('Optical Theta Peak Index', fontsize=18)
    plt.ylabel('LFP Theta Phase (degrees)', fontsize=18)
    plt.title('LFP Theta Phase at GEVI Peaks', fontsize=18)
    plt.yticks(np.arange(0, 360 * wrap_cycles + 1, 90))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()    

# ---------- helpers for phase splitting ----------
def _wrap_2pi(a):
    """Wrap angle to [0, 2π)."""
    return np.mod(a, 2*np.pi)

def classify_half_cycles(phases_rad, cut_rad):
    """
    Split angles into two halves relative to cut_rad.
    Returns boolean mask 'is_first_half' (True=first half).
    First half := (cut, cut+π]; second half := (cut+π, cut+2π].
    """
    ang = _wrap_2pi(phases_rad - cut_rad)  # shift so cut is 0
    return (ang > 0) & (ang <= np.pi)

def detect_optical_theta_peaks_idx(theta_phase_opt, fs, centre=np.pi, halfwidth=0.3, refractory_s=0.08):
    """Indices where optical theta phase is near π, with refractory."""
    idx = np.where((theta_phase_opt > (centre - halfwidth)) & (theta_phase_opt < (centre + halfwidth)))[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    min_dist = int(refractory_s * fs)
    kept = [int(idx[0])]
    for k in idx[1:]:
        if int(k) - kept[-1] >= min_dist:
            kept.append(int(k))
    return np.asarray(kept, dtype=int)

# ---------- 2D energy test with permutation ----------
def energy_statistic_2d(X, Y):
    """
    Energy distance (Székely–Rizzo). X, Y are (n,2) arrays.
    Returns E (float).
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    m, n = len(X), len(Y)
    if m < 2 or n < 2:
        return np.nan
    # pairwise Euclidean distances
    d_xy = np.linalg.norm(X[:,None,:] - Y[None,:,:], axis=2)
    d_xx = np.linalg.norm(X[:,None,:] - X[None,:,:], axis=2)
    d_yy = np.linalg.norm(Y[:,None,:] - Y[None,:,:], axis=2)
    term1 = 2.0/(m*n) * d_xy.sum()
    term2 = 1.0/(m*m) * d_xx.sum()
    term3 = 1.0/(n*n) * d_yy.sum()
    return term1 - term2 - term3

def energy_permutation_test_2d(X, Y, n_perm=2000, seed=0):
    """
    Permutation test for difference in 2D distributions using energy distance.
    Right-tailed: larger E = more different. Returns (E_obs, p_perm).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    E_obs = energy_statistic_2d(X, Y)
    if not np.isfinite(E_obs):
        return np.nan, np.nan
    Z = np.vstack([X, Y])
    m = len(X)
    cnt = 1  # add-one smoothing
    for _ in range(n_perm):
        perm = rng.permutation(len(Z))
        Xp, Yp = Z[perm[:m]], Z[perm[m:]]
        if energy_statistic_2d(Xp, Yp) >= E_obs:
            cnt += 1
    p = cnt / (n_perm + 1)
    return float(E_obs), float(p)

# ---------- plotting ----------
def plot_phase_split_arena(xy_blue, xy_red, cut_label, save_path=None, s=10):
    """
    xy_blue, xy_red: (n,2) arrays of positions. cut_label: string for title.
    """
    plt.figure(figsize=(5.2, 5.2))
    if len(xy_blue):
        plt.scatter(xy_blue[:,0], xy_blue[:,1], s=s, alpha=0.7, label="First half", color='C0')
    if len(xy_red):
        plt.scatter(xy_red[:,0], xy_red[:,1], s=s, alpha=0.7, label="Second half", color='C3')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X (pixel)", fontsize=14)
    plt.ylabel("Y (pixel)", fontsize=14)
    plt.title(f"Optical events by theta phase (cut at {cut_label})", fontsize=16)
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.grid(False)
    #plt.legend(frameon=False, fontsize=12)
    plt.legend('',frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
def analyse_phase_halves_in_arena(theta_part, fs=10000, use_lfp_phase=True,
                                  phase_cuts=(-np.pi/2, 0.0, np.pi/2),
                                  out_dir=None, base_name="phase_halves"):
    """
    Splits optical peaks into first/second half of theta cycle at multiple cuts and
    (1) plots 2D arena scatter (blue vs red) and
    (2) runs a 2D energy permutation test on positions.

    Args:
        theta_part: DataFrame with columns:
            'shoulder_x','shoulder_y','LFP_theta_angle','optical_theta_angle'
        fs: sampling rate
        use_lfp_phase: if True, classify by LFP phase at the optical-peak times
                       (this matches the paper’s population phase usage).
                       If False, use optical phase itself.
        phase_cuts: iterable of cut angles (rad)
        out_dir: if provided, saves PNGs there
        base_name: prefix for saved files

    Returns:
        results: list of dicts with summary per cut.
    """
    # 1) find optical peaks (times)
    peak_idx = detect_optical_theta_peaks_idx(theta_part['optical_theta_angle'].values,
                                              fs, centre=np.pi, halfwidth=0.30, refractory_s=0.08)
    if peak_idx.size == 0:
        print("No optical theta peaks detected.")
        return []

    # positions at peak times
    xy = theta_part.loc[peak_idx, ["shoulder_x","shoulder_y"]].to_numpy(float)

    # phase at those times
    if use_lfp_phase:
        phases = theta_part.loc[peak_idx, "LFP_theta_angle"].to_numpy(float)
    else:
        phases = theta_part.loc[peak_idx, "optical_theta_angle"].to_numpy(float)

    summaries = []
    for cut in phase_cuts:
        is_first = classify_half_cycles(phases, cut)          # bool mask
        blue = xy[is_first]
        red  = xy[~is_first]

        # stats
        E, p = energy_permutation_test_2d(blue, red, n_perm=2000, seed=1)

        # plot
        # cut_label = { -np.pi/2: "−π/2", 0.0: "0", np.pi/2: "π/2" }.get(cut, f"{cut:.2f} rad")
        # save_path = None
        # if out_dir:
        #     os.makedirs(out_dir, exist_ok=True)
        #     save_path = os.path.join(out_dir, f"{base_name}_cut_{cut_label.replace('π','pi')}.png")
        # plot_phase_split_arena(blue, red, cut_label=cut_label, save_path=save_path, s=12)
        
        cut_label = { -np.pi/2: "−π/2", 0.0: "0", np.pi/2: "π/2" }.get(cut, f"{cut:.2f} rad")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{base_name}_cut_{_safe_filename(cut_label)}.png"
            save_path = os.path.join(out_dir, fname)
        else:
            save_path = None
        
        plot_phase_split_arena(blue, red, cut_label=cut_label, save_path=save_path, s=12)


        # print summary
        print(f"[cut {cut_label}] n_blue={len(blue)}, n_red={len(red)}, "
              f"energy E={E:.4f}, p_perm={p:.4g}")

        summaries.append({
            "cut_rad": float(cut),
            "cut_label": cut_label,
            "n_blue": int(len(blue)),
            "n_red": int(len(red)),
            "energy_E": float(E) if np.isfinite(E) else np.nan,
            "p_perm": float(p) if np.isfinite(p) else np.nan,
        })

    return summaries

def collect_phase_split_positions(theta_part, fs=10000, use_lfp_phase=True, cut_rad=0.0):
    """Return (xy_blue, xy_red) arrays for one cut without plotting."""
    peak_idx = detect_optical_theta_peaks_idx(
        theta_part['optical_theta_angle'].to_numpy(float),
        fs, centre=np.pi, halfwidth=0.30, refractory_s=0.08
    )
    if peak_idx.size == 0:
        return np.empty((0,2)), np.empty((0,2))

    xy = theta_part.loc[peak_idx, ["shoulder_x","shoulder_y"]].to_numpy(float)
    phases = theta_part.loc[peak_idx, "LFP_theta_angle" if use_lfp_phase else "optical_theta_angle"].to_numpy(float)
    is_first = classify_half_cycles(phases, cut_rad)
    return xy[is_first], xy[~is_first]

import glob
import os
import pandas as pd
from math import pi

def run_phase_halves_all(
    parent_dir,
    LFP_channel="LFP_1",
    cuts=(0.0, pi/4),
    fs=10000,
    output_subdir="ThetaSave_Move_phase_halves_ALL",
    use_lfp_phase=True
):
    """
    For every SyncRecording* in parent_dir:
      - compute LFP/optical theta phase
      - plot blue/red arena scatter per cut
      - save per-recording summary CSV
    Then:
      - pool positions across all recordings per cut
      - plot pooled arena scatter
      - run 2D energy-distance permutation test on pooled blue vs red
      - save a master CSV with per-recording and pooled stats
    """
    rec_dirs = sorted(glob.glob(os.path.join(parent_dir, "SyncRecording*")))
    if not rec_dirs:
        print("No SyncRecording* folders found.")
        return

    master_out = os.path.join(parent_dir, output_subdir)
    os.makedirs(master_out, exist_ok=True)

    all_rows = []
    pooled_positions = {cut: {"blue": [], "red": []} for cut in cuts}

    for rec in rec_dirs:
        rec_name = os.path.basename(rec)
        print(f"\n=== {rec_name} ===")

        # Load aligned dataframe and compute phases
        sess = SyncOEpyPhotometrySession(
            parent_dir, rec_name, IsTracking=False,
            read_aligned_data_from_file=True,
            recordingMode='Atlas', indicator='GEVI'
        )
        df = sess.Ephys_tracking_spad_aligned.reset_index(drop=True)
        df['LFP_theta_angle']     = OE.calculate_theta_phase_angle(df[LFP_channel], theta_low=5, theta_high=12)
        df['optical_theta_angle'] = OE.calculate_theta_phase_angle(df['zscore_raw'],  theta_low=5, theta_high=12)

        # Per-recording output folder
        rec_out = os.path.join(master_out, rec_name)
        os.makedirs(rec_out, exist_ok=True)

        # Do plots + stats for each cut
        rec_summaries = []
        for cut in cuts:
            # Make per-recording plot + stats (re-using earlier function)
            cut_label = {0.0: "0", pi/4: "π/4"}.get(cut, f"{cut:.2f}rad")
            fig_path = os.path.join(rec_out, f"{rec_name}_cut_{_safe_filename(cut_label)}.png")
            summaries = analyse_phase_halves_in_arena(
                df, fs=fs, use_lfp_phase=use_lfp_phase,
                phase_cuts=(cut,), out_dir=rec_out, base_name=rec_name
            )
            if summaries:
                s = summaries[0]
                s["recording"] = rec_name
                rec_summaries.append(s)
                all_rows.append(s)

            # Also collect raw positions for pooled stats
            blue, red = collect_phase_split_positions(df, fs=fs, use_lfp_phase=use_lfp_phase, cut_rad=cut)
            if len(blue): pooled_positions[cut]["blue"].append(blue)
            if len(red):  pooled_positions[cut]["red"].append(red)

        # Save per-recording summary CSV
        if rec_summaries:
            pd.DataFrame(rec_summaries).to_csv(os.path.join(rec_out, f"{rec_name}_phase_halves_summary.csv"), index=False)

    # ---------- POOLED plots + stats ----------
    pooled_rows = []
    for cut in cuts:
        cut_label = {0.0: "0", pi/4: "π/4"}.get(cut, f"{cut:.2f}rad")
        blue = np.vstack(pooled_positions[cut]["blue"]) if pooled_positions[cut]["blue"] else np.empty((0,2))
        red  = np.vstack(pooled_positions[cut]["red"])  if pooled_positions[cut]["red"]  else np.empty((0,2))

        pooled_png = os.path.join(master_out, f"POOLED_cut_{_safe_filename(cut_label)}.png")
        plot_phase_split_arena(blue, red, cut_label=f"{cut_label} (pooled)", save_path=pooled_png, s=10)

        E, p = energy_permutation_test_2d(blue, red, n_perm=5000, seed=7)
        print(f"\n[POOLED cut {cut_label}] n_blue={len(blue)}, n_red={len(red)}, energy E={E:.4f}, p_perm={p:.4g}")
        pooled_rows.append({
            "recording": "POOLED",
            "cut_rad": float(cut),
            "cut_label": cut_label,
            "n_blue": int(len(blue)),
            "n_red": int(len(red)),
            "energy_E": float(E) if np.isfinite(E) else np.nan,
            "p_perm": float(p) if np.isfinite(p) else np.nan,
        })

    # ---------- MASTER CSV ----------
    master_csv = os.path.join(master_out, "MASTER_phase_halves_summary.csv")
    out_df = pd.DataFrame(all_rows + pooled_rows)
    out_df.to_csv(master_csv, index=False)
    print(f"\nSaved master summary → {master_csv}")

def run_theta_cycle_plot(dpath, LFP_channel, recordingName, savename, theta_low_thres=0.5, fs=10000):
    save_path = os.path.join(dpath, savename)
    os.makedirs(save_path, exist_ok=True)

    Recording1 = SyncOEpyPhotometrySession(
        dpath, recordingName, IsTracking=False,
        read_aligned_data_from_file=True,
        recordingMode='Atlas', indicator='GEVI'
    )

    theta_part = Recording1.Ephys_tracking_spad_aligned.reset_index(drop=True)
    theta_part['LFP_theta_angle'] = OE.calculate_theta_phase_angle(theta_part[LFP_channel], theta_low=5, theta_high=12)
    theta_part['optical_theta_angle'] = OE.calculate_theta_phase_angle(theta_part['zscore_raw'], theta_low=5, theta_high=12)
    
    out_dir = os.path.join(dpath, savename, "phase_halves_arena")
    summaries = analyse_phase_halves_in_arena(
        theta_part,
        fs=fs,
        use_lfp_phase=True,                 # match Skaggs et al.: population/LFP phase
        phase_cuts=(0.0, np.pi/4),
        out_dir=out_dir,
        base_name=f"{recordingName}"
    )
    
    # Optional: make a small CSV of the summary
    if summaries:
        pd.DataFrame(summaries).to_csv(os.path.join(out_dir, "phase_halves_summary.csv"), index=False)

    # results = run_theta_cycle_analysis(theta_part, LFP_channel=LFP_channel, fs=fs,
    #                                    bp_low=5, bp_high=12,
    #                                    win_s=2.0, step_s=0.2,
    #                                    n_perm=1000, min_shift_s=1.0,
    #                                    n_resample=200, make_plots=True)

    # # Example: save results dictionary
    # with open(os.path.join(save_path, "theta_cycle_results.pkl"), "wb") as f:
    #     pickle.dump(results, f)

    return theta_part
#%%
'This is to process a single or concatenated trial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
'Run single'
dpath=r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion'
recordingName='SyncRecording3'
savename='ThetaSave_Move'
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_1'
theta_part=run_theta_cycle_plot (dpath,LFP_channel,recordingName,savename,theta_low_thres=-0.7) #-0.3

#%%
'Run batch'
dpath = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion'
LFP_channel = 'LFP_1'

run_phase_halves_all(
    parent_dir=dpath,
    LFP_channel=LFP_channel,
    cuts=(0.0, np.pi/4),             # cut at 0 and π/4
    fs=10000,
    output_subdir="ThetaSave_Move_phase_halves_ALL",
    use_lfp_phase=True               # match Skaggs-style (population/LFP phase)
)
