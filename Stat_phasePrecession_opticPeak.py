# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 21:57:48 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Phase precession statistics & visualisation over saved bouts.

Reads every:
  ...\same_direction_outputs\SyncRecording*\same_direction_segments.pkl

Outputs to:
  ...\same_direction_outputs\stats_phase_precession\
    - MASTER_phase_precession_stats.csv
    - per-recording stats CSVs
    - example_bout_scatter_*.png
    - pooled_scatter.png
    - rho_histogram.png
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =========================
# CONFIG
# =========================
SAVE_ROOT = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\OpenField_DLCtracking\same_direction_outputs"
LFP_CHANNEL = "LFP_1"
THETA_LOW, THETA_HIGH = 4, 9
FS = 10_000

# Minimum number of optical peaks in a bout to analyse
MIN_PEAKS = 5

# How many example bouts to plot individually
N_EXAMPLES = 8

# =========================
# FILTERS / PHASE UTILS
# =========================
def butter_bandpass(x, fs, low, high, order=4):
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)

def detect_optical_peaks(opt_theta_phase, fs, phase_center=np.pi, phase_halfwidth=0.3, refractory_s=0.08):
    """
    Keep optical-theta 'peaks' where optical phase ~= pi, with a min refractory.
    Returns sample indices (int).
    """
    idx = np.where((opt_theta_phase > (phase_center - phase_halfwidth)) &
                   (opt_theta_phase < (phase_center + phase_halfwidth)))[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    # enforce refractory
    min_dist = int(refractory_s * fs)
    kept = [int(idx[0])]
    for k in idx[1:]:
        if int(k) - kept[-1] >= min_dist:
            kept.append(int(k))
    return np.array(kept, dtype=int)

def lfp_phase_and_opt_peaks(seg_df, fs=FS, lfp_channel=LFP_CHANNEL, theta_low=THETA_LOW, theta_high=THETA_HIGH):
    """Return (peak_idx, lfp_phase_at_peaks [rad], time_at_peaks [s], full lfp_phase, full time)"""
    lfp_f = butter_bandpass(seg_df[lfp_channel].to_numpy(), fs, theta_low, theta_high)
    opt_f = butter_bandpass(seg_df["zscore_raw"].to_numpy(), fs, theta_low, theta_high)
    lfp_phase = np.angle(signal.hilbert(lfp_f))
    opt_phase = np.angle(signal.hilbert(opt_f))
    peak_idx = detect_optical_peaks(opt_phase, fs)
    t = np.arange(len(seg_df)) / fs
    return peak_idx, lfp_phase[peak_idx], t[peak_idx], lfp_phase, t

# =========================
# STATS
# =========================
def circ_corrcl(alpha, x):
    """
    Circular-linear correlation (Berens, 2009).
    alpha: angles in radians, shape (n,)
    x: linear variable, shape (n,)
    Returns rho in [0,1], two-sided p-value.
    """
    alpha = np.asarray(alpha); x = np.asarray(x)
    C, S = np.cos(alpha), np.sin(alpha)
    rxc = np.corrcoef(x, C)[0,1]
    rxs = np.corrcoef(x, S)[0,1]
    rcs = np.corrcoef(C, S)[0,1]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs) / (1 - rcs**2))
    # t-stat as in Berens (uses effective sample size)
    n = len(alpha)
    l20 = np.mean(S**2)
    l02 = np.mean(C**2)
    l11 = np.mean(S*C)
    tval = rho * np.sqrt((n * (l20*l02 - l11**2)) / max(1e-12, (1 - rho**2)))
    from scipy.stats import t
    pval = 2 * (1 - t.cdf(np.abs(tval), n - 2))
    return float(rho), float(pval)

def unwrap_slope(alpha, x):
    """
    Simple linear slope using unwrapped phase (rad) vs x (time s).
    Returns slope_deg_per_s, r, p.
    """
    alpha_unwrapped = np.unwrap(alpha)
    slope, intercept, r, p, se = linregress(x, alpha_unwrapped)
    return slope * 180/np.pi, r, p

# =========================
# PLOTTING
# =========================
def plot_bout_scatter(seg_id, rec_name, time_s, phase_rad, save_path):
    plt.figure(figsize=(6,4))
    plt.scatter(time_s, (np.degrees(phase_rad) % 360), s=22, alpha=0.85)
    plt.ylim(0, 360); plt.yticks([0,90,180,270,360])
    plt.xlabel("Time in bout (s)")
    plt.ylabel("LFP phase at optical peaks (deg)")
    plt.title(f"{rec_name} | Bout {seg_id}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_pooled_scatter(all_points, save_path):
    """
    all_points: list of dicts with keys 't_norm' (0..1), 'phase_deg'
    Fits a simple linear regression on unwrapped phase vs t_norm.
    """
    if not all_points:
        return
    t_norm = np.array([p["t_norm"] for p in all_points])
    phase_deg = np.array([p["phase_deg"] for p in all_points])

    # For a simple line, unwrap in radians then convert back just for slope display
    phase_rad = np.radians(phase_deg)
    # Unwrap per-bout would be ideal, but for pooled overview we rely on trend
    slope, intercept, r, p, _ = linregress(t_norm, np.unwrap(phase_rad))

    plt.figure(figsize=(6,5))
    plt.scatter(t_norm, phase_deg % 360, s=10, alpha=0.5)
    plt.ylim(0, 360); plt.yticks([0,90,180,270,360])
    plt.xlabel("Normalised time in bout (0→1)")
    plt.ylabel("LFP phase at optical peaks (deg)")
    plt.title(f"Pooled precession (slope={slope*180/np.pi:.1f} deg per norm‑unit, r={r:.2f}, p={p:.3g})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_rho_hist(rhos, save_path):
    plt.figure(figsize=(6,4))
    plt.hist(rhos, bins=20, edgecolor="k", alpha=0.8)
    plt.axvline(np.mean(rhos), linestyle="--")
    plt.xlabel("Circular–linear correlation ρ")
    plt.ylabel("Bout count")
    plt.title("Phase precession strength across bouts")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
def plot_pooled_scatter_two_cycles(all_points, save_path, fit_line=True):
    """
    Plot pooled optical-peak phases against normalised bout time (0..1)
    and render *two* cycles (0–360° and 360–720°).
    For display we duplicate points at +360°. For stats, we still fit on an
    unwrapped phase (no duplication).
    """
    if not all_points:
        return

    t_norm = np.array([p["t_norm"] for p in all_points])
    phase_deg = np.array([p["phase_deg"] for p in all_points]) % 360.0

    # Duplicate for the second cycle
    t_all = np.concatenate([t_norm, t_norm])
    phase_two = np.concatenate([phase_deg, phase_deg + 360.0])

    # Fit (optional): use unwrapped radians WITHOUT duplication to avoid bias
    slope_txt = "n/a"; r_txt = "n/a"; p_txt = "n/a"
    if fit_line:
        from scipy.stats import linregress
        phase_rad = np.radians(phase_deg)
        slope, intercept, r, p, _ = linregress(t_norm, np.unwrap(phase_rad))
        slope_deg_per_norm = slope * 180/np.pi
        slope_txt = f"{slope_deg_per_norm:.1f}"
        r_txt = f"{r:.2f}"
        p_txt = f"{p:.3g}"

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(t_all, phase_two, s=10, alpha=0.5)
    plt.ylim(0, 720)
    plt.yticks([0, 90, 180, 270, 360, 450, 540, 630, 720])
    plt.xlabel("Normalised time in bout (0→1)")
    plt.ylabel("LFP phase at optical peaks (deg)")
    plt.title(f"Pooled precession (two cycles) | slope={slope_txt}°/norm, r={r_txt}, p={p_txt}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    stats_dir = os.path.join(SAVE_ROOT, "stats_phase_precession")
    os.makedirs(stats_dir, exist_ok=True)

    seg_pkls = glob.glob(os.path.join(SAVE_ROOT, "SyncRecording*", "same_direction_segments.pkl"))
    print(f"Found {len(seg_pkls)} recordings with saved bouts.")

    all_rows = []
    pooled_points = []   # for pooled scatter (normalised time)
    example_plotted = 0

    for seg_pkl in seg_pkls:
        rec_dir = os.path.dirname(seg_pkl)
        rec_name = os.path.basename(rec_dir)
        with open(seg_pkl, "rb") as f:
            segments = pickle.load(f)

        this_rec_rows = []
        examples_dir = os.path.join(stats_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)

        for seg_id, seg in enumerate(segments, start=1):
            # Compute LFP phase & optical peaks
            peak_idx, lfp_phase_at_peaks, t_at_peaks, lfp_phase_full, t_full = lfp_phase_and_opt_peaks(
                seg, fs=FS, lfp_channel=LFP_CHANNEL, theta_low=THETA_LOW, theta_high=THETA_HIGH
            )
            if len(peak_idx) < MIN_PEAKS:
                continue

            # --- TESTS ---
            # (1) circular-linear correlation: phase (rad) vs time (s)
            rho, p_rho = circ_corrcl(lfp_phase_at_peaks, t_at_peaks)

            # (2) simple unwrapped linear slope in deg/s (for intuition)
            slope_deg_s, r_lin, p_lin = unwrap_slope(lfp_phase_at_peaks, t_at_peaks)

            row = {
                "recording": rec_name,
                "segment_id": seg_id,
                "n_peaks": int(len(peak_idx)),
                "rho_circ_lin": rho,
                "p_circ_lin": p_rho,
                "slope_deg_per_s": slope_deg_s,
                "r_linear_unwrapped": r_lin,
                "p_linear_unwrapped": p_lin,
            }
            this_rec_rows.append(row)
            all_rows.append(row)

            # --- EXAMPLE PER-BOUT SCATTER (limit to N_EXAMPLES total across all recs) ---
            if example_plotted < N_EXAMPLES:
                bout_png = os.path.join(examples_dir, f"{rec_name}_bout{seg_id:03d}_scatter.png")
                plot_bout_scatter(seg_id, rec_name, t_at_peaks, lfp_phase_at_peaks, bout_png)
                example_plotted += 1

            # --- POOLED POINTS (normalise time to 0..1 per bout for comparability) ---
            t0, t1 = t_full[0], t_full[-1]
            dur = max(1e-12, (t1 - t0))
            t_norm = (t_at_peaks - t0) / dur
            for tn, ph in zip(t_norm, lfp_phase_at_peaks):
                pooled_points.append({"t_norm": float(tn), "phase_deg": float(np.degrees(ph))})

        # per-recording CSV
        if this_rec_rows:
            rec_csv = os.path.join(stats_dir, f"{rec_name}_phase_precession_stats.csv")
            pd.DataFrame(this_rec_rows).to_csv(rec_csv, index=False)
            print(f"[{rec_name}] saved {len(this_rec_rows)} bout stats → {rec_csv}")

    # MASTER CSV
    if all_rows:
        master_df = pd.DataFrame(all_rows)
        master_csv = os.path.join(stats_dir, "MASTER_phase_precession_stats.csv")
        master_df.to_csv(master_csv, index=False)
        print(f"\nMASTER stats across recordings: {len(master_df)} bouts → {master_csv}")

        # Plots: pooled scatter + rho histogram
        pooled_png = os.path.join(stats_dir, "pooled_scatter.png")
        plot_pooled_scatter(pooled_points, pooled_png)
        print(f"Pooled scatter → {pooled_png}")
        pooled_png = os.path.join(stats_dir, "pooled_scatter_two_cycles.png")
        plot_pooled_scatter_two_cycles(pooled_points, pooled_png)


        rho_png = os.path.join(stats_dir, "rho_histogram.png")
        plot_rho_hist(master_df["rho_circ_lin"].values, rho_png)
        print(f"ρ histogram → {rho_png}")

        # Quick text summary
        sig = (master_df["p_circ_lin"] < 0.05).sum()
        print(f"\nSignificant circular–linear correlation in {sig}/{len(master_df)} bouts (p<0.05).")
    else:
        print("No bouts with sufficient peaks were found. Check MIN_PEAKS or your detection window.")
