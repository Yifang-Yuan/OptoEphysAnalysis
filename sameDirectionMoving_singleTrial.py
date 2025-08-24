# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 22:56:41 2025

@author: yifan
"""

# ================================
# Same-direction segment extractor
# and theta precession plot helper
# ================================

import os
import pickle
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

#%%
# ---- config you may want to tweak ----
PKL_PATH = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\OpenField_DLCtracking\SyncRecording3/Ephys_tracking_photometry_aligned.pkl"
LFP_CHANNEL = "LFP_1"
HEAD_X, HEAD_Y = "head_x", "head_y"
SHOULDER_X, SHOULDER_Y = "shoulder_x", "shoulder_y"
OUT_DIR=r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\OpenField_DLCtracking"
FS = 10000                    # sampling rate of your aligned dataframe (Hz)
ANGLE_TOL_DEG = 60            # stay within ±20° (set to 30 if you prefer)
MIN_DURATION_S = 1.5          # at least 3 seconds

# --- SPEED CONFIG (new) ---
SPEED_THRESHOLD_CM = 1.5     # keep only periods with speed > 3 cm/s
SPEED_COLUMN = None           # e.g. "speed"; if None we'll try a few common names
SPEED_SCALE_TO_CM_S = 1.0     # multiply your column to convert to cm/s (e.g., mm/s -> 0.1)
SPEED_SMOOTH_WINDOW_S = 0.20  # light smoothing for robustness; set 0 to disable

SMOOTH_WINDOW_S = 0.2         # smooth head dir a bit to avoid jitter (Savitzky–Golay)
THETA_LOW, THETA_HIGH = 4, 9 # theta band for filtering


# ---------- helpers ----------
def load_pkl_dataframe(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v.copy()
    raise ValueError("No DataFrame found in pickle.")

def butter_bandpass(x, fs, low, high, order=4):
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)

def butter_lowpass(x, fs, cutoff, order=4):
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)

def angle_from_vec(dx, dy):
    """Return direction in degrees wrapped to [-180, 180)."""
    ang = np.degrees(np.arctan2(dy, dx))
    return (ang + 180) % 360 - 180

def smooth_deg_signal(deg, fs, window_s=0.25, polyorder=3):
    rad_unwrapped = np.unwrap(np.radians(deg))
    win = max(5, int(window_s * fs) // 2 * 2 + 1)  # odd >=5
    sm = signal.savgol_filter(rad_unwrapped, win, polyorder, mode="interp")
    return np.degrees(sm)

def shortest_angular_distance_deg(a, b):
    """Minimal signed angular distance a-b in degrees."""
    d = (a - b + 180) % 360 - 180
    return d

def contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def filter_close_peaks(peak_indices, min_distance_samples):
    if len(peak_indices) == 0:
        return np.array([], dtype=int)
    filtered = [int(peak_indices[0])]
    for idx in peak_indices[1:]:
        if int(idx) - filtered[-1] >= min_distance_samples:
            filtered.append(int(idx))
    return np.array(filtered, dtype=int)

# --- NEW: ensure we have speed in cm/s ---
def ensure_speed_cm_s(df, fs=FS):
    """
    Ensure df has a 'speed_cm_s' column.
    Priority:
      1) Use SPEED_COLUMN if set and present.
      2) Auto-detect common names.
      3) If none found, raise a helpful error (keeps your pipeline explicit).
    Applies an optional light smoothing for gating robustness.
    """
    out = df.copy()
    candidates = [SPEED_COLUMN] if SPEED_COLUMN else [
        "speed", "Speed", "speed_cm_s", "speed_cm_per_s",
        "inst_speed", "velocity", "vel", "speed_cm"
    ]
    col_found = None
    for c in candidates:
        if c and c in out.columns:
            col_found = c
            break
    if col_found is None:
        raise KeyError(
            "No speed column found. Set SPEED_COLUMN to the correct column name, "
            "or rename your speed column to one of: "
            "'speed', 'speed_cm_s', 'inst_speed', 'velocity', 'vel', 'speed_cm'."
        )

    speed_cm_s = out[col_found].to_numpy().astype(float) * float(SPEED_SCALE_TO_CM_S)

    # Optional small smoothing (helps avoid micro-spikes around the threshold)
    if SPEED_SMOOTH_WINDOW_S and SPEED_SMOOTH_WINDOW_S > 0:
        win = max(5, int(SPEED_SMOOTH_WINDOW_S * fs) // 2 * 2 + 1)
        speed_cm_s = signal.savgol_filter(speed_cm_s, win, polyorder=3, mode="interp")

    out["speed_cm_s"] = speed_cm_s
    return out

# ---------- core processing ----------
def add_head_direction(df, fs=FS):
    hx, hy = df[HEAD_X].to_numpy(), df[HEAD_Y].to_numpy()
    sx, sy = df[SHOULDER_X].to_numpy(), df[SHOULDER_Y].to_numpy()
    head_deg = angle_from_vec(hx - sx, hy - sy)
    head_deg_smooth = smooth_deg_signal(head_deg, fs, window_s=SMOOTH_WINDOW_S)
    df = df.copy()
    df["head_direction_deg"] = head_deg
    df["head_direction_deg_smooth"] = head_deg_smooth
    return df

def find_same_direction_segments(df,
                                 fs=FS,
                                 angle_tol_deg=ANGLE_TOL_DEG,
                                 min_duration_s=MIN_DURATION_S,
                                 speed_threshold_cm=SPEED_THRESHOLD_CM):
    """
    Segments where:
      (1) deviation from smoothed heading stays within ±angle_tol_deg
      (2) speed_cm_s > speed_threshold_cm
      (3) duration ≥ min_duration_s
    """
    dev = np.abs(shortest_angular_distance_deg(
        df["head_direction_deg"].to_numpy(),
        df["head_direction_deg_smooth"].to_numpy()
    ))
    mask_dir   = dev <= angle_tol_deg
    if "speed_cm_s" not in df.columns:
        raise KeyError("speed_cm_s not present. Call ensure_speed_cm_s(df) before segmentation.")
    mask_speed = df["speed_cm_s"].to_numpy() > speed_threshold_cm
    mask = mask_dir & mask_speed

    min_len = int(min_duration_s * fs)
    spans = [(s, e) for s, e in contiguous_regions(mask) if e - s >= min_len]
    segments = [df.iloc[s:e].copy() for s, e in spans]
    meta = pd.DataFrame({
        "start_idx": [s for s, _ in spans],
        "end_idx":   [e for _, e in spans],
        "start_time_s": [s / fs for s, _ in spans],
        "end_time_s":   [e / fs for _, e in spans],
        "duration_s":   [(e - s) / fs for s, e in spans],
        "mean_speed_cm_s": [df["speed_cm_s"].iloc[s:e].mean() for s, e in spans],
        "median_speed_cm_s": [df["speed_cm_s"].iloc[s:e].median() for s, e in spans],
    })
    return segments, meta, spans

def add_theta_angles(df, fs=FS, lfp_channel=LFP_CHANNEL, theta_low=THETA_LOW, theta_high=THETA_HIGH):
    """Bandpass → Hilbert → angles (in radians). Also keep filtered traces."""
    df = df.copy()
    lfp_f = butter_bandpass(df[lfp_channel].to_numpy(), fs, theta_low, theta_high)
    opt_f = butter_bandpass(df["zscore_raw"].to_numpy(), fs, theta_low, theta_high)
    df["LFP_theta_filt"] = lfp_f
    df["optical_theta_filt"] = opt_f
    df["LFP_theta_angle"] = np.angle(signal.hilbert(lfp_f))
    df["optical_theta_angle"] = np.angle(signal.hilbert(opt_f))
    return df

# ---------- plotting (added optional save) ----------
def plot_theta_traces(theta_part, LFP_channel, start_time, end_time, fs=FS, save_path=None):
    # Convert time to index
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    time_vector = np.arange(len(theta_part)) / fs

    # Subset and time axis
    segment = theta_part.iloc[start_idx:end_idx].copy()
    t = time_vector[start_idx:end_idx]

    # Apply theta bandpass filter
    filtered_LFP = butter_bandpass(segment[LFP_channel].values, fs, THETA_LOW, THETA_HIGH)
    filtered_zscore = butter_bandpass(segment['zscore_raw'].values, fs, THETA_LOW, THETA_HIGH)

    # Smooth optical raw (replacement for OE.smooth_signal)
    zscore_lowpass = butter_lowpass(segment['zscore_raw'].to_numpy(), fs, cutoff=100)

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
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Minimalist styling
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelsize=14)
        ax.set_yticks([])

    # 1. Raw LFP trace
    axes[0].plot(t, segment[LFP_CHANNEL], color='black')
    axes[0].set_ylabel(LFP_CHANNEL, fontsize=14)
    axes[0].set_title('Raw LFP trace', fontsize=14)

    # 2. zscore_raw (smoothed for readability)
    axes[1].plot(t, zscore_lowpass, color='green')
    axes[1].set_ylabel('zscore', fontsize=14)
    axes[1].set_title('Raw Optical Signal', fontsize=14)

    # 3. Filtered LFP + LFP peaks + optical peaks as dots
    axes[2].plot(t, filtered_LFP, color='black', label='Filtered LFP')
    for pt in peak_t_LFP:
        if start_time <= pt <= end_time:
            axes[2].axvline(x=pt, color='red', linestyle='--', alpha=0.6)

    # Overlay optical theta peaks as dots (sample at filtered LFP for y)
    dot_y = np.interp(peak_t_optical, t, filtered_LFP)
    axes[2].scatter(peak_t_optical, dot_y, color='green', marker='o', s=40, label='optical peaks')

    axes[2].set_ylabel('LFP theta', fontsize=14)
    axes[2].set_title('LFP Theta band + Optical Peaks', fontsize=14)

    # 4. Filtered optical + optical peaks (as verticals)
    axes[3].plot(t, filtered_zscore, color='green')
    for pt in peak_t_optical:
        if start_time <= pt <= end_time:
            axes[3].axvline(x=pt, color='red', linestyle='--', alpha=0.6)

    axes[3].set_ylabel('zscore theta', fontsize=14)
    axes[3].set_title('GEVI theta', fontsize=14)
    axes[3].set_xlabel('Time (s)', fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_phase_precession_dotplot(seg_df,
                                  fs=FS,
                                  lfp_channel=LFP_CHANNEL,
                                  theta_range=(THETA_LOW, THETA_HIGH),
                                  save_path=None):
    """Scatter of LFP theta phase at optical theta peaks."""
    # Bandpass both
    lfp_f = butter_bandpass(seg_df[lfp_channel].to_numpy(), fs, *theta_range)
    opt_f = butter_bandpass(seg_df["zscore_raw"].to_numpy(), fs, *theta_range)

    # Hilbert → phase
    lfp_phase = np.angle(signal.hilbert(lfp_f))
    opt_phase = np.angle(signal.hilbert(opt_f))

    # Optical theta peaks near π, ≥80 ms apart
    peak_idx = np.where((opt_phase > 2.9) & (opt_phase < 3.2))[0]
    min_dist = int(0.08 * fs)
    keep = []
    last = -min_dist
    for i in peak_idx:
        if i - last >= min_dist:
            keep.append(i)
            last = i
    keep = np.array(keep, dtype=int)

    # LFP phase at those optical peaks
    lfp_deg_at_opt_peaks = np.degrees(lfp_phase[keep]) % 360

    # Plot
    plt.figure(figsize=(5, 6))
    plt.scatter(np.arange(len(lfp_deg_at_opt_peaks)), lfp_deg_at_opt_peaks,
                alpha=0.8, color='purple')
    plt.ylim(0, 360)
    plt.yticks(np.arange(0, 361, 90))
    plt.xlabel("Optical theta peak index")
    plt.ylabel("LFP theta phase (deg)")
    plt.title("LFP phase at optical theta peaks")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return lfp_deg_at_opt_peaks

# ---------- run ----------
#%%
if __name__ == "__main__":
    df = load_pkl_dataframe(PKL_PATH)

    # 0) add heading and ensure we have a speed column in cm/s
    df = add_head_direction(df, fs=FS)
    df = ensure_speed_cm_s(df, fs=FS)

    # 1) segments with BOTH constraints (heading ±ANGLE_TOL_DEG AND speed > threshold)
    segments, meta, spans = find_same_direction_segments(
        df,
        fs=FS,
        angle_tol_deg=ANGLE_TOL_DEG,
        min_duration_s=MIN_DURATION_S,
        speed_threshold_cm=SPEED_THRESHOLD_CM
    )
    print(f"Found {len(segments)} same-direction segments "
          f"(≥{MIN_DURATION_S}s, ±{ANGLE_TOL_DEG}°, speed > {SPEED_THRESHOLD_CM} cm/s).")
    if not meta.empty:
        print(meta.head())

    # 2) save segments + meta
    if OUT_DIR is None:
        OUT_DIR = os.path.join(os.path.dirname(PKL_PATH), "same_direction_outputs")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "same_direction_segments.pkl"), "wb") as f:
        pickle.dump(segments, f)
    meta.to_csv(os.path.join(OUT_DIR, "segments_meta.csv"), index=False)

    # 3) compute theta angles ONCE for the whole dataframe
    df_theta = add_theta_angles(df, fs=FS, lfp_channel=LFP_CHANNEL,
                                theta_low=THETA_LOW, theta_high=THETA_HIGH)

    # 4) loop all segments and plot FULL spans (and SAVE)
    plots_dir = os.path.join(OUT_DIR, "plots_all_segments")
    os.makedirs(plots_dir, exist_ok=True)

    for i, (seg, (s, e)) in enumerate(zip(segments, spans), start=1):
        start_time = s / FS
        end_time   = e / FS    # full segment

        base = f"seg{i:03d}_{start_time:.3f}s_{end_time:.3f}s"
        trace_png = os.path.join(plots_dir, base + "_theta_traces.png")
        dots_png  = os.path.join(plots_dir, base + "_phase_precession.png")

        print(f"Plotting segment {i}: {start_time:.3f}–{end_time:.3f} s "
              f"({(e-s)/FS:.2f} s, mean speed {meta.loc[i-1,'mean_speed_cm_s']:.2f} cm/s)")

        # 4‑panel theta trace for the WHOLE segment (saved)
        plot_theta_traces(df_theta, LFP_CHANNEL, start_time, end_time, fs=FS, save_path=trace_png)

        # Phase‑precession dot plot for the SAME segment (saved)
        plot_phase_precession_dotplot(seg, fs=FS, lfp_channel=LFP_CHANNEL, save_path=dots_png)
