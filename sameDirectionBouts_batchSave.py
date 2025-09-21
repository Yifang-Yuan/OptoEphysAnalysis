# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 20:47:21 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Batch runner: process ALL SyncRecording* folders in a parent directory,
extract same-direction+speed bouts, and save all outputs to one central folder.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

#====================
# USER CONFIG
#====================
PARENT_DIR = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion"
SAVE_ROOT  = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion\same_direction_outputs"

LFP_CHANNEL = "LFP_1"
HEAD_X, HEAD_Y = "head_x", "head_y"
SHOULDER_X, SHOULDER_Y = "shoulder_x", "shoulder_y"

FS = 10000
ANGLE_TOL_DEG = 60
MIN_DURATION_S = 1.2

# --- speed config ---
SPEED_THRESHOLD_CM = 1
SPEED_COLUMN = None            # e.g. "speed" if you know it; None = auto-detect
SPEED_SCALE_TO_CM_S = 1.0      # set 0.1 if your column is mm/s
SPEED_SMOOTH_WINDOW_S = 0.20

# theta band
THETA_LOW, THETA_HIGH = 4, 12

SMOOTH_WINDOW_S = 0.2

#====================
# HELPERS (unchanged from your version)
#====================
def load_pkl_dataframe(path):
    import pickle
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
    ang = np.degrees(np.arctan2(dy, dx))
    return (ang + 180) % 360 - 180

def smooth_deg_signal(deg, fs, window_s=0.25, polyorder=3):
    rad_unwrapped = np.unwrap(np.radians(deg))
    win = max(5, int(window_s * fs) // 2 * 2 + 1)
    sm = signal.savgol_filter(rad_unwrapped, win, polyorder, mode="interp")
    return np.degrees(sm)

def shortest_angular_distance_deg(a, b):
    return (a - b + 180) % 360 - 180

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

def ensure_speed_cm_s(df, fs=FS):
    out = df.copy()
    candidates = [SPEED_COLUMN] if SPEED_COLUMN else [
        "speed", "Speed", "speed_cm_s", "speed_cm_per_s",
        "inst_speed", "velocity", "vel", "speed_cm"
    ]
    col_found = None
    for c in candidates:
        if c and c in out.columns:
            col_found = c; break
    if col_found is None:
        raise KeyError("No speed column found; set SPEED_COLUMN or rename to a common name.")
    speed_cm_s = out[col_found].to_numpy().astype(float) * float(SPEED_SCALE_TO_CM_S)
    if SPEED_SMOOTH_WINDOW_S and SPEED_SMOOTH_WINDOW_S > 0:
        win = max(5, int(SPEED_SMOOTH_WINDOW_S * fs) // 2 * 2 + 1)
        speed_cm_s = signal.savgol_filter(speed_cm_s, win, polyorder=3, mode="interp")
    out["speed_cm_s"] = speed_cm_s
    return out

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
    df = df.copy()
    lfp_f = butter_bandpass(df[lfp_channel].to_numpy(), fs, theta_low, theta_high)
    opt_f = butter_bandpass(df["zscore_raw"].to_numpy(), fs, theta_low, theta_high)
    df["LFP_theta_filt"] = lfp_f
    df["optical_theta_filt"] = opt_f
    df["LFP_theta_angle"] = np.angle(signal.hilbert(lfp_f))
    df["optical_theta_angle"] = np.angle(signal.hilbert(opt_f))
    return df

def plot_theta_traces(theta_part, LFP_channel, start_time, end_time, fs=FS, save_path=None):
    start_idx = int(start_time * fs); end_idx = int(end_time * fs)
    time_vector = np.arange(len(theta_part)) / fs
    segment = theta_part.iloc[start_idx:end_idx].copy()
    t = time_vector[start_idx:end_idx]
    filtered_LFP = butter_bandpass(segment[LFP_channel].values, fs, THETA_LOW, THETA_HIGH)
    filtered_zscore = butter_bandpass(segment['zscore_raw'].values, fs, THETA_LOW, THETA_HIGH)
    zscore_lowpass = butter_lowpass(segment['zscore_raw'].to_numpy(), fs, cutoff=100)

    segment_peak_LFP = segment[(segment['LFP_theta_angle'] > 2.9) & (segment['LFP_theta_angle'] < 3.2)].index
    segment_peak_optical = segment[(segment['optical_theta_angle'] > 2.9) & (segment['optical_theta_angle'] < 3.2)].index

    min_dist_samples = int(0.08 * fs)
    peak_idx_LFP = filter_close_peaks(segment_peak_LFP.to_numpy(), min_dist_samples)
    peak_idx_optical = filter_close_peaks(segment_peak_optical.to_numpy(), min_dist_samples)

    peak_t_LFP = peak_idx_LFP / fs
    peak_t_optical = peak_idx_optical / fs

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelsize=14); ax.set_yticks([])

    axes[0].plot(t, segment[LFP_CHANNEL], color='black')
    axes[0].set_ylabel(LFP_CHANNEL, fontsize=14); axes[0].set_title('Raw LFP trace', fontsize=14)

    axes[1].plot(t, zscore_lowpass, color='green')
    axes[1].set_ylabel('zscore', fontsize=14); axes[1].set_title('Raw Optical Signal', fontsize=14)

    axes[2].plot(t, filtered_LFP, color='black', label='Filtered LFP')
    for pt in peak_t_LFP:
        if start_time <= pt <= end_time:
            axes[2].axvline(x=pt, color='red', linestyle='--', alpha=0.6)
    dot_y = np.interp(peak_t_optical, t, filtered_LFP)
    axes[2].scatter(peak_t_optical, dot_y, color='green', s=40, label='optical peaks')
    axes[2].set_ylabel('LFP theta', fontsize=14); axes[2].set_title('LFP Theta band + Optical Peaks', fontsize=14)

    axes[3].plot(t, filtered_zscore, color='green')
    for pt in peak_t_optical:
        if start_time <= pt <= end_time:
            axes[3].axvline(x=pt, color='red', linestyle='--', alpha=0.6)
    axes[3].set_ylabel('zscore theta', fontsize=14); axes[3].set_title('GEVI theta', fontsize=14)
    axes[3].set_xlabel('Time (s)', fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_phase_precession_dotplot(seg_df, fs=FS, lfp_channel=LFP_CHANNEL,
                                  theta_range=(THETA_LOW, THETA_HIGH), save_path=None):
    lfp_f = butter_bandpass(seg_df[lfp_channel].to_numpy(), fs, *theta_range)
    opt_f = butter_bandpass(seg_df["zscore_raw"].to_numpy(), fs, *theta_range)
    lfp_phase = np.angle(signal.hilbert(lfp_f))
    opt_phase = np.angle(signal.hilbert(opt_f))
    peak_idx = np.where((opt_phase > 2.9) & (opt_phase < 3.2))[0]
    min_dist = int(0.08 * fs)
    keep = []; last = -min_dist
    for i in peak_idx:
        if i - last >= min_dist:
            keep.append(i); last = i
    keep = np.array(keep, dtype=int)
    lfp_deg_at_opt_peaks = np.degrees(lfp_phase[keep]) % 360

    fig = plt.figure(figsize=(5, 6))
    plt.scatter(np.arange(len(lfp_deg_at_opt_peaks)), lfp_deg_at_opt_peaks, alpha=0.8, color='purple')
    plt.ylim(0, 360); plt.yticks(np.arange(0, 361, 90))
    plt.xlabel("Optical theta peak index"); plt.ylabel("LFP theta phase (deg)")
    plt.title("LFP phase at optical theta peaks"); plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

#====================
# BATCH DRIVER
#====================
def process_one_recording(rec_dir):
    """Process one SyncRecording* directory. Returns (meta_df, n_segments)."""
    pkl_path = os.path.join(rec_dir, "Ephys_tracking_photometry_aligned.pkl")
    if not os.path.isfile(pkl_path):
        print(f"[skip] No pkl in {rec_dir}")
        return pd.DataFrame(), 0

    # load
    df = load_pkl_dataframe(pkl_path)
    # heading + speed
    df = add_head_direction(df, fs=FS)
    df = ensure_speed_cm_s(df, fs=FS)
    # find bouts
    segments, meta, spans = find_same_direction_segments(
        df, fs=FS, angle_tol_deg=ANGLE_TOL_DEG,
        min_duration_s=MIN_DURATION_S, speed_threshold_cm=SPEED_THRESHOLD_CM
    )
    meta["recording_dir"] = os.path.basename(rec_dir)
    meta["pkl_path"] = pkl_path

    # per-record save folder
    rec_save_dir = os.path.join(SAVE_ROOT, os.path.basename(rec_dir))
    os.makedirs(rec_save_dir, exist_ok=True)

    # save segments list + meta
    with open(os.path.join(rec_save_dir, "same_direction_segments.pkl"), "wb") as f:
        pickle.dump(segments, f)
    meta.to_csv(os.path.join(rec_save_dir, "segments_meta.csv"), index=False)

    # compute theta angles once for whole df
    df_theta = add_theta_angles(df, fs=FS, lfp_channel=LFP_CHANNEL,
                                theta_low=THETA_LOW, theta_high=THETA_HIGH)

    # plots
    plots_dir = os.path.join(rec_save_dir, "plots_all_segments")
    os.makedirs(plots_dir, exist_ok=True)

    for i, (seg, (s, e)) in enumerate(zip(segments, spans), start=1):
        start_time = s / FS; end_time = e / FS
        base = f"seg{i:03d}_{start_time:.3f}s_{end_time:.3f}s"
        trace_png = os.path.join(plots_dir, base + "_theta_traces.png")
        dots_png  = os.path.join(plots_dir, base + "_phase_precession.png")

        print(f"[{os.path.basename(rec_dir)}] seg {i}: {start_time:.3f}-{end_time:.3f}s "
              f"({(e-s)/FS:.2f}s, mean speed {meta.loc[i-1,'mean_speed_cm_s']:.2f} cm/s)")

        plot_theta_traces(df_theta, LFP_CHANNEL, start_time, end_time, fs=FS, save_path=trace_png)
        plot_phase_precession_dotplot(seg, fs=FS, lfp_channel=LFP_CHANNEL, save_path=dots_png)

    return meta, len(segments)

if __name__ == "__main__":
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # find all SyncRecording* folders
    rec_dirs = sorted([d for d in glob.glob(os.path.join(PARENT_DIR, "SyncRecording*"))
                       if os.path.isdir(d)])
    print(f"Found {len(rec_dirs)} recording folders.")

    all_meta = []
    total_segments = 0
    for rec in rec_dirs:
        try:
            meta, nseg = process_one_recording(rec)
            all_meta.append(meta)
            total_segments += nseg
        except Exception as e:
            print(f"[error] {os.path.basename(rec)}: {e}")

    # save master meta
    if len(all_meta):
        master = pd.concat(all_meta, ignore_index=True)
        master_out = os.path.join(SAVE_ROOT, "MASTER_segments_meta.csv")
        master.to_csv(master_out, index=False)
        print(f"\nSaved MASTER meta with {len(master)} bouts across {len(rec_dirs)} folders:")
        print(master_out)
    print(f"Total bouts: {total_segments}")
