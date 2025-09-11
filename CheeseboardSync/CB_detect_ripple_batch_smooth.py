# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 23:11:02 2025

@author: yifan
"""
from __future__ import annotations
import os
import pickle
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt

# --- pynapple / pynacollada ---
try:
    import pynapple as nap
    import pynacollada as pyna
except Exception:
    raise ImportError("Please install pynapple and pynacollada to run ripple detection.")
    
def _row_normalize(X: np.ndarray, T: np.ndarray,
                   method: str = "zscore",
                   baseline_window: tuple[float, float] = (-0.08, -0.02)) -> np.ndarray:
    """
    Normalise each row of X using stats from a baseline window on T.
    X: (n_events, nT), T: (nT,)
    """
    if X is None:
        return None
    X = X.copy()
    t0, t1 = baseline_window
    idx = (T >= t0) & (T <= t1)
    if not np.any(idx):
        # Fallback to whole window if baseline slice is empty
        idx = np.isfinite(T)

    base = X[:, idx]
    mu = np.nanmean(base, axis=1, keepdims=True)

    if method == "demean":
        X -= mu
        return X

    if method == "percent":
        denom = np.maximum(np.abs(mu), 1e-9)
        return (X - mu) / denom

    # default zscore
    sd = np.nanstd(base, axis=1, ddof=1, keepdims=True)
    sd = np.where(sd <= 0, 1.0, sd)
    return (X - mu) / sd

# ---------- smoothing (your function) ----------
def smooth_signal(x, window_len=10, window='flat'):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("smooth_signal only accepts 1D arrays.")
    n = x.size
    if n == 0:
        return x
    if window_len < 3:
        return x.copy()
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be one of 'flat','hanning','hamming','bartlett','blackman'")
    xi = x.copy()
    idx = np.arange(n); finite = np.isfinite(xi)
    if finite.any():
        first, last = idx[finite][0], idx[finite][-1]
        if first > 0: xi[:first] = xi[first]
        if last < n-1: xi[last+1:] = xi[last]
        bad = ~np.isfinite(xi)
        if bad.any(): xi[bad] = np.interp(idx[bad], idx[finite], xi[finite])
    else:
        return np.full_like(x, np.nan)
    window_len = int(max(3, window_len))
    if window_len % 2 == 0: window_len += 1
    if window_len > n:
        window_len = n if (n % 2 == 1) else (n - 1)
        window_len = max(3, window_len)
    w = np.ones(window_len, float) if window=='flat' else getattr(np, window)(window_len).astype(float)
    w = w / w.sum()
    pad = window_len // 2
    s = np.pad(xi, (pad, pad), mode='reflect')
    y = np.convolve(w, s, mode='valid')
    return y

# ---------- helpers ----------
def compute_px_to_cm_scale(
    cheeseboard_center: Tuple[float, float],
    cheeseboard_ends,
    real_diameter_cm: float = 100.0
) -> float:
    cx, cy = cheeseboard_center
    radii = [np.hypot(x - cx, y - cy) for x, y in cheeseboard_ends]
    diameter_px = 2.0 * np.mean(radii)
    return real_diameter_cm / diameter_px if diameter_px > 0 else 1.0

def compute_speed_cm_per_s(
    t: np.ndarray,
    head_xy: Tuple[np.ndarray, np.ndarray],
    px_to_cm: float
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = head_xy
    good = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t = t[good]; x = x[good]; y = y[good]
    if t.size < 2:
        return np.array([]), np.array([])
    dt = np.diff(t); dt[dt == 0] = np.nan
    speed_px = np.hypot(np.diff(x), np.diff(y)) / dt
    speed_cm = speed_px * px_to_cm
    tc = 0.5 * (t[:-1] + t[1:])
    return tc, speed_cm

def mask_to_intervals(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Convert a boolean mask into contiguous (start_time, end_time) intervals."""
    m = mask.astype(bool)
    if not m.any():
        return []
    d = np.diff(m.astype(int))
    starts = np.flatnonzero(d == 1) + 1
    ends   = np.flatnonzero(d == -1) + 1
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size]
    return [(float(t[s]), float(t[e - 1])) for s, e in zip(starts, ends)]

# ---------- ripple detector (same logic as your working version) ----------
def getRippleEvents(
    lfp_raw: nap.Tsd,
    Fs: float,
    windowlen: int = 500,
    Low_thres: float = 1.0,
    High_thres: float = 10.0,
    low_freq: float = 130.0,
    high_freq: float = 250.0
):
    """Return ripple-band filtered LFP, nSS (z), nSS thresholded, ripple epochs, and ripple peaks (Tsd)."""
    ripple_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, low_freq, high_freq, Fs)  # 130–250 Hz
    squared_signal = np.square(ripple_band_filtered.values)
    window = np.ones(int(windowlen)) / float(windowlen)

    nSS_arr = filtfilt(window, 1, squared_signal)
    nSS_arr = (nSS_arr - np.mean(nSS_arr)) / np.std(nSS_arr)
    nSS = nap.Tsd(
        t=ripple_band_filtered.index.values,
        d=nSS_arr,
        time_support=ripple_band_filtered.time_support
    )

    nSS2 = nSS.threshold(Low_thres, method='above')
    nSS3 = nSS2.threshold(High_thres, method='below')

    # duration constraints (ms)
    rip_ep = nSS3.time_support
    rip_ep = rip_ep.drop_short_intervals(20, time_units='ms')
    rip_ep = rip_ep.drop_long_intervals(200, time_units='ms')

    # peaks (one per epoch via argmax in nSS)
    rip_vals, rip_times = [], []
    for s, e in rip_ep.values:
        seg = nSS.loc[s:e]
        if len(seg) == 0:
            continue
        rip_times.append(seg.idxmax())
        rip_vals.append(seg.max())

    rip_tsd = nap.Tsd(t=np.array(rip_times), d=np.array(rip_vals))
    return ripple_band_filtered, nSS, nSS3, rip_ep, rip_tsd

# -------- per-sync collector (now smooths photometry @100 Hz *before* stacking) --------
def _collect_ripples_one_sync(
    sync_dir: str,
    landmarks: Dict,
    lfp_channel: str = "LFP_4",
    speed_thresh_cm_s: float = 3.0,
    ripple_win_s: float = 0.200,          # ±200 ms total window
    ripple_smooth_win_samples: int = 500,
    low_high_thresh: tuple[float, float] = (1.0, 10.0),
    ripple_band: tuple[float, float] = (130.0, 250.0),
    fs_plot: float = 1000.0               # resample for stacking/plots
):
    """
    Returns (T, RB_stack, LFP_stack, PH_stack, opt_label) for a single SyncRecording*.
      - T: time grid (s), shape (nT,)
      - RB_stack: ripple-band filtered LFP, shape (n_events, nT)
      - LFP_stack: raw LFP, shape (n_events, nT)
      - PH_stack: photometry z (smoothed @100 Hz), shape (n_events, nT) or None
    """
    import numpy as np
    import os, pickle

    pkl = os.path.join(sync_dir, "aligned_cheeseboard.pkl")
    if not os.path.isfile(pkl):
        return None, None, None, None, "GEVI"

    with open(pkl, "rb") as f:
        D = pickle.load(f)
    e = D["ephys"]; p = D["phot"]; b = D["beh"]

    # Raw arrays
    t_e  = np.asarray(e["t"], float)
    lfp  = np.asarray(e[lfp_channel], float)

    t_ph = np.asarray(p["t"], float)
    # Prefer 'green', else 'z'
    if p.get("z") is not None:
        zraw = np.asarray(p["z"], float); opt_label = "GEVI z"
    elif p.get("green") is not None:
        zraw = np.asarray(p["green"], float);     opt_label = "GEVI green"
    else:
        zraw = None;                          opt_label = "GEVI"
    fs_ph = float(p.get("fs", 1682.92))

    t_b = np.asarray(b["t"], float)
    headX, headY = np.asarray(b["head"][0], float), np.asarray(b["head"][1], float)
    neckX = neckY = None
    if "neck" in b and b["neck"] is not None:
        neckX, neckY = np.asarray(b["neck"][0], float), np.asarray(b["neck"][1], float)

    approach_t = b.get("approach_time", None)
    if approach_t is None or not np.isfinite(approach_t):
        return None, None, None, None, opt_label

    # Reward-zone leave time (after approach): head & neck outside zone
    cx, cy = landmarks["reward_pt"]; R = float(landmarks.get("reward_zone_radius", 15.0))
    in_head = ((headX - cx)**2 + (headY - cy)**2) <= (R*R)
    if neckX is not None:
        in_neck = ((neckX - cx)**2 + (neckY - cy)**2) <= (R*R)
        both_out = (~in_head) & (~in_neck)
    else:
        both_out = (~in_head)

    post_mask = (t_b >= approach_t)
    post_idx = np.flatnonzero(post_mask & both_out)
    if post_idx.size:
        leave_time = float(t_b[post_idx[0]])
    else:
        inside_post = np.flatnonzero(post_mask & (~both_out))
        leave_time = float(t_b[inside_post[-1]]) if inside_post.size else float(min(t_b[-1], approach_t + 1.0))

    # Immobility (< speed_thresh) INSIDE [approach, leave]
    px_to_cm = compute_px_to_cm_scale(landmarks["cheeseboard_center"], landmarks["cheeseboard_ends"], 100.0)
    ts_sp, sp = compute_speed_cm_per_s(t_b, (headX, headY), px_to_cm)
    rw_mask = (ts_sp >= approach_t) & (ts_sp <= leave_time)
    imm_mask = rw_mask & (sp < speed_thresh_cm_s) & np.isfinite(sp)

    imm_intervals = mask_to_intervals(ts_sp, imm_mask)
    if not imm_intervals:
        return None, None, None, None, opt_label

    ep_rw = nap.IntervalSet(start=approach_t, end=leave_time, time_units='s')
    ep_imm = nap.IntervalSet(start=[s for s,_ in imm_intervals],
                             end=[e for _,e in imm_intervals], time_units='s')

    # Build Tsd
    dt_e = np.nanmedian(np.diff(t_e))
    if not (np.isfinite(dt_e) and dt_e > 0):
        return None, None, None, None, opt_label
    fs_e = 1.0 / dt_e
    LFP = nap.Tsd(t=t_e, d=lfp, time_units='s')
    LFP_rw = LFP.restrict(ep_rw)

    # photometry (z for stacking)
    if (zraw is not None) and t_ph.size:
        z_mean = np.nanmean(zraw); z_std = np.nanstd(zraw, ddof=1)
        ZSTD = nap.Tsd(t=t_ph, d=((zraw - (z_mean if np.isfinite(z_mean) else 0.0)) /
                                   (z_std if np.isfinite(z_std) and z_std > 0 else 1.0)), time_units='s')
    else:
        ZSTD = None

    # Ripple detection on reward window; keep peaks/epochs inside immobility
    rb_lo, rb_hi = ripple_band
    low_th, high_th = low_high_thresh
    RB_rw, nSS_rw, _, rip_ep_rw, rip_tsd_rw = getRippleEvents(
        LFP_rw, Fs=fs_e, windowlen=ripple_smooth_win_samples,
        Low_thres=low_th, High_thres=high_th, low_freq=rb_lo, high_freq=rb_hi
    )
    rip_tsd = rip_tsd_rw.restrict(ep_imm)
    rip_ep  = rip_ep_rw.intersect(ep_imm)
    if rip_tsd.values.size == 0:
        return None, None, None, None, opt_label

    # Build aligned windows on a common grid
    half = ripple_win_s         # e.g., 0.200 s
    pad  = 0.030
    T = np.arange(-half, half + 1.0/fs_plot, 1.0/fs_plot)  # stacking grid (e.g. 1 kHz or 500 Hz)
    T100 = np.arange(-half, half + 0.01, 0.01)             # 100 Hz grid for photometry smoothing
    RB_stack, LFP_stack, PH_stack = [], [], []

    n = min(len(rip_ep), len(rip_tsd))
    for i in range(n):
        t_peak = float(rip_tsd.index.values[i])
        # guard edges on full LFP
        if (t_peak - LFP.index.values[0]) < (half + pad) or (LFP.index.values[-1] - t_peak) < (half + pad):
            continue

        # Filter around peak
        win_big   = nap.IntervalSet(start=t_peak - (half+pad), end=t_peak + (half+pad), time_units='s')
        LFP_big   = LFP.restrict(win_big)
        RB_big    = pyna.eeg_processing.bandpass_filter(LFP_big, rb_lo, rb_hi, fs_e)

        win_small = nap.IntervalSet(start=t_peak - half, end=t_peak + half, time_units='s')
        LFP_win   = LFP_big.restrict(win_small)
        RB_win    = RB_big.restrict(win_small)

        # Interp LFP/ripple-band onto stacking grid T (relative to t_peak)
        tL, xL = LFP_win.index.values, LFP_win.values
        tR, xR = RB_win.index.values,  RB_win.values
        if (tL.size < 2) or (tR.size < 2):
            continue
        LFP_stack.append(np.interp(t_peak + T, tL, xL, left=np.nan, right=np.nan))
        RB_stack.append( np.interp(t_peak + T, tR, xR, left=np.nan, right=np.nan))

        # Photometry: resample to 100 Hz, smooth, then (optionally) re-interp onto T
        if ZSTD is not None:
            Z_win = ZSTD.restrict(win_small)
            tZ, xZ = Z_win.index.values, Z_win.values
            if tZ.size >= 2:
                # relative to peak
                tZ_rel = tZ - t_peak
                # project onto 100 Hz grid
                z_100 = np.interp(T100, tZ_rel, xZ, left=np.nan, right=np.nan)
                # smooth at 100 Hz
                z_100_sm = smooth_signal(z_100, window_len=3, window='hanning')
                # re-interp smoothed 100 Hz trace back onto stacking grid T
                PH_stack.append(np.interp(T, T100, z_100_sm, left=np.nan, right=np.nan))

    if len(RB_stack) == 0:
        return None, None, None, None, opt_label

    RB_stack  = np.vstack(RB_stack)            # (n_events, nT)
    LFP_stack = np.vstack(LFP_stack)
    PH_stack  = np.vstack(PH_stack) if len(PH_stack) else None
    return T, RB_stack, LFP_stack, PH_stack, opt_label

# -------- batch across days / syncs; build mean/CI + heatmaps --------
def batch_ripple_aligned_summary(
    animal_root: str,
    landmarks: Dict,
    days: list[str] = ["Day1","Day2","Day3","Day4"],
    lfp_channel: str = "LFP_4",
    speed_thresh_cm_s: float = 3.0,
    ripple_win_s: float = 0.100,
    ripple_smooth_win_samples: int = 500,
    low_high_thresh: tuple[float, float] = (1.0, 10.0),
    ripple_band: tuple[float, float] = (130.0, 250.0),
    fs_plot: float = 1000.0,
    show: bool = True,
    savepath: str | None = None,
    row_norm_method: str = "zscore",                  # "zscore"|"demean"|"percent"|"none"
    row_norm_baseline: tuple[float, float] = (-0.08, -0.02)
):
    """
    Runs all Day*/SyncRecording* under animal_root and produces a figure:
      1) mean±95% CI ripple-band LFP
      2) mean±95% CI raw LFP
      3) mean±95% CI photometry (smoothed @100 Hz) if available
      4) heatmap of raw LFP (events × time)
      5) heatmap of photometry (smoothed) if available
    """
    import glob, os
    import numpy as np
    import matplotlib.pyplot as plt

    T_all = None
    RB_all, LFP_all, PH_all = [], [], []
    opt_label_seen = "GEVI"

    for day in days:
        day_dir = os.path.join(animal_root, day)
        if not os.path.isdir(day_dir):
            continue
        for sync in sorted(p for p in glob.glob(os.path.join(day_dir, "SyncRecording*")) if os.path.isdir(p)):
            out = _collect_ripples_one_sync(
                sync, landmarks,
                lfp_channel=lfp_channel,
                speed_thresh_cm_s=speed_thresh_cm_s,
                ripple_win_s=ripple_win_s,
                ripple_smooth_win_samples=ripple_smooth_win_samples,
                low_high_thresh=low_high_thresh,
                ripple_band=ripple_band,
                fs_plot=fs_plot
            )
            T, RB, LFP, PH, opt_label = out
            if T is None:
                continue
            if T_all is None:
                T_all = T
            RB_all.append(RB); LFP_all.append(LFP)
            if PH is not None:
                PH_all.append(PH)
                opt_label_seen = opt_label

    if not RB_all:
        print("No ripple epochs found across the requested days.")
        return

    # stack across all syncs
    RB = np.vstack(RB_all)           # (N, nT)
    L  = np.vstack(LFP_all)
    P  = np.vstack(PH_all) if PH_all else None
    n_events = RB.shape[0]
    # -------- per-epoch (row) normalisation BEFORE averaging/heatmaps --------
    if row_norm_method and row_norm_method.lower() != "none":
        RB = _row_normalize(RB, T_all, method=row_norm_method, baseline_window=row_norm_baseline)
        L  = _row_normalize(L,  T_all, method=row_norm_method, baseline_window=row_norm_baseline)
        if P is not None:
            P  = _row_normalize(P,  T_all, method=row_norm_method, baseline_window=row_norm_baseline)

    def _mean_ci95(X):
        m = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        n  = np.sum(np.isfinite(X), axis=0)
        ci = 1.96 * (sd / np.sqrt(np.maximum(n, 1)))
        return m, ci

    mRB, ciRB = _mean_ci95(RB)
    mL,  ciL  = _mean_ci95(L)
    if P is not None:
        mP,  ciP  = _mean_ci95(P)

    # ------------ figure ------------
    have_phot = P is not None
    nrows = 4 + (1 if have_phot else 0)
    fig, axes = plt.subplots(nrows, 1, figsize=(6, 10), sharex=True)
    idx = 0

    # Ripple band mean ± CI
    ax = axes[idx]; idx += 1
    ax.fill_between(T_all, mRB-ciRB, mRB+ciRB, color='0.8', alpha=0.8, linewidth=0)
    ax.plot(T_all, mRB, color='0.2', lw=1.5)
    ax.axvline(0, color='r', lw=1)
    ax.set_ylabel("Ripple band")
    ax.set_title(f"Aligned at ripple peak  (N ripples = {n_events})", pad=6)

    # LFP mean ± CI
    ax = axes[idx]; idx += 1
    ax.fill_between(T_all, mL-ciL, mL+ciL, color='#9ec9ff', alpha=0.7, linewidth=0)
    ax.plot(T_all, mL, color='#1f77b4', lw=2)
    ax.axvline(0, color='r', lw=1)
    ax.set_ylabel("LFP")

    # Photometry (smoothed @100 Hz) mean ± CI
    if have_phot:
        ax = axes[idx]; idx += 1
        ax.fill_between(T_all, mP-ciP, mP+ciP, color='#bfe7bf', alpha=0.8, linewidth=0)
        ax.plot(T_all, mP, color='#2ca02c', lw=2)
        ax.axvline(0, color='r', lw=1)
        ax.set_ylabel(opt_label_seen + " (z, smoothed)")

    # LFP heatmap
    ax = axes[idx]; idx += 1
    im = ax.imshow(L, aspect='auto', origin='lower',
                   extent=[T_all[0], T_all[-1], 1, L.shape[0]],
                   cmap='viridis')
    ax.axvline(0, color='r', lw=1)
    ax.set_ylabel("Ripple #")
    ax.set_title("LFP (events × time)")
    fig.colorbar(im, ax=ax, fraction=0.020, pad=0.02)

    # Photometry heatmap (smoothed)
    if have_phot:
        ax = axes[idx]
        im2 = ax.imshow(P, aspect='auto', origin='lower',
                        extent=[T_all[0], T_all[-1], 1, P.shape[0]],
                        cmap='viridis')
        ax.axvline(0, color='r', lw=1)
        ax.set_ylabel("Ripple #")
        ax.set_title(f"{opt_label_seen} (events × time, smoothed)")
        fig.colorbar(im2, ax=ax, fraction=0.020, pad=0.02)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# example call
batch_ripple_aligned_summary(
    animal_root=r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success",
    landmarks={
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 15.0,
    },
    days=["Day2","Day3","Day4"],
    lfp_channel="LFP_4",
    speed_thresh_cm_s=3.0,
    ripple_win_s=0.100,
    ripple_smooth_win_samples=100,
    low_high_thresh=(1.1, 10.0),
    ripple_band=(130.0, 250.0),
    fs_plot=500.0,
    show=True,
    savepath=None  # or r"C:\path\to\save\ripple_summary.png"
)
