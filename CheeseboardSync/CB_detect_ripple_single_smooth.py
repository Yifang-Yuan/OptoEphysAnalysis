# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:29:32 2025

@author: yifan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# ---------- smoothing ----------
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
    if window_len % 2 == 0:
        window_len += 1
    if window_len > n:
        window_len = n if (n % 2 == 1) else (n - 1)
        window_len = max(3, window_len)
    w = np.ones(window_len, float) if window=='flat' else getattr(np, window)(window_len).astype(float)
    w = w / w.sum()
    pad = window_len // 2
    s = np.pad(xi, (pad, pad), mode='reflect')
    y = np.convolve(w, s, mode='valid')
    return y

def resample_and_smooth_100Hz(tsd: nap.Tsd, window_len: int = 11, window: str = 'hanning'):
    """
    Interpolate a Tsd onto a 100 Hz uniform grid and smooth with smooth_signal().
    Returns (t_grid, y_smoothed). If tsd is empty, returns (None, None).
    """
    if tsd is None or len(tsd) == 0:
        return None, None
    t = tsd.index.values
    d = tsd.values
    t0, t1 = float(t[0]), float(t[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return None, None
    step = 0.01  # 100 Hz grid
    # include end point if it lands exactly on the grid
    t_grid = np.arange(t0, t1 + step*0.5, step)
    y = np.interp(t_grid, t, d)
    y_sm = smooth_signal(y, window_len=window_len, window=window)
    return t_grid, y_sm

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

# ---------- ripple detector ----------
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
    ripple_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, low_freq, high_freq, Fs)
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

    rip_ep = nSS3.time_support
    rip_ep = rip_ep.drop_short_intervals(20, time_units='ms')
    rip_ep = rip_ep.drop_long_intervals(200, time_units='ms')

    rip_vals, rip_times = [], []
    for s, e in rip_ep.values:
        seg = nSS.loc[s:e]
        if len(seg) == 0:
            continue
        rip_times.append(seg.idxmax())
        rip_vals.append(seg.max())

    rip_tsd = nap.Tsd(t=np.array(rip_times), d=np.array(rip_vals))
    return ripple_band_filtered, nSS, nSS3, rip_ep, rip_tsd

# ============ main single-trial entry ============
def single_trial_ripples(
    sync_dir: str,
    landmarks: Dict,
    lfp_channel: str = "LFP_4",
    speed_thresh_cm_s: float = 3.0,
    ripple_win_s: float = 0.100,
    ripple_smooth_win_samples: int = 500,
    low_high_thresh: tuple[float, float] = (1.0, 10.0),
    ripple_band: tuple[float, float] = (130.0, 250.0),
    show: bool = True
):
    pkl = os.path.join(sync_dir, "aligned_cheeseboard.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"aligned_cheeseboard.pkl not found in {sync_dir}")

    with open(pkl, "rb") as f:
        D = pickle.load(f)
    e = D["ephys"]; p = D["phot"]; b = D["beh"]

    # --- raw series ---
    t_e = np.asarray(e["t"], float)
    lfp = np.asarray(e[lfp_channel], float)

    t_ph = np.asarray(p["t"], float)
    if p.get("green") is not None:
        zraw = np.asarray(p["green"], float); opt_label = "GEVI green"
    elif p.get("z") is not None:
        zraw = np.asarray(p["z"], float);     opt_label = "GEVI z"
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
        print("No approach_time in this recording.")
        return

    # ----- reward-zone exit strictly after approach -----
    cx, cy = landmarks["reward_pt"]
    R = float(landmarks.get("reward_zone_radius", 15.0))
    in_head = ((headX - cx) ** 2 + (headY - cy) ** 2) <= (R * R)
    if neckX is not None:
        in_neck = ((neckX - cx) ** 2 + (neckY - cy) ** 2) <= (R * R)
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

    # ----- immobility INSIDE the reward window -----
    px_to_cm = compute_px_to_cm_scale(
        landmarks["cheeseboard_center"], landmarks["cheeseboard_ends"], 100.0
    )
    ts_sp, sp = compute_speed_cm_per_s(t_b, (headX, headY), px_to_cm)
    rw_mask = (ts_sp >= approach_t) & (ts_sp <= leave_time)
    imm_mask = rw_mask & (sp < speed_thresh_cm_s) & np.isfinite(sp)

    imm_intervals = mask_to_intervals(ts_sp, imm_mask)
    ep_rw = nap.IntervalSet(start=approach_t, end=leave_time, time_units='s')

    # ----- build nap.Tsd and restrict to windows -----
    dt_e = np.nanmedian(np.diff(t_e))
    if not (np.isfinite(dt_e) and dt_e > 0):
        print("Bad LFP timebase.")
        return
    fs_e = 1.0 / dt_e

    LFP = nap.Tsd(t=t_e, d=lfp, time_units='s')
    LFP_rw = LFP.restrict(ep_rw)

    # Photometry (z-scored), then 100 Hz resample + smooth for plotting
    if (zraw is not None) and t_ph.size:
        z_mean = np.nanmean(zraw); z_std = np.nanstd(zraw, ddof=1)
        ZSTD = nap.Tsd(
            t=t_ph,
            d=((zraw - (z_mean if np.isfinite(z_mean) else 0.0)) /
               (z_std if np.isfinite(z_std) and z_std > 0 else 1.0)),
            time_units='s'
        )
        ZSTD_rw = ZSTD.restrict(ep_rw)
        tZ_rw_100, z_rw_100 = resample_and_smooth_100Hz(ZSTD_rw, window_len=3, window='hanning')
    else:
        ZSTD = ZSTD_rw = None
        tZ_rw_100 = z_rw_100 = None

    # ================= overview plot (approach → leave) =================
    nrows = 2 if ZSTD is None else 3
    figO, ax = plt.subplots(nrows, 1, figsize=(10, 5), sharex=True)
    if ZSTD is None:
        axL, axS = ax
    else:
        axL, axZ, axS = ax

    axL.plot(LFP_rw.index.values, LFP_rw.values, color='k', lw=0.6, label='LFP')
    axL.axvline(approach_t, color='r', lw=1.5, label='approach')
    axL.axvline(leave_time, color='r', lw=1.5, linestyle='--', label='leave')
    if imm_intervals:
        for s, e in imm_intervals:
            axL.axvspan(s, e, color='orange', alpha=0.15, linewidth=0)
    axL.set_ylabel("LFP"); axL.legend(loc='upper right', frameon=False)

    if ZSTD is not None and (tZ_rw_100 is not None):
        axZ.plot(tZ_rw_100, z_rw_100, color='#2ca02c', lw=0.9, label=f'{opt_label} (z, 100 Hz smoothed)')
        axZ.axvline(approach_t, color='r', lw=1.5)
        axZ.axvline(leave_time, color='r', lw=1.5, linestyle='--')
        if imm_intervals:
            for s, e in imm_intervals:
                axZ.axvspan(s, e, color='orange', alpha=0.15, linewidth=0)
        axZ.set_ylabel("GEVI z")

    ts_rw = (ts_sp >= approach_t) & (ts_sp <= leave_time)
    axS.plot(ts_sp[ts_rw], sp[ts_rw], color='0.3', lw=0.8, label='speed (cm/s)')
    axS.axhline(speed_thresh_cm_s, color='0.2', ls=':', lw=1)
    axS.axvline(approach_t, color='r', lw=1.5)
    axS.axvline(leave_time, color='r', lw=1.5, linestyle='--')
    axS.set_ylabel("Speed"); axS.set_xlabel("Time (s)")
    axS.legend(loc='upper right', frameon=False)

    figO.suptitle(
        f"Reward window only — {os.path.basename(os.path.dirname(sync_dir))}/{os.path.basename(sync_dir)}"
    )
    figO.tight_layout(rect=[0, 0, 1, 0.95])
    if show:
        plt.show()
    plt.close(figO)

    # ================= ripple detection ONLY inside immobility ∩ reward window =================
    if not imm_intervals:
        print("No immobility within [approach, leave]; skipping ripple detection.")
        return

    rb_lo, rb_hi = ripple_band
    low_th, high_th = low_high_thresh

    ripple_band_rw, nSS_rw, nSS3_rw, rip_ep_rw, rip_tsd_rw = getRippleEvents(
        LFP_rw, Fs=fs_e, windowlen=ripple_smooth_win_samples,
        Low_thres=low_th, High_thres=high_th, low_freq=rb_lo, high_freq=rb_hi
    )

    ep_imm = nap.IntervalSet(
        start=[s for s, _ in imm_intervals],
        end=[e for _, e in imm_intervals],
        time_units='s'
    )
    rip_tsd = rip_tsd_rw.restrict(ep_imm)
    rip_ep  = rip_ep_rw.intersect(ep_imm)

    if rip_tsd.values.size == 0:
        print("No ripples detected in immobile reward window.")
        return

    # Per-ripple plots (± ripple_win_s around peak)
    save_dir = os.path.join(sync_dir, f"Ripples_{lfp_channel}")
    os.makedirs(save_dir, exist_ok=True)

    half = ripple_win_s
    pad = 0.030
    n = min(len(rip_ep), len(rip_tsd))
    for i in range(n):
        t_peak = float(rip_tsd.index.values[i])

        if (t_peak - LFP.index.values[0]) < (half + pad) or (LFP.index.values[-1] - t_peak) < (half + pad):
            continue

        win_big = nap.IntervalSet(start=t_peak - (half + pad), end=t_peak + (half + pad), time_units='s')
        LFP_big = LFP.restrict(win_big)
        RB_big = pyna.eeg_processing.bandpass_filter(LFP_big, rb_lo, rb_hi, fs_e)

        win_small = nap.IntervalSet(start=t_peak - half, end=t_peak + half, time_units='s')
        LFP_win = LFP_big.restrict(win_small)
        RB_win = RB_big.restrict(win_small)
        nSS_win = nSS_rw.restrict(win_small)

        # Photometry per-ripple (z-scored) → resample to 100 Hz and smooth
        if ZSTD is not None:
            Z_win = ZSTD.restrict(win_small)
            tZ_100, z_100 = resample_and_smooth_100Hz(Z_win, window_len=3, window='hanning')
        else:
            Z_win = None
            tZ_100 = z_100 = None

        # ---------- plotting ----------
        nrows = 2 if Z_win is None else 3
        fig, axs = plt.subplots(nrows, 1, figsize=(6, 6), sharex=True)
        if Z_win is None:
            ax1, ax2 = axs
        else:
            ax1, ax2, ax3 = axs

        ax1.plot(LFP_win.index.values - t_peak, LFP_win.values, color='k', lw=1.0, label='LFP')
        ax1b = ax1.twinx()
        ax1b.plot(RB_win.index.values - t_peak, RB_win.values, color='C0', lw=1.0, label='Ripple band (130–250 Hz)')
        ax1.axvline(0, color='r', lw=1.2)
        ax1.set_ylabel("LFP")
        ax1b.set_ylabel("Ripple band (a.u.)", color='C0')
        ax1b.tick_params(axis='y', labelcolor='C0')

        ax2.plot(nSS_win.index.values - t_peak, nSS_win.values, color='0.3', lw=1.0, label='nSS (z)')
        ax2.axvline(0, color='r', lw=1.2)
        ax2.set_ylabel("nSS (z)")

        if (tZ_100 is not None) and (z_100 is not None):
            ax3.plot(tZ_100 - t_peak, z_100, color='#2ca02c', lw=1.0, label=f'{opt_label} (z, 100 Hz smoothed)')
            ax3.axvline(0, color='r', lw=1.2)
            ax3.set_ylabel("GEVI z"); ax3.set_xlabel("Time (s)")
        else:
            ax2.set_xlabel("Time (s)")

        dur_ms = ((rip_ep.iloc[[i]]['end'] - rip_ep.iloc[[i]]['start']) * 1000)[0]
        fig.suptitle(f"Ripple {i+1} — peak @ {t_peak:.3f}s (dur {dur_ms:.0f} ms)")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fig.savefig(os.path.join(save_dir, f"Ripple_{i+1:03d}.png"), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    print(f"Done. Ripples saved in: {save_dir}")

# ===================== quick run =====================
if __name__ == "__main__":
    sync_dir = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success\Day2\SyncRecording2"
    landmarks = {
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 15.0,
    }
    single_trial_ripples(sync_dir, landmarks,
                         lfp_channel="LFP_4",
                         speed_thresh_cm_s=3.0,
                         ripple_win_s=0.100,
                         ripple_smooth_win_samples=1000,
                         low_high_thresh=(1.0, 10.0),
                         ripple_band=(130.0, 250.0),
                         show=True)
