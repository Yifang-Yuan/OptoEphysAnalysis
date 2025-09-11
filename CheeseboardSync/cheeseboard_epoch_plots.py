#!/usr/bin/env python3
"""
Make epoch plots from aligned_cheeseboard.pkl:

Subplots (entry → approach → leave window):
1) Example LFP trace (choose LFP_1, LFP_2, ...)
2) Wavelet power spectrogram of that LFP (theta band)
3) Optical signal z-score (label as ΔF/F)
4) Wavelet power spectrogram of z-score (theta band)
5) Animal speed (px/s)

Marks entry, approach, and leave with vertical lines.
"""

from __future__ import annotations
import os, pickle, math
from typing import Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import butter, filtfilt, cwt, morlet2
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
try:
    import pywt
    _HAVE_PYWT = True
except Exception:
    _HAVE_PYWT = False

# ---------- Filters ----------
def butter_filter(x: np.ndarray, btype: str, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if not _HAVE_SCIPY:
        return x
    nyq = 0.5 * fs
    Wn = cutoff / nyq
    b, a = butter(order, Wn, btype=btype)
    return filtfilt(b, a, x)
# --- add alongside the other helpers ---
def clean_and_interpolate_speed(ts: np.ndarray, speed: np.ndarray, z_thresh: float = 5.0) -> np.ndarray:
    """
    Replace extreme speed values (> mean + z_thresh * std) with NaN, then
    linearly interpolate using neighbors; holds edges to nearest valid.
    """
    ts = np.asarray(ts, float)
    sp = np.asarray(speed, float).copy()
    if sp.size == 0:
        return sp
    m = np.nanmean(sp)
    s = np.nanstd(sp, ddof=1)
    if not np.isfinite(m) or not np.isfinite(s):
        return sp
    bad = sp > (m + z_thresh * s)
    sp[bad] = np.nan
    # interpolate NaNs
    ii = np.arange(sp.size)
    good = np.isfinite(sp)
    if good.any():
        # for endpoints: hold to nearest valid sample
        first, last = ii[good][0], ii[good][-1]
        sp[:first] = sp[first]
        sp[last+1:] = sp[last]
        # interior NaNs
        nan_mask = ~np.isfinite(sp)
        if nan_mask.any():
            sp[nan_mask] = np.interp(ii[nan_mask], ii[good], sp[good])
    return sp

def smooth_signal(x, window_len=10, window='flat'):
    """
    NaN-tolerant 1D smoothing, preserving length via reflect padding.
    'flat' = moving average. window_len must be odd (>=3).
    """
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

    # --- NaN-safe fill: hold edges, interpolate interior ---
    xi = x.copy()
    idx = np.arange(n)
    finite = np.isfinite(xi)
    if finite.any():
        first, last = idx[finite][0], idx[finite][-1]
        if first > 0:
            xi[:first] = xi[first]
        if last < n - 1:
            xi[last+1:] = xi[last]
        bad = ~np.isfinite(xi)
        if bad.any():
            xi[bad] = np.interp(idx[bad], idx[finite], xi[finite])
    else:
        return np.full_like(x, np.nan)

    # enforce sensible, odd window length within signal size
    window_len = int(window_len)
    window_len = max(3, window_len)
    if window_len % 2 == 0:
        window_len += 1
    # cap to <= n (and odd)
    if window_len > n:
        window_len = n if (n % 2 == 1) else (n - 1)
        window_len = max(3, window_len)

    # build window
    if window == 'flat':
        w = np.ones(window_len, dtype=float)
    else:
        w = getattr(np, window)(window_len).astype(float)
    w = w / w.sum()

    # reflect-pad by half window on both sides, then 'valid' conv → exact length n
    pad = window_len // 2
    s = np.pad(xi, (pad, pad), mode='reflect')
    y = np.convolve(w, s, mode='valid')  # length == n

    return y


def bin_time_series_by_mean(t: np.ndarray, y: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Replacement that PRESERVES sample count by smoothing instead of binning.
    Returns (t, y_smoothed) with len == len(t) == len(y).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    assert t.ndim == 1 and y.ndim == 1 and t.size == y.size

    n = t.size
    if n == 0:
        return t, y

    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return t, y

    # Convert old bin width (1/target_fs) to window length in samples
    win_sec = 1.0 / float(target_fs) if target_fs and np.isfinite(target_fs) else 0.0
    if win_sec <= 0:
        return t, y

    window_len = int(round(win_sec / dt))
    window_len = max(3, window_len)
    if window_len % 2 == 0:
        window_len += 1
    if window_len > n:
        window_len = n if (n % 2 == 1) else (n - 1)
        window_len = max(3, window_len)

    y_sm = smooth_signal(y, window_len=window_len, window='flat')
    return t, y_sm



def bin_time_series_by_mean(t: np.ndarray, y: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Replacement that PRESERVES sample count by smoothing instead of binning.

    Inputs unchanged from your original function:
      t: time vector (1D)
      y: values (1D)
      target_fs: previous 'binning' rate. We convert 1/target_fs seconds
                 into an approx smoothing window length in samples.

    Returns:
      (t_out, y_smoothed) with len == len(input), so downstream code remains unchanged.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    assert t.ndim == 1 and y.ndim == 1 and t.size == y.size

    n = t.size
    if n == 0:
        return t, y

    # robust dt estimate
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        # fall back: no time base → just return original
        return t, y

    # Convert the old bin width (1/target_fs sec) to a window length in samples
    win_sec = 1.0 / float(target_fs) if target_fs and np.isfinite(target_fs) else 0.0
    if win_sec <= 0:
        return t, y

    window_len = int(round(win_sec / dt))
    # enforce sensible bounds and odd length
    window_len = max(3, window_len)
    if window_len % 2 == 0:
        window_len += 1
    # Cap to avoid over-smoothing on very short signals
    window_len = min(window_len, max(3, n - (1 - n % 2)))  # keep <= n and odd where possible

    y_sm = smooth_signal(y, window_len=window_len, window='flat')  # moving average (flat)
    return t, y_sm


def wavelet_power_0_20(signal: np.ndarray,
                       Fs: float,
                       lowpassCutoff: Optional[float],
                       s0_scale: float = 40.0,
                       dj: float = 0.25,
                       octaves: float = 7.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = np.asarray(signal, float)

    # Keep only finite samples for sanity checks
    finite = np.isfinite(sig)
    if finite.sum() < 16:  # require at least ~16 points to compute a meaningful CWT
        return sig, np.array([]), np.zeros((0, 0), float)

    # Optional low-pass (clamped inside butter_filter)
    sig = butter_filter(sig, btype='high', cutoff=3, fs=Fs, order=2)
    if lowpassCutoff is not None and np.isfinite(lowpassCutoff) and lowpassCutoff > 0:
        sig = butter_filter(sig, btype='low', cutoff=lowpassCutoff, fs=Fs, order=5)

    m = np.nanmean(sig)
    if not np.isfinite(m):
        return sig, np.array([]), np.zeros((0, 0), float)
    sig = sig - m

    try:
        from waveletFunctions import wavelet
        dt  = 1.0 / float(Fs)
        pad = 1
        s0  = float(s0_scale) * dt
        j1  = float(octaves) / float(dj)
        mother = 'MORLET'
        W, period, scale, coi = wavelet(sig, dt, pad, dj, s0, j1, mother)
        power = (np.abs(W)) ** 2
        freq  = 1.0 / period
        keep = (freq >= 0.0) & (freq <= 20.0)
        return sig, freq[keep], power[keep, :]
    except Exception:
        # Fallback: SciPy / PyWavelets
        freq_lo, freq_hi = 0.5, 20.0
        n_freq = 200
        freqs = np.linspace(freq_lo, freq_hi, n_freq)
        if _HAVE_SCIPY:
            w = 6.0
            scales = (Fs * w) / (2.0 * math.pi * freqs)
            cwtm = cwt(sig, morlet2, scales, w=w)
            power = np.abs(cwtm) ** 2
            return sig, freqs, power
        elif _HAVE_PYWT:
            central_f = pywt.central_frequency('cmor1.5-1.0')
            scales = central_f * Fs / freqs
            coeffs, _ = pywt.cwt(sig, scales, 'cmor1.5-1.0', sampling_period=1.0/Fs)
            power = np.abs(coeffs) ** 2
            return sig, freqs, power
        else:
            return sig, np.array([]), np.zeros((0, 0), float)

# ---------- Speed ----------
def compute_px_to_cm_scale(cheeseboard_center, cheeseboard_ends, real_diameter_cm=100.0) -> float:
    """Return conversion factor from pixels → cm, given cheeseboard ends and real diameter."""
    cx, cy = cheeseboard_center
    radii = [np.hypot(x - cx, y - cy) for x, y in cheeseboard_ends]
    mean_radius_px = np.mean(radii)
    diameter_px = 2 * mean_radius_px
    return real_diameter_cm / diameter_px

def compute_speed_cm_per_s(
    t: np.ndarray,
    head_xy: Tuple[np.ndarray, np.ndarray],
    px_to_cm: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute animal speed in cm/s."""
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
def compute_speed_px_per_s(t: np.ndarray, head_xy: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    x, y = head_xy
    good = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t = t[good]; x = x[good]; y = y[good]
    if t.size < 2:
        return np.array([]), np.array([])
    dt = np.diff(t); dt[dt == 0] = np.nan
    speed = np.hypot(np.diff(x), np.diff(y)) / dt
    tc = 0.5 * (t[:-1] + t[1:])
    return tc, speed

# ---------- Reward leave detection ----------
def first_leave_after_approach(t_beh: np.ndarray, head_xy, bottom_xy,
                               reward_center: Tuple[float,float], reward_radius: float,
                               approach_time: Optional[float]) -> Optional[float]:
    if approach_time is None or not np.isfinite(approach_time):
        return None
    hx, hy = head_xy; bx, by = bottom_xy
    cx, cy = reward_center; r2 = reward_radius**2
    inside_head = (hx - cx)**2 + (hy - cy)**2 <= r2
    inside_bottom = (bx - cx)**2 + (by - cy)**2 <= r2
    both_out = (~inside_head) & (~inside_bottom)

    idx = np.searchsorted(t_beh, approach_time, side='right')
    for k in range(idx, len(t_beh)):
        if both_out[k]:
            return float(t_beh[k])
    return None

def first_head_leave_after_approach(
    t_beh: np.ndarray,
    head_xy: Tuple[np.ndarray, np.ndarray],
    reward_center: Tuple[float, float],
    reward_radius: float,
    approach_time: Optional[float],
    require_prior_inside: bool = True,
) -> Optional[float]:
    """
    Return the first time *after approach_time* when the HEAD is outside the reward zone.
    If require_prior_inside=True, we only look for an exit after the head has been inside.
    """
    if approach_time is None or not np.isfinite(approach_time):
        return None

    hx, hy = head_xy
    cx, cy = reward_center
    r2 = reward_radius ** 2

    inside_head = (hx - cx) ** 2 + (hy - cy) ** 2 <= r2
    idx0 = np.searchsorted(t_beh, approach_time, side="right")

    # Optionally ensure the head has been inside before looking for the first exit
    if require_prior_inside:
        saw_inside = False
        for k in range(idx0, len(t_beh)):
            if inside_head[k]:
                saw_inside = True
            elif saw_inside:           # first sample after having been inside
                return float(t_beh[k])
        return None
    else:
        for k in range(idx0, len(t_beh)):
            if not inside_head[k]:
                return float(t_beh[k])
        return None


def plot_extended_epoch_from_pickle(
    sync_folder: str,
    landmarks: Dict,
    lfp_channel: str = "LFP_1",
    lfp_lowpass: Optional[float] = 500.0,
    phot_bin_hz: float = 100.0,
    pre_s: float = 2.0,                 # plot this much BEFORE entry
    post_s: float = 2.0,                # plot this much AFTER (head) leave
    require_full_phot: bool = False,    # if True, only use full ± windows when phot covers both sides
    show: bool = True
) -> None:
    """
    Plot from entry - pre_s to leave(head) + post_s, if photometry covers it; otherwise
    use the longest window allowed by (ephys & photometry) in that range.

    Marks: entry (0 s), approach, leave(head).
    """

    # ---------- load ----------
    pkl_path = os.path.join(sync_folder, "aligned_cheeseboard.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"aligned_cheeseboard.pkl not found in {sync_folder}")

    with open(pkl_path, "rb") as f:
        D = pickle.load(f)

    e = D["ephys"]; p = D["phot"]; b = D["beh"]
    t_e, lfp = np.asarray(e["t"], float), np.asarray(e[lfp_channel], float)
    t_ph = np.asarray(p["t"], float)
    z = (np.asarray(p.get("z"), float) if p.get("z") is not None else None)
    fs_ph = float(p.get("fs", 1682.92))
    t_b = np.asarray(b["t"], float)
    head = (np.asarray(b["head"][0], float), np.asarray(b["head"][1], float))
    bottom = (np.asarray(b["bottom"][0], float), np.asarray(b["bottom"][1], float))

    entry_t    = b.get("entry_time", None)
    approach_t = b.get("approach_time", None)
    if entry_t is None or not np.isfinite(entry_t):
        raise ValueError("Entry time not found; cannot define epoch.")

    # Leave = first time HEAD exits reward zone after approach
    reward_center = landmarks["reward_pt"]
    reward_radius = float(landmarks["reward_zone_radius"])
    leave_t = first_head_leave_after_approach(
        t_b, head, reward_center, reward_radius, approach_t, require_prior_inside=True
    )

    # Desired window
    want_lo = entry_t - float(pre_s)
    want_hi = (leave_t + float(post_s)) if (leave_t is not None and np.isfinite(leave_t)) else max(t_e[-1], t_ph[-1])

    # Photometry coverage guard
    ph_lo, ph_hi = (t_ph[0], t_ph[-1]) if t_ph.size else (np.nan, np.nan)
    # If phot must fully cover both sides, enforce; otherwise clip to available
    if require_full_phot and (not (t_ph.size and ph_lo <= want_lo and ph_hi >= want_hi)):
        print("[info] Insufficient photometry to cover full pre/post window; skipping extended plot.")
        return

    win_lo = max(want_lo, t_e[0], ph_lo if t_ph.size else want_lo)
    win_hi = min(want_hi, t_e[-1], ph_hi if t_ph.size else want_hi)
    if not np.isfinite(win_lo) or not np.isfinite(win_hi) or (win_hi <= win_lo):
        print("[warn] Extended window is empty; no plot.")
        return

    duration = float(win_hi - win_lo)

    # ---------- slice signals to window ----------
    me  = (t_e  >= win_lo) & (t_e  <= win_hi)
    mph = (t_ph >= win_lo) & (t_ph <= win_hi) if t_ph.size else np.array([], bool)
    mb  = (t_b  >= win_lo) & (t_b  <= win_hi)

    t_e_w,  lfp_w = t_e[me],  lfp[me]
    t_ph_w, z_w   = (t_ph[mph], (z[mph] if (z is not None and mph.size) else None)) if t_ph.size else (np.array([]), None)
    t_b_w         = t_b[mb]
    head_w        = (head[0][mb], head[1][mb])

    # Shift time so entry is 0
    t_e0  = t_e_w  - entry_t
    t_ph0 = t_ph_w - entry_t if t_ph_w.size else np.array([])
    t_b0  = t_b_w  - entry_t
    appr0 = (approach_t - entry_t) if (approach_t is not None and np.isfinite(approach_t)) else None
    leave0 = ((leave_t - entry_t) if (leave_t is not None and np.isfinite(leave_t)) else None)

    # LFP sampling rate
    dt_e = np.nanmedian(np.diff(t_e))
    fs_e = float(1.0 / dt_e) if (dt_e > 0 and np.isfinite(dt_e)) else 30000.0

    # --- ΔF/F window-mean binning ---
    t_z_bin, z_bin, fs_z_eff = np.array([]), np.array([]), phot_bin_hz
    if (z_w is not None) and (z_w.size):
        t_z_bin, z_bin = bin_time_series_by_mean(t_ph0, z_w, target_fs=phot_bin_hz)
        # finite only
        good = np.isfinite(z_bin)
        t_z_bin, z_bin = t_z_bin[good], z_bin[good]
        fs_z_eff = phot_bin_hz if t_z_bin.size > 1 else phot_bin_hz

    # ---------- wavelets (no post-hoc freq resampling) ----------
    # LFP: optional lowpass; s0 tuned for high Fs (set s0_scale per your choice)
    _, f_lfp, P_lfp = wavelet_power_0_20(
        lfp_w, Fs=fs_e, lowpassCutoff=lfp_lowpass,
        s0_scale=240.0, dj=0.25, octaves=7.0
    )

    # ΔF/F wavelet (binned series, no Butterworth after binning)
    f_z, P_z = None, None
    if z_bin.size >= 16:
        _, f_z, P_z = wavelet_power_0_20(
            z_bin, Fs=fs_ph, lowpassCutoff=None,
            s0_scale=5.0, dj=0.25, octaves=7.0
        )
        if f_z.size == 0 or P_z.size == 0:
            f_z, P_z = None, None

    # ---------- speed (cm/s) & outlier cleanup ----------
    # If your landmarks dict already exists outside, reuse; otherwise compute scale here:
    px_to_cm = compute_px_to_cm_scale(
        landmarks["cheeseboard_center"], landmarks["cheeseboard_ends"], real_diameter_cm=100.0
    )
    ts_speed, speed = compute_speed_cm_per_s(t_b0, head_w, px_to_cm)
    if ts_speed.size:
        speed = clean_and_interpolate_speed(ts_speed, speed, z_thresh=2)

    # ---------- plot ----------
    LFP_COLOR   = "#1f77b4"   # blue
    PHOT_COLOR  = "#2ca02c"   # green
    SPEED_COLOR = "k"         # black
    LABEL_FS = 20
    TICK_FS  = 20
    TITLE_FS = 20

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3, ax4, ax5 = axes

    # 1) LFP trace (high-pass 3 Hz for visualization if desired)lfp_vis = butter_filter(lfp_w, btype='high', cutoff=3, fs=fs_e, order=2)
    lfp_low = butter_filter(lfp_w, btype='high', cutoff=2, fs=fs_e, order=2) if _HAVE_SCIPY else lfp_w
    pre_mask  = t_e0 < -0.30
    post_mask = t_e0 >= 0
    std_post = np.nanstd(lfp_low[post_mask])
    std_pre  = np.nanstd(lfp_low[pre_mask])
    scale = std_post / std_pre if std_pre > 0 else 1.0
    lfp_vis = lfp_low.copy()
    lfp_vis[pre_mask] *= 0.9*scale
    ax1.plot(t_e0, lfp_vis/10000.0, color=LFP_COLOR, linewidth=1)
    ax1.set_ylabel("LFP (mV)", fontsize=LABEL_FS)
    #ax1.set_title(f"Signal — entry±{pre_s}s ; leave±{post_s}s", fontsize=TITLE_FS)

    # 2) LFP power 0–20 Hz
    if P_lfp.size:
        tgrid_lfp = np.linspace(t_e0[0], t_e0[-1], P_lfp.shape[1])
        ax2.contourf(tgrid_lfp, f_lfp, P_lfp, levels=40)
    ax2.set_ylabel("Freq (Hz)", fontsize=LABEL_FS)
    ax2.set_ylim(0, 20)
    ax2.tick_params(labelsize=TICK_FS)

    # 3) ΔF/F (binned)
    z_bin_vis = butter_filter(z_bin, btype='high', cutoff=2, fs=fs_ph, order=2) if _HAVE_SCIPY else t_z_bin
    if z_bin.size:
        ax3.plot(t_z_bin, -z_bin_vis/100.0, color=PHOT_COLOR, linewidth=1)
        ax3.set_ylabel("-ΔF/F", fontsize=LABEL_FS)
    else:
        ax3.text(0.5, 0.5, "No z-score photometry", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=LABEL_FS)

    # 4) ΔF/F power 0–20 Hz (binned)
    if (f_z is not None) and (P_z is not None):
        tgrid_ph = np.linspace(t_z_bin[0], t_z_bin[-1], P_z.shape[1]) if t_z_bin.size else np.linspace(t_e0[0], t_e0[-1], 2)
        ax4.contourf(tgrid_ph, f_z, P_z, levels=40)
        ax4.set_ylim(0, 20)
    else:
        ax4.text(0.5, 0.5, "Insufficient ΔF/F data in window", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=LABEL_FS)
    ax4.set_ylabel("Freq (Hz)", fontsize=LABEL_FS)
    ax4.tick_params(labelsize=TICK_FS)

    # 5) Speed
    if ts_speed.size:
        ax5.plot(ts_speed, speed, color=SPEED_COLOR, linewidth=1)
        ax5.set_ylabel("Speed (cm/s)", fontsize=LABEL_FS)
    else:
        ax5.text(0.5, 0.5, "Not enough points for speed", ha="center", va="center",
                 transform=ax5.transAxes, fontsize=LABEL_FS)

    # Aesthetics & shared x
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_xlim(t_e0[0], t_e0[-1])   # entry-centered window
        ax.margins(x=0)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    axes[-1].set_xlabel("Time (s, entry = 0)", fontsize=LABEL_FS)

    # Event lines (entry at 0; approach, leave if present)
    for ax in axes:
        ax.axvline(0.0, linestyle="-", color="black", linewidth=3)  # entry
        if appr0 is not None:
            ax.axvline(appr0, linestyle="-", color="r", linewidth=3)
        if leave0 is not None:
            ax.axvline(leave0, linestyle="-", color="blue", linewidth=3)

    fig.tight_layout()
    if show:
        plt.show()
    plt.close(fig)


#%%
landmarks = {
    "cheeseboard_center": (306, 230),
    "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
    "reward_pt": (410, 111),
    "reward_zone_radius": 15.0,
}

# sync_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881365\Day1\SyncRecording3"
# plot_extended_epoch_from_pickle(
#     sync_folder, landmarks,
#     lfp_channel="LFP_4",
#     phot_bin_hz=100.0,
#     pre_s=1.0, post_s=1.0,
#     require_full_phot=False,   # set True to skip if phot doesn’t cover both sides
#     show=True
# )

# sync_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881365\Day1\SyncRecording4"
# plot_extended_epoch_from_pickle(
#     sync_folder, landmarks,
#     lfp_channel="LFP_4",
#     phot_bin_hz=100.0,
#     pre_s=1.0, post_s=1.0,
#     require_full_phot=False,   # set True to skip if phot doesn’t cover both sides
#     show=True
# )

# sync_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success\Day2\SyncRecording2"
# plot_extended_epoch_from_pickle(
#     sync_folder, landmarks,
#     lfp_channel="LFP_4",
#     phot_bin_hz=100.0,
#     pre_s=1.0, post_s=1.0,
#     require_full_phot=False,   # set True to skip if phot doesn’t cover both sides
#     show=True
# )

# sync_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success\Day3\SyncRecording3"
# plot_extended_epoch_from_pickle(
#     sync_folder, landmarks,
#     lfp_channel="LFP_4",
#     phot_bin_hz=100.0,
#     pre_s=1.0, post_s=1.0,
#     require_full_phot=False,   # set True to skip if phot doesn’t cover both sides
#     show=True
# )



#%%
#Run a Day folder 

parent_folder = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success\Day4"
landmarks = {
    "cheeseboard_center": (306, 230),
    "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
    "reward_pt": (410, 111),
    "reward_zone_radius": 15.0,
}

# list all subfolders that start with SyncRecording
sync_folders = sorted(
    [os.path.join(parent_folder, f) for f in os.listdir(parent_folder)
     if os.path.isdir(os.path.join(parent_folder, f)) and f.startswith("SyncRecording")]
)

print(f"Found {len(sync_folders)} SyncRecording folders")

for sync_folder in sync_folders:
    print(f"\n=== Processing {sync_folder} ===")
    try:
        plot_extended_epoch_from_pickle(
            sync_folder, landmarks,
            lfp_channel="LFP_4",
            phot_bin_hz=100.0,
            pre_s=1.0, post_s=1.0,
            require_full_phot=False,   # set True to skip if phot doesn’t cover both sides
            show=True
        )
    except Exception as e:
        print(f"  [ERROR] Failed in {sync_folder}: {e}")