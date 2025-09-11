
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pickle, math, glob
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

# --- optional SciPy (faster CWT). If missing, spectrograms return zeros but code still runs. ---
try:
    from scipy.signal import butter, filtfilt, cwt, morlet2
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ========== small utilities ==========

def butter_filter(x: np.ndarray, btype: str, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    if not _HAVE_SCIPY or x.size == 0 or cutoff is None or not np.isfinite(cutoff):
        return x
    nyq = 0.5 * fs
    Wn = float(cutoff) / nyq
    Wn = np.clip(Wn, 1e-6, 0.999999)
    b, a = butter(order, Wn, btype=btype)
    return filtfilt(b, a, x)

def clean_and_interpolate_speed(ts: np.ndarray, speed: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    ts = np.asarray(ts, float)
    sp = np.asarray(speed, float).copy()
    if sp.size == 0:
        return sp
    m = np.nanmean(sp); s = np.nanstd(sp, ddof=1)
    if not np.isfinite(m) or not np.isfinite(s):
        return sp
    bad = sp > (m + z_thresh * s)
    sp[bad] = np.nan
    ii = np.arange(sp.size)
    good = np.isfinite(sp)
    if good.any():
        first, last = ii[good][0], ii[good][-1]
        sp[:first] = sp[first]
        sp[last+1:] = sp[last]
        nan_mask = ~np.isfinite(sp)
        if nan_mask.any():
            sp[nan_mask] = np.interp(ii[nan_mask], ii[good], sp[good])
    return sp

def compute_px_to_cm_scale(cheeseboard_center, cheeseboard_ends, real_diameter_cm=100.0) -> float:
    cx, cy = cheeseboard_center
    radii = [np.hypot(x - cx, y - cy) for x, y in cheeseboard_ends]
    mean_radius_px = np.mean(radii)
    diameter_px = 2 * mean_radius_px
    return real_diameter_cm / diameter_px if diameter_px > 0 else 1.0

def compute_speed_cm_per_s(t: np.ndarray, head_xy: Tuple[np.ndarray, np.ndarray], px_to_cm: float) -> Tuple[np.ndarray, np.ndarray]:
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

# ---- smoothing used to preserve sample count for plotting traces (unchanged behaviour) ----
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
    window_len = int(max(3, window_len)); 
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

def bin_time_series_by_mean(t: np.ndarray, y: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    # replaced by smoothing that preserves length & alignment
    t = np.asarray(t, float); y = np.asarray(y, float)
    assert t.ndim == y.ndim == 1 and t.size == y.size
    if t.size == 0: return t, y
    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0: return t, y
    win_sec = 1.0 / float(target_fs) if target_fs and np.isfinite(target_fs) else 0.0
    if win_sec <= 0: return t, y
    window_len = int(round(win_sec / dt))
    window_len = max(3, window_len)
    if window_len % 2 == 0: window_len += 1
    if window_len > t.size:
        window_len = t.size if (t.size % 2 == 1) else (t.size - 1)
        window_len = max(3, window_len)
    y_sm = smooth_signal(y, window_len=window_len, window='flat')
    return t, y_sm

def morlet_power_fixed_grid(x: np.ndarray, fs: float,
                            fmin: float = 0.5, fmax: float = 20.0, n_freq: int = 120,
                            w: float = 6.0,
                            pre_highpass_hz: Optional[float] = None,
                            pre_lowpass_hz: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = np.asarray(x, float)
    if pre_highpass_hz is not None:
        sig = butter_filter(sig, 'high', pre_highpass_hz, fs, order=2)
    if pre_lowpass_hz is not None:
        sig = butter_filter(sig, 'low',  pre_lowpass_hz, fs, order=4)
    m = np.nanmean(sig); sig = sig - (m if np.isfinite(m) else 0.0)
    freqs = np.linspace(fmin, fmax, n_freq)
    if not _HAVE_SCIPY:
        return sig, freqs, np.zeros((n_freq, sig.size), float)
    scales = (fs * w) / (2.0 * math.pi * freqs)
    coeffs = cwt(sig, morlet2, scales, w=w)
    power = np.abs(coeffs) ** 2
    return sig, freqs, power

# ========== trial extraction around APPROACH (t=0) ==========
def extract_trial_around_approach(pkl_path: str,
                                  landmarks: Dict,
                                  lfp_channel: str,
                                  window_s: Tuple[float, float] = (-2.0, 2.0),
                                  phot_bin_hz: float = 100.0,
                                  lfp_lowpass: Optional[float] = 500.0,
                                  freq_grid: np.ndarray = np.linspace(0.5, 20.0, 120),
                                  time_grid_100Hz: np.ndarray = np.linspace(-2.0, 2.0, 401),
                                  speed_grid_100Hz: np.ndarray = np.linspace(-2.0, 2.0, 401)):
    with open(pkl_path, "rb") as f:
        D = pickle.load(f)
    e = D["ephys"]; p = D["phot"]; b = D["beh"]

    t_e = np.asarray(e["t"], float)
    lfp = np.asarray(e[lfp_channel], float)
    t_ph = np.asarray(p["t"], float)
    z = np.asarray(p.get("z"), float) if p.get("z") is not None else None
    fs_ph = float(p.get("fs", 1682.92))
    t_b = np.asarray(b["t"], float)
    head = (np.asarray(b["head"][0], float), np.asarray(b["head"][1], float))

    approach_t = b.get("approach_time", None)
    if approach_t is None or not np.isfinite(approach_t):
        return None

    t0 = approach_t + window_s[0]
    t1 = approach_t + window_s[1]
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return None

    me = (t_e >= t0) & (t_e <= t1)
    if not np.any(me):
        return None
    t_e_w = t_e[me] - approach_t
    lfp_w = lfp[me]

    dt_e = np.nanmedian(np.diff(t_e))
    fs_e = float(1.0 / dt_e) if (dt_e > 0 and np.isfinite(dt_e)) else 30000.0

    _, f_lfp, P_lfp = morlet_power_fixed_grid(lfp_w, fs=fs_e,
                                              fmin=freq_grid[0], fmax=freq_grid[-1],
                                              n_freq=len(freq_grid),
                                              pre_highpass_hz=4.0,
                                              pre_lowpass_hz=lfp_lowpass)
    P_lfp_t = np.empty((P_lfp.shape[0], time_grid_100Hz.size), float)
    for i in range(P_lfp.shape[0]):
        P_lfp_t[i, :] = np.interp(time_grid_100Hz, t_e_w, P_lfp[i, :], left=np.nan, right=np.nan)

    # ΔF/F trace for plotting (smooth/bin to ~100 Hz) + per-trial z-score for averaging
    z_bin = np.full_like(time_grid_100Hz, np.nan)
    P_z_t = None
    if z is not None and t_ph.size:
        mph = (t_ph >= t0) & (t_ph <= t1)
        if np.any(mph):
            t_ph_w = t_ph[mph] - approach_t
            z_w = z[mph]
            z_bin_t, z_bin_vals = bin_time_series_by_mean(t_ph_w, z_w, target_fs=phot_bin_hz)
            m = np.nanmean(z_bin_vals); s = np.nanstd(z_bin_vals, ddof=1); s = 1.0 if (not np.isfinite(s) or s==0) else s
            z_bin_vals = (z_bin_vals - (m if np.isfinite(m) else 0.0)) / s
            z_bin = np.interp(time_grid_100Hz, z_bin_t, z_bin_vals, left=np.nan, right=np.nan)

            # GEVI spectrogram for fig1 (OK on ~100 Hz series up to 20 Hz)
            _, f_z, P_z = morlet_power_fixed_grid(z_bin_vals, fs=fs_ph,
                                                  fmin=freq_grid[0], fmax=freq_grid[-1],
                                                  n_freq=len(freq_grid),
                                                  pre_highpass_hz=4, pre_lowpass_hz=None)
            P_z_t = np.empty((P_z.shape[0], time_grid_100Hz.size), float)
            for i in range(P_z.shape[0]):
                P_z_t[i, :] = np.interp(time_grid_100Hz, z_bin_t, P_z[i, :], left=np.nan, right=np.nan)

    # speed (cm/s) on 100 Hz grid
    mb = (t_b >= t0) & (t_b <= t1)
    speed_interp = np.full_like(time_grid_100Hz, np.nan)
    if np.any(mb):
        t_b_w = t_b[mb] - approach_t
        head_w = (head[0][mb], head[1][mb])
        px_to_cm = compute_px_to_cm_scale(landmarks["cheeseboard_center"],
                                          landmarks["cheeseboard_ends"], 100.0)
        ts_speed, speed = compute_speed_cm_per_s(t_b_w, head_w, px_to_cm)
        if ts_speed.size:
            speed = clean_and_interpolate_speed(ts_speed, speed, z_thresh=2.5)
            speed_interp = np.interp(time_grid_100Hz, ts_speed, speed, left=np.nan, right=np.nan)

    return dict(
        lfp_spec=P_lfp_t,
        z_bin=z_bin,           # z-scored trace for averaging
        z_spec=P_z_t,
        speed=speed_interp
    )

# ========== batch runner & plotter (AVERAGING happens here) ==========
def average_across_trials(animal_root: str,
                          landmarks: Dict,
                          lfp_channel: str = "LFP_4",
                          days: List[str] = ["Day1","Day2","Day3","Day4"],
                          window_s: Tuple[float, float] = (-2.0, 2.0),
                          show: bool = True):
    freq_grid = np.linspace(0.5, 20.0, 120)
    time_grid = np.linspace(window_s[0], window_s[1], int((window_s[1]-window_s[0])*100)+1)  # 100 Hz grid
    nT = time_grid.size

    # accumulators
    lfp_sum = None; lfp_cnt = None                  # spectrogram sum/count
    zspec_sum = None; zspec_cnt = None              # spectrogram sum/count

    # traces (for CIs)
    lfp_tr_sum = np.zeros(nT, float); lfp_tr_sumsq = np.zeros(nT, float); lfp_tr_cnt = np.zeros(nT, float)  # NEW: LFP trace
    z_sum      = np.zeros(nT, float); z_sumsq      = np.zeros(nT, float); z_cnt      = np.zeros(nT, float)
    sp_sum     = np.zeros(nT, float); sp_sumsq     = np.zeros(nT, float); sp_cnt     = np.zeros(nT, float)
    n_trials = 0

    def _accum_spec(sum_, cnt_, M):
        if M is None: return sum_, cnt_
        if sum_ is None:
            sum_ = np.zeros_like(M, float)
            cnt_ = np.zeros_like(M, float)
        good = np.isfinite(M)
        sum_[good] += M[good]
        cnt_[good] += 1
        return sum_, cnt_

    for day in days:
        day_dir = os.path.join(animal_root, day)
        if not os.path.isdir(day_dir):
            continue
        for sync in sorted(p for p in glob.glob(os.path.join(day_dir, "SyncRecording*")) if os.path.isdir(p)):
            pkl = os.path.join(sync, "aligned_cheeseboard.pkl")
            if not os.path.isfile(pkl):
                continue

            T = extract_trial_around_approach(
                pkl, landmarks, lfp_channel,
                window_s=window_s,
                phot_bin_hz=100.0,
                lfp_lowpass=500.0,
                freq_grid=freq_grid,
                time_grid_100Hz=time_grid,
                speed_grid_100Hz=time_grid
            )
            if T is None:
                continue
            n_trials += 1

            # spectrogram accumulators
            lfp_sum,   lfp_cnt   = _accum_spec(lfp_sum,   lfp_cnt,   T.get("lfp_spec"))
            zspec_sum, zspec_cnt = _accum_spec(zspec_sum, zspec_cnt, T.get("z_spec"))

            # optical & speed traces
            z_bin = T.get("z_bin", None)
            sp    = T.get("speed", None)
            if z_bin is not None and z_bin.size == nT:
                good = np.isfinite(z_bin)
                z_sum[good]   += z_bin[good]
                z_sumsq[good] += (z_bin[good] ** 2)
                z_cnt[good]   += 1
            if sp is not None and sp.size == nT:
                good = np.isfinite(sp)
                sp_sum[good]   += sp[good]
                sp_sumsq[good] += (sp[good] ** 2)
                sp_cnt[good]   += 1

            # --- NEW: LFP trace at 100 Hz (use provided key if present; else derive once here) ---
            lfp_100 = T.get("lfp_100Hz", None)
            if (lfp_100 is None) or (lfp_100.size != nT):
                # derive from the same pickle (no param changes)
                with open(pkl, "rb") as f:
                    Dloc = pickle.load(f)
                e = Dloc["ephys"]; b = Dloc["beh"]
                t_e = np.asarray(e["t"], float)
                lfp = np.asarray(e[lfp_channel], float)
                appr = b.get("approach_time", None)
                if appr is None or not np.isfinite(appr):
                    lfp_100 = None
                else:
                    t0 = appr + window_s[0]; t1 = appr + window_s[1]
                    m = (t_e >= t0) & (t_e <= t1)
                    if m.any():
                        t_e_w = t_e[m] - appr
                        lfp_w = lfp[m]
                        lfp_100 = np.interp(time_grid, t_e_w, lfp_w, left=np.nan, right=np.nan)

            if lfp_100 is not None and lfp_100.size == nT:
                good = np.isfinite(lfp_100)
                lfp_tr_sum[good]   += lfp_100[good]
                lfp_tr_sumsq[good] += (lfp_100[good] ** 2)
                lfp_tr_cnt[good]   += 1

    if n_trials == 0:
        print("No usable trials found.")
        return

    def _mean(sum_, cnt_):
        if sum_ is None: return None
        return np.divide(sum_, cnt_, out=np.full_like(sum_, np.nan), where=cnt_ > 0)

    lfp_mean_spec   = _mean(lfp_sum,   lfp_cnt)
    zspec_mean      = _mean(zspec_sum, zspec_cnt)

    def _mean_ci90(sum_, sumsq_, cnt_):
        m = np.divide(sum_, cnt_, out=np.full_like(sum_, np.nan), where=cnt_ > 0)
        var = np.divide(sumsq_ - (sum_**2)/np.maximum(cnt_, 1.0),
                        np.maximum(cnt_ - 1.0, 1.0),
                        out=np.full_like(sum_, np.nan),
                        where=cnt_ > 1)
        sem = np.divide(np.sqrt(var), np.sqrt(cnt_),
                        out=np.full_like(sum_, np.nan),
                        where=cnt_ > 1)
        z90 = 1.645
        ci = z90 * sem
        return m, ci

    # NEW: LFP trace stats
    lfp_tr_mean, lfp_tr_ci = _mean_ci90(lfp_tr_sum, lfp_tr_sumsq, lfp_tr_cnt)
    # existing traces
    z_mean,  z_ci  = _mean_ci90(z_sum,  z_sumsq,  z_cnt)
    sp_mean, sp_ci = _mean_ci90(sp_sum, sp_sumsq, sp_cnt)

    # ===== figure with an extra TOP row for LFP trace =====
    LABEL_FS=20; TICK_FS=18
    LFP_TRACE_COLOR = "#1f77b4"   # blue-ish
    PHOT_COLOR="#2ca02c"; SPEED_COLOR="k"

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ax0, ax1, ax2, ax3, ax4 = axes  # ax0 is the NEW top row (LFP trace)

    # 0) LFP trace mean ± 90% CI
    if np.any(np.isfinite(lfp_tr_mean)):
        lo = lfp_tr_mean - lfp_tr_ci
        hi = lfp_tr_mean + lfp_tr_ci
        ax0.fill_between(time_grid, lo, hi, color=LFP_TRACE_COLOR, alpha=0.20, linewidth=0)
        ax0.plot(time_grid, lfp_tr_mean, color=LFP_TRACE_COLOR, linewidth=1.8)
    ax0.set_ylabel("LFP (μV)", fontsize=LABEL_FS)
    ax0.tick_params(labelsize=TICK_FS)

    # 1) LFP spectrogram (avg)
    if lfp_mean_spec is not None:
        ax1.contourf(time_grid, freq_grid, lfp_mean_spec, levels=40)
    ax1.set_ylabel("Freq (Hz)", fontsize=LABEL_FS)
    ax1.set_ylim(0, 20); ax1.tick_params(labelsize=TICK_FS)

    # 2) Optical trace (z-scored) mean ± 90% CI
    if np.any(np.isfinite(z_mean)):
        z_plot = (-1.0/100.0) * z_mean
        z_lo   = (-1.0/100.0) * (z_mean - z_ci)
        z_hi   = (-1.0/100.0) * (z_mean + z_ci)
        ax2.fill_between(time_grid, z_lo, z_hi, color=PHOT_COLOR, alpha=0.25, linewidth=0)
        ax2.plot(time_grid, z_plot, color=PHOT_COLOR, linewidth=2)
    ax2.set_ylabel("−ΔF/F (V)", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)

    # 3) Optical spectrogram (avg)
    if zspec_mean is not None:
        ax3.contourf(time_grid, freq_grid, zspec_mean, levels=40)
        ax3.set_ylim(0, 20)
    ax3.set_ylabel("Freq (Hz)", fontsize=LABEL_FS); ax3.tick_params(labelsize=TICK_FS)

    # 4) Speed mean ± 90% CI
    if np.any(np.isfinite(sp_mean)):
        sp_lo = sp_mean - sp_ci; sp_hi = sp_mean + sp_ci
        ax4.fill_between(time_grid, sp_lo, sp_hi, color=SPEED_COLOR, alpha=0.20, linewidth=0)
        ax4.plot(time_grid, sp_mean, color=SPEED_COLOR, linewidth=2)
    ax4.set_ylabel("Speed (cm/s)", fontsize=LABEL_FS); ax4.tick_params(labelsize=TICK_FS)
    ax4.set_ylim(0, 30)
    ax4.set_xlabel("Time (s, approach = 0)", fontsize=LABEL_FS)

    # cosmetics & event line
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0)
        ax.axvline(0.0, color='r', linewidth=2)

    title_animal = os.path.basename(os.path.normpath(animal_root))
    fig.suptitle(f"Peri-Event Average — (n={n_trials} trials)", fontsize=30)
    fig.tight_layout(rect=[0,0,1,0.96])
    if show:
        plt.show()
    plt.close(fig)


def average_across_trials_pctl95(animal_root: str,
                                 landmarks: Dict,
                                 lfp_channel: str = "LFP_4",
                                 days: List[str] = ["Day1","Day2","Day3","Day4"],
                                 window_s: Tuple[float, float] = (-2.0, 2.0),
                                 show: bool = True):
    freq_grid = np.linspace(0.5, 20.0, 120)
    time_grid = np.linspace(window_s[0], window_s[1], int((window_s[1]-window_s[0])*100)+1)  # 100 Hz grid
    nT = time_grid.size
    z_val_to_plot = -1.0/100.0  # negate & scale for plotting

    # accumulators
    lfp_sum = None; lfp_cnt = None
    zspec_sum = None; zspec_cnt = None
    z_sum = np.zeros(nT, float); z_sumsq = np.zeros(nT, float); z_cnt = np.zeros(nT, float)
    sp_sum = np.zeros(nT, float); sp_sumsq = np.zeros(nT, float); sp_cnt = np.zeros(nT, float)
    n_trials = 0

    # NEW: store per-trial optical (z-scored) traces for percentile band
    z_trials = []

    def _accum_spec(sum_, cnt_, M):
        if M is None: return sum_, cnt_
        if sum_ is None:
            sum_ = np.zeros_like(M, float)
            cnt_ = np.zeros_like(M, float)
        good = np.isfinite(M)
        sum_[good] += M[good]
        cnt_[good] += 1
        return sum_, cnt_

    for day in days:
        day_dir = os.path.join(animal_root, day)
        if not os.path.isdir(day_dir):
            continue
        for sync in sorted(p for p in glob.glob(os.path.join(day_dir, "SyncRecording*")) if os.path.isdir(p)):
            pkl = os.path.join(sync, "aligned_cheeseboard.pkl")
            if not os.path.isfile(pkl):
                continue
            T = extract_trial_around_approach(
                pkl, landmarks, lfp_channel,
                window_s=window_s,
                phot_bin_hz=100.0,
                lfp_lowpass=500.0,
                freq_grid=freq_grid,
                time_grid_100Hz=time_grid,
                speed_grid_100Hz=time_grid
            )
            if T is None:
                continue
            n_trials += 1

            lfp_sum, lfp_cnt     = _accum_spec(lfp_sum,   lfp_cnt,   T.get("lfp_spec"))
            zspec_sum, zspec_cnt = _accum_spec(zspec_sum, zspec_cnt, T.get("z_spec"))

            z_bin = T.get("z_bin", None)   # (already per-trial z-scored by extractor)
            sp    = T.get("speed", None)

            if z_bin is not None and z_bin.size == nT:
                z_trials.append(z_bin)  # keep for percentile band
                good = np.isfinite(z_bin)
                z_sum[good]   += z_bin[good]
                z_sumsq[good] += (z_bin[good] ** 2)
                z_cnt[good]   += 1

            if sp is not None and sp.size == nT:
                good = np.isfinite(sp)
                sp_sum[good]   += sp[good]
                sp_sumsq[good] += (sp[good] ** 2)
                sp_cnt[good]   += 1

    if n_trials == 0:
        print("No usable trials found.")
        return

    def _mean(sum_, cnt_):
        if sum_ is None: return None
        return np.divide(sum_, cnt_, out=np.full_like(sum_, np.nan), where=cnt_ > 0)

    lfp_mean   = _mean(lfp_sum,   lfp_cnt)
    zspec_mean = _mean(zspec_sum, zspec_cnt)

    # speed: keep SEM-based 95% CI (unchanged logic, just 95% instead of 90%)
    def _mean_ci95(sum_, sumsq_, cnt_):
        m = np.divide(sum_, cnt_, out=np.full_like(sum_, np.nan), where=cnt_ > 0)
        var = np.divide(sumsq_ - (sum_**2)/np.maximum(cnt_, 1.0),
                        np.maximum(cnt_ - 1.0, 1.0),
                        out=np.full_like(sum_, np.nan),
                        where=cnt_ > 1)
        sem = np.divide(np.sqrt(var), np.sqrt(cnt_),
                        out=np.full_like(sum_, np.nan),
                        where=cnt_ > 1)
        ci = 1.96 * sem
        return m, ci, cnt_

    sp_mean, sp_ci, _ = _mean_ci95(sp_sum, sp_sumsq, sp_cnt)

    # --- OPTICAL: percentile band across trials (2.5–97.5%)
    if len(z_trials):
        Z = np.vstack(z_trials)                      # shape: n_kept x nT
        z_mean = np.nanmean(Z, axis=0)
        z_lo   = np.nanpercentile(Z,  2.5, axis=0)
        z_hi   = np.nanpercentile(Z, 97.5, axis=0)
        n_opt  = Z.shape[0]
    else:
        z_mean = np.full(nT, np.nan)
        z_lo = z_hi = z_mean
        n_opt = 0

    # ===== fig: averaged plots =====
    LABEL_FS=20; TICK_FS=18
    PHOT_COLOR="#2ca02c"; SPEED_COLOR="k"
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    if lfp_mean is not None:
        ax1.contourf(time_grid, freq_grid, lfp_mean, levels=40)
    ax1.set_ylabel("Freq (Hz)", fontsize=LABEL_FS)
    ax1.set_ylim(0, 20); ax1.tick_params(labelsize=TICK_FS)

    # Optical: mean + 2.5–97.5% percentile band (still negated for plotting)
    if np.any(np.isfinite(z_mean)):
        ax2.fill_between(time_grid, z_val_to_plot*z_lo, z_val_to_plot*z_hi,
                         color=PHOT_COLOR, alpha=0.25, linewidth=0)
        ax2.plot(time_grid, z_val_to_plot*z_mean, color=PHOT_COLOR, linewidth=2)
    ax2.set_ylabel("−ΔF/F (V)", fontsize=LABEL_FS); ax2.tick_params(labelsize=TICK_FS)

    if zspec_mean is not None:
        ax3.contourf(time_grid, freq_grid, zspec_mean, levels=40)
        ax3.set_ylim(0, 20)
    ax3.set_ylabel("Freq (Hz)", fontsize=LABEL_FS); ax3.tick_params(labelsize=TICK_FS)

    if np.any(np.isfinite(sp_mean)):
        sp_lo = sp_mean - sp_ci; sp_hi = sp_mean + sp_ci
        ax4.fill_between(time_grid, sp_lo, sp_hi, color=SPEED_COLOR, alpha=0.20, linewidth=0)
        ax4.plot(time_grid, sp_mean, color=SPEED_COLOR, linewidth=2)
    ax4.set_ylabel("Speed (cm/s)", fontsize=LABEL_FS); ax4.tick_params(labelsize=TICK_FS)
    ax4.set_ylim(0, 30)
    ax4.set_xlabel("Time (s, approach = 0)", fontsize=LABEL_FS)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.margins(x=0)
        ax.axvline(0.0, color='r', linewidth=2)

    title_animal = os.path.basename(os.path.normpath(animal_root))
    fig.suptitle(f"Peri-Event Average — {title_animal}  (n={n_opt} trials)",
                 fontsize=25)
    fig.tight_layout(rect=[0,0,1,0.96])
    if show: plt.show()
    plt.close(fig)

# ================== run ==================
if __name__ == "__main__":
    landmarks = {
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 10.0,
    }
    
    animal_parent = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363\Success"
    average_across_trials(animal_root=animal_parent, landmarks=landmarks,
                          lfp_channel="LFP_4", days=["Day1","Day2","Day3","Day4"],
                          window_s=(-2.0, 2), show=True)

    animal_parent = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1907336\Success"
    average_across_trials(animal_root=animal_parent, landmarks=landmarks,
                          lfp_channel="LFP_4", days=["Day1","Day2","Day3","Day4"],
                          window_s=(-2.0, 2), show=True)
    
    # animal_parent = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881365\Success"
    # average_across_trials(animal_root=animal_parent, landmarks=landmarks,
    #                       lfp_channel="LFP_4", days=["Day1","Day2","Day3","Day4"],
    #                       window_s=(-2.0, 2), show=True)
    
    # animal_parent = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success"
    # average_across_trials(animal_root=animal_parent, landmarks=landmarks,
    #                       lfp_channel="LFP_3", days=["Day1","Day2","Day3","Day4"],
    #                       window_s=(-2.0, 2), show=True)
    
    