# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 23:57:34 2025

@author: yifan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pickle, glob
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------- helpers (raw-only PSD; no filtering, binning, or smoothing) ----------
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

def _legend_bottom(ax, ncol=1, dy=-0.10):
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, dy),  # vertical offset below the axis
        ncol=ncol,
        frameon=False,
        fontsize=10
    )


def _auto_ylim(ax, curves, band=(1.0, 20.0), top_pct=98.0, pad=1.08):
    """
    curves: iterable of (f, y) tuples. Computes y-limit from data within 'band'
    using the 'top_pct' percentile cap to avoid huge DC/near-DC peaks.
    """
    import numpy as np
    vals = []
    for f, y in curves:
        if f is None or y is None:
            continue
        f = np.asarray(f, float); y = np.asarray(y, float)
        m = (f >= band[0]) & (f <= band[1]) & np.isfinite(y)
        if np.any(m):
            vals.append(y[m])
    if vals:
        v = np.concatenate(vals)
        hi = np.nanpercentile(v, top_pct)
        if np.isfinite(hi) and hi > 0:
            ax.set_ylim(0, hi * pad)
    ax.set_xlim(*band)

def _longest_true_run(mask: np.ndarray) -> slice:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return slice(0, 0)
    # find breaks
    brk = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[brk+1]]
    ends   = np.r_[idx[brk], idx[-1]]
    lengths = ends - starts + 1
    k = int(np.argmax(lengths))
    return slice(int(starts[k]), int(ends[k]) + 1)

# --- replace your _welch_psd_raw with this version ---
def _welch_psd_raw(x: np.ndarray, fs: float, kind: str,
                   detrend_mode=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch on RAW segment (no filtering). Chooses parameters for good low-f resolution.
    kind: 'lfp' or 'phot'.
    """
    from scipy.signal import welch
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < int(0.5 * fs):  # need at least 0.5 s
        return None, None

    if kind == 'lfp':
        # Use the whole segment -> fine resolution (e.g., 2 s => 0.5 Hz bins)
        nperseg = len(x)
        noverlap = 0
        detrend = False if detrend_mode is None else detrend_mode  # keep DC unless you set otherwise
    else:
        # Photometry: up to ~2 s windows if available
        nperseg = min(len(x), int(2.0 * fs))
        noverlap = nperseg // 2 if nperseg < len(x) else 0
        detrend = 'constant' if detrend_mode is None else detrend_mode

    f, Pxx = welch(x, fs=fs, window='hann',
                   nperseg=nperseg, noverlap=noverlap,
                   detrend=detrend, scaling='density',
                   return_onesided=True)
    return f, Pxx

# ---------- main ----------
def averaged_psd(animal_root: str,
                 landmarks: Dict,
                 lfp_channel: str = "LFP_4",
                 days: List[str] = ["Day1","Day2","Day3","Day4"],
                 window_s: Tuple[float, float] = (-2.0, 2.0),
                 speed_thresh_cm_s: float = 4.0,
                 show: bool = True):
    """
    Build PSDs from RAW signals only (no filtering/smoothing/binning).
    - Per-trial figure: LFP & GEVI pre vs reward-immobile (speed<4) for each trial.
    - Averaged figures (two): mean±95% CI (SEM) and mean with 2.5–97.5% percentile band.
    Legends are drawn at the bottom of each plot.

    Optical source: phot['green'] if present, else phot['z'].
    """
    F_LFP = np.linspace(0.0, 50.0, 501)
    F_PH  = np.linspace(0.0, 20.0, 201)

    lfp_psd_pre, lfp_psd_postimm = [], []
    phot_psd_pre, phot_psd_postimm = [], []

    def _avg_sem_and_percentiles(psd_list):
        if not psd_list:
            return None
        M = np.vstack(psd_list)
        mean = np.nanmean(M, axis=0)
        sd   = np.nanstd(M, axis=0, ddof=1)
        nvec = np.sum(np.isfinite(M), axis=0)
        sem  = sd / np.sqrt(np.maximum(nvec, 1))
        ci95 = 1.96 * sem
        pct_lo = np.nanpercentile(M,  2.5, axis=0)
        pct_hi = np.nanpercentile(M, 97.5, axis=0)
        n_trials = int(np.sum(np.any(np.isfinite(M), axis=1)))
        return dict(mean=mean, ci95=ci95, pct_lo=pct_lo, pct_hi=pct_hi, n=n_trials)

    for day in days:
        day_dir = os.path.join(animal_root, day)
        if not os.path.isdir(day_dir):
            continue
        for sync in sorted(p for p in glob.glob(os.path.join(day_dir, "SyncRecording*")) if os.path.isdir(p)):
            pkl = os.path.join(sync, "aligned_cheeseboard.pkl")
            if not os.path.isfile(pkl):
                continue

            with open(pkl, "rb") as f:
                D = pickle.load(f)
            e = D["ephys"]; p = D["phot"]; b = D["beh"]

            t_e  = np.asarray(e["t"], float)
            lfp  = np.asarray(e[lfp_channel], float)
            t_ph = np.asarray(p["t"], float)

            # --- change here: prefer 'green', else 'z'
            if p.get("green") is not None:
                zraw = np.asarray(p["green"], float)
                opt_label = "GEVI"
            elif p.get("z") is not None:
                zraw = np.asarray(p["z"], float)
                opt_label = "GEVI z"
            else:
                zraw = None
                opt_label = "GEVI"

            fs_ph = float(p.get("fs", 1682.92))
            t_b = np.asarray(b["t"], float)
            head = (np.asarray(b["head"][0], float), np.asarray(b["head"][1], float))

            appr = b.get("approach_time", None)
            if appr is None or not np.isfinite(appr):
                continue

            t0 = appr + window_s[0]; t1 = appr + window_s[1]
            if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                continue

            dt_e = np.nanmedian(np.diff(t_e))
            if not (np.isfinite(dt_e) and dt_e > 0):
                continue
            fs_e = 30000

            mask_e = (t_e >= t0) & (t_e <= t1)
            if not np.any(mask_e):
                continue
            t_e_w = t_e[mask_e] - appr
            lfp_w = lfp[mask_e]

            mask_b = (t_b >= t0) & (t_b <= t1)
            imm_mask_e = None
            ts_sp = np.array([]); sp = np.array([])
            if np.any(mask_b):
                t_b_w = t_b[mask_b] - appr
                px_to_cm = compute_px_to_cm_scale(landmarks["cheeseboard_center"],
                                                  landmarks["cheeseboard_ends"], 100.0)
                ts_sp, sp = compute_speed_cm_per_s(t_b_w, (head[0][mask_b], head[1][mask_b]), px_to_cm)
                if ts_sp.size:
                    sp_lfp = np.interp(t_e_w, ts_sp, sp, left=np.nan, right=np.nan)
                    imm_mask_e = (t_e_w >= 0.0) & (t_e_w <= 2.0) & (sp_lfp < speed_thresh_cm_s) & np.isfinite(sp_lfp)
            pre_mask_e  = (t_e_w >= -2.0) & (t_e_w <= 0.0)
            post_mask_e = (t_e_w >=  0.0) & (t_e_w <= 3.0)
            if imm_mask_e is None:
                imm_mask_e = post_mask_e

            seg_e = _longest_true_run(imm_mask_e)
            x_pre  = lfp_w[pre_mask_e]
            x_post = lfp_w[seg_e]

            f_pre,  P_pre  = _welch_psd_raw(x_pre,  fs=fs_e, kind='lfp',  detrend_mode=False)
            f_post, P_post = _welch_psd_raw(x_post, fs=fs_e, kind='lfp',  detrend_mode=False)
            if f_pre is not None:
                lfp_psd_pre.append(np.interp(F_LFP, f_pre,  P_pre,  left=np.nan, right=np.nan))
            if f_post is not None:
                lfp_psd_postimm.append(np.interp(F_LFP, f_post, P_post, left=np.nan, right=np.nan))

            # --- GEVI
            if zraw is None or not t_ph.size:
                have_phot_trial = False
                fz_pre = fz_post = None
            else:
                mask_ph = (t_ph >= t0) & (t_ph <= t1)
                if not np.any(mask_ph):
                    have_phot_trial = False
                    fz_pre = fz_post = None
                else:
                    t_ph_w = t_ph[mask_ph] - appr
                    z_w    = zraw[mask_ph]
                    pre_mask_ph  = (t_ph_w >= -2.0) & (t_ph_w <= 0.0)
                    post_mask_ph = (t_ph_w >=  0.0) & (t_ph_w <= 2.0)
                    if ts_sp.size:
                        sp_ph = np.interp(t_ph_w, ts_sp, sp, left=np.nan, right=np.nan)
                        imm_mask_ph = post_mask_ph & (sp_ph < speed_thresh_cm_s) & np.isfinite(sp_ph)
                    else:
                        imm_mask_ph = post_mask_ph
                    seg_ph = _longest_true_run(imm_mask_ph)
                    x_pre_ph  = z_w[pre_mask_ph]
                    x_post_ph = z_w[seg_ph]

                    fz_pre,  Pz_pre  = _welch_psd_raw(x_pre_ph,  fs=fs_ph, kind='phot', detrend_mode='constant')
                    fz_post, Pz_post = _welch_psd_raw(x_post_ph, fs=fs_ph, kind='phot', detrend_mode='constant')
                    have_phot_trial = fz_pre is not None or fz_post is not None
                    if fz_pre is not None:
                        phot_psd_pre.append(np.interp(F_PH, fz_pre,  Pz_pre,  left=np.nan, right=np.nan))
                    if fz_post is not None:
                        phot_psd_postimm.append(np.interp(F_PH, fz_post, Pz_post, left=np.nan, right=np.nan))

            # ##-------- per-trial figure --------
            # figT, (axLT, axZT) = plt.subplots(
            #     1, 2, figsize=(6, 6), constrained_layout=True
            # )  # ≈1.5:1 tall:wide overall
            
            # if f_pre is not None:
            #     axLT.plot(f_pre,  P_pre,  label="LFP: pre (-2–0 s)")
            # if f_post is not None:
            #     axLT.plot(f_post, P_post, label="LFP: reward & immobile (0–2 s, speed<4)")
            # axLT.set_xlabel("Frequency (Hz)")
            # axLT.set_ylabel("PSD (V²/Hz)")
            # _auto_ylim(axLT, [(f_pre, P_pre), (f_post, P_post)], band=(1, 20), top_pct=99.9)
            # _legend_bottom(axLT, dy=-0.08)
            
            # if have_phot_trial:
            #     if fz_pre is not None:
            #         axZT.plot(fz_pre,  Pz_pre,  label=f"{opt_label}: pre (-2–0 s)")
            #     if fz_post is not None:
            #         axZT.plot(fz_post, Pz_post, label=f"{opt_label}: reward & immobile")
            # axZT.set_xlabel("Frequency (Hz)")
            # axZT.set_ylabel("PSD (units²/Hz)")
            # _auto_ylim(axZT, [(fz_pre, Pz_pre), (fz_post, Pz_post)], band=(1, 20), top_pct=99.9)
            # _legend_bottom(axZT, dy=-0.08)
            
            # trial_label = f"{os.path.basename(day_dir)} / {os.path.basename(sync)}"
            # figT.suptitle(f"Single-trial PSD — {trial_label}", fontsize=12, y=0.98)
            # # give extra room for bottom legends
            # figT.subplots_adjust(bottom=0.20, top=0.90, wspace=0.30)
            # if show: plt.show()
            # plt.close(figT)


    # ---------------- averaged figures ----------------
    aggL_pre  = _avg_sem_and_percentiles(lfp_psd_pre)
    aggL_post = _avg_sem_and_percentiles(lfp_psd_postimm)
    aggZ_pre  = _avg_sem_and_percentiles(phot_psd_pre)
    aggZ_post = _avg_sem_and_percentiles(phot_psd_postimm)

    have_lfp  = (aggL_pre is not None)  and (aggL_post is not None)
    have_phot = (aggZ_pre is not None)  and (aggZ_post is not None)
    if not (have_lfp or have_phot):
        print("No PSDs computed (check data/approach times).")
        return

    # Figure A: mean ± 95% CI (SEM)
    figA, (axL_A, axZ_A) = plt.subplots(
        1, 2, figsize=(6, 5), constrained_layout=True
    )
    
    if have_lfp:
        m_pre, ci_pre = aggL_pre["mean"],  aggL_pre["ci95"]
        m_post, ci_post = aggL_post["mean"], aggL_post["ci95"]
        axL_A.plot(F_LFP, m_pre,  label=f"LFP Running (n={aggL_pre['n']})")
        axL_A.fill_between(F_LFP, m_pre-ci_pre,  m_pre+ci_pre,  alpha=0.25, linewidth=0)
        axL_A.plot(F_LFP, m_post, label=f"LFP reward&immobile (n={aggL_post['n']})")
        axL_A.fill_between(F_LFP, m_post-ci_post, m_post+ci_post, alpha=0.25, linewidth=0)
        axL_A.set_xlabel("Frequency (Hz)")
        axL_A.set_ylabel("PSD (V²/Hz)")
        _auto_ylim(axL_A, [(F_LFP, m_pre+ci_pre), (F_LFP, m_post+ci_post)],
                   band=(1, 20), top_pct=95)
        _legend_bottom(axL_A, dy=-0.08)
    
    if have_phot:
        m_pre, ci_pre = aggZ_pre["mean"],  aggZ_pre["ci95"]
        m_post, ci_post = aggZ_post["mean"], aggZ_post["ci95"]
        axZ_A.plot(F_PH, m_pre,  label=f"{opt_label} Running (n={aggZ_pre['n']})")
        axZ_A.fill_between(F_PH, m_pre-ci_pre,  m_pre+ci_pre,  alpha=0.25, linewidth=0)
        axZ_A.plot(F_PH, m_post, label=f"{opt_label} reward&immobile (n={aggZ_post['n']})")
        axZ_A.fill_between(F_PH, m_post-ci_post, m_post+ci_post, alpha=0.25, linewidth=0)
        axZ_A.set_xlabel("Frequency (Hz)")
        axZ_A.set_ylabel("PSD (units²/Hz)")
        _auto_ylim(axZ_A, [(F_PH, m_pre+ci_pre), (F_PH, m_post+ci_post)],
                   band=(1, 20), top_pct=89)
        _legend_bottom(axZ_A, dy=-0.08)
    
    #figA.suptitle("Averaged PSDs — mean ± 95% CI (SEM)", y=0.98)
    figA.subplots_adjust(bottom=0.22, top=0.90, wspace=0.30)
    if show: plt.show()
    plt.close(figA)


    # Figure B: mean with 2.5–97.5% percentile band
    # figB, (axL_B, axZ_B) = plt.subplots(
    #     1, 2, figsize=(6, 6), constrained_layout=True
    # )
    
    # if have_lfp:
    #     m_pre, lo_pre, hi_pre = aggL_pre["mean"], aggL_pre["pct_lo"], aggL_pre["pct_hi"]
    #     m_post, lo_post, hi_post = aggL_post["mean"], aggL_post["pct_lo"], aggL_post["pct_hi"]
    #     axL_B.plot(F_LFP, m_pre,  label=f"LFP pre (n={aggL_pre['n']})")
    #     axL_B.fill_between(F_LFP, lo_pre,  hi_pre,  alpha=0.25, linewidth=0)
    #     axL_B.plot(F_LFP, m_post, label=f"LFP reward&immobile (n={aggL_post['n']})")
    #     axL_B.fill_between(F_LFP, lo_post, hi_post, alpha=0.25, linewidth=0)
    #     axL_B.set_xlabel("Frequency (Hz)")
    #     axL_B.set_ylabel("PSD (V²/Hz)")
    #     _auto_ylim(axL_B, [(F_LFP, hi_pre), (F_LFP, hi_post)], band=(1, 20), top_pct=99)
    #     _legend_bottom(axL_B, dy=-0.08)
    
    # if have_phot:
    #     m_pre, lo_pre, hi_pre = aggZ_pre["mean"], aggZ_pre["pct_lo"], aggZ_pre["pct_hi"]
    #     m_post, lo_post, hi_post = aggZ_post["mean"], aggZ_post["pct_lo"], aggZ_post["pct_hi"]
    #     axZ_B.plot(F_PH, m_pre,  label=f"{opt_label} pre (n={aggZ_pre['n']})")
    #     axZ_B.fill_between(F_PH, lo_pre,  hi_pre,  alpha=0.25, linewidth=0)
    #     axZ_B.plot(F_PH, m_post, label=f"{opt_label} reward&immobile (n={aggZ_post['n']})")
    #     axZ_B.fill_between(F_PH, lo_post, hi_post, alpha=0.25, linewidth=0)
    #     axZ_B.set_xlabel("Frequency (Hz)")
    #     axZ_B.set_ylabel("PSD (units²/Hz)")
    #     _auto_ylim(axZ_B, [(F_PH, hi_pre), (F_PH, hi_post)], band=(1, 20), top_pct=99)
    #     _legend_bottom(axZ_B, dy=-0.08)
    
    # figB.suptitle("Averaged PSDs — mean ± 2.5–97.5% percentile band", y=0.98)
    # figB.subplots_adjust(bottom=0.22, top=0.90, wspace=0.30)
    # if show: plt.show()
    # plt.close(figB)


# ================== run ==================
if __name__ == "__main__":
    
    landmarks = {
        "cheeseboard_center": (306, 230),
        "cheeseboard_ends": [(99, 246), (515, 214), (291, 26), (232, 436)],
        "reward_pt": (410, 111),
        "reward_zone_radius": 15.0,
    }

    
    # animal_root = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881365\Success"
    # averaged_psd(animal_root=animal_root, landmarks=landmarks,
    #                   lfp_channel="LFP_4",
    #                   days=["Day1","Day2","Day3","Day4"],
    #                   window_s=(-5.0, 5.0),
    #                   speed_thresh_cm_s=4.0,
    #                   show=True)   #top_pct=97
    
    # animal_root = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1910567\Success"
    # averaged_psd(animal_root=animal_root, landmarks=landmarks,
    #                   lfp_channel="LFP_4",
    #                   days=["Day1","Day2","Day3","Day4"],
    #                   window_s=(-5.0, 5.0),
    #                   speed_thresh_cm_s=3,
    #                   show=True)  #top_pct=90
    
    animal_root = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1881363\Success"
    averaged_psd(animal_root=animal_root, landmarks=landmarks,
                      lfp_channel="LFP_4",
                      days=["Day1","Day2","Day3","Day4"],
                      window_s=(-2.0, 2.0),
                      speed_thresh_cm_s=4.0,
                      show=True)
    
    animal_root = r"G:\2025_ATLAS_SPAD\CB_Jedi2P\1907336\Success"
    averaged_psd(animal_root=animal_root, landmarks=landmarks,
                      lfp_channel="LFP_4",
                      days=["Day1","Day2","Day3","Day4"],
                      window_s=(-2.0, 2.0),
                      speed_thresh_cm_s=4.0,
                      show=True)
    
