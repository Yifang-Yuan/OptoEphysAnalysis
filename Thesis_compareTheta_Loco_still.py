# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 00:21:58 2025

@author: yifan
"""

"""
Theta metrics (frequency & power): locomotion vs awake-stationary (pooled across sessions)

Additions vs previous:
- Also extracts theta power and relative theta power.
- Plots Locomotion vs Awake-stationary for power (LFP & GEVI).
- Saves power metrics to CSV; runs stats on them too.
"""

import os
import glob
import json
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy import stats

# ==========================
# === USER CONFIGURATION ===
# ==========================
LOCOMOTION_PARENT  = r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ALocomotion'
STATIONARY_PARENT  = r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationary'

OUTPUT_ROOT = os.path.join(os.path.dirname(LOCOMOTION_PARENT), "theta_freq_loco_vs_stationary_outputs")

Fs_raw      = 10_000
TARGET_FS   = 100
THETA_BAND  = (4.0, 12.0)
DELTA_BAND  = (1.0, 4.0)
TOTAL_BAND  = (1.0, 40.0)
LFP_CHANNEL = "LFP_3"

# Windowing
WINDOW_SEC      = 1.0
WINDOW_STEP_SEC = 0.5

# Gating
SPEED_MIN_CM_S   = 1.0          # locomotion definition: mean speed > 1 cm/s
SPEED_MAX_CM_S   = 50.0         # sanity cap
THETA_DELTA_MIN  = 1.0          # keep theta-rich windows if theta/delta > this (computed on LFP)

# Plot look
DPI = 140
FREQ_COLOURS     = ("#1f77b4", "#ff7f0e")  # blue vs orange
POWER_COLOURS    = ("#2ca02c", "#d62728")  # green vs red
RELPOWER_COLOURS = ("#a6cee3", "#fb9a99") # purple vs brown
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ================
# Helper functions
# ================
def find_syncrecording_folders(parent: str) -> List[str]:
    return sorted(glob.glob(os.path.join(parent, "SyncRecording*")))

def load_pickle_if_exists(folder: str) -> Optional[pd.DataFrame]:
    p = os.path.join(folder, "Ephys_tracking_photometry_aligned.pkl")
    if not os.path.isfile(p):
        hits = glob.glob(os.path.join(folder, "**", "Ephys_tracking_photometry_aligned.pkl"), recursive=True)
        if not hits:
            return None
        p = hits[0]
    try:
        df = pd.read_pickle(p)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    except Exception as e:
        print(f"Failed to read pickle in {folder}: {e}")
        return None

def bin_average(x: np.ndarray, bin_sz: int) -> np.ndarray:
    n = (len(x) // bin_sz) * bin_sz
    if n <= 0:
        return np.array([], dtype=float)
    return np.nanmean(np.asarray(x[:n], dtype=float).reshape(-1, bin_sz), axis=1)

# ==========================
# Spectral / theta features
# ==========================
def compute_theta_features_per_window(sig: np.ndarray,
                                      target_fs: float,
                                      win_len: int,
                                      step: int,
                                      freqs: np.ndarray,
                                      theta_mask: np.ndarray,
                                      delta_mask: np.ndarray,
                                      total_mask: np.ndarray,
                                      freq_theta: np.ndarray,
                                      df_theta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      theta_peak_freq [Hz],
      theta_band_power (integrated over theta band),
      delta_band_power (for gating),
      total_band_power (for relative power)
    """
    hann = get_window('hann', win_len)
    hann_norm = np.sum(hann**2)

    def psd(seg):
        sw = seg * hann
        spec = np.fft.rfft(sw)
        return (np.abs(spec) ** 2) / (hann_norm * target_fs)

    idx_starts = np.arange(0, len(sig) - win_len + 1, step)
    f_out, p_theta_out, p_delta_out, p_total_out = [], [], [], []

    dfreq = freqs[1] - freqs[0] if len(freqs) > 1 else np.nan

    for s in idx_starts:
        seg = sig[s:s+win_len]
        Pxx = psd(seg)

        p_th = Pxx[theta_mask]
        if p_th.size == 0 or not np.isfinite(p_th).any():
            f_out.append(np.nan)
            p_theta_out.append(np.nan)
            p_delta_out.append(np.nan)
            p_total_out.append(np.nan)
            continue

        i_max = int(np.nanargmax(p_th))
        # Quadratic interpolation inside theta band
        if len(p_th) >= 3 and 0 < i_max < (len(p_th) - 1) and np.isfinite(df_theta):
            y1, y2, y3 = p_th[i_max-1], p_th[i_max], p_th[i_max+1]
            denom = (y1 - 2*y2 + y3)
            if denom == 0 or not np.isfinite(denom):
                delta = 0.0
            else:
                delta = 0.5 * (y1 - y3) / denom
                if not np.isfinite(delta) or abs(delta) > 1.0:
                    delta = 0.0
            f_peak = freq_theta[i_max] + delta * df_theta
        else:
            f_peak = freq_theta[i_max]

        # Clamp to theta band
        if np.isfinite(f_peak):
            f_peak = min(max(f_peak, THETA_BAND[0]), THETA_BAND[1])
        else:
            f_peak = np.nan

        # Integrate band powers
        integ = (dfreq if np.isfinite(dfreq) else 1.0)
        p_theta = np.nansum(p_th) * integ
        p_delta = np.nansum(Pxx[delta_mask]) * integ
        p_total = np.nansum(Pxx[total_mask]) * integ

        f_out.append(f_peak)
        p_theta_out.append(p_theta)
        p_delta_out.append(p_delta)
        p_total_out.append(p_total)

    return (np.array(f_out),
            np.array(p_theta_out),
            np.array(p_delta_out),
            np.array(p_total_out))

def extract_theta_feats_from_session(df: pd.DataFrame,
                                     require_speed: bool,
                                     speed_min: float,
                                     speed_max: float,
                                     theta_delta_min: float
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (lfp_freq, opt_freq, lfp_theta_power, opt_theta_power, lfp_theta_relpower, opt_theta_relpower)
    for *kept* windows.
    If require_speed is True → applies speed gating; otherwise ignores speed (OK if no speed column).
    Always applies theta-rich gating via theta/delta > theta_delta_min on LFP.
    """
    # Down-bin raw to TARGET_FS by simple bin averaging
    bin_sz = int(round(Fs_raw / TARGET_FS))

    # Speed handling
    if 'speed' in df.columns:
        spd = np.clip(bin_average(df['speed'].to_numpy(), bin_sz), 0, None)
    else:
        spd = None

    # Signals
    if LFP_CHANNEL not in df.columns:
        raise KeyError(f"{LFP_CHANNEL} not found in dataframe columns.")
    lfp = bin_average(df[LFP_CHANNEL].to_numpy(), bin_sz)

    if 'zscore_raw' not in df.columns:
        raise KeyError("zscore_raw not found in dataframe columns.")
    opt = bin_average(df['zscore_raw'].to_numpy(), bin_sz)

    n = min(len(lfp), len(opt))
    if spd is not None:
        n = min(n, len(spd))
        spd = spd[:n]
    lfp = lfp[:n]
    opt = opt[:n]

    # Windowing grid
    win_len = int(round(WINDOW_SEC * TARGET_FS))
    step    = int(round(WINDOW_STEP_SEC * TARGET_FS))
    freqs   = np.fft.rfftfreq(win_len, d=1.0 / TARGET_FS)

    th_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    de_mask = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])

    freq_th = freqs[th_mask]
    df_th = (freq_th[1] - freq_th[0]) if len(freq_th) > 1 else np.nan

    # Per-window features
    lfp_f, lfp_pth, lfp_pde, lfp_ptot = compute_theta_features_per_window(
        lfp, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )
    opt_f, opt_pth, opt_pde, opt_ptot = compute_theta_features_per_window(
        opt, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )

    # Build window means for speed
    if spd is not None:
        idx_starts = np.arange(0, len(lfp) - win_len + 1, step)
        spd_win = np.array([np.nanmean(spd[s:s+win_len]) for s in idx_starts])
    else:
        spd_win = None

    # Gating
    keep = np.isfinite(lfp_f) & np.isfinite(opt_f)

    # Theta-rich via LFP theta/delta
    lfp_theta_delta = lfp_pth / (lfp_pde + 1e-12)
    keep &= (lfp_theta_delta > theta_delta_min)

    # Speed gating for locomotion condition only
    if require_speed:
        if spd_win is None:
            keep &= False
        else:
            keep &= np.isfinite(spd_win)
            keep &= (spd_win > speed_min) & (spd_win <= speed_max)

    # Relative power
    lfp_rel = lfp_pth / (lfp_ptot + 1e-12)
    opt_rel = opt_pth / (opt_ptot + 1e-12)

    return (lfp_f[keep], opt_f[keep],
            lfp_pth[keep], opt_pth[keep],
            lfp_rel[keep], opt_rel[keep])

# ================
# Stats & plotting
# ================
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float); b = b.astype(float)
    na, nb = len(a), len(b)
    va = np.var(a, ddof=1) if na > 1 else 0.0
    vb = np.var(b, ddof=1) if nb > 1 else 0.0
    s = np.sqrt(((na-1)*va + (nb-1)*vb) / max(na+nb-2, 1))
    return (np.nanmean(a) - np.nanmean(b)) / (s if s > 0 else np.nan)

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return np.nan
    sb = np.sort(b)
    import bisect
    greater = sum(n - bisect.bisect_right(sb, ai) for ai in a)
    less    = sum(bisect.bisect_left(sb, ai) for ai in a)
    return (greater - less) / (m * n)

def describe_array(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    return {
        "n": int(x.size),
        "mean": float(np.nanmean(x)) if x.size else np.nan,
        "std": float(np.nanstd(x, ddof=1)) if x.size > 1 else np.nan,
        "median": float(np.nanmedian(x)) if x.size else np.nan,
        "q25": float(np.nanquantile(x, 0.25)) if x.size else np.nan,
        "q75": float(np.nanquantile(x, 0.75)) if x.size else np.nan,
    }

def _auto_bins(a: np.ndarray, b: np.ndarray, nbins: int = 40) -> np.ndarray:
    x = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if x.size == 0:
        return np.linspace(0, 1, nbins)
    lo, hi = np.nanquantile(x, [0.01, 0.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            hi = lo + 1.0
    return np.linspace(lo, hi, nbins)

def plot_hist_violin(a: np.ndarray, b: np.ndarray, title: str, xlabel: str, out_prefix: str,
                     bins=None, colors=("black", "grey")):

    # Histograms
    plt.figure(figsize=(7,5), dpi=DPI)
    if bins is None:
        bins = _auto_bins(a, b, nbins=40)
    plt.hist(a, bins=bins, alpha=0.6, label='Locomotion', density=True, color=colors[0])
    plt.hist(b, bins=bins, alpha=0.6, label='Awake-stationary', density=True, color=colors[1])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(frameon=False,fontsize=15)
    plt.tight_layout()
    plt.savefig(out_prefix + "_hist.png"); plt.close()

    # Violin
    plt.figure(figsize=(6,5), dpi=DPI)
    vp = plt.violinplot([a, b], showmeans=True, showextrema=True)
    # Colour each violin body
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.5)
    # Make the summary lines visible
    for part in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
        if part in vp:
            vp[part].set_color('black')
    plt.xticks([1,2], ['Locomotion','Awake-stationary'], fontsize=16)
    plt.ylabel(xlabel, fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_prefix + "_violin.png"); plt.close()

def plot_log_hist(a: np.ndarray, b: np.ndarray, title: str, xlabel_raw: str, out_path_png: str, colors=("black", "grey")):

    # Log10 hist for skewed power
    a_pos = a[a > 0]; b_pos = b[b > 0]
    if a_pos.size == 0 or b_pos.size == 0:
        return
    la, lb = np.log10(a_pos), np.log10(b_pos)
    bins = _auto_bins(la, lb, nbins=40)
    plt.figure(figsize=(7,5), dpi=DPI)
    plt.hist(la, bins=bins, alpha=0.6, label='Locomotion', density=True, color=colors[0])
    plt.hist(lb, bins=bins, alpha=0.6, label='Awake-stationary', density=True, color=colors[1])

    plt.xlabel(f"log10({xlabel_raw})", fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path_png); plt.close()

# =====
# Main
# =====
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # -------- Collect locomotion --------
    loco_sessions = find_syncrecording_folders(LOCOMOTION_PARENT)
    loco_lfp_f, loco_opt_f = [], []
    loco_lfp_pow, loco_opt_pow = [], []
    loco_lfp_rel, loco_opt_rel = [], []

    for s in loco_sessions:
        df = load_pickle_if_exists(s)
        if df is None or df.empty:
            print(f"[LOCO] Skipping empty/missing: {s}")
            continue
        try:
            lf_f, op_f, lf_p, op_p, lf_r, op_r = extract_theta_feats_from_session(
                df,
                require_speed=True,
                speed_min=SPEED_MIN_CM_S,
                speed_max=SPEED_MAX_CM_S,
                theta_delta_min=THETA_DELTA_MIN
            )
        except Exception as e:
            print(f"[LOCO] Error in {s}: {e}")
            continue
        if lf_f.size:
            loco_lfp_f.append(lf_f);   loco_opt_f.append(op_f)
            loco_lfp_pow.append(lf_p); loco_opt_pow.append(op_p)
            loco_lfp_rel.append(lf_r); loco_opt_rel.append(op_r)

    def _concat_or_empty(lst):
        return np.concatenate(lst) if len(lst) else np.array([])

    loco_lfp_f   = _concat_or_empty(loco_lfp_f)
    loco_opt_f   = _concat_or_empty(loco_opt_f)
    loco_lfp_pow = _concat_or_empty(loco_lfp_pow)
    loco_opt_pow = _concat_or_empty(loco_opt_pow)
    loco_lfp_rel = _concat_or_empty(loco_lfp_rel)
    loco_opt_rel = _concat_or_empty(loco_opt_rel)

    # -------- Collect awake-stationary (theta-rich only) --------
    stat_sessions = find_syncrecording_folders(STATIONARY_PARENT)
    stat_lfp_f, stat_opt_f = [], []
    stat_lfp_pow, stat_opt_pow = [], []
    stat_lfp_rel, stat_opt_rel = [], []

    for s in stat_sessions:
        df = load_pickle_if_exists(s)
        if df is None or df.empty:
            print(f"[STAT] Skipping empty/missing: {s}")
            continue
        try:
            lf_f, op_f, lf_p, op_p, lf_r, op_r = extract_theta_feats_from_session(
                df,
                require_speed=False,    # ignore speed here
                speed_min=SPEED_MIN_CM_S,
                speed_max=SPEED_MAX_CM_S,
                theta_delta_min=THETA_DELTA_MIN
            )
        except Exception as e:
            print(f"[STAT] Error in {s}: {e}")
            continue
        if lf_f.size:
            stat_lfp_f.append(lf_f);   stat_opt_f.append(op_f)
            stat_lfp_pow.append(lf_p); stat_opt_pow.append(op_p)
            stat_lfp_rel.append(lf_r); stat_opt_rel.append(op_r)

    stat_lfp_f   = _concat_or_empty(stat_lfp_f)
    stat_opt_f   = _concat_or_empty(stat_opt_f)
    stat_lfp_pow = _concat_or_empty(stat_lfp_pow)
    stat_opt_pow = _concat_or_empty(stat_opt_pow)
    stat_lfp_rel = _concat_or_empty(stat_lfp_rel)
    stat_opt_rel = _concat_or_empty(stat_opt_rel)

    # Quick sanity
    print(f"Locomotion kept:    LFP n={loco_lfp_f.size}, GEVI n={loco_opt_f.size}")
    print(f"Stationary kept:    LFP n={stat_lfp_f.size}, GEVI n={stat_opt_f.size}")

    # ===== Save tidy CSV =====
    rows = []
    # Locomotion
    for f, p, r in zip(loco_lfp_f, loco_lfp_pow, loco_lfp_rel):
        rows.append({"modality":"LFP", "condition":"Locomotion",
                     "theta_freq_hz":float(f), "theta_power":float(p), "theta_rel_power":float(r)})
    for f, p, r in zip(loco_opt_f, loco_opt_pow, loco_opt_rel):
        rows.append({"modality":"Optical", "condition":"Locomotion",
                     "theta_freq_hz":float(f), "theta_power":float(p), "theta_rel_power":float(r)})
    # Stationary
    for f, p, r in zip(stat_lfp_f, stat_lfp_pow, stat_lfp_rel):
        rows.append({"modality":"LFP", "condition":"Awake-stationary",
                     "theta_freq_hz":float(f), "theta_power":float(p), "theta_rel_power":float(r)})
    for f, p, r in zip(stat_opt_f, stat_opt_pow, stat_opt_rel):
        rows.append({"modality":"Optical", "condition":"Awake-stationary",
                     "theta_freq_hz":float(f), "theta_power":float(p), "theta_rel_power":float(r)})

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_ROOT, "theta_metrics_locomotion_vs_stationary.csv")
    df_out.to_csv(csv_path, index=False)

    # ===== Stats (per modality × metric) =====
    def compare_and_report(a: np.ndarray, b: np.ndarray, name: str) -> Dict:
        desc_a = describe_array(a); desc_b = describe_array(b)
        t_res = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        af = a[np.isfinite(a)]; bf = b[np.isfinite(b)]
        if af.size > 0 and bf.size > 0:
            u_res = stats.mannwhitneyu(af, bf, alternative='two-sided')
        else:
            u_res = None
        cd = _cohens_d(af, bf) if af.size and bf.size else np.nan
        cliff = _cliffs_delta(af, bf) if af.size and bf.size else np.nan
        return {
            "measure": name,
            "locomotion": desc_a,
            "awake_stationary": desc_b,
            "welch_t": {"t": float(t_res.statistic) if t_res else np.nan,
                        "p": float(t_res.pvalue) if t_res else np.nan},
            "mannwhitney_u": {"U": float(u_res.statistic) if u_res else np.nan,
                              "p": float(u_res.pvalue) if u_res else np.nan} if u_res else {"U": np.nan, "p": np.nan},
            "effect_sizes": {"cohens_d": float(cd), "cliffs_delta": float(cliff)}
        }

    stats_dict = {
        "LFP": {
            "freq": compare_and_report(loco_lfp_f,   stat_lfp_f,   "theta_freq_hz"),
            "power": compare_and_report(loco_lfp_pow, stat_lfp_pow, "theta_power"),
            "rel_power": compare_and_report(loco_lfp_rel, stat_lfp_rel, "theta_rel_power")
        },
        "Optical": {
            "freq": compare_and_report(loco_opt_f,   stat_opt_f,   "theta_freq_hz"),
            "power": compare_and_report(loco_opt_pow, stat_opt_pow, "theta_power"),
            "rel_power": compare_and_report(loco_opt_rel, stat_opt_rel, "theta_rel_power")
        }
    }

    # ===== Plots =====
    # Frequency (kept from before, with your fontsize tweaks)
    def bins_freq():
        return np.linspace(THETA_BAND[0], THETA_BAND[1], 40)

    # Frequency (keep your pair here)
    plot_hist_violin(loco_lfp_f, stat_lfp_f,
                     title="Theta freq (LFP): Locomotion vs Stationary",
                     xlabel="Theta peak frequency (Hz)",
                     out_prefix=os.path.join(OUTPUT_ROOT, "lfp_theta_freq"),
                     bins=bins_freq(),
                     colors=FREQ_COLOURS)
    
    plot_hist_violin(loco_opt_f, stat_opt_f,
                     title="Theta freq (GEVI): Locomotion vs Stationary",
                     xlabel="Theta peak frequency (Hz)",
                     out_prefix=os.path.join(OUTPUT_ROOT, "opt_theta_freq"),
                     bins=bins_freq(),
                     colors=FREQ_COLOURS)
    
    # Absolute theta power (use a *different* pair from the same palette)
    plot_hist_violin(loco_lfp_pow, stat_lfp_pow,
                     title="Theta power (LFP): Locomotion vs Stationary",
                     xlabel="Theta band power (a.u.)",
                     out_prefix=os.path.join(OUTPUT_ROOT, "lfp_theta_power"),
                     colors=POWER_COLOURS)
    
    plot_hist_violin(loco_opt_pow, stat_opt_pow,
                     title="Theta power (GEVI): Locomotion vs Stationary",
                     xlabel="Theta band power (a.u.)",
                     out_prefix=os.path.join(OUTPUT_ROOT, "opt_theta_power"),
                     colors=POWER_COLOURS)
    
    # Log-hist for absolute power
    plot_log_hist(loco_lfp_pow, stat_lfp_pow,
                  title="Theta power (LFP): log10 histogram",
                  xlabel_raw="theta band power (a.u.)",
                  out_path_png=os.path.join(OUTPUT_ROOT, "lfp_theta_power_loghist.png"),
                  colors=POWER_COLOURS)
    
    plot_log_hist(loco_opt_pow, stat_opt_pow,
                  title="Theta power (GEVI): log10 histogram",
                  xlabel_raw="theta band power (a.u.)",
                  out_path_png=os.path.join(OUTPUT_ROOT, "opt_theta_power_loghist.png"),
                  colors=POWER_COLOURS)

    # Relative theta power (a third pair to stay visually distinct)
    plot_hist_violin(loco_lfp_rel, stat_lfp_rel,
                     title="Relative theta power (LFP): Locomotion vs Stationary",
                     xlabel="Theta / total (1–40 Hz)",
                     out_prefix=os.path.join(OUTPUT_ROOT, "lfp_theta_relpower"),
                     colors=RELPOWER_COLOURS)
    
    plot_hist_violin(loco_opt_rel, stat_opt_rel,
                     title="Relative theta power (GEVI): Locomotion vs Stationary",
                     xlabel="Theta / total (1–40 Hz)",
                     out_prefix=os.path.join(OUTPUT_ROOT, "opt_theta_relpower"),
                     colors=RELPOWER_COLOURS)


    # ===== Save summary =====
    summary = {
        "params": {
            "Fs_raw": Fs_raw, "TARGET_FS": TARGET_FS,
            "THETA_BAND": THETA_BAND, "DELTA_BAND": DELTA_BAND, "TOTAL_BAND": TOTAL_BAND,
            "WINDOW_SEC": WINDOW_SEC, "WINDOW_STEP_SEC": WINDOW_STEP_SEC,
            "SPEED_MIN_CM_S": SPEED_MIN_CM_S, "SPEED_MAX_CM_S": SPEED_MAX_CM_S,
            "THETA_DELTA_MIN": THETA_DELTA_MIN,
            "LFP_CHANNEL": LFP_CHANNEL,
            "LOCOMOTION_PARENT": LOCOMOTION_PARENT,
            "STATIONARY_PARENT": STATIONARY_PARENT
        },
        "counts": {
            "locomotion": {"LFP": int(loco_lfp_f.size), "Optical": int(loco_opt_f.size)},
            "awake_stationary": {"LFP": int(stat_lfp_f.size), "Optical": int(stat_opt_f.size)}
        },
        "stats": stats_dict
    }

    with open(os.path.join(OUTPUT_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUTPUT_ROOT, "summary.txt"), "w") as f:
        f.write("Theta metrics comparison: Locomotion vs Awake-stationary\n")
        f.write(json.dumps(summary, indent=2))

    print("\nDone.")
    print(f"CSV:      {csv_path}")
    print(f"Summary:  {os.path.join(OUTPUT_ROOT, 'summary.json')}")
    print(f"Plots:    *_theta_freq_*, *_theta_power_*, *_theta_relpower_* in {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
