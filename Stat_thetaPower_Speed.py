"""
Theta–speed relationship analysis (pooled windows + bin-averaged fit)

This script:
• Loads all SyncRecording* sessions and pools valid windows across sessions.
• Computes theta frequency (sub-bin via quadratic interpolation) and relative theta power from LFP and optical (zscore_raw).
• Excludes windows with mean speed < SPEED_MIN_CM_S and, optionally, with theta/delta ≤ 1.
• Runs regressions:
   1) vs speed (LFP freq, LFP rel power, Optical freq, Optical rel power)
   2) LFP vs Optical (freq; rel power)
• For each regression it plots BOTH:
   – Standard least-squares fit (red)
   – Bin-averaged fit over x (green), fitting a line to bin means
• Fixes spurious out-of-band/negative theta frequency by interpolating **inside the theta band only**, with clamping to [THETA_BAND[0], THETA_BAND[1]].
"""

import os
import glob
import json
from typing import List, Optional, Tuple
from scipy import odr
from scipy.stats import theilslopes, pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy import stats

# ==========================
# === USER CONFIGURATION ===
# ==========================
PARENT_DIR = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion'
OUTPUT_DIR = os.path.join(PARENT_DIR, "theta_speed_analysis_outputs")
Fs_raw = 10_000
TARGET_FS = 100
THETA_BAND = (4, 12.0)     # you can change to (6.0, 12.0) etc.
DELTA_BAND = (1.0, 4.0)
TOTAL_BAND = (1.0, 40.0)
LFP_CHANNEL = "LFP_1"
SPEED_MIN_CM_S = 0
SPEED_MAX_CM_S = 50.0   # <-- NEW: exclude windows with speed > 100 cm/s
USE_THETA_DELTA_GATE = True
WINDOW_SEC = 1
WINDOW_STEP_SEC = 0.5
BIN_WIDTH_FOR_GREEN = 0.5     # x-axis bin width for bin-averaged fit

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============
# I/O helpers
# =============

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

# ======================
# Analysis functionality
# ======================

def bin_average(x: np.ndarray, bin_sz: int) -> np.ndarray:
    n = (len(x) // bin_sz) * bin_sz
    if n == 0:
        return np.array([], dtype=float)
    return np.nanmean(np.asarray(x[:n], dtype=float).reshape(-1, bin_sz), axis=1)


def regress_and_plot_ls(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    dot_color: str = "k"   # <-- new input for scatter dot color
) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        print(f"Not enough points for {title}")
        return {"n": int(len(x))}

    slope, intercept, r, p, stderr = stats.linregress(x, y)

    plt.figure(figsize=(7, 5), dpi=140)
    plt.scatter(x, y, s=8, alpha=0.5, color=dot_color)  # <-- use custom color
    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, slope * xx + intercept, 'r', lw=2)

    # Larger font sizes
    plt.title(f"{title}\nR={r:.3f}, p={p:.3g}", fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {
        "n": int(len(x)),
        "r": float(r),
        "p": float(p),
        "slope": float(slope),
        "intercept": float(intercept)
    }




def process_session(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:

    bin_sz = int(round(Fs_raw / TARGET_FS))

    spd = np.clip(bin_average(df['speed'], bin_sz), 0, None)
    lfp = bin_average(df[LFP_CHANNEL], bin_sz)
    opt = bin_average(df['zscore_raw'], bin_sz)

    n = min(len(spd), len(lfp), len(opt))
    spd, lfp, opt = spd[:n], lfp[:n], opt[:n]

    # Windowing
    win_len = int(round(WINDOW_SEC * TARGET_FS))
    step = int(round(WINDOW_STEP_SEC * TARGET_FS))
    idx_starts = np.arange(0, n - win_len + 1, step)

    freqs = np.fft.rfftfreq(win_len, d=1.0/TARGET_FS)
    hann = get_window('hann', win_len)
    hann_norm = np.sum(hann**2)

    def psd(sig_win):
        sw = sig_win * hann
        spec = np.fft.rfft(sw)
        return (np.abs(spec)**2) / (hann_norm * TARGET_FS)

    # Precompute masks and theta-grid for interpolation inside the *theta band only*
    th_mask = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    de_mask = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    freq_th = freqs[th_mask]
    df_th = freq_th[1] - freq_th[0] if len(freq_th) > 1 else np.nan

    def theta_feats(sig):
        f_out, p_out, p_rel_out, de_out = [], [], [], []
        for s in idx_starts:
            seg = sig[s:s+win_len]
            p = psd(seg)
            p_th = p[th_mask]
            if p_th.size == 0 or not np.isfinite(p_th).any():
                f_out.append(np.nan); p_out.append(np.nan); p_rel_out.append(np.nan); de_out.append(np.nan)
                continue
            i_max = int(np.nanargmax(p_th))
            # Quadratic interpolation *within the theta sub-array*
            if len(p_th) >= 3 and 0 < i_max < (len(p_th) - 1) and np.isfinite(df_th):
                y1, y2, y3 = p_th[i_max-1], p_th[i_max], p_th[i_max+1]
                denom = (y1 - 2*y2 + y3)
                if denom == 0 or not np.isfinite(denom):
                    delta = 0.0
                else:
                    delta = 0.5 * (y1 - y3) / denom
                    # Guard against runaway parabolic offsets
                    if not np.isfinite(delta) or abs(delta) > 1.0:
                        delta = 0.0
                f_peak = freq_th[i_max] + delta * df_th
            else:
                f_peak = freq_th[i_max]
            # Clamp to theta band to avoid negative or out-of-range frequencies
            if not np.isfinite(f_peak):
                f_peak = np.nan
            else:
                f_peak = min(max(f_peak, THETA_BAND[0]), THETA_BAND[1])

            # Integrate powers
            dfreq = freqs[1] - freqs[0]
            p_theta = np.nansum(p_th) * dfreq
            p_tot = np.nansum(p[tot_mask]) * dfreq
            p_delta = np.nansum(p[de_mask]) * dfreq

            f_out.append(f_peak)
            p_out.append(p_theta)
            p_rel_out.append(p_theta/p_tot if p_tot > 0 else np.nan)
            de_out.append(p_delta)
        return np.array(f_out), np.array(p_out), np.array(p_rel_out), np.array(de_out)

    lfp_f, lfp_p, lfp_pr, lfp_de = theta_feats(lfp)
    opt_f, opt_p, opt_pr, opt_de = theta_feats(opt)

    spd_win = np.array([np.nanmean(spd[s:s+win_len]) for s in idx_starts])

    keep = (
    np.isfinite(spd_win) &
    (spd_win >= SPEED_MIN_CM_S) &
    (spd_win <= SPEED_MAX_CM_S)     # <-- NEW filter
    )
    if USE_THETA_DELTA_GATE:
        keep &= (lfp_p/(lfp_de + 1e-12) > 1.0)

    return spd_win[keep], lfp_f[keep], lfp_pr[keep], opt_f[keep], opt_pr[keep]

def regress_and_plot_pairwise(x, y, title, xlabel, ylabel, out_path,
                              trim_q=0.02,          # drop extreme 2% in both axes
                              bins2d=50,            # for density weights
                              label_fs=18, tick_fs=16, title_fs=20):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]; y = np.asarray(y)[mask]
    if len(x) < 3:
        print(f"Not enough points for {title}"); return {"n": int(len(x))}

    # ---- (0) optional quantile trim to ignore extreme edges ----------------
    xlo, xhi = np.quantile(x, [trim_q, 1-trim_q])
    ylo, yhi = np.quantile(y, [trim_q, 1-trim_q])
    keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    xt, yt = x[keep], y[keep]

    # ---- (1) OLS -----------------------------------------------------------
    slope_ols, intercept_ols, r, p, _ = stats.linregress(xt, yt)

    # ---- (2) Density-weighted OLS (follows the middle) ---------------------
    H, xe, ye = np.histogram2d(xt, yt, bins=bins2d)
    # map each point to its 2D bin count as weight
    ix = np.clip(np.searchsorted(xe, xt, side='right')-1, 0, H.shape[0]-1)
    iy = np.clip(np.searchsorted(ye, yt, side='right')-1, 0, H.shape[1]-1)
    w = H[ix, iy] + 1e-6
    slope_w, intercept_w = np.polyfit(xt, yt, 1, w=w)

    # ---- (3) Orthogonal (Deming/ODR) --------------------------------------
    def f(B, x): return B[0]*x + B[1]
    model = odr.Model(f)
    data  = odr.Data(xt, yt)                 # equal x/y errors ⇒ λ=1
    odr_inst = odr.ODR(data, model, beta0=[slope_ols, intercept_ols])
    out = odr_inst.run()
    slope_odr, intercept_odr = out.beta

    # ---- (4) Theil–Sen robust slope ---------------------------------------
    slope_ts, intercept_ts, _, _ = theilslopes(yt, xt, 0.95)

    # ---- plot --------------------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=140)
    plt.scatter(xt, yt, s=12, alpha=0.5)
    xx = np.linspace(xt.min(), xt.max(), 200)

    plt.plot(xx, slope_ols*xx + intercept_ols, 'r', lw=2, label=f'OLS')
    plt.plot(xx, slope_w *xx + intercept_w,   'g', lw=3, label='Density-weighted')
    plt.plot(xx, slope_odr*xx + intercept_odr, 'k--', lw=2.5, label='ODR (orthogonal)')
    plt.plot(xx, slope_ts*xx + intercept_ts,  color='#8e44ad', lw=2, ls=':', label='Theil–Sen')

    # identity line
    lo = min(xt.min(), yt.min()); hi = max(xt.max(), yt.max())
    plt.plot([lo, hi], [lo, hi], color='0.5', lw=1.5, ls='--', label='y = x')

    plt.title(f"{title}\nOLS: R={r:.3f}, p={p:.3g}", fontsize=title_fs)
    plt.xlabel(xlabel, fontsize=label_fs)
    plt.ylabel(ylabel, fontsize=label_fs)
    plt.xticks(fontsize=tick_fs); plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=12, frameon=False, ncol=2)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

    return {
        "n": int(len(xt)),
        "ols":   {"slope": float(slope_ols), "intercept": float(intercept_ols), "r": float(r), "p": float(p)},
        "wols":  {"slope": float(slope_w),   "intercept": float(intercept_w)},
        "odr":   {"slope": float(slope_odr), "intercept": float(intercept_odr)},
        "theil": {"slope": float(slope_ts),  "intercept": float(intercept_ts)},
        "trim":  {"x":[float(xlo), float(xhi)], "y":[float(ylo), float(yhi)]}
    }

def regress_and_plot_theilsen(x, y, title, xlabel, ylabel, out_path,
                              trim_q=0.02,              # quantile trim; set to 0 for none
                              show_identity=True,
                              label_fs=18, tick_fs=16, title_fs=20):
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 3:
        print(f"Not enough points for {title}")
        return {"n": int(x.size)}

    # Optional edge trim to de-emphasise sparse extremes
    if trim_q and 0 < trim_q < 0.5:
        xlo, xhi = np.quantile(x, [trim_q, 1-trim_q])
        ylo, yhi = np.quantile(y, [trim_q, 1-trim_q])
        keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
        x, y = x[keep], y[keep]
    else:
        xlo, xhi = float(np.min(x)), float(np.max(x))
        ylo, yhi = float(np.min(y)), float(np.max(y))

    # Robust slope/intercept
    slope, intercept, lo_slope, hi_slope = theilslopes(y, x, 0.95)

    # Correlation (for display only)
    r, p = pearsonr(x, y)

    # ---- plot ----
    plt.figure(figsize=(7, 5), dpi=140)
    plt.scatter(x, y, s=12, alpha=0.5)

    xx = np.linspace(x.min(), x.max(), 200)
    plt.plot(xx, slope*xx + intercept, color='#8e44ad', lw=2.5, ls=':', label='Theil–Sen')

    if show_identity:
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        plt.plot([lo, hi], [lo, hi], color='0.5', lw=1.5, ls='--', label='y = x')

    plt.title(f"{title}\nTheil–Sen slope = {slope:.3f} "
              f"(95% CI [{lo_slope:.3f}, {hi_slope:.3f}]); r={r:.3f}, p={p:.3g}",
              fontsize=title_fs)
    plt.xlabel(xlabel, fontsize=label_fs)
    plt.ylabel(ylabel, fontsize=label_fs)
    plt.xticks(fontsize=tick_fs); plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=12, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {
        "n": int(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci": [float(lo_slope), float(hi_slope)],
        "r": float(r), "p": float(p),
        "trim": {"x": [float(xlo), float(xhi)], "y": [float(ylo), float(yhi)]}
    }

def regress_and_plot_density_ols(x, y, title, xlabel, ylabel, out_path,
                                 trim_q=0.02,       # quantile trim on both axes (0.00–0.49). Use 0 to disable.
                                 bins2d=60,         # 2-D histogram bins for density weights
                                 show_identity=True,
                                 label_fs=18, tick_fs=16, title_fs=20,
                                 legend_fs=16,      # <-- NEW: bigger legend text
                                 five_xticks=False):
    """
    Density-weighted OLS (green) + standard OLS (red) + identity line (grey).
    Returns both fit params and weighted r,p.
    """
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3:
        print(f"Not enough points for {title}")
        return {"n": int(x.size)}

    # --- optional edge trim to de-emphasise sparse extremes -----------------
    if trim_q and 0 < trim_q < 0.5:
        xlo, xhi = np.quantile(x, [trim_q, 1-trim_q])
        ylo, yhi = np.quantile(y, [trim_q, 1-trim_q])
        keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
        x, y = x[keep], y[keep]
    else:
        xlo, xhi = float(np.min(x)), float(np.max(x))
        ylo, yhi = float(np.min(y)), float(np.max(y))

    # --- density weights from a 2-D histogram -------------------------------
    H, xe, ye = np.histogram2d(x, y, bins=bins2d)
    ix = np.clip(np.searchsorted(xe, x, side='right') - 1, 0, H.shape[0]-1)
    iy = np.clip(np.searchsorted(ye, y, side='right') - 1, 0, H.shape[1]-1)
    w  = H[ix, iy].astype(float) + 1e-6  # avoid zeros

    # --- weighted OLS via weighted least squares ----------------------------
    slope_w, intercept_w = np.polyfit(x, y, 1, w=w)

    # --- plain (unweighted) OLS --------------------------------------------
    slope_ols, intercept_ols, r_ols, p_ols, _ = stats.linregress(x, y)

    # --- weighted correlation (for display) ---------------------------------
    wsum = w.sum()
    neff = (wsum**2) / (np.sum(w**2) + 1e-12)
    mx = np.sum(w*x)/wsum; my = np.sum(w*y)/wsum
    cov_xy = np.sum(w*(x-mx)*(y-my))/wsum
    sx = np.sqrt(np.sum(w*(x-mx)**2)/wsum); sy = np.sqrt(np.sum(w*(y-my)**2)/wsum)
    r_w = (cov_xy / (sx*sy)) if sx>0 and sy>0 else np.nan
    df = max(int(neff - 2), 1)
    t = r_w * np.sqrt(df / max(1e-12, 1 - r_w**2))
    p_w = 2 * (1 - stats.t.cdf(abs(t), df))

    # --- plot ----------------------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=140)
    plt.scatter(x, y, s=12, alpha=0.5)

    xx = np.linspace(x.min(), x.max(), 200)
    # density-weighted fit (green)
    plt.plot(xx, slope_w*xx + intercept_w, color='g', lw=3, label='Density-weighted OLS')
    # plain OLS (red)
    plt.plot(xx, slope_ols*xx + intercept_ols, 'r', lw=2, label='OLS')

    if show_identity:
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        plt.plot([lo, hi], [lo, hi], color='0.5', lw=1.5, ls='--', label='y = x')

    plt.title(f"{title}\nweighted r={r_w:.3f}, p={p_w:.3g}", fontsize=title_fs)
    plt.xlabel(xlabel, fontsize=label_fs)
    plt.ylabel(ylabel, fontsize=label_fs)
    plt.xticks(fontsize=tick_fs); plt.yticks(fontsize=tick_fs)
    plt.legend(fontsize=legend_fs, frameon=False)  # <-- larger legend
    plt.tight_layout()

    if five_xticks:
        ax = plt.gca()
        lo, hi = ax.get_xlim()
        ticks = np.linspace(lo, hi, 5)
        ticks[np.isclose(ticks, 0.0)] = 0.0
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=tick_fs)

    plt.savefig(out_path)
    plt.close()

    return {
        "n": int(x.size),
        "wls": {"slope": float(slope_w), "intercept": float(intercept_w),
                "r_weighted": float(r_w), "p_weighted": float(p_w),
                "neff": float(neff)},
        "ols": {"slope": float(slope_ols), "intercept": float(intercept_ols),
                "r": float(r_ols), "p": float(p_ols)},
        "trim": {"x": [float(xlo), float(xhi)], "y": [float(ylo), float(yhi)]},
        "bins2d": int(bins2d)
    }
# =====
# Main
# =====

def main():
    sessions = find_syncrecording_folders(PARENT_DIR)
    all_speed, all_lfp_f, all_lfp_pr, all_opt_f, all_opt_pr = [], [], [], [], []

    for sess in sessions:
        df = load_pickle_if_exists(sess)
        if df is None or df.empty:
            continue
        spd, lfpf, lfppr, optf, optpr = process_session(df)
        if len(spd) == 0:
            continue
        all_speed.append(spd)
        all_lfp_f.append(lfpf)
        all_lfp_pr.append(lfppr)
        all_opt_f.append(optf)
        all_opt_pr.append(optpr)

    if not all_speed:
        print("No valid windows across sessions.")
        return

    all_speed = np.concatenate(all_speed)
    all_lfp_f = np.concatenate(all_lfp_f)
    all_lfp_pr = np.concatenate(all_lfp_pr)
    all_opt_f = np.concatenate(all_opt_f)
    all_opt_pr = np.concatenate(all_opt_pr)

    results = {}
    results['lfp_freq_vs_speed'] = regress_and_plot_ls(
        all_speed, all_lfp_f, "LFP theta frequency vs speed", "Speed (cm/s)", "Theta freq (Hz)",
        os.path.join(OUTPUT_DIR, "pooled_lfp_theta_freq_vs_speed.png"), dot_color="grey"
    )
    results['lfp_power_vs_speed'] = regress_and_plot_ls(
        all_speed, all_lfp_pr, "LFP theta relative power vs speed", "Speed (cm/s)", "Theta rel. power",
        os.path.join(OUTPUT_DIR, "pooled_lfp_theta_relpower_vs_speed.png"), dot_color="grey"
    )
    results['opt_freq_vs_speed'] = regress_and_plot_ls(
        all_speed, all_opt_f, "Optical theta frequency vs speed", "Speed (cm/s)", "Theta freq (Hz)",
        os.path.join(OUTPUT_DIR, "pooled_opt_theta_freq_vs_speed.png"), dot_color="green"
    )
    results['opt_power_vs_speed'] = regress_and_plot_ls(
        all_speed, all_opt_pr, "Optical theta relative power vs speed", "Speed (cm/s)", "Theta rel. power",
        os.path.join(OUTPUT_DIR, "pooled_opt_theta_relpower_vs_speed.png"), dot_color="green"
    )
    results['lfp_vs_opt_freq'] = regress_and_plot_density_ols(
    all_lfp_f, all_opt_f,
    "LFP vs Optical theta frequency",
    "LFP theta freq (Hz)", "Optical theta freq (Hz)",
    os.path.join(OUTPUT_DIR, "pooled_lfp_vs_opt_theta_freq.png"),
    trim_q=0.02, bins2d=60, show_identity=True, five_xticks=False
    )

    results['lfp_vs_opt_power'] = regress_and_plot_ls(
        all_lfp_pr, all_opt_pr, "LFP vs Optical theta relative power", "LFP theta rel. power", "Optical theta rel. power",
        os.path.join(OUTPUT_DIR, "pooled_lfp_vs_opt_theta_relpower.png"), dot_color="tab:blue"
    )

    with open(os.path.join(OUTPUT_DIR, "pooled_results.json"), "w") as jf, \
         open(os.path.join(OUTPUT_DIR, "pooled_results.txt"), "w") as tf:
        json.dump(results, jf, indent=2)
        for k, v in results.items():
            tf.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
