# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 22:20:37 2026

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Compare group-level theta–speed slopes for two SPEED_MIN thresholds:
  - SPEED_MIN = 0  (includes near-still windows)
  - SPEED_MIN = 1  (excludes near-still windows)

Locomotion sessions only (ALocomotion). Windows are overlapping, so:
- Primary inference uses animal-intercept model with SWEEP-CLUSTER robust SE
- Also reports bootstrap CI for slope differences (cluster bootstrap by sweep)

Outputs:
- theta_speed_windows_speedmin_<X>.csv
- speed_effects_speedmin_comparison.csv
- theta_speed_slope_comparison.png/pdf (panel of slope points ± CI)
- theta_speed_panels_speedmin_<X>.png/pdf (scatter panels, subsampled)

Author: yifan + ChatGPT
"""

import os
import glob
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy import stats


# ==========================
# USER CONFIGURATION
# ==========================

ANIMALS = [
    {"animal_id": "1765508",
     "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ALocomotion",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1844609",
     "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ALocomotion",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1881363",
     "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ALocomotion",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1851545",
     "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1851545_WT_Jedi2p_dis\ALocomotion",
     "lfp_channel": "LFP_1"},
]

OUTPUT_ROOT = r"G:\2025_ATLAS_SPAD\AcrossAnimal\theta_speed_compare_speedmin0_vs1"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Sampling / bands
Fs_raw      = 10_000
TARGET_FS   = 100
THETA_BAND  = (4.0, 12.0)
DELTA_BAND  = (1.0, 4.0)
TOTAL_BAND  = (1.0, 40.0)

# Windowing
WINDOW_SEC      = 1.0
WINDOW_STEP_SEC = 0.5

# Speed thresholds to compare
SPEED_MIN_LIST = [0.0, 1.0]
SPEED_MAX_CM_S = 50.0

# Theta-rich gating (optional, recommended)
USE_THETA_DELTA_GATE = False
THETA_DELTA_MIN = 0.5

# Optional speed transform (rarely needed; keep False for interpretability)
USE_LOG1P_SPEED = False

# Plot subsampling (stats use full data)
MAX_POINTS_PER_ANIMAL_FOR_PLOT = 12000

DPI = 180


# ==========================
# HELPERS
# ==========================

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
    x = np.asarray(x, dtype=float)
    n = (len(x) // bin_sz) * bin_sz
    if n <= 0:
        return np.array([], dtype=float)
    return np.nanmean(x[:n].reshape(-1, bin_sz), axis=1)

def p_to_stars(p: float) -> str:
    if p is None or not np.isfinite(p): return "n/a"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def per_animal_slopes_simple(df: pd.DataFrame,
                            ycol: str,
                            xcol: str = "speed_fit",
                            animal_col: str = "animal_id",
                            min_n: int = 200) -> pd.DataFrame:
    """
    Per-animal slope from simple OLS within each animal.
    Intended for visualisation (dots), not primary inference.
    """
    rows = []
    for aid, da in df[[animal_col, xcol, ycol]].dropna().groupby(animal_col):
        x = da[xcol].to_numpy(dtype=float)
        y = da[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < min_n:
            continue
        lr = stats.linregress(x, y)
        rows.append({
            "animal_id": str(aid),
            "n": int(x.size),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r": float(lr.rvalue),
            "p": float(lr.pvalue),
            "stderr": float(lr.stderr),
        })
    return pd.DataFrame(rows)

# ==========================
# SPECTRAL FEATURES
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
                                      df_theta: float
                                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-window arrays:
      theta_peak_freq [Hz], theta_power, delta_power, total_power
    """
    hann = get_window("hann", win_len)
    hann_norm = np.sum(hann**2)

    def psd(seg):
        sw = seg * hann
        spec = np.fft.rfft(sw)
        return (np.abs(spec) ** 2) / (hann_norm * target_fs)

    idx_starts = np.arange(0, len(sig) - win_len + 1, step)
    f_out, p_theta_out, p_delta_out, p_total_out = [], [], [], []

    dfreq = freqs[1] - freqs[0] if len(freqs) > 1 else np.nan
    integ = dfreq if np.isfinite(dfreq) else 1.0

    for s in idx_starts:
        seg = sig[s:s+win_len]
        Pxx = psd(seg)

        p_th = Pxx[theta_mask]
        if p_th.size == 0 or not np.isfinite(p_th).any():
            f_out.append(np.nan); p_theta_out.append(np.nan); p_delta_out.append(np.nan); p_total_out.append(np.nan)
            continue

        i_max = int(np.nanargmax(p_th))

        # Quadratic interpolation within theta band only
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

        # Clamp
        if np.isfinite(f_peak):
            f_peak = min(max(f_peak, THETA_BAND[0]), THETA_BAND[1])
        else:
            f_peak = np.nan

        p_theta = np.nansum(p_th) * integ
        p_delta = np.nansum(Pxx[delta_mask]) * integ
        p_total = np.nansum(Pxx[total_mask]) * integ

        f_out.append(f_peak)
        p_theta_out.append(p_theta)
        p_delta_out.append(p_delta)
        p_total_out.append(p_total)

    return (np.array(f_out), np.array(p_theta_out), np.array(p_delta_out), np.array(p_total_out))


def extract_theta_speed_windows(df: pd.DataFrame,
                               lfp_channel: str,
                               speed_min: float,
                               speed_max: float
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-window arrays:
      speed_win, lfp_freq, lfp_rel_power, opt_freq, opt_rel_power

    Key behaviour:
      - If speed_min == 0: DO NOT apply any lower-bound speed filtering.
        This keeps all windows from the full sweep (still/near-still included).
      - If speed_min > 0: apply speed_win >= speed_min.
      - Theta/delta gating is controlled by USE_THETA_DELTA_GATE (set False to disable).
    """
    bin_sz = int(round(Fs_raw / TARGET_FS))

    if "speed" not in df.columns:
        raise KeyError("speed not found.")
    if lfp_channel not in df.columns:
        raise KeyError(f"{lfp_channel} not found.")
    if "zscore_raw" not in df.columns:
        raise KeyError("zscore_raw not found.")

    spd = np.clip(bin_average(df["speed"].to_numpy(), bin_sz), 0, None)
    lfp = bin_average(df[lfp_channel].to_numpy(), bin_sz)
    opt = bin_average(df["zscore_raw"].to_numpy(), bin_sz)

    n = min(len(spd), len(lfp), len(opt))
    spd, lfp, opt = spd[:n], lfp[:n], opt[:n]

    win_len = int(round(WINDOW_SEC * TARGET_FS))
    step    = int(round(WINDOW_STEP_SEC * TARGET_FS))
    if n < win_len:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))

    freqs = np.fft.rfftfreq(win_len, d=1.0 / TARGET_FS)
    th_mask  = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    de_mask  = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])

    freq_th = freqs[th_mask]
    df_th = (freq_th[1] - freq_th[0]) if len(freq_th) > 1 else np.nan

    lfp_f, lfp_pth, lfp_pde, lfp_ptot = compute_theta_features_per_window(
        lfp, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )
    opt_f, opt_pth, opt_pde, opt_ptot = compute_theta_features_per_window(
        opt, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )

    idx_starts = np.arange(0, n - win_len + 1, step)
    spd_win = np.array([np.nanmean(spd[s:s+win_len]) for s in idx_starts])

    lfp_rel = lfp_pth / (lfp_ptot + 1e-12)
    opt_rel = opt_pth / (opt_ptot + 1e-12)

    # --- base validity ---
    keep = (
        np.isfinite(spd_win) &
        np.isfinite(lfp_f) & np.isfinite(opt_f) &
        np.isfinite(lfp_rel) & np.isfinite(opt_rel)
    )

    # --- speed gating ---
    # Always apply speed_max to guard against tracking artefacts.
    keep &= (spd_win <= speed_max)

    # ONLY apply speed_min if speed_min > 0 (so speed_min=0 keeps full sweep).
    if speed_min > 0:
        keep &= (spd_win >= speed_min)

    # --- theta/delta gating (disabled if USE_THETA_DELTA_GATE=False) ---
    if USE_THETA_DELTA_GATE:
        td = lfp_pth / (lfp_pde + 1e-12)
        keep &= np.isfinite(td) & (td > THETA_DELTA_MIN)

    return spd_win[keep], lfp_f[keep], lfp_rel[keep], opt_f[keep], opt_rel[keep]

# ==========================
# GROUP MODEL + CLUSTER ROBUST SE
# ==========================

def fit_common_slope_animal_intercepts_clustered(df: pd.DataFrame,
                                                 xcol: str,
                                                 ycol: str,
                                                 animal_col: str = "animal_id",
                                                 cluster_col: str = "sweep_id") -> Dict[str, float]:
    """
    Fits y = a_animal + beta*x + e via within-animal centring (animal fixed effects).
    Computes sweep-cluster robust SE for beta.

    Returns beta, cluster_se, cluster_p, ci95, n_obs, n_animals, n_clusters.
    """
    d = df[[animal_col, cluster_col, xcol, ycol]].dropna().copy()
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])].copy()
    if len(d) < 10:
        return {"beta": np.nan, "cluster_se": np.nan, "cluster_p": np.nan}

    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    g = d[animal_col].to_numpy()
    cl = d[cluster_col].to_numpy()

    # within-animal centring
    x_c = np.empty_like(x)
    y_c = np.empty_like(y)
    for aid in np.unique(g):
        m = (g == aid)
        x_c[m] = x[m] - np.mean(x[m])
        y_c[m] = y[m] - np.mean(y[m])

    Sxx = np.sum(x_c * x_c)
    Sxy = np.sum(x_c * y_c)
    if Sxx <= 0:
        return {"beta": np.nan, "cluster_se": np.nan, "cluster_p": np.nan}

    beta = Sxy / Sxx

    # residuals using animal intercepts
    yhat = np.empty_like(y)
    for aid in np.unique(g):
        m = (g == aid)
        ybar = np.mean(y[m])
        xbar = np.mean(x[m])
        yhat[m] = ybar + beta * (x[m] - xbar)
    resid = y - yhat

    # cluster robust variance of beta
    clusters = np.unique(cl)
    G = len(clusters)
    n_obs = len(y)
    n_animals = len(np.unique(g))

    if G < 3:
        return {"beta": float(beta), "cluster_se": np.nan, "cluster_p": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan,
                "n_obs": int(n_obs), "n_animals": int(n_animals), "n_clusters": int(G)}

    s = 0.0
    for cid in clusters:
        m = (cl == cid)
        s += (np.sum(x_c[m] * resid[m]) ** 2)

    var = s / (Sxx ** 2)

    # small-sample correction (helps when #clusters is small)
    k = n_animals + 1
    if (G > 1) and (n_obs > k):
        var *= (G / (G - 1)) * ((n_obs - 1) / (n_obs - k))

    se = float(np.sqrt(var))
    df_cl = max(G - 1, 1)
    t = beta / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=df_cl))

    ci_low = float(beta - 1.96 * se)
    ci_high = float(beta + 1.96 * se)

    return {
        "beta": float(beta),
        "cluster_se": float(se),
        "cluster_p": float(p),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n_obs": int(n_obs),
        "n_animals": int(n_animals),
        "n_clusters": int(G)
    }


def cluster_bootstrap_delta_beta(df_full: pd.DataFrame,
                                ycol: str,
                                xcol: str,
                                speed_col: str,
                                speed_min_a: float,
                                speed_min_b: float,
                                n_boot: int = 2000,
                                seed: int = 0) -> Dict[str, float]:
    """
    Cluster bootstrap by sweep_id:
      - resample sweeps with replacement
      - for each bootstrap sample, compute beta for threshold A and B
      - return CI for delta beta = beta(A) - beta(B)

    df_full should already be filtered to locomotion sessions, theta/delta gate, speed<=max etc.
    """
    rng = np.random.default_rng(seed)
    sweeps = df_full["sweep_id"].unique().tolist()
    if len(sweeps) < 3:
        return {"delta_ci_low": np.nan, "delta_ci_high": np.nan, "delta_p": np.nan}

    betas_a = []
    betas_b = []
    deltas = []

    for _ in range(n_boot):
        sampled = rng.choice(sweeps, size=len(sweeps), replace=True)
        db = pd.concat([df_full[df_full["sweep_id"] == sid] for sid in sampled], ignore_index=True)

        da = db[db[speed_col] >= speed_min_a].copy()
        db2 = db[db[speed_col] >= speed_min_b].copy()

        ra = fit_common_slope_animal_intercepts_clustered(da, xcol=xcol, ycol=ycol)
        rb = fit_common_slope_animal_intercepts_clustered(db2, xcol=xcol, ycol=ycol)

        ba = ra.get("beta", np.nan)
        bb = rb.get("beta", np.nan)
        if np.isfinite(ba) and np.isfinite(bb):
            betas_a.append(ba)
            betas_b.append(bb)
            deltas.append(ba - bb)

    deltas = np.asarray(deltas, dtype=float)
    if deltas.size < 200:
        return {"delta_ci_low": np.nan, "delta_ci_high": np.nan, "delta_p": np.nan}

    lo, hi = np.quantile(deltas, [0.025, 0.975])

    # two-sided bootstrap p-value: how often delta crosses 0
    p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))

    return {"delta_ci_low": float(lo), "delta_ci_high": float(hi), "delta_p": float(p)}


# ==========================
# PLOTTING
# ==========================

def plot_scatter_panels(df_plot: pd.DataFrame,
                        title_prefix: str,
                        outpath: str):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=DPI)
    axes = axes.flatten()

    panels = [
        ("lfp_theta_freq", "LFP theta frequency vs speed", "Theta peak frequency (Hz)"),
        ("lfp_theta_rel_power", "LFP relative theta power vs speed", "Relative theta power"),
        ("opt_theta_freq", "Optical theta frequency vs speed", "Theta peak frequency (Hz)"),
        ("opt_theta_rel_power", "Optical relative theta power vs speed", "Relative theta power"),
    ]

    for ax, (ycol, t, yl) in zip(axes, panels):
        for aid, da in df_plot.groupby("animal_id"):
            ax.scatter(da["speed"], da[ycol], s=8, alpha=0.35, label=str(aid))
        ax.set_title(f"{title_prefix}\n{t}", fontsize=12)
        ax.set_xlabel("Speed (cm/s)")
        ax.set_ylabel(yl)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=9)

    plt.tight_layout()
    fig.savefig(outpath + ".png")
    fig.savefig(outpath + ".pdf")
    plt.close(fig)


def plot_slope_comparison(summary: pd.DataFrame,
                          per_animal: Dict[Tuple[str, float], pd.DataFrame],
                          outpath: str,
                          seed: int = 0):
    """
    For each metric:
      - big dots + CI = group slope beta (cluster robust)
      - small dots = per-animal slopes
      - thin lines connect per-animal slopes across thresholds when available
    """
    rng = np.random.default_rng(seed)

    order = ["lfp_theta_freq", "lfp_theta_rel_power", "opt_theta_freq", "opt_theta_rel_power"]
    name_map = {
        "lfp_theta_freq": "LFP theta frequency",
        "lfp_theta_rel_power": "LFP relative theta power",
        "opt_theta_freq": "Optical theta frequency",
        "opt_theta_rel_power": "Optical relative theta power",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), dpi=DPI)
    axes = axes.flatten()

    for ax, m in zip(axes, order):
        d = summary[summary["metric"] == m].sort_values("speed_min")
        if d.empty:
            ax.axis("off")
            continue

        # ---- group points + CI ----
        x = d["speed_min"].to_numpy(dtype=float)  # [0, 1]
        y = d["beta"].to_numpy(dtype=float)
        lo = d["ci95_low"].to_numpy(dtype=float)
        hi = d["ci95_high"].to_numpy(dtype=float)
        p  = d["cluster_p"].to_numpy(dtype=float)

        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt='o', capsize=4, linewidth=1.5)
        ax.plot(x, y, linewidth=1.5)

        # annotate group significance (ns/*/** etc.)
        for xi, yi, pi in zip(x, y, p):
            ax.text(xi + 0.03, yi, f"{p_to_stars(pi)}", va="center", fontsize=11)

        # ---- per-animal dots (and optional connecting lines) ----
        # Collect per-animal slopes at each threshold
        df0 = per_animal.get((m, 0.0), pd.DataFrame())
        df1 = per_animal.get((m, 1.0), pd.DataFrame())

        # Build a union of animal IDs (some animals may be missing at one threshold)
        aids = sorted(set(df0.get("animal_id", [])) | set(df1.get("animal_id", [])))

        # map slopes for fast lookup
        s0 = dict(zip(df0.get("animal_id", []), df0.get("slope", [])))
        s1 = dict(zip(df1.get("animal_id", []), df1.get("slope", [])))

        # jitter so points don’t overlap perfectly
        jitter0 = rng.uniform(-0.02, 0.02, size=len(aids))
        jitter1 = rng.uniform(-0.02, 0.02, size=len(aids))

        for i, aid in enumerate(aids):
            y0 = s0.get(aid, np.nan)
            y1 = s1.get(aid, np.nan)

            # dots
            if np.isfinite(y0):
                ax.scatter(0.0 + jitter0[i], y0, s=28, alpha=0.8, zorder=3)
            if np.isfinite(y1):
                ax.scatter(1.0 + jitter1[i], y1, s=28, alpha=0.8, zorder=3)

            # connect if both exist
            if np.isfinite(y0) and np.isfinite(y1):
                ax.plot([0.0 + jitter0[i], 1.0 + jitter1[i]], [y0, y1],
                        linewidth=1.0, alpha=0.5, zorder=2)

        ax.axhline(0, color="0.6", linewidth=1)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["speed≥0", "speed≥1"])
        ax.set_title(name_map.get(m, m), fontsize=12)
        ax.set_ylabel("Slope β (per cm/s)", fontsize=11)

    plt.tight_layout()
    fig.savefig(outpath + ".png")
    fig.savefig(outpath + ".pdf")
    plt.close(fig)

# ==========================
# DATA COLLECTION
# ==========================

def collect_windows_for_speedmin(speed_min: float) -> pd.DataFrame:
    rows = []
    for a in ANIMALS:
        animal_id = a["animal_id"]
        parent = a["loco_parent"]
        lfp_channel = a.get("lfp_channel", "LFP_1")

        sessions = find_syncrecording_folders(parent)
        if len(sessions) == 0:
            print(f"[{animal_id}] No SyncRecording* in {parent}")
            continue

        for sess in sessions:
            df = load_pickle_if_exists(sess)
            if df is None or df.empty:
                continue

            try:
                spd, lfpf, lfp_rel, optf, opt_rel = extract_theta_speed_windows(
                    df, lfp_channel=lfp_channel, speed_min=speed_min, speed_max=SPEED_MAX_CM_S
                )
            except Exception as e:
                print(f"[{animal_id}] error in {sess}: {e}")
                continue

            if spd.size == 0:
                continue

            sweep_id = os.path.basename(sess)
            for i in range(spd.size):
                rows.append({
                    "animal_id": animal_id,
                    "sweep_id": sweep_id,
                    "speed": float(spd[i]),
                    "lfp_theta_freq": float(lfpf[i]),
                    "lfp_theta_rel_power": float(lfp_rel[i]),
                    "opt_theta_freq": float(optf[i]),
                    "opt_theta_rel_power": float(opt_rel[i]),
                })

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    # speed for fit
    if USE_LOG1P_SPEED:
        d["speed_fit"] = np.log1p(d["speed"].to_numpy(dtype=float))
    else:
        d["speed_fit"] = d["speed"].to_numpy(dtype=float)

    return d


# ==========================
# MAIN
# ==========================

def main():
    metrics = [
        "lfp_theta_freq",
        "lfp_theta_rel_power",
        "opt_theta_freq",
        "opt_theta_rel_power",
    ]

    # Run both thresholds
    dfs = {}
    summaries = []
    per_animal = {}  # key: (metric, speed_min) -> df slopes

    for sm in SPEED_MIN_LIST:
        d = collect_windows_for_speedmin(sm)
        if d.empty:
            raise RuntimeError(f"No data extracted for SPEED_MIN={sm}. Check gates/paths.")
        dfs[sm] = d

        out_csv = os.path.join(OUTPUT_ROOT, f"theta_speed_windows_speedmin_{sm:g}.csv")
        d.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}  (n={len(d)} windows)")
        
        for ycol in metrics:
            pa = per_animal_slopes_simple(d, ycol=ycol, xcol="speed_fit", min_n=200)
            per_animal[(ycol, float(sm))] = pa
            pa.to_csv(os.path.join(OUTPUT_ROOT, f"per_animal_slopes_{ycol}_speedmin_{sm:g}.csv"), index=False)
        

        # scatter panels (subsampled)
        dplot_parts = []
        for aid, da in d.groupby("animal_id"):
            if len(da) > MAX_POINTS_PER_ANIMAL_FOR_PLOT:
                da = da.sample(n=MAX_POINTS_PER_ANIMAL_FOR_PLOT, random_state=0)
            dplot_parts.append(da)
        dplot = pd.concat(dplot_parts, ignore_index=True)

        plot_scatter_panels(
            dplot,
            title_prefix=f"Locomotion sessions only | SPEED_MIN={sm:g} cm/s | theta/delta gate={USE_THETA_DELTA_GATE}",
            outpath=os.path.join(OUTPUT_ROOT, f"theta_speed_panels_speedmin_{sm:g}")
        )

        # group slopes
        for ycol in metrics:
            res = fit_common_slope_animal_intercepts_clustered(
                d, xcol="speed_fit", ycol=ycol, animal_col="animal_id", cluster_col="sweep_id"
            )
            summaries.append({
                "metric": ycol,
                "speed_min": sm,
                **res,
                "stars": p_to_stars(res.get("cluster_p", np.nan)),
                "speed_max": SPEED_MAX_CM_S,
                "use_log1p_speed": USE_LOG1P_SPEED,
                "use_theta_delta_gate": USE_THETA_DELTA_GATE,
                "theta_delta_min": THETA_DELTA_MIN
            })

    summary = pd.DataFrame(summaries)

    # Compare slopes (delta) using cluster bootstrap on the SPEED_MIN=0 dataset
    # (same sweeps resampled; threshold applied within each replicate)
    d0 = dfs[min(SPEED_MIN_LIST)]
    for ycol in metrics:
        boot = cluster_bootstrap_delta_beta(
            df_full=d0,
            ycol=ycol,
            xcol="speed_fit",
            speed_col="speed",
            speed_min_a=0.0,
            speed_min_b=1.0,
            n_boot=2000,
            seed=0
        )
        summary.loc[summary["metric"] == ycol, "delta_beta_ci95_low"] = np.nan
        summary.loc[summary["metric"] == ycol, "delta_beta_ci95_high"] = np.nan
        summary.loc[summary["metric"] == ycol, "delta_beta_p"] = np.nan
        # attach delta results to BOTH rows of that metric for convenience
        summary.loc[summary["metric"] == ycol, "delta_beta_ci95_low"] = boot["delta_ci_low"]
        summary.loc[summary["metric"] == ycol, "delta_beta_ci95_high"] = boot["delta_ci_high"]
        summary.loc[summary["metric"] == ycol, "delta_beta_p"] = boot["delta_p"]
        summary.loc[summary["metric"] == ycol, "delta_beta_stars"] = p_to_stars(boot["delta_p"])

    out_summary = os.path.join(OUTPUT_ROOT, "speed_effects_speedmin_comparison.csv")
    summary.to_csv(out_summary, index=False)
    print(f"Saved summary: {out_summary}")

    # Plot slope comparison
    plot_slope_comparison(summary, per_animal, outpath=os.path.join(OUTPUT_ROOT, "theta_speed_slope_comparison"))


    # Print a readable console summary
    print("\n=== Interpretation guide ===")
    print("beta = slope per 1 cm/s. Compare beta(speed>=0) vs beta(speed>=1).")
    print("cluster_p uses sweep-cluster robust SE (recommended).")
    print("delta_beta_p tests whether including 0 changes the slope (cluster bootstrap).")

    for ycol in metrics:
        s0 = summary[(summary.metric == ycol) & (summary.speed_min == 0.0)].iloc[0]
        s1 = summary[(summary.metric == ycol) & (summary.speed_min == 1.0)].iloc[0]
        print(f"\n[{ycol}]")
        print(f"  speed>=0: beta={s0.beta:.4g} CI[{s0.ci95_low:.4g},{s0.ci95_high:.4g}] "
              f"p={s0.cluster_p:.3g} {s0.stars}")
        print(f"  speed>=1: beta={s1.beta:.4g} CI[{s1.ci95_low:.4g},{s1.ci95_high:.4g}] "
              f"p={s1.cluster_p:.3g} {s1.stars}")
        print(f"  delta (>=0 minus >=1): CI[{s0.delta_beta_ci95_low:.4g},{s0.delta_beta_ci95_high:.4g}] "
              f"p={s0.delta_beta_p:.3g} {s0.delta_beta_stars}")

    print("\nDone. Outputs in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
