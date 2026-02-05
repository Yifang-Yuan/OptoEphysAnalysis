# -*- coding: utf-8 -*-
"""
Group-level theta metrics analysis: locomotion vs awake-stationary
- Avoids pseudoreplication by summarising overlapping windows to ONE value per sweep.
- Primary inference: animal is the experimental unit (paired across conditions).
- Also reports within-animal variability across sweeps.
- Keeps: theta peak frequency + relative theta power
- Removes: absolute theta power / log power

Author: yifan + ChatGPT
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

# Optional mixed model
try:
    import statsmodels.formula.api as smf
    HAS_MIXEDLM = True
except Exception:
    HAS_MIXEDLM = False


# ==========================
# === USER CONFIGURATION ===
# ==========================

ANIMALS = [
    {"animal_id": "1765508", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\AwakeStationary",
     "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ASleepREM",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1844609", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\AwakeStationary",
     "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ASleepREM",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1881363", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\AwakeStationary",
     "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ASleepREM",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1887933", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\AwakeStationary",
     "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ASleepREM",
     "lfp_channel": "LFP_3"},
]

MULTI_ANIMAL_ROOT = r"G:\2025_ATLAS_SPAD\AcrossAnimal"
OUTPUT_ROOT = os.path.join(MULTI_ANIMAL_ROOT, "theta_group_outputs_freq_relpower")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Signal / analysis parameters
Fs_raw      = 10_000
TARGET_FS   = 100
THETA_BAND  = (4.0, 12.0)
DELTA_BAND  = (1.0, 4.0)
TOTAL_BAND  = (1.0, 40.0)

# Windowing
WINDOW_SEC      = 1.0
WINDOW_STEP_SEC = 0.5

# Gating
SPEED_MIN_CM_S   = 0.5
SPEED_MAX_CM_S   = 50.0
THETA_DELTA_MIN  = 1.0  # theta-rich gating computed on LFP

# Plot settings
DPI = 160


# ==========================
# === Helper functions    ===
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
    n = (len(x) // bin_sz) * bin_sz
    if n <= 0:
        return np.array([], dtype=float)
    return np.nanmean(np.asarray(x[:n], dtype=float).reshape(-1, bin_sz), axis=1)

def _nanmedian(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.nanmedian(x)) if x.size else np.nan

def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.nanmean(x)) if x.size else np.nan


# ==========================
# === Spectral features   ===
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
    Return per-window arrays:
      theta_peak_freq [Hz],
      theta_band_power,
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
                                     lfp_channel: str,
                                     require_speed: bool,
                                     speed_min: float,
                                     speed_max: float,
                                     theta_delta_min: float
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns per-window arrays (kept windows):
      (lfp_freq, opt_freq, lfp_rel_power, opt_rel_power)

    Windows overlap; inference must NOT use them directly.
    We summarise per sweep.
    """
    bin_sz = int(round(Fs_raw / TARGET_FS))

    # Speed
    if 'speed' in df.columns:
        spd = np.clip(bin_average(df['speed'].to_numpy(), bin_sz), 0, None)
    else:
        spd = None

    # Signals
    if lfp_channel not in df.columns:
        raise KeyError(f"{lfp_channel} not found in dataframe columns.")
    lfp = bin_average(df[lfp_channel].to_numpy(), bin_sz)

    if 'zscore_raw' not in df.columns:
        raise KeyError("zscore_raw not found in dataframe columns.")
    opt = bin_average(df['zscore_raw'].to_numpy(), bin_sz)

    n = min(len(lfp), len(opt))
    if spd is not None:
        n = min(n, len(spd))
        spd = spd[:n]
    lfp = lfp[:n]
    opt = opt[:n]

    win_len = int(round(WINDOW_SEC * TARGET_FS))
    step    = int(round(WINDOW_STEP_SEC * TARGET_FS))
    if n < win_len:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    freqs   = np.fft.rfftfreq(win_len, d=1.0 / TARGET_FS)
    th_mask  = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    de_mask  = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])

    freq_th = freqs[th_mask]
    df_th = (freq_th[1] - freq_th[0]) if len(freq_th) > 1 else np.nan

    # Per-window features for LFP and Optical
    lfp_f, lfp_pth, lfp_pde, lfp_ptot = compute_theta_features_per_window(
        lfp, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )
    opt_f, opt_pth, opt_pde, opt_ptot = compute_theta_features_per_window(
        opt, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )

    # Speed windows
    if spd is not None:
        idx_starts = np.arange(0, len(lfp) - win_len + 1, step)
        spd_win = np.array([np.nanmean(spd[s:s+win_len]) for s in idx_starts])
    else:
        spd_win = None

    # Theta-rich gating based on LFP theta/delta
    lfp_theta_delta = lfp_pth / (lfp_pde + 1e-12)
    keep = np.isfinite(lfp_theta_delta) & (lfp_theta_delta > theta_delta_min)

    # Speed gating for locomotion
    if require_speed:
        if spd_win is None:
            keep &= False
        else:
            keep &= np.isfinite(spd_win)
            keep &= (spd_win > speed_min) & (spd_win <= speed_max)

    # Relative power
    lfp_rel = lfp_pth / (lfp_ptot + 1e-12)
    opt_rel = opt_pth / (opt_ptot + 1e-12)

    keep &= np.isfinite(lfp_f) & np.isfinite(opt_f) & np.isfinite(lfp_rel) & np.isfinite(opt_rel)

    return (lfp_f[keep], opt_f[keep], lfp_rel[keep], opt_rel[keep])


# ==========================
# === Sweep-level tidy df ===
# ==========================

def summarise_sweep(freq: np.ndarray, rel: np.ndarray) -> Dict[str, float]:
    freq = np.asarray(freq, dtype=float)
    rel = np.asarray(rel, dtype=float)
    freq_f = freq[np.isfinite(freq)]
    rel_f  = rel[np.isfinite(rel)]
    return {
        "theta_freq_median": _nanmedian(freq_f),
        "theta_freq_mean": _nanmean(freq_f),
        "rel_power_median": _nanmedian(rel_f),
        "rel_power_mean": _nanmean(rel_f),
        "n_windows": int(min(freq_f.size, rel_f.size))
    }

def collect_tidy_sweep_table(animals: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for a in animals:
        animal_id = a["animal_id"]
        lfp_channel = a["lfp_channel"]

        for condition, parent, require_speed in [
            ("Locomotion", a["loco_parent"], True),
            ("Awake-stationary", a["stat_parent"], False),
        ]:
            sessions = find_syncrecording_folders(parent)
            for s in sessions:
                df = load_pickle_if_exists(s)
                if df is None or df.empty:
                    continue

                try:
                    lf_f, op_f, lf_r, op_r = extract_theta_feats_from_session(
                        df,
                        lfp_channel=lfp_channel,
                        require_speed=require_speed,
                        speed_min=SPEED_MIN_CM_S,
                        speed_max=SPEED_MAX_CM_S,
                        theta_delta_min=THETA_DELTA_MIN
                    )
                except Exception as e:
                    print(f"[{animal_id} | {condition}] Error in {s}: {e}")
                    continue

                sweep_id = os.path.basename(s)

                lfp_sum = summarise_sweep(lf_f, lf_r)
                opt_sum = summarise_sweep(op_f, op_r)

                rows.append({
                    "animal_id": animal_id,
                    "sweep_id": sweep_id,
                    "condition": condition,
                    "modality": "LFP",
                    "lfp_channel": lfp_channel,
                    **lfp_sum,
                    "folder": s
                })
                rows.append({
                    "animal_id": animal_id,
                    "sweep_id": sweep_id,
                    "condition": condition,
                    "modality": "Optical",
                    "lfp_channel": lfp_channel,
                    **opt_sum,
                    "folder": s
                })

    return pd.DataFrame(rows)


# ==========================
# === Animal-level stats  ===
# ==========================

def cohens_dz(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size < 2:
        return np.nan
    return float(np.nanmean(delta) / (np.nanstd(delta, ddof=1) + 1e-12))

def bootstrap_ci_mean(delta: np.ndarray, n_boot: int = 20000, seed: int = 0) -> Tuple[float, float]:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(delta, size=(n_boot, delta.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)

def animal_level_table(df: pd.DataFrame,
                       metric_col: str,
                       agg: str = "median",
                       min_windows: int = 5) -> pd.DataFrame:
    d = df.copy()
    d = d[d["n_windows"] >= min_windows].copy()

    if agg == "median":
        g = d.groupby(["animal_id", "modality", "condition"], as_index=False)[metric_col].median()
    elif agg == "mean":
        g = d.groupby(["animal_id", "modality", "condition"], as_index=False)[metric_col].mean()
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    wide = g.pivot_table(index=["animal_id", "modality"], columns="condition", values=metric_col).reset_index()
    wide.columns.name = None
    return wide

def paired_stats_for_modality(wide: pd.DataFrame, modality: str, metric_col: str) -> Dict[str, float]:
    w = wide[wide["modality"] == modality].copy()
    if "Locomotion" not in w.columns or "Awake-stationary" not in w.columns:
        return {}

    a = w["Locomotion"].to_numpy(dtype=float)
    b = w["Awake-stationary"].to_numpy(dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]; b = b[keep]
    n = int(a.size)
    if n < 2:
        return {"n_animals": n}

    delta = a - b
    t_res = stats.ttest_rel(a, b, nan_policy="omit")
    try:
        w_res = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = np.nan

    ci_lo, ci_hi = bootstrap_ci_mean(delta)
    return {
        "n_animals": n,
        "mean_delta": float(np.mean(delta)),
        "dz": cohens_dz(delta),
        "t_p": float(t_res.pvalue),
        "wilcoxon_p": w_p,
        "mean_delta_ci95_low": ci_lo,
        "mean_delta_ci95_high": ci_hi
    }


# ==========================
# === Optional mixed model ===
# ==========================

def run_mixedlm(df: pd.DataFrame, metric_col: str, modality: str, min_windows: int = 5) -> Optional[Dict[str, float]]:
    if not HAS_MIXEDLM:
        return None

    d = df.copy()
    d = d[(d["n_windows"] >= min_windows) & (d["modality"] == modality)].copy()
    d = d[np.isfinite(d[metric_col])]
    d["condition"] = pd.Categorical(d["condition"], categories=["Awake-stationary", "Locomotion"], ordered=True)

    if d["animal_id"].nunique() < 2 or len(d) < 6:
        return None

    try:
        model = smf.mixedlm(f"{metric_col} ~ condition", d, groups=d["animal_id"])
        res = model.fit(reml=True, method="lbfgs", maxiter=500, disp=False)
        b = float(res.params.get("condition[T.Locomotion]", np.nan))
        p = float(res.pvalues.get("condition[T.Locomotion]", np.nan))
        return {
            "modality": modality,
            "n_obs_sweeps": int(len(d)),
            "n_animals": int(d["animal_id"].nunique()),
            "beta_condition_Loco_minus_Stat": b,
            "p_value": p
        }
    except Exception as e:
        print(f"MixedLM failed for {modality} {metric_col}: {e}")
        return None


# ==========================
# === Plotting            ===
# ==========================

def plot_animal_paired_single_modality(wide: pd.DataFrame,
                                       modality: str,
                                       title: str,
                                       ylabel: str,
                                       outpath: str):
    w = wide[wide["modality"] == modality].copy()
    if "Awake-stationary" not in w.columns or "Locomotion" not in w.columns:
        return

    y_stat = w["Awake-stationary"].to_numpy(dtype=float)
    y_loco = w["Locomotion"].to_numpy(dtype=float)

    plt.figure(figsize=(5.6, 5.0), dpi=DPI)
    x_stat, x_loco = 1, 2

    for i in range(len(w)):
        if np.isfinite(y_stat[i]) and np.isfinite(y_loco[i]):
            plt.plot([x_stat, x_loco], [y_stat[i], y_loco[i]], linewidth=1.2, alpha=0.7)

    plt.scatter(np.full_like(y_stat, x_stat, dtype=float), y_stat, s=55, label="Awake-stationary")
    plt.scatter(np.full_like(y_loco, x_loco, dtype=float), y_loco, s=55, label="Locomotion")

    ys = y_stat[np.isfinite(y_stat)]
    yl = y_loco[np.isfinite(y_loco)]
    if ys.size >= 2:
        plt.errorbar([x_stat], [np.mean(ys)], yerr=[stats.sem(ys)], fmt='o', capsize=4, markersize=0)
    if yl.size >= 2:
        plt.errorbar([x_loco], [np.mean(yl)], yerr=[stats.sem(yl)], fmt='o', capsize=4, markersize=0)

    plt.xticks([1, 2], ["Awake-stationary", "Locomotion"], fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(outpath + ".png")
    plt.savefig(outpath + ".pdf")
    plt.close()

def plot_within_animal_variability(df: pd.DataFrame,
                                   metric_col: str,
                                   modality: str,
                                   title: str,
                                   ylabel: str,
                                   outpath: str,
                                   min_windows: int = 5):
    d = df[(df["modality"] == modality) & (df["n_windows"] >= min_windows)].copy()
    d = d[np.isfinite(d[metric_col])]
    animals = sorted(d["animal_id"].unique().tolist())
    if len(animals) == 0:
        return

    n = len(animals)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10.5, 3.2*nrows), dpi=DPI, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, animal_id in enumerate(animals):
        ax = axes[i]
        da = d[d["animal_id"] == animal_id].copy()

        x_stat, x_loco = 1, 2
        v_stat = da[da["condition"] == "Awake-stationary"][metric_col].to_numpy(dtype=float)
        v_loco = da[da["condition"] == "Locomotion"][metric_col].to_numpy(dtype=float)

        rng = np.random.default_rng(0)
        j_stat = (rng.random(v_stat.size) - 0.5) * 0.18
        j_loco = (rng.random(v_loco.size) - 0.5) * 0.18

        ax.scatter(np.full_like(v_stat, x_stat, dtype=float) + j_stat, v_stat, s=28, alpha=0.8)
        ax.scatter(np.full_like(v_loco, x_loco, dtype=float) + j_loco, v_loco, s=28, alpha=0.8)

        if v_stat.size:
            ax.hlines(np.nanmedian(v_stat), x_stat-0.22, x_stat+0.22, linewidth=2)
        if v_loco.size:
            ax.hlines(np.nanmedian(v_loco), x_loco-0.22, x_loco+0.22, linewidth=2)

        ax.set_title(f"{animal_id}", fontsize=11)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Stationary", "Locomotion"], fontsize=10)
        ax.grid(False)

    for j in range(len(animals), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=13)
    fig.supylabel(ylabel, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(outpath + ".png")
    fig.savefig(outpath + ".pdf")
    plt.close(fig)


# ==========================
# === Main
# ==========================

def main():
    animals = ANIMALS
    if len(animals) == 0:
        raise RuntimeError("ANIMALS list is empty.")

    print(f"Found {len(animals)} animals:")
    for a in animals:
        print("  ", a["animal_id"], "|", a["lfp_channel"])

    # 1) Tidy table
    tidy = collect_tidy_sweep_table(animals)
    tidy_path = os.path.join(OUTPUT_ROOT, "theta_freq_relpower_sweep_summary_tidy.csv")
    tidy.to_csv(tidy_path, index=False)
    print(f"Saved tidy table: {tidy_path}")

    # 2) Animal-level paired inference
    agg_mode = "median"
    min_windows = 5

    metrics = [
        ("theta_freq_median", "Theta peak frequency (Hz)", "Theta peak frequency"),
        ("rel_power_median", "Relative theta power (theta / total 1–40 Hz)", "Relative theta power"),
    ]

    stats_rows = []
    mixed_rows = []

    for metric_col, ylabel, short in metrics:
        wide = animal_level_table(tidy, metric_col=metric_col, agg=agg_mode, min_windows=min_windows)

        # stats separate by modality
        s_lfp = paired_stats_for_modality(wide, "LFP", metric_col)
        s_opt = paired_stats_for_modality(wide, "Optical", metric_col)

        if s_lfp:
            stats_rows.append({"modality": "LFP", "metric": metric_col, **s_lfp,
                               "animal_agg_across_sweeps": agg_mode, "min_windows_per_sweep": min_windows})
        if s_opt:
            stats_rows.append({"modality": "Optical", "metric": metric_col, **s_opt,
                               "animal_agg_across_sweeps": agg_mode, "min_windows_per_sweep": min_windows})

        # paired plots: separate figures
        plot_animal_paired_single_modality(
            wide=wide,
            modality="LFP",
            title=f"LFP {short}: animal-level paired ({agg_mode} across sweeps; n_windows≥{min_windows}/sweep)",
            ylabel=ylabel,
            outpath=os.path.join(OUTPUT_ROOT, f"paired_LFP_{metric_col}_{agg_mode}")
        )
        plot_animal_paired_single_modality(
            wide=wide,
            modality="Optical",
            title=f"GEVI {short}: animal-level paired ({agg_mode} across sweeps; n_windows≥{min_windows}/sweep)",
            ylabel=ylabel,
            outpath=os.path.join(OUTPUT_ROOT, f"paired_Optical_{metric_col}_{agg_mode}")
        )

        # within-animal variability (also separate)
        plot_within_animal_variability(
            df=tidy, metric_col=metric_col, modality="LFP",
            title=f"LFP {short}: sweep-to-sweep variability",
            ylabel=ylabel,
            outpath=os.path.join(OUTPUT_ROOT, f"within_LFP_{metric_col}"),
            min_windows=min_windows
        )
        plot_within_animal_variability(
            df=tidy, metric_col=metric_col, modality="Optical",
            title=f"GEVI {short}: sweep-to-sweep variability",
            ylabel=ylabel,
            outpath=os.path.join(OUTPUT_ROOT, f"within_Optical_{metric_col}"),
            min_windows=min_windows
        )

        # optional mixed model
        m1 = run_mixedlm(tidy, metric_col=metric_col, modality="LFP", min_windows=min_windows)
        m2 = run_mixedlm(tidy, metric_col=metric_col, modality="Optical", min_windows=min_windows)
        if m1:
            mixed_rows.append({"metric": metric_col, **m1, "min_windows_per_sweep": min_windows})
        if m2:
            mixed_rows.append({"metric": metric_col, **m2, "min_windows_per_sweep": min_windows})

    # save stats
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(OUTPUT_ROOT, "animal_level_stats_freq_relpower.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved animal-level stats: {stats_path}")

    mixed_path = None
    if mixed_rows:
        mixed_df = pd.DataFrame(mixed_rows)
        mixed_path = os.path.join(OUTPUT_ROOT, "mixedlm_sweep_level_stats_freq_relpower.csv")
        mixed_df.to_csv(mixed_path, index=False)
        print(f"Saved mixed-effects stats: {mixed_path}")
    else:
        print("Mixed-effects model not run (statsmodels not available or insufficient data).")

    summary = {
        "params": {
            "Fs_raw": Fs_raw, "TARGET_FS": TARGET_FS,
            "THETA_BAND": THETA_BAND, "DELTA_BAND": DELTA_BAND, "TOTAL_BAND": TOTAL_BAND,
            "WINDOW_SEC": WINDOW_SEC, "WINDOW_STEP_SEC": WINDOW_STEP_SEC,
            "SPEED_MIN_CM_S": SPEED_MIN_CM_S, "SPEED_MAX_CM_S": SPEED_MAX_CM_S,
            "THETA_DELTA_MIN": THETA_DELTA_MIN
        },
        "output": {
            "tidy_csv": tidy_path,
            "animal_level_stats_csv": stats_path,
            "mixedlm_csv": mixed_path,
            "output_root": OUTPUT_ROOT
        }
    }
    with open(os.path.join(OUTPUT_ROOT, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Figures saved as PNG+PDF in:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
