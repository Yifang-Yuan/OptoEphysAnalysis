# -*- coding: utf-8 -*-
"""
Group-level theta–speed analysis (locomotion sessions only)

Key points:
- Uses NON-overlapping windows for inference (WINDOW_STEP_SEC = WINDOW_SEC) to reduce dependence.
- Fits mixed-effects models: metric ~ speed + (1 + speed | animal)
- Optional sweep random intercept via variance components: + (1 | sweep_key)
- Also reports per-animal OLS slopes and tests slope vs 0 across animals.

Outputs (in OUTPUT_ROOT):
- theta_speed_windows_tidy.csv
- mixedlm_speed_effects.csv
- per_animal_slopes_*.csv
- Figures: speed_effect_*.png and .pdf (one per metric)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy import stats

# Mixed model
try:
    import statsmodels.formula.api as smf
    HAS_MIXEDLM = True
except Exception:
    HAS_MIXEDLM = False

# ==========================
# USER CONFIG
# ==========================
ANIMALS = [
    {"animal_id": "1765508", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\AwakeStationary",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1844609", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\AwakeStationary",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1881363", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\AwakeStationary",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1851545", "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1851545_WT_Jedi2p_dis\ALocomotion",
     "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1851545_WT_Jedi2p_dis\AwakeStationary",
     "lfp_channel": "LFP_1"},
    {"animal_id": "1887933",
     "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ALocomotion",
     "lfp_channel": "LFP_2"},
]

OUTPUT_ROOT = r"G:\2025_ATLAS_SPAD\AcrossAnimal\theta_speed_group_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

Fs_raw = 10_000
TARGET_FS = 100

THETA_BAND = (4.0, 12.0)
DELTA_BAND = (1.0, 4.0)
TOTAL_BAND = (1.0, 40.0)

# Inference windows: NON-overlapping to reduce temporal dependence
WINDOW_SEC = 1.0
WINDOW_STEP_SEC = 1.0

# Speed gating (change to 0 if you want to include near-zero)
SPEED_MIN_CM_S = 1.0
SPEED_MAX_CM_S = 50.0

USE_THETA_DELTA_GATE = True
THETA_DELTA_MIN = 1.0

# Optional: log-transform speed if you want a more linear relationship
USE_LOG1P_SPEED = False

# Plot control
DPI = 170
MAX_PLOT_POINTS = 25000  # downsample for plotting if huge

# ==========================
# I/O helpers
# ==========================
def find_syncrecording_folders(parent: str):
    return sorted(glob.glob(os.path.join(parent, "SyncRecording*")))

def load_pickle_if_exists(folder: str):
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
# Theta feature extraction
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
                                      df_theta: float):
    hann = get_window('hann', win_len)
    hann_norm = np.sum(hann**2)

    def psd(seg):
        sw = seg * hann
        spec = np.fft.rfft(sw)
        return (np.abs(spec) ** 2) / (hann_norm * target_fs)

    idx_starts = np.arange(0, len(sig) - win_len + 1, step)
    f_out, p_theta_out, p_delta_out, p_total_out = [], [], [], []

    dfreq = freqs[1] - freqs[0] if len(freqs) > 1 else np.nan
    integ = (dfreq if np.isfinite(dfreq) else 1.0)

    for s in idx_starts:
        seg = sig[s:s+win_len]
        Pxx = psd(seg)

        p_th = Pxx[theta_mask]
        if p_th.size == 0 or not np.isfinite(p_th).any():
            f_out.append(np.nan); p_theta_out.append(np.nan); p_delta_out.append(np.nan); p_total_out.append(np.nan)
            continue

        i_max = int(np.nanargmax(p_th))

        # Quadratic interpolation within theta band
        if len(p_th) >= 3 and 0 < i_max < (len(p_th) - 1) and np.isfinite(df_theta):
            y1, y2, y3 = p_th[i_max-1], p_th[i_max], p_th[i_max+1]
            denom = (y1 - 2*y2 + y3)
            if denom == 0 or not np.isfinite(denom):
                delta = 0.0
            else:
                delta = 0.5 * (y1 - y3) / denom
                if (not np.isfinite(delta)) or abs(delta) > 1.0:
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

    return (np.array(f_out), np.array(p_theta_out), np.array(p_delta_out), np.array(p_total_out), idx_starts)

def extract_theta_speed_windows(df: pd.DataFrame, lfp_channel: str) -> pd.DataFrame:
    """
    One row per window:
      speed, lfp_theta_freq, lfp_theta_rel_power, opt_theta_freq, opt_theta_rel_power
    """
    if ("speed" not in df.columns) or ("zscore_raw" not in df.columns) or (lfp_channel not in df.columns):
        return pd.DataFrame()

    bin_sz = int(round(Fs_raw / TARGET_FS))
    spd = np.clip(bin_average(df["speed"].to_numpy(), bin_sz), 0, None)
    lfp = bin_average(df[lfp_channel].to_numpy(), bin_sz)
    opt = bin_average(df["zscore_raw"].to_numpy(), bin_sz)

    n = min(len(spd), len(lfp), len(opt))
    spd, lfp, opt = spd[:n], lfp[:n], opt[:n]

    win_len = int(round(WINDOW_SEC * TARGET_FS))
    step    = int(round(WINDOW_STEP_SEC * TARGET_FS))
    if n < win_len:
        return pd.DataFrame()

    freqs = np.fft.rfftfreq(win_len, d=1.0 / TARGET_FS)
    th_mask  = (freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1])
    de_mask  = (freqs >= DELTA_BAND[0]) & (freqs <= DELTA_BAND[1])
    tot_mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    freq_th = freqs[th_mask]
    df_th = (freq_th[1] - freq_th[0]) if len(freq_th) > 1 else np.nan

    lfp_f, lfp_pth, lfp_pde, lfp_ptot, idx_starts = compute_theta_features_per_window(
        lfp, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )
    opt_f, opt_pth, opt_pde, opt_ptot, _ = compute_theta_features_per_window(
        opt, TARGET_FS, win_len, step, freqs, th_mask, de_mask, tot_mask, freq_th, df_th
    )

    # speed per window
    spd_win = np.array([np.nanmean(spd[s:s+win_len]) for s in idx_starts])

    lfp_rel = lfp_pth / (lfp_ptot + 1e-12)
    opt_rel = opt_pth / (opt_ptot + 1e-12)

    keep = (
        np.isfinite(spd_win) &
        (spd_win >= SPEED_MIN_CM_S) &
        (spd_win <= SPEED_MAX_CM_S) &
        np.isfinite(lfp_f) & np.isfinite(opt_f) &
        np.isfinite(lfp_rel) & np.isfinite(opt_rel)
    )

    if USE_THETA_DELTA_GATE:
        keep &= (lfp_pth / (lfp_pde + 1e-12) > THETA_DELTA_MIN)

    return pd.DataFrame({
        "speed": spd_win[keep],
        "lfp_theta_freq": lfp_f[keep],
        "lfp_theta_rel_power": lfp_rel[keep],
        "opt_theta_freq": opt_f[keep],
        "opt_theta_rel_power": opt_rel[keep],
    })

# ==========================
# Mixed model + animal slopes
# ==========================
def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def collect_all_animals_windows(animals) -> pd.DataFrame:
    rows = []
    for a in animals:
        animal_id = a["animal_id"]
        lfp_channel = a.get("lfp_channel", "LFP_1")
        sessions = find_syncrecording_folders(a["loco_parent"])
        for s in sessions:
            df = load_pickle_if_exists(s)
            if df is None or df.empty:
                continue
            w = extract_theta_speed_windows(df, lfp_channel=lfp_channel)
            if w.empty:
                continue
            w["animal_id"] = animal_id
            w["sweep_id"] = os.path.basename(s)
            w["sweep_key"] = animal_id + "_" + os.path.basename(s)
            rows.append(w)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def fit_mixedlm(df: pd.DataFrame, ycol: str):
    """
    Model:
      y ~ x_c
      random: (1 + x_c | animal_id)
      variance component: (1 | sweep_key)  [optional but helpful]
    """
    if not HAS_MIXEDLM:
        raise RuntimeError("statsmodels not available: cannot run MixedLM.")

    d = df[["animal_id", "sweep_key", "speed", ycol]].dropna().copy()
    d = d[np.isfinite(d["speed"]) & np.isfinite(d[ycol])].copy()

    # x = speed or log1p(speed)
    d["x"] = np.log1p(d["speed"]) if USE_LOG1P_SPEED else d["speed"]
    d["x_c"] = d["x"] - d["x"].mean()

    # Sweep random intercept via variance component
    vc = {"sweep": "0 + C(sweep_key)"}

    # Try full random slope model; fallback to random intercept only if needed
    try:
        model = smf.mixedlm(f"{ycol} ~ x_c", d,
                            groups=d["animal_id"],
                            re_formula="~x_c",
                            vc_formula=vc)
        res = model.fit(reml=True, method="lbfgs", maxiter=3000, disp=False)
        model_type = "RI+RS (animal) + sweep VC"
    except Exception as e:
        print(f"[WARN] Random-slope MixedLM failed for {ycol}: {e}")
        model = smf.mixedlm(f"{ycol} ~ x_c", d,
                            groups=d["animal_id"],
                            re_formula="1",
                            vc_formula=vc)
        res = model.fit(reml=True, method="lbfgs", maxiter=3000, disp=False)
        model_type = "RI (animal) + sweep VC"

    beta = float(res.params.get("x_c", np.nan))
    se = float(res.bse.get("x_c", np.nan))
    p = show_p = float(res.pvalues.get("x_c", np.nan))
    ci_lo = beta - 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan
    ci_hi = beta + 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan

    return {
        "metric": ycol,
        "x": "log1p(speed)" if USE_LOG1P_SPEED else "speed",
        "model": model_type,
        "n_obs_windows": int(len(d)),
        "n_animals": int(d["animal_id"].nunique()),
        "beta": beta,
        "se": se,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_value": float(show_p),
    }, res, d

def per_animal_slopes(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    rows = []
    for animal_id, da in df.groupby("animal_id"):
        x = da["speed"].to_numpy(dtype=float)
        y = da[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 20:
            continue
        x = np.log1p(x) if USE_LOG1P_SPEED else x
        lr = stats.linregress(x, y)
        rows.append({
            "animal_id": animal_id,
            "n_windows": int(x.size),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r": float(lr.rvalue),
            "p": float(lr.pvalue),
            "stderr": float(lr.stderr),
            "ci95_low": float(lr.slope - 1.96*lr.stderr),
            "ci95_high": float(lr.slope + 1.96*lr.stderr),
        })
    return pd.DataFrame(rows)

def test_slopes_vs_zero(slopes_df: pd.DataFrame) -> dict:
    s = slopes_df["slope"].to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return {"n_animals": int(s.size), "t_p": np.nan, "wilcoxon_p": np.nan, "mean_slope": np.nan}
    t = stats.ttest_1samp(s, 0.0)
    try:
        w = stats.wilcoxon(s, zero_method="wilcox", alternative="two-sided")
        w_p = float(w.pvalue)
    except Exception:
        w_p = np.nan
    return {
        "n_animals": int(s.size),
        "mean_slope": float(np.mean(s)),
        "t_p": float(t.pvalue),
        "wilcoxon_p": w_p,
    }

# ==========================
# Plotting
# ==========================
def plot_speed_effect(df_model: pd.DataFrame,
                      slopes_df: pd.DataFrame,
                      mixed_summary: dict,
                      ycol: str,
                      title: str,
                      ylabel: str,
                      outpath: str):
    """
    Pooled scatter (coloured by animal) + per-animal OLS lines + mixed-model fixed-effect line.
    Annotate β, CI, p (mixed model) and animal-slope test (secondary).
    """
    d = df_model.copy()
    # downsample for plotting if needed
    if len(d) > MAX_PLOT_POINTS:
        d = d.sample(n=MAX_PLOT_POINTS, random_state=0)

    plt.figure(figsize=(7.2, 5.4), dpi=DPI)
    ax = plt.gca()

    # scatter per animal
    for animal_id, da in d.groupby("animal_id"):
        ax.scatter(da["speed"], da[ycol], s=10, alpha=0.35, label=animal_id)

    # per-animal lines in speed space
    xx = np.linspace(d["speed"].min(), d["speed"].max(), 200)
    for _, row in slopes_df.iterrows():
        # slope was fit in x-space (speed or log1p(speed)); draw line in original speed axis
        if USE_LOG1P_SPEED:
            xline = np.log1p(xx)
        else:
            xline = xx
        yline = row["intercept"] + row["slope"] * xline
        ax.plot(xx, yline, linewidth=1.2, alpha=0.7)

    # mixed model line (fixed effect only)
    # model was fit: y = b0 + beta*(x - mean_x)  => y = (b0 - beta*mean_x) + beta*x
    # statsmodels intercept key is 'Intercept'
    beta = mixed_summary["beta"]
    ci_lo, ci_hi = mixed_summary["ci95_low"], mixed_summary["ci95_high"]
    pval = mixed_summary["p_value"]

    # Recover intercept (on centred x), then convert to uncentred scale
    # We can approximate from per-window centred x stored in df_model:
    x = np.log1p(df_model["speed"].to_numpy()) if USE_LOG1P_SPEED else df_model["speed"].to_numpy()
    x_mean = np.nanmean(x)

    # Fit intercept by OLS on centred x using mixed beta (stable display only)
    # Using: intercept_display = mean(y) - beta*mean(x)
    y = df_model[ycol].to_numpy()
    y_mean = np.nanmean(y)
    intercept_unc = y_mean - beta * x_mean

    xline = np.log1p(xx) if USE_LOG1P_SPEED else xx
    ax.plot(xx, intercept_unc + beta * xline, linewidth=3)

    # annotate
    slopes_test = test_slopes_vs_zero(slopes_df)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Speed (cm/s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    annot = (
        f"MixedLM: β={beta:.4g} (95% CI [{ci_lo:.4g}, {ci_hi:.4g}]), p={pval:.3g} ({p_to_stars(pval)})\n"
        f"Animal slopes vs 0: n={slopes_test['n_animals']}, Wilcoxon p={slopes_test['wilcoxon_p']:.3g}"
    )
    ax.text(0.02, 0.98, annot, transform=ax.transAxes, va="top", ha="left", fontsize=10)

    # legend (animal IDs)
    ax.legend(frameon=False, fontsize=9, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(outpath + ".png")
    plt.savefig(outpath + ".pdf")
    plt.close()

# ==========================
# Main
# ==========================
def main():
    dfw = collect_all_animals_windows(ANIMALS)
    if dfw.empty:
        raise RuntimeError("No valid windows found. Check paths, channels, and gating thresholds.")

    out_csv = os.path.join(OUTPUT_ROOT, "theta_speed_windows_tidy.csv")
    dfw.to_csv(out_csv, index=False)
    print("Saved:", out_csv, "n_rows=", len(dfw))

    if not HAS_MIXEDLM:
        print("statsmodels not available; cannot run mixed-effects model.")
        return

    targets = [
        ("lfp_theta_freq", "LFP theta peak frequency vs speed", "Theta peak frequency (Hz)"),
        ("lfp_theta_rel_power", "LFP relative theta power vs speed", "Relative theta power (theta/total 1–40 Hz)"),
        ("opt_theta_freq", "Optical (GEVI) theta peak frequency vs speed", "Theta peak frequency (Hz)"),
        ("opt_theta_rel_power", "Optical (GEVI) relative theta power vs speed", "Relative theta power (theta/total 1–40 Hz)"),
    ]

    mixed_rows = []

    for ycol, title, ylabel in targets:
        summ, res, d_model = fit_mixedlm(dfw, ycol=ycol)
        mixed_rows.append(summ)

        # per-animal slopes (secondary, animal unit)
        sl = per_animal_slopes(d_model.rename(columns={"speed":"speed"}), ycol=ycol)
        sl_path = os.path.join(OUTPUT_ROOT, f"per_animal_slopes_{ycol}.csv")
        sl.to_csv(sl_path, index=False)

        # plot
        outfig = os.path.join(OUTPUT_ROOT, f"speed_effect_{ycol}")
        plot_speed_effect(
            df_model=d_model,
            slopes_df=sl,
            mixed_summary=summ,
            ycol=ycol,
            title=title,
            ylabel=ylabel,
            outpath=outfig
        )

        # also save raw mixedlm text summary for record
        with open(os.path.join(OUTPUT_ROOT, f"mixedlm_summary_{ycol}.txt"), "w") as f:
            f.write(str(res.summary()))

        print(f"{ycol}: beta={summ['beta']:.4g}, p={summ['p_value']:.3g}, model={summ['model']}")

    mixed_df = pd.DataFrame(mixed_rows)
    mixed_path = os.path.join(OUTPUT_ROOT, "mixedlm_speed_effects.csv")
    mixed_df.to_csv(mixed_path, index=False)
    print("Saved:", mixed_path)
    print("Done. Figures saved as PNG+PDF in:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
