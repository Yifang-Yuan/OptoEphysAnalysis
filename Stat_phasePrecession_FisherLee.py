# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:47:31 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
"""
Theta phase precession with Fisher–Lee circular–linear regression (Python-only MLE).
Replaces OLS-on-unwrapped-phase with a von Mises GLM (Fisher & Lee link).

Inputs:
  ...\same_direction_outputs\SyncRecording*\same_direction_segments.pkl  (list of bout DataFrames)

Outputs:
  ...\same_direction_outputs\stats_phase_precession_localaxis\
    - MASTER_phase_precession_localaxis.csv
    - per-recording CSVs
    - examples/*.png (per-bout scatter with Fisher–Lee curve)
    - pooled_scatter_two_cycles.png
    - rho_histogram.png
"""

import os, glob, pickle, warnings
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import minimize
from numpy.linalg import inv
from scipy.special import i0

# =========================
# CONFIG (adjust if needed)
# =========================
SAVE_ROOT = r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion\same_direction_outputs"
LFP_CHANNEL = "LFP_1"
THETA_LOW, THETA_HIGH = 4, 12
FS = 10_000

# tracking columns for position (shoulder as requested)
SHOULDER_X, SHOULDER_Y = "shoulder_x", "shoulder_y"
POS_SCALE_TO_CM = 1.0   # set to 0.1 if your x/y are mm; set to px->cm if pixels, etc.

# Local-heading segmentation
HEADING_SMOOTH_S   = 0.15    # SavGol smoothing of heading (s)
TURN_THRESH_DEG    = 20.0    # start a new local axis when |heading - segment_mean| > this
MIN_SEG_DUR_S      = 0.30    # drop tiny local segments (< this duration)

# Optical-peak detection around π with a refractory
OPT_PEAK_HALFWIDTH = 0.30    # rad (≈17°)
OPT_PEAK_REFRACT_S = 0.08    # 80 ms

# Statistics
MIN_PEAKS          = 5       # skip bouts with too few peaks
N_EXAMPLES         = 20      # per-bout example scatters
N_PERM             = 2000    # permutations for effect-size p-value

# =========================
# Filters / phase helpers
# =========================
def butter_bandpass(x, fs, low, high, order=4):
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)

def heading_series(x, y, fs, smooth_s=HEADING_SMOOTH_S):
    """Instantaneous heading (deg, wrapped -180..180) from shoulder trajectory, smoothed."""
    dx = np.diff(x, prepend=x[0]); dy = np.diff(y, prepend=y[0])
    raw = np.degrees(np.arctan2(dy, dx))
    unwrapped = np.unwrap(np.radians(raw))
    win = max(5, int(smooth_s * fs) // 2 * 2 + 1)
    sm = signal.savgol_filter(unwrapped, win, polyorder=3, mode="interp")
    return (np.degrees(sm) + 180) % 360 - 180

def detect_optical_peaks(opt_theta_phase, fs, phase_center=np.pi,
                         phase_halfwidth=OPT_PEAK_HALFWIDTH, refractory_s=OPT_PEAK_REFRACT_S):
    idx = np.where((opt_theta_phase > (phase_center - phase_halfwidth)) &
                   (opt_theta_phase < (phase_center + phase_halfwidth)))[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    min_dist = int(refractory_s * fs)
    kept = [int(idx[0])]
    for k in idx[1:]:
        if int(k) - kept[-1] >= min_dist:
            kept.append(int(k))
    return np.array(kept, dtype=int)

def lfp_phase_and_opt_peaks(seg_df, fs=FS, lfp_channel=LFP_CHANNEL, theta_low=THETA_LOW, theta_high=THETA_HIGH):
    """Return (peak_idx, lfp_phase_at_peaks [rad], full lfp_phase [rad])."""
    lfp_f = butter_bandpass(seg_df[lfp_channel].to_numpy(), fs, theta_low, theta_high)
    opt_f = butter_bandpass(seg_df["zscore_raw"].to_numpy(), fs, theta_low, theta_high)
    lfp_phase = np.angle(signal.hilbert(lfp_f))
    opt_phase = np.angle(signal.hilbert(opt_f))
    peak_idx = detect_optical_peaks(opt_phase, fs)
    return peak_idx, lfp_phase[peak_idx], lfp_phase

# =========================
# Local position axis
# =========================
def split_local_heading_segments(head_deg, fs, turn_thresh_deg=TURN_THRESH_DEG, min_seg_s=MIN_SEG_DUR_S):
    """
    Split a bout into near-straight local segments based on heading stability.
    Start a new segment whenever |heading - current_segment_mean| > threshold.
    Enforce a minimum duration per segment.
    Returns list of (start_idx, end_idx) half-open.
    """
    n = len(head_deg)
    spans, s = [], 0
    seg_mean = head_deg[0]
    for i in range(1, n):
        d = (head_deg[i] - seg_mean + 180) % 360 - 180
        if np.abs(d) > turn_thresh_deg:
            if i - s >= int(min_seg_s * fs):
                spans.append((s, i))
            s = i
            seg_mean = head_deg[i]
        else:
            seg_mean = 0.98*seg_mean + 0.02*head_deg[i]
    if n - s >= int(min_seg_s * fs):
        spans.append((s, n))
    return spans

def projected_distance_cm_in_segments(seg_df, fs=FS,
                                      x_col=SHOULDER_X, y_col=SHOULDER_Y,
                                      scale_to_cm=POS_SCALE_TO_CM):
    """
    For each local heading segment, project displacement onto the segment's mean heading.
    Returns:
      proj_cm: array len==len(seg_df) with piecewise-cumulative projected distance (cm)
      seg_spans: list of (s,e)
    """
    x = seg_df[x_col].to_numpy().astype(float)
    y = seg_df[y_col].to_numpy().astype(float)
    head_deg = heading_series(x, y, fs)
    spans = split_local_heading_segments(head_deg, fs)

    proj = np.zeros_like(x, dtype=float)
    for (s, e) in spans:
        mean_hd = np.degrees(np.angle(np.mean(np.exp(1j*np.radians(head_deg[s:e])))))
        ux, uy = np.cos(np.radians(mean_hd)), np.sin(np.radians(mean_hd))
        dx = np.diff(x[s:e], prepend=x[s]); dy = np.diff(y[s:e], prepend=y[s])
        step_proj = dx*ux + dy*uy
        seg_proj = np.cumsum(step_proj) * scale_to_cm
        proj[s:e] = seg_proj - seg_proj[0]
    return proj, spans

# =========================
# Classic circ–lin correlation (kept for reference)
# =========================
def circ_corrcl(alpha, x):
    """Circular-linear correlation (Berens, 2009). alpha in radians; x linear."""
    alpha = np.asarray(alpha); x = np.asarray(x)
    C, S = np.cos(alpha), np.sin(alpha)
    rxc = np.corrcoef(x, C)[0,1]; rxs = np.corrcoef(x, S)[0,1]; rcs = np.corrcoef(C, S)[0,1]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs) / max(1e-12, (1 - rcs**2)))
    n = len(alpha)
    l20 = np.mean(S**2); l02 = np.mean(C**2); l11 = np.mean(S*C)
    tval = rho * np.sqrt((n * (l20*l02 - l11**2)) / max(1e-12, (1 - rho**2)))
    from scipy.stats import t
    pval = 2 * (1 - t.cdf(np.abs(tval), n - 2))
    return float(rho), float(pval)

# =========================
# === NEW: Fisher–Lee circular~linear regression (Python-only MLE)
# =========================
def _wrap_to_pi(a):
    """Map angles to (-pi, pi]; mainly for numeric stability (not strictly required)."""
    return (a + np.pi) % (2*np.pi) - np.pi

def fisher_lee_fit(phase_rad, d_norm, init=None):
    """
    Fit Fisher–Lee model:
      mu(x) = mu0 + 2*atan(beta0 + beta1*x)
      theta_i ~ von Mises(mu_i, kappa)
    Parameters estimated by MLE: [mu0, beta0, beta1, log_kappa]
    Returns dict with MLEs, SE(beta1) (Wald), p_beta1 (2-sided normal approx),
    slope at x=0.5 (rad/unit), and success flag.
    """
    y = np.asarray(phase_rad, float).ravel()
    x = np.asarray(d_norm, float).ravel()
    n = y.size
    if n < 3:
        return dict(success=False)

    # Negative log-likelihood
    def nll(params):
        mu0, b0, b1, logk = params
        mu = mu0 + 2.0*np.arctan(b0 + b1*x)
        kappa = np.exp(logk)
        # von Mises log-likelihood up to constants: kappa*cos(y-mu) - log(2π I0(kappa))
        # Keep constants to stabilise scaling
        return -(kappa*np.cos(_wrap_to_pi(y - mu))).sum() + n*(np.log(2*np.pi) + np.log(i0(kappa)))

    # sensible initials
    if init is None:
        init = np.array([0.0, 0.0, 0.0, np.log(1.0)], dtype=float)

    res = minimize(nll, init, method="BFGS")
    success = bool(res.success)
    mu0, b0, b1, logk = res.x
    kappa = float(np.exp(logk))

    # Wald SE & p for beta1 using inverse Hessian (BFGS approx)
    try:
        Hinv = np.array(res.hess_inv)  # may be ndarray or scipy object; coerce
        se_b1 = float(np.sqrt(max(Hinv[2,2], 1e-12)))
        z = b1 / max(se_b1, 1e-12)
        from scipy.stats import norm
        p_beta1 = float(2*norm.sf(abs(z)))
    except Exception:
        se_b1, p_beta1 = np.nan, np.nan

    # instantaneous slope at mid-bout
    slope_mid_rad = 2.0 * b1 / (1.0 + (b0 + b1*0.5)**2)

    return dict(success=success, mu0=float(mu0), beta0=float(b0), beta1=float(b1),
                log_kappa=float(logk), kappa=kappa,
                se_beta1=float(se_b1), p_beta1=float(p_beta1),
                slope_mid_rad=float(slope_mid_rad), fun=float(res.fun),
                n_iter=int(res.nit))

def perm_pvalue_fisher_lee(phase_rad, d_norm, seg_labels, n_perm=2000, seed=0, effect="beta1"):
    """
    Permute d_norm within local-heading segments, refit, and compute p for |effect|.
    effect ∈ {'beta1','slope_mid'}.
    Returns (obs_effect, p_perm).
    """
    rng = np.random.default_rng(seed)
    fit = fisher_lee_fit(phase_rad, d_norm)
    if not fit["success"]:
        return np.nan, np.nan
    obs = fit["beta1"] if effect == "beta1" else fit["slope_mid_rad"]

    count = 1
    for _ in range(n_perm):
        dn = d_norm.copy()
        for lab in np.unique(seg_labels):
            idx = np.where(seg_labels == lab)[0]
            if idx.size > 1:
                dn[idx] = rng.permutation(dn[idx])
        f = fisher_lee_fit(phase_rad, dn)
        if not f["success"]:
            continue
        stat = f["beta1"] if effect == "beta1" else f["slope_mid_rad"]
        if abs(stat) >= abs(obs):
            count += 1
    p = count / (n_perm + 1)
    return float(obs), float(p)

# =========================
# Plotting helpers
# =========================
def plot_bout_scatter_pos(seg_id, rec_name, d_norm_at_peaks, phase_rad, fl_fit, save_path):
    """Scatter of data + Fisher–Lee fitted curve (display wrapped 0..360°)."""
    plt.figure(figsize=(6,4))
    plt.scatter(d_norm_at_peaks, (np.degrees(phase_rad) % 360), s=22, alpha=0.85, label="Data")

    # --- Fisher–Lee curve ---
    if fl_fit is not None and fl_fit.get("success", False):
        xg = np.linspace(0, 1, 200)
        mu = fl_fit["mu0"] + 2*np.arctan(fl_fit["beta0"] + fl_fit["beta1"]*xg)
        mu_deg = (np.degrees(mu) % 360)
        plt.plot(xg, mu_deg, linewidth=2, label="Fisher–Lee fit")

    plt.ylim(0, 360); plt.yticks([0,90,180,270,360])
    plt.xlabel("Normalised projected distance (0→1)")
    plt.ylabel("LFP phase at optical peaks (deg)")
    plt.title(f"{rec_name} | Bout {seg_id}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def plot_pooled_scatter_two_cycles_pos(all_points, save_path, fit_line=True):
    if not all_points: return
    d_norm = np.array([p["d_norm"] for p in all_points])
    phase_deg = np.array([p["phase_deg"] for p in all_points]) % 360.0

    d_all = np.concatenate([d_norm, d_norm])
    phase_two = np.concatenate([phase_deg, phase_deg + 360.0])

    slope_txt = "n/a"; r_txt = "n/a"; p_txt = "n/a"
    if fit_line:
        phase_rad = np.radians(phase_deg)
        s, b, r, p, _ = linregress(d_norm, np.unwrap(phase_rad))
        slope_txt = f"{s*180/np.pi:.1f}"; r_txt = f"{r:.2f}"; p_txt = f"{p:.3g}"

    plt.figure(figsize=(6,5))
    plt.scatter(d_all, phase_two, s=10, alpha=0.5)
    plt.ylim(0, 720); plt.yticks([0,90,180,270,360,450,540,630,720])
    plt.xlabel("Normalised projected distance (0→1)")
    plt.ylabel("LFP phase at optical peaks (deg)")
    plt.title(f"Pooled precession (two cycles) | slope={slope_txt}°/norm, r={r_txt}, p={p_txt}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def plot_rho_hist(rhos, save_path):
    plt.figure(figsize=(6,4))
    plt.hist(rhos, bins=20, edgecolor="k", alpha=0.8)
    plt.axvline(np.mean(rhos), linestyle="--")
    plt.xlabel("Circular–linear correlation ρ (phase ↔ projected distance)")
    plt.ylabel("Bout count")
    plt.title("Phase–position correlation across bouts (local axes)")
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

# =========================
# Labels/utility
# =========================
def labels_for_peaks(peak_idx, seg_spans):
    """Map every peak index to its local-segment ID."""
    labels = np.zeros(len(peak_idx), dtype=int)
    for k, (s, e) in enumerate(seg_spans):
        mask = (peak_idx >= s) & (peak_idx < e)
        labels[mask] = k
    return labels

# =========================
# Main
# =========================
if __name__ == "__main__":
    stats_dir = os.path.join(SAVE_ROOT, "stats_phase_precession_localaxis")
    os.makedirs(stats_dir, exist_ok=True)
    examples_dir = os.path.join(stats_dir, "examples"); os.makedirs(examples_dir, exist_ok=True)

    seg_pkls = glob.glob(os.path.join(SAVE_ROOT, "SyncRecording*", "same_direction_segments.pkl"))
    print(f"Found {len(seg_pkls)} recordings with saved bouts.")

    all_rows, pooled_points = [], []
    example_plotted = 0

    for seg_pkl in seg_pkls:
        rec_dir = os.path.dirname(seg_pkl)
        rec_name = os.path.basename(rec_dir)
        with open(seg_pkl, "rb") as f:
            segments = pickle.load(f)

        this_rec_rows = []

        for seg_id, seg in enumerate(segments, start=1):
            # local projected distance (cm) and segment spans on the full bout
            proj_cm_full, seg_spans = projected_distance_cm_in_segments(
                seg, fs=FS, x_col=SHOULDER_X, y_col=SHOULDER_Y, scale_to_cm=POS_SCALE_TO_CM
            )

            # optical peaks + phases
            peak_idx, lfp_phase_at_peaks, _lfp_phase_full = lfp_phase_and_opt_peaks(
                seg, fs=FS, lfp_channel=LFP_CHANNEL, theta_low=THETA_LOW, theta_high=THETA_HIGH
            )
            if len(peak_idx) < MIN_PEAKS:
                continue

            # 1-D position at peaks
            x_proj_at_peaks = proj_cm_full[peak_idx]

            # segment label for each peak, for permutation within local segments
            seg_labels = labels_for_peaks(peak_idx, seg_spans)

            # Normalise distance within this bout
            d0, d1 = proj_cm_full.min(), proj_cm_full.max()
            span = max(1e-9, (d1 - d0))
            d_norm_at_peaks = (x_proj_at_peaks - d0) / span

            # --- circ–lin correlation (kept for reference)
            rho_pos, p_rho_pos = circ_corrcl(lfp_phase_at_peaks, x_proj_at_peaks)

            # === NEW: Fisher–Lee fit (Python MLE)
            fl = fisher_lee_fit(lfp_phase_at_peaks, d_norm_at_peaks)
            if not fl["success"]:
                warnings.warn(f"Fisher–Lee failed to converge: {rec_name} bout {seg_id}")
                continue

            # Permutation p-values for |beta1| and |slope_mid|
            _, p_perm_beta1 = perm_pvalue_fisher_lee(
                lfp_phase_at_peaks, d_norm_at_peaks, seg_labels, n_perm=N_PERM, seed=0, effect="beta1"
            )
            _, p_perm_slope = perm_pvalue_fisher_lee(
                lfp_phase_at_peaks, d_norm_at_peaks, seg_labels, n_perm=N_PERM, seed=0, effect="slope_mid"
            )

            row = {
                "recording": rec_name,
                "segment_id": seg_id,
                "n_peaks": int(len(peak_idx)),
                "rho_circ_lin_local": rho_pos,
                "p_circ_lin_local": p_rho_pos,

                # --- Fisher–Lee outputs ---
                "fl_mu0": fl["mu0"],
                "fl_beta0": fl["beta0"],
                "fl_beta1": fl["beta1"],
                "fl_se_beta1": fl["se_beta1"],
                "fl_p_beta1": fl["p_beta1"],
                "fl_kappa": fl["kappa"],
                "fl_slope_mid_deg_per_norm": fl["slope_mid_rad"] * 180/np.pi,
                "fl_perm_p_beta1": p_perm_beta1,
                "fl_perm_p_slope_mid": p_perm_slope,

                "bout_proj_span_cm": float(proj_cm_full.max() - proj_cm_full.min()),
                "n_local_segments": len(seg_spans),
            }
            this_rec_rows.append(row); all_rows.append(row)

            # --- example per-bout plot with Fisher–Lee curve ---
            if example_plotted < N_EXAMPLES:
                bout_png = os.path.join(examples_dir, f"{rec_name}_bout{seg_id:03d}_scatter_localaxis.png")
                plot_bout_scatter_pos(seg_id, rec_name, d_norm_at_peaks, lfp_phase_at_peaks, fl, bout_png)
                example_plotted += 1

            # --- pooled 2-cycle points (for display only) ---
            for dn, ph in zip(d_norm_at_peaks, lfp_phase_at_peaks):
                pooled_points.append({"d_norm": float(dn), "phase_deg": float(np.degrees(ph))})

        # per-recording CSV
        if this_rec_rows:
            rec_csv = os.path.join(stats_dir, f"{rec_name}_phase_precession_localaxis.csv")
            pd.DataFrame(this_rec_rows).to_csv(rec_csv, index=False)
            print(f"[{rec_name}] saved {len(this_rec_rows)} Fisher–Lee bout stats → {rec_csv}")

    # MASTER CSV + pooled plots
    if all_rows:
        master_df = pd.DataFrame(all_rows)
        master_csv = os.path.join(stats_dir, "MASTER_phase_precession_localaxis.csv")
        master_df.to_csv(master_csv, index=False)
        print(f"\nMASTER (local axis, Fisher–Lee): {len(master_df)} bouts → {master_csv}")

        pooled_png = os.path.join(stats_dir, "pooled_scatter_two_cycles.png")
        plot_pooled_scatter_two_cycles_pos(pooled_points, pooled_png)
        print(f"Pooled two-cycle scatter → {pooled_png}")

        rho_png = os.path.join(stats_dir, "rho_histogram.png")
        plot_rho_hist(master_df["rho_circ_lin_local"].values, rho_png)
        print(f"ρ histogram → {rho_png}")

        sig_perm = (master_df["fl_perm_p_slope_mid"] < 0.05).sum()
        print(f"\nPermutation-significant Fisher–Lee effects in {sig_perm}/{len(master_df)} bouts (p<0.05).")
    else:
        print("No bouts with sufficient peaks. Consider loosening MIN_PEAKS or detection params.")
