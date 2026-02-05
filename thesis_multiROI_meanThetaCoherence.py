# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 20:28:05 2026

@author: yifan
"""

"""
Batch-average coherence spectra across all SyncRecording* folders
and plot mean spectrum with 95% CI (moving vs not moving), matching your
plot_coherence_spectrum style (raw + smoothed + theta-band shading).

Outputs:
  - CSV: per-recording spectra summary (optional: saves only metadata)
  - NPZ: per-pair, per-state stacked spectra arrays (frequency grid + traces)
  - PNG: per-pair figure with 2 panels (moving / not moving)

Dependencies:
  - SyncOECPySessionClass.py (SyncOEpyPhotometrySession)
  - numpy, pandas, scipy, matplotlib
"""

from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import coherence, savgol_filter


# =========================
# USER CONFIG
# =========================
#dpath = r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day1and2DLC'
dpath = r'G:\2025_ATLAS_SPAD\MultiFibre\1887932_Jedi2p_Multi_ephysbad\MovingTrialsDLC'

pairs = [
    {"name": "CA1_L–CA1_R", "col_a": "sig_raw",    "col_b": "ref_raw"},
    {"name": "CA1_R–CA3_L", "col_a": "ref_raw",    "col_b": "zscore_raw"},
    {"name": "CA1_L–CA3_L", "col_a": "sig_raw",    "col_b": "zscore_raw"},
]

movement_col = "movement"
moving_states = ("moving",)          # e.g. ("moving","running") if you want
stationary_states = None             # None => everything NOT in moving_states treated as "not moving"

# coherence params (match your single-recording calls)
fmax = 20.0
nperseg_sec = 2.0
overlap = 0.5
window = "hann"
detrend = "constant"

# averaging + plotting
smooth_hz = 0.5                      # same meaning as in your plot_coherence_spectrum
theta_band = (4, 12)
f_step_hz = 0.1                      # frequency grid for averaging (0.1 Hz gives 201 points from 0–20)
min_valid_samples = 64               # per state, per recording

# CI params
ci = 0.95
n_boot = 5000
seed = 0

# session kwargs
session_kwargs = dict(
    IsTracking=False,
    read_aligned_data_from_file=True,
    recordingMode="Atlas",
    indicator="GEVI",
)

# output folder
save_dir = os.path.join(dpath, "coherence_avg_spectra")
os.makedirs(save_dir, exist_ok=True)


# =========================
# HELPERS
# =========================
def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def find_sync_recordings(parent_dir: str, prefix="SyncRecording"):
    recs = []
    for name in os.listdir(parent_dir):
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p) and name.startswith(prefix):
            recs.append(name)
    recs.sort(key=_natural_sort_key)
    return recs

def build_state_masks(df: pd.DataFrame,
                      movement_col: str,
                      moving_states=("moving",),
                      stationary_states=None):
    """
    Returns (mask_moving, mask_notmoving) as boolean numpy arrays.

    If stationary_states is None:
      not moving = movement label exists AND not in moving_states.
    """
    if movement_col not in df.columns:
        raise ValueError(f"Column '{movement_col}' not found.")

    mv = df[movement_col]
    mv_ok = mv.notna()

    if isinstance(moving_states, str):
        moving_states = (moving_states,)

    mask_m = mv.isin(moving_states) & mv_ok

    if stationary_states is None:
        mask_s = (~mv.isin(moving_states)) & mv_ok
    else:
        if isinstance(stationary_states, str):
            stationary_states = (stationary_states,)
        mask_s = mv.isin(stationary_states) & mv_ok

    return mask_m.to_numpy(bool), mask_s.to_numpy(bool)

def _compute_coherence_spectrum(x, y, fs,
                               fmax=20.0,
                               nperseg_sec=2.0,
                               overlap=0.5,
                               window="hann",
                               detrend="constant",
                               min_valid_samples=64):
    """
    Returns (f, Cxy) up to fmax.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]

    N = min(x.size, y.size)
    if N < min_valid_samples:
        return None, None

    nperseg = max(32, int(round(nperseg_sec * fs)))
    if nperseg > N:
        nperseg = max(32, N // 4)
    noverlap = int(round(np.clip(overlap, 0.0, 0.95) * nperseg))

    f, Cxy = coherence(
        x[:N], y[:N],
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        detrend=detrend
    )
    sel = f <= float(fmax)
    return f[sel], Cxy[sel]

def _interp_to_grid(f, y, f_grid):
    """
    Interpolate y(f) onto f_grid, filling edges with boundary values.
    """
    f = np.asarray(f, float)
    y = np.asarray(y, float)
    good = np.isfinite(f) & np.isfinite(y)
    f = f[good]; y = y[good]
    if f.size < 2:
        return np.full_like(f_grid, np.nan, dtype=float)

    # ensure increasing frequency
    order = np.argsort(f)
    f = f[order]; y = y[order]

    yg = np.interp(f_grid, f, y, left=y[0], right=y[-1])
    return yg

def _smooth_spectrum(y_grid, f_step_hz, smooth_hz=0.5):
    """
    Savitzky–Golay smoothing on the frequency grid.
    smooth_hz ~ approximate smoothing bandwidth.
    """
    y = np.asarray(y_grid, float)
    if not (smooth_hz and smooth_hz > 0) or np.all(~np.isfinite(y)):
        return y.copy()

    dfreq = float(f_step_hz)
    k = max(5, int(round(smooth_hz / dfreq)) | 1)   # odd, >=5
    poly = 2 if k > 3 else 1

    # handle NaNs by interpolating across them before filtering
    yy = y.copy()
    idx = np.arange(yy.size)
    good = np.isfinite(yy)
    if not np.any(good):
        return yy
    if not good[0]:
        first = np.argmax(good)
        yy[:first] = yy[first]
        good[:first] = True
    if not good[-1]:
        last = yy.size - 1 - np.argmax(good[::-1])
        yy[last+1:] = yy[last]
        good[last+1:] = True
    yy[~good] = np.interp(idx[~good], idx[good], yy[good])

    return savgol_filter(yy, window_length=min(k, yy.size - (1 - yy.size % 2)),
                         polyorder=poly, mode="interp")

def bootstrap_mean_ci_traces(Y, n_boot=5000, ci=0.95, seed=0):
    """
    Bootstrap 95% CI for the mean at each frequency.
    Y: shape (n_rec, n_freq)
    Returns mean, lo, hi arrays of shape (n_freq,)
    """
    Y = np.asarray(Y, float)
    # drop rows that are all NaN
    keep = np.any(np.isfinite(Y), axis=1)
    Y = Y[keep]
    n, m = Y.shape
    if n == 0:
        return (np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan))
    if n == 1:
        mu = Y[0].copy()
        return (mu, mu.copy(), mu.copy())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.nanmean(Y[idx, :], axis=1)   # (n_boot, n_freq)

    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    mu = np.nanmean(Y, axis=0)
    lo = np.nanquantile(boot_means, lo_q, axis=0)
    hi = np.nanquantile(boot_means, hi_q, axis=0)
    return mu, lo, hi


# =========================
# MAIN: compute spectra across recordings
# =========================
def collect_coherence_spectra_all(dpath, pairs,
                                 movement_col="movement",
                                 moving_states=("moving",),
                                 stationary_states=None,
                                 fmax=20.0,
                                 f_step_hz=0.1,
                                 nperseg_sec=2.0,
                                 overlap=0.5,
                                 window="hann",
                                 detrend="constant",
                                 smooth_hz=0.5,
                                 min_valid_samples=64,
                                 session_kwargs=None,
                                 verbose=True):
    """
    Returns:
      f_grid
      results[pair_name][state] = dict(
          raw = array(n_rec, n_freq),
          smoothed = array(n_rec, n_freq),
          recordings = list of recording names used
      )
    """
    if session_kwargs is None:
        session_kwargs = {}

    recs = find_sync_recordings(dpath, prefix="SyncRecording")
    if verbose:
        print(f"Found {len(recs)} SyncRecording* folders.")
        print(recs)

    f_grid = np.arange(0.0, float(fmax) + 1e-9, float(f_step_hz))

    results = {p["name"]: {"moving": {"raw": [], "smoothed": [], "recordings": []},
                           "not moving": {"raw": [], "smoothed": [], "recordings": []}}
               for p in pairs}

    for rec in recs:
        try:
            S = SyncOEpyPhotometrySession(dpath, rec, **session_kwargs)
            df = S.Ephys_tracking_spad_aligned
            fs = float(S.fs)

            mask_m, mask_s = build_state_masks(df, movement_col, moving_states, stationary_states)

            for p in pairs:
                pair_name = p["name"]
                a = p["col_a"]; b = p["col_b"]
                if (a not in df.columns) or (b not in df.columns):
                    if verbose:
                        print(f"[SKIP] {rec} | {pair_name}: missing columns ({a}, {b})")
                    continue

                x = df[a].to_numpy(float)
                y = df[b].to_numpy(float)

                # ---- moving
                f_m, C_m = _compute_coherence_spectrum(
                    x[mask_m], y[mask_m], fs,
                    fmax=fmax, nperseg_sec=nperseg_sec, overlap=overlap,
                    window=window, detrend=detrend, min_valid_samples=min_valid_samples
                )
                if f_m is not None:
                    Cg = _interp_to_grid(f_m, C_m, f_grid)
                    Cg_s = _smooth_spectrum(Cg, f_step_hz=f_step_hz, smooth_hz=smooth_hz)
                    results[pair_name]["moving"]["raw"].append(Cg)
                    results[pair_name]["moving"]["smoothed"].append(Cg_s)
                    results[pair_name]["moving"]["recordings"].append(rec)

                # ---- not moving
                f_s, C_s = _compute_coherence_spectrum(
                    x[mask_s], y[mask_s], fs,
                    fmax=fmax, nperseg_sec=nperseg_sec, overlap=overlap,
                    window=window, detrend=detrend, min_valid_samples=min_valid_samples
                )
                if f_s is not None:
                    Cg = _interp_to_grid(f_s, C_s, f_grid)
                    Cg_s = _smooth_spectrum(Cg, f_step_hz=f_step_hz, smooth_hz=smooth_hz)
                    results[pair_name]["not moving"]["raw"].append(Cg)
                    results[pair_name]["not moving"]["smoothed"].append(Cg_s)
                    results[pair_name]["not moving"]["recordings"].append(rec)

        except Exception as e:
            if verbose:
                print(f"[FAIL] {rec}: {repr(e)}")
            continue

    # convert lists to arrays
    for pair_name in results:
        for st in ["moving", "not moving"]:
            raw_list = results[pair_name][st]["raw"]
            sm_list  = results[pair_name][st]["smoothed"]
            results[pair_name][st]["raw"] = np.vstack(raw_list) if len(raw_list) else np.empty((0, f_grid.size))
            results[pair_name][st]["smoothed"] = np.vstack(sm_list) if len(sm_list) else np.empty((0, f_grid.size))

    return f_grid, results


# =========================
# PLOTTING: mean spectrum + 95% CI, in your style
# =========================
def plot_avg_spectrum_pair_states(f, results_for_pair,
                                 pair_title="PAIR",
                                 theta_band=(4, 12),
                                 smooth_hz=0.5,
                                 ci=0.95,
                                 n_boot=5000,
                                 seed=0,
                                 label_fs=20,
                                 tick_fs=18,
                                 title_fs=22,
                                 legend_fs=16,
                                 line_w=2.8,
                                 raw_line_w=1.4,
                                 raw_alpha=0.55,
                                 ci_alpha=0.20,
                                 figsize=(11, 5),
                                 savepath=None):
    """
    Makes a 1x2 figure: [moving] [not moving]
    Each panel shows:
      - mean raw
      - mean smoothed
      - 95% CI shading around mean smoothed
      - theta band shading
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True, sharey=True)

    for ax, st in zip(axes, ["moving", "not moving"]):
        Y_raw = results_for_pair[st]["raw"]
        Y_sm  = results_for_pair[st]["smoothed"]
        n = Y_sm.shape[0]

        if n == 0:
            ax.set_title(f"{pair_title} | {st} (n=0)", fontsize=title_fs, pad=8)
            ax.set_xlim(0, f.max())
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("Frequency (Hz)", fontsize=label_fs)
            ax.grid(alpha=0.25, linewidth=0.8)
            # theta shading still
            ax.axvspan(theta_band[0], theta_band[1], color="grey", alpha=0.15, label="theta band")
            ax.tick_params(axis="both", which="both", labelsize=tick_fs)
            continue

        # mean raw + mean smoothed
        mu_raw = np.nanmean(Y_raw, axis=0)
        mu_sm, lo_sm, hi_sm = bootstrap_mean_ci_traces(Y_sm, n_boot=n_boot, ci=ci, seed=seed)

        # plot mean raw first (so it becomes "raw" in legend; matches your style)
        #ax.plot(f, mu_raw, lw=raw_line_w, alpha=raw_alpha, label="raw")

        # CI shading (smoothed)
        ax.fill_between(f, lo_sm, hi_sm, alpha=ci_alpha, linewidth=0)

        # plot mean smoothed
        ax.plot(f, mu_sm, lw=line_w, label=f"smoothed (~{smooth_hz:g} Hz)")

        # theta shading
        ax.axvspan(theta_band[0], theta_band[1], color="grey", alpha=0.15, label="theta band")

        ax.set_xlim(0, f.max())
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Frequency (Hz)", fontsize=label_fs)
        ax.tick_params(axis="both", which="both", labelsize=tick_fs)
        ax.grid(alpha=0.25, linewidth=0.8)

        ax.set_title(f"{pair_title} | {st} (n={n})", fontsize=title_fs, pad=8)
        ax.legend(frameon=False, fontsize=legend_fs, loc="upper right")

    axes[0].set_ylabel("Coherence", fontsize=label_fs)
    return fig, axes


# =========================
# RUN
# =========================
if __name__ == "__main__":
    f_grid, all_results = collect_coherence_spectra_all(
        dpath=dpath,
        pairs=pairs,
        movement_col=movement_col,
        moving_states=moving_states,
        stationary_states=stationary_states,
        fmax=fmax,
        f_step_hz=f_step_hz,
        nperseg_sec=nperseg_sec,
        overlap=overlap,
        window=window,
        detrend=detrend,
        smooth_hz=smooth_hz,
        min_valid_samples=min_valid_samples,
        session_kwargs=session_kwargs,
        verbose=True
    )

    # Save arrays for later reuse (optional but useful)
    # One NPZ per pair, containing f_grid and stacked traces for moving/not moving.
    for p in pairs:
        pair_name = p["name"]
        out_npz = os.path.join(save_dir, f"{pair_name.replace('–','-').replace(' ','')}_spectra.npz")
        np.savez(
            out_npz,
            f=f_grid,
            moving_raw=all_results[pair_name]["moving"]["raw"],
            moving_smoothed=all_results[pair_name]["moving"]["smoothed"],
            moving_recordings=np.array(all_results[pair_name]["moving"]["recordings"], dtype=object),
            notmoving_raw=all_results[pair_name]["not moving"]["raw"],
            notmoving_smoothed=all_results[pair_name]["not moving"]["smoothed"],
            notmoving_recordings=np.array(all_results[pair_name]["not moving"]["recordings"], dtype=object),
        )
        print(f"Saved spectra arrays: {out_npz}")

    # Make plots (one figure per pair)
    for p in pairs:
        pair_name = p["name"]
        fig, axes = plot_avg_spectrum_pair_states(
            f=f_grid,
            results_for_pair=all_results[pair_name],
            pair_title=pair_name,
            theta_band=theta_band,
            smooth_hz=smooth_hz,
            ci=ci,
            n_boot=n_boot,
            seed=seed,
            label_fs=20,
            tick_fs=18,
            title_fs=22,
            legend_fs=16,
            line_w=2.8,
            raw_line_w=1.4,
            raw_alpha=0.6,
            ci_alpha=0.18,
            figsize=(11.5, 5),
            savepath=None
        )

        out_png = os.path.join(save_dir, f"{pair_name.replace('–','-').replace(' ','')}_avg_spectrum_95CI.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {out_png}")

    plt.show()
