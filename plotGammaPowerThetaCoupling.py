# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 11:24:12 2025

@author: yifan
"""
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle, gzip, time, uuid
from scipy.signal import butter, sosfiltfilt, hilbert
import os
import OpenEphysTools as OE

_TWO_PI = 2*np.pi
def _butter_bandpass_sos(low, high, fs, order=4):
    return butter(order, [low, high], btype='band', fs=fs, output='sos')

def _inst_power(x):
    # analytic amplitude -> power
    a = hilbert(x)
    return np.abs(a)**2

def _normalised_contour(bin_centers, counts):
    """Return closed contour (theta, r) where r is normalised to [0, 1]."""
    counts = np.asarray(counts)
    r = counts / counts.max() if counts.max() > 0 else counts.astype(float)
    theta_circ = np.append(bin_centers, bin_centers[0])
    r_circ     = np.append(r, r[0])
    return theta_circ, r_circ, r  # return open r as well for averaging


def _cycle_ids(theta_phase):
    """Return integer cycle ids from an unwrapped theta phase array."""
    th_unw = np.unwrap(np.asarray(theta_phase))
    return np.floor(th_unw / _TWO_PI).astype(int)

def _cyclewise_power_pref_angles(theta_phase, power, min_samples_per_cycle=10):
    """
    For each theta cycle, compute a single power-weighted preferred phase angle.
    Returns: angles (radians), n_cycles_used, per-cycle weights (sum power).
    """
    theta_phase = np.asarray(theta_phase) % _TWO_PI
    power = np.asarray(power).astype(float)
    cid = _cycle_ids(theta_phase)
    angles = []
    weights = []
    for c in range(cid.min(), cid.max()+1):
        sel = (cid == c)
        if sel.sum() < min_samples_per_cycle:
            continue
        # Power-weighted complex vector
        C = np.sum(power[sel] * np.cos(theta_phase[sel]))
        S = np.sum(power[sel] * np.sin(theta_phase[sel]))
        if C == 0 and S == 0:
            continue
        angles.append(np.arctan2(S, C) % _TWO_PI)
        weights.append(np.sum(power[sel]))
    return np.array(angles, dtype=float), len(angles), np.array(weights, dtype=float)

def _rayleigh_test(angles):
    """
    Classic Rayleigh test for non-uniformity of circular data.
    angles: array of radians (unweighted).
    Returns dict with n, Rbar, Z, p.
    Berens (2009) approximation for p with small-sample correction.
    """
    a = np.asarray(angles, dtype=float)
    n = a.size
    if n == 0:
        return {'n': 0, 'Rbar': np.nan, 'Z': np.nan, 'p': np.nan}
    C = np.sum(np.cos(a))
    S = np.sum(np.sin(a))
    R = np.sqrt(C*C + S*S)
    Rbar = R / n
    Z = n * (Rbar**2)
    # small-sample corrected p (Berens, CircStat toolbox notes)
    p = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2))
    p = float(np.clip(p, 0.0, 1.0))
    return {'n': int(n), 'Rbar': float(Rbar), 'Z': float(Z), 'p': p}


def save_trial_phase_metrics(theta_phase,
                             signal,
                             save_path,
                             *,
                             bins=30,
                             height_factor=3.0,
                             distance=20,
                             prominence=None,
                             min_events=50,
                             alpha=0.01,
                             use_event_indices=None,
                             meta=None,
                             plot=True):
    """
    Compute per-trial contour, preferred phase and modulation depth, then save to pickle.gz.

    save_path : str/Path
        e.g. 'session01_trial03_phase.pkl.gz'
    meta : dict or None
        Optional metadata to embed (animal, session, trial, Fs, etc.).
    """
    res = OE.compute_optical_phase_preference(
        theta_phase, signal,
        bins=bins,
        height_factor=height_factor,
        distance=distance,
        prominence=prominence,
        min_events=min_events,
        alpha=alpha,
        use_event_indices=use_event_indices,
        plot=plot
    )

    theta_circ, r_circ, r_open = _normalised_contour(res['bin_centers'], res['counts'])

    out = {
        'version'             : 1,
        'uuid'                : str(uuid.uuid4()),
        'timestamp'           : time.time(),
        'bins'                : int(bins),
        'bin_centers'         : res['bin_centers'],
        'counts'              : res['counts'],          # raw counts (for reference)
        'contour_theta'       : theta_circ,             # closed for quick plotting
        'contour_r'           : r_circ,                 # closed for quick plotting
        'contour_r_open'      : r_open,                 # length = bins (for averaging)
        'preferred_phase_rad' : res['preferred_phase_rad'],
        'preferred_phase_deg' : res['preferred_phase_deg'],
        'R'                   : res['modulation_depth_R'],
        'Z'                   : res['Z'],
        'p'                   : res['p'],
        'n_events'            : res['n_events'],
        'significant'         : res['significant'],
        'meta'                : meta or {}
    }

    save_path = Path(save_path)
    if save_path.suffix != '.gz':
        # default to gzip compression
        save_path = save_path.with_suffix(save_path.suffix + '.gz')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_path, out
def _power_phase_stats(theta_phase, power, bins=30):
    """
    Returns:
      bin_centers (rad), mean_power_per_bin (normalised to mean=1),
      preferred_phase (rad), modulation_depth (weighted R), weights_sum
    """
    theta_phase = np.asarray(theta_phase) % (2*np.pi)
    power = np.asarray(power).astype(float)
    # Bin means
    edges = np.linspace(0, 2*np.pi, bins+1)
    idx = np.digitize(theta_phase, edges) - 1
    idx[idx == bins] = bins-1
    mean_pow = np.zeros(bins, dtype=float)
    for b in range(bins):
        sel = (idx == b)
        mean_pow[b] = power[sel].mean() if np.any(sel) else 0.0
    # Normalise contour to unit mean (common in Tort-style plots)
    mean_pow_norm = mean_pow / (mean_pow.mean() + 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2

    # Power-weighted circular stats
    C = np.sum(power * np.cos(theta_phase))
    S = np.sum(power * np.sin(theta_phase))
    preferred = (np.arctan2(S, C)) % (2*np.pi)
    Rw = np.sqrt(C**2 + S**2) / (np.sum(power) + 1e-12)

    return centers, mean_pow_norm, preferred, Rw, float(np.sum(power))

def compute_gamma_power_on_theta_phase(theta_phase,
                                       signal,
                                       Fs,
                                       gamma_band=(30, 80),
                                       bins=30,
                                       zscore=False,
                                       plot=True,
                                       title=None):
    """
    Gamma power on theta phase (Tort et al., 2010 approach).
    - theta_phase: radians [0, 2π), 0° = trough (your convention), len = N
    - signal: raw LFP (µV) or optical trace (ΔF/F or z-score), len = N
    - Fs: sampling rate (Hz)
    - gamma_band: (low, high) Hz
    """
    x = np.asarray(signal).astype(float)
    if zscore:
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    sos = _butter_bandpass_sos(gamma_band[0], gamma_band[1], Fs, order=4)
    xg = sosfiltfilt(sos, x)
    Pg = _inst_power(xg)

    ctr, contour, pref, Rw, wsum = _power_phase_stats(theta_phase, Pg, bins=bins)

    fig = None
    if plot:
        # Close contour for plotting
        theta_circ = np.append(ctr, ctr[0])
        r_circ     = np.append(contour, contour[0])

        fig = plt.figure(figsize=(5.2, 5.2))
        ax  = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)  # anticlockwise
        ax.spines['polar'].set_linewidth(2.5)
        ax.grid(True, linewidth=0.8, alpha=0.4)
        ax.set_thetagrids([0, 90, 180, 270], labels=['0', '90', '180', '270'], fontsize=14, weight='bold')

        ax.set_ylim(0, np.max(r_circ) if np.max(r_circ) > 1 else 1.0)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['0.5', '1'], fontsize=11)

        ax.plot(theta_circ, r_circ, linewidth=2.5)
        ax.annotate("", xy=(pref, 0.9*Rw), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", linewidth=2.5))

        ttl = title if title else f"pref={np.degrees(pref):.1f}°, R={Rw:.3f}"
        ax.set_title(ttl, va="bottom", pad=12, fontsize=12)
        plt.tight_layout()
    
    # --- NEW: cycle-level preferred phases + Rayleigh test (n = # theta cycles)
    angles_cyc, n_cyc, w_cyc = _cyclewise_power_pref_angles(theta_phase, Pg, min_samples_per_cycle=10)
    ray = _rayleigh_test(angles_cyc)

    return {
        'bin_centers'          : ctr,
        'contour_norm_mean'    : contour,
        'preferred_phase_rad'  : float(pref),
        'preferred_phase_deg'  : float(np.degrees(pref) % 360.0),
        'modulation_depth_R'   : float(Rw),
        'gamma_band'           : tuple(gamma_band),
        'weights_sum'          : wsum,
        'fig'                  : fig,
        'cycle_pref_angles'    : angles_cyc,   # array (radians) per cycle
        'n_cycles'             : int(n_cyc),   # sample size for significance test
        'rayleigh'             : ray,          # {'n','Rbar','Z','p'}
    }

def save_trial_gamma_power_phase(theta_phase,
                                 lfp_signal,
                                 opt_signal,
                                 Fs,
                                 save_path,
                                 bins=30,
                                 lfp_gamma=(30,80),
                                 opt_gamma=(30,80),
                                 zscore_opt=False,
                                 meta=None,
                                 plot=True):
    """
    Compute gamma-power-on-theta-phase for LFP and optical; save to .pkl.gz
    Also optionally generate a side-by-side polar plot for visualisation.
    """
    res_lfp = compute_gamma_power_on_theta_phase(theta_phase, lfp_signal, Fs,
                                                 gamma_band=lfp_gamma, bins=bins,
                                                 zscore=False, plot=False)
    res_opt = compute_gamma_power_on_theta_phase(theta_phase, opt_signal, Fs,
                                                 gamma_band=opt_gamma, bins=bins,
                                                 zscore=zscore_opt, plot=False)
    fig, axes = plot_gamma_power_on_theta_cartesian_single(
    res_lfp, cycles=2, theta_amp=1.0, title="LFP")
    plot_gamma_power_on_theta_cartesian_single(res_opt, cycles=2, title="Optical")
    
    out = {
        'version'  : 1,
        'uuid'     : str(uuid.uuid4()),
        'timestamp': time.time(),
        'bins'     : int(bins),

        'lfp' : {
            'gamma_band'          : res_lfp['gamma_band'],
            'bin_centers'         : res_lfp['bin_centers'],
            'contour_norm_mean'   : res_lfp['contour_norm_mean'],
            'preferred_phase_rad' : res_lfp['preferred_phase_rad'],
            'preferred_phase_deg' : res_lfp['preferred_phase_deg'],
            'R'                   : res_lfp['modulation_depth_R'],
            'weights_sum'         : res_lfp['weights_sum'],
            # NEW
            'cycle_pref_angles'   : res_lfp['cycle_pref_angles'],
            'n_cycles'            : res_lfp['n_cycles'],
            'rayleigh'            : res_lfp['rayleigh'],
        },
        'opt' : {
            'gamma_band'          : res_opt['gamma_band'],
            'bin_centers'         : res_opt['bin_centers'],
            'contour_norm_mean'   : res_opt['contour_norm_mean'],
            'preferred_phase_rad' : res_opt['preferred_phase_rad'],
            'preferred_phase_deg' : res_opt['preferred_phase_deg'],
            'R'                   : res_opt['modulation_depth_R'],
            'weights_sum'         : res_opt['weights_sum'],
            # NEW
            'cycle_pref_angles'   : res_opt['cycle_pref_angles'],
            'n_cycles'            : res_opt['n_cycles'],
            'rayleigh'            : res_opt['rayleigh'],
        },
        'meta': meta or {}
    }

    # Save results
    save_path = Path(save_path)
    if save_path.suffix != '.gz':
        save_path = save_path.with_suffix(save_path.suffix + '.gz')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(9, 4.5))
        for ax, res, label in zip(axes, [res_lfp, res_opt], ["LFP", "Optical"]):
            bc = res['bin_centers']
            r  = res['contour_norm_mean']
            theta_circ = np.append(bc, bc[0])
            r_circ     = np.append(r, r[0])

            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.spines['polar'].set_linewidth(2.0)
            ax.grid(True, linewidth=0.8, alpha=0.4)
            ax.set_thetagrids([0, 90, 180, 270],
                              labels=['0', '90', '180', '270'],
                              fontsize=12, weight='bold')
            ax.set_ylim(0, max(1.0, r_circ.max()))
            ax.set_yticks([0.5, 1.0])
            ax.set_yticklabels(['0.5', '1'], fontsize=9)

            ax.plot(theta_circ, r_circ, linewidth=2.0)
            ax.annotate("", xy=(res['preferred_phase_rad'], 0.9*res['modulation_depth_R']),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", linewidth=2.0))
            ax.set_title(f"{label}\nPref={res['preferred_phase_deg']:.1f}°, R={res['modulation_depth_R']:.3f}",
                         fontsize=11)

        fig.suptitle("Gamma power on theta phase (single trial)", fontsize=13)
        fig.tight_layout()

        # Save the figure alongside the pickle
        fig_save_path = save_path.with_suffix('.png')
        fig.savefig(fig_save_path, dpi=300)
        print(f"📈 Saved polar plot to {fig_save_path}")

    return save_path, out, fig

def plot_gamma_power_on_theta_cartesian_single(
    res,
    cycles=2,
    theta_amp=1.0,
    gamma_color=(0.45, 0.55, 0.20),  # olive-ish
    title="",
    gamma_label="Gamma relative magnitude (a.u.)",
    theta_label="Theta amplitude (a.u.)",
    *,
    title_fs=18,
    label_fs=14,
    tick_fs=14
):
    """
    Plot gamma power vs theta phase alongside a theta sine wave (single trial).
    `res` is the dict returned by compute_gamma_power_on_theta_phase(...).
    """
    bc_rad   = np.asarray(res['bin_centers'])
    bc_deg   = np.degrees(bc_rad) % 360.0
    gamma_y  = np.asarray(res['contour_norm_mean'])  # unit-mean

    x_deg        = np.concatenate([bc_deg + 360*k for k in range(cycles)])
    gamma_tiled  = np.tile(gamma_y, cycles)

    x_line = np.linspace(0, 360*cycles, 1000)
    theta_sine = theta_amp * np.sin(np.deg2rad(x_line))

    fig, ax_left = plt.subplots(figsize=(6.2, 4.6))
    # Thicker spines
    for s in ax_left.spines.values():
        s.set_linewidth(2)

    # Theta (left axis)
    ax_left.plot(x_line, theta_sine, color='black', linewidth=2)
    ax_left.set_xlim(0, 360*cycles)
    ax_left.set_ylim(-1.2*theta_amp, 1.2*theta_amp)
    ax_left.set_xlabel("LFP Theta Phase (deg.)", fontsize=label_fs, fontweight='bold')
    ax_left.set_ylabel(theta_label, fontsize=label_fs, fontweight='bold')
    ax_left.set_title(title or f"pref={res['preferred_phase_deg']:.1f}°, R={res['modulation_depth_R']:.3f}",
                      fontsize=title_fs, fontweight='bold', pad=12)

    # Gamma (right axis)
    ax_right = ax_left.twinx()
    ax_right.plot(x_deg, gamma_tiled, color=gamma_color, linewidth=3, alpha=0.9)
    ax_right.set_ylabel(gamma_label, color=gamma_color, fontsize=label_fs, fontweight='bold')
    ax_right.tick_params(axis='y', colors=gamma_color, labelsize=tick_fs)
    ax_left.tick_params(axis='both', labelsize=tick_fs)
    # Right spine thicker
    ax_right.spines['right'].set_linewidth(2)

    # X ticks every 180°
    ax_left.set_xticks(np.arange(0, 360*cycles+1, 180))
    ax_left.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax_left, ax_right)


def plot_gamma_power_on_theta_cartesian_group(
    agg_side,
    cycles=2,
    theta_amp=1.0,
    gamma_color=(0.45, 0.55, 0.20),
    title_prefix="",
    *,
    title_fs=18,
    label_fs=14,
    tick_fs=14
):
    """
    Plot group mean gamma power vs theta phase with 95% CI (from aggregate_gamma_power_phase).
    `agg_side` is either agg['lfp'] or agg['opt'].
    """
    bc_rad  = np.asarray(agg_side['bin_centers'])
    bc_deg  = np.degrees(bc_rad) % 360.0
    mean_y  = np.asarray(agg_side['mean_contour'])
    lo_y    = np.asarray(agg_side['ci_lower'])
    hi_y    = np.asarray(agg_side['ci_upper'])

    x_deg       = np.concatenate([bc_deg + 360*k for k in range(cycles)])
    mean_tiled  = np.tile(mean_y, cycles)
    lo_tiled    = np.tile(lo_y, cycles)
    hi_tiled    = np.tile(hi_y, cycles)

    x_line = np.linspace(0, 360*cycles, 1000)
    theta_sine = theta_amp * np.sin(np.deg2rad(x_line))

    fig, ax_left = plt.subplots(figsize=(6.2, 4.6))
    for s in ax_left.spines.values():
        s.set_linewidth(2)

    ax_left.plot(x_line, theta_sine, color='black', linewidth=2)
    ax_left.set_xlim(0, 360*cycles)
    ax_left.set_ylim(-1.2*theta_amp, 1.2*theta_amp)
    ax_left.set_xlabel("LFP Theta Phase (deg.)", fontsize=label_fs, fontweight='bold')
    ax_left.set_ylabel("Theta amplitude (a.u.)", fontsize=label_fs, fontweight='bold')

    ax_right = ax_left.twinx()
    ax_right.fill_between(x_deg, lo_tiled, hi_tiled, color=gamma_color, alpha=0.25, linewidth=0)
    ax_right.plot(x_deg, mean_tiled, color=gamma_color, linewidth=3)
    ax_right.set_ylabel("Gamma relative magnitude (a.u.)",
                        color=gamma_color, fontsize=label_fs, fontweight='bold')
    ax_right.tick_params(axis='y', colors=gamma_color, labelsize=tick_fs)
    ax_left.tick_params(axis='both', labelsize=tick_fs)
    ax_right.spines['right'].set_linewidth(2)

    # ttl = (f"{title_prefix}pref={agg_side['group_pref_phase_deg']:.1f}°, "
    #        f"R={agg_side['group_R']:.3f} (n={agg_side['n_trials']})")
    ttl = (f"{title_prefix}pref={agg_side['group_pref_phase_deg']:.1f}°, "
       f"R={agg_side['group_R']:.3f} \n  "
       f"Rayleigh (cycles): n={agg_side.get('n_cycles', 0)}, p={agg_side.get('rayleigh',{}).get('p', np.nan):.3g}  "
       f"(sweeps={agg_side.get('n_trials', 0)})")

    
    ax_left.set_title(ttl, fontsize=title_fs, fontweight='bold', pad=12)

    ax_left.set_xticks(np.arange(0, 360*cycles+1, 180))
    ax_left.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, (ax_left, ax_right)

def _load_pickle(path):
    path = Path(path)
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

def aggregate_gamma_power_phase(paths, ci_method="bootstrap", n_boot=2000, ci_alpha=0.05):
    """
    Aggregate many per-trial gamma-power-on-theta-phase pickles produced above.
    Returns separate aggregates for LFP and optical.
    """
    recs = [_load_pickle(p) for p in paths]
    if not recs:
        raise ValueError("No files provided.")

    # Consistency checks
    bins0 = recs[0]['bins']
    bc0_lfp = recs[0]['lfp']['bin_centers']
    bc0_opt = recs[0]['opt']['bin_centers']
    for r in recs[1:]:
        assert r['bins'] == bins0
        if not np.allclose(r['lfp']['bin_centers'], bc0_lfp): raise ValueError("LFP bins differ.")
        if not np.allclose(r['opt']['bin_centers'], bc0_opt): raise ValueError("Opt bins differ.")

    def _stack(side):
        contours = np.vstack([r[side]['contour_norm_mean'] for r in recs])  # (T, bins)
        weights  = np.array([r[side]['weights_sum'] for r in recs], dtype=float)
        phi      = np.array([r[side]['preferred_phase_rad'] for r in recs], dtype=float)
        R        = np.array([r[side]['R'] for r in recs], dtype=float)
        # NEW: cycle-level arrays (ragged; we’ll concat)
        cyc_lists = [np.asarray(r[side]['cycle_pref_angles'], dtype=float) for r in recs]
        ncyc      = np.array([int(r[side]['n_cycles']) for r in recs], dtype=int)
        return contours, weights, phi, R, cyc_lists, ncyc

    agg = {}
    for side in ['lfp', 'opt']:
        contours, w, phi, R, cyc_lists, ncyc = _stack(side)
        T, B = contours.shape
        mean_contour = contours.mean(axis=0)
        sem_contour  = contours.std(axis=0, ddof=1) / np.sqrt(T)

        # CI
        if ci_method == "sem":
            from scipy.stats import norm
            z = norm.ppf(1 - ci_alpha/2)
            ci_lower = mean_contour - z * sem_contour
            ci_upper = mean_contour + z * sem_contour
        else:
            rng = np.random.default_rng()
            boot = np.empty((n_boot, B))
            for b in range(n_boot):
                idx = rng.integers(0, T, size=T)
                boot[b] = contours[idx].mean(axis=0)
            lo = 100 * (ci_alpha/2)
            hi = 100 * (1 - ci_alpha/2)
            ci_lower = np.percentile(boot, lo, axis=0)
            ci_upper = np.percentile(boot, hi, axis=0)

        # Power-weighted circular mean across trials
        # weight each trial by its total gamma power (w) — analogous to event weighting
        Cx = np.sum(w * R * np.cos(phi))
        Sy = np.sum(w * R * np.sin(phi))
        group_pref = (np.arctan2(Sy, Cx)) % (2*np.pi)
        group_R    = np.sqrt(Cx**2 + Sy**2) / (np.sum(w) + 1e-12)
        # NEW: pooled cycle-level Rayleigh across all sweeps (n = total cycles)
        all_cycle_angles = np.concatenate([c for c in cyc_lists if c.size > 0], axis=0) if len(cyc_lists) else np.array([])
        ray_pool = _rayleigh_test(all_cycle_angles)

        agg[side] = {
        'bin_centers' : (bc0_lfp if side=='lfp' else bc0_opt),
        'mean_contour': mean_contour,
        'sem_contour' : sem_contour,
        'ci_lower'    : ci_lower,
        'ci_upper'    : ci_upper,
        'group_pref_phase_rad': float(group_pref),
        'group_pref_phase_deg': float(np.degrees(group_pref) % 360.0),
        'group_R'     : float(group_R),
        'n_trials'    : int(T),
        # NEW
        'n_cycles'    : int(ray_pool['n']),
        'rayleigh'    : ray_pool,  # {'n','Rbar','Z','p'}
    }

    return agg

def plot_power_phase_agg(agg_side, title=None):
    bc = agg_side['bin_centers']
    mean_r = agg_side['mean_contour']
    lo = agg_side['ci_lower']
    hi = agg_side['ci_upper']

    theta_circ = np.append(bc, bc[0])
    mean_circ  = np.append(mean_r, mean_r[0])
    lo_circ    = np.append(lo, lo[0])
    hi_circ    = np.append(hi, hi[0])

    fig = plt.figure(figsize=(5.2, 5.2))
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.spines['polar'].set_linewidth(2.5)
    ax.grid(True, linewidth=0.8, alpha=0.4)
    ax.set_thetagrids([0, 90, 180, 270],
                      labels=['0', '90', '180', '270'],
                      fontsize=14, weight='bold')
    ax.set_ylim(0, max(1.0, mean_circ.max()))
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1'], fontsize=11)

    # CI band and mean contour
    ax.fill_between(theta_circ, lo_circ, hi_circ, alpha=0.25, linewidth=0)
    ax.plot(theta_circ, mean_circ, linewidth=2.5)

    # Preferred phase arrow
    arrow_len = 0.9 * agg_side['group_R']
    ax.annotate("", xy=(agg_side['group_pref_phase_rad'], arrow_len),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", linewidth=2.5))

    # Label preferred phase and modulation index
    label_r = min(1.05, arrow_len + 0.15)  # place slightly beyond arrow tip
    ax.text(agg_side['group_pref_phase_rad'], label_r,
            f"{agg_side['group_pref_phase_deg']:.1f}°\nR={agg_side['group_R']:.3f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    # Title
    if title is None:
        title = (f"pref={agg_side['group_pref_phase_deg']:.1f}°, "
                 f"R={agg_side['group_R']:.3f} (n={agg_side['n_trials']})")
        # title = (f"pref={agg_side['group_pref_phase_deg']:.1f}°, "
        #          f"R={agg_side['group_R']:.3f} \n  "
        #          f"Rayleigh (cycles): n={agg_side.get('n_cycles',0)}, "
        #          f"p={agg_side.get('rayleigh',{}).get('p', np.nan):.3g}  "
        #          f"(sweeps={agg_side.get('n_trials',0)})")

    ax.set_title(title, va="bottom", pad=12, fontsize=12)

    plt.tight_layout()
    return fig, ax

import re
from pathlib import Path

def run_gamma_power_on_theta_batch(
    dpath,
    savename,
    LFP_channel,
    theta_low_thres=-0.5,
    theta_band_for_phase=(5, 12),
    lfp_gamma=(30, 80),
    opt_gamma=(30, 80),
    overwrite=False,
    plot_theta=True,
    behaviour='Rest'
):
    """
    Loop over all SyncRecording* folders in `dpath`, in numeric order, and run:
      1) save_trial_phase_metrics (event counts vs theta phase)
      2) save_trial_gamma_power_phase (gamma power vs theta phase; LFP + Optical)
    Saves outputs into:  <dpath>/<savename>/

    Parameters
    ----------
    dpath : str or Path
    savename : str
        Subfolder under dpath to store outputs (e.g., 'GammaPowerOnTheta').
    LFP_channel : str
        e.g., 'LFP_1'
    theta_low_thres : float
        Threshold used when labelling theta epochs (your existing function).
    theta_band_for_phase : (low, high)
        Band (Hz) to compute theta phase for `OE.calculate_theta_phase_angle`.
    lfp_gamma, opt_gamma : (low, high)
        Gamma bands for power–phase analysis (Hz).
    overwrite : bool
        If False, skip trials that already have the target files.
    plot_theta : bool
        Whether to show theta plots in your label step.

    Returns
    -------
    list of dict
        Per-trial summaries with file paths and key metrics.
    """
    dpath = Path(dpath)
    save_root = dpath / savename
    save_root.mkdir(parents=True, exist_ok=True)

    # 1) Find and sort SyncRecording* numerically
    all_recs = [p for p in dpath.iterdir() if p.is_dir() and p.name.startswith("SyncRecording")]
    def rec_num(name):
        m = re.search(r"SyncRecording(\d+)", name)
        return int(m.group(1)) if m else 10**9
    all_recs_sorted = sorted(all_recs, key=lambda p: rec_num(p.name))

    if not all_recs_sorted:
        print(f"No 'SyncRecording*' folders found in {dpath}")
        return []

    results = []
    for idx, rec_path in enumerate(all_recs_sorted, start=1):
        recordingName = rec_path.name
        print(f"\n=== Processing {recordingName}  →  Trial {idx:02d} ===")

        # File names
        trial_phase_fname   = f"TrialPhase_trial{idx:02d}.pkl.gz"           # your original naming
        trial_phase_path    = save_root / trial_phase_fname
        gp_on_theta_fname   = f"GammaPowerOnTheta_trial{idx:02d}.pkl.gz"
        gp_on_theta_path    = save_root / gp_on_theta_fname
        gp_on_theta_png     = gp_on_theta_path.with_suffix(".png")

        if not overwrite and trial_phase_path.exists() and gp_on_theta_path.exists() and gp_on_theta_png.exists():
            print(f"Skip (already exists): {trial_phase_path.name}, {gp_on_theta_path.name}")
            results.append({
                'recording' : recordingName,
                'idx'       : idx,
                'skipped'   : True,
                'phase_pkl' : str(trial_phase_path),
                'gamma_pkl' : str(gp_on_theta_path),
                'gamma_png' : str(gp_on_theta_png)
            })
            continue

        # 2) Load session
        Recording1 = SyncOEpyPhotometrySession(
            str(dpath), recordingName,
            IsTracking=False,
            read_aligned_data_from_file=True,
            recordingMode='Atlas',
            indicator='GEVI'
        )

        # 3) Label theta epochs and take the selected part
        if behaviour=='Rest':
            'Label theta part'
            Recording1.pynacollada_label_theta(
                LFP_channel,
                Low_thres=theta_low_thres,
                High_thres=10,
                save=False,
                plot_theta=plot_theta
            )
            theta_part = Recording1.theta_part
        if behaviour=='REM':
            'Label REM sleep'
            Recording1.Label_REM_sleep (LFP_channel)
            theta_part =  Recording1.Ephys_tracking_spad_aligned[
            Recording1.Ephys_tracking_spad_aligned['REMstate'] == 'REM']
        if behaviour == 'Moving':
            '''Choose MOving state from open field trials '''
            theta_part =  Recording1.Ephys_tracking_spad_aligned[
            Recording1.Ephys_tracking_spad_aligned['movement'] == 'moving']

        # 4) Theta phase for the selected band (your convention: 0 at trough)
        tl, th = theta_band_for_phase
        theta_angle = OE.calculate_theta_phase_angle(theta_part[LFP_channel], theta_low=tl, theta_high=th)

        # 5) Save event-on-phase metrics (counts vs phase; your existing function)
        try:
            save_trial_phase_metrics(
                theta_angle,
                theta_part['zscore_raw'],
                trial_phase_path,
                bins=30, height_factor=3.0, distance=20, plot=True
            )
        except Exception as e:
            print(f"Warning: save_trial_phase_metrics failed for {recordingName}: {e}")

        # 6) Gamma power on theta phase (LFP + Optical) and plot
        lfp_sig = theta_part[LFP_channel].to_numpy() / 1000.0  # µV→mV (optional)
        opt_sig = theta_part['zscore_raw'].to_numpy()
        Fs = Recording1.fs

        try:
            _, out, fig = save_trial_gamma_power_phase(
                theta_phase=theta_angle,
                lfp_signal=lfp_sig,
                opt_signal=opt_sig,
                Fs=Fs,
                save_path=gp_on_theta_path,
                bins=30,
                lfp_gamma=lfp_gamma,
                opt_gamma=opt_gamma,
                zscore_opt=False,
                meta={'recording': recordingName, 'trial_index': idx, 'state': 'awake_stationary'},
                plot=True
            )
        except Exception as e:
            print(f"Warning: save_trial_gamma_power_phase failed for {recordingName}: {e}")
            out, fig = None, None

        # 7) Collect summary
        trial_summary = {
            'recording' : recordingName,
            'idx'       : idx,
            'skipped'   : False,
            'phase_pkl' : str(trial_phase_path),
            'gamma_pkl' : str(gp_on_theta_path),
            'gamma_png' : str(gp_on_theta_png)
        }
        if out is not None:
            trial_summary.update({
                'lfp_pref_deg' : out['lfp']['preferred_phase_deg'],
                'lfp_R'        : out['lfp']['R'],
                'opt_pref_deg' : out['opt']['preferred_phase_deg'],
                'opt_R'        : out['opt']['R'],
            })
        results.append(trial_summary)

    print(f"\nDone. Processed {len([r for r in results if not r['skipped']])} / {len(results)} trials.")
    return results

#%%
#dpath      = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion'
dpath      = r'G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\ALocomotion'
savename   = 'GammaPowerOnTheta'
LFP_channel= 'LFP_1'
'Run all SyncRecording'
results = run_gamma_power_on_theta_batch(
    dpath=dpath,
    savename=savename,
    LFP_channel=LFP_channel,
    theta_low_thres=-0.3,         # your usual threshold
    theta_band_for_phase=(4, 12), # theta band for phase
    lfp_gamma=(65, 100),
    opt_gamma=(65, 100),
    overwrite=True,
    plot_theta=True,
    behaviour='Moving'
)
#%%
dpath      = r'G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\ALocomotion'
save_path=os.path.join(dpath,'GammaPowerOnTheta')
pp_paths = sorted(Path(save_path).glob("GammaPowerOnTheta_trial*.pkl.gz"))
agg = aggregate_gamma_power_phase(pp_paths, ci_method="bootstrap", n_boot=2000, ci_alpha=0.05)
'Plot LFP gamma phase'
fig, axes = plot_gamma_power_on_theta_cartesian_group(
    agg['lfp'], cycles=2, theta_amp=1.0, title_prefix="LFP ")
'Plot optical gamma phase'
plot_gamma_power_on_theta_cartesian_group(agg['opt'], cycles=2, title_prefix="Optical ")
# Plot LFP and optical separately
plot_power_phase_agg(agg['lfp'],  title="LFP γ power on θ phase (group)")
plot_power_phase_agg(agg['opt'],  title="Optical γ power on θ phase (group)")

print("\n=== Group γ-on-θ phase significance (cycle-level Rayleigh) ===")
for side in ['lfp','opt']:
    r = agg[side]['rayleigh']
    print(f"{side.upper():>3}: n_cycles={r['n']}, Rbar={r['Rbar']:.3f}, Z={r['Z']:.3f}, p={r['p']:.3g}")
