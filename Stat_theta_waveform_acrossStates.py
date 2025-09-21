# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 20:32:21 2025

@author: yifan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path
import os, re, pickle
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
from scipy.signal import hilbert
from scipy.stats import sem
import importlib.util
# ---------- small helpers (from earlier replies) ----------
def bandpass_filter(x, fs, low=5, high=12, order=4):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)

def _phase(x): return np.angle(hilbert(x))
def _envelope(x): return np.abs(hilbert(x))
def _wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi
def _circmean(ang): return np.angle(np.mean(np.exp(1j*ang)))

def cross_correlation_theta(lfp_theta, opt_theta, fs, max_lag_s=0.5):
    corr = signal.correlate(lfp_theta, opt_theta, mode='full')
    lags = signal.correlation_lags(len(lfp_theta), len(opt_theta), mode='full') / fs
    corr = corr / (np.std(lfp_theta) * np.std(opt_theta) * len(lfp_theta))
    m = (lags >= -max_lag_s) & (lags <= max_lag_s)
    return lags[m], corr[m]

def estimate_peak_lag_from_xcorr(lags_s, corr, win=(-0.25, 0.25)):
    lags_s = np.asarray(lags_s); corr = np.asarray(corr)
    m = (lags_s >= win[0]) & (lags_s <= win[1])
    l, c = lags_s[m], corr[m]
    i = int(np.argmax(c))
    if i == 0 or i == len(c)-1: return float(l[i]), float(c[i])
    y1, y2, y3 = c[i-1], c[i], c[i+1]; x1, x2, x3 = l[i-1], l[i], l[i+1]
    h = x2 - x1; denom = (y1 - 2*y2 + y3)
    if denom == 0: return float(x2), float(y2)
    delta = 0.5 * (y1 - y3) / denom
    return float(x2 + delta*h), float(y2 - 0.25*(y1 - y3)*delta)

def detect_lfp_troughs(lfp_theta, fs, min_period_s=0.08):
    distance = int(min_period_s * fs)
    trough_idx, _ = signal.find_peaks(-lfp_theta, distance=distance)
    return trough_idx

def resample_cycle(x, n=200):
    old = np.linspace(0, 1, len(x), endpoint=True)
    new = np.linspace(0, 1, n, endpoint=True)
    return np.interp(new, old, x)

def cycle_analysis(lfp_theta, opt_theta, fs, n_resample=200, cycle_normalize="zscore"):
    troughs = detect_lfp_troughs(lfp_theta, fs)
    troughs = troughs[(troughs > 0) & (troughs < len(lfp_theta)-1)]
    if len(troughs) < 3:
        return dict(cycle_phase_shift=np.array([]), lfp_cycles=None, opt_cycles=None, inst_freq_hz=np.array([]))
    phi_l, phi_o = _phase(lfp_theta), _phase(opt_theta)
    phase_shift, lfp_stack, opt_stack, instf = [], [], [], []
    for i in range(len(troughs)-1):
        a, b = troughs[i], troughs[i+1]
        if b <= a+2: continue
        instf.append(1.0 / ((b-a)/fs))
        dphi = _wrap_pi(phi_l[a:b] - phi_o[a:b])
        phase_shift.append(_circmean(dphi))
        lc = resample_cycle(lfp_theta[a:b], n_resample)
        oc = resample_cycle(opt_theta[a:b], n_resample)
        if cycle_normalize == "zscore":
            eps = 1e-12
            lc = (lc - lc.mean()) / (lc.std()+eps)
            oc = (oc - oc.mean()) / (oc.std()+eps)
        lfp_stack.append(lc); opt_stack.append(oc)
    return {
        "cycle_phase_shift": np.array(phase_shift),
        "lfp_cycles": np.vstack(lfp_stack) if lfp_stack else None,
        "opt_cycles": np.vstack(opt_stack) if opt_stack else None,
        "inst_freq_hz": np.array(instf),
    }

def align_cycles_to_lfp_trough(lfp_cycles, opt_cycles):
    n = lfp_cycles.shape[1]; centre = n//2
    l_al = np.empty_like(lfp_cycles); o_al = np.empty_like(opt_cycles)
    for i in range(lfp_cycles.shape[0]):
        k = int(np.argmin(lfp_cycles[i]))
        shift = centre - k
        l_al[i] = np.roll(lfp_cycles[i], shift)
        o_al[i] = np.roll(opt_cycles[i], shift)
    return l_al, o_al

def phase_locking_value(lfp_theta, opt_theta):
    dphi = _wrap_pi(_phase(lfp_theta) - _phase(opt_theta))
    return float(np.abs(np.mean(np.exp(1j*dphi))))

def amplitude_correlation(lfp_theta, opt_theta):
    return float(np.corrcoef(_envelope(lfp_theta), _envelope(opt_theta))[0,1])

# ---------- state-wise loader ----------
def _iter_sync_recordings(root: Path):
    for p in sorted(root.glob("SyncRecording*")):
        if p.is_dir():
            yield p.name

def _select_theta_segment(rec: SyncOEpyPhotometrySession, behaviour: str, LFP_channel: str, theta_low_thres: float):
    if behaviour == "Moving":
        df = rec.Ephys_tracking_spad_aligned
        return df[df["movement"] == "moving"]
    if behaviour == "Rest":
        rec.pynacollada_label_theta(LFP_channel, Low_thres=theta_low_thres, High_thres=10, save=False, plot_theta=False)
        return rec.theta_part
    if behaviour == "REM":
        rec.Label_REM_sleep(LFP_channel)
        df = rec.Ephys_tracking_spad_aligned
        return df[df["REMstate"] == "REM"]
    raise ValueError("Unknown behaviour")

# ---------- main routine per state ----------
def analyse_state(root_dir: Path, behaviour: str, LFP_channel: str, fs=10000, bp_low=5, bp_high=12,
                  theta_low_thres=0.0, n_resample=200, make_plots=True):
    all_cycles_lfp, all_cycles_opt = [], []
    all_phase_shifts = []
    concat_lfp, concat_opt = [], []

    for rec_name in _iter_sync_recordings(root_dir):
        rec = SyncOEpyPhotometrySession(str(root_dir), rec_name,
                                        IsTracking=False,
                                        read_aligned_data_from_file=True,
                                        recordingMode='Atlas', indicator='GEVI')
        seg = _select_theta_segment(rec, behaviour, LFP_channel, theta_low_thres)
        if seg is None or len(seg)==0:
            continue

        lfp = bandpass_filter(seg[LFP_channel].to_numpy(), fs, bp_low, bp_high)
        opt = bandpass_filter(seg["zscore_raw"].to_numpy(), fs, bp_low, bp_high)

        # concat for cross-corr/PLV
        concat_lfp.append(lfp); concat_opt.append(opt)

        # cycles
        cyc = cycle_analysis(lfp, opt, fs, n_resample=n_resample, cycle_normalize="zscore")
        if cyc["lfp_cycles"] is not None:
            all_cycles_lfp.append(cyc["lfp_cycles"])
            all_cycles_opt.append(cyc["opt_cycles"])
            all_phase_shifts.append(cyc["cycle_phase_shift"])

    if len(concat_lfp) == 0:
        raise RuntimeError(f"No usable segments found for {behaviour}")

    lfp_cat = np.concatenate(concat_lfp)
    opt_cat = np.concatenate(concat_opt)

    # metrics on concatenated signals
    lags, xcorr = cross_correlation_theta(lfp_cat, opt_cat, fs)
    lag_peak_s, corr_peak = estimate_peak_lag_from_xcorr(lags, xcorr, win=(-0.25, 0.25))
    plv = phase_locking_value(lfp_cat, opt_cat)
    env_r = amplitude_correlation(lfp_cat, opt_cat)

    # pooled cycles
    if len(all_cycles_lfp):
        lfp_pool = np.vstack(all_cycles_lfp)
        opt_pool = np.vstack(all_cycles_opt)
        lfp_al, opt_al = align_cycles_to_lfp_trough(lfp_pool, opt_pool)

        lfp_mean = np.nanmean(lfp_al, axis=0); lfp_sem = sem(lfp_al, axis=0, nan_policy='omit')
        opt_mean = np.nanmean(opt_al, axis=0); opt_sem = sem(opt_al, axis=0, nan_policy='omit')
        phase_shifts = np.concatenate(all_phase_shifts)
    else:
        lfp_mean = opt_mean = lfp_sem = opt_sem = None
        phase_shifts = np.array([])

    # ------- plots --------
    if make_plots:
        # (1) cross-correlation
        plt.figure(figsize=(6.5,3.6)); ax = plt.gca()
        ax.plot(lags, xcorr, lw=2); ax.axvline(0, color='k', ls='--')
        ax.set_xlabel('Lag (s)', fontsize=15); ax.set_ylabel('Correlation', fontsize=15)
        ax.set_title(f'{behaviour}: Cross-correlation (LFP vs GEVI θ)', fontsize=15)
        ax.tick_params(labelsize=13); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); plt.show()

        # (2) cycle-averaged waveforms in radians (0→2π), troughs marked
        if lfp_mean is not None:
            # ---- cycle-averaged waveforms (0 rad = LFP trough) ----
            plt.figure(figsize=(7,4))
            ax = plt.gca()
            
            # 1) roll so LFP trough is at index 0 (phase = 0 rad)
            k0 = int(np.argmin(lfp_mean))
            lfp_mean_r = np.roll(lfp_mean, -k0)
            lfp_sem_r  = np.roll(lfp_sem,  -k0)
            opt_mean_r = np.roll(opt_mean, -k0)
            opt_sem_r  = np.roll(opt_sem,  -k0)
            
            # 2) x in radians, 0→2π (endpoint=False avoids duplicating 0 at 2π)
            n = len(lfp_mean_r)
            x = np.linspace(0, 2*np.pi, n, endpoint=False)
            
            # 3) plot mean ± SEM
            sl = slice(1, None)
            ax.plot(x[sl], lfp_mean_r[sl], label='LFP θ (mean)', color='C0')
            ax.fill_between(x[sl], (lfp_mean_r-lfp_sem_r)[sl], (lfp_mean_r+lfp_sem_r)[sl], alpha=0.25, color='C0')
            ax.plot(x[sl], opt_mean_r[sl], label='GEVI θ (mean)', color='C1')
            ax.fill_between(x[sl], (opt_mean_r-opt_sem_r)[sl], (opt_mean_r+opt_sem_r)[sl], alpha=0.25, color='C1')
            
            # 4) mark troughs (LFP at 0 by construction; GEVI from rolled mean)
            ax.axvline(2*np.pi, color='C0', ls='--', lw=1.5, label='LFP trough')
            gevi_trough_idx = int(np.argmin(opt_mean_r))
            ax.axvline(x[gevi_trough_idx], color='C1', ls='--', lw=1.5, label='GEVI trough')
            
            # (optional) mark peaks
            # lfp_peak_idx  = int(np.argmax(lfp_mean_r))
            # gevi_peak_idx = int(np.argmax(opt_mean_r))
            # ax.axvline(x[lfp_peak_idx],  color='C0', ls=':', lw=1.5, label='LFP peak')
            # ax.axvline(x[gevi_peak_idx], color='C1', ls=':', lw=1.5, label='GEVI peak')
            
            # 5) labels, ticks, and style
            ax.set_xlabel('Theta phase (rad)', fontsize=16)
            ax.set_ylabel('Amplitude\n(normalised per cycle)', fontsize=16)
            ax.set_title('Cycle-averaged waveforms (0 = LFP trough)', fontsize=16)
            ax.legend(fontsize=12, ncol=2)
            
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=14)
            
            ax.tick_params(axis='both', labelsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.show()


        # (3) phase-shift histogram
        if phase_shifts.size:
            plt.figure(figsize=(5,3.6)); ax = plt.gca()
            bins = np.linspace(-np.pi, np.pi, 37)
            ax.hist(_wrap_pi(phase_shifts), bins=bins, density=True, alpha=0.85)
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r'-$\pi$', r'-$\pi/2$', '0', r'$\pi/2$', r'$\pi$'], fontsize=13)
            ax.set_xlabel('Cycle-wise phase shift (rad)\n(φ_LFP − φ_GEVI)', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.set_title(f'{behaviour}: Phase-shift distribution', fontsize=15)
            ax.tick_params(labelsize=13); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout(); plt.show()

    results = {
        "behaviour": behaviour,
        "n_sweeps": len(list(_iter_sync_recordings(root_dir))),
        "n_cycles": 0 if phase_shifts.size == 0 else int(len(phase_shifts)),
        "xcorr_lags_s": lags, "xcorr": xcorr,
        "xcorr_peak_lag_s": lag_peak_s, "xcorr_peak_corr": corr_peak,
        "PLV_global": plv, "Envelope_r_global": env_r,
        "phase_shift_rad": phase_shifts,
        "lfp_mean_cycle": lfp_mean, "lfp_sem_cycle": lfp_sem,
        "gevi_mean_cycle": opt_mean, "gevi_sem_cycle": opt_sem,
    }
    return results

# ---------- convenience wrapper for the three states ----------
def analyse_three_states(locomotion_dir, awake_dir, rem_dir, LFP_channel='LFP_1',
                         fs=10000, theta_low_thres=0.0, bp_low=5, bp_high=12,
                         n_resample=200, make_plots=True):
    state_dirs = {
        "Moving": Path(locomotion_dir),
        "Rest":   Path(awake_dir),
        "REM":    Path(rem_dir),
    }
    out = {}
    for behav, p in state_dirs.items():
        print(f"\n=== {behav} | {p} ===")
        out[behav] = analyse_state(
            p, behav, LFP_channel, fs=fs, bp_low=bp_low, bp_high=bp_high,
            theta_low_thres=theta_low_thres, n_resample=n_resample, make_plots=make_plots
        )
    return out


def run_circular_comparisons(state_results, n_perm=5000, seed=0):
    import numpy as np

    # Gather phase-shift arrays (wrapped to [-pi, pi))
    def _wrap_pi(x): return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi
    groups = {}
    for name in ["Moving", "Rest", "REM"]:
        arr = state_results.get(name, {}).get("phase_shift_rad", np.array([]))
        if arr is None or len(arr) == 0:
            raise ValueError(f"No phase-shift data for state: {name}")
        groups[name] = _wrap_pi(arr)

    names = list(groups.keys())

    # Pretty print helper
    def _print_header(txt):
        print("\n" + "="*len(txt))
        print(txt)
        print("="*len(txt))

    results = {"omnibus": None, "pairwise": []}

    # ---------- try parametric circular tests (pycircstat) ----------
    try:
        import pycircstat as circ

        # Omnibus: Watson–Williams (≈ circular ANOVA)
        F, p = circ.tests.watson_williams(*[groups[n] for n in names])
        results["omnibus"] = {"test": "Watson–Williams", "stat": float(F), "p": float(p)}

        # Pairwise: Watson two-sample (nonparametric)
        pair_ps = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                U2, p_ = circ.tests.watson_two(groups[names[i]], groups[names[j]])
                results["pairwise"].append({
                    "pair": f"{names[i]} vs {names[j]}",
                    "test": "Watson two-sample",
                    "stat": float(U2),
                    "p": float(p_)
                })
                pair_ps.append(float(p_))

    except Exception:
        # ---------- fallback: permutation on circular mean direction ----------
        rng = np.random.default_rng(seed)

        def circ_mean(a):
            return np.angle(np.mean(np.exp(1j*np.asarray(a))))

        # simple omnibus via max pairwise difference (permutation)
        all_vals = np.concatenate([groups[n] for n in names])
        lengths = [len(groups[n]) for n in names]
        means_obs = [circ_mean(groups[n]) for n in names]
        # test statistic: max absolute pairwise difference in mean direction
        def circ_diff(a, b):
            return np.abs(np.angle(np.exp(1j*(a-b))))
        T_obs = max(circ_diff(means_obs[i], means_obs[j])
                    for i in range(len(names)) for j in range(i+1, len(names)))

        count = 0
        for _ in range(n_perm):
            rng.shuffle(all_vals)
            splits = np.cumsum(lengths[:-1])
            parts = np.split(all_vals, splits)
            mus = [circ_mean(p) for p in parts]
            T = max(circ_diff(mus[i], mus[j]) for i in range(len(names)) for j in range(i+1, len(names)))
            if T >= T_obs:
                count += 1
        p_omni = (count + 1) / (n_perm + 1)
        results["omnibus"] = {"test": "Permutation omnibus (mean direction)", "stat": float(T_obs), "p": float(p_omni)}

        # Pairwise permutation tests
        pair_ps = []
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                A, B = groups[names[i]], groups[names[j]]
                obs = circ_diff(circ_mean(A), circ_mean(B))
                comb = np.concatenate([A, B]); nA = len(A)
                cnt = 0
                for _ in range(n_perm):
                    rng.shuffle(comb)
                    muA = circ_mean(comb[:nA]); muB = circ_mean(comb[nA:])
                    stat = circ_diff(muA, muB)
                    if stat >= obs: cnt += 1
                p_pair = (cnt + 1) / (n_perm + 1)
                results["pairwise"].append({
                    "pair": f"{names[i]} vs {names[j]}",
                    "test": "Permutation (mean direction)",
                    "stat": float(obs),
                    "p": float(p_pair)
                })
                pair_ps.append(p_pair)

    # ---------- FDR (Benjamini–Hochberg) on pairwise p-values ----------
    if results["pairwise"]:
        ps = np.array([r["p"] for r in results["pairwise"]], float)
        order = np.argsort(ps); m = len(ps)
        q = np.empty_like(ps)
        cummin = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            i = order.size - rank + 1  # BH uses i = rank in ascending order
            val = ps[idx] * m / i
            cummin = min(cummin, val)
            q[idx] = cummin
        for r, qv in zip(results["pairwise"], q):
            r["q_FDR"] = float(min(1.0, qv))

    # ---------- print summary ----------
    _print_header("Circular comparisons across behavioural states")
    om = results["omnibus"]
    print(f"Omnibus: {om['test']}: stat = {om['stat']:.3f}, p = {om['p']:.4g}")

    print("\nPairwise comparisons (FDR-corrected):")
    for r in results["pairwise"]:
        extra = f", q(FDR) = {r['q_FDR']:.4g}" if "q_FDR" in r else ""
        print(f"  {r['pair']}: {r['test']} → stat = {r['stat']:.3f}, p = {r['p']:.4g}{extra}")

    return results

import numpy as np
import matplotlib.pyplot as plt

def _wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi

def _circmean(rad):
    return float(np.angle(np.mean(np.exp(1j*np.asarray(rad)))))

def _deg(x):  # radians -> degrees in [-180, 180)
    return (np.degrees(_wrap_pi(x)))

def plot_phase_shift_comparison(state_results, stats_summary=None, bins=36, savepath=None):
    """
    Make two panels:
      A) Overlaid density histograms of (φ_LFP − φ_GEVI) for Moving / Rest / REM
      B) Violin plots per state (angles in degrees)
    If stats_summary is provided (from run_circular_comparisons), annotate pairwise q-values.
    """
    # --- gather & wrap
    states_order = ["Moving", "Rest", "REM"]
    colors = {"Moving":"C0", "Rest":"C1", "REM":"C2"}
    data_deg = {}
    means_deg = {}

    for s in states_order:
        rad = state_results[s]["phase_shift_rad"]
        deg = _deg(rad)
        data_deg[s] = deg
        means_deg[s] = _deg(_circmean(rad))

    # ---------- Figure ----------
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.3)

    # ---- (A) Overlaid densities ----
    ax = fig.add_subplot(gs[0, 0])
    bin_edges = np.linspace(-180, 180, bins+1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for s in states_order:
        counts, _ = np.histogram(data_deg[s], bins=bin_edges, density=True)
        ax.plot(bin_centres, counts, lw=2, label=s, color=colors[s])
        # circular mean marker
        ax.axvline(means_deg[s], color=colors[s], ls='--', lw=1.6, alpha=0.9)

    ax.set_xlabel('Cycle-wise phase shift (deg)\n(φ_LFP − φ_GEVI)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_title('Overlaid phase-shift distributions', fontsize=14)
    ax.legend(fontsize=11, frameon=False)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ---- (B) Violin plots per state ----
    ax2 = fig.add_subplot(gs[0, 1])

    # To avoid wrap artifacts, jitter each distribution slightly (visual only)
    vio_data = [data_deg[s] for s in states_order]
    parts = ax2.violinplot(vio_data, showmeans=False, showmedians=False, showextrema=False)

    # Color violins
    for i, b in enumerate(parts['bodies']):
        s = states_order[i]
        b.set_facecolor(colors[s]); b.set_alpha(0.25); b.set_edgecolor('none')

    # Overlay circular means (thick lines)
    for i, s in enumerate(states_order, start=1):
        mu = means_deg[s]
        ax2.plot([i-0.25, i+0.25], [mu, mu], color=colors[s], lw=3, solid_capstyle='round')

    # Scatter raw (downsample if huge)
    rng = np.random.default_rng(0)
    for i, s in enumerate(states_order, start=1):
        d = data_deg[s]
        if len(d) > 1500:  # light downsample for plotting speed
            idx = rng.choice(len(d), 1500, replace=False)
            d = d[idx]
        xj = i + 0.04 * rng.standard_normal(len(d))
        ax2.plot(xj, d, '.', ms=2.2, alpha=0.25, color=colors[s])

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(states_order, fontsize=12)
    ax2.set_ylabel('Phase shift (deg)\n(φ_LFP − φ_GEVI)', fontsize=14)
    ax2.set_ylim(-180, 180)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_title('Violin comparison across states', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # ---- Annotate q-values if provided ----
    if stats_summary and stats_summary.get("pairwise"):
        # Map pair names to q
        qmap = {}
        for r in stats_summary["pairwise"]:
            pair = r["pair"].replace(" ", "")
            qmap[pair] = r.get("q_FDR", r["p"])  # fall back to p if q not present

        def qtxt(q):
            return f"q={q:.3g}" if q >= 0.001 else "q<0.001"

        # Place annotations between category positions (1,2,3)
        pairs_pos = {
            ("Moving","Rest"): (1, 2, 150),   # lower bar
            ("Rest","REM")  : (2, 3, 155),   # middle bar
            ("Moving","REM"): (1, 3, 180),   # top bar
        }
        # extend y-limits so the top bar fits
        for (a,b),(xa,xb,y) in pairs_pos.items():
            key1, key2 = f"{a}vs{b}", f"{b}vs{a}"
            qv = qmap.get(key1, qmap.get(key2))
            if qv is None: 
                continue
            # bracket
            ax2.plot([xa, xb], [y, y], color='k', lw=1.2)
            ax2.plot([xa, xa], [y-6, y], color='k', lw=1.2)
            ax2.plot([xb, xb], [y-6, y], color='k', lw=1.2)
            # label with small horizontal nudge for aesthetics
            xmid = (xa + xb) / 2
            xmid += {-1.0: -0.02, 0.0: 0.0, 1.0: 0.02}.get(np.sign(xb-xa), 0.0)
            ax2.text(xmid, y + 3, qtxt(qv), ha='center', va='bottom', fontsize=11)
        ax2.set_ylim(-180, 200)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

#%%
from pathlib import Path

locomotion_dir = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion")
awake_dir      = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary")
rem_dir        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepREM")

# locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion")
# awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationaryTheta")
# rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepREM")

# locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ALocomotion")
# awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationaryTheta")
# rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepREM")

state_results = analyse_three_states(
    locomotion_dir, awake_dir, rem_dir,
    LFP_channel='LFP_1',   # or your best channel
    fs=10000,
    theta_low_thres=0.0,   # adjust if you use a threshold for Rest
    bp_low=5, bp_high=12,
    n_resample=200,
    make_plots=True
)

# Example: access numbers
for k, r in state_results.items():
    lag_ms = 1000 * r["xcorr_peak_lag_s"]
    print(f"{k}: xcorr peak lag = {lag_ms:+.1f} ms, PLV = {r['PLV_global']:.3f}, "
          f"env r = {r['Envelope_r_global']:.3f}, n_cycles = {r['n_cycles']}")
#%%

# ================= circular comparisons across states =================

# ---- call it after you have `state_results` ----
stats_summary = run_circular_comparisons(state_results, n_perm=5000, seed=0)
# After you already computed:
# state_results = analyse_three_states(...)
# stats_summary = run_circular_comparisons(state_results, ...)

fig = plot_phase_shift_comparison(state_results, stats_summary=stats_summary, bins=36, savepath=None)
