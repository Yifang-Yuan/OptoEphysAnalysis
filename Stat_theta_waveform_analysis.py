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

import numpy as np
import pandas as pd
from pathlib import Path
import itertools

# -------------------------
# Circular utilities
# -------------------------
def wrap_pi(x):
    x = np.asarray(x)
    return (x + np.pi) % (2*np.pi) - np.pi

def circmean(ang):
    ang = np.asarray(ang)
    return float(np.angle(np.mean(np.exp(1j * ang))))

def circR(ang):
    ang = np.asarray(ang)
    return float(np.abs(np.mean(np.exp(1j * ang))))

def vector_mean_angles(angles_rad):
    """Vector-mean a set of angles (e.g., sweep-wise circular means)."""
    z = np.mean(np.exp(1j * np.asarray(angles_rad)))
    return float(np.angle(z)), float(np.abs(z))

def circ_diff(a, b):
    """Signed circular difference a-b in [-pi, pi)."""
    return float(np.angle(np.exp(1j*(a - b))))

# -------------------------
# Sweep-level analysis: one row per SyncRecording
# -------------------------
def analyse_state_sweepwise(root_dir: Path, behaviour: str, animal_id: str,
                           LFP_channel: str, fs=10000, bp_low=5, bp_high=12,
                           theta_low_thres=0.0, n_resample=200,
                           indicator='GEVI', opt_col='zscore_raw'):
    rows = []
    root_dir = Path(root_dir)

    for rec_name in _iter_sync_recordings(root_dir):
        rec = SyncOEpyPhotometrySession(str(root_dir), rec_name,
                                        IsTracking=False,
                                        read_aligned_data_from_file=True,
                                        recordingMode='Atlas', indicator=indicator)

        seg = _select_theta_segment(rec, behaviour, LFP_channel, theta_low_thres)
        if seg is None or len(seg) == 0:
            continue

        if opt_col not in seg.columns:
            # fall back gracefully if naming differs
            candidates = [c for c in seg.columns if "zscore" in c.lower() and "raw" in c.lower()]
            if len(candidates) == 0:
                raise KeyError(f"[{animal_id} | {behaviour} | {rec_name}] No '{opt_col}' column found. "
                               f"Available columns include: {list(seg.columns)[:20]}")
            opt_use = candidates[0]
        else:
            opt_use = opt_col

        lfp = bandpass_filter(seg[LFP_channel].to_numpy(), fs, bp_low, bp_high)
        opt = bandpass_filter(seg[opt_use].to_numpy(), fs, bp_low, bp_high)

        # per-sweep linear metrics
        lags, xcorr = cross_correlation_theta(lfp, opt, fs)
        lag_peak_s, corr_peak = estimate_peak_lag_from_xcorr(lags, xcorr, win=(-0.25, 0.25))
        plv = phase_locking_value(lfp, opt)
        env_r = amplitude_correlation(lfp, opt)

        # per-sweep cycle-wise phase shift summary
        cyc = cycle_analysis(lfp, opt, fs, n_resample=n_resample, cycle_normalize="zscore")
        ph = cyc.get("cycle_phase_shift", None)
        instf = cyc.get("inst_freq_hz", None)

        if ph is None or len(ph) == 0:
            continue

        ph = wrap_pi(ph)
        mu = circmean(ph)
        R = circR(ph)
        f_theta = float(np.nanmean(instf)) if instf is not None and len(instf) else np.nan

        # optional: convert mean phase shift to ms using sweep theta frequency
        lag_from_phase_ms = np.nan
        if np.isfinite(f_theta) and f_theta > 0:
            lag_from_phase_ms = (mu / (2*np.pi)) * (1000.0 / f_theta)

        rows.append({
            "animal": animal_id,
            "state": behaviour,
            "sweep": rec_name,
            "lfp_channel": LFP_channel,
            "n_cycles": int(len(ph)),
            "theta_freq_hz": f_theta,
            "xcorr_peak_lag_ms": 1000.0 * float(lag_peak_s),
            "xcorr_peak_corr": float(corr_peak),
            "PLV": float(plv),
            "env_r": float(env_r),
            "mu_phase_rad": float(mu),
            "R_phase": float(R),
            "lag_from_phase_ms": float(lag_from_phase_ms),
        })

    return pd.DataFrame(rows)

# -------------------------
# Animal-level aggregation: one row per animal × state
# -------------------------
def summarise_animal_state(df_sweeps_one_animal_state: pd.DataFrame):
    if df_sweeps_one_animal_state.empty:
        return None

    g = df_sweeps_one_animal_state

    # Linear metrics: equal-weight mean across sweeps
    lin_cols = ["xcorr_peak_lag_ms", "PLV", "env_r", "theta_freq_hz", "lag_from_phase_ms"]
    out = {c: float(g[c].mean()) for c in lin_cols if c in g.columns}

    # Circular: vector-mean across sweep means (equal sweep weight)
    mu_sweeps = g["mu_phase_rad"].to_numpy()
    mu_an, R_an = vector_mean_angles(mu_sweeps)

    out.update({
        "animal": g["animal"].iloc[0],
        "state": g["state"].iloc[0],
        "n_sweeps": int(len(g)),
        "n_cycles_total": int(g["n_cycles"].sum()),
        "mu_phase_rad": float(mu_an),
        "R_phase": float(R_an),
    })
    return out

def run_across_animals(ANIMALS, fs=10000, bp_low=5, bp_high=12,
                      theta_low_thres=0.0, n_resample=200,
                      indicator='GEVI', opt_col='zscore_raw'):
    all_sweeps = []

    for a in ANIMALS:
        animal_id = a["animal_id"]
        LFP_channel = a["lfp_channel"]

        state_dirs = {
            "Moving": Path(a["loco_parent"]),
            "Rest":   Path(a["stat_parent"]),
            "REM":    Path(a["rem_parent"]),
        }

        for state, p in state_dirs.items():
            opt_use = a.get("opt_col", opt_col)
            df = analyse_state_sweepwise(
                p, state, animal_id, LFP_channel=LFP_channel,
                fs=fs, bp_low=bp_low, bp_high=bp_high,
                theta_low_thres=theta_low_thres, n_resample=n_resample,
                indicator=indicator, opt_col=opt_use
            )
            all_sweeps.append(df)

    df_sweep = pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame()

    # animal-level summary
    rows = []
    for (animal, state), g in df_sweep.groupby(["animal", "state"]):
        r = summarise_animal_state(g)
        if r is not None:
            rows.append(r)
    df_animal = pd.DataFrame(rows)

    return df_sweep, df_animal

# -------------------------
# Exact permutation tests for n=4 (recommended)
# -------------------------
def exact_signflip_pvalue(values, stat_fn, center_null=True):
    """
    Exact sign-flip test (2^n).
    values: array of within-animal contrasts (e.g., Rest-Moving per animal).
    stat_fn: maps array -> scalar statistic (larger = more extreme).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan

    obs = stat_fn(v)
    # enumerate all sign flips
    count = 0
    total = 0
    for flips in itertools.product([-1, 1], repeat=n):
        flips = np.asarray(flips, float)
        vv = v * flips
        T = stat_fn(vv)
        if T >= obs - 1e-15:
            count += 1
        total += 1
    p = count / total
    return float(obs), float(p)

def exact_within_animal_label_perm_3states(df_animal, metric_col, states=("Moving","Rest","REM")):
    """
    Omnibus test: within each animal, permute state labels (6^n animals, exact).
    Statistic: sum across animals of variance across the 3 state values.
    """
    # build animal-> vector of values in fixed state order
    animals = sorted(df_animal["animal"].unique())
    vals = {}
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if not all(s in g.index for s in states):
            continue
        vals[a] = np.array([g.loc[s, metric_col] for s in states], float)

    animals = list(vals.keys())
    if len(animals) == 0:
        return {"metric": metric_col, "stat_obs": np.nan, "p": np.nan, "n_animals": 0}

    def stat_from_assign(assign_dict):
        # assign_dict[a] is a length-3 array in (Moving,Rest,REM) order after relabelling
        T = 0.0
        for a in animals:
            x = assign_dict[a]
            if np.any(~np.isfinite(x)):
                continue
            T += float(np.var(x, ddof=0))
        return T

    # observed
    obs_assign = {a: vals[a].copy() for a in animals}
    T_obs = stat_from_assign(obs_assign)

    # enumerate 6^n permutations of labels per animal
    perms = list(itertools.permutations([0,1,2], 3))  # 6 permutations
    count = 0
    total = 0
    for choices in itertools.product(range(6), repeat=len(animals)):
        assign = {}
        for ai, c in enumerate(choices):
            a = animals[ai]
            perm = perms[c]
            assign[a] = vals[a][list(perm)]
        T = stat_from_assign(assign)
        if T >= T_obs - 1e-15:
            count += 1
        total += 1

    return {"metric": metric_col, "stat_obs": float(T_obs), "p": float(count/total), "n_animals": len(animals)}

def pairwise_linear_signflip(df_animal, metric_col, s1="Rest", s2="Moving"):
    """
    Pairwise within-animal test: contrast = value(s1)-value(s2), exact sign-flip.
    Statistic: abs(mean(contrast)).
    """
    animals = sorted(df_animal["animal"].unique())
    diffs = []
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if s1 in g.index and s2 in g.index:
            diffs.append(float(g.loc[s1, metric_col] - g.loc[s2, metric_col]))
    diffs = np.asarray(diffs, float)

    stat_fn = lambda x: float(np.abs(np.mean(x)))
    obs, p = exact_signflip_pvalue(diffs, stat_fn)
    return {"metric": metric_col, "pair": f"{s1} - {s2}", "diffs": diffs, "stat_obs": obs, "p": p, "n": int(np.isfinite(diffs).sum())}

def pairwise_circular_signflip(df_animal, s1="Rest", s2="Moving"):
    """
    Pairwise within-animal circular test on mean directions.
    For each animal: delta = wrap(mu_s1 - mu_s2).
    Null: labels exchangeable => delta sign-flips.
    Statistic: 1 - Rbar, equivalently use Rbar as extremeness (larger = more extreme).
    """
    animals = sorted(df_animal["animal"].unique())
    deltas = []
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if s1 in g.index and s2 in g.index:
            d = circ_diff(g.loc[s1, "mu_phase_rad"], g.loc[s2, "mu_phase_rad"])
            deltas.append(d)
    deltas = wrap_pi(np.asarray(deltas, float))

    def stat_fn(x):
        return float(np.abs(np.mean(np.exp(1j*np.asarray(x)))))  # resultant length of mean delta

    R_obs, p = exact_signflip_pvalue(deltas, stat_fn)
    mu_obs = circmean(deltas) if len(deltas) else np.nan

    return {
        "pair": f"{s1} - {s2}",
        "mu_delta_rad": float(mu_obs) if np.isfinite(mu_obs) else np.nan,
        "R_delta": float(R_obs) if np.isfinite(R_obs) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
        "n": int(np.isfinite(deltas).sum())
    }

#%%
'RUN WITH SINGLE ANIMALS'
from pathlib import Path

# locomotion_dir = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion")
# awake_dir      = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary")
# rem_dir        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepREM")

# locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion")
# awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationaryTheta")
# rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepREM")

# locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ALocomotion")
# awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationaryTheta")
# rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepREM")

locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\ALocomotion")
awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\AwakeStationary")
rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\AsleepNonREM")

state_results = analyse_three_states(
    locomotion_dir, awake_dir, rem_dir,
    LFP_channel='LFP_4',   # or your best channel
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

# ================= circular comparisons across states =================

# ---- call it after you have `state_results` ----
stats_summary = run_circular_comparisons(state_results, n_perm=5000, seed=0)
# After you already computed:
# state_results = analyse_three_states(...)
# stats_summary = run_circular_comparisons(state_results, ...)

fig = plot_phase_shift_comparison(state_results, stats_summary=stats_summary, bins=36, savepath=None)

#%%
'MUlTIPLE ANIMALS'

# ANIMALS = [ {"animal_id": "1765508", 
#              "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ALocomotion",
#              "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\AwakeStationary", 
#              "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ASleepREM", 
#              "lfp_channel": "LFP_1"}, 
#            {"animal_id": "1844609", 
#             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ALocomotion", 
#             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\AwakeStationary", 
#             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ASleepREM", 
#             "lfp_channel": "LFP_1"}, 
#            {"animal_id": "1881363", 
#             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ALocomotion",
#             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\AwakeStationary",
#             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ASleepREM", 
#             "lfp_channel": "LFP_1"}, 
#            {"animal_id": "1887933", 
#             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ALocomotion", 
#             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\AwakeStationary", 
#             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ASleepREM", 
#             "lfp_channel": "LFP_2", "opt_col": "ref_raw"},]

ANIMALS = [ 
            {"animal_id": "1842515", 
                         "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\ALocomotion",
                         "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\AwakeStationary", 
                         "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\ASleepREM", 
                         "lfp_channel": "LFP_1"},
            
            {"animal_id": "1842516", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\ASleepNonREM", 
            "lfp_channel": "LFP_4"}, 
            
            {"animal_id": "1844607", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1844607_PV_mNeon_F\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1844607_PV_mNeon_F\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1844607_PV_mNeon_F\ASleepNonREM", 
            "lfp_channel": "LFP_2"}, 
            
            {"animal_id": "1887930", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\ASleepNonREM", 
            "lfp_channel": "LFP_4"}, 
            
            ]

df_sweep, df_animal = run_across_animals(
    ANIMALS,
    fs=10000, bp_low=5, bp_high=12,
    theta_low_thres=0.0,   # your Rest theta gate threshold
    n_resample=200,
    indicator='GEVI',       # change if needed for mCherry sessions
    opt_col='zscore_raw'
)

print("Sweep-level rows:", len(df_sweep))
print(df_sweep.head())

print("\nAnimal-level rows:", len(df_animal))
print(df_animal.sort_values(["animal","state"]))
# Omnibus across 3 states (exact 6^n)
for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "lag_from_phase_ms"]:
    res = exact_within_animal_label_perm_3states(df_animal, metric)
    print(res)

# Pairwise within-animal (exact 2^n)
pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "lag_from_phase_ms"]:
    for s1, s2 in pairs:
        res = pairwise_linear_signflip(df_animal, metric, s1=s1, s2=s2)
        print(res)
# Omnibus across 3 states (exact 6^n)
for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "lag_from_phase_ms"]:
    res = exact_within_animal_label_perm_3states(df_animal, metric)
    print(res)

# Pairwise within-animal (exact 2^n)
pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "lag_from_phase_ms"]:
    for s1, s2 in pairs:
        res = pairwise_linear_signflip(df_animal, metric, s1=s1, s2=s2)
        print(res)
pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
for s1, s2 in pairs:
    res = pairwise_circular_signflip(df_animal, s1=s1, s2=s2)
    # report in degrees for readability
    mu_deg = np.degrees(wrap_pi(res["mu_delta_rad"])) if np.isfinite(res["mu_delta_rad"]) else np.nan
    print(f"{res['pair']}: mu_delta={mu_deg:.1f} deg, R={res['R_delta']:.3f}, p={res['p']:.4g}, n={res['n']}")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _prep_state_order(df, states_order):
    df = df.copy()
    df["state"] = pd.Categorical(df["state"], categories=states_order, ordered=True)
    return df.sort_values(["animal", "state"])

def _format_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"

def _extract_p(stats_list, metric, pair=None):
    """
    stats_list: list of dicts, like your printed outputs.
    metric: e.g. 'xcorr_peak_lag_ms'
    pair: e.g. 'Rest - Moving' (exact string used in your dict)
    """
    for r in stats_list:
        if r.get("metric") != metric:
            continue
        if pair is None and "pair" not in r:
            return r.get("p", None)
        if pair is not None and r.get("pair") == pair:
            return r.get("p", None)
    return None

def annotate_pairwise_p(ax, pvals, states_order=("Moving","Rest","REM"),
                        pad_frac=0.06, step_frac=0.06, fontsize=12):
    """
    Draw pairwise brackets ABOVE the data. Returns (y_top, step) for placing extra text above.
    pvals keys: ('Moving','Rest'), ('Rest','REM'), ('Moving','REM') etc.
    """
    x_map = {s: i for i, s in enumerate(states_order)}

    # Data limits AFTER plotting (robust)
    y_min, y_max = ax.dataLim.y0, ax.dataLim.y1
    yr = (y_max - y_min) if (y_max > y_min) else 1.0

    y_base = y_max + pad_frac * yr
    step = step_frac * yr

    # Order brackets from low to high (adjacent lowest; widest highest)
    bracket_specs = [
        (("Moving", "Rest"), y_base),
        (("Rest",   "REM"),  y_base + step),
        (("Moving", "REM"),  y_base + 2*step),
    ]

    def _format_p(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "p=NA"
        if p < 0.001:
            return "p<0.001"
        return f"p={p:.3g}"

    for (a, b), y in bracket_specs:
        key = (a, b) if (a, b) in pvals else (b, a)
        p = pvals.get(key, None)
        if p is None:
            continue

        xa, xb = x_map[a], x_map[b]

        # bracket
        ax.plot([xa, xa, xb, xb],
                [y - 0.5*step, y, y, y - 0.5*step],
                lw=1.2, color="k")

        # label
        ax.text((xa + xb) / 2, y + 0.15*step, _format_p(p),
                ha="center", va="bottom", fontsize=fontsize)

    # Expand y-limits to guarantee space for brackets
    y_top = y_base + 2.8*step
    ax.set_ylim(y_min - 0.02*yr, y_top)

    return y_top, step


def plot_spaghetti_metric(df_animal, metric, states_order=("Moving","Rest","REM"),
                          ylabel=None, title=None, ax=None, add_legend=False):
    df = _prep_state_order(df_animal, states_order)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))

    x = np.arange(len(states_order))

    # animal lines
    handles = []
    labels = []
    for animal, g in df.groupby("animal"):
        g = g.set_index("state").reindex(states_order)
        y = g[metric].to_numpy(dtype=float)
        h, = ax.plot(x, y, marker='o', linewidth=1.8, alpha=0.9, label=str(animal))
        handles.append(h)
        labels.append(str(animal))

    # mean ± SEM
    means = df.groupby("state")[metric].mean().reindex(states_order).to_numpy()
    sems  = df.groupby("state")[metric].sem().reindex(states_order).to_numpy()
    ax.errorbar(x, means, yerr=sems, fmt='-s', linewidth=2.2, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(states_order)
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.set_title(title if title else metric)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.25)

    if add_legend:
        ax.legend(title="Animal", fontsize=9, frameon=False, loc="best")

    return ax, handles, labels

def plot_two_panel_with_pvalues(df_animal, stats_summary_list,
                                states_order=("Moving","Rest","REM"),
                                figsize=(11, 4.2)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)

    # --- Panel 1: xcorr lag ---
    ax0, handles, labels = plot_spaghetti_metric(
        df_animal, "xcorr_peak_lag_ms",
        ylabel="Xcorr peak lag (ms)",
        title="LFP–GEVI theta lag (xcorr peak)",
        states_order=states_order,
        ax=axes[0],
        add_legend=False
    )

    # extract p-values
    p_omni = _extract_p(stats_summary_list, "xcorr_peak_lag_ms", pair=None)
    p_rm = _extract_p(stats_summary_list, "xcorr_peak_lag_ms", pair="Rest - Moving")
    p_mm = _extract_p(stats_summary_list, "xcorr_peak_lag_ms", pair="REM - Moving")
    p_mr = _extract_p(stats_summary_list, "xcorr_peak_lag_ms", pair="REM - Rest")

    # annotate omnibus in title corner
    # ---- annotate pairwise brackets first (so we know where the top is) ----
    y_top, step = annotate_pairwise_p(
        ax0,
        {("Rest","Moving"): p_rm, ("REM","Moving"): p_mm, ("REM","Rest"): p_mr},
        states_order=states_order
    )
    
    # ---- place omnibus above brackets (x in axes coords, y in data coords) ----
    y_omni = y_top + 0.55*step
    ax0.set_ylim(ax0.get_ylim()[0], y_omni + 0.6*step)  # ensure extra headroom
    ax0.text(0.02, y_omni, f"Omnibus {_format_p(p_omni)}",
             transform=ax0.get_yaxis_transform(),  # x in axes, y in data
             ha="left", va="bottom", fontsize=12)

    # --- Panel 2: PLV ---
    ax1, _, _ = plot_spaghetti_metric(
        df_animal, "PLV",
        ylabel="PLV (theta)",
        title="Phase-locking value (theta)",
        states_order=states_order,
        ax=axes[1],
        add_legend=False
    )

    p_omni = _extract_p(stats_summary_list, "PLV", pair=None)
    p_rm = _extract_p(stats_summary_list, "PLV", pair="Rest - Moving")
    p_mm = _extract_p(stats_summary_list, "PLV", pair="REM - Moving")
    p_mr = _extract_p(stats_summary_list, "PLV", pair="REM - Rest")

    # ---- annotate pairwise brackets first (so we know where the top is) ----
    y_top, step = annotate_pairwise_p(
        ax1,
        {("Rest","Moving"): p_rm, ("REM","Moving"): p_mm, ("REM","Rest"): p_mr},
        states_order=states_order
    )
    
    # ---- place omnibus above brackets (x in axes coords, y in data coords) ----
    y_omni = y_top + 0.55*step
    ax1.set_ylim(ax1.get_ylim()[0], y_omni + 0.6*step)  # ensure extra headroom
    ax1.text(0.02, y_omni, f"Omnibus {_format_p(p_omni)}",
             transform=ax1.get_yaxis_transform(),  # x in axes, y in data
             ha="left", va="bottom", fontsize=12)

    # --- Single shared legend for animals (use the handles from panel 1) ---
    # Place outside to avoid covering data
    fig.legend(handles, labels, title="Animal", frameon=False,
               loc="center right", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.92, 1])  # leave space for the right legend
    plt.show()
    return fig

# -------------------------
# Call it like this:
# stats_summary_list should be the list of dict outputs you printed
# e.g. if you stored them in `all_stats` (list), pass that.
# -------------------------
# fig = plot_two_panel_with_pvalues(df_animal, stats_summary_list=all_stats)

all_stats = []
for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "lag_from_phase_ms"]:
    all_stats.append(exact_within_animal_label_perm_3states(df_animal, metric))

pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "lag_from_phase_ms"]:
    for s1, s2 in pairs:
        all_stats.append(pairwise_linear_signflip(df_animal, metric, s1=s1, s2=s2))

# then plot
plot_two_panel_with_pvalues(df_animal, stats_summary_list=all_stats)


import numpy as np
import matplotlib.pyplot as plt

def wrap_pi(x):
    x = np.asarray(x)
    return (x + np.pi) % (2*np.pi) - np.pi

def plot_phase_polar(df_animal, states_order=("Moving","Rest","REM"), ax=None, title=None):
    df = df_animal.copy()
    df["state"] = pd.Categorical(df["state"], categories=states_order, ordered=True)
    df = df.sort_values(["state","animal"])

    if ax is None:
        fig = plt.figure(figsize=(6.0, 5.4))
        ax = fig.add_subplot(111, projection='polar')

    # put 0 at the top and go clockwise (optional; comment out if you prefer default)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # plot each state in turn
    for s in states_order:
        g = df[df["state"] == s]
        ang = wrap_pi(g["mu_phase_rad"].to_numpy(dtype=float))
        r = np.ones_like(ang)

        ax.scatter(ang, r, label=s, alpha=0.85)

        # group resultant vector across animals for this state
        z = np.mean(np.exp(1j * ang))
        mu = np.angle(z)
        R  = np.abs(z)
        ax.plot([mu, mu], [0, R], linewidth=3)

    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rmax(1.05)
    ax.set_title(title if title else "Animal mean phase direction per state", pad=18)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.15))

    return ax

plot_phase_polar(df_animal, title="Mean phase shift (φ_LFP − φ_GEVI): animal-level directions")
plt.tight_layout()
plt.show()

def plot_pairwise_diffs(df_animal, metric, s1="Rest", s2="Moving", ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))

    # compute within-animal differences
    diffs = []
    animals = sorted(df_animal["animal"].unique())
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if s1 in g.index and s2 in g.index:
            diffs.append((a, float(g.loc[s1, metric] - g.loc[s2, metric])))

    x = np.arange(len(diffs))
    y = np.array([d for _, d in diffs], float)

    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.scatter(x, y)
    for i, (a, d) in enumerate(diffs):
        ax.text(i, d, str(a), fontsize=9, ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([""]*len(x))
    ax.set_ylabel(f"{metric}: {s1} − {s2}")
    ax.set_title(title if title else f"Within-animal contrasts: {s1} − {s2}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.25)
    return ax

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
plot_pairwise_diffs(df_animal, "xcorr_peak_lag_ms", "Rest", "Moving", ax=axes[0])
plot_pairwise_diffs(df_animal, "xcorr_peak_lag_ms", "REM", "Moving",  ax=axes[1])
plot_pairwise_diffs(df_animal, "xcorr_peak_lag_ms", "REM", "Rest",    ax=axes[2])
plt.tight_layout()
plt.show()
#%%

'Plot animal-wide average'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem

# --- helpers ---
def _two_cycles(x, y):
    """
    Duplicate a 0→2π signal to 0→4π.
    Assumes x is 0→2π (endpoint=False) and y has same length.
    """
    x2 = np.concatenate([x, x + 2*np.pi])
    y2 = np.concatenate([y, y])
    return x2, y2

import matplotlib as mpl

def set_large_fonts(base=14):
    mpl.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 2,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 1,
    })

def _wrap_pi(x):
    x = np.asarray(x)
    return (x + np.pi) % (2*np.pi) - np.pi

def _align_cycle_to_lfp_trough(lfp_cycle, gevi_cycle):
    """Roll both so LFP trough is at index 0."""
    k0 = int(np.argmin(lfp_cycle))
    return np.roll(lfp_cycle, -k0), np.roll(gevi_cycle, -k0)

def run_state_per_animal(ANIMALS, fs=10000, bp_low=5, bp_high=12, theta_low_thres=0.0,
                         n_resample=200, max_lag_s=0.5):
    """
    Returns:
      results[state][animal_id] = dict from analyse_state(...)
    """
    results = {"Moving": {}, "Rest": {}, "REM": {}}

    for a in ANIMALS:
        aid = a["animal_id"]
        ch = a["lfp_channel"]

        state_dirs = {
            "Moving": Path(a["loco_parent"]),
            "Rest":   Path(a["stat_parent"]),
            "REM":    Path(a["rem_parent"]),
        }

        for state, p in state_dirs.items():
            try:
                r = analyse_state(
                    p, state, ch,
                    fs=fs, bp_low=bp_low, bp_high=bp_high,
                    theta_low_thres=theta_low_thres,
                    n_resample=n_resample,
                    make_plots=False
                )
                # store animal id for convenience
                r["animal_id"] = aid
                results[state][aid] = r
            except Exception as e:
                print(f"[WARN] {aid} {state}: skipped ({e})")

    return results

def plot_cohort_state_summary(cohort_results, fs=10000, max_lag_s=0.5,
                             n_resample=200, bins=36, show_individual=False):
    """
    cohort_results: output of run_state_per_animal()
    Produces 3 rows (Moving/Rest/REM) × 3 cols (xcorr / cycle mean / phase histogram),
    with animal-level mean ± SEM curves.
    """
    states = ["Moving", "Rest", "REM"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 8.8))
    plt.subplots_adjust(wspace=0.28, hspace=0.45)

    # --- common x grids ---
    lags_common = np.arange(-max_lag_s, max_lag_s + 1/fs, 1/fs)
    phase_bins = np.linspace(-np.pi, np.pi, bins + 1)
    phase_centres = 0.5 * (phase_bins[:-1] + phase_bins[1:])
    x_phase = np.linspace(0, 2*np.pi, n_resample, endpoint=False)

    for i, state in enumerate(states):
        animal_dict = cohort_results.get(state, {})
        if len(animal_dict) == 0:
            continue

        # ========== (1) Cross-correlation ==========
        ax = axes[i, 0]
        corr_stack = []

        for aid, r in animal_dict.items():
            lags = np.asarray(r["xcorr_lags_s"])
            corr = np.asarray(r["xcorr"])

            # interpolate onto common grid
            c_i = np.interp(lags_common, lags, corr)
            corr_stack.append(c_i)

            if show_individual:
                ax.plot(lags_common, c_i, lw=1.0, alpha=0.35)

        corr_stack = np.vstack(corr_stack)
        m = np.nanmean(corr_stack, axis=0)
        s = sem(corr_stack, axis=0, nan_policy='omit')

        ax.plot(lags_common, m, lw=2.2)
        ax.fill_between(lags_common, m - s, m + s, alpha=0.25)
        ax.axvline(0, color='k', ls='--', lw=1.2)
        ax.set_title(f"{state}: Cross-correlation (LFP vs GEVI θ)")
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # ========== (2) Cycle-averaged waveforms ==========
        ax = axes[i, 1]
        lfp_cycles = []
        gevi_cycles = []
        
        for aid, r in animal_dict.items():
            lfp = r.get("lfp_mean_cycle", None)
            gevi = r.get("gevi_mean_cycle", None)
            if lfp is None or gevi is None:
                continue
        
            lfp = np.asarray(lfp).copy()
            gevi = np.asarray(gevi).copy()
        
            # align each animal’s mean cycle to its LFP trough (index 0)
            lfp_a, gevi_a = _align_cycle_to_lfp_trough(lfp, gevi)
            lfp_cycles.append(lfp_a)
            gevi_cycles.append(gevi_a)
        
        lfp_cycles = np.vstack(lfp_cycles)
        gevi_cycles = np.vstack(gevi_cycles)
        
        lfp_m = np.nanmean(lfp_cycles, axis=0); lfp_s = sem(lfp_cycles, axis=0, nan_policy='omit')
        gevi_m = np.nanmean(gevi_cycles, axis=0); gevi_s = sem(gevi_cycles, axis=0, nan_policy='omit')
        
        # ---- base x is 0→2π ----
        x_phase = np.linspace(0, 2*np.pi, n_resample, endpoint=False)
        
        # ---- expand to 2 cycles (0→4π) ----
        x2, lfp_m2  = _two_cycles(x_phase, lfp_m)
        _,  lfp_s2  = _two_cycles(x_phase, lfp_s)
        _,  gevi_m2 = _two_cycles(x_phase, gevi_m)
        _,  gevi_s2 = _two_cycles(x_phase, gevi_s)
        
        ax.plot(x2, lfp_m2, lw=2.6, label="LFP θ")
        ax.fill_between(x2, lfp_m2 - lfp_s2, lfp_m2 + lfp_s2, alpha=0.25)
        
        ax.plot(x2, gevi_m2, lw=2.6, label="GEVI θ")
        ax.fill_between(x2, gevi_m2 - gevi_s2, gevi_m2 + gevi_s2, alpha=0.25)
        
        # mark troughs: LFP trough at 0 (and also at 2π)
        ax.axvline(0, color='k', ls='--', lw=1.5)
        ax.axvline(2*np.pi, color='k', ls='--', lw=1.5)
        
        # GEVI trough from the first-cycle mean, then repeat at +2π
        gevi_tr = int(np.argmin(gevi_m))
        x_gevi_tr = x_phase[gevi_tr]
        ax.axvline(x_gevi_tr, ls='--', lw=1.5)
        ax.axvline(x_gevi_tr + 2*np.pi, ls='--', lw=1.5)
        
        ax.set_title("mean θ waveforms (0 = LFP trough)")
        ax.set_xlabel("Theta phase (rad)")
        ax.set_ylabel("Amplitude")
        
        ax.set_xlim(0, 4*np.pi)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi,
                       5*np.pi/2, 3*np.pi, 7*np.pi/2, 4*np.pi])
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$',
                            r'$5\pi/2$', r'$3\pi$', r'$7\pi/2$', r'$4\pi$'])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:
            ax.legend(frameon=False, ncol=2, loc="lower center")


        # ========== (3) Phase-shift distribution (animal-weighted) ==========
        ax = axes[i, 2]
        dens_stack = []

        for aid, r in animal_dict.items():
            ph = r.get("phase_shift_rad", None)
            if ph is None or len(ph) == 0:
                continue
            ph = _wrap_pi(np.asarray(ph))

            dens, _ = np.histogram(ph, bins=phase_bins, density=True)
            dens_stack.append(dens)

            if show_individual:
                ax.plot(phase_centres, dens, lw=1.0, alpha=0.25)

        dens_stack = np.vstack(dens_stack)
        dm = np.nanmean(dens_stack, axis=0)
        ds = sem(dens_stack, axis=0, nan_policy='omit')

        ax.plot(phase_centres, dm, lw=2.2)
        ax.fill_between(phase_centres, dm - ds, dm + ds, alpha=0.25)

        ax.set_title(f"{state}: Phase-shift distribution")
        ax.set_xlabel(r"Cycle-wise phase shift $(\varphi_{LFP}-\varphi_{GEVI})$")
        ax.set_ylabel("Density")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'-$\pi$', r'-$\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig

# 1) compute per-animal per-state results (no plots)
cohort_results = run_state_per_animal(
    ANIMALS,
    fs=10000, bp_low=5, bp_high=12,
    theta_low_thres=0.0,
    n_resample=200,
    max_lag_s=0.5
)

# 2) plot cohort-level 3×3 summary
fig = plot_cohort_state_summary(
    cohort_results,
    fs=10000,
    max_lag_s=0.5,
    n_resample=200,
    bins=36,
    show_individual=False  # set True if you want thin per-animal curves behind mean±SEM
)

set_large_fonts(base=14)   # try 14–16 for thesis figures
