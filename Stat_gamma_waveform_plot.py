# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 23:01:53 2026

@author: yifan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path
from scipy.signal import hilbert
from scipy.stats import sem
from SyncOECPySessionClass import SyncOEpyPhotometrySession
# -------------------------
# Reuse / keep your helpers
# -------------------------
def bandpass_filter(x, fs, low, high, order=4):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)

def _phase(x): return np.angle(hilbert(x))
def _envelope(x): return np.abs(hilbert(x))
def _wrap_pi(x): return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi
def _circmean(ang): return float(np.angle(np.mean(np.exp(1j*np.asarray(ang)))))
def _circR(ang): return float(np.abs(np.mean(np.exp(1j*np.asarray(ang)))))

def cross_correlation_band(x, y, fs, max_lag_s=0.05):
    corr = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(len(x), len(y), mode='full') / fs
    corr = corr / (np.std(x) * np.std(y) * len(x))
    m = (lags >= -max_lag_s) & (lags <= max_lag_s)
    return lags[m], corr[m]

def estimate_peak_lag_from_xcorr(lags_s, corr, win=(-0.02, 0.02)):
    lags_s = np.asarray(lags_s); corr = np.asarray(corr)
    m = (lags_s >= win[0]) & (lags_s <= win[1])
    l, c = lags_s[m], corr[m]
    i = int(np.argmax(c))
    if i == 0 or i == len(c)-1:
        return float(l[i]), float(c[i])
    y1, y2, y3 = c[i-1], c[i], c[i+1]
    x1, x2, x3 = l[i-1], l[i], l[i+1]
    h = x2 - x1
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return float(x2), float(y2)
    delta = 0.5 * (y1 - y3) / denom
    return float(x2 + delta*h), float(y2 - 0.25*(y1 - y3)*delta)

def phase_locking_value(x, y):
    dphi = _wrap_pi(_phase(x) - _phase(y))
    return float(np.abs(np.mean(np.exp(1j*dphi))))

def amplitude_correlation(x, y):
    return float(np.corrcoef(_envelope(x), _envelope(y))[0, 1])

def resample_cycle(x, n=200):
    old = np.linspace(0, 1, len(x), endpoint=True)
    new = np.linspace(0, 1, n, endpoint=True)
    return np.interp(new, old, x)

def align_cycles_to_lfp_trough(lfp_cycles, opt_cycles):
    n = lfp_cycles.shape[1]; centre = n//2
    l_al = np.empty_like(lfp_cycles); o_al = np.empty_like(opt_cycles)
    for i in range(lfp_cycles.shape[0]):
        k = int(np.argmin(lfp_cycles[i]))
        shift = centre - k
        l_al[i] = np.roll(lfp_cycles[i], shift)
        o_al[i] = np.roll(opt_cycles[i], shift)
    return l_al, o_al

# -------------------------
# Your existing loaders
# -------------------------
def _iter_sync_recordings(root: Path):
    for p in sorted(Path(root).glob("SyncRecording*")):
        if p.is_dir():
            yield p.name

def _select_state_segment(rec, behaviour: str, LFP_channel: str, theta_low_thres: float):
    """
    Kept identical logic to your theta selection, so Moving/Rest/REM are comparable.
    """
    if behaviour == "Moving":
        df = rec.Ephys_tracking_spad_aligned
        return df[df["movement"] == "moving"]
    if behaviour == "Rest":
        rec.pynacollada_label_theta(LFP_channel, Low_thres=theta_low_thres, High_thres=10,
                                    save=False, plot_theta=False)
        return rec.theta_part
    if behaviour == "REM":
        rec.Label_REM_sleep(LFP_channel)
        df = rec.Ephys_tracking_spad_aligned
        return df[df["REMstate"] == "REM"]
    raise ValueError("Unknown behaviour")

# -------------------------
# Gamma-specific: gating + cycle segmentation
# -------------------------
def _mask_to_segments(mask: np.ndarray, min_len: int):
    """Return list of (start, end) indices where mask is True and length >= min_len."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    segs = []
    for a, b in zip(starts, ends):
        if (b - a) >= min_len:
            segs.append((a, b))
    return segs

def detect_lfp_troughs_band(lfp_band, fs, f_high, safety=0.8):
    """
    Trough detection distance derived from highest frequency bound.
    safety<1 gives a slightly smaller min distance to avoid missing cycles.
    """
    min_period_s = safety * (1.0 / float(f_high))
    distance = max(1, int(min_period_s * fs))
    trough_idx, _ = signal.find_peaks(-lfp_band, distance=distance)
    return trough_idx

def cycle_analysis_band(lfp_band, opt_band, fs, f_high, n_resample=200, cycle_normalize="zscore"):
    troughs = detect_lfp_troughs_band(lfp_band, fs, f_high=f_high, safety=0.8)
    troughs = troughs[(troughs > 0) & (troughs < len(lfp_band)-1)]
    if len(troughs) < 3:
        return dict(cycle_phase_shift=np.array([]), lfp_cycles=None, opt_cycles=None, inst_freq_hz=np.array([]))

    phi_l, phi_o = _phase(lfp_band), _phase(opt_band)
    phase_shift, lfp_stack, opt_stack, instf = [], [], [], []

    for i in range(len(troughs)-1):
        a, b = troughs[i], troughs[i+1]
        if b <= a + 2:
            continue

        instf.append(1.0 / ((b-a)/fs))
        dphi = _wrap_pi(phi_l[a:b] - phi_o[a:b])
        phase_shift.append(_circmean(dphi))

        lc = resample_cycle(lfp_band[a:b], n_resample)
        oc = resample_cycle(opt_band[a:b], n_resample)

        if cycle_normalize == "zscore":
            eps = 1e-12
            lc = (lc - lc.mean()) / (lc.std() + eps)
            oc = (oc - oc.mean()) / (oc.std() + eps)

        lfp_stack.append(lc); opt_stack.append(oc)

    return {
        "cycle_phase_shift": np.asarray(phase_shift),
        "lfp_cycles": np.vstack(lfp_stack) if lfp_stack else None,
        "opt_cycles": np.vstack(opt_stack) if opt_stack else None,
        "inst_freq_hz": np.asarray(instf),
    }

def _two_cycles(x, y):
    x2 = np.concatenate([x, x + 2*np.pi])
    y2 = np.concatenate([y, y])
    return x2, y2

def _align_cycle_to_lfp_trough_1cycle(lfp_cycle, opt_cycle):
    k0 = int(np.argmin(lfp_cycle))
    return np.roll(lfp_cycle, -k0), np.roll(opt_cycle, -k0)

# -------------------------
# Main per-state gamma analysis (single animal, pooled sweeps)
# -------------------------
def analyse_state_gamma(root_dir: Path, behaviour: str, LFP_channel: str,
                        fs=10000,
                        gamma_band=(30, 55),
                        theta_low_thres=0.0,
                        opt_col="zscore_raw",
                        indicator="GEVI",
                        gate_quantile=0.7,
                        gate_on="lfp",          # "lfp" or "both"
                        min_epoch_s=0.10,       # require >=100 ms high-gamma run
                        n_resample=200,
                        max_lag_s=0.05,
                        xcorr_win=(-0.02, 0.02),
                        make_plots=True):
    """
    Gamma analysis with amplitude gating.
    Returns dict similar to your theta results, plus gating diagnostics.
    """
    root_dir = Path(root_dir)
    f_low, f_high = gamma_band

    all_cycles_lfp, all_cycles_opt = [], []
    all_phase_shifts = []
    concat_lfp, concat_opt = [], []
    concat_env_l, concat_env_o = [], []
    n_epochs_total = 0
    n_samples_total = 0
    n_samples_gated = 0

    for rec_name in _iter_sync_recordings(root_dir):
        rec = SyncOEpyPhotometrySession(str(root_dir), rec_name,
                                        IsTracking=False,
                                        read_aligned_data_from_file=True,
                                        recordingMode='Atlas', indicator=indicator)

        seg = _select_state_segment(rec, behaviour, LFP_channel, theta_low_thres)
        if seg is None or len(seg) == 0:
            continue

        if opt_col not in seg.columns:
            candidates = [c for c in seg.columns if "zscore" in c.lower() and "raw" in c.lower()]
            if len(candidates) == 0:
                raise KeyError(f"[{behaviour} | {rec_name}] No '{opt_col}' found. "
                               f"Columns include: {list(seg.columns)[:30]}")
            opt_use = candidates[0]
        else:
            opt_use = opt_col

        lfp_raw = seg[LFP_channel].to_numpy()
        opt_raw = seg[opt_use].to_numpy()

        n_samples_total += len(lfp_raw)

        # bandpass
        lfp_g = bandpass_filter(lfp_raw, fs, f_low, f_high, order=4)
        opt_g = bandpass_filter(opt_raw, fs, f_low, f_high, order=4)

        # gating on gamma envelope (recommended)
        env_l = _envelope(lfp_g)
        env_o = _envelope(opt_g)

        th_l = np.quantile(env_l, gate_quantile)
        mask = env_l > th_l
        if gate_on == "both":
            th_o = np.quantile(env_o, gate_quantile)
            mask = mask & (env_o > th_o)

        segs = _mask_to_segments(mask, min_len=int(min_epoch_s * fs))
        if len(segs) == 0:
            continue

        n_epochs_total += len(segs)

        # extract epochs
        for a, b in segs:
            lfp_e = lfp_g[a:b]
            opt_e = opt_g[a:b]
            env_le = env_l[a:b]
            env_oe = env_o[a:b]

            n_samples_gated += len(lfp_e)

            # concat for "global" metrics
            concat_lfp.append(lfp_e); concat_opt.append(opt_e)
            concat_env_l.append(env_le); concat_env_o.append(env_oe)

            # cycle-wise phase shifts & cycle-average waveforms (visual supplement)
            cyc = cycle_analysis_band(lfp_e, opt_e, fs, f_high=f_high, n_resample=n_resample, cycle_normalize="zscore")
            if cyc["lfp_cycles"] is not None:
                all_cycles_lfp.append(cyc["lfp_cycles"])
                all_cycles_opt.append(cyc["opt_cycles"])
                all_phase_shifts.append(cyc["cycle_phase_shift"])

    if len(concat_lfp) == 0:
        raise RuntimeError(f"No usable high-gamma epochs found for {behaviour} in {root_dir} "
                           f"(try lowering gate_quantile or min_epoch_s).")

    lfp_cat = np.concatenate(concat_lfp)
    opt_cat = np.concatenate(concat_opt)
    env_l_cat = np.concatenate(concat_env_l)
    env_o_cat = np.concatenate(concat_env_o)

    # metrics on concatenated gated band signals
    lags, xcorr = cross_correlation_band(lfp_cat, opt_cat, fs, max_lag_s=max_lag_s)
    lag_peak_s, corr_peak = estimate_peak_lag_from_xcorr(lags, xcorr, win=xcorr_win)

    plv = phase_locking_value(lfp_cat, opt_cat)
    env_r = float(np.corrcoef(env_l_cat, env_o_cat)[0, 1])

    # envelope lag (often more stable than raw gamma xcorr)
    lags_e, xcorr_e = cross_correlation_band(env_l_cat - env_l_cat.mean(),
                                             env_o_cat - env_o_cat.mean(),
                                             fs, max_lag_s=0.5)
    env_lag_peak_s, env_corr_peak = estimate_peak_lag_from_xcorr(lags_e, xcorr_e, win=(-0.25, 0.25))

    # pooled cycles
    if len(all_cycles_lfp):
        lfp_pool = np.vstack(all_cycles_lfp)
        opt_pool = np.vstack(all_cycles_opt)
        lfp_al, opt_al = align_cycles_to_lfp_trough(lfp_pool, opt_pool)

        lfp_mean = np.nanmean(lfp_al, axis=0); lfp_sem = sem(lfp_al, axis=0, nan_policy='omit')
        opt_mean = np.nanmean(opt_al, axis=0); opt_sem = sem(opt_al, axis=0, nan_policy='omit')
        phase_shifts = _wrap_pi(np.concatenate(all_phase_shifts))
    else:
        lfp_mean = opt_mean = lfp_sem = opt_sem = None
        phase_shifts = np.array([])

    # optional: convert mean phase shift to ms using estimated cycle frequency
    lag_from_phase_ms = np.nan
    mu_phase_rad = np.nan
    R_phase = np.nan
    if phase_shifts.size:
        mu_phase_rad = _circmean(phase_shifts)
        R_phase = _circR(phase_shifts)
        # crude conversion: use centre frequency (safer than noisy instf across segments)
        f_c = 0.5 * (f_low + f_high)
        lag_from_phase_ms = (mu_phase_rad / (2*np.pi)) * (1000.0 / f_c)

    if make_plots:
        # (1) xcorr
        plt.figure(figsize=(6.8, 3.8)); ax = plt.gca()
        ax.plot(lags, xcorr, lw=2.2)
        ax.axvline(0, color='k', ls='--', lw=1.2)
        ax.axvline(lag_peak_s, ls=':', lw=1.8)
        ax.set_xlabel('Lag (s)', fontsize=16); ax.set_ylabel('Correlation', fontsize=16)
        ax.set_title(f'{behaviour}: Cross-correlation (LFP vs GEVI γ {f_low}-{f_high} Hz)', fontsize=16)
        ax.tick_params(labelsize=14); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); plt.show()

        # (2) cycle-averaged waveforms (2 cycles)
        if lfp_mean is not None:
            plt.figure(figsize=(8.2, 4.4)); ax = plt.gca()

            lfp_r, opt_r = _align_cycle_to_lfp_trough_1cycle(lfp_mean, opt_mean)
            lfp_sem_r = np.roll(lfp_sem, -int(np.argmin(lfp_mean)))
            opt_sem_r = np.roll(opt_sem, -int(np.argmin(lfp_mean)))

            n = len(lfp_r)
            x = np.linspace(0, 2*np.pi, n, endpoint=False)

            x2, lfp2 = _two_cycles(x, lfp_r)
            _,  lfp2s = _two_cycles(x, lfp_sem_r)
            _,  opt2 = _two_cycles(x, opt_r)
            _,  opt2s = _two_cycles(x, opt_sem_r)

            ax.plot(x2, lfp2, lw=2.6, label=f"LFP γ")
            ax.fill_between(x2, lfp2 - lfp2s, lfp2 + lfp2s, alpha=0.25)

            ax.plot(x2, opt2, lw=2.6, label="GEVI γ")
            ax.fill_between(x2, opt2 - opt2s, opt2 + opt2s, alpha=0.25)

            ax.axvline(0, color='k', ls='--', lw=1.4)
            ax.axvline(2*np.pi, color='k', ls='--', lw=1.4)

            gevi_tr = int(np.argmin(opt_r))
            ax.axvline(x[gevi_tr], ls='--', lw=1.4)
            ax.axvline(x[gevi_tr] + 2*np.pi, ls='--', lw=1.4)

            ax.set_xlim(0, 4*np.pi)
            ax.set_xlabel('Phase (rad; 0 = LFP trough)', fontsize=16)
            ax.set_ylabel('Amplitude (normalised per cycle)', fontsize=16)
            ax.set_title(f'Cycle-averaged γ waveforms ({f_low}-{f_high} Hz)', fontsize=16)
            ax.tick_params(labelsize=14)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, fontsize=12, ncol=2)
            plt.tight_layout(); plt.show()

        # (3) phase-shift histogram
        if phase_shifts.size:
            plt.figure(figsize=(5.6, 3.9)); ax = plt.gca()
            bins = np.linspace(-np.pi, np.pi, 37)
            ax.hist(_wrap_pi(phase_shifts), bins=bins, density=True, alpha=0.85)
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r'-$\pi$', r'-$\pi/2$', '0', r'$\pi/2$', r'$\pi$'], fontsize=13)
            ax.set_xlabel(r'Cycle-wise phase shift (rad)  $(\varphi_{LFP}-\varphi_{GEVI})$', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.set_title(f'{behaviour}: γ phase-shift distribution', fontsize=15)
            ax.tick_params(labelsize=13); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout(); plt.show()

    return {
        "behaviour": behaviour,
        "band": (f_low, f_high),
        "n_epochs": int(n_epochs_total),
        "gated_fraction": float(n_samples_gated / max(1, n_samples_total)),
        "xcorr_lags_s": lags,
        "xcorr": xcorr,
        "xcorr_peak_lag_s": float(lag_peak_s),
        "xcorr_peak_corr": float(corr_peak),
        "PLV_global": float(plv),
        "Envelope_r_global": float(env_r),
        "env_xcorr_lags_s": lags_e,
        "env_xcorr": xcorr_e,
        "env_xcorr_peak_lag_s": float(env_lag_peak_s),
        "env_xcorr_peak_corr": float(env_corr_peak),
        "phase_shift_rad": phase_shifts,
        "mu_phase_rad": float(mu_phase_rad) if np.isfinite(mu_phase_rad) else np.nan,
        "R_phase": float(R_phase) if np.isfinite(R_phase) else np.nan,
        "lag_from_phase_ms": float(lag_from_phase_ms) if np.isfinite(lag_from_phase_ms) else np.nan,
        "lfp_mean_cycle": lfp_mean,
        "lfp_sem_cycle": lfp_sem,
        "gevi_mean_cycle": opt_mean,
        "gevi_sem_cycle": opt_sem,
    }

# -------------------------
# Single-animal: run 3 states for BOTH slow/fast gamma
# -------------------------
def analyse_three_states_gamma(locomotion_dir, awake_dir, rem_dir,
                              LFP_channel="LFP_1",
                              fs=10000,
                              gamma_bands=None,
                              theta_low_thres=0.0,
                              indicator="GEVI",
                              opt_col="zscore_raw",
                              gate_quantile=0.7,
                              gate_on="lfp",
                              min_epoch_s=0.10,
                              n_resample=200,
                              make_plots=True):
    if gamma_bands is None:
        gamma_bands = {
            "slow_gamma": (30, 55),
            "fast_gamma": (65, 100),
        }

    state_dirs = {
        "Moving": Path(locomotion_dir),
        "Rest":   Path(awake_dir),
        "REM":    Path(rem_dir),
    }

    out = {}
    for band_name, band in gamma_bands.items():
        out[band_name] = {}
        for state, p in state_dirs.items():
            print(f"\n=== {band_name} {band} | {state} | {p} ===")
            out[band_name][state] = analyse_state_gamma(
                p, state, LFP_channel,
                fs=fs,
                gamma_band=band,
                theta_low_thres=theta_low_thres,
                indicator=indicator,
                opt_col=opt_col,
                gate_quantile=gate_quantile,
                gate_on=gate_on,
                min_epoch_s=min_epoch_s,
                n_resample=n_resample,
                max_lag_s=0.05,
                xcorr_win=(-0.02, 0.02),
                make_plots=make_plots
            )
    return out

# -------------------------
# Multi-animal: per-animal per-state per-band results
# -------------------------
def run_gamma_state_per_animal(ANIMALS,
                               fs=10000,
                               gamma_bands=None,
                               theta_low_thres=0.0,
                               indicator="GEVI",
                               opt_col_default="zscore_raw",
                               gate_quantile=0.7,
                               gate_on="lfp",
                               min_epoch_s=0.10,
                               n_resample=200):
    """
    Returns:
      cohort[band_name][state][animal_id] = analyse_state_gamma(...) dict
    """
    if gamma_bands is None:
        gamma_bands = {
            "slow_gamma": (30, 55),
            "fast_gamma": (65, 100),
        }

    cohort = {bn: {"Moving": {}, "Rest": {}, "REM": {}} for bn in gamma_bands.keys()}

    for a in ANIMALS:
        aid = a["animal_id"]
        ch = a["lfp_channel"]
        opt_col = a.get("opt_col", opt_col_default)

        state_dirs = {
            "Moving": Path(a["loco_parent"]),
            "Rest":   Path(a["stat_parent"]),
            "REM":    Path(a["rem_parent"]),
        }

        for band_name, band in gamma_bands.items():
            for state, p in state_dirs.items():
                try:
                    r = analyse_state_gamma(
                        p, state, ch,
                        fs=fs,
                        gamma_band=band,
                        theta_low_thres=theta_low_thres,
                        indicator=indicator,
                        opt_col=opt_col,
                        gate_quantile=gate_quantile,
                        gate_on=gate_on,
                        min_epoch_s=min_epoch_s,
                        n_resample=n_resample,
                        max_lag_s=0.05,
                        xcorr_win=(-0.02, 0.02),
                        make_plots=False
                    )
                    r["animal_id"] = aid
                    cohort[band_name][state][aid] = r
                except Exception as e:
                    print(f"[WARN] {aid} {band_name} {state}: skipped ({e})")

    return cohort

# -------------------------
# Cohort-level 3×3 plotting (matches your theta figure style)
# -------------------------
def plot_cohort_gamma_summary(cohort_band_results, band_label="γ",
                             fs=10000, max_lag_s=0.05, n_resample=200, bins=36,
                             show_individual=False):
    """
    cohort_band_results: cohort[band_name] from run_gamma_state_per_animal
      -> dict with keys "Moving","Rest","REM", each containing animal->result dict

    Produces 3 rows × 3 cols:
      xcorr (gamma), cycle-average (2 cycles), phase-shift density (animal-weighted mean±SEM)
    """
    states = ["Moving", "Rest", "REM"]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 9.2))
    plt.subplots_adjust(wspace=0.30, hspace=0.55)

    lags_common = np.arange(-max_lag_s, max_lag_s + 1/fs, 1/fs)
    phase_bins = np.linspace(-np.pi, np.pi, bins + 1)
    phase_centres = 0.5 * (phase_bins[:-1] + phase_bins[1:])
    x_phase = np.linspace(0, 2*np.pi, n_resample, endpoint=False)

    for i, state in enumerate(states):
        animal_dict = cohort_band_results.get(state, {})
        if len(animal_dict) == 0:
            continue

        # --- (1) xcorr mean±SEM across animals
        ax = axes[i, 0]
        corr_stack = []
        for aid, r in animal_dict.items():
            lags = np.asarray(r["xcorr_lags_s"])
            corr = np.asarray(r["xcorr"])
            c_i = np.interp(lags_common, lags, corr)
            corr_stack.append(c_i)
            if show_individual:
                ax.plot(lags_common, c_i, lw=1.0, alpha=0.25)
        corr_stack = np.vstack(corr_stack)
        m = np.nanmean(corr_stack, axis=0)
        s = sem(corr_stack, axis=0, nan_policy='omit')
        ax.plot(lags_common, m, lw=2.2)
        ax.fill_between(lags_common, m - s, m + s, alpha=0.25)
        ax.axvline(0, color='k', ls='--', lw=1.1)
        ax.set_title(f"{state}: Cross-corr (LFP vs GEVI {band_label})", fontsize=15)
        ax.set_xlabel("Lag (s)", fontsize=14)
        ax.set_ylabel("Correlation", fontsize=14)
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # --- (2) cycle-average mean±SEM across animals (2 cycles)
        ax = axes[i, 1]
        lfp_cycles = []
        gevi_cycles = []
        for aid, r in animal_dict.items():
            lfp = r.get("lfp_mean_cycle", None)
            gevi = r.get("gevi_mean_cycle", None)
            if lfp is None or gevi is None:
                continue
            lfp_a, gevi_a = _align_cycle_to_lfp_trough_1cycle(np.asarray(lfp), np.asarray(gevi))
            lfp_cycles.append(lfp_a); gevi_cycles.append(gevi_a)

        if len(lfp_cycles):
            lfp_cycles = np.vstack(lfp_cycles)
            gevi_cycles = np.vstack(gevi_cycles)
            lfp_m = np.nanmean(lfp_cycles, axis=0); lfp_s = sem(lfp_cycles, axis=0, nan_policy='omit')
            gevi_m = np.nanmean(gevi_cycles, axis=0); gevi_s = sem(gevi_cycles, axis=0, nan_policy='omit')

            x2, lfp_m2 = _two_cycles(x_phase, lfp_m)
            _,  lfp_s2 = _two_cycles(x_phase, lfp_s)
            _,  gevi_m2 = _two_cycles(x_phase, gevi_m)
            _,  gevi_s2 = _two_cycles(x_phase, gevi_s)

            ax.plot(x2, lfp_m2, lw=2.6, label="LFP")
            ax.fill_between(x2, lfp_m2 - lfp_s2, lfp_m2 + lfp_s2, alpha=0.25)
            ax.plot(x2, gevi_m2, lw=2.6, label="GEVI")
            ax.fill_between(x2, gevi_m2 - gevi_s2, gevi_m2 + gevi_s2, alpha=0.25)

            ax.axvline(0, color='k', ls='--', lw=1.4)
            ax.axvline(2*np.pi, color='k', ls='--', lw=1.4)

            gevi_tr = int(np.argmin(gevi_m))
            ax.axvline(x_phase[gevi_tr], ls='--', lw=1.4)
            ax.axvline(x_phase[gevi_tr] + 2*np.pi, ls='--', lw=1.4)

            ax.set_xlim(0, 4*np.pi)
            ax.set_title(f"Cycle-avg (0 = LFP trough)", fontsize=15)
            ax.set_xlabel("Phase (rad)", fontsize=14)
            ax.set_ylabel("Amplitude (z per cycle)", fontsize=14)
            ax.tick_params(labelsize=13)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if i == 0:
                ax.legend(frameon=False, ncol=2, fontsize=12, loc="lower center")

        # --- (3) phase-shift density mean±SEM across animals (animal-weighted)
        ax = axes[i, 2]
        dens_stack = []
        for aid, r in animal_dict.items():
            ph = r.get("phase_shift_rad", None)
            if ph is None or len(ph) == 0:
                continue
            dens, _ = np.histogram(_wrap_pi(ph), bins=phase_bins, density=True)
            dens_stack.append(dens)
            if show_individual:
                ax.plot(phase_centres, dens, lw=1.0, alpha=0.20)

        if len(dens_stack):
            dens_stack = np.vstack(dens_stack)
            dm = np.nanmean(dens_stack, axis=0)
            ds = sem(dens_stack, axis=0, nan_policy='omit')
            ax.plot(phase_centres, dm, lw=2.2)
            ax.fill_between(phase_centres, dm - ds, dm + ds, alpha=0.25)

        ax.set_title(f"{state}: Phase-shift dist", fontsize=15)
        ax.set_xlabel(r"$(\varphi_{LFP}-\varphi_{GEVI})$ (rad)", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'-$\pi$', r'-$\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle(f"Cohort summary: {band_label} (gated high-γ epochs)", fontsize=17, y=0.98)
    plt.tight_layout()
    plt.show()
    return fig

#%%
'Single animal analysis'
locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\ALocomotion")
awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\AwakeStationary")
rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\AsleepNonREM")

gamma_out = analyse_three_states_gamma(
    locomotion_dir, awake_dir, rem_dir,
    LFP_channel="LFP_1",
    fs=10000,
    gamma_bands={"slow_gamma": (30,55), "fast_gamma": (65,100)},
    opt_col="zscore_raw",
    gate_quantile=0.4,     # if too strict, try 0.6
    min_epoch_s=0.05,      # if too strict, try 0.05
    make_plots=True
)

# Access numbers
for band_name, states in gamma_out.items():
    for s, r in states.items():
        print(band_name, s,
              "xcorr lag(ms)=", 1000*r["xcorr_peak_lag_s"],
              "PLV=", r["PLV_global"],
              "env_r=", r["Envelope_r_global"],
              "env_lag(ms)=", 1000*r["env_xcorr_peak_lag_s"],
              "gated_frac=", r["gated_fraction"])


#statistics
#%%
'Multiple animal analysis'
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

cohort = run_gamma_state_per_animal(
    ANIMALS,
    fs=10000,
    gamma_bands={"slow_gamma": (30,55), "fast_gamma": (65,100)},
    opt_col_default="zscore_raw",
    gate_quantile=0.4,
    gate_on="lfp",
    min_epoch_s=0.05,
    n_resample=200
)

plot_cohort_gamma_summary(cohort["slow_gamma"], band_label="Slow γ (30–55 Hz)")
plot_cohort_gamma_summary(cohort["fast_gamma"], band_label="Fast γ (65–100 Hz)")
