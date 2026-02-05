# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 22:26:47 2025

@author: yifan
Independent theta coherence heatmap among optical signals
(Behaviour-state filtering version)
"""
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, sosfiltfilt, hilbert, coherence, correlate

# ---------- helpers ----------
def _sos_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def _theta_envelope(x, fs, theta_band=(4, 12)):
    th = _sos_bandpass(x, fs, theta_band[0], theta_band[1])
    return np.abs(hilbert(th))

def _summarise_theta_coherence(x, y, fs, theta_band=(4, 12), nperseg=None, noverlap=None):
    if nperseg is None:
        nperseg = max(256, int(round(fs * 2)))  # ~2 s window
    if noverlap is None:
        noverlap = nperseg // 2
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    sel = (f >= theta_band[0]) & (f <= theta_band[1])
    return float(np.nanmean(Cxy[sel])) if np.any(sel) else np.nan

def _build_state_mask(df, movement_col='movement', states=None):
    """Return boolean mask selecting rows whose movement_col is in states.
    If states is None: all rows. Accepts a single string or an iterable.
    """
    if states is None:
        return np.ones(len(df), dtype=bool)
    if isinstance(states, str):
        states = [states]
    col = df.get(movement_col, None)
    if col is None:
        raise ValueError(f"Column '{movement_col}' not found in df.")
    return col.isin(states).to_numpy(dtype=bool)

# ---------- main plotter (state-filtered) ----------
def plot_coherence_heatmap_from_df(
        df: pd.DataFrame,
        fs: float,
        chan_map: dict,
        theta_band=(4, 12),
        use_envelope=False,                 # envelope coherence (True) vs bandpassed signals (False)
        order=("CA1_R", "CA1_L", "CA3_PR"),
        vmin=0.0, vmax=1.0,
        palette="greyred",
        annot_fs=12, tick_fs=14, label_fs=16, title_fs=18,
        savepath=None, prefix="ex_events",
        nperseg=None, noverlap=None,
        # NEW:
        movement_col='movement',
        states=None                         # e.g. 'moving' or ('moving','running')
    ):
    """
    Pairwise theta-band coherence heatmap among optical signals, restricted to rows
    whose behaviour state matches `states` in column `movement_col`.
    """

    # collect available optical channels
    present = [(col, name) for col, name in chan_map.items() if col in df.columns]
    if len(present) < 2:
        raise ValueError("Need at least two optical channels present in df.")

    # behaviour-state mask
    mask_state = _build_state_mask(df, movement_col=movement_col, states=states)

    # compute signals for coherence (after filtering)
    sigs = {}
    for col, name in present:
        x = df[col].to_numpy(dtype=float)
        x = x[mask_state & np.isfinite(x)]
        if use_envelope:
            sigs[name] = _theta_envelope(x, fs, theta_band)
        else:
            sigs[name] = _sos_bandpass(x, fs, theta_band[0], theta_band[1])

    # resolve names & order (CA3_PR fallback)
    present_names = set([n for _, n in present])
    desired = list(order)
    if "CA3_PR" in desired and "CA3_PR" not in present_names and "CA3_L" in present_names:
        desired = [("CA3_L" if x == "CA3_PR" else x) for x in desired]
    names = [n for n in desired if n in present_names]
    extras = sorted(list(present_names.difference(names)))
    names += extras
    n = len(names)

    # build coherence matrix
    M = np.full((n, n), np.nan, float)
    np.fill_diagonal(M, 1.0)
    for i in range(n):
        for j in range(i+1, n):
            a, b = names[i], names[j]
            xi, xj = sigs[a], sigs[b]
            if min(len(xi), len(xj)) < 256:
                val = np.nan
            else:
                L = min(len(xi), len(xj))
                val = _summarise_theta_coherence(xi[:L], xj[:L], fs, theta_band, nperseg, noverlap)
            M[i, j] = M[j, i] = val

    # choose palette
    if palette.lower() in ("grey", "gray", "greys", "grays"):
        cmap_use = plt.get_cmap("Greys")
    else:
        cmap_use = LinearSegmentedColormap.from_list("greyred",
                                                     ["#f7f7f7", "#cfcfcf", "#ef8a62", "#b2182b"])

    # nice title scope
    if states is None:
        scope = "all states"
    else:
        scope = f"state: {states}" if isinstance(states, str) else f"states: {tuple(states)}"
    mode = "envelope" if use_envelope else "bandpassed"

    # plot
    fig, ax = plt.subplots(figsize=(1.3*n + 2, 1.3*n + 2), constrained_layout=True)
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap_use)
    
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(names, fontsize=tick_fs)

    mid = 0.5*(vmin+vmax)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.2f}",
                    ha="center", va="center",
                    color=("white" if (np.isfinite(v) and v >= mid) else "black"),
                    fontsize=annot_fs)

    ax.set_title(f"Theta coherence (4–12 Hz) | {scope}", fontsize=title_fs, pad=20)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("coherence", rotation=90, fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    if savepath:
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(os.path.join(savepath, f"{prefix}_coherence_heatmap.png"),
                    dpi=300, bbox_inches="tight")

    return fig, ax

def _zscore(x):
    mu = np.nanmean(x); sd = np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x*0.0

def plot_env_xcorr_optical(
        df_theta: pd.DataFrame,
        fs: float,
        chan_map = {'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
        theta_band=(4,12),
        max_lag_ms=250,
        # NEW: behaviour-state filtering
        movement_col='movement',
        states=None,  # e.g. 'moving' or ('moving','running'); None => all
        # styling
        bar_title_fs=18, bar_tick_fs=14, bar_label_fs=16,
        savepath=None, prefix='ex_events'
    ):
    """
    Theta envelope cross-correlation between every pair of optical signals,
    restricted to rows whose behaviour state matches `states` in `movement_col`.
    Positive lag_ms => second name lags the first (first leads).
    """

    # behaviour-state mask
    mask_state = _build_state_mask(df_theta, movement_col=movement_col, states=states)

    # Collect present channels
    present = [(col, name) for col, name in chan_map.items() if col in df_theta.columns]
    names   = [name for _, name in present]
    if len(present) < 2:
        raise ValueError("Need at least two optical channels present in df_theta.")

    # Compute theta envelopes, then apply mask
    env = {}
    for col, name in present:
        e = _theta_envelope(df_theta[col].to_numpy(float), fs, theta_band)
        e = e[mask_state & np.isfinite(e)]
        env[name] = e

    # Build ordered pairs
    desired_order = ['CA1_R','CA1_L','CA3_L']
    ordered = [n for n in desired_order if n in names]
    # append any extras
    extras = [n for n in names if n not in ordered]
    ordered += extras
    pairs = [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i+1, len(ordered))]

    # Compute normalized xcorr peak within ±max_lag
    max_lag = int((max_lag_ms/1000.0) * fs)
    labels, rvals, lags = [], [], []
    for a_name, b_name in pairs:
        a_full = env[a_name]; b_full = env[b_name]
        n = min(a_full.size, b_full.size)
        labels.append(f"{a_name}–{b_name}")
        if n < 32:
            rvals.append(np.nan); lags.append(np.nan); continue
        a = _zscore(a_full[:n]); b = _zscore(b_full[:n])
        xcorr = correlate(a - np.mean(a), b - np.mean(b), mode='full')
        lags_samp = np.arange(-n+1, n)
        sel = (lags_samp >= -max_lag) & (lags_samp <= max_lag)
        if not np.any(sel):
            rvals.append(np.nan); lags.append(np.nan); continue
        xs  = xcorr[sel]; ls = lags_samp[sel]
        k   = int(np.nanargmax(xs))
        denom = (np.std(a)*np.std(b)*n)
        peak_r = float(xs[k] / denom) if denom > 0 else np.nan
        lag_ms = 1000.0 * ls[k] / fs   # + => b lags a
        rvals.append(peak_r); lags.append(float(lag_ms))

    # figures
    scope = "all states" if states is None else (f"state: {states}" if isinstance(states, str) else f"states: {tuple(states)}")

    # -------- figure 1: peak r --------
    fig_r, ax_r = plt.subplots(figsize=(max(5, 1*len(labels)), 5), constrained_layout=True)
    ax_r.bar(np.arange(len(labels)), rvals)
    ax_r.set_ylabel('peak r', fontsize=bar_label_fs)
    ax_r.set_ylim(0, 1)
    scope = "all states" if states is None else (f"{states}" if isinstance(states, str) else f"{tuple(states)}")
    ax_r.set_title(f'Theta envelope cross-correlation | {scope}', fontsize=bar_title_fs, pad=20)
    ax_r.set_xticks(np.arange(len(labels)))
    ax_r.set_xticklabels(labels, rotation=15, fontsize=bar_tick_fs)
    ax_r.tick_params(axis='y', labelsize=bar_tick_fs)
    # fig_r.tight_layout()  # not needed with constrained_layout
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        fig_r.savefig(os.path.join(savepath, f"{prefix}_env_xcorr_peak.png"),
                      dpi=300, bbox_inches='tight')

    # -------- figure 2: lag (ms) --------
    fig_l, ax_l = plt.subplots(figsize=(max(7, 0.6*len(labels)), 4))
    ax_l.bar(np.arange(len(labels)), lags)
    ax_l.axhline(0, color='k', lw=0.8)
    ax_l.set_ylabel('lag (ms)', fontsize=bar_label_fs)
    ax_l.set_xticks(np.arange(len(labels)))
    ax_l.set_xticklabels(labels, rotation=15, fontsize=bar_tick_fs)
    ax_l.tick_params(axis='y', labelsize=bar_tick_fs)
    finite_lags = [x for x in lags if np.isfinite(x)]
    if finite_lags:
        lim = max(10.0, np.ceil(max(abs(min(finite_lags)), abs(max(finite_lags))) / 5.0) * 5.0)
        ax_l.set_ylim(-lim, lim)
    fig_l.tight_layout()
    if savepath:
        fig_l.savefig(os.path.join(savepath, f"{prefix}_env_xcorr_lag.png"),
                      dpi=300, bbox_inches='tight')

    data = {'labels': labels, 'peak_r': np.array(rvals, float), 'lag_ms': np.array(lags, float),
            'movement_col': movement_col, 'states': states}
    return (fig_r, ax_r), (fig_l, ax_l), data

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, medfilt, savgol_filter
from scipy.ndimage import gaussian_filter

def _interp_speed_over_threshold(spd: np.ndarray, thr: float = 100.0) -> np.ndarray:
    y = np.asarray(spd, dtype=float).copy()
    bad = y > float(thr)
    y[bad] = np.nan
    idx = np.arange(y.size)
    good = np.isfinite(y)
    if not np.any(good):
        return np.zeros_like(y)
    # edge fill
    first = np.argmax(good); last = y.size - 1 - np.argmax(good[::-1])
    if not good[0]:  y[:first] = y[first]; good[:first] = True
    if not good[-1]: y[last+1:] = y[last]; good[last+1:] = True
    # linear interp
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y

def _smooth_speed(spd: np.ndarray, fs: float, med_kernel_s=0.15, sg_win_s=0.35, sg_poly=3):
    # median kernel size must be odd and >=1
    k_med = max(1, int(round(med_kernel_s * fs)) | 1)
    s_med = medfilt(spd, kernel_size=k_med)
    # Savitzky–Golay window must be odd and > poly
    k_sg = max(5, int(round(sg_win_s * fs)) | 1)
    if k_sg <= sg_poly: k_sg = sg_poly + 2 + (sg_poly % 2 == 0)
    s_sg = savgol_filter(s_med, window_length=k_sg, polyorder=sg_poly, mode='mirror')
    return s_sg

def plot_timefreq_coherence_with_speed_pretty(
        df, fs,
        col_a, col_b, pair_label=('A','B'),
        speed_col='speed',
        fmax=20.0,
        # outer time windowing
        win_sec=3.0, step_sec=0.15,
        # inner Welch segmentation
        welch_sec=0.5, welch_overlap=0.5,
        # theta band for optional time-course
        theta_band=(4,12),
        # visuals
        vmin=0.0, vmax=1.0, cmap='viridis',
        blur_sigma=0.6,                 # Gaussian blur (visual only); set 0 to disable
        # behaviour/state filter
        movement_col=None, states=None,
        # speed cleaning
        speed_max=100.0,
        speed_med_kernel_s=0.15,
        speed_sg_win_s=0.35,
        savepath=None, prefix='pair_tfr'
    ):
    # --------- masks and data
    mask = _build_state_mask(df, movement_col, states) if (movement_col and states is not None) else np.ones(len(df), bool)
    xa = df[col_a].to_numpy(float)[mask]
    xb = df[col_b].to_numpy(float)[mask]

    if 'time' in df.columns:
        t_full = df['time'].to_numpy(float)[mask]
    elif 'timestamp' in df.columns:
        t_full = df['timestamp'].to_numpy(float)[mask]
    else:
        t_full = np.arange(mask.sum()) / fs

    if speed_col in df.columns:
        spd = df[speed_col].to_numpy(float)[mask]
    else:
        spd = np.zeros_like(t_full)

    finite = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(t_full) & np.isfinite(spd)
    xa, xb, t, spd = xa[finite], xb[finite], t_full[finite], spd[finite]

    # clean speed
    if speed_max is not None:
        spd = _interp_speed_over_threshold(spd, thr=float(speed_max))
    spd = _smooth_speed(spd, fs, med_kernel_s=speed_med_kernel_s, sg_win_s=speed_sg_win_s)

    # --------- window params
    win  = int(round(win_sec  * fs))
    hop  = int(round(step_sec * fs));  hop = max(1, hop)
    nper = int(round(welch_sec * fs)); nper = max(16, nper)
    novl = int(round(nper * np.clip(welch_overlap, 0.0, 0.95)))
    if nper > win:
        nper = max(16, win // 3); novl = nper // 2

    # frequency grid
    f_ref, _ = coherence(xa[:win], xb[:win], fs=fs, nperseg=nper, noverlap=novl, window='hann')
    sel = f_ref <= float(fmax); f = f_ref[sel]
    if f.size == 0: raise ValueError("fmax too low.")

    starts = np.arange(0, xa.size - win + 1, hop, dtype=int)
    T = np.empty(starts.size, float)
    C = np.full((f.size, starts.size), np.nan, float)

    theta_sel = (f >= theta_band[0]) & (f <= theta_band[1])
    theta_t = np.empty(starts.size, float); theta_t[:] = np.nan
    theta_val = np.empty(starts.size, float); theta_val[:] = np.nan

    for k, s in enumerate(starts):
        e = s + win
        xi = xa[s:e]; xj = xb[s:e]
        T[k] = 0.5 * (t[s] + t[e-1])
        ff, Cxy = coherence(xi, xj, fs=fs, nperseg=nper, noverlap=novl, window='hann')
        C[:, k] = Cxy[sel]
        theta_t[k] = T[k]
        theta_val[k] = np.nanmean(C[theta_sel, k]) if np.any(theta_sel) else np.nan

    # visual smoothing (heatmap only)
    C_plot = gaussian_filter(C, sigma=blur_sigma) if (blur_sigma and blur_sigma > 0) else C

    # --------- plot
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, constrained_layout=True)

    im = ax_top.pcolormesh(T, f, C_plot, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    pair = f"{pair_label[0]}–{pair_label[1]}"
    scope = "all states" if (movement_col is None or states is None) else (f"state: {states}" if isinstance(states, str) else f"states: {tuple(states)}")
    ax_top.set_ylabel("Frequency (Hz)")
    ax_top.set_title(f"Coherence over time (0–{int(fmax)} Hz) | {pair} | {scope}")
    ax_top.set_ylim(0, fmax)
    cbar = fig.colorbar(im, ax=ax_top, pad=0.02); cbar.set_label("Coherence")

    ax_bot.plot(t, spd, lw=1.0,color='k')
    # Optional: overlay theta time-course on a twin y-axis
    ax_theta = ax_bot.twinx()
    ax_theta.plot(theta_t, theta_val, lw=1.2, alpha=0.8)
    ax_theta.set_ylabel("θ-coherence", rotation=90)
    ax_theta.set_ylim(0, 1)

    ax_bot.set_ylabel("Speed")
    ax_bot.set_xlabel("Time (s)")
    subtitle = f"Speed over time (>{int(speed_max)} interpolated)" if speed_max is not None else "Speed over time"
    ax_bot.set_title(subtitle)

    if savepath:
        import os
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(os.path.join(savepath, f"{prefix}_TFR_{pair.replace(' ','')}.png"), dpi=300, bbox_inches='tight')

    return fig, (ax_top, ax_bot), {"T":T, "f":f, "C":C, "theta_t":theta_t, "theta":theta_val}

import numpy as np

def plot_coherence_spectrum(
        df, fs,
        col_a, col_b,
        pair_label=('A','B'),
        fmax=20.0,
        nperseg_sec=2.0,
        overlap=0.5,
        window='hann',
        detrend='constant',
        movement_col=None, states=None,
        smooth_hz=0.0,
        theta_band=(4,12),
        savepath=None, prefix='coh_spectrum',
        # --- NEW: typography / style ---
        label_fs=18,          # x/y-axis label size
        tick_fs=16,           # tick label size
        title_fs=20,          # title size
        legend_fs=14,         # legend size
        line_w=2.0,           # smoothed line width
        raw_line_w=1.4        # raw line width
    ):
    """Plot coherence spectrum Cxy(f) between col_a and col_b with larger fonts."""
    # Behaviour-state mask
    mask = _build_state_mask(df, movement_col, states) if (movement_col and states is not None) \
           else np.ones(len(df), bool)

    # Extract & clean
    x = df[col_a].to_numpy(float)[mask]
    y = df[col_b].to_numpy(float)[mask]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if min(len(x), len(y)) < 64:
        raise ValueError("Not enough valid samples for coherence.")

    # Welch params
    nperseg = max(32, int(round(nperseg_sec * fs)))
    N = min(len(x), len(y))
    if nperseg > N:
        nperseg = max(32, N // 4)
    noverlap = int(round(np.clip(overlap, 0, 0.95) * nperseg))

    # Coherence
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap,
                       window=window, detrend=detrend)
    sel = f <= float(fmax)
    f, Cxy = f[sel], Cxy[sel]

    # Optional smoothing
    Cxy_smooth = None
    if smooth_hz and smooth_hz > 0:
        dfreq = np.median(np.diff(f))
        k = max(5, int(round(smooth_hz / dfreq)) | 1)   # odd length ≥5
        poly = 2 if k > 3 else 1
        Cxy_smooth = savgol_filter(Cxy, window_length=k, polyorder=poly, mode='interp')

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.plot(f, Cxy, lw=raw_line_w, alpha=0.6, label='raw')
    if Cxy_smooth is not None:
        ax.plot(f, Cxy_smooth, lw=line_w, label=f'smoothed (~{smooth_hz:g} Hz)')

    ax.set_xlim(0, fmax); ax.set_ylim(0, 1.0)
    ax.set_xlabel('Frequency (Hz)', fontsize=label_fs)
    ax.set_ylabel('Coherence', fontsize=label_fs)

    pair = f"{pair_label[0]}–{pair_label[1]}"
    scope = "all states" if (movement_col is None or states is None) else (f"{states}" if isinstance(states, str) else f"{tuple(states)}")
    ax.set_title(f'{pair} | {scope}', fontsize=title_fs, pad=8)

    # Theta shading
    th_lo, th_hi = theta_band
    ax.axvspan(th_lo, th_hi, color='grey', alpha=0.15, label='theta band')

    # Ticks, grid, legend
    ax.tick_params(axis='both', which='both', labelsize=tick_fs)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=legend_fs)

    if savepath:
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(os.path.join(savepath, f"{prefix}_{pair.replace(' ','')}_coh_spectrum.png"),
                    dpi=300, bbox_inches='tight')
    return fig, ax, (f, Cxy if Cxy_smooth is None else Cxy_smooth)

#%%
# dpath=r'G:\2025_ATLAS_SPAD\MultiFibre\1887932_Jedi2p_Multi_ephysbad\MovingTrialsDLC'
# recordingName='SyncRecording2'

dpath=r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day1and2DLC'
recordingName='SyncRecording2'

Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 
# Grab once from your session
df = Recording1.Ephys_tracking_spad_aligned
fs = Recording1.fs

savename='ThetaSave'
save_path = os.path.join(dpath,savename)
save_dir  = os.path.join(dpath, "theta_figs_events")
os.makedirs(save_dir, exist_ok=True)
# Choose the raw columns that correspond to CA1_L and CA3_L in your df
col_CA1_L = 'sig_raw'     # maps to CA1_L in your chan_map
col_CA3_L = 'zscore_raw'  # maps to CA3_L in your chan_map
col_CA1_R = 'ref_raw'  # maps to CA3_L in your chan_map

# fig_pair, axes_pair, out = plot_timefreq_coherence_with_speed_pretty(
#     df=Recording1.Ephys_tracking_spad_aligned, fs=Recording1.fs,
#     col_a='sig_raw', col_b='ref_raw', pair_label=('CA1_L','CA1_R'),
#     fmax=20, win_sec=3.0, step_sec=0.15, welch_sec=0.5, welch_overlap=0.5,
#     movement_col='movement', states=None,
#     speed_max=60.0,           # interpolate spikes >100
#     speed_med_kernel_s=0.15,   # ~150 ms median
#     speed_sg_win_s=0.35,       # ~350 ms Savitzky–Golay
#     blur_sigma=0.6,            # light visual smoothing of the heatmap
#     savepath=save_dir, prefix=recordingName
# )

# CA1_L vs CA1_R
plot_coherence_spectrum(
    df=Recording1.Ephys_tracking_spad_aligned, fs=Recording1.fs,
    col_a='sig_raw', col_b='ref_raw',
    pair_label=('CA1_L','CA1_R'),
    fmax=20, nperseg_sec=2.0, overlap=0.5,
    movement_col='movement', states='moving',
    smooth_hz=0.5,
    savepath=save_dir, prefix=recordingName,
    label_fs=20, tick_fs=18, title_fs=22, legend_fs=16, line_w=2.4
)
# CA1_L vs CA1_R
plot_coherence_spectrum(
    df=Recording1.Ephys_tracking_spad_aligned, fs=Recording1.fs,
    col_a='ref_raw', col_b='zscore_raw',
    pair_label=('CA1_R','CA3_L'),
    fmax=20, nperseg_sec=2.0, overlap=0.5,
    movement_col='movement', states='moving',
    smooth_hz=0.5,
    savepath=save_dir, prefix=recordingName,
    label_fs=20, tick_fs=18, title_fs=22, legend_fs=16, line_w=2.4
)

# CA1_L vs CA3_L
plot_coherence_spectrum(
    df=Recording1.Ephys_tracking_spad_aligned, fs=Recording1.fs,
    col_a='sig_raw', col_b='zscore_raw',
    pair_label=('CA1_L','CA3_L'),
    fmax=20, nperseg_sec=2.0, overlap=0.5,
    movement_col='movement', states='moving',
    smooth_hz=0.5,
    savepath=save_dir, prefix=recordingName,
    label_fs=20, tick_fs=18, title_fs=22, legend_fs=16, line_w=2.4
)
#%%
# Reuse your existing plotters for coherence/xcorr using theta_res_ev:
'This is to plot theta coherence among optical signals'
# Coherence among optical signals, theta-rich only by CA1_R:
# moving-only:
figC_m, axC_m = plot_coherence_heatmap_from_df(
    df=Recording1.Ephys_tracking_spad_aligned,
    fs=Recording1.fs,
    chan_map={'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
    theta_band=(4,12),
    use_envelope=False,
    order=("CA1_R","CA1_L","CA3_L"),
    palette="greyred",
    savepath=save_dir, prefix="ex_events_moving",
    movement_col='movement',
    states='moving'       # <-- behaviour-state filter
)

# not-moving (example; adjust to your label):
figC_nm, axC_nm = plot_coherence_heatmap_from_df(
    df=Recording1.Ephys_tracking_spad_aligned,
    fs=Recording1.fs,
    chan_map={'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
    theta_band=(4,12),
    use_envelope=False,
    order=("CA1_R","CA1_L","CA3_L"),
    palette="greyred",
    savepath=save_dir, prefix="ex_events_notmoving",
    movement_col='movement',
    states='notmoving'   # <-- whatever your label is
)

(fig_peak_m, ax_peak_m), (fig_lag_m, ax_lag_m), data_bars_m = plot_env_xcorr_optical(
    df_theta=Recording1.Ephys_tracking_spad_aligned,
    fs=Recording1.fs,
    chan_map={'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
    theta_band=(5,11),
    max_lag_ms=250,
    savepath=save_dir,
    prefix="ex_events_moving",
    movement_col='movement',
    states='moving'       # <-- behaviour-state filter
)

(fig_peak_m, ax_peak_m), (fig_lag_m, ax_lag_m), data_bars_m = plot_env_xcorr_optical(
    df_theta=Recording1.Ephys_tracking_spad_aligned,
    fs=Recording1.fs,
    chan_map={'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
    theta_band=(5,11),
    max_lag_ms=250,
    savepath=save_dir,
    prefix="ex_events_moving",
    movement_col='movement',
    states='notmoving'       # <-- behaviour-state filter
)

