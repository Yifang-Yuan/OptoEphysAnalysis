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

#%%
dpath= r'G:\2025_ATLAS_SPAD\MultiFibre\1887932_Jedi2p_Multi_ephysbad\MovingTrialsDLC'
recordingName='SyncRecording6'

Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 

# Grab once from your session
df = Recording1.Ephys_tracking_spad_aligned
fs = Recording1.fs

savename='ThetaSave'
save_path = os.path.join(dpath,savename)
save_dir  = os.path.join(dpath, "theta_figs_events")
os.makedirs(save_dir, exist_ok=True)


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

