# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 12:37:32 2025

@author: yifan
"""
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import MakePlots
from scipy.stats import circmean
from scipy.signal import butter, sosfiltfilt, hilbert, coherence, correlate, find_peaks
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
import seaborn as sns
from matplotlib.lines import Line2D
# ---------- core helpers ----------
def _sos_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band', output='sos')
    return sosfiltfilt(sos, np.asarray(x, float))

def _theta_phase_trough0(lfp, fs, theta_band=(4,12)):
    """Return theta phase in [0, 2π) with 0 at LFP troughs + trough indices used."""
    th = _sos_bandpass(lfp, fs, *theta_band)
    ph = np.angle(hilbert(th))              # (-π, π], 0 at LFP peak
    # find troughs on bandpassed LFP
    min_dist = int(max(1, 0.8 * fs / float(theta_band[1])))
    trough_idx, _ = find_peaks(-th, distance=min_dist)
    if trough_idx.size >= 3:
        phi0 = np.angle(np.nanmean(np.exp(1j*ph[trough_idx])))
    else:
        phi0 = np.pi  # fallback: for a cosine, trough ≈ π
    phi = (ph - phi0) % (2*np.pi)          # 0..2π, 0 at trough
    return phi, trough_idx

def _rayleigh_test(ph):
    """Rayleigh uniformity test for angles ph in radians."""
    ph = np.asarray(ph, float)
    ph = ph[np.isfinite(ph)]
    n = ph.size
    if n == 0:
        return np.nan, np.nan
    C = np.sum(np.cos(ph)); S = np.sum(np.sin(ph))
    R = np.sqrt(C*C + S*S) / n
    Z = n * R * R
    # Berens (2009) approximation (CircStat)
    p = np.exp(-Z) * (1 + (2*Z - Z*Z)/(4*n) - (24*Z - 132*Z*Z + 76*Z**3 - 9*Z**4)/(288*n**2))
    p = float(np.clip(p, 0.0, 1.0))
    return float(Z), p

# ---------- EVENT-BASED preferred phase ----------
def preferred_phase_from_events(df_theta, fs, lfp_col, opt_col,
                                theta_band=(4,12),
                                height_factor=3.0,
                                distance_samples=20,
                                prominence=None,
                                use_event_indices=None,
                                plot=True):
    """
    Matches your event-based definition: preferred phase of optical EVENTS
    relative to the LFP, with 0° = LFP trough.
    - distance_samples: minimal spacing between optical events (in samples).
    """
    # pull arrays
    lfp = pd.to_numeric(df_theta[lfp_col], errors='coerce').to_numpy(float)
    opt = pd.to_numeric(df_theta[opt_col], errors='coerce').to_numpy(float)

    # LFP phase with trough at 0°
    theta_phase, trough_idx = _theta_phase_trough0(lfp, fs, theta_band=theta_band)  # 0..2π

    # events
    if use_event_indices is None:
        baseline = np.nanmedian(opt)
        mad = median_abs_deviation(opt, scale=1.0, nan_policy='omit')
        thr = baseline + height_factor * mad
        event_idx, _ = find_peaks(opt, height=thr,
                                  distance=int(distance_samples),
                                  prominence=prominence)
    else:
        event_idx = np.asarray(use_event_indices, int)

    if event_idx.size == 0:
        raise ValueError(f"No events detected for {opt_col}.")

    # event phases (0..2π)
    event_phases = theta_phase[event_idx]

    # mean direction (preferred) & depth
    C = np.mean(np.cos(event_phases)); S = np.mean(np.sin(event_phases))
    pref = (np.arctan2(S, C)) % (2*np.pi)
    R = float(np.sqrt(C*C + S*S))
    Z, p = _rayleigh_test(event_phases)

    # optional plot in your exact convention
    fig = None
    if plot:
        bins = 30
        bin_edges = np.linspace(0, 2*np.pi, bins+1)
        counts, _ = np.histogram(event_phases, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        r = counts/counts.max() if counts.max()>0 else counts
        theta_circ = np.r_[bin_centers, bin_centers[0]]
        r_circ     = np.r_[r, r[0]]

        fig = plt.figure(figsize=(5.2, 5.2))
        ax  = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('E')      # 0° at right
        ax.set_theta_direction(1)            # anticlockwise
        ax.spines['polar'].set_linewidth(2.5)
        ax.grid(True, linewidth=0.8, alpha=0.4)
        ax.set_thetagrids([0,90,180,270], labels=['0','90','180','270'], fontsize=14, weight='bold')
        ax.set_ylim(0, 1.0); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(['0.5','1'])
        ax.plot(theta_circ, r_circ, linewidth=2.5)
        ax.annotate("", xy=(pref, 0.9*R), xytext=(0,0), arrowprops=dict(arrowstyle="->", linewidth=4))
        ax.set_title(f"pref={np.degrees(pref):.1f}°, R={R:.3f}, p={p:.3g} (n={event_idx.size})",
                     va="bottom", pad=12, fontsize=16)
        plt.tight_layout()

    return {
        'preferred_phase_deg': float(np.degrees(pref) % 360.0),
        'modulation_depth_R' : float(R),
        'Z'                  : float(Z),
        'p'                  : float(p),
        'n_events'           : int(event_idx.size),
        'event_indices'      : event_idx,
        'trough_indices'     : trough_idx,
        'fig'                : fig
    }

# ---------- batch wrapper for multiple optical channels ----------
def preferred_phase_events_multi(df_theta, fs, lfp_col, chan_map,
                                 theta_band=(4,12),
                                 height_factor=3.0,
                                 distance_samples=20,
                                 prominence=None,
                                 events_dict=None,   # optional: dict {opt_col: indices}
                                 plot_each=False):
    out = {}
    for opt_col, region_name in chan_map.items():
        idxs = None if events_dict is None else events_dict.get(opt_col, None)
        res = preferred_phase_from_events(df_theta, fs, lfp_col, opt_col,
                                          theta_band=theta_band,
                                          height_factor=height_factor,
                                          distance_samples=distance_samples,
                                          prominence=prominence,
                                          use_event_indices=idxs,
                                          plot=plot_each)
        out[region_name] = res
    return out

# ---------- helpers (unchanged) ----------

def _sos_highpass(x, fs, cutoff=0.5, order=4):
    sos = butter(order, cutoff/(fs/2), btype='highpass', output='sos')
    return sosfiltfilt(sos, np.asarray(x, float))

def _time_seconds(df, fs):
    if 'timestamps' in df.columns:
        return pd.to_numeric(df['timestamps'], errors='coerce').to_numpy(float)
    if 'time' in df.columns:
        return pd.to_numeric(df['time'], errors='coerce').to_numpy(float)
    idx = df.index
    if isinstance(idx, pd.TimedeltaIndex): return idx.total_seconds().astype(float)
    if isinstance(idx, pd.DatetimeIndex):  return (idx.view('int64')/1e9).astype(float)
    return np.arange(len(df), dtype=float) / float(fs)

def _interp_fix(df, cols, tsec):
    d = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    for c in cols: d[c] = pd.to_numeric(d[c], errors='coerce')
    order = np.argsort(tsec); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    d = d.iloc[order].interpolate('linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
    return d.iloc[inv]

def _lfp_phase_trough0(lfp, fs, theta_band=(4,12)):
    th  = _sos_bandpass(lfp, fs, *theta_band)
    ph  = np.angle(hilbert(th))   # (-π, π], ~0 at peak
    # troughs = negative peaks in theta LFP
    min_dist = int(max(1, 0.8 * fs / float(theta_band[1])))
    trough_idx, _ = find_peaks(-th, distance=min_dist)
    # global phase offset so troughs ~ 0 rad
    if trough_idx.size >= 3:
        phi0 = np.angle(np.nanmean(np.exp(1j * ph[trough_idx])))
    else:
        phi0 = np.pi
    phi = (ph - phi0) % (2*np.pi)     # [0, 2π), 0 = trough
    return phi, trough_idx

def _rayleigh(ph):
    ph = np.asarray(ph, float); ph = ph[np.isfinite(ph)]
    n = ph.size
    if n == 0: return np.nan, np.nan
    C = np.sum(np.cos(ph)); S = np.sum(np.sin(ph))
    Rbar = np.sqrt(C*C + S*S) / n
    Z = n * Rbar * Rbar
    p = np.exp(-Z) * (1 + (2*Z - Z*Z)/(4*n) - (24*Z - 132*Z*Z + 76*Z**3 - 9*Z**4)/(288*n**2))
    p = float(np.clip(p, 0.0, 1.0))
    return float(Z), p

def _zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

# ---------- core: event-based preferred phases for each optical channel ----------
def compare_regions_theta_events_on_subset(
        df_theta: pd.DataFrame,
        fs: float,
        lfp_col: str = 'LFP_2',
        chan_map = {'sig_raw': 'CA1_L', 'ref_raw': 'CA1_R', 'zscore_raw': 'CA3_L'},
        theta_band = (4,12),
        # event detection knobs (must match preferred_phase_from_events)
        height_factor=3.0,
        distance_samples=20,
        prominence=None,
        # optionally provide precomputed event indices per channel: {opt_col: np.ndarray}
        events_dict=None,
        # continuous metrics
        max_lag_ms: int = 250
    ):
    """
    Returns:
      {
        'phase_reference': 'lfp_trough_zero (events)',
        'trough_indices': <np.ndarray>,
        'event_phase_mu_deg': {region: deg},
        'event_plv_R': {region: R},
        'event_rayleigh_p': {region: p},
        'event_n': {region: n},
        'events_used': {opt_col: np.ndarray of indices},
        'coherence_theta': {'A-B': value},
        'env_xcorr': {'CA3_L->CA1_L': {...}, ...},
        'LI_stats': {'mean':..., 'std':...}
      }
    """
    # ---- event-based phase preference (delegates to your function) -----------
    event_phase_mu_deg, event_plv_R = {}, {}
    event_rayleigh_p, event_n = {}, {}
    events_used = {}
    trough_global = None  # will store trough indices from the first call

    for opt_col, region_name in chan_map.items():
        if opt_col not in df_theta.columns:
            continue
        ev_idx = None if events_dict is None else events_dict.get(opt_col, None)

        res = preferred_phase_from_events(
            df_theta, fs,
            lfp_col=lfp_col,
            opt_col=opt_col,
            theta_band=theta_band,
            height_factor=height_factor,
            distance_samples=distance_samples,
            prominence=prominence,
            use_event_indices=ev_idx,
            plot=False
        )

        event_phase_mu_deg[region_name] = res['preferred_phase_deg']
        event_plv_R[region_name]        = res['modulation_depth_R']
        event_rayleigh_p[region_name]   = res['p']
        event_n[region_name]            = res['n_events']
        events_used[opt_col]            = res['event_indices']
        # keep the first trough set as reference (all are computed from same LFP & df)
        if trough_global is None:
            trough_global = res['trough_indices']

    out = {
        'phase_reference': 'lfp_trough_zero (events)',
        'trough_indices': trough_global if trough_global is not None else np.array([], int),
        'event_phase_mu_deg': event_phase_mu_deg,
        'event_plv_R': event_plv_R,
        'event_rayleigh_p': event_rayleigh_p,
        'event_n': event_n,
        'events_used': events_used,
        'coherence_theta': {},
        'env_xcorr': {},
        'LI_stats': {}
    }

    # ---- continuous metrics (same as before) ---------------------------------
    # For these we clean/interpolate minimal columns to avoid NaNs breaking Welch/coherence.
    tsec = _time_seconds(df_theta, fs)
    cols_cont = [lfp_col] + [c for c in chan_map.keys() if c in df_theta.columns]
    X = _interp_fix(df_theta, cols_cont, tsec)

    # Build continuous versions (raw high-pass + theta env) for coherence/x-corr/LI
    regs = {}
    for opt_col, region_name in chan_map.items():
        if opt_col not in X.columns:
            continue
        x_raw   = _sos_highpass(X[opt_col].to_numpy(float), fs, cutoff=0.5)
        x_theta = _sos_bandpass(x_raw, fs, *theta_band)
        regs[region_name] = {'raw': x_raw, 'env': np.abs(hilbert(x_theta))}

    # Pairwise theta coherence across all regions (use raw high-pass)
    names = list(regs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            a = regs[n1]['raw']; b = regs[n2]['raw']
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            if a.size < 256:
                out['coherence_theta'][f'{n1}-{n2}'] = np.nan
                continue
            nper = int(min(max(256, 4*fs), a.size, b.size))
            f, Cxy = coherence(a, b, fs=fs, nperseg=nper)
            sel = (f >= theta_band[0]) & (f <= theta_band[1])
            out['coherence_theta'][f'{n1}-{n2}'] = float(np.nanmean(Cxy[sel])) if np.any(sel) else np.nan

    # Envelope cross-correlation: CA3_L vs CA1_L/R (if CA3_L present)
    max_lag = int((max_lag_ms/1000.0)*fs)
    if 'CA3_L' in regs:
        for target in ['CA1_L', 'CA1_R']:
            if target not in regs:
                continue
            a_full = regs['CA3_L']['env']; b_full = regs[target]['env']
            m = np.isfinite(a_full) & np.isfinite(b_full)
            a = _zscore(a_full[m]); b = _zscore(b_full[m])
            n = min(a.size, b.size)
            if n < 32:
                out['env_xcorr'][f'CA3_L->{target}'] = {'peak_r': np.nan, 'lag_ms': np.nan, 'note': 'insufficient'}
            else:
                a, b = a[:n], b[:n]
                xcorr = correlate(a - np.mean(a), b - np.mean(b), mode='full')
                lags  = np.arange(-n+1, n)
                sel   = (lags >= -max_lag) & (lags <= max_lag)
                lags_sel = lags[sel]; xcorr_sel = xcorr[sel]
                k = int(np.nanargmax(xcorr_sel))
                denom = (np.std(a)*np.std(b)*n)
                peak_r = float(xcorr_sel[k] / denom) if denom > 0 else np.nan
                out['env_xcorr'][f'CA3_L->{target}'] = {
                    'peak_r': peak_r,
                    'lag_ms': float(1000.0 * lags_sel[k] / fs)
                }

    # Lateralisation index on raw high-pass CA1_L vs CA1_R
    if 'CA1_L' in regs and 'CA1_R' in regs:
        L = _zscore(regs['CA1_L']['raw']); Rr = _zscore(regs['CA1_R']['raw'])
        li = (L - Rr) / (np.abs(L) + np.abs(Rr) + 1e-9)
        good = np.isfinite(li)
        out['LI_stats'] = {
            'mean': float(np.nanmean(li[good])) if np.any(good) else np.nan,
            'std':  float(np.nanstd(li[good]))  if np.any(good) else np.nan
        }

    return out


# ==== VIS PLOTS FOR THETA RESULTS (works with your df_theta structure) =====
def plot_phase_roses_events(
        df_theta, fs, lfp_col='LFP_2',
        chan_map={'sig_raw':'CA1_L','ref_raw':'CA1_R','zscore_raw':'CA3_L'},
        theta_band=(4,12),
        height_factor=3.0, distance_samples=20, prominence=None,
        events_dict=None, savepath=None, prefix='events',
        # styling
        suptitle_fs=22, rose_title_fs=18, polar_tick_fs=16,
        radial_tick_fs=14, bar_title_fs=18, bar_tick_fs=16, bar_label_fs=16,
        line_w=2.8, arrow_w=4.0
    ):
    """
    Event-based phase roses with 0° = LFP trough.
    Panels are ordered left→right as CA1_R, CA1_L, CA3_L when present.
    Uses helper `_lfp_phase_trough0` and standard SciPy peak detection.
    """
    # ---- LFP phase (trough = 0°) ----
    lfp = pd.to_numeric(df_theta[lfp_col], errors='coerce').to_numpy(float)
    theta_phase, trough_idx = _lfp_phase_trough0(lfp, fs, theta_band)

    # ---- Collect event-phase stats per channel ----
    dphi_dict, R_dict, n_dict = {}, {}, {}
    for opt_col, name in chan_map.items():
        if opt_col not in df_theta.columns:
            continue
        x = pd.to_numeric(df_theta[opt_col], errors='coerce').to_numpy(float)
        if events_dict is not None and opt_col in events_dict:
            ev_idx = np.asarray(events_dict[opt_col], int)
        else:
            base = np.nanmedian(x)
            mad  = median_abs_deviation(x, scale=1.0, nan_policy='omit')
            thr  = base + height_factor * mad
            ev_idx, _ = find_peaks(x, height=thr, distance=int(distance_samples), prominence=prominence)

        if ev_idx.size == 0:
            dphi, Rval, nval = np.array([]), np.nan, 0
        else:
            dphi = theta_phase[ev_idx]  # [0, 2π), trough=0
            C = np.mean(np.cos(dphi)); S = np.mean(np.sin(dphi))
            Rval = float(np.sqrt(C*C + S*S))
            nval = int(ev_idx.size)

        dphi_dict[name] = dphi
        R_dict[name]    = Rval
        n_dict[name]    = nval

    # ---- Desired left→right order: CA1_R, CA1_L, CA3_L ----
    desired = ['CA1_R', 'CA1_L', 'CA3_L']
    names = [n for n in desired if n in dphi_dict]
    if not names:
        names = list(dphi_dict.keys())

    # ---- Polar roses ----
    n = len(names)
    fig = plt.figure(figsize=(4.8*n, 5.2))
    for i, name in enumerate(names, 1):
        ax = fig.add_subplot(1, n, i, projection='polar')
        phi = dphi_dict[name]

        ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
        ax.set_thetagrids([0,90,180,270], labels=['0','90','180','270'], fontsize=polar_tick_fs)
        ax.set_ylim(0, 1.0); ax.set_yticks([0.5,1.0]); ax.set_yticklabels(['0.5','1'], fontsize=radial_tick_fs)

        if phi.size == 0:
            ax.set_title(f"{name}\nno events", fontsize=rose_title_fs)
            continue

        bins = 30
        edges   = np.linspace(0, 2*np.pi, bins+1)
        counts, _ = np.histogram(phi, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        r = counts/counts.max() if counts.max()>0 else counts.astype(float)
        ax.plot(np.r_[centers, centers[0]], np.r_[r, r[0]], linewidth=line_w)

        # mean vector
        C = np.mean(np.cos(phi)); S = np.mean(np.sin(phi))
        mu = (np.arctan2(S, C)) % (2*np.pi)
        R  = R_dict[name] if np.isfinite(R_dict[name]) else 0.0
        ax.annotate("", xy=(mu, 0.9*R), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", linewidth=arrow_w))

        ax.set_title(f"{name}\nμ={np.degrees(mu):.1f}°, R={R:.2f}, n={n_dict[name]}",
                     fontsize=rose_title_fs)

    fig.suptitle("Event-based preferred phase (0° = LFP trough)", y=1.02, fontsize=suptitle_fs)
    fig.tight_layout()
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(os.path.join(savepath, f"{prefix}_rose.png"), dpi=300, bbox_inches='tight')

    # ---- Modulation Depth (R) bars in same order ----
    fig2, ax2 = plt.subplots(figsize=(0.6*n+2, 3.8))
    bars = [R_dict.get(k, np.nan) for k in names]
    ax2.bar(np.arange(n), bars)
    ax2.set_xticks(np.arange(n)); ax2.set_xticklabels(names, fontsize=bar_tick_fs)
    ax2.set_ylabel('Modulation Depth (R)', fontsize=bar_label_fs)
    ax2.set_ylim(0, 1)
    ax2.set_title('Modulation Depth (R)', fontsize=bar_title_fs)
    ax2.tick_params(axis='y', labelsize=bar_tick_fs)
    fig2.tight_layout()
    if savepath:
        fig2.savefig(os.path.join(savepath, f"{prefix}_plv.png"), dpi=300, bbox_inches='tight')

    return fig, fig2



def _theta_envelope(x, fs, theta_band=(4,12)):
    th = _sos_bandpass(x, fs, theta_band[0], theta_band[1])
    return np.abs(hilbert(th))


def plot_env_xcorr_optical(
        df_theta: pd.DataFrame,
        fs: float,
        chan_map = {'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
        theta_band=(4,12),
        max_lag_ms=250,
        # styling
        bar_title_fs=18, bar_tick_fs=14, bar_label_fs=16,
        savepath=None, prefix='ex_events'
    ):
    """
    Theta envelope cross-correlation between every pair of optical signals.
    Pairs (unordered): CA1_R–CA1_L, CA1_R–CA3_L, CA1_L–CA3_L (if present).

    Positive lag_ms => second name lags the first (first leads).

    Now outputs TWO plots: (1) peak r bars, (2) lag (ms) bars.
    """
    # Collect available optical channels
    present = [(col, name) for col, name in chan_map.items() if col in df_theta.columns]
    names   = [name for _, name in present]
    if len(present) < 2:
        raise ValueError("Need at least two optical channels present in df_theta.")

    # Compute theta envelopes once
    env = {}
    for col, name in present:
        env[name] = _theta_envelope(df_theta[col].to_numpy(), fs, theta_band)

    # Build pairs in the requested order (CA1_R, CA1_L, CA3_L)
    desired_order = ['CA1_R','CA1_L','CA3_L']
    ordered = [n for n in desired_order if n in names]
    pairs = [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i+1, len(ordered))]

    # Compute normalized xcorr peak within ±max_lag
    max_lag = int((max_lag_ms/1000.0) * fs)
    labels, rvals, lags = [], [], []
    for a_name, b_name in pairs:
        a_full = env[a_name]; b_full = env[b_name]
        m = np.isfinite(a_full) & np.isfinite(b_full)
        a = _zscore(a_full[m]); b = _zscore(b_full[m])
        n = min(a.size, b.size)
        labels.append(f"{a_name}–{b_name}")
        if n < 32:
            rvals.append(np.nan); lags.append(np.nan)
            continue
        a = a[:n]; b = b[:n]
        xcorr = correlate(a - np.mean(a), b - np.mean(b), mode='full')
        lags_samp = np.arange(-n+1, n)
        sel = (lags_samp >= -max_lag) & (lags_samp <= max_lag)
        if not np.any(sel):
            rvals.append(np.nan); lags.append(np.nan)
            continue
        xs  = xcorr[sel]; ls = lags_samp[sel]
        k   = int(np.nanargmax(xs))
        denom = (np.std(a)*np.std(b)*n)
        peak_r = float(xs[k] / denom) if denom > 0 else np.nan
        lag_ms = 1000.0 * ls[k] / fs   # + => b lags a (a leads)
        rvals.append(peak_r); lags.append(float(lag_ms))

    # -------- figure 1: peak r --------
    fig_r, ax_r = plt.subplots(figsize=(max(5, 1*len(labels)), 5))
    ax_r.bar(np.arange(len(labels)), rvals)
    ax_r.set_ylabel('peak r', fontsize=bar_label_fs)
    ax_r.set_ylim(0, 1)
    ax_r.set_title('Theta envelope cross-correlation', fontsize=bar_title_fs)
    ax_r.set_xticks(np.arange(len(labels)))
    ax_r.set_xticklabels(labels, rotation=15, fontsize=bar_tick_fs)
    ax_r.tick_params(axis='y', labelsize=bar_tick_fs)
    fig_r.tight_layout()

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

    data = {'labels': labels, 'peak_r': np.array(rvals, float), 'lag_ms': np.array(lags, float)}
    return (fig_r, ax_r), (fig_l, ax_l), data

        
# ---- helpers ----

def _slice_df_by_time(df, fs, start_time, end_time):
    t = _time_seconds(df, fs)
    m = (t >= start_time) & (t <= end_time)
    return df.loc[m], t[m]

def _bandpass_theta(x, fs, band=(4,12), order=4):
    x = np.asarray(pd.to_numeric(x, errors='coerce'), float)
    sos = butter(order, [band[0]/(fs/2), band[1]/(fs/2)], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def _find_theta_peaks(x_theta, fs, distance_ms=80, height_factor=None, prominence=None):
    """Find peaks on theta-filtered optical trace."""
    dist = max(1, int(round(distance_ms/1000.0 * fs)))
    height = None
    if height_factor is not None:
        base = np.nanmedian(x_theta)
        mad  = median_abs_deviation(x_theta, scale=1.0, nan_policy='omit')
        height = base + height_factor * mad
    idx, _ = find_peaks(x_theta, distance=dist, prominence=prominence, height=height)
    return idx

# ---- main ----
def plot_theta_peaks_twoLFP(
        df_aligned: pd.DataFrame,
        fs: float,
        start_time: float,
        end_time: float,
        lfp_cols=('LFP_2','LFP_3'),
        opt_cols=('ref_raw','sig_raw','zscore_raw'),  # CA1_R, CA1_L, CA3_L
        theta_band=(4,12),
        distance_ms=80,
        height_factor=1.5,
        prominence=None,
        label_fs=16,
        tick_fs=14,
        leg_fs=16,                 # <- legend fontsize
        legend_loc='upper right',  # <- legend location on each subplot
        savepath=None,
        filename='theta_twoLFP_optDots.png'
    ):
    """
    Plot theta-filtered LFP_2 & LFP_3 and theta-filtered CA1_R/CA1_L/CA3_L.
    Overlay optical theta-peak dots on BOTH LFP subplots.
    Legends show ONLY the trace on that subplot (no dot legend entries).
    """
    # slice by time
    seg, t = _slice_df_by_time(df_aligned, fs, start_time, end_time)
    if seg.empty:
        raise ValueError("Selected time window has no data.")
    for c in lfp_cols + opt_cols:
        if c not in seg.columns:
            raise ValueError(f"Missing column '{c}' in df_aligned.")

    pal = sns.color_palette("husl", 8)
    color_map = {'CA1_R': pal[0], 'CA1_L': pal[3], 'CA3_L': pal[2],
                 'LFP_2': pal[5], 'LFP_3': pal[6]}

    # theta-filter
    lfp_name_1, lfp_name_2 = lfp_cols
    lfp1_th = _bandpass_theta(seg[lfp_name_1].to_numpy(), fs, theta_band)
    lfp2_th = _bandpass_theta(seg[lfp_name_2].to_numpy(), fs, theta_band)

    name_map = {opt_cols[0]:'CA1_R', opt_cols[1]:'CA1_L', opt_cols[2]:'CA3_L'}
    opt_theta, peaks_t = {}, {}
    for col, region in name_map.items():
        xt = _bandpass_theta(seg[col].to_numpy(), fs, theta_band)
        opt_theta[region] = xt
        idx = _find_theta_peaks(xt, fs, distance_ms=distance_ms,
                                height_factor=height_factor, prominence=prominence)
        peaks_t[region] = t[idx]

    # figure
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=tick_fs)

    # --- LFP_2 θ with dots ---
    ax = axes[3]
    line1, = ax.plot(t, lfp1_th, color=color_map['LFP_2'], lw=2.5, label=f'{lfp_name_1} θ')
    for region in ('CA1_R','CA1_L','CA3_L'):
        if peaks_t[region].size:
            y = np.interp(peaks_t[region], t, lfp1_th)
            ax.scatter(peaks_t[region], y, s=30, marker='o', color=color_map[region], zorder=3)
    ax.set_ylabel(f'{lfp_name_1} θ', fontsize=label_fs)
    ax.legend(handles=[line1], fontsize=leg_fs, loc=legend_loc, frameon=False)

    # --- LFP_3 θ with dots ---
    ax = axes[4]
    line2, = ax.plot(t, lfp2_th, color=color_map['LFP_3'], lw=2.5, label=f'{lfp_name_2} θ')
    for region in ('CA1_R','CA1_L','CA3_L'):
        if peaks_t[region].size:
            y = np.interp(peaks_t[region], t, lfp2_th)
            ax.scatter(peaks_t[region], y, s=30, marker='o', color=color_map[region], zorder=3)
    ax.set_ylabel(f'{lfp_name_2} θ', fontsize=label_fs)
    ax.legend(handles=[line2], fontsize=leg_fs, loc=legend_loc, frameon=False)

    # --- Optical θ traces (each with its own legend) ---
    ax = axes[0]
    line3, = ax.plot(t, opt_theta['CA1_R'], color=color_map['CA1_R'], lw=2.5, label='CA1_R θ')
    ax.set_ylabel('CA1_R θ', fontsize=label_fs)
    ax.legend(handles=[line3], fontsize=leg_fs, loc=legend_loc, frameon=False)

    ax = axes[1]
    line4, = ax.plot(t, opt_theta['CA1_L'], color=color_map['CA1_L'], lw=2.5, label='CA1_L θ')
    ax.set_ylabel('CA1_L θ', fontsize=label_fs)
    ax.legend(handles=[line4], fontsize=leg_fs, loc=legend_loc, frameon=False)

    ax = axes[2]
    line5, = ax.plot(t, opt_theta['CA3_L'], color=color_map['CA3_L'], lw=2.5, label='CA3_L θ')
    ax.set_ylabel('CA3_L θ', fontsize=label_fs)
    ax.set_xlabel('Time (s)', fontsize=label_fs)
    ax.legend(handles=[line5], fontsize=leg_fs, loc=legend_loc, frameon=False)

    fig.suptitle('Theta-filtered LFPs and optical peaks', fontsize=label_fs+2, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        out_path = os.path.join(savepath, filename)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    else:
        out_path = None

    return fig, axes, {'peaks_t': peaks_t}

#%%
dpath= r'G:\2025_ATLAS_SPAD\MultiFibreOEC\1887932_Jedi2p_Multi_ephysbad\MovingTrialsDLC'

recordingName='SyncRecording2'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 

# Grab once from your session
df = Recording1.Ephys_tracking_spad_aligned
fs = Recording1.fs
LFP = 'LFP_3'
optical_channel='zscore_raw'
savename='ThetaSave'
save_path = os.path.join(dpath,savename)
save_dir  = os.path.join(dpath, "theta_figs_events")
os.makedirs(save_dir, exist_ok=True)
# 1) You already labelled theta + troughs:
Recording1.pynacollada_label_theta(LFP, Low_thres=-0.3, High_thres=10, save=False, plot_theta=True)
df_theta  = Recording1.theta_part

trough_idx, peak_idx = Recording1.plot_theta_correlation(df_theta, LFP, save_path,optical_channel)
#%%
# # 2) All three optical channels at once (no plots)
ev_all = preferred_phase_events_multi(
    df, fs, lfp_col='LFP_3',
    chan_map={'sig_raw':'CA1_L','ref_raw':'CA1_R','zscore_raw':'CA3_L'},
    theta_band=(4,12),
    height_factor=3.0, distance_samples=20, prominence=None,
    plot_each=False
)
for region, res in ev_all.items():
    print(region, "pref=", f"{res['preferred_phase_deg']:.1f}°",
          "R=", f"{res['modulation_depth_R']:.3f}", "p=", f"{res['p']:.2g}", "n=", res['n_events'])
#%%
# Compute event-based metrics (0° = trough)
theta_res_ev = compare_regions_theta_events_on_subset(
    df_theta=df,
    fs=fs,
    lfp_col='LFP_2',
    chan_map={'sig_raw':'CA1_L','ref_raw':'CA1_R','zscore_raw':'CA3_L'},
    theta_band=(4,12),
    height_factor=3.0,
    distance_samples=20,
    prominence=None,
    events_dict=None,        # or provide your own indices per channel
    max_lag_ms=250
)

# Print in the same style
print('Preferred phase (deg, events):', theta_res_ev['event_phase_mu_deg'])
print('PLV R (events):', theta_res_ev['event_plv_R'])
print('Rayleigh p (events):', theta_res_ev['event_rayleigh_p'])
print('n events:', theta_res_ev['event_n'])
print('Theta coherence:', theta_res_ev['coherence_theta'])
print('CA3↔CA1 env xcorr:', theta_res_ev['env_xcorr'])
print('LI stats:', theta_res_ev['LI_stats'])
#%%
# Plots
'This is to plot theta phase preference of optical signal on LFP theta'
plot_phase_roses_events(
    df_theta, fs, lfp_col='LFP_2',
    chan_map={'sig_raw':'CA1_L','ref_raw':'CA1_R','zscore_raw':'CA3_L'},
    theta_band=(5,11),
    height_factor=3.0, distance_samples=20, prominence=None,
    events_dict=None,
    savepath=save_dir, prefix='ex_events'
)

#%%
# Bars (peak r and lag) across pairs
(fig_peak, ax_peak), (fig_lag, ax_lag), data_bars = plot_env_xcorr_optical(
    df_theta=df,
    fs=Recording1.fs,
    chan_map={'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'},
    theta_band=(4,12),
    max_lag_ms=250,
    savepath=save_dir,
    prefix="ex_events"
)
#%%
'''Plot example traces to select best one'''
dpath= r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day2'
recordingName='SyncRecording4'

# dpath= r'G:\2025_ATLAS_SPAD\MultiFibre\1887932_Jedi2p_Multi_ephysbad\MovingTrialsDLC'
# recordingName='SyncRecording1'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 

LFP = 'LFP_3'
# Grab what you need from the class instance once
df = Recording1.Ephys_tracking_spad_aligned
fs = Recording1.fs
savepath = Recording1.savepath  # or a custom folder path

timewindow = 3
viewNum = 9
for i in range(viewNum):
    start = timewindow * i
    end   = timewindow * (i + 1)
    MakePlots.plot_segment_feature_multiROI_independent(
        df_aligned=df,
        fs=fs,
        savepath=savepath,
        LFP_channel=LFP,
        start_time=start,
        end_time=end,
        SPAD_cutoff=50,
        lfp_cutoff=500,
        five_xticks=False
    )
#%%
'''Plot example traces with power spectrogram'''
start = 15.1
end   = 18.1
# start = 14
# end   = 19
fig, ax, out = MakePlots.plot_segment_feature_multiROI_twoLFP(
    df_aligned=df, fs=Recording1.fs, savepath=Recording1.savepath,
    LFP_channels=('LFP_2','LFP_3'),
    start_time=start, end_time=end, SPAD_cutoff=50, lfp_cutoff=500,
    label_fs=18, tick_fs=16, leg_fs=18, five_xticks=True
)
#%%
'''Plot example theta band trace with power spectrogram'''
fig, axes, peaks = plot_theta_peaks_twoLFP(
    df_aligned=df,   # or Recording1.Ephys_tracking_spad_aligned
    fs=Recording1.fs,
    start_time=start, end_time=end,
    lfp_cols=('LFP_2','LFP_3'),
    opt_cols=('ref_raw','sig_raw','zscore_raw'),
    theta_band=(4,12),
    distance_ms=80, height_factor=1.5, prominence=None,
    leg_fs=18, label_fs=18, tick_fs=16,
    savepath=os.path.join(dpath, 'theta_figs'),
    filename='theta_twoLFP_optDots.png'
)