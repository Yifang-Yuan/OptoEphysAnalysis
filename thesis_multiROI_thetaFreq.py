# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 00:04:40 2025

@author: yifan
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, coherence, welch, correlate
from scipy.stats import circmean

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, welch, coherence

# -------------------- helpers --------------------
def _time_seconds(df, fs, pref_cols=("timestamps","time","time_s")):
    """Return a float seconds vector for time (prefers 'timestamps')."""
    for c in pref_cols:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            return t
    # fallback: handle time-like index
    idx = df.index
    if isinstance(idx, pd.TimedeltaIndex):  return idx.total_seconds().astype(float)
    if isinstance(idx, pd.DatetimeIndex):   return (idx.view("int64")/1e9).astype(float)
    # last resort: assume uniform sampling
    return np.arange(len(df), dtype=float) / float(fs)

def _numeric_interp_sorted(df, cols, tsec):
    """Coerce to numeric + interpolate in time order → return arrays (sorted by time)."""
    d = df[cols].replace([np.inf,-np.inf], np.nan).copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    order = np.argsort(tsec)
    d = d.iloc[order]
    d = d.interpolate(method="linear", limit_direction="both").fillna(method="ffill").fillna(method="bfill")
    # sanity: ensure not all-NaN
    for c in cols:
        arr = d[c].to_numpy()
        if not np.isfinite(arr).any():
            raise ValueError(f"Column '{c}' is all-NaN after interpolation.")
    return d, tsec[order]

def _sos_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float))

def _phase(x): return np.angle(hilbert(x))
def _inst_freq_hz(x_theta, fs):
    ph = np.unwrap(_phase(x_theta))
    w  = np.diff(ph) * fs / (2*np.pi)
    return np.r_[w, np.nan]

def _plv(phA, phB):
    m = np.isfinite(phA) & np.isfinite(phB)
    if m.sum() < 8: return np.nan
    dphi = np.angle(np.exp(1j*(phA[m] - phB[m])))
    return float(np.abs(np.nanmean(np.exp(1j*dphi))))

def _peak_theta_welch(x, fs, band):
    if not np.isfinite(x).any(): return np.nan
    nperseg = int(min(len(x), max(2048, int(8*fs))))  # good resolution but <= length
    if nperseg < 128: return np.nan
    f, P = welch(x, fs=fs, nperseg=nperseg, detrend=False)
    sel = (f >= band[0]) & (f <= band[1])
    if not np.any(sel): return np.nan
    return float(f[sel][np.nanargmax(P[sel])])

# -------------------- ONE-DF ANALYSIS --------------------
def theta_compare_single_df(df, fs,
                            lfp_a="LFP_2", lfp_b="LFP_3",
                            optical_cols=("sig_raw","ref_raw","zscore_raw"),
                            theta_band=(4,12),
                            make_plot=True):
    """
    Run theta frequency/PLV/coherence using JUST this DataFrame.
    Uses df['timestamps'] if present; otherwise tries index; otherwise assumes uniform time @ fs.
    """
    # pick columns that exist
    have_lfp = [c for c in ["LFP_1","LFP_2","LFP_3","LFP_4"] if c in df.columns]
    if lfp_a not in have_lfp and have_lfp:
        lfp_a = have_lfp[0]; print(f"[info] LFP A not found; using '{lfp_a}'.")
    if (lfp_b not in have_lfp) or (lfp_b == lfp_a):
        lfp_b = next((c for c in have_lfp if c != lfp_a), None)
        if lfp_b is None:
            raise ValueError("Need two distinct LFP columns in this DataFrame.")
        print(f"[info] Using '{lfp_b}' as LFP B.")
    optical_cols = tuple(c for c in optical_cols if c in df.columns)
    if not optical_cols:
        raise ValueError("None of ('sig_raw','ref_raw','zscore_raw') are in this DataFrame.")

    cols = [lfp_a, lfp_b, *optical_cols]

    # 1) build time (seconds) and clean/interpolate in time order
    tsec = _time_seconds(df, fs, pref_cols=("timestamps","time","time_s"))
    X, tsec = _numeric_interp_sorted(df, cols, tsec)

    # 2) theta-band filter (SOS for numerical stability at 10 kHz)
    th = {c: _sos_bandpass(X[c].to_numpy(), fs, *theta_band) for c in cols}

    # 3) LFP–LFP metrics
    A, B = th[lfp_a], th[lfp_b]
    phA, phB = _phase(A), _phase(B)
    peakA = _peak_theta_welch(A, fs, theta_band)
    peakB = _peak_theta_welch(B, fs, theta_band)
    ifA   = float(np.nanmedian(_inst_freq_hz(A, fs)))
    ifB   = float(np.nanmedian(_inst_freq_hz(B, fs)))
    plvAB = _plv(phA, phB)

    # coherence (safe nperseg)
    m = np.isfinite(A) & np.isfinite(B)
    coh = np.nan
    if m.sum() >= 256:
        nper = int(min(m.sum(), max(1024, int(4*fs))))
        fC, Cxy = coherence(A[m], B[m], fs=fs, nperseg=nper)
        sel = (fC >= theta_band[0]) & (fC <= theta_band[1])
        coh = float(np.nanmean(Cxy[sel])) if np.any(sel) else np.nan

    # 4) optical PLVs to each LFP
    rows = []
    for opt in optical_cols:
        phO = _phase(th[opt])
        vA = _plv(phO, phA)
        vB = _plv(phO, phB)
        best = None if (np.isnan(vA) and np.isnan(vB)) else (lfp_a if np.nan_to_num(vA) >= np.nan_to_num(vB) else lfp_b)
        rows.append({"optical_channel": opt,
                     f"PLV_to_{lfp_a}": vA,
                     f"PLV_to_{lfp_b}": vB,
                     "best_lfp": best})
    opt_df = pd.DataFrame(rows)

    # 5) tidy summary
    lfp_table = pd.DataFrame({
        "metric": ["fs_input_Hz",
                   f"{lfp_a}_peak_freq", f"{lfp_b}_peak_freq",
                   f"{lfp_a}_inst_freq_median", f"{lfp_b}_inst_freq_median",
                   "lfp_phase_diff_plv_R", "lfp_theta_coherence"],
        "value": [fs, peakA, peakB, ifA, ifB, plvAB, coh]
    })

    # 6) quick plot (optional)
    if make_plot and len(opt_df):
        import matplotlib.pyplot as plt
        x = np.arange(len(opt_df)); w = 0.35
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(x-w/2, opt_df[f"PLV_to_{lfp_a}"], width=w, label=f"PLV→{lfp_a}")
        ax.bar(x+w/2, opt_df[f"PLV_to_{lfp_b}"], width=w, label=f"PLV→{lfp_b}")
        ax.set_xticks(x); ax.set_xticklabels(opt_df["optical_channel"])
        ax.set_ylabel("PLV (R)"); ax.set_title("Optical PLV to each LFP (theta)")
        ax.legend(); fig.tight_layout(); plt.show()

    return {"peak_freq": {lfp_a: peakA, lfp_b: peakB},
            "inst_freq_median": {lfp_a: ifA, lfp_b: ifB},
            "lfp_phase_diff_plv_R": plvAB,
            "lfp_theta_coherence": coh}, lfp_table, opt_df


import matplotlib.pyplot as plt
from scipy.signal import coherence, csd

# ---------- small robust helpers ----------

def _get_numeric_sorted(df, cols, tsec):
    d = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    for c in cols: d[c] = pd.to_numeric(d[c], errors='coerce')
    order = np.argsort(tsec)
    t = tsec[order]
    d = d.iloc[order].interpolate('linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
    return d, t

# ---------- 1) Frequency-resolved coherence & cross-spectral phase ----------
def plot_lfp_coherence_and_phase(df, fs, lfp_a='LFP_2', lfp_b='LFP_3',
                                 fmin=2, fmax=20, nperseg_s=2.0, overlap=0.5,
                                 title_suffix=' (theta focus 4–12 Hz)'):
    cols = [lfp_a, lfp_b]
    tsec = _time_seconds(df, fs)
    X, t = _get_numeric_sorted(df, cols, tsec)
    xa = X[lfp_a].to_numpy(); xb = X[lfp_b].to_numpy()

    nperseg = int(max(128, min(len(xa), nperseg_s*fs)))
    noverlap = int(nperseg*overlap)

    # coherence
    f, Cxy = coherence(xa, xb, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # cross-spectral phase (degrees) from CSD
    f2, Sxy = csd(xa, xb, fs=fs, nperseg=nperseg, noverlap=noverlap)
    phase_deg = np.degrees(np.angle(Sxy))

    # focus freq range
    m = (f>=fmin) & (f<=fmax)
    m2 = (f2>=fmin) & (f2<=fmax)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6), sharex=True)
    ax1.plot(f[m], Cxy[m])
    ax1.set_ylabel('Coherence')
    ax1.set_ylim(0, 1)
    ax1.set_title(f'Coherence: {lfp_a} vs {lfp_b}{title_suffix}')

    ax2.plot(f2[m2], phase_deg[m2])
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Cross-spectral phase (deg)')
    ax2.set_title('Cross-spectral phase (angle of CSD)')

    # highlight theta band
    ax1.axvspan(4, 12, alpha=0.1)
    ax2.axvspan(4, 12, alpha=0.1)
    fig.tight_layout()
    return {'f': f, 'coh': Cxy, 'f_phase': f2, 'phase_deg': phase_deg}

# ---------- 2) Bout-wise theta peak frequency boxplots ----------
def plot_boutwise_theta_peaks(df, fs, lfp_a='LFP_2', lfp_b='LFP_3',
                              theta_band=(4,12), win_s=2.0, step_s=1.0,
                              nperseg_min=256):
    cols = [lfp_a, lfp_b]
    tsec = _time_seconds(df, fs)
    X, t = _get_numeric_sorted(df, cols, tsec)
    xa = X[lfp_a].to_numpy(); xb = X[lfp_b].to_numpy()

    win = int(win_s*fs); step = int(step_s*fs)
    peaks_a, peaks_b = [], []

    for start in range(0, max(0, len(xa)-win+1), step):
        seg_a = xa[start:start+win]
        seg_b = xb[start:start+win]
        if len(seg_a) < nperseg_min or len(seg_b) < nperseg_min:
            continue
        # Welch per window
        fA, PA = welch(seg_a, fs=fs, nperseg=len(seg_a), detrend=False)
        fB, PB = welch(seg_b, fs=fs, nperseg=len(seg_b), detrend=False)
        selA = (fA>=theta_band[0]) & (fA<=theta_band[1])
        selB = (fB>=theta_band[0]) & (fB<=theta_band[1])
        if not np.any(selA) or not np.any(selB):
            continue
        peaks_a.append(float(fA[selA][np.argmax(PA[selA])]))
        peaks_b.append(float(fB[selB][np.argmax(PB[selB])]))

    if len(peaks_a)==0 or len(peaks_b)==0:
        print("Not enough data for bout-wise peaks with the current windowing.")
        return {'peaks_a': peaks_a, 'peaks_b': peaks_b}

    # Boxplot + jitter
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([peaks_a, peaks_b], labels=[lfp_a, lfp_b], showmeans=True)
    # jittered points
    xj = np.concatenate([np.full(len(peaks_a), 1.05), np.full(len(peaks_b), 1.95)])
    yj = np.concatenate([peaks_a, peaks_b])
    ax.scatter(xj, yj, alpha=0.5, s=10)
    ax.set_ylabel('Bout-wise theta peak (Hz)')
    ax.set_title(f'Bout-wise peaks (win={win_s}s, step={step_s}s)')
    fig.tight_layout()

    print(f"Medians: {lfp_a}={np.median(peaks_a):.2f} Hz, {lfp_b}={np.median(peaks_b):.2f} Hz")
    return {'peaks_a': peaks_a, 'peaks_b': peaks_b}


#%%
# Use an unbiased segment (e.g., the whole recording or a union of theta bouts).
# If you insist on using Recording1.theta_part (made from LFP_3), that may bias results toward LFP_3.
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import MakePlots
# Your loading code
dpath = r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day8'
recordingName = 'SyncRecording3'
Recording1 = SyncOEpyPhotometrySession(
    dpath, recordingName,
    IsTracking=False,
    read_aligned_data_from_file=True,
    recordingMode='Atlas', indicator='GEVI'
)

# Define theta using LFP_3 (fine; just note this can slightly bias PLVs toward LFP_3)
Recording1.pynacollada_label_theta('LFP_2', Low_thres=-0.5, High_thres=10, save=False, plot_theta=True)

df_full  = Recording1.Ephys_tracking_spad_aligned
fs       = Recording1.fs
df_theta = Recording1.theta_part  # your theta subset

# A) theta only
results_theta, lfp_table_theta, opt_df_theta = theta_compare_single_df(
    df_theta, Recording1.fs,
    lfp_a='LFP_2', lfp_b='LFP_3',
    optical_cols=('sig_raw','ref_raw','zscore_raw'),
    theta_band=(4,11),
    make_plot=True
)
print(lfp_table_theta.to_string(index=False))
print(opt_df_theta.to_string(index=False))


# Use theta-only subset (recommended), or df_full if you prefer
df_use = df_theta    # or: Recording1.Ephys_tracking_spad_aligned
fs = Recording1.fs   # e.g., 10000

# 1) Coherence + cross-spectral phase
spec = plot_lfp_coherence_and_phase(df_use, fs, lfp_a='LFP_2', lfp_b='LFP_3',
                                    fmin=2, fmax=20, nperseg_s=2.0, overlap=0.5)

# 2) Bout-wise theta peaks (2 s windows sliding by 1 s)
peaks = plot_boutwise_theta_peaks(df_use, fs, lfp_a='LFP_2', lfp_b='LFP_3',
                                  theta_band=(4,12), win_s=2.0, step_s=1.0)


# B) whole recording (no pre-sliced theta)
# results_full, lfp_table_full, opt_df_full = theta_compare_single_df(
#     Recording1.Ephys_tracking_spad_aligned, Recording1.fs,
#     lfp_a='LFP_2', lfp_b='LFP_3',
#     optical_cols=('sig_raw','ref_raw','zscore_raw'),
#     theta_band=(4,12),
#     make_plot=True
# )
# print(lfp_table_theta.to_string(index=False))
# print(opt_df_theta.to_string(index=False))