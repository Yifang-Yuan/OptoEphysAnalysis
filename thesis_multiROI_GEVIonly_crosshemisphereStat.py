# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 19:24:28 2025

@author: yifan
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, coherence
from scipy.stats import wilcoxon



def _sos_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def _theta_coherence_pair(x, y, fs, theta_band=(4,12), nperseg=None, noverlap=None):
    if nperseg is None:
        nperseg = max(256, int(round(fs*2)))  # ~2 s
    if noverlap is None:
        noverlap = nperseg // 2
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    sel = (f >= theta_band[0]) & (f <= theta_band[1])
    return float(np.nanmean(Cxy[sel])) if np.any(sel) else np.nan

def _state_mask(df, movement_col='movement', states='moving'):
    if isinstance(states, str): states = [states]
    if movement_col not in df.columns:
        raise ValueError(f"movement column '{movement_col}' not in df")
    return df[movement_col].isin(states).to_numpy(bool)

def coherence_pair_by_state(df, fs, chan_map, pair=('CA1_R','CA1_L'),
                            theta_band=(4,12), movement_col='movement', state='moving'):
    """
    Theta coherence between an arbitrary label pair (e.g., ('CA1_R','CA3_L'))
    within rows of df where df[movement_col]==state.
    """
    label2col = {lab: col for col, lab in chan_map.items()}
    a, b = pair
    if a not in label2col or b not in label2col:
        raise ValueError(f"pair labels {pair} not in chan_map labels {list(label2col)}")
    ca, cb = label2col[a], label2col[b]
    if ca not in df.columns or cb not in df.columns:
        raise ValueError(f"df missing columns {ca} or {cb}")

    m = _state_mask(df, movement_col=movement_col, states=state)
    xa = df[ca].to_numpy(float)[m]
    xb = df[cb].to_numpy(float)[m]
    if min(xa.size, xb.size) < max(256, int(fs*2)):
        return np.nan

    xa_th = _sos_bandpass(xa, fs, theta_band[0], theta_band[1])
    xb_th = _sos_bandpass(xb, fs, theta_band[0], theta_band[1])
    L = min(len(xa_th), len(xb_th))
    return _theta_coherence_pair(xa_th[:L], xb_th[:L], fs, theta_band)

def batch_pair_theta_coherence_by_state(
        root_dir, recording_names, chan_map, pair=('CA1_R','CA1_L'),
        theta_band=(4,12), movement_col='movement',
        moving_label='moving', notmoving_label='notmoving',
        save_csv_path=None):
    """
    For each recording, compute pair coherence for moving vs notmoving.
    Returns tidy df and paired Wilcoxon stat.
    """
    from SyncOECPySessionClass import SyncOEpyPhotometrySession

    rows = []
    for rec in recording_names:
        sess = SyncOEpyPhotometrySession(root_dir, rec, IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas', indicator='GEVI')
        df = sess.Ephys_tracking_spad_aligned
        fs = sess.fs

        c_m  = coherence_pair_by_state(df, fs, chan_map, pair, theta_band, movement_col, moving_label)
        c_nm = coherence_pair_by_state(df, fs, chan_map, pair, theta_band, movement_col, notmoving_label)

        rows += [
            {'recording': rec, 'state': moving_label,    'coherence': c_m,  'pair': f'{pair[0]}–{pair[1]}'},
            {'recording': rec, 'state': notmoving_label, 'coherence': c_nm, 'pair': f'{pair[0]}–{pair[1]}'},
        ]

    out = pd.DataFrame(rows)

    # Paired test
    pivot = out.pivot(index='recording', columns='state', values='coherence')[[moving_label, notmoving_label]]
    test_df = pivot.dropna()
    if len(test_df) >= 3:
        W, p = wilcoxon(test_df[moving_label], test_df[notmoving_label], alternative='two-sided')
    else:
        W, p = np.nan, np.nan

    if save_csv_path:
        out.to_csv(save_csv_path, index=False)

    return out, (W, p), test_df

# ---------- signal helpers ----------

def coherence_CA1R_CA1L_by_state(df, fs, chan_map, theta_band=(4,12),
                                 movement_col='movement', state='moving'):
    """
    Returns theta coherence between CA1_R and CA1_L during rows where df[movement_col] == state.
    """
    # which df columns correspond to CA1_R and CA1_L?
    label2col = {lab: col for col, lab in chan_map.items()}
    if 'CA1_R' not in label2col or 'CA1_L' not in label2col:
        raise ValueError("chan_map must include CA1_R and CA1_L labels.")

    col_R = label2col['CA1_R']
    col_L = label2col['CA1_L']
    if (col_R not in df.columns) or (col_L not in df.columns):
        raise ValueError(f"Columns {col_R} and/or {col_L} not found in df")

    m = _state_mask(df, movement_col=movement_col, states=state)
    xR = df[col_R].to_numpy(float)[m]
    xL = df[col_L].to_numpy(float)[m]

    # require enough samples for spectral estimate
    if min(xR.size, xL.size) < max(256, int(fs*2)):
        return np.nan

    xR_th = _sos_bandpass(xR, fs, theta_band[0], theta_band[1])
    xL_th = _sos_bandpass(xL, fs, theta_band[0], theta_band[1])

    # trim to common length (in case of any small mismatch)
    L = min(len(xR_th), len(xL_th))
    return _theta_coherence_pair(xR_th[:L], xL_th[:L], fs, theta_band)



# ---------- batch runner across multiple recordings ----------
def batch_cross_hemi_theta_coherence(
        root_dir,
        recording_names,         # e.g., ['SyncRecording1','SyncRecording2', ...]
        chan_map,                # e.g., {'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'}
        theta_band=(4,12),
        movement_col='movement',
        moving_label='moving',
        notmoving_label='notmoving',     # set to your actual label
        save_csv_path=None
    ):
    """
    Loads each recording, computes CA1_R–CA1_L theta coherence for moving and not-moving,
    returns a tidy DataFrame and (statistic, p) from paired Wilcoxon.
    """
    from SyncOECPySessionClass import SyncOEpyPhotometrySession

    rows = []
    for rec in recording_names:
        sess = SyncOEpyPhotometrySession(root_dir, rec, IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas', indicator='GEVI')
        df = sess.Ephys_tracking_spad_aligned
        fs = sess.fs

        coh_m = coherence_CA1R_CA1L_by_state(df, fs, chan_map,
                                             theta_band=theta_band,
                                             movement_col=movement_col,
                                             state=moving_label)
        coh_nm = coherence_CA1R_CA1L_by_state(df, fs, chan_map,
                                              theta_band=theta_band,
                                              movement_col=movement_col,
                                              state=notmoving_label)

        rows.append({'recording': rec, 'state': moving_label,    'coherence': coh_m})
        rows.append({'recording': rec, 'state': notmoving_label, 'coherence': coh_nm})

    out = pd.DataFrame(rows)

    # paired test across recordings (only where both states present and finite)
    pivot = out.pivot(index='recording', columns='state', values='coherence')
    pivot = pivot[[moving_label, notmoving_label]]  # stable order
    df_test = pivot.dropna(axis=0, how='any')
    if len(df_test) >= 3:
        stat, p = wilcoxon(df_test[moving_label], df_test[notmoving_label], alternative='two-sided')
    else:
        stat, p = np.nan, np.nan  # not enough paired samples

    if save_csv_path:
        out.to_csv(save_csv_path, index=False)

    return out, (stat, p), df_test

# ---------- quick plotting helper ----------

def _p_to_stars(p):
    if not np.isfinite(p): return "n/a"
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"

def _add_sig_bar(ax, x1, x2, y, text, h=0.02, lw=1.6, fs=14):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='k', lw=lw)
    ax.text((x1+x2)/2.0, y+h, text, ha='center', va='bottom', fontsize=fs)

def plot_state_boxplot(df_tidy, moving_label='moving', notmoving_label='notmoving',
                       title='Theta coherence by state',
                       fs_title=20, fs_axes=18, fs_tick=16, fs_star=14,
                       show_lines=True):
    pivot = df_tidy.pivot(index='recording', columns='state', values='coherence')
    pivot = pivot[[moving_label, notmoving_label]].dropna()

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)

    data = [pivot[moving_label].values, pivot[notmoving_label].values]
    ax.boxplot(data, positions=[1,2], widths=0.6, showfliers=False)

    # coloured paired points/lines
    colors = plt.cm.tab10(np.linspace(0,1,len(pivot)))
    for (c, (_, row)) in zip(colors, pivot.iterrows()):
        if show_lines:
            ax.plot([1,2], [row[moving_label], row[notmoving_label]],
                    marker='o', lw=2.0, alpha=0.9, color=c)
        else:
            ax.scatter([1,2], [row[moving_label], row[notmoving_label]],
                       s=50, color=c, edgecolor='k', linewidth=0.5, zorder=3)

    # labels
    ax.set_xticks([1,2])
    ax.set_xticklabels([moving_label, notmoving_label], fontsize=fs_tick)
    ax.set_ylabel('theta coherence (4–12 Hz)', fontsize=fs_axes)
    ax.tick_params(axis='y', labelsize=fs_tick)
    if title is None:
        # expects a single pair in df_tidy['pair']
        if 'pair' in df_tidy.columns and df_tidy['pair'].nunique() == 1:
            pair_txt = df_tidy['pair'].iloc[0]
            title = f'{pair_txt} theta coherence by state'
        else:
            title = 'Theta coherence by state'

    ax.set_title(title, pad=12, fontsize=fs_title)  # tighter title pad
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # stats
    if len(pivot) >= 3:
        W, p = wilcoxon(pivot[moving_label], pivot[notmoving_label], alternative='two-sided')
    else:
        W, p = np.nan, np.nan

    # Prism-style bar, closer to data
    ymax = np.nanmax(np.concatenate(data)) if all(len(d)>0 for d in data) else ax.get_ylim()[1]
    y0, y1 = ax.get_ylim()
    step = 0.04 * (y1 - y0)   # smaller vertical step
    ybar = ymax + step
    ax.set_ylim(top=ybar + 1.5*step)

    label = f"{_p_to_stars(p)} (p={p:.3g})" if np.isfinite(p) else "n/a"
    _add_sig_bar(ax, 1, 2, ybar, label, h=0.01*(y1-y0), fs=fs_star)

    return fig, ax, (W, p)

from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon

def p_to_stars(p):
    return ("ns" if p >= 0.05 else
            "*"  if p < 0.05 and p >= 0.01 else
            "**" if p < 0.01 and p >= 0.001 else
            "***" if p < 0.001 and p >= 1e-4 else
            "****")

def holm_bonferroni(pvals):
    """
    Holm–Bonferroni step-down correction.
    Returns list of adjusted p-values in the original order.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        k = m - rank
        adj_p = pvals[idx] * k
        running_max = max(running_max, adj_p)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()

def add_sig_bar(ax, x1, x2, y, text, h=0.02, lw=1.5, fs=14):
    """
    Draws a significance bar between x1 and x2 at height y with label 'text'.
    x positions are category indices (1-based to match the boxplot positions below).
    """
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='k', lw=lw)
    ax.text((x1+x2)/2.0, y+h, text, ha='center', va='bottom', fontsize=fs)

PAIR_ORDER_DEFAULT = [('CA1_R','CA1_L'),  ('CA1_R','CA3_L'),('CA1_L','CA3_L')]

def _coherence_pair_for_state(df, fs, chan_map, pair_labels, theta_band=(4,12),
                              movement_col='movement', state='moving'):
    label2col = {lab: col for col, lab in chan_map.items()}
    a, b = pair_labels
    if a not in label2col or b not in label2col:
        return np.nan
    col_a, col_b = label2col[a], label2col[b]
    if (col_a not in df.columns) or (col_b not in df.columns):
        return np.nan

    mask = _state_mask(df, movement_col=movement_col, states=state)
    xa = df[col_a].to_numpy(float)[mask]
    xb = df[col_b].to_numpy(float)[mask]
    if min(xa.size, xb.size) < max(256, int(fs*2)):
        return np.nan
    xa_th = _sos_bandpass(xa, fs, theta_band[0], theta_band[1])
    xb_th = _sos_bandpass(xb, fs, theta_band[0], theta_band[1])
    L = min(len(xa_th), len(xb_th))
    return _theta_coherence_pair(xa_th[:L], xb_th[:L], fs, theta_band)

def batch_theta_coherence_three_pairs(
        root_dir, recording_names, chan_map,
        theta_band=(4,12),
        movement_col='movement',
        state='moving',
        pair_order=PAIR_ORDER_DEFAULT,
        save_csv_path=None):
    """
    Returns a tidy DataFrame with columns: recording, pair, coherence, state.
    """
    from SyncOECPySessionClass import SyncOEpyPhotometrySession

    rows = []
    for rec in recording_names:
        sess = SyncOEpyPhotometrySession(root_dir, rec, IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas', indicator='GEVI')
        df = sess.Ephys_tracking_spad_aligned
        fs = sess.fs
        for pair in pair_order:
            val = _coherence_pair_for_state(df, fs, chan_map, pair, theta_band,
                                            movement_col, state)
            rows.append({'recording': rec,
                         'pair': f"{pair[0]}–{pair[1]}",
                         'coherence': val,
                         'state': state})
    out = pd.DataFrame(rows)
    if save_csv_path:
        out.to_csv(save_csv_path, index=False)
    return out

def three_pair_stats(df_tidy, pair_order=PAIR_ORDER_DEFAULT):
    """
    Friedman global test across the three pairs (within-recording),
    then pairwise Wilcoxon with Holm–Bonferroni correction.
    Returns: global_stat, global_p, pairwise_table (DataFrame with raw & adjusted p).
    """
    pairs_txt = [f"{a}–{b}" for a,b in pair_order]
    pivot = df_tidy.pivot(index='recording', columns='pair', values='coherence')
    pivot = pivot[pairs_txt]  # keep order
    pivot = pivot.dropna(how='any')      # only complete cases

    global_stat, global_p = np.nan, np.nan
    if len(pivot) >= 3:
        a, b, c = (pivot[pairs_txt[0]].values,
                   pivot[pairs_txt[1]].values,
                   pivot[pairs_txt[2]].values)
        global_stat, global_p = friedmanchisquare(a, b, c)
    # pairwise
    pairwise = []
    for (i, j) in combinations(range(3), 2):
        p1, p2 = pairs_txt[i], pairs_txt[j]
        x, y = pivot[p1].values, pivot[p2].values
        if len(x) >= 3:
            W, p = wilcoxon(x, y, alternative='two-sided')
        else:
            W, p = np.nan, np.nan
        pairwise.append({'pairA': p1, 'pairB': p2, 'W': W, 'p_raw': p})

    # Holm–Bonferroni
    pvals = [r['p_raw'] if np.isfinite(r['p_raw']) else 1.0 for r in pairwise]
    p_adj = holm_bonferroni(pvals) if len(pvals) > 0 else []
    for r, padj in zip(pairwise, p_adj):
        r['p_adj'] = padj
        r['stars'] = p_to_stars(padj)
    pairwise_df = pd.DataFrame(pairwise)
    return (global_stat, global_p), pairwise_df, pivot

def plot_three_pairs_boxstrip(df_tidy, pair_order=PAIR_ORDER_DEFAULT,
                              title='Theta coherence (4–12 Hz) by pair',
                              fs_title=20, fs_axes=18, fs_tick=16, fs_star=16):
    pairs_txt = [f"{a}–{b}" for a,b in pair_order]
    pivot = df_tidy.pivot(index='recording', columns='pair', values='coherence')
    pivot = pivot[pairs_txt].dropna(how='any')

    fig, ax = plt.subplots(figsize=(8,7), constrained_layout=True)

    data = [pivot[p].values for p in pairs_txt]
    positions = [1,2,3]
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)

    # --- jittered coloured dots (no lines) ---
    colors = plt.cm.tab10(np.linspace(0,1,len(pivot)))  # distinct colours per trial
    jitter = 0.12
    for idx, (_, row) in enumerate(pivot.iterrows()):
        xs = [positions[k] + np.random.uniform(-jitter, jitter) for k in range(3)]
        ax.scatter(xs, [row[p] for p in pairs_txt],
                   color=colors[idx], s=50, zorder=3, edgecolor='k', linewidth=0.5)

    # axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(pairs_txt, rotation=0, fontsize=fs_tick)
    ax.set_ylabel('theta coherence (4–12 Hz)', fontsize=fs_axes)
    ax.tick_params(axis='y', labelsize=fs_tick)
    ax.set_title(title, fontsize=fs_title, pad=24)

    # remove top/right frames
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # stats and sig bars
    (gstat, gp), pairwise_df, _ = three_pair_stats(df_tidy, pair_order=pair_order)
    if np.isfinite(gp):
        ax.set_title(f"{title}\nFriedman χ²={gstat:.2f}, p={gp:.3g}",
                     fontsize=fs_title, pad=24)

    ymax = np.nanmax(np.concatenate(data)) if len(data) else 1.0
    y = ymax + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    step = 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    loc = {pairs_txt[i]: positions[i] for i in range(3)}
    for k, r in enumerate(pairwise_df.itertuples(index=False)):
        x1 = loc[r.pairA]; x2 = loc[r.pairB]
        label = f"{r.stars} (p={r.p_adj:.3g})"
        add_sig_bar(ax, x1, x2, y + k*step, label,
                    h=0.01*(ax.get_ylim()[1]-ax.get_ylim()[0]), fs=fs_star)

    if len(data) and np.isfinite(ymax):
        ax.set_ylim(top=y + (len(pairwise_df)+2)*step)

    return fig, ax, (gstat, gp), pairwise_df

#%%
# dpath = r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day1and2DLC'
# recordings = ['SyncRecording1','SyncRecording2','SyncRecording3',
#               'SyncRecording4','SyncRecording5','SyncRecording6',
#               'SyncRecording7','SyncRecording8','SyncRecording9',
#               'SyncRecording10']

dpath = r'G:\2025_ATLAS_SPAD\MultiFibre\1887932_Jedi2p_Multi_ephysbad\MovingTrialsDLC'
recordings = ['SyncRecording1','SyncRecording2','SyncRecording3',
              'SyncRecording4','SyncRecording5','SyncRecording6',
              'SyncRecording7']

chan_map = {'ref_raw':'CA1_R','sig_raw':'CA1_L','zscore_raw':'CA3_L'}
theta_band=(4,12)
moving_label='moving'
notmoving_label='notmoving'

for pair in [('CA1_R','CA1_L'), ('CA1_R','CA3_L'), ('CA1_L','CA3_L')]:
    pair_txt = f'{pair[0]}–{pair[1]}'
    out_df_pair, (W, p), paired_table = batch_pair_theta_coherence_by_state(
        root_dir=dpath,
        recording_names=recordings,
        chan_map=chan_map,
        pair=pair,
        theta_band=theta_band,
        movement_col='movement',
        moving_label=moving_label,
        notmoving_label=notmoving_label,
        save_csv_path=os.path.join(dpath, f'{pair_txt.replace("–","-")}_theta_coh_by_state.csv')
    )

    print(f"\n{pair_txt} — Paired Wilcoxon (moving vs {notmoving_label}): W={W:.3f}, p={p:.3g}")
    print(paired_table.describe())

    fig, ax, (W, p) = plot_state_boxplot(
    out_df_pair,                      # tidy df for a given pair
    moving_label='moving',
    notmoving_label='notmoving',
    title=None,
    fs_title=18, fs_axes=18, fs_tick=18, fs_star=18,
    show_lines=True   # or False if you prefer dots only
)
#%%
# Choose a state to analyse; run once for 'moving' and again for 'notmoving' if you want two figures
state_to_analyse = 'moving'   # or 'notmoving'

# Compute tidy table across recordings for the three pairs
df_three = batch_theta_coherence_three_pairs(
    root_dir=dpath,
    recording_names=recordings,
    chan_map=chan_map,
    theta_band=(4,12),
    movement_col='movement',
    state=state_to_analyse,
    pair_order=PAIR_ORDER_DEFAULT,
    save_csv_path=os.path.join(dpath, f'theta_coherence_three_pairs_{state_to_analyse}.csv')
)
#%%
# Plot + stats
fig3, ax3, (gstat, gp), pairwise_df = plot_three_pairs_boxstrip(
    df_three[df_three['state']==state_to_analyse],
    pair_order=PAIR_ORDER_DEFAULT,
    title=f"Theta coherence by pair | state: {state_to_analyse}",
    fs_title=18, fs_axes=18, fs_tick=18, fs_star=18
)

print(f"[{state_to_analyse}] Friedman χ²={gstat:.3f}, p={gp:.3g}")
print("Pairwise Wilcoxon with Holm–Bonferroni correction:")
print(pairwise_df[['pairA','pairB','W','p_raw','p_adj','stars']])