# -*- coding: utf-8 -*-
"""
Adds pooled plotting for Awake-stationary and NonREM:
 - raster + histogram of ripple-aligned events
 - centre (±10 ms) vs flanks (total 20 ms)
 - pre (−20–0 ms) vs post (0–20 ms)
 - null distribution (if available) with observed RMI
"""

import os, glob, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# Configure your 6 animals
# =========================
NONREM_ROOTS = [
    r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepNonREM',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepNonREM',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepNonREM',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\ASleepNonREM',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1910567_Jedi2p_OF\ASleepNonREM',
    r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ASleepNonREM',
]

AWAKE_ROOTS = [
    r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1910567_Jedi2p_OF\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationary',
    
]

OUT_DIR = os.path.join(os.path.dirname(NONREM_ROOTS[0]), 'RMI_animal_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# --------- Search patterns / keys (ADAPT HERE IF YOUR KEYS DIFFER) ----------
CANDIDATE_EVENT_FILENAMES = [
    "*event*align*.pkl", "*align*event*.pkl",
    "*event*align*.npy", "*align*event*.npy",
    "*events_rel*.pkl", "*events_rel*.npy"
]
CANDIDATE_EVENT_KEYS = [
    "rel_event_times_s", "events_rel_s", "t_rel_by_epoch", "t_rel_list"
]
CANDIDATE_NULL_KEYS = [
    "null_RMI", "null_dist", "null_distribution", "nulls"
]

# ------------------------- Helpers (existing + new) -------------------------
def _find_session_pkls(state_root):
    direct = glob.glob(os.path.join(state_root, 'SyncRecording*',
                                    'RippleSave_*', 'session_ripple_RMI.pkl'))
    if direct:
        return sorted(direct)
    return sorted(glob.glob(os.path.join(state_root, 'SyncRecording*',
                                         '**', 'session_ripple_RMI.pkl'), recursive=True))

def _animal_id_from_path(path):
    m = re.search(r'(\d{7})', path)
    return m.group(1) if m else 'UNKNOWN'

def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _load_one_rmi_pkl(pkl_path):
    d = _load_pickle(pkl_path)
    pooled = float(d.get('pooled_RMI'))
    null = None
    for k in CANDIDATE_NULL_KEYS:
        if k in d and d[k] is not None:
            null = np.asarray(d[k]).ravel()
            break
    return pooled, null

def _collect_state_df(roots, state):
    rows = []
    for root in roots:
        for p in _find_session_pkls(root):
            try:
                rmi, null = _load_one_rmi_pkl(p)
            except Exception as e:
                print(f'! skipping {p}: {e}')
                continue
            rows.append({
                'animal': _animal_id_from_path(p),
                'state' : state,
                'path'  : p,
                'RMI'   : rmi,
                'null'  : null
            })
    return pd.DataFrame(rows)

def _mean_sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = np.mean(x) if x.size else np.nan
    s = np.std(x, ddof=1)/np.sqrt(x.size) if x.size > 1 else np.nan
    return float(m), float(s)

def _paired_perm_p(diff, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.mean(diff)
    signs = rng.choice([-1, 1], size=(n_perm, diff.size))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p), null

# ---------- NEW: event loader (list of 1D arrays per epoch) ----------
def _find_event_file(session_dir):
    for pat in CANDIDATE_EVENT_FILENAMES:
        hits = glob.glob(os.path.join(session_dir, pat))
        if hits:
            return hits[0]
    # also look one level up (some projects save next to pickle)
    parent = os.path.dirname(session_dir)
    for pat in CANDIDATE_EVENT_FILENAMES:
        hits = glob.glob(os.path.join(parent, pat))
        if hits:
            return hits[0]
    return None

def _extract_rel_times_list(obj):
    """Return [np.array([...]), ...] one per epoch, or None if not possible."""
    # numpy file could already be an object array of arrays
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [np.asarray(x, float).ravel() for x in obj.tolist()]
        # 2D array fallback -> treat rows as epochs
        if obj.ndim == 2:
            return [np.asarray(row, float)[~np.isnan(row)] for row in obj]
        # 1D array -> single epoch
        if obj.ndim == 1:
            return [np.asarray(obj, float)]
    # dict-like pickle with various key names
    if isinstance(obj, dict):
        for k in CANDIDATE_EVENT_KEYS:
            if k in obj:
                val = obj[k]
                if isinstance(val, list):
                    return [np.asarray(x, float).ravel() for x in val]
                if isinstance(val, np.ndarray) and val.dtype == object:
                    return [np.asarray(x, float).ravel() for x in val.tolist()]
                if isinstance(val, np.ndarray) and val.ndim == 2:
                    return [np.asarray(row, float)[~np.isnan(row)] for row in val]
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    return [np.asarray(val, float)]
    # pandas DataFrame with a column of lists/arrays
    try:
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            for k in CANDIDATE_EVENT_KEYS + ['t_rel', 't_rel_s']:
                if k in obj.columns:
                    return [np.asarray(x, float).ravel() for x in obj[k].tolist()]
    except Exception:
        pass
    return None

def _load_session_events(session_pkl_path):
    """Given a session_ripple_RMI.pkl path, try to load aligned events (s) per epoch."""
    sess_dir = os.path.dirname(session_pkl_path)
    ev_file = _find_event_file(sess_dir)
    if ev_file is None:
        warnings.warn(f"No aligned-event file found near {sess_dir}")
        return []
    try:
        if ev_file.endswith('.npy'):
            obj = np.load(ev_file, allow_pickle=True)
        else:
            obj = _load_pickle(ev_file)
        rel_list = _extract_rel_times_list(obj)
        if rel_list is None:
            warnings.warn(f"Could not parse event arrays from {ev_file}")
            return []
        return [np.asarray(x, float) for x in rel_list]
    except Exception as e:
        warnings.warn(f"Failed loading events from {ev_file}: {e}")
        return []

# ---------- NEW: pooled collectors ----------
def collect_pooled_events(df_state):
    """Return list-of-arrays (per epoch) of relative times (s) for the given state."""
    pooled = []
    for p in df_state['path'].tolist():
        pooled.extend(_load_session_events(p))
    return pooled  # list of length n_epochs, each np.array of event times (s)

def collect_pooled_nulls(df_state):
    """Concatenate any available null RMI arrays across sessions."""
    nulls = [x for x in df_state['null'].tolist() if x is not None]
    if not nulls:
        return None
    return np.concatenate(nulls)

# ------------------------- NEW: plotting -------------------------
def _figsave(fig, out_png):
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", out_png)

def plot_raster_and_hist(rel_times_by_epoch, out_png,
                         xlim_s=(-0.05, 0.05), bin_ms=2):
    if not rel_times_by_epoch:
        warnings.warn("No events to plot for raster/hist.")
        return
    fig = plt.figure(figsize=(8, 6), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.15)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0])

    # raster
    y = []
    x = []
    for i, arr in enumerate(rel_times_by_epoch):
        if arr.size == 0: 
            continue
        sel = arr[(arr >= xlim_s[0]) & (arr <= xlim_s[1])]
        x.append(sel)
        y.append(np.full(sel.size, i+1))
    if x:
        ax_r.plot(np.concatenate(x), np.concatenate(y), 'o', ms=2)
    ax_r.axvline(0, ls='--', lw=1, color='k')
    ax_r.set_xlim(*xlim_s)
    ax_r.set_ylabel('Epoch')
    ax_r.set_title('Ripple-aligned optical events (pooled)')

    # histogram
    bins = int((xlim_s[1]-xlim_s[0])/(bin_ms/1000.0))
    all_events = np.concatenate([arr[(arr>=xlim_s[0])&(arr<=xlim_s[1])] for arr in rel_times_by_epoch]) if rel_times_by_epoch else np.array([])
    ax_h.hist(all_events, bins=bins)
    ax_h.axvline(0, ls='--', lw=1, color='k')
    ax_h.set_xlim(*xlim_s)
    ax_h.set_xlabel('Time from ripple peak (s)')
    ax_h.set_ylabel('Count')

    _figsave(fig, out_png)

def _per_epoch_counts(rel_times_by_epoch, center_ms=10, half_win_ms=20):
    """Return per-epoch counts (centre, flanks, pre, post) within ±half_win."""
    c = center_ms/1000.0
    W = half_win_ms/1000.0
    per_epoch = []
    for arr in rel_times_by_epoch:
        if arr.size == 0:
            per_epoch.append((0,0,0,0))
            continue
        inwin = arr[(arr>=-W)&(arr<=W)]
        k_center = np.sum((inwin>=-c)&(inwin<=c))
        k_flank  = inwin.size - k_center
        k_pre    = np.sum((inwin>=-W)&(inwin<0))
        k_post   = np.sum((inwin>=0)&(inwin<=W))
        per_epoch.append((int(k_center), int(k_flank), int(k_pre), int(k_post)))
    return np.array(per_epoch, int)

def plot_centre_vs_flanks(rel_times_by_epoch, out_png, center_ms=10, half_win_ms=20):
    import numpy.random as npr
    if not rel_times_by_epoch:
        import warnings; warnings.warn("No events to plot for centre vs flanks."); return

    counts = _per_epoch_counts(rel_times_by_epoch, center_ms, half_win_ms)
    kc, kf = counts[:,0], counts[:,1]
    diffs = kc - kf

    # stats (one-sided: centre > flanks)
    from scipy.stats import wilcoxon, binomtest
    nz = diffs != 0
    p_wil = (wilcoxon(kc[nz], kf[nz], alternative='greater', zero_method='wilcox').pvalue
             if np.sum(nz) >= 1 else 1.0)
    # sign-flip permutation on paired diffs
    rng = npr.default_rng(0)
    null = np.sum(diffs * rng.choice([-1,1], size=(10000, diffs.size)), axis=1)
    p_perm = float(np.mean(null >= np.sum(diffs)))
    N = int(kc.sum() + kf.sum())
    p_bin = binomtest(int(kc.sum()), N, 0.5, alternative='greater').pvalue if N>0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
    axes[0].bar([0,1],[kc.sum(), kf.sum()], tick_label=['Centre','Flanks'])
    axes[0].set_title(f'Pooled counts (±{half_win_ms} ms; centre ±{center_ms} ms)')
    axes[0].set_ylabel('Events')

    parts = axes[1].violinplot([kc, kf], showmeans=False, showextrema=False)
    for pc in parts['bodies']: pc.set_alpha(0.5)
    x1, x2 = 1, 2
    axes[1].scatter(np.full_like(kc, x1), kc, s=10); axes[1].scatter(np.full_like(kf, x2), kf, s=10)
    for a, b in zip(kc, kf): axes[1].plot([x1, x2], [a, b], color='0.7', lw=0.8)
    axes[1].set_xticks([1,2]); axes[1].set_xticklabels(['Centre','Flanks'])
    axes[1].set_title(f'Per-epoch paired counts\nWilcoxon p={p_wil:.3g}; Perm p={p_perm:.3g}; Binom p={p_bin:.3g}')

    _figsave(fig, out_png)


def plot_pre_vs_post(rel_times_by_epoch, out_png, half_win_ms=20):
    import numpy.random as npr
    if not rel_times_by_epoch:
        import warnings; warnings.warn("No events to plot for pre vs post."); return

    counts = _per_epoch_counts(rel_times_by_epoch, center_ms=10, half_win_ms=half_win_ms)
    kpre, kpost = counts[:,2], counts[:,3]
    diffs = kpre - kpost

    from scipy.stats import wilcoxon, binomtest
    nz = diffs != 0
    p_wil = (wilcoxon(kpre[nz], kpost[nz], alternative='two-sided', zero_method='wilcox').pvalue
             if np.sum(nz) >= 1 else 1.0)

    rng = npr.default_rng(0)
    null = np.sum(diffs * rng.choice([-1,1], size=(10000, diffs.size)), axis=1)
    p_perm = float(np.mean(np.abs(null) >= abs(np.sum(diffs))))

    N = int(kpre.sum() + kpost.sum())
    p_bin = binomtest(int(kpre.sum()), N, 0.5).pvalue if N>0 else 1.0

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    parts = ax.violinplot([kpre, kpost], showmeans=False, showextrema=False)
    for pc in parts['bodies']: pc.set_alpha(0.5)
    x1, x2 = 1, 2
    ax.scatter(np.full_like(kpre, x1), kpre, s=10); ax.scatter(np.full_like(kpost, x2), kpost, s=10)
    for a, b in zip(kpre, kpost): ax.plot([x1, x2], [a, b], color='0.7', lw=0.8)
    ax.set_xticks([1,2]); ax.set_xticklabels([f'Pre (−{half_win_ms}–0 ms)', f'Post (0–+{half_win_ms} ms)'])
    ax.set_ylabel('Events per epoch')
    ax.set_title(f'Pre vs Post — per-epoch counts\nWilcoxon p={p_wil:.3g}; Perm p={p_perm:.3g}; Binom p={p_bin:.3g}')
    _figsave(fig, out_png)


def plot_null_distribution(null_values, observed, out_png, state_label,
                          n_events=None, n_epochs=None, p_perm_one_sided=None):
    if null_values is None or null_values.size == 0:
        import warnings; warnings.warn(f"No null distribution for {state_label}."); return

    # if not provided, compute one-sided p = Pr(null >= observed)
    if p_perm_one_sided is None:
        p_perm_one_sided = float((np.sum(null_values >= observed) + 1) / (null_values.size + 1))

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=150)
    ax.hist(null_values, bins=40)
    ax.axvline(observed, color='r', lw=3, label=f'Observed RMI = {observed:.3f}')
    ax.set_xlabel('RMI (null)',fontsize=16); ax.set_ylabel('Frequency',fontsize=16)
    ax.set_title(f'Null distribution of RMI — {state_label}')
    ax.legend(frameon=False)

    # annotation (events/epochs + p)
    lines = [f"perm p (≥obs) = {p_perm_one_sided:.3g}"]
    # if n_events is not None: lines.append(f"events = {int(n_events)}")
    # if n_epochs is not None: lines.append(f"epochs = {int(n_epochs)}")
    ax.text(0.98, 0.02, "\n".join(lines), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=18,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='0.7'))

    _figsave(fig, out_png)


# ------------------------- Existing plot on animal means ---------------------
def plot_paired_animal_means(df_animal_means, out_png):
    A = df_animal_means['Awake-stationary'].values
    N = df_animal_means['NonREM'].values
    animals = df_animal_means['animal'].astype(str).values
    n = len(df_animal_means)

    # deterministic colour per animal
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap('tab10')
    colour_map = {a: cmap(i % 10) for i, a in enumerate(sorted(animals))}
    colours = [colour_map[a] for a in animals]

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    xs = [1, 2]

    for i in range(n):
        ax.plot(xs, [A[i], N[i]], '-', color='0.80', lw=1.5, zorder=1)
        ax.plot(xs[0], A[i], 'o', color=colours[i], ms=7, zorder=2)
        ax.plot(xs[1], N[i], 'o', color=colours[i], ms=7, zorder=2)

    mA, sA = _mean_sem(A); mN, sN = _mean_sem(N)
    for x, m, s in zip(xs, [mA, mN], [sA, sN]):
        ax.hlines(m, x-0.20, x+0.20, colors='k', lw=2.5, zorder=3)
        ax.vlines(x, m-s, m+s, colors='k', lw=2.5, zorder=3)
        ax.hlines([m-s, m+s], x-0.10, x+0.10, colors='k', lw=2.5, zorder=3)

    diff = N - A
    wl_stat, p_wil = stats.wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
    p_perm, _ = _paired_perm_p(diff, n_perm=10000, seed=0)
    dz = np.mean(diff) / (np.std(diff, ddof=1) if len(diff) > 1 else np.nan)

    ax.set_xticks(xs); ax.set_xticklabels(['Awake-stationary', 'NonREM'], fontsize=14)
    ax.set_ylabel('Ripple Modulation Index (RMI)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14); ax.axhline(0, color='0.7', lw=1)
    ax.set_title('Animal-averaged RMI by state (paired)', fontsize=16, pad=10)

    txt = (f"n={n} animals\n"
           f"Mean±SEM: Awake={mA:.3f}±{sA:.3f},  NonREM={mN:.3f}±{sN:.3f}\n"
           f"Δmean (NonREM−Awake) = {np.mean(diff):.3f}\n"
           f"Wilcoxon p={p_wil:.4f}; permutation p={p_perm:.4f}; dz={dz:.2f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none'))

    _figsave(fig, out_png)
    return {'n_animals': int(n), 'Awake_mean': float(mA), 'Awake_sem': float(sA),
            'NonREM_mean': float(mN), 'NonREM_sem': float(sN),
            'wilcoxon_p': float(p_wil), 'perm_p': float(p_perm), 'cohens_dz': float(dz)}


# ------------------------- Main -------------------------
def main():
    # session tables
    df_awake  = _collect_state_df(AWAKE_ROOTS,  'Awake-stationary')
    df_nonrem = _collect_state_df(NONREM_ROOTS, 'NonREM')

    if df_awake.empty or df_nonrem.empty:
        print("No session_ripple_RMI.pkl found in one or both state folders.")
        return

    # per-animal means within each state
    gA = df_awake.groupby('animal')['RMI'].mean().rename('Awake-stationary')
    gN = df_nonrem.groupby('animal')['RMI'].mean().rename('NonREM')

    # animals that have BOTH states
    df_animals = pd.concat([gA, gN], axis=1).dropna().reset_index()
    df_animals = df_animals.sort_values('animal')

    # save tables
    df_awake.to_csv(os.path.join(OUT_DIR, 'sessions_Awake.csv'), index=False)
    df_nonrem.to_csv(os.path.join(OUT_DIR, 'sessions_NonREM.csv'), index=False)
    df_animals.to_csv(os.path.join(OUT_DIR, 'animal_means_Awake_vs_NonREM.csv'), index=False)

    # plot paired scatter on animal means
    stats_out = plot_paired_animal_means(
        df_animals,
        out_png=os.path.join(OUT_DIR, 'RMI_animal_means_paired.png')
    )

    with open(os.path.join(OUT_DIR, 'RMI_animal_compare_stats.txt'), 'w') as f:
        f.write(df_animals.to_string(index=False) + "\n\n")
        for k, v in stats_out.items():
            f.write(f"{k}: {v}\n")

    # -------- NEW: pooled plots for each state --------
    for state_name, df_state in [('Awake-stationary', df_awake), ('NonREM', df_nonrem)]:
        state_slug = 'Awake' if state_name.startswith('Awake') else 'NonREM'
        out_state = os.path.join(OUT_DIR, f'POOLED_{state_slug}')
        os.makedirs(out_state, exist_ok=True)

        # events pooled across sessions (list of arrays, one per epoch)
        rel_times = collect_pooled_events(df_state)
        # observed RMI pooled across sessions (mean of session pooled_RMI)
        obs_rmi = float(np.nanmean(df_state['RMI'].values))
        # null distribution concatenated if available
        nulls = collect_pooled_nulls(df_state)

        # raster + histogram
        plot_raster_and_hist(rel_times,
            out_png=os.path.join(out_state, f'{state_slug}_raster_hist.png'),
            xlim_s=(-0.05, 0.05), bin_ms=2)

        # centre vs flanks (centre ±10 ms within ±20 ms)
        plot_centre_vs_flanks(rel_times,
            out_png=os.path.join(out_state, f'{state_slug}_centre_vs_flanks.png'),
            center_ms=10, half_win_ms=20)

        # pre vs post (−20–0 vs 0–20 ms)
        plot_pre_vs_post(rel_times,
            out_png=os.path.join(out_state, f'{state_slug}_pre_vs_post.png'),
            half_win_ms=20)

        # totals for annotation
        n_epochs = len(rel_times)                          # epochs with aligned data
        n_events = sum(len(a) for a in rel_times)          # pooled events
        
        # null p (one-sided)
        p_null = None if nulls is None else float((np.sum(nulls >= obs_rmi) + 1) / (nulls.size + 1))
        
        plot_null_distribution(
            nulls, obs_rmi,
            out_png=os.path.join(out_state, f'{state_slug}_RMI_null_vs_observed.png'),
            state_label=state_name,
            n_events=n_events, n_epochs=n_epochs,
            p_perm_one_sided=p_null
        )

    print(f"Animals included (both states): {list(df_animals['animal'])}")
    print("Outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
