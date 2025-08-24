# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:51:59 2025

@author: yifan
"""

import os, glob, re, pickle
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
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\PyramidalWT\1910567_Jedi2p_OF\AwakeStationary',
    r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationary',
    r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary',
]

OUT_DIR = os.path.join(os.path.dirname(NONREM_ROOTS[0]), 'RMI_animal_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def _find_session_pkls(state_root):
    """Look inside each SyncRecording for the session RMI pickle."""
    direct = glob.glob(os.path.join(state_root, 'SyncRecording*',
                                    'RippleSave_*', 'session_ripple_RMI.pkl'))
    if direct:
        return sorted(direct)
    return sorted(glob.glob(os.path.join(state_root, 'SyncRecording*',
                                         '**', 'session_ripple_RMI.pkl'), recursive=True))

def _animal_id_from_path(path):
    m = re.search(r'(\d{7})', path)
    return m.group(1) if m else 'UNKNOWN'

def _load_one_rmi(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    return float(d.get('pooled_RMI'))

def _collect_state_df(roots, state):
    rows = []
    for root in roots:
        for p in _find_session_pkls(root):
            try:
                rmi = _load_one_rmi(p)
            except Exception as e:
                print(f'! skipping {p}: {e}')
                continue
            rows.append({
                'animal': _animal_id_from_path(p),
                'state' : state,
                'path'  : p,
                'RMI'   : rmi
            })
    return pd.DataFrame(rows)

def _mean_sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = np.mean(x) if x.size else np.nan
    s = np.std(x, ddof=1)/np.sqrt(x.size) if x.size > 1 else np.nan
    return float(m), float(s)

def _paired_perm_p(diff, n_perm=10000, seed=0):
    """Sign-flip permutation on paired differences (two-sided, T = mean(diff))."""
    rng = np.random.default_rng(seed)
    obs = np.mean(diff)
    signs = rng.choice([-1, 1], size=(n_perm, diff.size))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p), null

# -------------------------
# Plot paired animal means
# -------------------------
def plot_paired_animal_means(df_animal_means, out_png):
    """
    df_animal_means: columns ['animal','Awake-stationary','NonREM']
    """
    A = df_animal_means['Awake-stationary'].values
    N = df_animal_means['NonREM'].values
    animals = df_animal_means['animal'].values
    n = len(df_animal_means)

    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    xs = [1, 2]
    colors = ['#1a9641', '#2c7fb8']

    # one paired line per animal
    for i in range(n):
        ax.plot(xs, [A[i], N[i]], '-', color='0.80', lw=1.5, zorder=1)
        ax.plot(xs[0], A[i], 'o', color=colors[0], ms=7, zorder=2)
        ax.plot(xs[1], N[i], 'o', color=colors[1], ms=7, zorder=2)

    # group mean ± SEM bars
    mA, sA = _mean_sem(A); mN, sN = _mean_sem(N)
    for x, m, s in zip(xs, [mA, mN], [sA, sN]):
        ax.hlines(m, x-0.20, x+0.20, colors='k', lw=2.5, zorder=3)
        ax.vlines(x, m-s, m+s, colors='k', lw=2.5, zorder=3)
        ax.hlines([m-s, m+s], x-0.10, x+0.10, colors='k', lw=2.5, zorder=3)

    # paired stats on animal means
    diff = N - A
    wl_stat, p_wil = stats.wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
    p_perm, _ = _paired_perm_p(diff, n_perm=10000, seed=0)
    dz = np.mean(diff) / (np.std(diff, ddof=1) if len(diff) > 1 else np.nan)

    ax.set_xticks(xs)
    ax.set_xticklabels(['Awake-stationary', 'NonREM'], fontsize=14)
    ax.set_ylabel('Ripple Modulation Index (RMI)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.axhline(0, color='0.7', lw=1)

    ax.set_title('Animal-averaged RMI by state (paired)', fontsize=16, pad=10)
    txt = (f"n={n} animals\n"
           f"Mean±SEM:Awake={mA:.3f}±{sA:.3f},  NonREM={mN:.3f}±{sN:.3f}\n"
           f"Δmean (NonREM−Awake) = {np.mean(diff):.3f}\n"
           f"Wilcoxon p={p_wil:.4f}; permutation p={p_perm:.4f}; dz={dz:.2f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none'))

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print (txt)

    return {
        'n_animals': int(n),
        'Awake_mean': float(mA), 'Awake_sem': float(sA),
        'NonREM_mean': float(mN), 'NonREM_sem': float(sN),
        'wilcoxon_p': float(p_wil), 'perm_p': float(p_perm), 'cohens_dz': float(dz)
    }

# -------------------------
# Main
# -------------------------
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

    # also dump stats to text
    with open(os.path.join(OUT_DIR, 'RMI_animal_compare_stats.txt'), 'w') as f:
        f.write(df_animals.to_string(index=False) + "\n\n")
        for k, v in stats_out.items():
            f.write(f"{k}: {v}\n")

    print(f"Animals included (both states): {list(df_animals['animal'])}")
    print("Outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
