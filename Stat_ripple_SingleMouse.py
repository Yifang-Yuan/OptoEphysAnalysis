import os, glob, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- set your two state roots (NOT the RipplePooled folders) ---
AWAKE_ROOT  = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary'
NONREM_ROOT = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepNonREM'

OUT_DIR = os.path.join(os.path.dirname(AWAKE_ROOT), 'RMI_session_compare')
os.makedirs(OUT_DIR, exist_ok=True)

def _find_session_pkls(state_root):
    """
    Look *inside each SyncRecording* for RippleSave_* / session_ripple_RMI.pkl.
    Falls back to recursive search if the direct pattern finds none.
    """
    direct = glob.glob(os.path.join(state_root, 'SyncRecording*',
                                    'RippleSave_*', 'session_ripple_RMI.pkl'))
    if direct:
        return sorted(direct)

    # fallback: recursive search (in case your save dir is nested differently)
    rec = glob.glob(os.path.join(state_root, 'SyncRecording*',
                                 '**', 'session_ripple_RMI.pkl'), recursive=True)
    return sorted(rec)

def _load_rmis(pkl_paths):
    rows = []
    for p in pkl_paths:
        try:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            rmi = float(d.get('pooled_RMI'))
            rec = d.get('recording', os.path.basename(os.path.dirname(os.path.dirname(p))))
            rows.append({'recording': rec, 'path': p, 'RMI': rmi})
        except Exception as e:
            print(f'! skip {p}: {e}')
    return pd.DataFrame(rows)

def _mean_sem(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.mean(a)), float(np.std(a, ddof=1)/np.sqrt(max(1, len(a))))

def _cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / max(1, na+nb-2))
    if sp == 0: return 0.0
    J = 1 - (3/(4*(na+nb)-9)) if (na+nb) > 2 else 1.0  # Hedges’ correction
    return float(((np.mean(a)-np.mean(b))/sp) * J)

def _cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = sum(x>y for x in a for y in b); lt = sum(x<y for x in a for y in b)
    return float((gt - lt) / (len(a)*len(b)))

def _perm_label_test(a, b, n_perm=10000, seed=0, which='mean'):
    """Within-animal session-label permutation test (two-sided)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    pool = np.concatenate([a, b])
    na, nb = len(a), len(b)
    stat = (np.mean(a)-np.mean(b)) if which=='mean' else (np.median(a)-np.median(b))
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pool)
        aa = pool[:na]; bb = pool[na:na+nb]
        null[i] = (np.mean(aa)-np.mean(bb)) if which=='mean' else (np.median(aa)-np.median(bb))
    p = (np.sum(np.abs(null) >= abs(stat)) + 1) / (n_perm + 1)
    return float(stat), float(p)

def plot_session_scatter(a_vals, n_vals, out_png):
    plt.figure(figsize=(6.6, 5.0), dpi=150)
    ax = plt.gca()
    labels = ['Awake-stationary', 'NonREM']; colors = ['#1a9641', '#2c7fb8']; xs = [1,2]

    for i, (vals, col) in enumerate(zip([a_vals, n_vals], colors)):
        xj = np.random.normal(xs[i], 0.06, size=len(vals))
        ax.plot(xj, vals, 'o', ms=6, alpha=0.85, color=col, label=f"{labels[i]} (n={len(vals)})")
        m, s = _mean_sem(vals)
        ax.hlines(m, xs[i]-0.18, xs[i]+0.18, colors='k', lw=2)
        ax.vlines(xs[i], m-s, m+s, colors='k', lw=2)
        ax.hlines([m-s, m+s], xs[i]-0.08, xs[i]+0.08, colors='k', lw=2)

    # stats (session-level, within-animal label permutation)
    try:
        U, p_mwu = stats.mannwhitneyu(a_vals, n_vals, alternative='two-sided')
    except ValueError:
        U, p_mwu = np.nan, 1.0
    dm, p_perm_m   = _perm_label_test(a_vals, n_vals, which='mean')
    dmed, p_perm_s = _perm_label_test(a_vals, n_vals, which='median')
    d = _cohens_d(a_vals, n_vals); delt = _cliffs_delta(a_vals, n_vals)

    # simple title + stats text inside axes (same fontsize)
    title_fs = 14
    ax.set_title("Ripple Modulation Index sessions by state", fontsize=title_fs, pad=10)

    txt = (f"Δmean={dm:.3f} (perm p={p_perm_m:.4f});\n"
           f"Δmedian={dmed:.3f} (perm p={p_perm_s:.4f});\n "
           f"Mann–Whitney p={p_mwu:.4f}; d={d:.2f}, Δ={delt:.2f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=title_fs,
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='none',   # <-- note the string 'none'
                  edgecolor='none',   # or set a color if you want an outline
                  lw=0))

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylabel("Ripple Modulation Index (RMI)", fontsize=16)
    ax.axhline(0, color='0.7', lw=1)
    ax.legend(frameon=True, fontsize=14, loc='upper left')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return {'mwu_p': float(p_mwu), 'perm_p_mean': float(p_perm_m),
            'perm_p_median': float(p_perm_s), 'delta_mean': float(dm),
            'delta_median': float(dmed), 'cohens_d': float(d), 'cliffs_delta': float(delt)}

    return {'mwu_p': float(p_mwu), 'perm_p_mean': float(p_perm_m),
            'perm_p_median': float(p_perm_s), 'delta_mean': float(dm),
            'delta_median': float(dmed), 'cohens_d': float(d), 'cliffs_delta': float(delt)}

def main():
    awake_pkls  = _find_session_pkls(AWAKE_ROOT)
    nonrem_pkls = _find_session_pkls(NONREM_ROOT)
    print(f"Found {len(awake_pkls)} Awake session pickles")
    print(f"Found {len(nonrem_pkls)} NonREM session pickles")

    df_a = _load_rmis(awake_pkls);  df_a['state'] = 'Awake-stationary'
    df_n = _load_rmis(nonrem_pkls); df_n['state'] = 'NonREM'
    if df_a.empty or df_n.empty:
        print("No session_ripple_RMI.pkl found in one/both roots.")
        return

    # save raw tables
    df_a.to_csv(os.path.join(OUT_DIR, 'Awake_sessions_RMI.csv'), index=False)
    df_n.to_csv(os.path.join(OUT_DIR, 'NonREM_sessions_RMI.csv'), index=False)

    # scatter + mean±SEM
    out_png = os.path.join(OUT_DIR, 'RMI_sessions_scatter_meanSEM.png')
    stats_out = plot_session_scatter(df_a['RMI'].values, df_n['RMI'].values, out_png)

    df_all = pd.concat([df_a, df_n], ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, 'RMI_sessions_combined.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'RMI_session_compare_stats.txt'), 'w') as f:
        f.write(f"Awake sessions: n={len(df_a)}, mean={df_a['RMI'].mean():.3f}, SEM={df_a['RMI'].sem():.3f}\n")
        f.write(f"NonREM sessions: n={len(df_n)}, mean={df_n['RMI'].mean():.3f}, SEM={df_n['RMI'].sem():.3f}\n")
        for k,v in stats_out.items():
            f.write(f"{k}: {v}\n")

    print("Saved outputs to:", OUT_DIR)

if __name__ == "__main__":
    main()
