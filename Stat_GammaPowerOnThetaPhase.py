# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:21:51 2025

@author: yifan
"""

"""
Gamma power on theta phase: statistics & plots across 3 groups
Creates four separate figures:
  1) LFP preferred phase (deg, wrapped)
  2) LFP modulation depth (R)
  3) Optical preferred phase (deg, wrapped)
  4) Optical modulation depth (R)

Author: yifan (script assembled by ChatGPT)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle
from scipy.stats import kruskal, mannwhitneyu

# ======================== USER PATHS =========================
# Point each to the folder containing GammaPowerOnTheta_trial*.pkl.gz
LOCOMOTION_DIR = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\OpenField_DLCtracking\GammaPowerOnTheta")
AWAKE_DIR      = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\GammaPowerOnTheta")
REM_DIR        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleep\GammaPowerOnTheta")

OUT_DIR        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\GammaPhase_Group_Stats")  # where to save the four PNGs
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN        = "GammaPowerOnTheta_trial*.pkl.gz"
N_PERM         = 5000     # permutations for circular tests
ALPHA          = 0.05
# ============================================================


# ----------------- IO helpers -----------------
def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_gamma_group(folder: Path, pattern=PATTERN):
    """
    Read all pickles in a folder and extract:
      - lfp preferred phase (rad), lfp R
      - opt preferred phase (rad), opt R
    """
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise ValueError(f"No files matching {pattern} in {folder}")

    lfp_phi = []
    lfp_R   = []
    opt_phi = []
    opt_R   = []
    for p in paths:
        rec = _load_pickle(p)
        lfp_phi.append(float(rec["lfp"]["preferred_phase_rad"]))
        lfp_R.append(float(rec["lfp"]["R"]))
        opt_phi.append(float(rec["opt"]["preferred_phase_rad"]))
        opt_R.append(float(rec["opt"]["R"]))
    return {
        "lfp_phi": np.array(lfp_phi) % (2*np.pi),
        "lfp_R"  : np.array(lfp_R, dtype=float),
        "opt_phi": np.array(opt_phi) % (2*np.pi),
        "opt_R"  : np.array(opt_R, dtype=float),
        "n"      : len(paths),
        "paths"  : paths
    }


# -------------- Circular helpers --------------
def circ_mean(alpha):
    """Mean direction (rad) + resultant length of the mean (Rbar)."""
    C = np.nanmean(np.cos(alpha))
    S = np.nanmean(np.sin(alpha))
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rbar = np.sqrt(C**2 + S**2)
    return mu, Rbar

def circ_dist(a, b):
    """Smallest signed circular difference a-b in (-pi, pi]."""
    return (a - b + np.pi) % (2*np.pi) - np.pi

def perm_test_circ_means(groups_phi, n_perm=5000, rng=None):
    """
    Global multi-group permutation test for equality of mean directions.
    groups_phi: list of 1D arrays of radians
    """
    if rng is None: rng = np.random.default_rng()
    mus = np.array([circ_mean(g)[0] for g in groups_phi])
    ux, uy = np.cos(mus), np.sin(mus)
    obs = 0.0
    for i in range(len(mus)):
        for j in range(i+1, len(mus)):
            obs += (ux[i]-ux[j])**2 + (uy[i]-uy[j])**2

    pooled = np.concatenate(groups_phi)
    sizes  = [len(g) for g in groups_phi]
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        idx = 0
        mus_p = []
        for sz in sizes:
            ph = pooled[idx:idx+sz]; idx += sz
            mus_p.append(circ_mean(ph)[0])
        mus_p = np.array(mus_p)
        ux_p, uy_p = np.cos(mus_p), np.sin(mus_p)
        stat = 0.0
        for i in range(len(mus_p)):
            for j in range(i+1, len(mus_p)):
                stat += (ux_p[i]-ux_p[j])**2 + (uy_p[i]-uy_p[j])**2
        if stat >= obs - 1e-15:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)

def perm_test_circ_means_pair(a, b, n_perm=5000, rng=None):
    """
    Pairwise permutation test on |mean-direction difference|.
    Returns p-value and observed difference (deg).
    """
    if rng is None: rng = np.random.default_rng()
    mu_a, _ = circ_mean(a); mu_b, _ = circ_mean(b)
    obs = np.abs(circ_dist(mu_a, mu_b))
    pooled = np.concatenate([a, b])
    na = len(a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        aa = pooled[:na]; bb = pooled[na:]
        d = np.abs(circ_dist(circ_mean(aa)[0], circ_mean(bb)[0]))
        if d >= obs - 1e-15: count_ge += 1
    return (count_ge + 1) / (n_perm + 1), np.degrees(obs)

def holm_bonferroni(pvals, alpha=0.05):
    """Return adjusted p-values (Holm) in original order + significance decisions."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m, float)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running_max)
    sig = adj < alpha
    return adj, sig


# -------------- Plot helpers --------------
def jitter(n, width=0.1, rng=None):
    if rng is None: rng = np.random.default_rng()
    return rng.uniform(-width, width, size=n)

def smart_ylim(data_list, pad_frac=0.06):
    all_data = np.concatenate([np.asarray(d) for d in data_list])
    dmin, dmax = np.nanmin(all_data), np.nanmax(all_data)
    if np.isclose(dmin, dmax):
        return dmin - 1, dmax + 1
    pad = (dmax - dmin) * pad_frac
    return dmin - pad, dmax + pad

def style_ax(ax, fontsize=16):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize + 2)

def p_to_asterisks(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

def add_all_pairwise_bars(ax, positions, data_list, pairs, pvals,
                          text_fs=16, line_h_frac=0.06, pad_top_frac=0.04, gap_mult=2):
    """
    Draw Prism-style bars and extend ylim as needed.
    """
    y0, y1 = ax.get_ylim()
    data_max = max(np.nanmax(d) for d in data_list)
    span = (y1 - y0) if (y1 > y0) else 1.0
    h = span * line_h_frac*0.5
    gap = span * line_h_frac * gap_mult
    base = max(y1, data_max) + span * pad_top_frac

    for i, ((g1, g2), p) in enumerate(zip(pairs, pvals)):
        x1, x2 = positions[g1], positions[g2]
        y = base + i * gap
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.7, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, y + h, p_to_asterisks(p),
                ha='center', va='bottom', fontsize=text_fs, clip_on=False)

    ax.set_ylim(y0, base + len(pairs) * gap + h + span * 0.02)

def wrap_angles_for_plot(phi_rad, ref_rad=0.0, deg_range=(-180.0, 180.0)):
    """Rotate by ref_rad and wrap to [low, high] degrees for plotting only."""
    low, high = deg_range
    width = high - low
    shifted = (phi_rad - ref_rad) % (2*np.pi)
    deg = np.degrees(shifted)
    return ((deg - low) % width) + low


# ----------------- Load three groups -----------------
G_loc = load_gamma_group(LOCOMOTION_DIR)
G_aw  = load_gamma_group(AWAKE_DIR)
G_rem = load_gamma_group(REM_DIR)

# Order: Locomotion (left), Awake-stationary, REM
labels = ["Locomotion", "Awake-stationary", "REM"]
positions = [1, 2, 3]

# ----------- LFP preferred phase (circular) -----------
groups_phi = [G_loc["lfp_phi"], G_aw["lfp_phi"], G_rem["lfp_phi"]]
p_global = perm_test_circ_means(groups_phi, n_perm=N_PERM)

# Pairwise circular permutation tests on raw radians
pairs = [(0, 1), (0, 2), (1, 2)]
p_pair = []
for i, j in pairs:
    p, _ = perm_test_circ_means_pair(groups_phi[i], groups_phi[j], n_perm=N_PERM)
    p_pair.append(p)
p_adj, _sig = holm_bonferroni(p_pair, alpha=ALPHA)  # not shown on bars, but you can print if needed

# Wrapped for plotting around grand mean
grand_mu, _ = circ_mean(np.concatenate(groups_phi))
deg_plot = [
    wrap_angles_for_plot(G_loc["lfp_phi"], ref_rad=grand_mu),
    wrap_angles_for_plot(G_aw["lfp_phi"],  ref_rad=grand_mu),
    wrap_angles_for_plot(G_rem["lfp_phi"], ref_rad=grand_mu),
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(deg_plot, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(deg_plot, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.85)

ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("deg, wrapped", fontsize=16)
ax.set_title(f"LFP:γ preferred θ phase (global p={p_global:.3g})", fontsize=18)
ax.set_ylim(*smart_ylim(deg_plot, pad_frac=0.08))
style_ax(ax, fontsize=16)

add_all_pairwise_bars(ax, positions, deg_plot, pairs, p_pair,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04, gap_mult=2)

plt.tight_layout()
fig.savefig(OUT_DIR / "LFP_preferred_phase.png", dpi=300)

# ----------- LFP modulation depth R (linear) -----------
data_R = [G_loc["lfp_R"], G_aw["lfp_R"], G_rem["lfp_R"]]
kw_stat, kw_p = kruskal(*data_R)

# Pairwise Mann–Whitney
p_R = []
for i, j in pairs:
    _, p = mannwhitneyu(data_R[i], data_R[j], alternative="two-sided")
    p_R.append(p)
p_R_adj, _ = holm_bonferroni(p_R, alpha=ALPHA)

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(data_R, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(data_R, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.85)

ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("Modulation Depth (R)", fontsize=16)
ax.set_title(f"LFP:θ-γ modulation depth R (Kruskal–Wallis p={kw_p:.3g})", fontsize=18)
ax.set_ylim(*smart_ylim(data_R, pad_frac=0.08))
style_ax(ax, fontsize=16)

add_all_pairwise_bars(ax, positions, data_R, pairs, p_R,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04, gap_mult=2)

plt.tight_layout()
fig.savefig(OUT_DIR / "LFP_R.png", dpi=300)

# ----------- Optical preferred phase (circular) -----------
groups_phi_opt = [G_loc["opt_phi"], G_aw["opt_phi"], G_rem["opt_phi"]]
p_global_opt = perm_test_circ_means(groups_phi_opt, n_perm=N_PERM)

p_pair_opt = []
for i, j in pairs:
    p, _ = perm_test_circ_means_pair(groups_phi_opt[i], groups_phi_opt[j], n_perm=N_PERM)
    p_pair_opt.append(p)

grand_mu_opt, _ = circ_mean(np.concatenate(groups_phi_opt))
deg_plot_opt = [
    wrap_angles_for_plot(G_loc["opt_phi"], ref_rad=grand_mu_opt),
    wrap_angles_for_plot(G_aw["opt_phi"],  ref_rad=grand_mu_opt),
    wrap_angles_for_plot(G_rem["opt_phi"], ref_rad=grand_mu_opt),
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(deg_plot_opt, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(deg_plot_opt, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.85)

ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("deg, wrapped", fontsize=16)
ax.set_title(f"Optical γ preferred LFP θ phase (global p={p_global_opt:.3g})", fontsize=18)
ax.set_ylim(*smart_ylim(deg_plot_opt, pad_frac=0.08))
style_ax(ax, fontsize=16)

add_all_pairwise_bars(ax, positions, deg_plot_opt, pairs, p_pair_opt,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04, gap_mult=1.6)

plt.tight_layout()
fig.savefig(OUT_DIR / "Opt_preferred_phase.png", dpi=300)

# ----------- Optical modulation depth R (linear) -----------
data_R_opt = [G_loc["opt_R"], G_aw["opt_R"], G_rem["opt_R"]]
kw_stat_opt, kw_p_opt = kruskal(*data_R_opt)

p_R_opt = []
for i, j in pairs:
    _, p = mannwhitneyu(data_R_opt[i], data_R_opt[j], alternative="two-sided")
    p_R_opt.append(p)

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(data_R_opt, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(data_R_opt, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.85)

ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("Modulation depth (R)", fontsize=16)
ax.set_title(f"θ-Optical γ modulation depth R (p={kw_p_opt:.3g})", fontsize=18)
ax.set_ylim(*smart_ylim(data_R_opt, pad_frac=0.08))
style_ax(ax, fontsize=16)

add_all_pairwise_bars(ax, positions, data_R_opt, pairs, p_R_opt,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04, gap_mult=1.6)

plt.tight_layout()
fig.savefig(OUT_DIR / "Opt_R.png", dpi=300)

print("✅ Saved four figures to:", OUT_DIR)
#%%
# ========== PRINT RESULTS ==========
print("\n=== LFP preferred phase (deg, wrapped) ===")
print(f"Global circular permutation test p = {p_global:.5g}")
for (g1, g2), p_raw, p_adj_val in zip(pairs, p_pair, p_adj):
    print(f"{labels[g1]} vs {labels[g2]}: raw p = {p_raw:.5g}, Holm-adjusted p = {p_adj_val:.5g}")

print("\n=== LFP modulation depth (R) ===")
print(f"Kruskal–Wallis H = {kw_stat:.3f}, p = {kw_p:.5g}")
for (g1, g2), p_raw, p_adj_val in zip(pairs, p_R, p_R_adj):
    print(f"{labels[g1]} vs {labels[g2]}: raw p = {p_raw:.5g}, Holm-adjusted p = {p_adj_val:.5g}")

print("\n=== Optical preferred phase (deg, wrapped) ===")
print(f"Global circular permutation test p = {p_global_opt:.5g}")
# Compute adjusted p for optical phase
p_pair_opt_adj, _ = holm_bonferroni(p_pair_opt, alpha=ALPHA)
for (g1, g2), p_raw, p_adj_val in zip(pairs, p_pair_opt, p_pair_opt_adj):
    print(f"{labels[g1]} vs {labels[g2]}: raw p = {p_raw:.5g}, Holm-adjusted p = {p_adj_val:.5g}")

print("\n=== Optical modulation depth (R) ===")
print(f"Kruskal–Wallis H = {kw_stat_opt:.3f}, p = {kw_p_opt:.5g}")
# Compute adjusted p for optical R
p_R_opt_adj, _ = holm_bonferroni(p_R_opt, alpha=ALPHA)
for (g1, g2), p_raw, p_adj_val in zip(pairs, p_R_opt, p_R_opt_adj):
    print(f"{labels[g1]} vs {labels[g2]}: raw p = {p_raw:.5g}, Holm-adjusted p = {p_adj_val:.5g}")