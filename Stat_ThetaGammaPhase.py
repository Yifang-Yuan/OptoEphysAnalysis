# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:32:12 2025

@author: yifan
"""

import numpy as np
import pandas as pd
import gzip, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

# =============== CONFIG: set your three folders =================
'THETA modulation'
locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\aLocomotion2\ThetaPhase_Save")
awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\AwakeStationary\ThetaPhase_Save")  # e.g. folder containing AwakeStationary_trial*.pkl.gz
rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\Asleep\ThetaPhase_Save")

'GAMMA modulation'
# locomotion_dir = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\OpenField_DLCtracking\GammaPhase_Save")
# awake_dir      = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\GammaPhase_Save")  # e.g. folder containing AwakeStationary_trial*.pkl.gz
# rem_dir        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleep\GammaPhase_Save")

pattern        = "*.pkl.gz"  # leave as-is unless your filenames differ
n_perm         = 5000         # permutation iterations for circular tests
alpha          = 0.05
# ================================================================

def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_group(folder: Path, pattern="*.pkl.gz"):
    paths = sorted(folder.glob(pattern))
    recs  = [_load_pickle(p) for p in paths]
    if not recs:
        raise ValueError(f"No files found in {folder}")
    phi = np.array([r["preferred_phase_rad"] for r in recs], dtype=float)  # radians [0,2pi)
    R   = np.array([r["R"] for r in recs], dtype=float)
    return {"phi": phi % (2*np.pi), "R": R, "n": len(recs), "paths": paths}

# ---------- Circular helpers ----------
def circ_mean(alpha):
    """Mean direction (rad) and resultant length Rbar for 1D angles."""
    C = np.nanmean(np.cos(alpha))
    S = np.nanmean(np.sin(alpha))
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rbar = np.sqrt(C**2 + S**2)
    return mu, Rbar

def circ_dist(a, b):
    """Smallest signed circular difference a-b, in radians in (-pi, pi]."""
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return d

def perm_test_circ_means(groups, n_perm=5000, rng=None):
    """
    Global multi-group permutation test for equality of mean directions.
    Test statistic: sum of squared circular mean differences across groups
                    (equivalently, resultant-vector dispersion between groups).
    Returns p-value.
    """
    if rng is None: rng = np.random.default_rng()
    # observed statistic
    mus = np.array([circ_mean(g["phi"])[0] for g in groups])
    # Use unit vectors
    ux = np.cos(mus); uy = np.sin(mus)
    # Sum of pairwise squared Euclidean distances between mean-direction vectors
    # (equivalent to a dispersion; higher => more separated means)
    obs = 0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            obs += (ux[i]-ux[j])**2 + (uy[i]-uy[j])**2

    # pool angles
    pooled = np.concatenate([g["phi"] for g in groups])
    sizes  = [len(g["phi"]) for g in groups]
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        start = 0
        mus_perm = []
        for sz in sizes:
            ph = pooled[start:start+sz]
            start += sz
            mus_perm.append(circ_mean(ph)[0])
        mus_perm = np.array(mus_perm)
        ux_p = np.cos(mus_perm); uy_p = np.sin(mus_perm)
        stat = 0.0
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                stat += (ux_p[i]-ux_p[j])**2 + (uy_p[i]-uy_p[j])**2
        if stat >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, obs

def perm_test_circ_means_pair(a, b, n_perm=5000, rng=None):
    """
    Pairwise permutation test on |mean direction difference|.
    Returns p-value and observed absolute difference in degrees.
    """
    if rng is None: rng = np.random.default_rng()
    mu_a, _ = circ_mean(a); mu_b, _ = circ_mean(b)
    obs = np.abs(circ_dist(mu_a, mu_b))  # radians in [0, pi]
    pooled = np.concatenate([a, b])
    na = len(a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        aa = pooled[:na]
        bb = pooled[na:]
        mu1, _ = circ_mean(aa); mu2, _ = circ_mean(bb)
        d = np.abs(circ_dist(mu1, mu2))
        if d >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, np.degrees(obs)

def holm_bonferroni(pvals, alpha=0.05, labels=None):
    """Return adjusted decisions for multiple comparisons (Holm)."""
    idx = np.argsort(pvals)
    m = len(pvals)
    decisions = [False]*m
    adj_p = np.empty(m)
    for rank, i in enumerate(idx, start=1):
        adj = pvals[i] * (m - rank + 1)
        adj_p[i] = min(1.0, adj)
    # step-down decisions
    thresh = [alpha/(m - k) for k in range(m)]
    # Apply monotonicity
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj_p[i] = min(1.0, running_max)
        decisions[i] = adj_p[i] < alpha
    out = list(zip(labels if labels else range(m), pvals, adj_p, decisions))
    return out

# -------- Load data ----------
A = load_group(awake_dir, pattern)
R_ = load_group(rem_dir, pattern)
L = load_group(locomotion_dir, pattern)

groups = [A, R_, L]
group_names = ["Awake-stationary", "REM", "Locomotion"]

# --------- Circular tests: preferred phase ----------
p_global, stat = perm_test_circ_means(groups, n_perm=n_perm)
print(f"[Preferred phase] Global permutation p = {p_global:.4g}")

# Pairwise
pairs = [(0,1), (0,2), (1,2)]
p_pair = []
labels = []
pair_deg = []
for i,j in pairs:
    p, ddeg = perm_test_circ_means_pair(groups[i]["phi"], groups[j]["phi"], n_perm=n_perm)
    p_pair.append(p); pair_deg.append(ddeg)
    labels.append(f"{group_names[i]} vs {group_names[j]}")
print("Pairwise circular mean differences (deg) and p:")
for lab, d, p in zip(labels, pair_deg, p_pair):
    print(f"  {lab:>28s}: Δμ = {d:6.2f}°, p = {p:.4g}")
print("\nHolm–Bonferroni (pairwise preferred phase):")
for lab, p, padj, sig in holm_bonferroni(p_pair, alpha=alpha, labels=labels):
    print(f"  {lab:>28s}: p={p:.4g}, p_adj={padj:.4g}, sig={sig}")

# --------- Linear tests: modulation depth R ----------
kw_stat, kw_p = kruskal(A["R"], R_["R"], L["R"])
print(f"\n[R] Kruskal–Wallis across groups: H={kw_stat:.3f}, p={kw_p:.4g}")

# Pairwise Mann–Whitney + Holm
mw_pairs = []
mw_p = []
for i,j in pairs:
    stat, p = mannwhitneyu(groups[i]["R"], groups[j]["R"], alternative="two-sided")
    mw_pairs.append(f"{group_names[i]} vs {group_names[j]}")
    mw_p.append(p)
print("Pairwise Mann–Whitney on R:")
for lab, p in zip(mw_pairs, mw_p):
    print(f"  {lab:>28s}: p = {p:.4g}")
print("\nHolm–Bonferroni (pairwise R):")
for lab, p, padj, sig in holm_bonferroni(mw_p, alpha=alpha, labels=mw_pairs):
    print(f"  {lab:>28s}: p={p:.4g}, p_adj={padj:.4g}, sig={sig}")

#%%
# ------------- Plots with smart y-axis & clean style -------------
def jitter(n, width=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-width, width, size=n)

def smart_ylim(data_list, pad_frac=0.05):
    """Return (ymin, ymax) padded by pad_frac of data range."""
    all_data = np.concatenate([np.asarray(d) for d in data_list])
    dmin, dmax = np.nanmin(all_data), np.nanmax(all_data)
    if np.isclose(dmin, dmax):
        return dmin - 1, dmax + 1
    pad = (dmax - dmin) * pad_frac
    return dmin - pad, dmax + pad

def style_ax(ax, fontsize=16):
    """Remove top/right spines and enlarge ticks/labels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize + 2)

def plot_with_no_stat_bars():
    # Preferred phase (deg)
    fig, ax = plt.subplots(figsize=(8, 5))
    degA = (np.degrees(A["phi"]) % 360.0)
    degR = (np.degrees(R_["phi"]) % 360.0)
    degL = (np.degrees(L["phi"]) % 360.0)
    data_deg = [degA, degR, degL]
    positions = [1, 2, 3]
    ax.boxplot(data_deg, positions=positions, widths=0.5, showfliers=False)
    for x, pos in zip(data_deg, positions):
        ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(group_names, rotation=10)
    ax.set_ylabel("Preferred phase (deg)", fontsize=16)
    ax.set_title(f"Preferred phase by condition (global p={p_global:.3g})")
    ax.set_ylim(*smart_ylim(data_deg, pad_frac=0.05))
    ax.grid(alpha=0.3, axis='y')
    style_ax(ax, fontsize=16)
    plt.tight_layout()
    
    # Modulation depth R
    fig, ax = plt.subplots(figsize=(8, 5))
    data_R = [A["R"], R_["R"], L["R"]]
    ax.boxplot(data_R, positions=positions, widths=0.5, showfliers=False)
    for x, pos in zip(data_R, positions):
        ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(group_names, rotation=10)
    ax.set_ylabel("Modulation depth (R)", fontsize=16)
    ax.set_title(f"R by condition (Kruskal–Wallis p={kw_p:.3g})")
    ax.set_ylim(*smart_ylim(data_R, pad_frac=0.05))
    ax.grid(alpha=0.3, axis='y')
    style_ax(ax, fontsize=16)
    plt.tight_layout()
    
    plt.show()

from scipy.stats import mannwhitneyu

def p_to_asterisks(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

def add_all_pairwise_bars(ax, positions, data_list, pairs, pvals,
                          text_fs=14, line_h_frac=0.08, pad_top_frac=0.05):
    """
    Draw Prism-style bars for given pairs; auto-extends ylim so bars are visible.
    """
    y0, y1 = ax.get_ylim()
    data_max = max(np.nanmax(d) for d in data_list)
    span = (y1 - y0) if (y1 > y0) else 1.0
    h = span * line_h_frac *0.5         # vertical height of each bar end
    gap = span * line_h_frac * 2  # reduced gap for compact stacking
    base = max(y1, data_max) + span * pad_top_frac

    for i, ((g1, g2), p) in enumerate(zip(pairs, pvals)):
        x1, x2 = positions[g1], positions[g2]
        y = base + i * gap
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.7, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, y + h, p_to_asterisks(p),
                ha='center', va='bottom', fontsize=text_fs, clip_on=False)

    ax.set_ylim(y0, base + len(pairs) * gap + h + span * 0.02)



# --- helpers for plotting circular data on a linear axis ---
def wrap_angles_for_plot(phi_rad, ref_rad=0.0, deg_range=(-180.0, 180.0)):
    """
    Rotate phases by ref_rad and wrap into [low, high] degrees (e.g., [-180, 180]).
    Returns degrees.
    """
    low, high = deg_range
    width = high - low
    # rotate
    shifted = (phi_rad - ref_rad) % (2*np.pi)
    # map to [0, 360) deg
    deg = np.degrees(shifted)
    # shift to [low, high)
    deg = ((deg - low) % width) + low
    return deg

# ---- choose a common reference for plotting: grand circular mean across all groups ----
all_phi = np.concatenate([L["phi"], A["phi"], R_["phi"]])
grand_mu, _ = circ_mean(all_phi)  # radians

# === Preferred phase (wrapped for plotting) — Locomotion, Awake-stationary, REM ===
fig, ax = plt.subplots(figsize=(8, 5))

# Wrap each group around grand mean into [-180, 180] deg
degL_plot = wrap_angles_for_plot(L["phi"], ref_rad=grand_mu, deg_range=(-180, 180))
degA_plot = wrap_angles_for_plot(A["phi"], ref_rad=grand_mu, deg_range=(-180, 180))
degR_plot = wrap_angles_for_plot(R_["phi"], ref_rad=grand_mu, deg_range=(-180, 180))

data_deg_plot = [degL_plot, degA_plot, degR_plot]
labels_ord    = ["Locomotion", "Awake-stationary", "REM"]
positions     = [1, 2, 3]

ax.boxplot(data_deg_plot, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(data_deg_plot, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.8)

ax.set_xticks(positions)
ax.set_xticklabels(labels_ord, rotation=10)
ax.set_ylabel("Preferred phase (deg)", fontsize=16)
ax.set_title(f"Preferred phase (global p={p_global:.4g})", fontsize=18)

# smart y-limits (now around ~[-180, 180])
ax.set_ylim(*smart_ylim(data_deg_plot, pad_frac=0.08))
style_ax(ax, fontsize=16)

# Pairwise p-values for phase: use circular permutation test on RAW radians
pairs = [(0, 1), (0, 2), (1, 2)]
group_phi_raw = [L["phi"], A["phi"], R_["phi"]]
pvals_phase = []
for g1, g2 in pairs:
    p, _ = perm_test_circ_means_pair(group_phi_raw[g1], group_phi_raw[g2], n_perm=n_perm)
    pvals_phase.append(p)

# Bars (close to boxes; adjust pad/gap as you like)
add_all_pairwise_bars(ax, positions, data_deg_plot, pairs, pvals_phase,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04)

plt.tight_layout()

# === Modulation depth (R) — same order; unchanged stats ===
fig, ax = plt.subplots(figsize=(8, 5))
data_R = [L["R"], A["R"], R_["R"]]
ax.boxplot(data_R, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(data_R, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12), x, s=50, alpha=0.8)

ax.set_xticks(positions)
ax.set_xticklabels(labels_ord, rotation=10)
ax.set_ylabel("Modulation depth (R)", fontsize=16)
ax.set_title(f"R by condition (Kruskal–Wallis p={kw_p:.4g})", fontsize=18)

ax.set_ylim(*smart_ylim(data_R, pad_frac=0.08))
style_ax(ax, fontsize=16)

# Pairwise p-values for R (Mann–Whitney as before)
pvals_R = []
for g1, g2 in pairs:
    _, p = mannwhitneyu(data_R[g1], data_R[g2], alternative='two-sided')
    pvals_R.append(p)
add_all_pairwise_bars(ax, positions, data_R, pairs, pvals_R,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04)

plt.tight_layout()
plt.show()