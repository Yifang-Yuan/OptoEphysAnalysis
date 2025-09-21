# -*- coding: utf-8 -*-
"""
Single-animal across-behaviour comparison (weighted circular means for phase;
weighted permutation tests for modulation depth R).

- Phase tests: weighted (events × R) permutation at sweep level.
- R tests: weighted by n_events at sweep level (permutation omnibus + pairwise).
- Plots: phase and R with pairwise bars; R plot overlays weighted mean ± 95% CI.

Author: you + helper
"""

import numpy as np
import gzip, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu  # kept for reference if you want unweighted tests, not used for weighted
# ================== CONFIG ==================
locomotion_dir = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion\GammaPhase_Save")
awake_dir      = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\GammaPhase_Save")
rem_dir        = Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepREM\GammaPhase_Save")

# locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion\GammaPhase_Save")
# awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationaryTheta\GammaPhase_Save")
# rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepREM\GammaPhase_Save")
# locomotion_dir = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ALocomotion\ThetaPhase_Save")
# awake_dir      = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationaryTheta\ThetaPhase_Save")
# rem_dir        = Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepREM\ThetaPhase_Save")

pattern  = "*.pkl.gz"
n_perm   = 5000
n_boot   = 2000   # for R weighted CI ribbons
alpha    = 0.05
rng      = np.random.default_rng(12345)
# ============================================

# ---------- IO ----------
def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def load_group(folder: Path, pattern="*.pkl.gz"):
    """
    Returns sweep-level arrays for one behaviour state:
    - phi_w: preferred phase (rad) for sweeps with positive phase weight
    - w_phase: phase weights = n_events * R
    - R_all: modulation depth for all sweeps
    - R_pos / w_R: R and weights (n_events) for sweeps with n_events>0 (for weighted R tests)
    - counts: n_sweeps, total events
    """
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder}")
    paths = sorted(folder.glob(pattern))
    recs  = [_load_pickle(p) for p in paths]
    if not recs:
        raise ValueError(f"No files found in {folder}")

    phi = np.array([r["preferred_phase_rad"] for r in recs], dtype=float) % (2*np.pi)
    R   = np.array([r["R"] for r in recs], dtype=float)
    n_events = np.array([r.get("n_events", 0) for r in recs], dtype=float)

    # Phase weights and mask
    w_phase = n_events * R
    mask_phase = w_phase > 0

    # R weights and mask (use n_events as precision weights)
    w_R   = n_events
    mask_R = w_R > 0

    return {
        # for phase
        "phi_w": phi[mask_phase],
        "w_phase": w_phase[mask_phase],
        # for R
        "R_all": R,                  # unfiltered (for raw plotting if you want)
        "R_pos": R[mask_R],          # filtered for weighted inference
        "w_R": w_R[mask_R],
        # counts
        "n_sweeps": len(recs),
        "n_events": int(np.sum(n_events))
    }

# ---------- Circular (phase) helpers ----------
def circ_weighted_mean(alpha, w):
    alpha = np.asarray(alpha, float) % (2*np.pi)
    w     = np.asarray(w, float)
    C = np.sum(w * np.cos(alpha))
    S = np.sum(w * np.sin(alpha))
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rbar = np.hypot(C, S) / (np.sum(w) + 1e-12)
    return mu, Rbar

def wrap_angles_for_plot(phi_rad, ref_rad=0.0, deg_range=(-180.0, 180.0)):
    low, high = deg_range
    width = high - low
    shifted = (phi_rad - ref_rad) % (2*np.pi)
    deg = np.degrees(shifted)
    deg = ((deg - low) % width) + low
    return deg

def holm_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running_max)
    sig = adj < alpha
    return adj, sig

# ---------- Phase: weighted permutation tests ----------
def perm_test_circ_means_weighted(groups, n_perm=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sizes = [len(g["phi_w"]) for g in groups]
    pooled_phi = np.concatenate([g["phi_w"] for g in groups])
    pooled_w   = np.concatenate([g["w_phase"] for g in groups])

    mus = [circ_weighted_mean(g["phi_w"], g["w_phase"])[0] for g in groups]
    ux, uy = np.cos(mus), np.sin(mus)
    obs = sum((ux[i]-ux[j])**2 + (uy[i]-uy[j])**2
              for i in range(len(groups)) for j in range(i+1, len(groups)))

    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled_phi.size)
        phi_p = pooled_phi[idx]; w_p = pooled_w[idx]
        start = 0; mus_p = []
        for sz in sizes:
            mu_p, _ = circ_weighted_mean(phi_p[start:start+sz], w_p[start:start+sz])
            mus_p.append(mu_p); start += sz
        ux_p, uy_p = np.cos(mus_p), np.sin(mus_p)
        stat = sum((ux_p[i]-ux_p[j])**2 + (uy_p[i]-uy_p[j])**2
                   for i in range(len(sizes)) for j in range(i+1, len(sizes)))
        if stat >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, obs

def perm_test_circ_means_pair_weighted(g1, g2, n_perm=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mu1, _ = circ_weighted_mean(g1["phi_w"], g1["w_phase"])
    mu2, _ = circ_weighted_mean(g2["phi_w"], g2["w_phase"])
    obs = np.abs(((mu1 - mu2 + np.pi) % (2*np.pi)) - np.pi)  # radians ∈ [0, π]

    pooled_phi = np.concatenate([g1["phi_w"], g2["phi_w"]])
    pooled_w   = np.concatenate([g1["w_phase"], g2["w_phase"]])
    n1 = g1["phi_w"].size

    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled_phi.size)
        mu_a, _ = circ_weighted_mean(pooled_phi[idx[:n1]],  pooled_w[idx[:n1]])
        mu_b, _ = circ_weighted_mean(pooled_phi[idx[n1:]], pooled_w[idx[n1:]])
        d = np.abs(((mu_a - mu_b + np.pi) % (2*np.pi)) - np.pi)
        if d >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, np.degrees(obs)

# ---------- R: weighted tests (weights = n_events) ----------
def weighted_mean_R(R, w):
    R = np.asarray(R, float); w = np.asarray(w, float)
    return np.sum(w * R) / (np.sum(w) + 1e-12)

def perm_test_weighted_R_multi(groups, n_perm=5000, rng=None):
    """
    Omnibus test for differences in weighted means of R across >2 groups.
    Statistic = weighted between-groups sum of squares around the global weighted mean.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build pooled arrays using only sweeps with positive weight
    Rs = [g["R_pos"] for g in groups]
    Ws = [g["w_R"]   for g in groups]
    sizes = [len(r) for r in Rs]

    pooled_R = np.concatenate(Rs)
    pooled_W = np.concatenate(Ws)

    # Observed
    mu_g = [weighted_mean_R(r, w) for r, w in zip(Rs, Ws)]
    W_g  = [np.sum(w) for w in Ws]
    mu_all = weighted_mean_R(pooled_R, pooled_W)
    obs = sum(Wg * (mug - mu_all)**2 for mug, Wg in zip(mu_g, W_g))

    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled_R.size)
        R_p = pooled_R[idx]; W_p = pooled_W[idx]
        start = 0; stat = 0.0
        # recompute group means and global mean under permutation
        mu_all_p = weighted_mean_R(R_p, W_p)
        for sz in sizes:
            r = R_p[start:start+sz]; w = W_p[start:start+sz]; start += sz
            mug = weighted_mean_R(r, w); Wg = np.sum(w)
            stat += Wg * (mug - mu_all_p)**2
        if stat >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, obs

def perm_test_weighted_R_pair(g1, g2, n_perm=5000, rng=None):
    """
    Pairwise test on |difference of weighted means| of R between two groups.
    """
    if rng is None:
        rng = np.random.default_rng()
    R1, W1 = g1["R_pos"], g1["w_R"]
    R2, W2 = g2["R_pos"], g2["w_R"]
    obs = abs(weighted_mean_R(R1, W1) - weighted_mean_R(R2, W2))

    pooled_R = np.concatenate([R1, R2])
    pooled_W = np.concatenate([W1, W2])
    n1 = R1.size

    count_ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled_R.size)
        mu_a = weighted_mean_R(pooled_R[idx[:n1]],  pooled_W[idx[:n1]])
        mu_b = weighted_mean_R(pooled_R[idx[n1:]], pooled_W[idx[n1:]])
        d = abs(mu_a - mu_b)
        if d >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, obs

def weighted_R_bootstrap_CI(R, w, n_boot=2000, rng=None, alpha=0.05):
    if rng is None: rng = np.random.default_rng()
    R = np.asarray(R, float); w = np.asarray(w, float)
    n = R.size
    boot = np.empty(n_boot, float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample sweeps
        boot[b] = weighted_mean_R(R[idx], w[idx])
    lo = np.percentile(boot, 100*(alpha/2))
    hi = np.percentile(boot, 100*(1 - alpha/2))
    return weighted_mean_R(R, w), lo, hi

# ---------- Plot helpers ----------
def jitter(n, width=0.12, rng=None):
    if rng is None: rng = np.random.default_rng()
    return rng.uniform(-width, width, size=n)

def smart_ylim(data_list, pad_frac=0.08):
    arr = np.concatenate([np.asarray(d) for d in data_list])
    dmin, dmax = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(dmin, dmax):
        return dmin - 1, dmax + 1
    pad = (dmax - dmin) * pad_frac
    return dmin - pad, dmax + pad

def style_ax(ax, fontsize=16):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize); ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize+2)

def p_to_asterisks(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

def add_all_pairwise_bars(ax, positions, data_list, pairs, pvals,
                          text_fs=14, line_h_frac=0.08, pad_top_frac=0.05):
    y0, y1 = ax.get_ylim()
    data_max = max(np.nanmax(d) for d in data_list)
    span = (y1 - y0) if (y1 > y0) else 1.0
    h = span * line_h_frac * 0.5
    gap = span * line_h_frac * 2
    base = max(y1, data_max) + span * pad_top_frac
    for i, ((g1, g2), p) in enumerate(zip(pairs, pvals)):
        x1, x2 = positions[g1], positions[g2]
        y = base + i * gap
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.7, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, y + h, p_to_asterisks(p),
                ha='center', va='bottom', fontsize=text_fs, clip_on=False)
    ax.set_ylim(y0, base + len(pairs) * gap + h + span * 0.02)

# ================== LOAD ==================
L = load_group(locomotion_dir, pattern)
A = load_group(awake_dir, pattern)
R_ = load_group(rem_dir, pattern)
labels = ["Locomotion", "Awake–stationary", "REM"]
print("Counts per state (sweeps / events):")
print(f"  L: n_sweeps={L['n_sweeps']:>3}, events={L['n_events']}")
print(f"  A: n_sweeps={A['n_sweeps']:>3}, events={A['n_events']}")
print(f"  R: n_sweeps={R_['n_sweeps']:>3}, events={R_['n_events']}")

# ================== PHASE: weighted stats ==================
groups_phase = [L, A, R_]
pairs_idx = [(0,1), (0,2), (1,2)]

p_global_phase, _ = perm_test_circ_means_weighted(groups_phase, n_perm=n_perm, rng=rng)
print(f"\n[Phase] weighted omnibus permutation p = {p_global_phase:.4g}")

p_phase_raw = []; d_phase_deg = []
for i, j in pairs_idx:
    p, ddeg = perm_test_circ_means_pair_weighted(groups_phase[i], groups_phase[j], n_perm=n_perm, rng=rng)
    p_phase_raw.append(p); d_phase_deg.append(ddeg)
p_phase_adj, _ = holm_bonferroni(p_phase_raw, alpha=alpha)

print("[Phase] pairwise (raw p, Holm-adjusted p; stars use adjusted p):")
for k, (i, j) in enumerate(pairs_idx):
    print(f"  {labels[i]} vs {labels[j]}: |Δμ| = {d_phase_deg[k]:5.1f}°, "
          f"p_raw = {p_phase_raw[k]:.4g}, p_adj = {p_phase_adj[k]:.4g}, "
          f"stars: {p_to_asterisks(p_phase_adj[k])}")

# ================== R: weighted stats ==================
groups_R = [L, A, R_]
p_global_R, _ = perm_test_weighted_R_multi(groups_R, n_perm=n_perm, rng=rng)
print(f"\n[R] weighted omnibus permutation p = {p_global_R:.4g}")

p_R_raw = []; d_R_raw = []
for i, j in pairs_idx:
    p, d = perm_test_weighted_R_pair(groups_R[i], groups_R[j], n_perm=n_perm, rng=rng)
    p_R_raw.append(p); d_R_raw.append(d)
p_R_adj, _ = holm_bonferroni(p_R_raw, alpha=alpha)

print("[R] pairwise (raw p, Holm-adjusted p; stars use adjusted p):")
for k, (i, j) in enumerate(pairs_idx):
    print(f"  {labels[i]} vs {labels[j]}: |Δ(mean_R_w)| = {d_R_raw[k]:.4f}, "
          f"p_raw = {p_R_raw[k]:.4g}, p_adj = {p_R_adj[k]:.4g}, "
          f"stars: {p_to_asterisks(p_R_adj[k])}")

# ================== PLOTS ==================
positions = [1, 2, 3]

# Phase plot (make all groups either absolute or aligned; here keep your current mix)
all_phi = np.concatenate([L["phi_w"], A["phi_w"], R_["phi_w"]])
all_w   = np.concatenate([L["w_phase"], A["w_phase"], R_["w_phase"]])
grand_mu, _ = circ_weighted_mean(all_phi, all_w)

# ABSOLUTE for Locomotion; aligned for others (your current choice)
#degL = (np.degrees(L["phi_w"]) % 360.0)
degL = wrap_angles_for_plot(L["phi_w"], ref_rad=grand_mu, deg_range=(-180, 180))
degA = wrap_angles_for_plot(A["phi_w"], ref_rad=grand_mu, deg_range=(-180, 180))
degR = wrap_angles_for_plot(R_["phi_w"], ref_rad=grand_mu, deg_range=(-180, 180))
data_deg_plot = [degL, degA, degR]

fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.boxplot(data_deg_plot, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(data_deg_plot, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + rng.uniform(-0.12, 0.12, size=len(x)),
               x, s=50, alpha=0.85)
ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("Preferred phase (deg)")
ax.set_title(f"Preferred phase (weighted) | omnibus p={p_global_phase:.3g}\n"
             f"n_sweeps L/A/R={L['n_sweeps']}/{A['n_sweeps']}/{R_['n_sweeps']}\n"
             f"events (weights) L/A/R={L['n_events']}/{A['n_events']}/{R_['n_events']}",
             fontsize=12)
ax.set_ylim(*smart_ylim(data_deg_plot, pad_frac=0.08))
ax.grid(alpha=0.3, axis='y'); style_ax(ax, fontsize=16)

# Use ADJUSTED p-values for plot stars
add_all_pairwise_bars(ax, positions, data_deg_plot, pairs_idx, p_phase_adj,
                      text_fs=14, line_h_frac=0.06, pad_top_frac=0.05)
plt.tight_layout()

# R plot (weighted): show raw sweeps + overlay weighted mean ± 95% CI
fig, ax = plt.subplots(figsize=(8.6, 5.4))
R_lists_raw = [L["R_all"], A["R_all"], R_["R_all"]]
ax.boxplot(R_lists_raw, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(R_lists_raw, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + rng.uniform(-0.12, 0.12, size=len(x)),
               x, s=50, alpha=0.6)

groups_for_CI = [L, A, R_]
weighted_means = []; ci_lo = []; ci_hi = []
for g in groups_for_CI:
    mu, lo, hi = weighted_R_bootstrap_CI(g["R_pos"], g["w_R"], n_boot=n_boot, rng=rng, alpha=0.05)
    weighted_means.append(mu); ci_lo.append(lo); ci_hi.append(hi)

ax.errorbar(positions, weighted_means,
            yerr=[np.array(weighted_means)-np.array(ci_lo),
                  np.array(ci_hi)-np.array(weighted_means)],
            fmt='o', ms=8, lw=1.8, capsize=5, label="Weighted mean ± 95% CI")

ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=10)
ax.set_ylabel("Modulation depth (R)")
ax.set_title(f"R by state | weighted omnibus p={p_global_R:.3g}",
             fontsize=14)
ax.set_ylim(*smart_ylim(R_lists_raw, pad_frac=0.08))
ax.grid(alpha=0.3, axis='y'); style_ax(ax, fontsize=16)
ax.legend(loc="upper left", frameon=False, fontsize=16)

# Use ADJUSTED p-values for plot stars
add_all_pairwise_bars(ax, positions, R_lists_raw, pairs_idx, p_R_adj,
                      text_fs=14, line_h_frac=0.06, pad_top_frac=0.05)
plt.tight_layout()
plt.show()
