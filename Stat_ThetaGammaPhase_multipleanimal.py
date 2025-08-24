# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:28:18 2025

@author: yifan
"""
import numpy as np
import pandas as pd
import gzip, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

'THETA modulation'
# ====== Multi-animal, multi-state phase/R aggregation & stats ======
import numpy as np, pandas as pd, gzip, pickle, os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

# ---------- 1) Tell us where each animal's state folders are ----------
# Use ANY subset you have; missing states are ok (they'll be skipped).
'THETA ANALYSIS'
# ANIMALS = {
#     # animal_id : { "Locomotion": Path(...), "Awake-stationary": Path(...), "REM": Path(...)}
#     "1765508": {
#         "Locomotion": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion\ThetaPhase_Save"),
#         "Awake-stationary": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\ThetaPhase_Save"),
#         "REM": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepREM\ThetaPhase_Save"),
#     },
#     "1844609": {
#         "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion\ThetaPhase_Save"),
#         "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationaryTheta\ThetaPhase_Save"),
#         "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepREM\ThetaPhase_Save"),
#     },
#     "1881363": {
#         "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ALocomotion\ThetaPhase_Save"),
#         "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationaryTheta\ThetaPhase_Save"),
#         "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepREM\ThetaPhase_Save"),
#     },
#     "1851545": {
#         "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1851545_WT_Jedi2p_dis\ALocomotion\ThetaPhase_Save"),
#         # no Awake-stationary listed
#         # no REM listed
#     },
#     "1881365": {
#         # note: your pasted path had a trailing 'e' in 'Savee'; fix if needed
#         "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\AwakeStationaryTheta\ThetaPhase_Save"),
#         "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\ASleepREM\ThetaPhase_Save"),
#     },
#     "1887933": {
#         # You listed a REM path that points to AwakeStationaryTheta; fix here if needed.
#         "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationaryTheta\ThetaPhase_Save"),
#         "REM": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ASleepREM\ThetaPhase_Save"),
#         "Locomotion": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ALocomotion\ThetaPhase_Save")
#         # "REM": Path(r"...\ASleepREM\ThetaPhase_Save")  # add if present
#     },
# }

ANIMALS = {
    # animal_id : { "Locomotion": Path(...), "Awake-stationary": Path(...), "REM": Path(...)}
    "1765508": {
        "Locomotion": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion\GammaPhase_Save"),
        "Awake-stationary": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\GammaPhase_Save"),
        "REM": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepREM\GammaPhase_Save"),
    },
    "1844609": {
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion\GammaPhase_Save"),
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationaryTheta\GammaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepREM\GammaPhase_Save"),
    },
    "1881363": {
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ALocomotion\GammaPhase_Save"),
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationaryTheta\GammaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepREM\GammaPhase_Save"),
    },
    "1851545": {
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1851545_WT_Jedi2p_dis\ALocomotion\GammaPhase_Save"),
        # no Awake-stationary listed
        # no REM listed
    },
    "1881365": {
        # note: your pasted path had a trailing 'e' in 'Savee'; fix if needed
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\AwakeStationaryTheta\GammaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\ASleepREM\GammaPhase_Save"),
    },
    "1887933": {
        # You listed a REM path that points to AwakeStationaryTheta; fix here if needed.
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationaryTheta\GammaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ASleepREM\GammaPhase_Save"),
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ALocomotion\GammaPhase_Save")
        # "REM": Path(r"...\ASleepREM\ThetaPhase_Save")  # add if present
    },
}


# ---------- 2) Helpers ----------
def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def _load_trial_records(folder: Path, pattern="*.pkl.gz"):
    if not folder.exists():
        return []
    return [_load_pickle(p) for p in sorted(folder.glob(pattern))]

def circ_weighted_mean(phi, w):
    """Weighted circular mean direction (rad) and resultant length (R_bar)."""
    phi = np.asarray(phi, float) % (2*np.pi)
    w   = np.asarray(w, float)
    C = np.sum(w * np.cos(phi))
    S = np.sum(w * np.sin(phi))
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rw = np.sqrt(C**2 + S**2) / (np.sum(w) + 1e-12)
    return mu, Rw

def wrap_angles_for_plot(phi_rad, ref_rad=0.0, deg_range=(-180, 180)):
    """Rotate phases by ref_rad and wrap into desired degree window."""
    low, high = deg_range
    width = high - low
    shifted = (phi_rad - ref_rad) % (2*np.pi)
    deg = np.degrees(shifted)
    deg = ((deg - low) % width) + low
    return deg

# ---------- 3) Reduce trials -> animal means per state ----------
def animal_state_mean(folder: Path):
    """
    Returns dict with per-animal state mean:
      - pref_phase_rad (weighted by n_events * R)
      - R_mean (weighted by n_events)
      - n_trials, n_events_total
    """
    recs = _load_trial_records(folder)
    if not recs:
        return None

    # pull arrays
    phi = np.array([r["preferred_phase_rad"] for r in recs], float)
    R   = np.array([r["R"] for r in recs], float)
    n   = np.array([r.get("n_events", 1) for r in recs], float)  # fall back to 1

    # phase: weight by events * R (stable & standard in your previous code)
    w_phase = n * R
    mu, Rw  = circ_weighted_mean(phi, w_phase)

    # per-animal R: weight by events
    R_mean = np.sum(n * R) / (np.sum(n) + 1e-12)

    return {
        "pref_phase_rad": float(mu),
        "pref_phase_deg": float(np.degrees(mu) % 360.0),
        "R_animal": float(R_mean),
        "R_resultant_within": float(Rw),  # optional: resultant of the weighted mean vector
        "n_trials": int(len(recs)),
        "n_events_total": int(np.sum(n))
    }

def build_animal_dataframe(ANIMALS_dict):
    """
    Build a tidy DataFrame with columns:
      animal_id, state, pref_phase_rad, pref_phase_deg, R, n_trials, n_events_total
    """
    rows = []
    for aid, states in ANIMALS_dict.items():
        for state_name, folder in states.items():
            res = animal_state_mean(folder)
            if res is None: 
                continue
            rows.append({
                "animal_id": aid,
                "state": state_name,
                "pref_phase_rad": res["pref_phase_rad"],
                "pref_phase_deg": res["pref_phase_deg"],
                "R": res["R_animal"],
                "n_trials": res["n_trials"],
                "n_events_total": res["n_events_total"]
            })
    if not rows:
        raise ValueError("No data found in any listed folders.")
    return pd.DataFrame(rows)

df_animals = build_animal_dataframe(ANIMALS)
print(df_animals)

# ---------- 4) Stats across ANIMALS (states = groups) ----------
# Group into lists by state (samples are ANIMALS now)
states_order = ["Locomotion", "Awake-stationary", "REM"]
present_states = [s for s in states_order if s in df_animals["state"].unique()]

# circular helpers (reuse weighted-unweighted simple mean for group-level tests)
def circ_mean(alpha):
    C = np.nanmean(np.cos(alpha)); S = np.nanmean(np.sin(alpha))
    return (np.arctan2(S, C)) % (2*np.pi), np.sqrt(C**2 + S**2)

def circ_dist(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi

def perm_test_circ_means(groups, n_perm=5000, rng=None):
    if rng is None: rng = np.random.default_rng()
    mus = np.array([circ_mean(g)[0] for g in groups])
    ux, uy = np.cos(mus), np.sin(mus)
    obs = 0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            obs += (ux[i]-ux[j])**2 + (uy[i]-uy[j])**2

    pooled = np.concatenate(groups)
    sizes  = [len(g) for g in groups]
    count_ge = 0
    for _ in range(5000):
        rng.shuffle(pooled)
        start = 0; mus_p = []
        for sz in sizes:
            ph = pooled[start:start+sz]; start += sz
            mus_p.append(circ_mean(ph)[0])
        mus_p = np.array(mus_p)
        stat = 0.0
        ux_p, uy_p = np.cos(mus_p), np.sin(mus_p)
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                stat += (ux_p[i]-ux_p[j])**2 + (uy_p[i]-uy_p[j])**2
        if stat >= obs - 1e-15:
            count_ge += 1
    return (count_ge + 1) / (5000 + 1)

def perm_test_circ_means_pair(a, b, n_perm=5000, rng=None):
    if rng is None: rng = np.random.default_rng()
    mu_a, _ = circ_mean(a); mu_b, _ = circ_mean(b)
    obs = np.abs(circ_dist(mu_a, mu_b))
    pooled = np.concatenate([a, b]); na = len(a)
    count_ge = 0
    for _ in range(5000):
        rng.shuffle(pooled)
        aa, bb = pooled[:na], pooled[na:]
        d = np.abs(circ_dist(circ_mean(aa)[0], circ_mean(bb)[0]))
        if d >= obs - 1e-15:
            count_ge += 1
    return (count_ge + 1) / (5000 + 1), np.degrees(obs)

def holm_bonferroni(pvals, alpha=0.05):
    idx = np.argsort(pvals); m = len(pvals)
    adj = np.empty(m)
    # step-down adjusted p-values (monotone)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running_max)
    sig = adj < alpha
    return adj, sig

# Build arrays by state (animals are samples)
phi_by_state = []
R_by_state   = []
for s in present_states:
    phi_by_state.append(np.radians(df_animals.loc[df_animals["state"]==s, "pref_phase_deg"].values))
    R_by_state.append(df_animals.loc[df_animals["state"]==s, "R"].values)

p_global = perm_test_circ_means(phi_by_state)
H, p_kw  = kruskal(*R_by_state)

# Pairwise (only for states that actually exist)
pairs = [(i,j) for i in range(len(present_states)) for j in range(i+1, len(present_states))]
p_phase_raw, ddeg = [], []
p_R_raw = []
for i,j in pairs:
    p, d = perm_test_circ_means_pair(phi_by_state[i], phi_by_state[j]); p_phase_raw.append(p); ddeg.append(d)
    _, pR = mannwhitneyu(R_by_state[i], R_by_state[j], alternative="two-sided"); p_R_raw.append(pR)
p_phase_adj, sig_phase = holm_bonferroni(np.array(p_phase_raw))
p_R_adj, sig_R = holm_bonferroni(np.array(p_R_raw))

print(f"[Preferred phase] global permutation p = {p_global:.4g}")
for (i,j), p, pa, dd in zip(pairs, p_phase_raw, p_phase_adj, ddeg):
    print(f"  {present_states[i]} vs {present_states[j]}: Δμ={dd:5.1f}°, p={p:.4g}, p_adj={pa:.4g}")
print(f"[R] Kruskal–Wallis H={H:.3g}, p={p_kw:.4g}")
for (i,j), p, pa in zip(pairs, p_R_raw, p_R_adj):
    print(f"  {present_states[i]} vs {present_states[j]}: p={p:.4g}, p_adj={pa:.4g}")

# ---------- 5) Plots (per-animal samples) ----------
def jitter(n, width=0.1, rng=None):
    if rng is None: rng = np.random.default_rng()
    return rng.uniform(-width, width, size=n)

def smart_ylim(data_list, pad_frac=0.08):
    arr = np.concatenate([np.asarray(d) for d in data_list])
    dmin, dmax = np.nanmin(arr), np.nanmax(arr)
    pad = (dmax - dmin) * pad_frac if dmax>dmin else 1.0
    return dmin - pad, dmax + pad

def style_ax(ax, fontsize=16):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize); ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize+2)
from scipy.stats import mannwhitneyu

def wrap_to_pm180(deg):
    """Wrap degrees into [-180, 180] by subtracting 360° only for values >180°."""
    return (deg + 180) % 360 - 180

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
    h = span * line_h_frac * 0.5      # bar-end height
    gap = span * line_h_frac * 2      # vertical spacing between bars
    base = max(y1, data_max) + span * pad_top_frac

    for i, ((g1, g2), p) in enumerate(zip(pairs, pvals)):
        x1, x2 = positions[g1], positions[g2]
        y = base + i * gap
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.7, c='k', clip_on=False)
        ax.text((x1 + x2) / 2, y + h, p_to_asterisks(p),
                ha='center', va='bottom', fontsize=text_fs, clip_on=False)

    ax.set_ylim(y0, base + len(pairs) * gap + h + span * 0.02)

def holm_bonferroni(pvals, alpha=0.05):
    """Return Holm–Bonferroni adjusted p-values (monotone step-down)."""
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


# Build displayed data: wrap only values > 180° into [-180, 180]
deg_pm180_lists = [
    wrap_to_pm180(df_animals.loc[df_animals["state"]==s, "pref_phase_deg"].values)
    for s in present_states
]
positions = np.arange(1, len(present_states)+1)

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.boxplot(deg_pm180_lists, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(deg_pm180_lists, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12),
               x, s=60, alpha=0.9)

ax.set_xticks(positions)
ax.set_xticklabels(present_states, rotation=10)
ax.set_ylabel("Preferred phase (deg)")
ax.set_title(f"Preferred phase (global p={p_global:.3g})")

for y in (-180, -90, 0, 90, 180):
    ax.axhline(y, color='k', lw=0.6, alpha=0.25)

ax.set_ylim(-180, 180)
ax.grid(alpha=0.3, axis='y')
style_ax(ax, fontsize=16)
plt.tight_layout()

# --- Pairwise p-values for phase (use RAW radians, not displayed values) ---
pairs = [(i, j) for i in range(len(present_states)) for j in range(i+1, len(present_states))]

# Build raw radians per state
phi_by_state = [np.radians(df_animals.loc[df_animals["state"]==s, "pref_phase_deg"].values)
                for s in present_states]

# Your permutation test function from earlier:
pvals_phase_raw = []
for i, j in pairs:
    p, _ = perm_test_circ_means_pair(phi_by_state[i], phi_by_state[j], n_perm=5000)
    pvals_phase_raw.append(p)

# Choose whether to annotate with adjusted p or raw p
pvals_phase_adj, _ = holm_bonferroni(pvals_phase_raw, alpha=0.05)
# Annotate with adjusted p-values:
add_all_pairwise_bars(ax, positions, deg_pm180_lists, pairs, pvals_phase_adj,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04)


fig, ax = plt.subplots(figsize=(8.2, 5.2))
R_lists = [df_animals.loc[df_animals["state"]==s, "R"].values for s in present_states]

ax.boxplot(R_lists, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(R_lists, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12),
               x, s=60, alpha=0.9)

ax.set_xticks(positions)
ax.set_xticklabels(present_states, rotation=10)
ax.set_ylabel("Modulation depth (R)")
ax.set_title(f"R by state (Kruskal–Wallis p={p_kw:.3g})")
ax.set_ylim(*smart_ylim(R_lists))
ax.grid(alpha=0.3, axis='y')
style_ax(ax, fontsize=16)
plt.tight_layout()

# Pairwise Mann–Whitney on R
pvals_R_raw = []
for i, j in pairs:
    _, p = mannwhitneyu(R_lists[i], R_lists[j], alternative="two-sided")
    pvals_R_raw.append(p)

pvals_R_adj, _ = holm_bonferroni(pvals_R_raw, alpha=0.05)
add_all_pairwise_bars(ax, positions, R_lists, pairs, pvals_R_adj,
                      text_fs=16, line_h_frac=0.06, pad_top_frac=0.04)
