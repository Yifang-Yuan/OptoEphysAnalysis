# -*- coding: utf-8 -*-
"""
Trial-wise plots per animal + pooled stats (new dataset)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------
# NEW DATASET (from your screenshot)
# -----------------------
animals = {
    "Animal 1": {
        "Day 1": [0.101, 0.120, 0.066, 0.070],
        "Day 2": [0.154, 0.084, 0.094, 0.096],
        "Day 3": [0.112, 0.094, 0.085, 0.114],
        "Day 4": [0.083, 0.057, 0.091, 0.082],
    },
    "Animal 2": {
        "Day 1": [0.097, 0.085, 0.057, 0.083],
        "Day 2": [0.083, 0.071, 0.073, 0.093],
        "Day 3": [0.091, 0.111, 0.065, 0.062],
        "Day 4": [0.112, 0.079, 0.054, 0.074],
    },
    "Animal 3": {
        "Day 1": [0.065, 0.076, 0.109, 0.132],
        "Day 2": [0.065, 0.068, 0.070, 0.047],
        "Day 3": [0.081, 0.067, 0.085, 0.079],
        "Day 4": [0.065, 0.052, 0.095, 0.110],
    },
    "Animal 4": {
        "Day 1": [0.105, 0.082, 0.110, 0.062],
        "Day 2": [0.069, 0.076, 0.083, 0.066],
        "Day 3": [0.156, 0.068, 0.046, 0.072],
        "Day 4": [0.091, 0.109, 0.073, 0.067],
    },
    "Animal 5": {
        "Day 1": [0.085, 0.092, 0.080, 0.072],
        "Day 2": [0.141, 0.075, 0.074, 0.079],
        "Day 3": [0.103, 0.081, 0.080, 0.057],
        "Day 4": [0.086, 0.098, 0.075, 0.076],
    },
}

# -----------------------
# Tidy DataFrame
# -----------------------
records = []
for animal, days in animals.items():
    for day, vals in days.items():
        for trial_idx, val in enumerate(vals, start=1):
            records.append({"Animal": animal, "Day": day, "Trial": trial_idx, "Value": val})
df = pd.DataFrame.from_records(records)

# -----------------------
# Helpers (significance utils)
# -----------------------
def holm_bonferroni(pvals):
    """Holm–Bonferroni adjusted p-values (vector)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for k, idx in enumerate(order):
        padj = (m - k) * pvals[order[k]]
        running_max = max(running_max, padj)
        adj[order[k]] = min(running_max, 1.0)
    return adj

def p_to_stars(p):
    return "ns" if p >= 0.05 else ("*" if p < 0.05 and p >= 0.01 else ("**" if p < 0.01 and p >= 0.001 else "***"))

def add_sig_bracket(ax, x1, x2, y, text, height=0.006, lw=1.5, fs=14):
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], lw=lw, c='k')
    ax.text((x1+x2)/2, y+height, text, ha='center', va='bottom', fontsize=fs)

# -----------------------
# 1) Trial‑wise line plot for a single animal
# -----------------------
def plot_animal_trialwise(animal_name, savepath=None):
    assert animal_name in animals, f"{animal_name} not in dataset"
    trials = [1,2,3,4]
    plt.figure(figsize=(7,5))
    for day in ["Day 1","Day 2","Day 3","Day 4"]:
        y = animals[animal_name][day]
        plt.plot(trials, y, marker='o', label=day)
    plt.xlabel("Trial Number", fontsize=16)
    plt.ylabel("Correlation Value", fontsize=16)
    plt.title(f"{animal_name}: Trial-wise Correlation", fontsize=18)
    plt.xticks(trials, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

# Example: plot_animal_trialwise("Animal 1")

# -----------------------
# 2) Combined scatter across trials (all animals & days) + pairwise trial brackets
# -----------------------
def plot_all_scatter_with_trial_brackets(use_holm=True, pairs=None, savepath=None):
    if pairs is None:
        pairs = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]  # all pairs

    plt.figure(figsize=(8,6))
    day_names = ["Day 1","Day 2","Day 3","Day 4"]
    jitter_map = {d: (i - (len(day_names)-1)/2) * 0.05 for i, d in enumerate(day_names)}

    for d in day_names:
        sub = df[df["Day"] == d]
        x = sub["Trial"].values + np.array([jitter_map[d]] * len(sub))
        plt.scatter(x, sub["Value"].values, label=d, alpha=0.8)

    plt.xlabel("Trial Number", fontsize=18)
    plt.ylabel("Correlation Value", fontsize=18)
    plt.title("All Animals & Days: Correlation Values Across Trials", fontsize=20)
    plt.xticks([1,2,3,4], fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12,  loc="upper left") #'best'
    plt.grid(True, linestyle="--", alpha=0.6)

    # Pairwise trial tests (Welch t-tests, pooled over animals & days)
    # trial_vals = {t: df.loc[df["Trial"] == t, "Value"].values for t in [1,2,3,4]}
    # raw_p = []
    # for (a,b) in pairs:
    #     _, p = stats.ttest_ind(trial_vals[a], trial_vals[b], equal_var=False)
    #     raw_p.append(p)
    # p_use = holm_bonferroni(raw_p) if use_holm else raw_p

    # # Brackets
    # ax = plt.gca()
    # y_max = df["Value"].max()
    # base = y_max + 0.01
    # step = 0.02
    # for i, ((a,b), p) in enumerate(zip(pairs, p_use)):
    #     add_sig_bracket(ax, a, b, base + i*step, p_to_stars(p), height=0.006, fs=14)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

# Example: plot_all_scatter_with_trial_brackets(use_holm=True)

# -----------------------
# 3) Stats: across trials, across days, across animals
# -----------------------
def print_pooled_stats():
    # Trials
    trial_groups = [df[df["Trial"]==t]["Value"].values for t in [1,2,3,4]]
    F_t, p_t = stats.f_oneway(*trial_groups)
    H_t, p_kw_t = stats.kruskal(*trial_groups)
    print(f"[Trials]  ANOVA F={F_t:.3f}, p={p_t:.3f} | Kruskal H={H_t:.3f}, p={p_kw_t:.3f}")
    print(df.groupby("Trial")["Value"].agg(['count','mean','std']).assign(sem=lambda s: s['std']/np.sqrt(s['count'])))

    # Days
    day_groups = [df[df["Day"]==d]["Value"].values for d in ["Day 1","Day 2","Day 3","Day 4"]]
    F_d, p_d = stats.f_oneway(*day_groups)
    H_d, p_kw_d = stats.kruskal(*day_groups)
    print(f"\n[Days]    ANOVA F={F_d:.3f}, p={p_d:.3f} | Kruskal H={H_d:.3f}, p={p_kw_d:.3f}")
    print(df.groupby("Day")["Value"].agg(['count','mean','std']).assign(sem=lambda s: s['std']/np.sqrt(s['count'])))

    # Animals
    animal_groups = [df[df["Animal"]==a]["Value"].values for a in sorted(df["Animal"].unique())]
    F_a, p_a = stats.f_oneway(*animal_groups)
    H_a, p_kw_a = stats.kruskal(*animal_groups)
    print(f"\n[Animals] ANOVA F={F_a:.3f}, p={p_a:.3f} | Kruskal H={H_a:.3f}, p={p_kw_a:.3f}")
    print(df.groupby("Animal")["Value"].agg(['count','mean','std']).assign(sem=lambda s: s['std']/np.sqrt(s['count'])))

# Example calls:
# plot_animal_trialwise("Animal 1")
# plot_animal_trialwise("Animal 2")
# plot_animal_trialwise("Animal 3")
# plot_animal_trialwise("Animal 4")
# plot_animal_trialwise("Animal 5")
plot_all_scatter_with_trial_brackets(use_holm=True)
print_pooled_stats()
#%%
import numpy as np
import pandas as pd
from scipy import stats

# Data for Animal 1
data = {
    "Day 1": [0.101, 0.120, 0.066, 0.070],
    "Day 2": [0.154, 0.084, 0.094, 0.096],
    "Day 3": [0.112, 0.094, 0.085, 0.114],
    "Day 4": [0.083, 0.057, 0.091, 0.082],
}

# Convert to DataFrame
records = []
for day, vals in data.items():
    for trial, val in enumerate(vals, start=1):
        records.append({"Day": day, "Trial": trial, "Value": val})
df = pd.DataFrame.from_records(records)

# Pool across days, group by trial
trial_groups = [df[df["Trial"]==t]["Value"].values for t in [1,2,3,4]]

# ANOVA (parametric)
F, p = stats.f_oneway(*trial_groups)
print(f"ANOVA across trials: F = {F:.3f}, p = {p:.4f}")

# Kruskal–Wallis (nonparametric)
H, p_kw = stats.kruskal(*trial_groups)
print(f"Kruskal–Wallis across trials: H = {H:.3f}, p = {p_kw:.4f}")

# Summary per trial
print("\nSummary per trial:")
print(df.groupby("Trial")["Value"].agg(['count','mean','std']).assign(sem=lambda s: s['std']/np.sqrt(s['count'])))
