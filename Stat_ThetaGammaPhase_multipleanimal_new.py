# -*- coding: utf-8 -*-
"""
Multi-animal, multi-state analysis:
- Build per-animal state means (phase & R) from your saved trial pickles
- Across-animal stats:
    * Phase: repeated-measures permutation on cos/sin (complete cases); paired circular pairwise tests
    * R: MixedLM on logit(R) with random intercept for animal (handles unbalanced),
         omnibus via LRT (full vs null), optional within-animal label permutation p,
         pairwise Wald contrasts (Holm adjusted) + Wilcoxon as nonparam check
"""

import os, gzip, pickle, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm, chi2, wilcoxon

# --------------------------- 0) CONFIG: paths ---------------------------
# Fill these with your folders (ThetaPhase_Save). Missing states are OK.
'THETA ANALYSIS'
ANIMALS = {
    # animal_id : { "Locomotion": Path(...), "Awake-stationary": Path(...), "REM": Path(...)}
    "1765508": {
        "Locomotion": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion\ThetaPhase_Save"),
        "Awake-stationary": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\AwakeStationary\ThetaPhase_Save"),
        "REM": Path(r"G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepREM\ThetaPhase_Save"),
    },
    "1844609": {
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion\ThetaPhase_Save"),
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\AwakeStationaryTheta\ThetaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ASleepREM\ThetaPhase_Save"),
    },
    "1881363": {
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ALocomotion\ThetaPhase_Save"),
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\AwakeStationaryTheta\ThetaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\ASleepREM\ThetaPhase_Save"),
    },
    "1851545": {
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1851545_WT_Jedi2p_dis\ALocomotion\ThetaPhase_Save"),
        # no Awake-stationary listed
        # no REM listed
    },
    "1881365": {
        # note: your pasted path had a trailing 'e' in 'Savee'; fix if needed
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\AwakeStationaryTheta\ThetaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\ASleepREM\ThetaPhase_Save"),
    },
    "1887933": {
        # You listed a REM path that points to AwakeStationaryTheta; fix here if needed.
        "Awake-stationary": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationaryTheta\ThetaPhase_Save"),
        "REM": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ASleepREM\ThetaPhase_Save"),
        "Locomotion": Path(r"G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\ALocomotion\ThetaPhase_Save")
    },
}

STATES_ORDER = ["Locomotion", "Awake-stationary", "REM"]
PATTERN = "*.pkl.gz"
RNG = np.random.default_rng(12345)
N_PERM = 5000          # for circular omnibus & optional MixedLM label-permutation
MIXED_PERM = 2000      # permutations for MixedLM omnibus (set 0 to skip)

# --------------------------- 1) I/O helpers -----------------------------
def _load_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def _load_trial_records(folder: Path, pattern="*.pkl.gz"):
    if not folder or not Path(folder).exists():
        return []
    return [_load_pickle(p) for p in sorted(Path(folder).glob(pattern))]

# ---------------------- 2) Per-animal state reduction -------------------
def circ_weighted_mean(phi, w):
    """Weighted circular mean direction (rad) & resultant length."""
    phi = np.asarray(phi, float) % (2*np.pi)
    w   = np.asarray(w, float)
    C = np.sum(w * np.cos(phi)); S = np.sum(w * np.sin(phi))
    mu = (np.arctan2(S, C)) % (2*np.pi)
    Rw = np.hypot(C, S) / (np.sum(w) + 1e-12)
    return float(mu), float(Rw)

def animal_state_mean(folder: Path):
    """
    From trial pickles -> per-animal state mean:
      - pref_phase_rad: weighted by (events * R) per sweep (stable)
      - R_animal: events-weighted mean of R per sweep
    """
    recs = _load_trial_records(folder, PATTERN)
    if not recs:
        return None
    phi = np.array([r["preferred_phase_rad"] for r in recs], float)
    R   = np.array([r["R"] for r in recs], float)
    n   = np.array([r.get("n_events", 1) for r in recs], float)

    mu, _ = circ_weighted_mean(phi, n * R)
    R_mean = float(np.sum(n * R) / (np.sum(n) + 1e-12))
    return {
        "pref_phase_rad": mu,
        "pref_phase_deg": float(np.degrees(mu) % 360.0),
        "R": R_mean,
        "n_trials": int(len(recs)),
        "n_events_total": int(np.sum(n))
    }

def build_animal_dataframe(ANIMALS_dict):
    rows = []
    for aid, states in ANIMALS_dict.items():
        for sname, folder in states.items():
            res = animal_state_mean(folder)
            if res is None: 
                continue
            rows.append({
                "animal_id": str(aid),
                "state": sname,
                "pref_phase_rad": res["pref_phase_rad"],
                "pref_phase_deg": res["pref_phase_deg"],
                "R": res["R"],
                "n_trials": res["n_trials"],
                "n_events_total": res["n_events_total"]
            })
    if not rows:
        raise ValueError("No data found. Check your ANIMALS paths.")
    return pd.DataFrame(rows)

df_animals = build_animal_dataframe(ANIMALS)
print("Per-animal state means:\n", df_animals.sort_values(["animal_id","state"]))

# ---------------------- 3) Phase across animals ------------------------
def circ_dist(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi

def vtest_toward_zero(alpha_rad):
    """One-sample V-test toward 0 rad (directional). Returns (u, p, Rbar, psi)."""
    alpha = np.asarray(alpha_rad, float)
    n = alpha.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    C = np.mean(np.cos(alpha)); S = np.mean(np.sin(alpha))
    Rbar = float(np.hypot(C, S)); psi = float(np.arctan2(S, C))
    m = Rbar * np.cos(psi - 0.0)
    u = np.sqrt(2 * n) * m
    p = float(norm.sf(u))  # one-sided
    return u, p, Rbar, psi

def rayleigh_onesample(alpha_rad):
    alpha = np.asarray(alpha_rad, float)
    n = alpha.size
    if n < 3:
        return np.nan, np.nan, np.nan
    C = np.sum(np.cos(alpha)); S = np.sum(np.sin(alpha))
    Rbar = np.hypot(C, S) / n
    Z = n * (Rbar ** 2)
    p = np.exp(-Z) * (1 + (2*Z - Z*Z) / (4*n))  # Berens 2009 approx
    p = float(np.clip(p, 0, 1))
    return Z, p, float(Rbar)

def holm_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running_max)
    return adj, (adj < alpha)

def make_wide(df, value_col, states):
    return (df.pivot(index="animal_id", columns="state", values=value_col)
              .reindex(columns=states))

# Omnibus (phase): RM-permutation on cos/sin, COMPLETE CASES
wide_phase = make_wide(df_animals, "pref_phase_rad", STATES_ORDER)
cc_phase = wide_phase.dropna(axis=0, how="any")
if cc_phase.shape[0] >= 2 and cc_phase.shape[1] >= 2:
    X = cc_phase.to_numpy()
    cos_mat, sin_mat = np.cos(X), np.sin(X)
    nA, nS = cos_mat.shape

    def omnibus_stat(cos_m, sin_m):
        C = cos_m.mean(axis=0); S = sin_m.mean(axis=0)
        stat = 0.0
        for i in range(nS):
            for j in range(i+1, nS):
                stat += (C[i]-C[j])**2 + (S[i]-S[j])**2
        return stat

    obs_stat = omnibus_stat(cos_mat, sin_mat)
    count_ge = 0
    for _ in range(N_PERM):
        perm_idx = np.array([RNG.permutation(nS) for _ in range(nA)])
        cos_p = np.take_along_axis(cos_mat, perm_idx, axis=1)
        sin_p = np.take_along_axis(sin_mat, perm_idx, axis=1)
        if omnibus_stat(cos_p, sin_p) >= obs_stat - 1e-15:
            count_ge += 1
    p_global_phase = (count_ge + 1) / (N_PERM + 1)
else:
    p_global_phase = np.nan

print(f"\n[Phase] RM-permutation omnibus (complete cases) p = {p_global_phase:.4g}")

# Pairwise (phase): paired circular tests on within-animal Δφ
pairs = [(i, j) for i in range(len(STATES_ORDER)) for j in range(i+1, len(STATES_ORDER))]
pair_results_phase = []
pV_raw, pRay_raw = [], []

for i, j in pairs:
    s1, s2 = STATES_ORDER[i], STATES_ORDER[j]
    w = wide_phase[[s1, s2]].dropna()
    dphi = circ_dist(w[s2].to_numpy(), w[s1].to_numpy())  # S2 − S1
    if len(dphi) >= 2:
        u, p_v, Rbar, psi = vtest_toward_zero(dphi)
        Z, p_ray, _ = rayleigh_onesample(dphi)
    else:
        u = p_v = Z = p_ray = np.nan
    ddeg = np.degrees(np.arctan2(np.mean(np.sin(dphi)), np.mean(np.cos(dphi)))) if len(dphi) else np.nan
    pair_results_phase.append((s1, s2, len(dphi), ddeg, u, p_v, Z, p_ray))
    pV_raw.append(p_v); pRay_raw.append(p_ray)

pV_adj, _ = holm_bonferroni(np.array([p if np.isfinite(p) else 1.0 for p in pV_raw]))
pRay_adj,_= holm_bonferroni(np.array([p if np.isfinite(p) else 1.0 for p in pRay_raw]))

print("[Phase] pairwise (Δμ°; V-test p_adj [p_raw])")
for (s1,s2,n,ddeg,u,pv,Z,pray), pv_adj, pr_adj in zip(pair_results_phase, pV_adj, pRay_adj):
    print(f"  {s1} vs {s2}: n={n}, Δμ≈{ddeg:5.1f}°,  V-test p_adj={pv_adj:.4g} (raw {pv:.4g});  Rayleigh p_adj={pr_adj:.4g} (raw {pray:.4g})")

# ---------------------- 4) MixedLM for R across animals ----------------
# -*- coding: utf-8 -*-
"""
Across-animal stats (phase & R) with robust handling of unbalanced data.
- Phase: RM-permutation on cos/sin (complete cases) + paired circular pairwise tests.
- R:
   * If statsmodels is available -> MixedLM on logit(R) with random intercept (animal):
       - Omnibus via Likelihood Ratio Test (full vs null; ML)
       - Optional within-animal label-permutation omnibus p
       - Pairwise Wald contrasts (Holm-adjusted), report |Δ mean R| on probability scale
   * If statsmodels is NOT available -> permutation-only fallback:
       - Omnibus: within-animal label permutations of state labels, test statistic = ANOVA-style between-state SS
       - Pairwise: within-animal sign-flip permutation on paired differences for all available animals
"""

import numpy as np, pandas as pd
from scipy.stats import norm, chi2, wilcoxon
import warnings

# ---------------- CONFIG ----------------
RNG = np.random.default_rng(12345)
N_PERM = 5000        # permutations for phase omnibus & pairwise fallback tests
MIXED_PERM = 2000    # permutations for MixedLM omnibus (set 0 to skip)
STATES_ORDER = ["Locomotion", "Awake-stationary", "REM"]

# If you already built df_animals earlier, keep it. Otherwise ensure df_animals exists with:
# columns: animal_id, state, pref_phase_rad, pref_phase_deg, R, n_trials, n_events_total

# -------------- Helpers --------------
def make_wide(df, value_col, states):
    return (df.pivot(index="animal_id", columns="state", values=value_col)
              .reindex(columns=states))

def circ_dist(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi

def vtest_toward_zero(alpha_rad):
    alpha = np.asarray(alpha_rad, float)
    n = alpha.size
    if n < 2: return np.nan, np.nan, np.nan, np.nan
    C = np.mean(np.cos(alpha)); S = np.mean(np.sin(alpha))
    Rbar = float(np.hypot(C, S)); psi = float(np.arctan2(S, C))
    m = Rbar * np.cos(psi - 0.0)
    u = np.sqrt(2 * n) * m
    p = float(norm.sf(u))
    return u, p, Rbar, psi

def rayleigh_onesample(alpha_rad):
    alpha = np.asarray(alpha_rad, float)
    n = alpha.size
    if n < 3: return np.nan, np.nan, np.nan
    C = np.sum(np.cos(alpha)); S = np.sum(np.sin(alpha))
    Rbar = np.hypot(C, S) / n
    Z = n * (Rbar ** 2)
    p = np.exp(-Z) * (1 + (2*Z - Z*Z) / (4*n))
    return Z, float(np.clip(p, 0, 1)), float(Rbar)

def holm_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running_max)
    return adj, (adj < alpha)

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return np.log(p/(1-p))
def inv_logit(x): return 1/(1+np.exp(-x))

# -------------- PHASE: RM-permutation omnibus + paired circular tests --------------
wide_phase = make_wide(df_animals, "pref_phase_rad", STATES_ORDER)
cc_phase = wide_phase.dropna(axis=0, how="any")

if cc_phase.shape[0] >= 2 and cc_phase.shape[1] >= 2:
    X = cc_phase.to_numpy()
    cos_mat, sin_mat = np.cos(X), np.sin(X)
    nA, nS = cos_mat.shape

    def phase_omnibus_stat(cos_m, sin_m):
        C = cos_m.mean(axis=0); S = sin_m.mean(axis=0)
        stat = 0.0
        for i in range(nS):
            for j in range(i+1, nS):
                stat += (C[i]-C[j])**2 + (S[i]-S[j])**2
        return stat

    obs = phase_omnibus_stat(cos_mat, sin_mat)
    count_ge = 0
    for _ in range(N_PERM):
        perm_idx = np.array([RNG.permutation(nS) for _ in range(nA)])
        cos_p = np.take_along_axis(cos_mat, perm_idx, axis=1)
        sin_p = np.take_along_axis(sin_mat, perm_idx, axis=1)
        if phase_omnibus_stat(cos_p, sin_p) >= obs - 1e-15:
            count_ge += 1
    p_global_phase = (count_ge + 1) / (N_PERM + 1)
else:
    p_global_phase = np.nan

print(f"\n[Preferred phase] RM-permutation omnibus (complete cases) p = {p_global_phase:.4g}")

pairs = [("Locomotion","Awake-stationary"),
         ("Locomotion","REM"),
         ("Awake-stationary","REM")]

print("[Preferred phase] paired circular comparisons (Δμ°, V-test & Rayleigh on within-animal differences):")
pV_raw, pRay_raw, ddeg_list = [], [], []
for s1, s2 in pairs:
    w = wide_phase[[s1, s2]].dropna()
    dphi = circ_dist(w[s2].to_numpy(), w[s1].to_numpy())
    if len(dphi) >= 2:
        u, p_v, _, _ = vtest_toward_zero(dphi)
        Z, p_ray, _ = rayleigh_onesample(dphi)
        ddeg = np.degrees(np.arctan2(np.mean(np.sin(dphi)), np.mean(np.cos(dphi))))
    else:
        p_v = p_ray = ddeg = np.nan
    pV_raw.append(p_v); pRay_raw.append(p_ray); ddeg_list.append(ddeg)
    print(f"  {s1} vs {s2}: n={len(dphi)}, Δμ≈{(ddeg if np.isfinite(ddeg) else float('nan')):5.1f}°, "
          f"V-test p={p_v if np.isfinite(p_v) else 'NA'}, Rayleigh p={p_ray if np.isfinite(p_ray) else 'NA'}")

pV_adj,_ = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in pV_raw])
pRay_adj,_ = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in pRay_raw])
print("  Holm-adjusted p (V-test):", [f"{x:.4g}" for x in pV_adj])
print("  Holm-adjusted p (Rayleigh):", [f"{x:.4g}" for x in pRay_adj])

# -------------- R: MixedLM if available; else permutation fallback --------------
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except Exception:
    HAS_SM = False
    warnings.warn("statsmodels not installed; using permutation-only fallback for R.")

df_R = df_animals[["animal_id","state","R"]].dropna().copy()
df_R["state"] = pd.Categorical(df_R["state"], categories=STATES_ORDER, ordered=False)

if HAS_SM:
    # --- MixedLM on logit(R) ---
    df_R["R_logit"] = logit(df_R["R"].values)

    full = smf.mixedlm('R_logit ~ C(state, Treatment(reference="Locomotion"))',
                       df_R, groups=df_R["animal_id"])
    res_full = full.fit(method='lbfgs', reml=False, maxiter=200, disp=False)

    null = smf.mixedlm('R_logit ~ 1', df_R, groups=df_R["animal_id"])
    res_null = null.fit(method='lbfgs', reml=False, maxiter=200, disp=False)

    LR = 2*(res_full.llf - res_null.llf)
    df_diff = res_full.df_modelwc - res_null.df_modelwc
    p_LRT = chi2.sf(LR, df_diff)

    print(f"\n[R] MixedLM omnibus (logit R): LR={LR:.3f}, df≈{df_diff:.0f}, p≈{p_LRT:.4g}")

    # Optional label-permutation omnibus p
    p_perm_mixed = np.nan
    if MIXED_PERM and MIXED_PERM > 0:
        count_ge = 0
        per_animal_idx = df_R.groupby("animal_id").indices
        for _ in range(MIXED_PERM):
            df_p = df_R.copy()
            for aid, idxs in per_animal_idx.items():
                df_p.loc[df_R.index[idxs], "state"] = np.random.permutation(df_R.loc[df_R.index[idxs], "state"].values)
            try:
                res_full_p = smf.mixedlm('R_logit ~ C(state, Treatment(reference="Locomotion"))',
                                         df_p, groups=df_p["animal_id"]).fit(method='lbfgs', reml=False, maxiter=200, disp=False)
                LR_p = 2*(res_full_p.llf - res_null.llf)
            except Exception:
                continue
            if LR_p >= LR - 1e-12:
                count_ge += 1
        p_perm_mixed = (count_ge + 1) / (MIXED_PERM + 1)
        print(f"  Label-permutation omnibus p (within-animal shuffles, {MIXED_PERM} iters): {p_perm_mixed:.4g}")

    # Pairwise Wald contrasts (logit scale) + Holm; also show |Δ mean R| on probability scale
    params = res_full.params; cov = res_full.cov_params()
    mu_logit = {
        "Locomotion": params["Intercept"],
        "Awake-stationary": params["Intercept"] + params.get('C(state, Treatment(reference="Locomotion"))[T.Awake–stationary]', 0.0),
        "REM": params["Intercept"] + params.get('C(state, Treatment(reference="Locomotion"))[T.REM]', 0.0),
    }
    mu_prob = {k: float(inv_logit(v)) for k,v in mu_logit.items()}

    def wald_contrast(cvec, params, cov):
        est = float(np.dot(cvec, params))
        se = float(np.sqrt(np.dot(cvec, np.dot(cov, cvec))))
        z = est / (se + 1e-12)
        p = 2*norm.sf(abs(z))
        return est, se, z, p

    pair_labels = [("Locomotion","Awake-stationary"), ("Locomotion","REM"), ("Awake-stationary","REM")]
    contrasts = []
    for a, b in pair_labels:
        if a=="Locomotion" and b=="Awake-stationary":
            cvec = np.array([0.0, -1.0, 0.0])
        elif a=="Locomotion" and b=="REM":
            cvec = np.array([0.0, 0.0, -1.0])
        else:  # A - R
            cvec = np.array([0.0, 1.0, -1.0])
        est, se, z, p = wald_contrast(cvec, params, cov)
        est_abs = abs(est)
        diff_prob = abs(mu_prob[b] - mu_prob[a])
        contrasts.append((a, b, est_abs, se, p, diff_prob))

    p_mix_raw = np.array([c[4] for c in contrasts])
    p_mix_adj, _ = holm_bonferroni(p_mix_raw)

    print("\n[R] MixedLM pairwise (logit-scale |Δ| with Wald p; Holm-adjusted) and prob-scale |Δ mean R|:")
    for (a,b,est,se,p,dp), padj in zip(contrasts, p_mix_adj):
        print(f"  {a} vs {b}: |Δ_logit|={est:.3f} (SE {se:.3f}), p_raw={p:.4g}, p_adj={padj:.4g}; "
              f"|Δ mean R|≈{dp:.4f}   (means: L={mu_prob['Locomotion']:.3f}, A={mu_prob['Awake-stationary']:.3f}, R={mu_prob['REM']:.3f})")

else:
    # -------- Permutation-only fallback for R (no statsmodels) --------
    warnings.warn("MixedLM skipped (statsmodels not found). Using permutation omnibus and pairwise tests for R.")
    wide_R = make_wide(df_animals, "R", STATES_ORDER)

    # Omnibus: between-state SS across animals, with label permutations within animal
    def omnibus_stat_R(dfR):
        means = dfR.groupby("state")["R"].mean()
        ns = dfR.groupby("state")["R"].size()
        gmean = dfR["R"].mean()
        # ANOVA-style between-groups SS (unbalanced): sum_s n_s * (mean_s - grand_mean)^2
        return float(np.sum(ns * (means - gmean)**2))

    df_present = df_R.copy()  # columns: animal_id,state,R

    obs = omnibus_stat_R(df_present)
    count_ge = 0
    # Pre-compute per-animal row indices
    groups = df_present.groupby("animal_id").indices
    for _ in range(N_PERM):
        df_p = df_present.copy()
        for aid, idxs in groups.items():
            # permute labels within-animal among its observed states
            df_p.loc[df_present.index[idxs], "state"] = RNG.permutation(df_present.loc[df_present.index[idxs], "state"].values)
        stat_p = omnibus_stat_R(df_p)
        if stat_p >= obs - 1e-12:
            count_ge += 1
    p_global_R = (count_ge + 1) / (N_PERM + 1)
    print(f"\n[R] Permutation omnibus (within-animal label shuffles): p = {p_global_R:.4g}")

    # Pairwise: within-animal differences, sign-flip permutation (handles unbalanced pairs)
    print("[R] Pairwise differences (|Δ mean R|) with sign-flip permutation (Holm-adjusted):")
    pair_labels = [("Locomotion","Awake-stationary"), ("Locomotion","REM"), ("Awake-stationary","REM")]
    p_raw, effects = [], []
    for a, b in pair_labels:
        w = wide_R[[a,b]].dropna()
        if w.shape[0] >= 2:
            d = (w[b] - w[a]).to_numpy()
            obs = np.mean(np.abs(d))
            count = 0
            for _ in range(N_PERM):
                flips = RNG.choice([-1,1], size=d.size)
                if np.mean(np.abs(flips * d)) >= obs - 1e-12:
                    count += 1
            p = (count + 1) / (N_PERM + 1)
            p_raw.append(p); effects.append((a,b,obs,w.shape[0]))
        else:
            p_raw.append(np.nan); effects.append((a,b,np.nan,w.shape[0]))
    p_adj, _ = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in p_raw])
    for (a, b, e, n), pr, pa in zip(effects, p_raw, p_adj):
        # pretty strings
        e_str  = "NA" if not np.isfinite(e)  else f"{e:.4f}"
        pr_str = "NA" if not np.isfinite(pr) else f"{pr:.4g}"
        # if raw p is NA (not enough paired animals), show NA for adjusted p too
        pa_str = "NA" if not np.isfinite(pr) else f"{pa:.4g}"
    
        print(f"  {a} vs {b}: n={n}, |Δ mean R|≈{e_str}, p_raw={pr_str}, p_adj={pa_str}")
# ---------------------- 5) Wilcoxon (secondary, complete pairs) --------
wide_R = make_wide(df_animals, "R", STATES_ORDER)
print("\n[R] Wilcoxon paired tests (complete pairs, two-sided) with Holm adjustment:")
wilc_p = []
for a, b in pair_labels:
    w = wide_R[[a,b]].dropna()
    if len(w) >= 2:
        try:
            _, pW = wilcoxon(w[a].to_numpy(), w[b].to_numpy(), alternative="two-sided", zero_method="wilcox")
        except ValueError:
            # fall back to sign-flip permutation on diffs
            d = (w[b] - w[a]).to_numpy()
            obs = np.mean(np.abs(d))
            count = 0
            for _ in range(N_PERM):
                flips = RNG.choice([-1,1], size=d.size)
                if np.mean(np.abs(flips*d)) >= obs - 1e-12:
                    count += 1
            pW = (count+1)/(N_PERM+1)
        wilc_p.append(pW)
    else:
        wilc_p.append(np.nan)
p_adj_w, _ = holm_bonferroni(np.array([p if np.isfinite(p) else 1.0 for p in wilc_p]))
for (a,b), pr, pa in zip(pair_labels, wilc_p, p_adj_w):
    print(f"  {a} vs {b}: n={wide_R[[a,b]].dropna().shape[0]}, p_raw={pr if np.isfinite(pr) else 'NA'}, p_adj={pa:.4g}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= Config =========
states_order = ["Locomotion", "Awake-stationary", "REM"]
alpha = 0.05
n_perm = 5000
rng = np.random.default_rng(12345)

# ========= Small helpers (local, self-contained) =========
def jitter(n, width=0.12, rng=None):
    if rng is None: rng = np.random.default_rng()
    return rng.uniform(-width, width, size=n)

def style_ax(ax, fontsize=16):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize); ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize+1)

def smart_ylim(data_list, pad_frac=0.08):
    arr = np.concatenate([np.asarray(d, float) for d in data_list if len(d)>0])
    if arr.size == 0:
        return -1, 1
    dmin, dmax = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(dmin, dmax): 
        return dmin - 1, dmax + 1
    pad = (dmax - dmin) * pad_frac
    return dmin - pad, dmax + pad

def p_to_asterisks(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"

def add_pairwise_bars(ax, positions, data_list, pairs, pvals, text_fs=14, line_h_frac=0.06, pad_top_frac=0.05):
    y0, y1 = ax.get_ylim()
    data_max = max(np.nanmax(d) if len(d)>0 else -np.inf for d in data_list)
    span = (y1 - y0) if (y1 > y0) else 1.0
    h = span * line_h_frac * 0.5
    gap = span * line_h_frac * 2
    base = max(y1, data_max) + span * pad_top_frac
    for i, ((g1, g2), p) in enumerate(zip(pairs, pvals)):
        x1, x2 = positions[g1], positions[g2]
        y = base + i * gap
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.7, c='k', clip_on=False)
        ax.text((x1 + x2)/2, y + h, p_to_asterisks(p), ha='center', va='bottom', fontsize=text_fs, clip_on=False)
    ax.set_ylim(y0, base + len(pairs) * gap + h + span * 0.02)

def holm_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty_like(pvals)
    running_max = 0.0
    for rank, i in enumerate(idx, start=1):
        running_max = max(running_max, pvals[i] * (m - rank + 1))
        adj[i] = min(1.0, running_max)
    return adj, (adj < alpha)

def circ_mean(alpha):
    C = np.nanmean(np.cos(alpha)); S = np.nanmean(np.sin(alpha))
    return (np.arctan2(S, C)) % (2*np.pi), float(np.hypot(C, S))

def circ_dist(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi

def wrap_angles_for_plot(phi_rad, ref_rad=0.0, deg_range=(-180, 180)):
    low, high = deg_range; width = high - low
    shifted = (phi_rad - ref_rad) % (2*np.pi)
    deg = np.degrees(shifted)
    return ((deg - low) % width) + low

def perm_test_circ_means_pair(a, b, n_perm=5000, rng=None):
    if rng is None: rng = np.random.default_rng()
    if len(a)==0 or len(b)==0: 
        return np.nan, np.nan
    mu_a, _ = circ_mean(a); mu_b, _ = circ_mean(b)
    obs = np.abs(circ_dist(mu_a, mu_b))
    pooled = np.concatenate([a, b]); na = len(a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        aa, bb = pooled[:na], pooled[na:]
        d = np.abs(circ_dist(circ_mean(aa)[0], circ_mean(bb)[0]))
        if d >= obs - 1e-15:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    return p, np.degrees(obs)

def perm_test_R_pair(x, y, n_perm=5000, rng=None):
    """Paired sign-flip permutation on within-animal differences (handles unbalanced pairs)."""
    if rng is None: rng = np.random.default_rng()
    if len(x) != len(y) or len(x) < 2:
        return np.nan, np.nan
    d = (np.asarray(y) - np.asarray(x))
    obs = np.mean(np.abs(d))
    count = 0
    for _ in range(n_perm):
        flips = rng.choice([-1, 1], size=d.size)
        if np.mean(np.abs(flips * d)) >= obs - 1e-12:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return p, obs

# ========= Build data vectors by state (per-animal values) =========
present_states = [s for s in states_order if s in df_animals["state"].unique()]
positions = list(range(1, len(present_states)+1))

# Phase: wrap to grand mean (across all animals & states) for display
all_phi = np.radians(df_animals["pref_phase_deg"].values)
grand_mu, _ = circ_mean(all_phi)

phase_deg_by_state = [
    wrap_angles_for_plot(np.radians(df_animals.loc[df_animals["state"]==s, "pref_phase_deg"].values),
                         ref_rad=grand_mu, deg_range=(-180, 180))
    for s in present_states
]

# R: per-animal R by state
R_by_state = [df_animals.loc[df_animals["state"]==s, "R"].values for s in present_states]

# ========= Pairwise stats (displayed on plots; Holm-adjusted) =========
pairs_idx = [(i,j) for i in range(len(present_states)) for j in range(i+1, len(present_states))]

# Phase pairwise (circular permutation on means across animals)
p_phase_raw, dmu_deg = [], []
for i,j in pairs_idx:
    s1, s2 = present_states[i], present_states[j]
    a = np.radians(df_animals.loc[df_animals["state"]==s1, "pref_phase_deg"].dropna().values)
    b = np.radians(df_animals.loc[df_animals["state"]==s2, "pref_phase_deg"].dropna().values)
    p, d = perm_test_circ_means_pair(a, b, n_perm=n_perm, rng=rng)
    p_phase_raw.append(p); dmu_deg.append(d)
p_phase_adj, _ = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in p_phase_raw])

# R pairwise (paired animals only, sign-flip permutation on diffs)
p_R_raw, d_R = [], []
for i,j in pairs_idx:
    s1, s2 = present_states[i], present_states[j]
    wide = (df_animals.pivot(index="animal_id", columns="state", values="R")
                       .reindex(columns=present_states))[[s1, s2]].dropna()
    p, d = perm_test_R_pair(wide[s1].values, wide[s2].values, n_perm=n_perm, rng=rng)
    p_R_raw.append(p); d_R.append(d)
p_R_adj, _ = holm_bonferroni([p if np.isfinite(p) else 1.0 for p in p_R_raw])
#%%
# ========= PLOTS =========

# ---- 1) Preferred phase (deg, wrapped to grand mean) ----
fig, ax = plt.subplots(figsize=(8.8, 5.6))
# boxplots
ax.boxplot(phase_deg_by_state, positions=positions, widths=0.5, showfliers=False)
# jittered points
for x, pos in zip(phase_deg_by_state, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12, rng),
               x, s=60, alpha=0.9)
ax.set_xticks(positions); ax.set_xticklabels(present_states, rotation=10)
ax.set_ylabel("deg, wrapped to grand mean")
yl = smart_ylim(phase_deg_by_state, pad_frac=0.08)
ax.set_ylim(yl)
ax.set_title("Across animals: preferred phase by state")
ax.grid(alpha=0.3, axis='y'); style_ax(ax, fontsize=18)
# annotate with Holm-adjusted p-values
add_pairwise_bars(ax, positions, phase_deg_by_state, pairs_idx, p_phase_adj,
                  text_fs=16, line_h_frac=0.06, pad_top_frac=0.05)
plt.tight_layout()

# ---- 2) Modulation depth R ----
fig, ax = plt.subplots(figsize=(8.8, 5.6))
ax.boxplot(R_by_state, positions=positions, widths=0.5, showfliers=False)
for x, pos in zip(R_by_state, positions):
    ax.scatter(np.full_like(x, pos, dtype=float) + jitter(len(x), 0.12, rng),
               x, s=60, alpha=0.85)
ax.set_xticks(positions); ax.set_xticklabels(present_states, rotation=10)
ax.set_ylabel("Modulation depth (R)")
yl = smart_ylim(R_by_state, pad_frac=0.08)
ax.set_ylim(yl)
ax.set_title("Across animals: modulation depth (R) by state")
ax.grid(alpha=0.3, axis='y'); style_ax(ax, fontsize=18)
# annotate with Holm-adjusted p-values
add_pairwise_bars(ax, positions, R_by_state, pairs_idx, p_R_adj,
                  text_fs=16, line_h_frac=0.06, pad_top_frac=0.05)
plt.tight_layout()

# ---- Console summary (optional) ----
print("\nPairwise phase (Δμ°, Holm-adjusted p shown on plot):")
for (i,j), d, pr, pa in zip(pairs_idx, dmu_deg, p_phase_raw, p_phase_adj):
    print(f"  {present_states[i]} vs {present_states[j]}: Δμ={d:5.1f}°, p_raw={pr if np.isfinite(pr) else 'NA'}, p_adj={pa:.4g}")

print("\nPairwise R (|Δ mean R|, Holm-adjusted p shown on plot):")
for (i,j), d, pr, pa in zip(pairs_idx, d_R, p_R_raw, p_R_adj):
    d_str = "NA" if not np.isfinite(d) else f"{d:.4f}"
    print(f"  {present_states[i]} vs {present_states[j]}: |Δ mean R|={d_str}, p_raw={pr if np.isfinite(pr) else 'NA'}, p_adj={pa:.4g}")
