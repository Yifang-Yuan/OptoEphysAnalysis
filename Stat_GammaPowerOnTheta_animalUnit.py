# -*- coding: utf-8 -*-
"""
Theta–gamma coupling (gamma power on theta phase) across animals, states, and gamma bands
Recomputes directly from SyncRecording folders (does NOT rely on GammaPowerOnTheta_trial*.pkl.gz).

Outputs:
  df_sweep  : one row per sweep × band × state × animal
  df_animal : one row per animal × band × state (aggregated across sweeps)
  stats     : exact circular paired tests:
                - LFP vs Optical within each state
                - Between-state comparisons (Moving/Rest/REM) for delta phase (Opt-LFP)
                - Optional omnibus across 3 states (exact label permutation)

Author: yifan + ChatGPT (assembled)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE

_TWO_PI = 2*np.pi


# -----------------------------
# Basic helpers
# -----------------------------
def _iter_sync_recordings(parent: Path):
    parent = Path(parent)
    recs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("SyncRecording")]
    def rec_num(name):
        m = re.search(r"SyncRecording(\d+)", name)
        return int(m.group(1)) if m else 10**9
    for p in sorted(recs, key=lambda x: rec_num(x.name)):
        yield p.name

def _butter_bandpass_sos(low, high, fs, order=4):
    return butter(order, [low, high], btype='band', fs=fs, output='sos')

def _inst_power(x):
    a = hilbert(x)
    return np.abs(a)**2

def wrap_pi(x):
    x = np.asarray(x)
    return (x + np.pi) % (2*np.pi) - np.pi

def circ_dist(a, b):
    """Signed circular difference a-b in (-pi, pi]."""
    return (a - b + np.pi) % (2*np.pi) - np.pi

def circ_mean(angles, w=None):
    """
    Circular mean direction (rad) and resultant length of the mean (Rbar).
    If weights w provided, uses weighted vector mean.
    """
    ang = np.asarray(angles, float) % _TWO_PI
    if ang.size == 0:
        return np.nan, np.nan
    if w is None:
        C = np.sum(np.cos(ang))
        S = np.sum(np.sin(ang))
        n = ang.size
        R = np.sqrt(C*C + S*S)
        mu = (np.arctan2(S, C)) % _TWO_PI
        return float(mu), float(R / n)
    w = np.asarray(w, float)
    w = np.where(np.isfinite(w), w, 0.0)
    C = np.sum(w * np.cos(ang))
    S = np.sum(w * np.sin(ang))
    denom = np.sum(w) + 1e-12
    R = np.sqrt(C*C + S*S)
    mu = (np.arctan2(S, C)) % _TWO_PI
    return float(mu), float(R / denom)

def rayleigh_test(angles):
    """
    Classic Rayleigh test (unweighted) for non-uniformity of circular data.
    Returns dict with n, Rbar, Z, p. Uses Berens (CircStat toolbox) small-sample correction.
    """
    a = np.asarray(angles, float) % _TWO_PI
    n = a.size
    if n == 0:
        return {"n": 0, "Rbar": np.nan, "Z": np.nan, "p": np.nan}
    C = np.sum(np.cos(a))
    S = np.sum(np.sin(a))
    R = np.sqrt(C*C + S*S)
    Rbar = R / n
    Z = n * (Rbar**2)
    p = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2))
    p = float(np.clip(p, 0.0, 1.0))
    return {"n": int(n), "Rbar": float(Rbar), "Z": float(Z), "p": p}

def _cycle_ids(theta_phase):
    th_unw = np.unwrap(np.asarray(theta_phase, float))
    return np.floor(th_unw / _TWO_PI).astype(int)

def cyclewise_power_pref_angles(theta_phase, power, min_samples_per_cycle=10):
    """
    For each theta cycle, compute a single power-weighted preferred phase.
    Returns per-cycle preferred angles (rad), n_cycles, per-cycle weight (sum power).
    """
    th = np.asarray(theta_phase, float) % _TWO_PI
    pw = np.asarray(power, float)
    cid = _cycle_ids(th)

    angles, weights = [], []
    for c in range(cid.min(), cid.max() + 1):
        sel = (cid == c)
        if sel.sum() < min_samples_per_cycle:
            continue
        C = np.sum(pw[sel] * np.cos(th[sel]))
        S = np.sum(pw[sel] * np.sin(th[sel]))
        if C == 0 and S == 0:
            continue
        angles.append((np.arctan2(S, C)) % _TWO_PI)
        weights.append(np.sum(pw[sel]))
    angles = np.asarray(angles, float)
    weights = np.asarray(weights, float)
    return angles, int(len(angles)), weights


# -----------------------------
# Core: gamma power on theta phase for one modality
# -----------------------------
def gamma_power_on_theta(theta_phase, signal, fs, gamma_band, bins=30,
                         gate_percentile=None, gate_on_power=None,
                         min_samples_per_cycle=10):
    """
    Compute gamma power (Hilbert amplitude^2) on theta phase.
    Optionally gate to "high-gamma" samples:
      - gate_percentile: e.g. 75 means keep samples where gate_on_power >= 75th percentile
      - gate_on_power: if provided, uses this power array for gating; else uses this modality's power
    Returns dict with:
      bin_centers, mean_pow_norm, preferred_phase_rad, Rw, weights_sum,
      cycle_pref_angles, rayleigh
    """
    th = np.asarray(theta_phase, float) % _TWO_PI
    x = np.asarray(signal, float)

    sos = _butter_bandpass_sos(gamma_band[0], gamma_band[1], fs, order=4)
    xg = sosfiltfilt(sos, x)
    Pg = _inst_power(xg)

    # gating mask (optional)
    if gate_percentile is not None:
        refP = Pg if gate_on_power is None else np.asarray(gate_on_power, float)
        thr = np.nanpercentile(refP, gate_percentile)
        m = np.isfinite(th) & np.isfinite(Pg) & (refP >= thr)
        th_use = th[m]
        Pg_use = Pg[m]
    else:
        m = np.isfinite(th) & np.isfinite(Pg)
        th_use = th[m]
        Pg_use = Pg[m]

    # bin mean power
    edges = np.linspace(0, _TWO_PI, bins+1)
    idx = np.digitize(th_use, edges) - 1
    idx[idx == bins] = bins - 1

    mean_pow = np.zeros(bins, float)
    for b in range(bins):
        sel = (idx == b)
        mean_pow[b] = np.nanmean(Pg_use[sel]) if np.any(sel) else 0.0
    mean_pow_norm = mean_pow / (np.nanmean(mean_pow) + 1e-12)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # preferred phase and modulation depth (power-weighted)
    C = np.sum(Pg_use * np.cos(th_use))
    S = np.sum(Pg_use * np.sin(th_use))
    pref = (np.arctan2(S, C)) % _TWO_PI
    Rw = np.sqrt(C*C + S*S) / (np.sum(Pg_use) + 1e-12)
    wsum = float(np.sum(Pg_use))

    # cycle-level angles (for Rayleigh)
    cyc_ang, ncyc, cyc_w = cyclewise_power_pref_angles(th_use, Pg_use, min_samples_per_cycle=min_samples_per_cycle)
    ray = rayleigh_test(cyc_ang)

    return {
        "bin_centers": bin_centers,
        "contour_norm_mean": mean_pow_norm,
        "preferred_phase_rad": float(pref),
        "preferred_phase_deg": float(np.degrees(pref) % 360.0),
        "R": float(Rw),
        "weights_sum": wsum,
        "n_samples_used": int(len(th_use)),
        "cycle_pref_angles": cyc_ang,
        "n_cycles": int(ncyc),
        "rayleigh": ray,
        "gamma_band": tuple(gamma_band),
        "gate_percentile": gate_percentile,
    }


# -----------------------------
# State segment selection (mirrors your theta workflow)
# -----------------------------
def select_state_segment(rec: SyncOEpyPhotometrySession, state: str, LFP_channel: str,
                         theta_low_thres=0.0, plot_theta=False):
    """
    Returns a dataframe segment with at least:
      - LFP channel column
      - optical column (later selected)
    """
    if state == "Moving":
        df = rec.Ephys_tracking_spad_aligned
        return df[df["movement"] == "moving"]

    if state == "Rest":
        rec.pynacollada_label_theta(LFP_channel, Low_thres=theta_low_thres, High_thres=10,
                                    save=False, plot_theta=plot_theta)
        return rec.theta_part

    if state == "REM":
        rec.Label_REM_sleep(LFP_channel)
        df = rec.Ephys_tracking_spad_aligned
        return df[df["REMstate"] == "REM"]

    raise ValueError(f"Unknown state: {state}")


# -----------------------------
# Sweep-level analysis
# -----------------------------
def analyse_one_sweep(parent_dir: Path, recording_name: str, state: str, animal_id: str,
                      LFP_channel: str, opt_col: str,
                      theta_band=(5, 12), gamma_bands=None, bins=30,
                      theta_low_thres=0.0, plot_theta=False,
                      gate_percentile=75, min_samples_per_cycle=10,
                      indicator="GEVI"):
    """
    Computes slow/fast gamma power-on-theta-phase for LFP and Optical in one sweep.
    Gating uses LFP gamma power threshold (percentile) to define "high-gamma samples"
    and applies the same mask to Optical for comparability.
    """
    if gamma_bands is None:
        gamma_bands = {
            "slow_gamma": (30, 55),
            "fast_gamma": (65, 100),
        }

    rec = SyncOEpyPhotometrySession(str(parent_dir), recording_name,
                                   IsTracking=False,
                                   read_aligned_data_from_file=True,
                                   recordingMode='Atlas',
                                   indicator=indicator)

    seg = select_state_segment(rec, state, LFP_channel, theta_low_thres=theta_low_thres, plot_theta=plot_theta)
    if seg is None or len(seg) == 0:
        return []

    # choose optical column robustly
    if opt_col not in seg.columns:
        # fallback: first column containing 'zscore' or 'ref'
        candidates = [c for c in seg.columns if ("zscore" in c.lower()) or ("ref" in c.lower())]
        if len(candidates) == 0:
            raise KeyError(f"[{animal_id} | {state} | {recording_name}] opt_col '{opt_col}' not found. "
                           f"Available columns: {list(seg.columns)[:30]}")
        opt_use = candidates[0]
    else:
        opt_use = opt_col

    fs = float(rec.fs)
    tl, th = theta_band
    theta_phase = OE.calculate_theta_phase_angle(seg[LFP_channel], theta_low=tl, theta_high=th)

    lfp_sig = seg[LFP_channel].to_numpy(dtype=float)
    opt_sig = seg[opt_use].to_numpy(dtype=float)

    rows = []
    for band_name, gband in gamma_bands.items():

        # First compute LFP gamma (also gives us power for gating)
        # We'll compute gating power from LFP and reuse it for Optical.
        # Do that by computing LFP gamma power once, then calling gamma_power_on_theta for both.
        sos = _butter_bandpass_sos(gband[0], gband[1], fs, order=4)
        lfp_g = sosfiltfilt(sos, lfp_sig)
        lfp_P = _inst_power(lfp_g)

        lfp_res = gamma_power_on_theta(theta_phase, lfp_sig, fs, gband, bins=bins,
                                       gate_percentile=gate_percentile,
                                       gate_on_power=lfp_P,
                                       min_samples_per_cycle=min_samples_per_cycle)

        opt_res = gamma_power_on_theta(theta_phase, opt_sig, fs, gband, bins=bins,
                                       gate_percentile=gate_percentile,
                                       gate_on_power=lfp_P,  # same gate definition as LFP
                                       min_samples_per_cycle=min_samples_per_cycle)

        # delta preferred phase (Opt - LFP)
        delta = circ_dist(opt_res["preferred_phase_rad"], lfp_res["preferred_phase_rad"])
        delta_deg = float(np.degrees(wrap_pi(delta)))

        rows.append({
            "animal": animal_id,
            "state": state,
            "band": band_name,
            "sweep": recording_name,
            "fs": fs,
            "theta_band": str(theta_band),
            "gamma_band": str(gband),
            "gate_percentile": gate_percentile,

            "lfp_pref_rad": lfp_res["preferred_phase_rad"],
            "lfp_pref_deg": lfp_res["preferred_phase_deg"],
            "lfp_R": lfp_res["R"],
            "lfp_weights_sum": lfp_res["weights_sum"],
            "lfp_n_cycles": lfp_res["n_cycles"],
            "lfp_ray_p": lfp_res["rayleigh"]["p"],

            "opt_pref_rad": opt_res["preferred_phase_rad"],
            "opt_pref_deg": opt_res["preferred_phase_deg"],
            "opt_R": opt_res["R"],
            "opt_weights_sum": opt_res["weights_sum"],
            "opt_n_cycles": opt_res["n_cycles"],
            "opt_ray_p": opt_res["rayleigh"]["p"],

            "delta_pref_rad": float(delta),
            "delta_pref_deg": delta_deg,
        })

    return rows


# -----------------------------
# Animal-level aggregation
# -----------------------------
def summarise_animal_state_band(df_one: pd.DataFrame):
    """
    df_one: subset for (animal, state, band), multiple sweeps.
    Aggregates:
      - preferred phase via weighted vector mean across sweeps
      - Rayleigh on pooled cycle-level angles across sweeps (approximate; uses per-sweep cycle angles if stored elsewhere)
    Here we do:
      - weighted mean phase using weights = weights_sum * R (per sweep)
      - rayleigh across sweeps based on sweep-wise preferred phases is NOT ideal;
        better is pooled cycle angles, but we didn't store them in df for size reasons.
    """
    g = df_one.copy()
    if g.empty:
        return None

    # weights for sweep-level aggregation (mirrors your aggregate_gamma_power_phase idea)
    w_lfp = (g["lfp_weights_sum"].to_numpy(float) * g["lfp_R"].to_numpy(float))
    w_opt = (g["opt_weights_sum"].to_numpy(float) * g["opt_R"].to_numpy(float))

    mu_lfp, Rbar_lfp = circ_mean(g["lfp_pref_rad"].to_numpy(float), w=w_lfp)
    mu_opt, Rbar_opt = circ_mean(g["opt_pref_rad"].to_numpy(float), w=w_opt)

    # delta per sweep and aggregated delta
    delta_sweep = g["delta_pref_rad"].to_numpy(float)
    # symmetric weights for delta
    w_delta = np.sqrt(np.maximum(w_lfp, 0) * np.maximum(w_opt, 0))
    mu_delta, Rbar_delta = circ_mean(delta_sweep, w=w_delta)

    out = {
        "animal": g["animal"].iloc[0],
        "state": g["state"].iloc[0],
        "band": g["band"].iloc[0],
        "n_sweeps": int(len(g)),

        "lfp_mu_rad": mu_lfp,
        "lfp_mu_deg": float(np.degrees(mu_lfp) % 360.0) if np.isfinite(mu_lfp) else np.nan,
        "lfp_Rbar_sweep": Rbar_lfp,

        "opt_mu_rad": mu_opt,
        "opt_mu_deg": float(np.degrees(mu_opt) % 360.0) if np.isfinite(mu_opt) else np.nan,
        "opt_Rbar_sweep": Rbar_opt,

        "delta_mu_rad": mu_delta,
        "delta_mu_deg": float(np.degrees(wrap_pi(mu_delta))) if np.isfinite(mu_delta) else np.nan,
        "delta_Rbar_sweep": Rbar_delta,

        # simple summaries of within-sweep Rayleigh p-values (informative, not a formal pooled test)
        "lfp_ray_p_median": float(np.nanmedian(g["lfp_ray_p"])),
        "opt_ray_p_median": float(np.nanmedian(g["opt_ray_p"])),
        "lfp_n_cycles_total": int(np.nansum(g["lfp_n_cycles"])),
        "opt_n_cycles_total": int(np.nansum(g["opt_n_cycles"])),
    }
    return out


# -----------------------------
# Exact circular tests (n=4 => exact sign-flip feasible)
# -----------------------------
def exact_signflip_circular(deltas_rad):
    """
    Exact sign-flip test for H0: mean direction of deltas == 0.
    Statistic: resultant length of mean vector |mean(exp(i*delta))|.
    For n animals, enumerate all 2^n sign flips: delta -> ±delta.
    Returns dict with mu_delta, R, p.
    """
    d = wrap_pi(np.asarray(deltas_rad, float))
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n": 0, "mu_delta_rad": np.nan, "mu_delta_deg": np.nan, "R": np.nan, "p": np.nan}

    # observed
    z = np.mean(np.exp(1j * d))
    mu = float(np.angle(z))
    R_obs = float(np.abs(z))

    # enumerate all sign flips
    import itertools
    count = 0
    total = 0
    for flips in itertools.product([-1, 1], repeat=n):
        flips = np.asarray(flips, float)
        zz = np.mean(np.exp(1j * (d * flips)))
        R = np.abs(zz)
        if R >= R_obs - 1e-15:
            count += 1
        total += 1

    p = count / total
    return {
        "n": int(n),
        "mu_delta_rad": float(mu),
        "mu_delta_deg": float(np.degrees(wrap_pi(mu))),
        "R": float(R_obs),
        "p": float(p)
    }

def pairwise_state_diff_signflip(df_animal, band, metric_col="delta_mu_rad", s1="Rest", s2="Moving"):
    """
    For each animal, compute signed circular difference metric(s1) - metric(s2) and test via exact signflip.
    """
    animals = sorted(df_animal["animal"].unique())
    diffs = []
    for a in animals:
        g = df_animal[(df_animal["animal"] == a) & (df_animal["band"] == band)].set_index("state")
        if (s1 in g.index) and (s2 in g.index):
            diffs.append(circ_dist(g.loc[s1, metric_col], g.loc[s2, metric_col]))
    diffs = wrap_pi(np.asarray(diffs, float))
    res = exact_signflip_circular(diffs)
    res.update({"band": band, "metric": metric_col, "pair": f"{s1} - {s2}", "diffs_rad": diffs})
    return res

def omnibus_state_label_perm(df_animal, band, metric_col="delta_mu_rad", states=("Moving","Rest","REM")):
    """
    Exact within-animal label permutation across 3 states (6^n animals).
    Statistic: sum across animals of circular dispersion across the 3 state values:
      dispersion_a = 1 - R_a, where R_a = |mean(exp(i*angle_state))|
    Larger => more state-dependence (more spread).
    """
    import itertools
    dfb = df_animal[df_animal["band"] == band].copy()
    animals = sorted(dfb["animal"].unique())

    vals = {}
    for a in animals:
        g = dfb[dfb["animal"] == a].set_index("state")
        if not all(s in g.index for s in states):
            continue
        vals[a] = np.array([g.loc[s, metric_col] for s in states], float)

    animals = list(vals.keys())
    if len(animals) == 0:
        return {"band": band, "metric": metric_col, "stat_obs": np.nan, "p": np.nan, "n_animals": 0}

    def dispersion(x3):
        z = np.mean(np.exp(1j*np.asarray(x3)))
        R = np.abs(z)
        return 1.0 - R

    def stat(assign_dict):
        T = 0.0
        for a in animals:
            T += dispersion(assign_dict[a])
        return float(T)

    # observed
    obs_assign = {a: vals[a].copy() for a in animals}
    T_obs = stat(obs_assign)

    perms = list(itertools.permutations([0,1,2], 3))  # 6 perms
    count = 0
    total = 0
    for choices in itertools.product(range(6), repeat=len(animals)):
        assign = {}
        for ai, c in enumerate(choices):
            a = animals[ai]
            perm = perms[c]
            assign[a] = vals[a][list(perm)]
        T = stat(assign)
        if T >= T_obs - 1e-15:
            count += 1
        total += 1

    return {"band": band, "metric": metric_col, "stat_obs": float(T_obs),
            "p": float(count/total), "n_animals": int(len(animals))}


# -----------------------------
# Main runner across animals
# -----------------------------
def run_theta_gamma_across_animals(ANIMALS,
                                  theta_band=(5,12),
                                  gamma_bands=None,
                                  bins=30,
                                  theta_low_thres=0.0,
                                  gate_percentile=75,
                                  min_samples_per_cycle=10,
                                  plot_theta=False):
    """
    Returns:
      df_sweep, df_animal, stats (dict)
    """
    if gamma_bands is None:
        gamma_bands = {
            "slow_gamma": (30, 55),
            "fast_gamma": (65, 100),
        }

    all_rows = []
    for a in ANIMALS:
        animal_id = a["animal_id"]
        LFP_channel = a["lfp_channel"]
        opt_col = a.get("opt_col", "zscore_raw")

        state_dirs = {
            "Moving": Path(a["loco_parent"]),
            "Rest":   Path(a["stat_parent"]),
            "REM":    Path(a["rem_parent"]),
        }

        for state, parent in state_dirs.items():
            for rec_name in _iter_sync_recordings(parent):
                rows = analyse_one_sweep(parent, rec_name, state, animal_id,
                                        LFP_channel=LFP_channel,
                                        opt_col=opt_col,
                                        theta_band=theta_band,
                                        gamma_bands=gamma_bands,
                                        bins=bins,
                                        theta_low_thres=theta_low_thres,
                                        plot_theta=plot_theta,
                                        gate_percentile=gate_percentile,
                                        min_samples_per_cycle=min_samples_per_cycle,
                                        indicator="GEVI")
                all_rows.extend(rows)

    df_sweep = pd.DataFrame(all_rows)
    if df_sweep.empty:
        raise RuntimeError("No sweep-level results produced. Check paths, SyncRecording folders, and state labelling.")

    # Animal-level aggregation
    animal_rows = []
    for (animal, state, band), g in df_sweep.groupby(["animal","state","band"]):
        r = summarise_animal_state_band(g)
        if r is not None:
            animal_rows.append(r)
    df_animal = pd.DataFrame(animal_rows)

    # ---------- Stats ----------
    stats = {"paired_LFP_vs_Opt_within_state": [], "between_states_delta": [], "omnibus_delta": []}

    for band in gamma_bands.keys():
        # paired LFP vs Opt within each state: use delta_mu_rad per animal-state-band
        for state in ["Moving", "Rest", "REM"]:
            d = df_animal[(df_animal["band"] == band) & (df_animal["state"] == state)]["delta_mu_rad"].to_numpy(float)
            res = exact_signflip_circular(d)
            res.update({"band": band, "state": state, "metric": "delta_mu_rad (Opt-LFP)"})
            stats["paired_LFP_vs_Opt_within_state"].append(res)

        # between-state comparisons on delta
        pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
        for s1, s2 in pairs:
            res = pairwise_state_diff_signflip(df_animal, band, metric_col="delta_mu_rad", s1=s1, s2=s2)
            stats["between_states_delta"].append(res)

        # omnibus across 3 states on delta
        stats["omnibus_delta"].append(omnibus_state_label_perm(df_animal, band, metric_col="delta_mu_rad",
                                                               states=("Moving","Rest","REM")))

    return df_sweep, df_animal, stats


# -----------------------------
# Example call (uses your ANIMALS list)
# -----------------------------
'Multiple animal analysis'
ANIMALS = [ {"animal_id": "1765508", 
             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ALocomotion",
             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\AwakeStationary", 
             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ASleepREM", 
             "lfp_channel": "LFP_1"}, 
           {"animal_id": "1844609", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ASleepREM", 
            "lfp_channel": "LFP_1"}, 
           {"animal_id": "1881363", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ALocomotion",
            "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\AwakeStationary",
            "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ASleepREM", 
            "lfp_channel": "LFP_1"}, 
           {"animal_id": "1887933", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ASleepREM", 
            "lfp_channel": "LFP_2", "opt_col": "ref_raw"},]


df_sweep, df_animal, stats = run_theta_gamma_across_animals(
    ANIMALS,
    theta_band=(5,12),
    gamma_bands={"slow_gamma": (30,55), "fast_gamma": (65,100)},
    bins=30,
    theta_low_thres=-0.5,
    gate_percentile=75,          # "high gamma" = top 25% of LFP gamma power within each sweep
    min_samples_per_cycle=10,
    plot_theta=False
)

print(df_sweep.head())
print(df_animal.sort_values(["band","animal","state"]))
print("\nPaired LFP vs Opt within state:")
for r in stats["paired_LFP_vs_Opt_within_state"]:
    print(r)
print("\nBetween-state delta comparisons:")
for r in stats["between_states_delta"]:
    print(r)
print("\nOmnibus delta across 3 states:")
for r in stats["omnibus_delta"]:
    print(r)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Helpers
# --------------------------
def wrap_pm180(deg):
    """Wrap degrees to (-180, 180]."""
    d = ((deg + 180.0) % 360.0) - 180.0
    return 180.0 if np.isclose(d, -180.0) else d

def _p_to_text(p):
    if p < 1e-3: return "p<0.001"
    return f"p={p:.3g}"

def _format_pair_label(pair):
    # input like "Rest - Moving"
    return pair.replace(" - ", " vs ")

def add_sig_bracket(ax, x1, x2, y, text, h=2.5, lw=1.6, fs=12):
    """Draw a bracket with text at height y."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=lw, c="k", clip_on=False)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=fs, clip_on=False)

def add_text_above(ax, x, y, text, fs=12):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, clip_on=False)

def _prep_states(df, states=("Moving","Rest","REM")):
    df = df.copy()
    df["state"] = pd.Categorical(df["state"], categories=states, ordered=True)
    return df.sort_values(["band","animal","state"])

# --------------------------
# Main plotting function
# --------------------------

def plot_delta_mu_by_state_clean(
    df_animal,
    paired_within_state_stats,
    between_state_stats,
    states=("Moving", "Rest", "REM"),
    bands=("slow_gamma", "fast_gamma"),
    ylim=None,
    title="Theta–gamma coupling: Opt–LFP preferred θ phase offset (Δμ)",
    figsize=(12, 5.4),
    fontsize=14,
    show_p="significant_only",   # "all" | "significant_only" | "none"
    alpha_sig=0.05,
    legend_loc="bottom",         # "bottom" | "right"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def _prep_states(df, states=("Moving","Rest","REM")):
        df = df.copy()
        df["state"] = pd.Categorical(df["state"], categories=states, ordered=True)
        return df.sort_values(["band","animal","state"])

    def _p_to_text(p):
        if p is None or (not np.isfinite(p)):
            return "p=nan"
        if p < 1e-3:
            return "p<0.001"
        return f"p={p:.3g}"

    def _maybe_keep(p):
        if show_p == "none":
            return False
        if show_p == "all":
            return True
        # significant_only
        return (p is not None) and np.isfinite(p) and (p < alpha_sig)

    df = _prep_states(df_animal, states=states)

    # lookups
    within_lookup = {(d["band"], d["state"]): d for d in paired_within_state_stats}
    between_lookup = {(d["band"], d["pair"]): d for d in between_state_stats}

    x = np.arange(len(states))

    # y-lims
    if ylim is None:
        y_all = df["delta_mu_deg"].dropna().to_numpy(float)
        if y_all.size == 0:
            ylim = (-30, 30)
        else:
            y_min, y_max = np.min(y_all), np.max(y_all)
            # keep symmetric if range is large (helps readability for wrapped angles)
            span = max(abs(y_min), abs(y_max))
            ylim = (-1.15 * span, 1.15 * span)

    fig, axes = plt.subplots(1, len(bands), figsize=figsize, sharey=True)
    if len(bands) == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None

    for ax, band in zip(axes, bands):
        sub = df[df["band"] == band].copy()

        ax.set_title(band.replace("_", " ").title(), fontsize=fontsize+2, pad=10)

        # per-animal lines
        for animal, g in sub.groupby("animal"):
            g = g.set_index("state").reindex(states)
            y = g["delta_mu_deg"].to_numpy(float)
            line = ax.plot(x, y, marker="o", linewidth=1.8, alpha=0.9, label=str(animal))[0]

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # group mean ± SEM
        means = sub.groupby("state")["delta_mu_deg"].mean().reindex(states).to_numpy(float)
        sems  = sub.groupby("state")["delta_mu_deg"].sem().reindex(states).to_numpy(float)
        ax.errorbar(x, means, yerr=sems, fmt="-s", linewidth=2.2, capsize=4)

        ax.axhline(0, lw=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=fontsize)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=fontsize)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---- compact p-value textbox (no brackets) ----
        # Within-state (Δμ ≠ 0)
        within_parts = []
        for st in states:
            key = (band, st)
            if key in within_lookup:
                p = within_lookup[key]["p"]
                if _maybe_keep(p):
                    within_parts.append(f"{st}:{_p_to_text(p)}")
        # Between-state
        between_parts = []
        for pair in ["Rest - Moving", "REM - Moving", "REM - Rest"]:
            key = (band, pair)
            if key in between_lookup:
                p = between_lookup[key]["p"]
                if _maybe_keep(p):
                    short = pair.replace(" - ", "–")
                    between_parts.append(f"{short}:{_p_to_text(p)}")


        lines = []
        if show_p != "none":
            if len(within_parts) > 0:
                lines.append("Within-state: " + ", ".join(within_parts))
            if len(between_parts) > 0:
                lines.append("Between-state: " + ", ".join(between_parts))
            # If we are hiding non-sig values and nothing remains, state that explicitly
            if (show_p == "significant_only") and (len(lines) == 0):
                lines = [f"All tests ns (α={alpha_sig:g})"]

        if len(lines) > 0:
            ax.text(
                0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=fontsize-3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
            )

        ax.set_xlabel("State", fontsize=fontsize)

    axes[0].set_ylabel("Δμ (Opt − LFP) preferred θ phase (deg)", fontsize=fontsize)

    # ---- legend placement ----
    if legend_loc == "bottom":
        fig.legend(
            legend_handles, legend_labels, title="Animal",
            loc="lower center", bbox_to_anchor=(0.5, -0.02),
            ncol=len(legend_labels), frameon=False,
            fontsize=fontsize-2, title_fontsize=fontsize-1
        )
        fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.15)
    else:
        fig.legend(
            legend_handles, legend_labels, title="Animal",
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=fontsize-2, title_fontsize=fontsize-1
        )
        fig.subplots_adjust(right=0.82, top=0.86, wspace=0.15)

    fig.suptitle(title, fontsize=fontsize+4, y=0.97)
    return fig, axes



# --------------------------
# Call the plotting function
# --------------------------

# Ensure degrees column exists (your summariser already creates delta_mu_deg, but keep this guard)
if "delta_mu_deg" not in df_animal.columns:
    df_animal["delta_mu_deg"] = df_animal["delta_mu_rad"].apply(lambda r: wrap_pm180(np.degrees(r)))

paired_within = stats["paired_LFP_vs_Opt_within_state"]
between_state = stats["between_states_delta"]

fig, axes = plot_delta_mu_by_state_clean(
    df_animal=df_animal,
    paired_within_state_stats=paired_within,
    between_state_stats=between_state,
    show_p="significant_only",  # hides all those p=0.375/0.875 labels
    alpha_sig=0.05,
    legend_loc="bottom",
    figsize=(12, 5.4),
    fontsize=16
)
fig.subplots_adjust(top=0.80)  # more space for the shared legend + title + brackets

out_png = r"G:\2025_ATLAS_SPAD\AcrossAnimal\ThetaGamma_deltaMu_byState_clean.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print("Saved:", out_png)

#%%
def compute_within_state_lfp_vs_opt_stats(df_animal, bands=("slow_gamma","fast_gamma"), states=("Moving","Rest","REM")):
    """
    Tests, within each state and band, whether Opt and LFP preferred θ phases differ across animals.
    Uses exact sign-flip test on per-animal delta_mu_rad (= Opt - LFP, aggregated within animal/state/band).
    Returns a dict lookup and a tidy DataFrame.
    """
    rows = []
    for band in bands:
        for st in states:
            d = df_animal[(df_animal["band"] == band) & (df_animal["state"] == st)]["delta_mu_rad"].to_numpy(float)
            res = exact_signflip_circular(d)
            res.update({"band": band, "state": st, "metric": "Opt-LFP (delta_mu_rad)"})
            rows.append(res)

    stats_df = pd.DataFrame(rows)
    lookup = {(r["band"], r["state"]): r for r in rows}
    return lookup, stats_df

from matplotlib.lines import Line2D

def plot_lfp_vs_opt_pref_theta_phase(
    df_animal,
    bands=("slow_gamma","fast_gamma"),
    states=("Moving","Rest","REM"),
    ylim=(-180, 180),
    figsize=(12.5, 4.8),
    fontsize=14
):
    """
    Plots LFP and GEVI preferred theta phase (animal-level mu) across states.
    Same animal = same colour; LFP solid, GEVI dashed.
    Adds within-state paired circular stats (Opt-LFP != 0) in a compact text box.
    """

    # Wrapped degrees for display
    tmp = df_animal.copy()
    tmp["lfp_mu_deg_wrapped"] = tmp["lfp_mu_deg"].apply(wrap_pm180)
    tmp["opt_mu_deg_wrapped"] = tmp["opt_mu_deg"].apply(wrap_pm180)

    # Stats lookup (within each state/band)
    within_lookup, within_df = compute_within_state_lfp_vs_opt_stats(tmp, bands=bands, states=states)

    # Colour map by animal (stable order)
    animals = sorted(tmp["animal"].unique())
    base_cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {a: base_cols[i % len(base_cols)] for i, a in enumerate(animals)}

    x = np.arange(len(states))

    fig, axes = plt.subplots(1, len(bands), figsize=figsize, sharey=True)
    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        sub = tmp[tmp["band"] == band].copy()
        sub["state"] = pd.Categorical(sub["state"], categories=states, ordered=True)

        # Plot per animal: LFP solid, Opt dashed, same colour
        for animal in animals:
            g = sub[sub["animal"] == animal].set_index("state").reindex(states)
            col = color_map[animal]

            ax.plot(x, g["lfp_mu_deg_wrapped"].to_numpy(float),
                    linestyle="-", marker="o", linewidth=2.0, alpha=0.9, color=col)

            ax.plot(x, g["opt_mu_deg_wrapped"].to_numpy(float),
                    linestyle="--", marker="s", linewidth=2.0, alpha=0.9, color=col)

        # Formatting
        ax.set_title(band.replace("_", " ").title(), fontsize=fontsize+2, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(states, fontsize=fontsize)
        ax.axhline(0, lw=1, alpha=0.5)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Compact within-state p-values (Opt vs LFP)
        lines = []
        for st in states:
            r = within_lookup.get((band, st), None)
            if r is None:
                continue
            lines.append(f"{st}: p={r['p']:.3g}, Δμ={r['mu_delta_deg']:+.1f}°")
        txt = "LFP vs GEVI (Opt–LFP)\n" + "\n".join(lines)

        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize-3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
        )

    axes[0].set_ylabel("Preferred θ phase (deg, wrapped)", fontsize=fontsize)

    # Two-part legend: modality + animal colours
    modality_handles = [
        Line2D([0],[0], color="k", lw=2, ls="-", marker="o", label="LFP (μ)"),
        Line2D([0],[0], color="k", lw=2, ls="--", marker="s", label="GEVI (μ)"),
    ]
    animal_handles = [Line2D([0],[0], color=color_map[a], lw=3, label=a) for a in animals]

    # Put legends cleanly under the plot
    fig.legend(handles=modality_handles, loc="lower center",
               bbox_to_anchor=(0.33, -0.02), ncol=2, frameon=False,
               fontsize=fontsize-2)
    fig.legend(handles=animal_handles, title="Animal", loc="lower center",
               bbox_to_anchor=(0.77, -0.02), ncol=len(animals), frameon=False,
               fontsize=fontsize-2, title_fontsize=fontsize-2)

    fig.suptitle("Preferred θ phase for gamma power (animal-level μ): LFP vs GEVI", fontsize=fontsize+4, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.18)

    return fig, axes, within_df

fig, axes, within_df = plot_lfp_vs_opt_pref_theta_phase(
    df_animal,
    bands=("slow_gamma","fast_gamma"),
    states=("Moving","Rest","REM"),
    ylim=(-180, 180),
    fontsize=16
)
plt.show()

print(within_df.sort_values(["band","state"]))
