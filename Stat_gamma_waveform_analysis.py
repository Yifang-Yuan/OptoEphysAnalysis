# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 10:23:12 2026

@author: yifan
"""
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import hilbert
from scipy.stats import sem
from pathlib import Path
from SyncOECPySessionClass import SyncOEpyPhotometrySession
from pathlib import Path
import itertools
# -------------------------
# Core helpers (reuse style)
# -------------------------
def bandpass_filter(x, fs, low, high, order=4):
    sos = signal.butter(order, [low, high], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)

def _phase(x): return np.angle(hilbert(x))
def _envelope(x): return np.abs(hilbert(x))
def _wrap_pi(x): return (np.asarray(x) + np.pi) % (2*np.pi) - np.pi
def _circmean(ang): return float(np.angle(np.mean(np.exp(1j*np.asarray(ang)))))
def _circR(ang): return float(np.abs(np.mean(np.exp(1j*np.asarray(ang)))))

# -------------------------
# Circular utilities
# -------------------------
def wrap_pi(x):
    x = np.asarray(x)
    return (x + np.pi) % (2*np.pi) - np.pi

def circmean(ang):
    ang = np.asarray(ang)
    return float(np.angle(np.mean(np.exp(1j * ang))))

def circR(ang):
    ang = np.asarray(ang)
    return float(np.abs(np.mean(np.exp(1j * ang))))

def vector_mean_angles(angles_rad):
    """Vector-mean a set of angles (e.g., sweep-wise circular means)."""
    z = np.mean(np.exp(1j * np.asarray(angles_rad)))
    return float(np.angle(z)), float(np.abs(z))

def circ_diff(a, b):
    """Signed circular difference a-b in [-pi, pi)."""
    return float(np.angle(np.exp(1j*(a - b))))

def cross_correlation_band(x, y, fs, max_lag_s=0.05):
    corr = signal.correlate(x, y, mode='full')
    lags = signal.correlation_lags(len(x), len(y), mode='full') / fs
    corr = corr / (np.std(x) * np.std(y) * len(x))
    m = (lags >= -max_lag_s) & (lags <= max_lag_s)
    return lags[m], corr[m]

def estimate_peak_lag_from_xcorr(lags_s, corr, win=(-0.02, 0.02)):
    lags_s = np.asarray(lags_s); corr = np.asarray(corr)
    m = (lags_s >= win[0]) & (lags_s <= win[1])
    l, c = lags_s[m], corr[m]
    i = int(np.argmax(c))
    if i == 0 or i == len(c)-1:
        return float(l[i]), float(c[i])
    y1, y2, y3 = c[i-1], c[i], c[i+1]
    x1, x2, x3 = l[i-1], l[i], l[i+1]
    h = x2 - x1
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return float(x2), float(y2)
    delta = 0.5 * (y1 - y3) / denom
    return float(x2 + delta*h), float(y2 - 0.25*(y1 - y3)*delta)

def phase_locking_value(x, y):
    dphi = _wrap_pi(_phase(x) - _phase(y))
    return float(np.abs(np.mean(np.exp(1j*dphi))))

def amplitude_correlation(x, y):
    return float(np.corrcoef(_envelope(x), _envelope(y))[0, 1])

# -------------------------
# Rayleigh test (phase-lock significance)
#   H0: angles uniform on circle
#   Input angles ideally "one per cycle" (less autocorrelation)
# -------------------------
def rayleigh_test(angles_rad):
    ang = _wrap_pi(np.asarray(angles_rad, float))
    ang = ang[np.isfinite(ang)]
    n = len(ang)
    if n < 10:
        return {"n": int(n), "R": np.nan, "z": np.nan, "p": np.nan}

    Rbar = _circR(ang)              # resultant length of mean vector
    z = n * (Rbar ** 2)

    # CircStat (Berens 2009) style approximation (good for n>=10)
    p = np.exp(-z) * (
        1
        + (2*z - z**2) / (4*n)
        - (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288 * (n**2))
    )
    p = float(np.clip(p, 0.0, 1.0))
    return {"n": int(n), "R": float(Rbar), "z": float(z), "p": float(p)}

# -------------------------
# Your existing iterators / state selection
# -------------------------
def _iter_sync_recordings(root: Path):
    for p in sorted(Path(root).glob("SyncRecording*")):
        if p.is_dir():
            yield p.name

def _select_state_segment(rec, behaviour: str, LFP_channel: str, theta_low_thres: float):
    if behaviour == "Moving":
        df = rec.Ephys_tracking_spad_aligned
        return df[df["movement"] == "moving"]
    if behaviour == "Rest":
        rec.pynacollada_label_theta(LFP_channel, Low_thres=theta_low_thres, High_thres=10,
                                    save=False, plot_theta=False)
        return rec.theta_part
    if behaviour == "REM":
        rec.Label_REM_sleep(LFP_channel)
        df = rec.Ephys_tracking_spad_aligned
        return df[df["REMstate"] == "REM"]
    raise ValueError("Unknown behaviour")

# -------------------------
# Gamma-specific: gating and cycle-wise phase shift
# -------------------------
def _mask_to_segments(mask: np.ndarray, min_len: int):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    segs = []
    for a, b in zip(starts, ends):
        if (b - a) >= min_len:
            segs.append((a, b))
    return segs

def detect_lfp_troughs_band(lfp_band, fs, f_high, safety=0.8):
    min_period_s = safety * (1.0 / float(f_high))
    distance = max(1, int(min_period_s * fs))
    trough_idx, _ = signal.find_peaks(-lfp_band, distance=distance)
    return trough_idx

def resample_cycle(x, n=200):
    old = np.linspace(0, 1, len(x), endpoint=True)
    new = np.linspace(0, 1, n, endpoint=True)
    return np.interp(new, old, x)

def cycle_analysis_band(lfp_band, opt_band, fs, f_high, n_resample=200, cycle_normalize="zscore"):
    troughs = detect_lfp_troughs_band(lfp_band, fs, f_high=f_high, safety=0.8)
    troughs = troughs[(troughs > 0) & (troughs < len(lfp_band)-1)]
    if len(troughs) < 3:
        return dict(cycle_phase_shift=np.array([]), inst_freq_hz=np.array([]))

    phi_l, phi_o = _phase(lfp_band), _phase(opt_band)
    phase_shift, instf = [], []

    for i in range(len(troughs)-1):
        a, b = troughs[i], troughs[i+1]
        if b <= a + 2:
            continue

        instf.append(1.0 / ((b-a)/fs))
        dphi = _wrap_pi(phi_l[a:b] - phi_o[a:b])
        phase_shift.append(_circmean(dphi))

    return {
        "cycle_phase_shift": np.asarray(phase_shift),
        "inst_freq_hz": np.asarray(instf),
    }

# -------------------------
# Sweep-wise gamma stats (one row per SyncRecording)
# -------------------------
def analyse_state_sweepwise_gamma(root_dir: Path, behaviour: str, animal_id: str,
                                 LFP_channel: str, fs=10000,
                                 gamma_band=(30, 55), band_name="slow_gamma",
                                 theta_low_thres=0.0,
                                 indicator="GEVI", opt_col="zscore_raw",
                                 gate_quantile=0.7, gate_on="lfp",
                                 min_epoch_s=0.10,
                                 max_lag_s=0.05, xcorr_win=(-0.02, 0.02)):
    rows = []
    phase_pool = []  # pooled cycle-wise phase shifts across sweeps (for animal-level significance)

    root_dir = Path(root_dir)
    f_low, f_high = gamma_band

    for rec_name in _iter_sync_recordings(root_dir):
        rec = SyncOEpyPhotometrySession(str(root_dir), rec_name,
                                        IsTracking=False,
                                        read_aligned_data_from_file=True,
                                        recordingMode='Atlas', indicator=indicator)

        seg = _select_state_segment(rec, behaviour, LFP_channel, theta_low_thres)
        if seg is None or len(seg) == 0:
            continue

        if opt_col not in seg.columns:
            candidates = [c for c in seg.columns if "zscore" in c.lower() and "raw" in c.lower()]
            if len(candidates) == 0:
                raise KeyError(f"[{animal_id}|{behaviour}|{rec_name}] No '{opt_col}' column. "
                               f"Columns: {list(seg.columns)[:30]}")
            opt_use = candidates[0]
        else:
            opt_use = opt_col

        lfp_raw = seg[LFP_channel].to_numpy()
        opt_raw = seg[opt_use].to_numpy()

        # gamma band
        lfp_g = bandpass_filter(lfp_raw, fs, f_low, f_high, order=4)
        opt_g = bandpass_filter(opt_raw, fs, f_low, f_high, order=4)

        env_l = _envelope(lfp_g)
        env_o = _envelope(opt_g)

        th_l = np.quantile(env_l, gate_quantile)
        mask = env_l > th_l
        if gate_on == "both":
            th_o = np.quantile(env_o, gate_quantile)
            mask = mask & (env_o > th_o)

        segs = _mask_to_segments(mask, min_len=int(min_epoch_s * fs))
        if len(segs) == 0:
            continue

        # concatenate gated epochs for robust metrics
        lfp_cat, opt_cat = [], []
        env_l_cat, env_o_cat = [], []
        ph_all = []
        instf_all = []

        for a, b in segs:
            lfp_e = lfp_g[a:b]
            opt_e = opt_g[a:b]
            lfp_cat.append(lfp_e); opt_cat.append(opt_e)
            env_l_cat.append(env_l[a:b]); env_o_cat.append(env_o[a:b])

            cyc = cycle_analysis_band(lfp_e, opt_e, fs, f_high=f_high, n_resample=200)
            ph = cyc.get("cycle_phase_shift", np.array([]))
            if len(ph):
                ph_all.append(ph)
            instf = cyc.get("inst_freq_hz", np.array([]))
            if len(instf):
                instf_all.append(instf)

        if len(lfp_cat) == 0:
            continue

        lfp_cat = np.concatenate(lfp_cat)
        opt_cat = np.concatenate(opt_cat)
        env_l_cat = np.concatenate(env_l_cat)
        env_o_cat = np.concatenate(env_o_cat)

        # linear metrics (gamma band)
        lags, xcorr = cross_correlation_band(lfp_cat, opt_cat, fs, max_lag_s=max_lag_s)
        lag_peak_s, corr_peak = estimate_peak_lag_from_xcorr(lags, xcorr, win=xcorr_win)
        plv = phase_locking_value(lfp_cat, opt_cat)
        env_r = float(np.corrcoef(env_l_cat, env_o_cat)[0, 1])

        # envelope lag (often more stable than raw gamma xcorr)
        lags_e, xcorr_e = cross_correlation_band(env_l_cat - env_l_cat.mean(),
                                                 env_o_cat - env_o_cat.mean(),
                                                 fs, max_lag_s=0.5)
        env_lag_peak_s, env_corr_peak = estimate_peak_lag_from_xcorr(lags_e, xcorr_e, win=(-0.25, 0.25))

        # cycle-wise phase shifts (one per gamma cycle, gated)
        if len(ph_all):
            ph_all = _wrap_pi(np.concatenate(ph_all))
        else:
            ph_all = np.array([])

        # pool for animal-level tests
        if len(ph_all):
            phase_pool.append(ph_all)

        # sweep-level circular summary + Rayleigh significance
        mu = np.nan
        R = np.nan
        ray_p = np.nan
        lag_from_phase_ms = np.nan
        f_gamma = np.nan

        if len(ph_all) >= 10:
            mu = _circmean(ph_all)
            R = _circR(ph_all)
            ray = rayleigh_test(ph_all)
            ray_p = ray["p"]

            if len(instf_all):
                f_gamma = float(np.nanmean(np.concatenate(instf_all)))
            else:
                f_gamma = float(0.5 * (f_low + f_high))

            if np.isfinite(f_gamma) and f_gamma > 0:
                lag_from_phase_ms = (mu / (2*np.pi)) * (1000.0 / f_gamma)

        gated_fraction = len(lfp_cat) / max(1, len(lfp_raw))

        rows.append({
            "animal": animal_id,
            "state": behaviour,
            "sweep": rec_name,
            "band": band_name,
            "band_low": float(f_low),
            "band_high": float(f_high),
            "lfp_channel": LFP_channel,
            "n_epochs": int(len(segs)),
            "gated_fraction": float(gated_fraction),

            "xcorr_peak_lag_ms": 1000.0 * float(lag_peak_s),
            "xcorr_peak_corr": float(corr_peak),
            "PLV": float(plv),
            "env_r": float(env_r),

            "env_xcorr_peak_lag_ms": 1000.0 * float(env_lag_peak_s),
            "env_xcorr_peak_corr": float(env_corr_peak),

            "n_cycles": int(len(ph_all)),
            "gamma_freq_hz": float(f_gamma) if np.isfinite(f_gamma) else np.nan,
            "mu_phase_rad": float(mu) if np.isfinite(mu) else np.nan,
            "R_phase": float(R) if np.isfinite(R) else np.nan,
            "rayleigh_p": float(ray_p) if np.isfinite(ray_p) else np.nan,
            "lag_from_phase_ms": float(lag_from_phase_ms) if np.isfinite(lag_from_phase_ms) else np.nan,
        })

    df = pd.DataFrame(rows)

    # pooled angles across sweeps (for animal-level Rayleigh)
    if len(phase_pool):
        phase_pool = _wrap_pi(np.concatenate(phase_pool))
    else:
        phase_pool = np.array([])

    return df, phase_pool

# -------------------------
# Animal-level summary (one row per animal × state × band)
#   - linear metrics averaged across sweeps
#   - phase stats computed from pooled cycles across sweeps (rayleigh_p_pool)
# -------------------------
def summarise_animal_state_gamma(df_sweeps_one_animal_state: pd.DataFrame, pooled_angles_rad: np.ndarray):
    if df_sweeps_one_animal_state.empty:
        return None

    g = df_sweeps_one_animal_state

    lin_cols = [
        "xcorr_peak_lag_ms", "PLV", "env_r",
        "env_xcorr_peak_lag_ms", "gated_fraction",
        "lag_from_phase_ms", "gamma_freq_hz"
    ]
    out = {c: float(g[c].mean()) for c in lin_cols if c in g.columns}

    out.update({
        "animal": g["animal"].iloc[0],
        "state": g["state"].iloc[0],
        "band": g["band"].iloc[0],
        "band_low": float(g["band_low"].iloc[0]),
        "band_high": float(g["band_high"].iloc[0]),
        "n_sweeps": int(len(g)),
        "n_cycles_total": int(g["n_cycles"].sum()),
        "n_epochs_total": int(g["n_epochs"].sum()),
    })

    # pooled cycle-wise angles across sweeps (animal-level phase-lock + mean direction)
    pooled_angles_rad = _wrap_pi(np.asarray(pooled_angles_rad, float))
    pooled_angles_rad = pooled_angles_rad[np.isfinite(pooled_angles_rad)]

    if len(pooled_angles_rad) >= 10:
        out["mu_phase_rad"] = float(_circmean(pooled_angles_rad))
        out["R_phase"] = float(_circR(pooled_angles_rad))
        ray = rayleigh_test(pooled_angles_rad)
        out["rayleigh_p_pool"] = float(ray["p"])
        out["rayleigh_z_pool"] = float(ray["z"])
        out["n_cycles_pool"] = int(ray["n"])
    else:
        out["mu_phase_rad"] = np.nan
        out["R_phase"] = np.nan
        out["rayleigh_p_pool"] = np.nan
        out["rayleigh_z_pool"] = np.nan
        out["n_cycles_pool"] = int(len(pooled_angles_rad))

    return out

# -------------------------
# Across animals runner (matches your theta runner pattern)
# -------------------------
def run_across_animals_gamma(ANIMALS, fs=10000,
                            gamma_bands=None,
                            theta_low_thres=0.0,
                            indicator="GEVI",
                            opt_col_default="zscore_raw",
                            gate_quantile=0.7, gate_on="lfp",
                            min_epoch_s=0.10):
    """
    Returns:
      df_sweep_gamma: rows per sweep
      df_animal_gamma: rows per animal×state×band, includes rayleigh_p_pool
    """
    if gamma_bands is None:
        gamma_bands = {
            "slow_gamma": (30, 55),
            "fast_gamma": (65, 100),
        }

    all_sweeps = []
    pool_map = {}  # (animal, state, band) -> pooled angles

    for a in ANIMALS:
        animal_id = a["animal_id"]
        LFP_channel = a["lfp_channel"]
        opt_use = a.get("opt_col", opt_col_default)

        state_dirs = {
            "Moving": Path(a["loco_parent"]),
            "Rest":   Path(a["stat_parent"]),
            "REM":    Path(a["rem_parent"]),
        }

        for band_name, band in gamma_bands.items():
            for state, p in state_dirs.items():
                df_state, ph_pool = analyse_state_sweepwise_gamma(
                    p, state, animal_id,
                    LFP_channel=LFP_channel,
                    fs=fs,
                    gamma_band=band,
                    band_name=band_name,
                    theta_low_thres=theta_low_thres,
                    indicator=indicator,
                    opt_col=opt_use,
                    gate_quantile=gate_quantile,
                    gate_on=gate_on,
                    min_epoch_s=min_epoch_s
                )
                all_sweeps.append(df_state)
                pool_map[(animal_id, state, band_name)] = ph_pool

    df_sweep = pd.concat(all_sweeps, ignore_index=True) if all_sweeps else pd.DataFrame()

    # animal-level rows
    rows = []
    for (animal, state, band), g in df_sweep.groupby(["animal", "state", "band"]):
        ph_pool = pool_map.get((animal, state, band), np.array([]))
        r = summarise_animal_state_gamma(g, ph_pool)
        if r is not None:
            rows.append(r)
    df_animal = pd.DataFrame(rows)

    return df_sweep, df_animal

def exact_signflip_pvalue(values, stat_fn, center_null=True):
    """
    Exact sign-flip test (2^n).
    values: array of within-animal contrasts (e.g., Rest-Moving per animal).
    stat_fn: maps array -> scalar statistic (larger = more extreme).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan

    obs = stat_fn(v)
    # enumerate all sign flips
    count = 0
    total = 0
    for flips in itertools.product([-1, 1], repeat=n):
        flips = np.asarray(flips, float)
        vv = v * flips
        T = stat_fn(vv)
        if T >= obs - 1e-15:
            count += 1
        total += 1
    p = count / total
    return float(obs), float(p)
def exact_within_animal_label_perm_3states(df_animal, metric_col, states=("Moving","Rest","REM")):
    """
    Omnibus test: within each animal, permute state labels (6^n animals, exact).
    Statistic: sum across animals of variance across the 3 state values.
    """
    # build animal-> vector of values in fixed state order
    animals = sorted(df_animal["animal"].unique())
    vals = {}
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if not all(s in g.index for s in states):
            continue
        vals[a] = np.array([g.loc[s, metric_col] for s in states], float)

    animals = list(vals.keys())
    if len(animals) == 0:
        return {"metric": metric_col, "stat_obs": np.nan, "p": np.nan, "n_animals": 0}

    def stat_from_assign(assign_dict):
        # assign_dict[a] is a length-3 array in (Moving,Rest,REM) order after relabelling
        T = 0.0
        for a in animals:
            x = assign_dict[a]
            if np.any(~np.isfinite(x)):
                continue
            T += float(np.var(x, ddof=0))
        return T

    # observed
    obs_assign = {a: vals[a].copy() for a in animals}
    T_obs = stat_from_assign(obs_assign)

    # enumerate 6^n permutations of labels per animal
    perms = list(itertools.permutations([0,1,2], 3))  # 6 permutations
    count = 0
    total = 0
    for choices in itertools.product(range(6), repeat=len(animals)):
        assign = {}
        for ai, c in enumerate(choices):
            a = animals[ai]
            perm = perms[c]
            assign[a] = vals[a][list(perm)]
        T = stat_from_assign(assign)
        if T >= T_obs - 1e-15:
            count += 1
        total += 1

    return {"metric": metric_col, "stat_obs": float(T_obs), "p": float(count/total), "n_animals": len(animals)}

def pairwise_linear_signflip(df_animal, metric_col, s1="Rest", s2="Moving"):
    """
    Pairwise within-animal test: contrast = value(s1)-value(s2), exact sign-flip.
    Statistic: abs(mean(contrast)).
    """
    animals = sorted(df_animal["animal"].unique())
    diffs = []
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if s1 in g.index and s2 in g.index:
            diffs.append(float(g.loc[s1, metric_col] - g.loc[s2, metric_col]))
    diffs = np.asarray(diffs, float)

    stat_fn = lambda x: float(np.abs(np.mean(x)))
    obs, p = exact_signflip_pvalue(diffs, stat_fn)
    return {"metric": metric_col, "pair": f"{s1} - {s2}", "diffs": diffs, "stat_obs": obs, "p": p, "n": int(np.isfinite(diffs).sum())}

def pairwise_circular_signflip(df_animal, s1="Rest", s2="Moving"):
    """
    Pairwise within-animal circular test on mean directions.
    For each animal: delta = wrap(mu_s1 - mu_s2).
    Null: labels exchangeable => delta sign-flips.
    Statistic: 1 - Rbar, equivalently use Rbar as extremeness (larger = more extreme).
    """
    animals = sorted(df_animal["animal"].unique())
    deltas = []
    for a in animals:
        g = df_animal[df_animal["animal"] == a].set_index("state")
        if s1 in g.index and s2 in g.index:
            d = circ_diff(g.loc[s1, "mu_phase_rad"], g.loc[s2, "mu_phase_rad"])
            deltas.append(d)
    deltas = wrap_pi(np.asarray(deltas, float))

    def stat_fn(x):
        return float(np.abs(np.mean(np.exp(1j*np.asarray(x)))))  # resultant length of mean delta

    R_obs, p = exact_signflip_pvalue(deltas, stat_fn)
    mu_obs = circmean(deltas) if len(deltas) else np.nan

    return {
        "pair": f"{s1} - {s2}",
        "mu_delta_rad": float(mu_obs) if np.isfinite(mu_obs) else np.nan,
        "R_delta": float(R_obs) if np.isfinite(R_obs) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
        "n": int(np.isfinite(deltas).sum())
    }

#%%
'Multiple animal analysis'
# ANIMALS = [ {"animal_id": "1765508", 
#              "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ALocomotion",
#              "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\AwakeStationary", 
#              "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1765508_Jedi2p_Atlas\ASleepREM", 
#              "lfp_channel": "LFP_1"}, 
#            {"animal_id": "1844609", 
#             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ALocomotion", 
#             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\AwakeStationary", 
#             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1844609_WT_Jedi2p\ASleepREM", 
#             "lfp_channel": "LFP_1"}, 
#            {"animal_id": "1881363", 
#             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ALocomotion",
#             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\AwakeStationary",
#             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1881363_Jedi2p_mCherry\ASleepREM", 
#             "lfp_channel": "LFP_1"}, 
#            {"animal_id": "1887933", 
#             "loco_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ALocomotion", 
#             "stat_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\AwakeStationary", 
#             "rem_parent": r"G:\2025_ATLAS_SPAD\AcrossAnimal\1887933_Jedi2P_Multi\ASleepREM", 
#             "lfp_channel": "LFP_2", "opt_col": "ref_raw"},]

ANIMALS = [ 
            {"animal_id": "1842515", 
                         "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\ALocomotion",
                         "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\AwakeStationary", 
                         "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842515_PV_mNeon\ASleepREM", 
                         "lfp_channel": "LFP_1"},
            
            {"animal_id": "1842516", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\ASleepNonREM", 
            "lfp_channel": "LFP_4"}, 
            
            {"animal_id": "1844607", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1844607_PV_mNeon_F\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1844607_PV_mNeon_F\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1844607_PV_mNeon_F\ASleepNonREM", 
            "lfp_channel": "LFP_2"}, 
            
            {"animal_id": "1887930", 
            "loco_parent": r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\ALocomotion", 
            "stat_parent": r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\AwakeStationary", 
            "rem_parent": r"G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\ASleepNonREM", 
            "lfp_channel": "LFP_4"}, 
            ]
df_sweep_g, df_animal_g = run_across_animals_gamma(
    ANIMALS,
    fs=10000,
    gamma_bands={"slow_gamma": (30,55), "fast_gamma": (65,100)},
    theta_low_thres=0.0,
    indicator="GEVI",
    opt_col_default="zscore_raw",
    gate_quantile=0.7,     # relax to 0.6 if too few epochs
    gate_on="lfp",         # or "both"
    min_epoch_s=0.10
)

print(df_animal_g.sort_values(["band","animal","state"]))
# Example: list significant phase-locks
sig = df_animal_g[df_animal_g["rayleigh_p_pool"] < 0.05]
print(sig[["band","animal","state","n_cycles_pool","R_phase","mu_phase_rad","rayleigh_p_pool"]]
      .sort_values(["band","state","rayleigh_p_pool"]))
# your existing functions (already in your script):
# exact_within_animal_label_perm_3states(df_animal_g, metric_col)
# pairwise_linear_signflip(df_animal_g, metric_col, s1, s2)
# pairwise_circular_signflip(df_animal_g, s1, s2)

for band in ["slow_gamma", "fast_gamma"]:
    dfB = df_animal_g[df_animal_g["band"] == band].copy()
    print("\n====================")
    print("BAND:", band, "| n animals:", dfB["animal"].nunique())
    print("====================")

    # Omnibus + pairwise for linear metrics
    for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "env_xcorr_peak_lag_ms", "lag_from_phase_ms"]:
        print(exact_within_animal_label_perm_3states(dfB, metric))

    pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
    for metric in ["xcorr_peak_lag_ms", "PLV", "env_r", "env_xcorr_peak_lag_ms", "lag_from_phase_ms"]:
        for s1, s2 in pairs:
            print(pairwise_linear_signflip(dfB, metric, s1=s1, s2=s2))

    # Pairwise circular tests on mean phase direction (mu_phase_rad)
    for s1, s2 in pairs:
        res = pairwise_circular_signflip(dfB, s1=s1, s2=s2)
        mu_deg = np.degrees(_wrap_pi(res["mu_delta_rad"])) if np.isfinite(res["mu_delta_rad"]) else np.nan
        print(f"{band} {res['pair']}: mu_delta={mu_deg:.1f} deg, R={res['R_delta']:.3f}, p={res['p']:.4g}, n={res['n']}")

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers (same pattern as your theta plotting) ----------
def _prep_state_order(df, states_order):
    df = df.copy()
    df["state"] = pd.Categorical(df["state"], categories=states_order, ordered=True)
    return df.sort_values(["animal", "state"])

def _format_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3g}"

def _extract_p(stats_list, metric, pair=None):
    for r in stats_list:
        if r.get("metric") != metric:
            continue
        if pair is None and "pair" not in r:
            return r.get("p", None)
        if pair is not None and r.get("pair") == pair:
            return r.get("p", None)
    return None

def plot_spaghetti_metric(df_animal, metric, states_order=("Moving","Rest","REM"),
                          ylabel=None, title=None, ax=None, add_legend=False):
    df = _prep_state_order(df_animal, states_order)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 4.0))

    x = np.arange(len(states_order))

    handles, labels = [], []
    for animal, g in df.groupby("animal"):
        g = g.set_index("state").reindex(states_order)
        y = g[metric].to_numpy(dtype=float)
        h, = ax.plot(x, y, marker='o', linewidth=1.8, alpha=0.9, label=str(animal))
        handles.append(h); labels.append(str(animal))

    means = df.groupby("state")[metric].mean().reindex(states_order).to_numpy()
    sems  = df.groupby("state")[metric].sem().reindex(states_order).to_numpy()
    ax.errorbar(x, means, yerr=sems, fmt='-s', linewidth=2.2, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(states_order)
    ax.set_ylabel(ylabel if ylabel else metric)
    ax.set_title(title if title else metric)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.25)

    if add_legend:
        ax.legend(title="Animal", fontsize=9, frameon=False, loc="best")

    return ax, handles, labels

def annotate_pairwise_p(ax, pvals, states_order=("Moving","Rest","REM"),
                        pad_frac=0.07, step_frac=0.07, fontsize=12):
    """
    Draw pairwise brackets ABOVE data and return (y_top, step) so we can place omnibus above.
    """
    x_map = {s: i for i, s in enumerate(states_order)}

    # y-range from plotted data
    y_min, y_max = ax.dataLim.y0, ax.dataLim.y1
    yr = (y_max - y_min) if (y_max > y_min) else 1.0

    y_base = y_max + pad_frac * yr
    step = step_frac * yr

    bracket_specs = [
        (("Moving", "Rest"), y_base),
        (("Rest",   "REM"),  y_base + step),
        (("Moving", "REM"),  y_base + 2*step),
    ]

    for (a, b), y in bracket_specs:
        key = (a, b) if (a, b) in pvals else (b, a)
        p = pvals.get(key, None)
        if p is None:
            continue

        xa, xb = x_map[a], x_map[b]
        ax.plot([xa, xa, xb, xb],
                [y - 0.5*step, y, y, y - 0.5*step],
                lw=1.2, color="k")
        ax.text((xa + xb) / 2, y + 0.15*step, _format_p(p),
                ha="center", va="bottom", fontsize=fontsize)

    y_top = y_base + 2.8*step
    ax.set_ylim(y_min - 0.02*yr, y_top)
    return y_top, step

def collect_stats_for_band(df_animal_gamma, band,
                           metrics=("xcorr_peak_lag_ms","PLV","env_r","env_xcorr_peak_lag_ms","lag_from_phase_ms")):
    """
    Recompute the exact tests for ONE band using your existing functions:
      - exact_within_animal_label_perm_3states
      - pairwise_linear_signflip
    """
    dfB = df_animal_gamma[df_animal_gamma["band"] == band].copy()
    stats = []

    # omnibus (exact 6^n)
    for m in metrics:
        if m in dfB.columns:
            stats.append(exact_within_animal_label_perm_3states(dfB, m))

    # pairwise (exact 2^n)
    pairs = [("Rest","Moving"), ("REM","Moving"), ("REM","Rest")]
    for m in metrics:
        if m in dfB.columns:
            for s1, s2 in pairs:
                stats.append(pairwise_linear_signflip(dfB, m, s1=s1, s2=s2))

    return dfB, stats

def plot_gamma_band_summary(df_animal_gamma, band, states_order=("Moving","Rest","REM"),
                            figsize=(12.8, 7.2), show_lag_from_phase=False):
    """
    2×2 panel per band:
      - xcorr_peak_lag_ms
      - PLV
      - env_r
      - env_xcorr_peak_lag_ms
    with omnibus p above the pairwise brackets, and a single shared animal legend.
    """
    dfB, stats = collect_stats_for_band(df_animal_gamma, band)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.ravel()

    panels = [
        ("xcorr_peak_lag_ms",        "Xcorr peak lag (ms)",            f"{band}: LFP–GEVI gamma lag (xcorr peak)"),
        ("PLV",                      "PLV (gamma)",                    f"{band}: Phase-locking value (gamma)"),
        ("env_r",                    "Envelope correlation r",         f"{band}: Gamma envelope correlation"),
        ("env_xcorr_peak_lag_ms",    "Envelope xcorr lag (ms)",        f"{band}: Envelope lag (xcorr peak)"),
    ]

    legend_handles = legend_labels = None

    for ax, (metric, ylabel, title) in zip(axes, panels):
        if metric not in dfB.columns:
            ax.axis("off")
            continue

        ax, handles, labels = plot_spaghetti_metric(
            dfB, metric, states_order=states_order,
            ylabel=ylabel, title=title,
            ax=ax, add_legend=False
        )
        if legend_handles is None:
            legend_handles, legend_labels = handles, labels

        # p-values
        p_omni = _extract_p(stats, metric, pair=None)
        p_mr   = _extract_p(stats, metric, pair="Rest - Moving")
        p_rm   = _extract_p(stats, metric, pair="REM - Moving")
        p_rr   = _extract_p(stats, metric, pair="REM - Rest")

        y_top, step = annotate_pairwise_p(
            ax,
            {("Rest","Moving"): p_mr, ("REM","Moving"): p_rm, ("REM","Rest"): p_rr},
            states_order=states_order
        )

        # omnibus above brackets (avoids overlap)
        y_omni = y_top + 0.55*step
        ax.set_ylim(ax.get_ylim()[0], y_omni + 0.6*step)
        ax.text(0.02, y_omni, f"Omnibus {_format_p(p_omni)}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=12)

    # Single shared legend (animal IDs) on the right
    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, title="Animal", frameon=False,
                   loc="center right", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.show()
    return fig

# -------------------------
# USAGE
# -------------------------
# df_animal_g is your gamma animal-level table (one row per animal×state×band)
plot_gamma_band_summary(df_animal_g, "slow_gamma")
plot_gamma_band_summary(df_animal_g, "fast_gamma")
