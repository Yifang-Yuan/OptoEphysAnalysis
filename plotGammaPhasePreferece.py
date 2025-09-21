# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 23:19:03 2025
The same as plotThetaPhasePreference.py, use this as gamma plot for convenience.

@author: yifan
"""

import os
import re
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import pickle, gzip, time, uuid
from pathlib import Path
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Optional

# --- assumes your compute_optical_phase_preference is already defined in scope ---

def _normalised_contour(bin_centers, counts):
    """Return closed contour (theta, r) where r is normalised to [0, 1]."""
    counts = np.asarray(counts)
    r = counts / counts.max() if counts.max() > 0 else counts.astype(float)
    theta_circ = np.append(bin_centers, bin_centers[0])
    r_circ     = np.append(r, r[0])
    return theta_circ, r_circ, r  # return open r as well for averaging

def save_trial_phase_metrics(theta_phase,
                             signal,
                             save_path,
                             *,
                             bins=30,
                             height_factor=3.0,
                             distance=20,
                             prominence=None,
                             min_events=50,
                             alpha=0.01,
                             use_event_indices=None,
                             meta=None,
                             plot=True):
    """
    Compute per-trial contour, preferred phase and modulation depth, then save to pickle.gz.

    save_path : str/Path
        e.g. 'session01_trial03_phase.pkl.gz'
    meta : dict or None
        Optional metadata to embed (animal, session, trial, Fs, etc.).
    """
    res = OE.compute_optical_phase_preference(
        theta_phase, signal,
        bins=bins,
        height_factor=height_factor,
        distance=distance,
        prominence=prominence,
        min_events=min_events,
        alpha=alpha,
        use_event_indices=use_event_indices,
        plot=plot
    )

    theta_circ, r_circ, r_open = _normalised_contour(res['bin_centers'], res['counts'])

    out = {
        'version'             : 1,
        'uuid'                : str(uuid.uuid4()),
        'timestamp'           : time.time(),
        'bins'                : int(bins),
        'bin_centers'         : res['bin_centers'],
        'counts'              : res['counts'],          # raw counts (for reference)
        'contour_theta'       : theta_circ,             # closed for quick plotting
        'contour_r'           : r_circ,                 # closed for quick plotting
        'contour_r_open'      : r_open,                 # length = bins (for averaging)
        'preferred_phase_rad' : res['preferred_phase_rad'],
        'preferred_phase_deg' : res['preferred_phase_deg'],
        'R'                   : res['modulation_depth_R'],
        'Z'                   : res['Z'],
        'p'                   : res['p'],
        'n_events'            : res['n_events'],
        'significant'         : res['significant'],
        'meta'                : meta or {}
    }

    save_path = Path(save_path)
    if save_path.suffix != '.gz':
        # default to gzip compression
        save_path = save_path.with_suffix(save_path.suffix + '.gz')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_path, out


def _load_pickle(path):
    path = Path(path)
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

def aggregate_phase_pickles_plot_simple(paths):
    """
    Aggregate multiple trial pickle files saved by save_trial_phase_metrics.

    Returns
    -------
    agg : dict
        - 'bin_centers': common bin centres
        - 'mean_contour_r_open': mean normalised contour (length = bins)
        - 'sem_contour_r_open' : SEM of contour across trials
        - 'group_pref_phase_rad/deg': weighted circular mean phase
        - 'group_R' : weighted mean resultant length (length of the weighted mean vector)
        - 'per_trial': table-like dict with per-trial phi, R, n
        - 'n_trials': number of files aggregated
    """
    paths = [Path(p) for p in paths]
    records = [_load_pickle(p) for p in paths]

    if len(records) == 0:
        raise ValueError("No pickle files provided.")

    # --- Ensure all have the same bins/bin_centers ---
    bins0 = records[0]['bins']
    bc0 = records[0]['bin_centers']
    for r in records[1:]:
        if r['bins'] != bins0 or not np.allclose(r['bin_centers'], bc0, atol=1e-9):
            raise ValueError("Inconsistent binning across pickles; re-run with the same 'bins'.")

    # --- Stack contours ---
    contours = np.vstack([r['contour_r_open'] for r in records])  # shape: (T, bins)
    mean_contour = contours.mean(axis=0)
    sem_contour  = contours.std(axis=0, ddof=1) / np.sqrt(contours.shape[0])

    # --- Weighted circular mean of preferred phase, weighted by n_events * R ---
    phi  = np.array([r['preferred_phase_rad'] for r in records])
    R    = np.array([r['R'] for r in records])
    n    = np.array([r['n_events'] for r in records])

    w = n.astype(float)
    Cx = np.sum(w * R * np.cos(phi))
    Sy = np.sum(w * R * np.sin(phi))
    
    group_pref_phase = (np.arctan2(Sy, Cx)) % (2*np.pi)
    group_R = np.sqrt(Cx**2 + Sy**2) / np.sum(w)

    # simple dispersion: circular standard error of the mean direction (approx)
    # (uses effective resultant length; for large-sample heuristic)
    # This is optional and approximate; feel free to ignore if not needed.
    R_bar = group_R
    kappa_se = np.sqrt((1 - R_bar) / (np.sum(w)))  # heuristic
    sem_phase = kappa_se  # radians (very rough)

    agg = {
        'bin_centers'           : bc0,
        'mean_contour_r_open'   : mean_contour,
        'sem_contour_r_open'    : sem_contour,
        'group_pref_phase_rad'  : float(group_pref_phase),
        'group_pref_phase_deg'  : float(np.degrees(group_pref_phase) % 360.0),
        'group_R'               : float(group_R),
        'sem_pref_phase_rad'    : float(sem_phase),
        'n_trials'              : int(len(records)),
        'per_trial'             : {
            'preferred_phase_rad': phi,
            'preferred_phase_deg': np.degrees(phi) % 360.0,
            'R'                  : R,
            'n_events'           : n
        }
    }
    return agg
# --------- Group circular stats helpers (events-weighted) ---------
@dataclass
class RayleighResult:
    n: int
    R_bar: float
    Z: float
    p: float
    note: str

@dataclass
class VTestResult:
    n: int
    mu_rad: float
    u: float
    p_one_sided: float
    note: str

def _rayleigh_from_summary(phi, R, n, mode: str) -> RayleighResult:
    """
    Rayleigh test from per-sweep summaries.
    mode='events' -> uses grand resultant from (n_i * R_i) vectors (correct).
    mode='equal'  -> treats sweep means as unit vectors (approx.).
    """
    phi = np.asarray(phi, float)
    R   = np.asarray(R, float)
    n   = np.asarray(n,  int)

    if mode == "events":
        C = np.sum(n * R * np.cos(phi))
        S = np.sum(n * R * np.sin(phi))
        n_tot = int(np.sum(n))
        R_bar = (np.hypot(C, S) / n_tot) if n_tot > 0 else 0.0
        note  = "events-weighted Rayleigh (uses total events)"
    else:
        C = np.sum(np.cos(phi))
        S = np.sum(np.sin(phi))
        n_tot = len(phi)
        R_bar = (np.hypot(C, S) / n_tot) if n_tot > 0 else 0.0
        note  = "approximate Rayleigh on sweep means (equal-weighted)"

    Z = n_tot * (R_bar ** 2)
    # Berens (2009) first-order p approximation
    p = math.exp(-Z) * (1 + (2*Z - Z*Z) / (4*max(n_tot,1))) if n_tot > 0 else 1.0
    p = float(np.clip(p, 0.0, 1.0))
    return RayleighResult(n=n_tot, R_bar=float(R_bar), Z=float(Z), p=p, note=note)

def _vtest_from_summary(phi, R, n, mu_rad: float, mode: str) -> VTestResult:
    """
    V-test toward specified mean direction mu_rad.
    mode='events' -> grand resultant from (n_i * R_i) vectors.
    mode='equal'  -> unit vectors at sweep means (approx.).
    """
    phi = np.asarray(phi, float)
    R   = np.asarray(R, float)
    n   = np.asarray(n,  int)

    if mode == "events":
        C = np.sum(n * R * np.cos(phi))
        S = np.sum(n * R * np.sin(phi))
        n_tot = int(np.sum(n))
        note  = "events-weighted V-test (uses total events)"
    else:
        C = np.sum(np.cos(phi))
        S = np.sum(np.sin(phi))
        n_tot = len(phi)
        note  = "approximate V-test on sweep means (equal-weighted)"

    if n_tot == 0:
        return VTestResult(n=0, mu_rad=float(mu_rad), u=float("nan"), p_one_sided=1.0, note=note)

    psi   = math.atan2(S, C) % (2*np.pi)
    R_bar = math.hypot(C, S) / n_tot
    m     = R_bar * math.cos((psi - mu_rad) % (2*np.pi))  # component toward mu
    u     = math.sqrt(2 * n_tot) * m

    try:
        from scipy.stats import norm
        p_one = float(norm.sf(u))  # one-sided
    except Exception:
        p_one = 0.5 * math.erfc(u / math.sqrt(2))
    return VTestResult(n=n_tot, mu_rad=float(mu_rad), u=float(u), p_one_sided=p_one, note=note)

def add_rayleigh_to_agg(agg, overwrite: bool = True):
    phi = agg['per_trial']['preferred_phase_rad']
    R   = agg['per_trial']['R']
    n   = agg['per_trial']['n_events']
    mode = "events" if agg.get('weight_mode', 'events') == 'events' else "equal"
    res  = _rayleigh_from_summary(phi, R, n, mode)
    agg.setdefault('group_stats', {})
    if overwrite or ('rayleigh' not in agg['group_stats']):
        agg['group_stats']['rayleigh'] = res
    return agg

def add_vtest_to_agg(agg, mu_deg: float, overwrite: bool = True):
    mu_rad = np.deg2rad(mu_deg % 360.0)
    phi = agg['per_trial']['preferred_phase_rad']
    R   = agg['per_trial']['R']
    n   = agg['per_trial']['n_events']
    mode = "events" if agg.get('weight_mode', 'events') == 'events' else "equal"
    vres = _vtest_from_summary(phi, R, n, mu_rad, mode)
    agg.setdefault('group_stats', {})
    if overwrite or ('vtest' not in agg['group_stats']):
        agg['group_stats']['vtest'] = vres
    agg['target_mu_rad'] = float(mu_rad)
    agg['target_mu_deg'] = float(mu_deg % 360.0)
    return agg

def print_group_phase_stats(agg, mu_deg: Optional[float] = None):
    """Pretty print group μ, R plus Rayleigh; optionally V-test toward μ."""
    print("=== Group Circular Statistics ===")
    print(f"n_trials: {agg['n_trials']}")
    print(f"group preferred phase: {agg['group_pref_phase_deg']:.1f}°")
    print(f"group resultant length R: {agg['group_R']:.3f}")

    ray = agg.get('group_stats', {}).get('rayleigh', None)
    if ray is not None:
        print(f"\nRayleigh test [{ray.note}]")
        print(f"  n = {ray.n}, R̄ = {ray.R_bar:.4f}, Z = {ray.Z:.3f}, p = {ray.p:.3g}")

    if mu_deg is not None:
        vts = agg.get('group_stats', {}).get('vtest', None)
        if vts is not None:
            print(f"\nV-test toward μ = {mu_deg:.1f}° [{vts.note}]")
            print(f"  n = {vts.n}, u = {vts.u:.3f}, one-sided p = {vts.p_one_sided:.3g}")

def aggregate_phase_pickles(paths, ci_method="bootstrap", n_boot=2000, ci_alpha=0.05, weight_mode="events"):
    """
    Aggregate saved trial pickles and compute mean contour + CI, and group phase/R.

    Parameters
    ----------
    paths : list[str|Path]
    ci_method : "bootstrap" | "sem"
        - "bootstrap": percentile CI from resampling trials with replacement
        - "sem": mean ± z * SEM (Gaussian approx), z≈1.96 for 95%
    n_boot : int
        Number of bootstrap replicates if ci_method="bootstrap".
    ci_alpha : float
        1 - CI level (0.05 -> 95% CI).
    weight_mode : "events" | "equal"
        Weights for group preferred phase/R: event-count weighting or equal trials.

    Returns
    -------
    dict with:
        'bin_centers', 'mean_contour_r_open', 'sem_contour_r_open',
        'ci_lower', 'ci_upper', 'group_pref_phase_rad/deg', 'group_R',
        'n_trials', 'per_trial' (phi, R, n)
    """
    records = [_load_pickle(p) for p in paths]
    if len(records) == 0:
        raise ValueError("No pickle files provided.")

    # --- Consistency check
    bins0 = records[0]['bins']
    bc0   = records[0]['bin_centers']
    for r in records[1:]:
        if r['bins'] != bins0 or not np.allclose(r['bin_centers'], bc0, atol=1e-9):
            raise ValueError("Inconsistent binning across pickles.")

    # --- Stack normalised contours (T, bins)
    contours = np.vstack([r['contour_r_open'] for r in records])
    T, B = contours.shape

    mean_contour = contours.mean(axis=0)
    sem_contour  = contours.std(axis=0, ddof=1) / np.sqrt(T)

    # --- CI
    if ci_method == "sem":
        from scipy.stats import norm
        z = norm.ppf(1 - ci_alpha/2)
        ci_lower = mean_contour - z * sem_contour
        ci_upper = mean_contour + z * sem_contour
    elif ci_method == "bootstrap":
        rng = np.random.default_rng()
        boot = np.empty((n_boot, B))
        for b in range(n_boot):
            idx = rng.integers(0, T, size=T)
            boot[b] = contours[idx].mean(axis=0)
        lo = 100 * (ci_alpha/2)
        hi = 100 * (1 - ci_alpha/2)
        ci_lower = np.percentile(boot, lo, axis=0)
        ci_upper = np.percentile(boot, hi, axis=0)
    else:
        raise ValueError("ci_method must be 'bootstrap' or 'sem'.")

    # --- Group preferred phase & R (fixed formula)
    phi = np.array([r['preferred_phase_rad'] for r in records])
    R   = np.array([r['R'] for r in records])
    n   = np.array([r['n_events'] for r in records])

    if weight_mode == "events":
        w = n.astype(float)
    elif weight_mode == "equal":
        w = np.ones_like(n, dtype=float)
    else:
        raise ValueError("weight_mode must be 'events' or 'equal'.")

    Cx = np.sum(w * R * np.cos(phi))
    Sy = np.sum(w * R * np.sin(phi))
    group_pref_phase = (np.arctan2(Sy, Cx)) % (2*np.pi)
    group_R = np.sqrt(Cx**2 + Sy**2) / np.sum(w)

    return {
        'bin_centers'          : bc0,
        'mean_contour_r_open'  : mean_contour,
        'sem_contour_r_open'   : sem_contour,
        'ci_lower'             : ci_lower,
        'ci_upper'             : ci_upper,
        'group_pref_phase_rad' : float(group_pref_phase),
        'group_pref_phase_deg' : float(np.degrees(group_pref_phase) % 360.0),
        'group_R'              : float(group_R),
        'n_trials'             : int(T),
        'weight_mode'          : weight_mode,   # <— ADD THIS LINE
        'per_trial'            : {
            'preferred_phase_rad': phi,
            'preferred_phase_deg': np.degrees(phi) % 360.0,
            'R'                  : R,
            'n_events'           : n
        }
    }


def plot_group_contour_with_ci(agg, title=None):
    """
    Draw mean contour with shaded CI and the group preferred-phase arrow.
    Expects output dict from `aggregate_phase_pickles`.
    """
    bc = agg['bin_centers']
    mean_r = agg['mean_contour_r_open']
    lo = agg['ci_lower']
    hi = agg['ci_upper']

    # Close the contours for plotting
    theta_circ = np.append(bc, bc[0])
    mean_circ  = np.append(mean_r, mean_r[0])
    lo_circ    = np.append(lo, lo[0])
    hi_circ    = np.append(hi, hi[0])

    fig = plt.figure(figsize=(5.2, 5.2))
    ax  = fig.add_subplot(111, projection='polar')

    # Style: 0° right, anticlockwise; thicker spine; bigger angle labels
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.spines['polar'].set_linewidth(2.5)
    ax.grid(True, linewidth=0.8, alpha=0.4)
    ax.set_thetagrids([0, 90, 180, 270], labels=['0', '90', '180', '270'], fontsize=14, weight='bold')

    # Radial limits/ticks
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1'], fontsize=11)

    # CI band (shaded)
    ax.fill_between(theta_circ, lo_circ, hi_circ, alpha=0.25, linewidth=0)

    # Mean contour (solid line)
    ax.plot(theta_circ, mean_circ, linewidth=2.5)

    # Group preferred-phase arrow (length ∝ group R)
    arrow_len = 0.9 * agg['group_R']
    ax.annotate(
        "",
        xy=(agg['group_pref_phase_rad'], arrow_len),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", linewidth=2.5)
    )

    if title is None:
        title = (f"Group pref={agg['group_pref_phase_deg']:.1f}°, "
                 f"R={agg['group_R']:.3f} (n={agg['n_trials']})")
    ax.set_title(title, va="bottom", pad=12, fontsize=12)

    plt.tight_layout()
    return fig, ax

def calculate_theta_phase_session_all(dpath,savename,LFP_channel,theta_low_thres,behaviour):

    # Find all folders starting with SyncRecording
    all_folders = [
        f for f in os.listdir(dpath)
        if os.path.isdir(os.path.join(dpath, f)) and f.startswith('SyncRecording')
    ]

    # Sort numerically by number after 'SyncRecording'
    def extract_number(name):
        match = re.search(r"SyncRecording(\d+)", name)
        return int(match.group(1)) if match else float("inf")
    all_folders_sorted = sorted(all_folders, key=extract_number)

    # Create save folder if it doesn't exist
    save_path = os.path.join(dpath, savename)
    os.makedirs(save_path, exist_ok=True)

    # Loop through each recording
    for idx, recordingName in enumerate(all_folders_sorted, start=1):
        print(f"Processing {recordingName} → Trial {idx:02d}")

        # Set save filename
        trial_filename = f"AwakeStationary_trial{idx:02d}.pkl.gz"
        pickle_save_path = os.path.join(save_path, trial_filename)

        # Run your analysis
        Recording1 = SyncOEpyPhotometrySession(
            dpath, recordingName,
            IsTracking=False,
            read_aligned_data_from_file=True,
            recordingMode='Atlas', indicator='GEVI'
        )

        # Label theta part
        if behaviour == 'Rest':
            Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_low_thres,High_thres=10,save=False,plot_theta=True)
            theta_part=Recording1.theta_part
        
        if behaviour == 'REM':
            '''Choose REM and nonREM and select REM'''
            Recording1.Label_REM_sleep (LFP_channel)
            theta_part =  Recording1.Ephys_tracking_spad_aligned[
            Recording1.Ephys_tracking_spad_aligned['REMstate'] == 'REM']
        if behaviour == 'Moving':
            '''Choose MOving state from open field trials '''
            theta_part =  Recording1.Ephys_tracking_spad_aligned[
            Recording1.Ephys_tracking_spad_aligned['movement'] == 'moving']

        # Calculate gamma angle
        gamma_angle = OE.calculate_theta_phase_angle(theta_part[LFP_channel], theta_low=30, theta_high=60)

        # Save results
        save_trial_phase_metrics(gamma_angle, theta_part['zscore_raw'], pickle_save_path,
                                 bins=30, height_factor=3.0, distance=20, plot=True)

        # Optional: compute optical event on phase
        OE.compute_optical_event_on_phase(gamma_angle, theta_part['zscore_raw'], bins=30, distance=10, plot=True)

        print(f"✅ Saved {pickle_save_path}")

def average_all_theta_phase_results(dpath, *, weight_mode="events", mu_deg=0.0, do_print=True):
    """
    Load all *.pkl.gz in dpath, aggregate, compute group Rayleigh and V-test (toward mu_deg),
    plot the group contour, and optionally print stats. Returns (agg, (fig, ax)).
    """
    folder = Path(dpath)
    paths = sorted(folder.glob("*.pkl.gz"))
    if len(paths) == 0:
        raise ValueError(f"No *.pkl.gz files found in {folder}")

    agg = aggregate_phase_pickles(paths, ci_method="bootstrap",
                                  n_boot=2000, ci_alpha=0.05,
                                  weight_mode=weight_mode)

    # attach stats
    agg = add_rayleigh_to_agg(agg)
    agg = add_vtest_to_agg(agg, mu_deg=mu_deg)

    if do_print:
        print_group_phase_stats(agg, mu_deg=mu_deg)

    fig, ax = plot_group_contour_with_ci(agg)
    return agg, (fig, ax)


'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_theta_plot_cycle_singleTrial (dpath,LFP_channel,recordingName,savename,theta_low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    os.makedirs(save_path, exist_ok=True)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GEVI') 
    '''Label Theta part'''
    # Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_low_thres,High_thres=10,save=False,plot_theta=True)
    # theta_part=Recording1.theta_part
    
    '''Label REM and nonREM'''
    # Recording1.Label_REM_sleep (LFP_channel)
    # theta_part =  Recording1.Ephys_tracking_spad_aligned[
    # Recording1.Ephys_tracking_spad_aligned['REMstate'] == 'REM']
    
    '''Choose MOving state from open field trials '''
    theta_part =  Recording1.Ephys_tracking_spad_aligned[
    Recording1.Ephys_tracking_spad_aligned['movement'] == 'moving']

    #theta_part=Recording1.Ephys_tracking_spad_aligned
    gamma_angle=OE.calculate_theta_phase_angle(theta_part[LFP_channel], theta_low=30, theta_high=60) #range 5 to 9
    pickle_save_path= os.path.join(dpath,savename,'Moving_trial10.pkl.gz')
    save_trial_phase_metrics(gamma_angle, theta_part['zscore_raw'], pickle_save_path,
                         bins=30, height_factor=3.0, distance=20, plot=True)
    OE.compute_optical_event_on_phase(gamma_angle, 
                                theta_part['zscore_raw'], bins=30, distance=10,plot=True)
    return -1

def calculate_theta_phase_session(dpath):
    '''This is to process a single or concatenated trial, 
    with a Ephys_tracking_photometry_aligned.pkl in the recording folder'''
    savename='GammaPhase_Save'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'
    theta_low_thres=-0.5
    behaviour='Moving'
    calculate_theta_phase_session_all(dpath,savename,LFP_channel,theta_low_thres,behaviour)
    #recordingName='SyncRecording10'
    #run_theta_plot_cycle_singleTrial (dpath,LFP_channel,recordingName,savename,theta_low_thres) #-0.3

    
def main():    
    # dpath=r'G:\2025_ATLAS_SPAD\PyramidalWT\1844609_WT_Jedi2p\ALocomotion'
    # calculate_theta_phase_session(dpath)
    
    spath = r"G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\ASleepREM\GammaPhase_Save"
    agg, _ = average_all_theta_phase_results(spath, weight_mode="events", mu_deg=0.0)
if __name__ == "__main__":
    main()