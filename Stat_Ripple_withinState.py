# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:27:38 2025

@author: yifan
"""
# ---------- NEW / UPDATED UTILITIES -----------------------------------------
import os, glob, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, hilbert, butter, filtfilt
import matplotlib.pyplot as plt
import pickle
import plotRipple as Ripple
from SyncOECPySessionClass import SyncOEpyPhotometrySession
from scipy.stats import median_abs_deviation, sem, wilcoxon
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import re

# --- fix: consistent filenames (also read legacy 'ailgned_*') ---------------
ALIGNED_FILES = {
    "lfp_bp":  ["aligned_ripple_bandpass_LFP.pkl",  "ailgned_ripple_bandpass_LFP.pkl"],
    "lfp_bb":  ["aligned_ripple_LFP.pkl",           "ailgned_ripple_LFP.pkl"],
    "zscore":  ["aligned_ripple_Zscore.pkl",        "ailgned_ripple_Zscore.pkl"],
}

ANIMAL_REGEX = r'(\d{6,})'  # adjust if your IDs look different

def _extract_animal_id(path_str: str) -> str:
    m = re.search(ANIMAL_REGEX, path_str.replace('\\','/'))
    return m.group(1) if m else "unknown"

def _safe_save_pickle(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(arr, f)

def _safe_load_first(path_list):
    """Try multiple filenames and return the first that exists."""
    for p in path_list:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"None of these files exist: {path_list}")

# --- small helper used by phase analyses ------------------------------------
def _butter_bandpass(data, fs, low=130, high=250, order=4, axis=-1):
    b, a = butter(order, [low/(fs*0.5), high/(fs*0.5)], btype='band')
    return filtfilt(b, a, data, axis=axis)

def detect_peaks_times(zscores_aligned, fs=10_000, detect_window=(-0.1, 0.1),
                       peak_thr=3.0, distance=15):
    """
    Detect peaks ONCE per epoch within detect_window and return event times (s)
    relative to the ripple peak (t=0). This matches 3a/3b detection.
    """
    from scipy.signal import find_peaks
    from scipy.stats import median_abs_deviation

    n_ep, n_samp = zscores_aligned.shape
    mid = n_samp // 2
    t_full = (np.arange(n_samp) - mid) / fs
    mask_det = (t_full >= detect_window[0]) & (t_full <= detect_window[1])

    times_per_epoch = []
    for e in range(n_ep):
        tr = zscores_aligned[e, mask_det]
        thr = np.median(tr) + peak_thr * median_abs_deviation(tr)
        idx, _ = find_peaks(tr, height=thr, distance=distance)
        # Convert indices to times using the DETECTION window’s time axis
        t_det = t_full[mask_det]
        times_per_epoch.append(t_det[idx])
    return times_per_epoch  # list of 1-D arrays of times (seconds)

# ---------- PATCH your saver to use consistent filenames --------------------
def plot_aligned_ripple_save_FIXED(save_path, LFP_channel, recordingName,
                                   ripple_triggered_lfps, ripple_triggered_zscores, Fs=10000):
    """
    Identical to your plot_aligned_ripple_save but:
    - saves under filenames starting with 'aligned_' (consistent)
    - keeps your heatmaps as-is
    """
    os.makedirs(save_path, exist_ok=True)
    ripple_sample_numbers = len(ripple_triggered_lfps[0])
    midpoint = ripple_sample_numbers // 2
    start_idx = int(midpoint - 0.08*Fs)
    end_idx   = int(midpoint + 0.08*Fs)

    aligned_ripple_band_lfps, aligned_lfps, aligned_zscores = Ripple.align_ripples(
        ripple_triggered_lfps, ripple_triggered_zscores, start_idx, end_idx, midpoint, Fs
    )

    fig = Ripple.plot_ripple_heatmap(aligned_ripple_band_lfps, aligned_lfps, aligned_zscores, Fs)
    fig.savefig(os.path.join(save_path, f"{recordingName}{LFP_channel}_Ripple_aligned_heatmap_400ms.png"), transparent=True)

    fig = Ripple.plot_ripple_heatmap(aligned_ripple_band_lfps[:, start_idx:end_idx],
                              aligned_lfps[:, start_idx:end_idx],
                              aligned_zscores[:, start_idx:end_idx], Fs)
    fig.savefig(os.path.join(save_path, f"{recordingName}{LFP_channel}_Ripple_aligned_heatmap_200ms.png"), transparent=True)

    # --- consistent filenames (plus legacy keepers already handled on load) --
    _safe_save_pickle(aligned_lfps,               os.path.join(save_path, "aligned_ripple_LFP.pkl"))
    _safe_save_pickle(aligned_ripple_band_lfps,   os.path.join(save_path, "aligned_ripple_bandpass_LFP.pkl"))
    _safe_save_pickle(aligned_zscores,            os.path.join(save_path, "aligned_ripple_Zscore.pkl"))

    return aligned_ripple_band_lfps, aligned_zscores

def compute_and_save_trial_rmi(aligned_zscores,
                               out_dir,
                               recording_name,
                               fs=10_000,
                               ripple_dur=0.06,
                               thresh_factor=3.0,
                               distance=80,
                               n_shuffle=2000,
                               seed=0):
    """
    Compute pooled and per-epoch RMI for a *single SyncRecording* (trial) and save:
      - session_ripple_RMI.pkl  (dict with pooled RMI, per-epoch RMI, null, etc.)
      - session_ripple_RMI_null.png  (large-font null histogram)
    Returns the saved dict.
    """
    os.makedirs(out_dir, exist_ok=True)

    # pooled RMI via shuffling
    rmi_real, p_perm, rmi_null = Ripple.ripple_modulation_index_events(
        aligned_zscores, fs=fs, ripple_dur=ripple_dur,
        thresh_factor=thresh_factor, distance=distance,
        n_shuffle=n_shuffle, seed=seed
    )
    # per-epoch RMIs
    per_epoch_rmi, per_epoch_in, per_epoch_out, _ = _per_epoch_rmi_events(
        aligned_zscores, fs=fs, ripple_dur=ripple_dur,
        thresh_factor=thresh_factor, distance=distance
    )

    payload = {
        "recording": recording_name,
        "pooled_RMI": float(rmi_real),
        "pooled_perm_p": float(p_perm),
        "null_RMI": rmi_null,
        "per_epoch_RMI": per_epoch_rmi,
        "per_epoch_in_counts": per_epoch_in,
        "per_epoch_out_counts": per_epoch_out,
        "fs": fs, "ripple_dur": ripple_dur,
        "thresh_factor": thresh_factor, "distance": distance,
        "n_shuffle": n_shuffle, "seed": seed
    }

    # save PKL
    pkl_path = os.path.join(out_dir, "session_ripple_RMI.pkl")
    _safe_save_pickle(payload, pkl_path)

    # save null figure (large fonts, fixed-point p-value)
    plot_rmi_null_distribution(
        rmi_null, rmi_real,
        out_png=os.path.join(out_dir, "session_ripple_RMI_null.png"),
        title=f"RMI null (perm p={p_perm:.2f})",
        title_fs=26, label_fs=22, tick_fs=20, legend_fs=20
    )
    return payload

# ---------- Run one recording folder ----------------------------------------
def run_one_recording_folder(rec_dir, LFP_channel="LFP_1", savename="RippleSave", theta_cutoff=0.5):
    """
    rec_dir points directly to .../SyncRecordingX
    Saves aligned epochs + per-session RMI (PKL + PNG) into this recording's save folder,
    and appends a row into a state-level CSV summary.
    """
    rec_dir = Path(rec_dir)
    dpath   = str(rec_dir.parent)        # state/session folder path
    recordingName = rec_dir.name         # "SyncRecordingX"
    save_path = str(rec_dir / f"{savename}_{LFP_channel}")

    Recording1 = SyncOEpyPhotometrySession(dpath, recordingName,
                                           IsTracking=False,
                                           read_aligned_data_from_file=True,
                                           recordingMode='Atlas', indicator='GEVI')

    Recording1.pynacollada_label_theta(LFP_channel, Low_thres=theta_cutoff, High_thres=8,
                                       save=False, plot_theta=True)

    # Detect ripples (exclude theta)
    Recording1.pynappleAnalysis(lfp_channel=LFP_channel,
                                ep_start=0, ep_end=20,
                                Low_thres=1.5, High_thres=10,
                                plot_segment=False, plot_ripple_ep=False,
                                excludeTheta=True)

    idx = LFP_channel.split('_')[-1]
    if idx == '1':
        ripple_triggered_LFP_values = Recording1.ripple_triggered_LFP_values_1
    elif idx == '2':
        ripple_triggered_LFP_values = Recording1.ripple_triggered_LFP_values_2
    elif idx == '3':
        ripple_triggered_LFP_values = Recording1.ripple_triggered_LFP_values_3
    else:
        ripple_triggered_LFP_values = Recording1.ripple_triggered_LFP_values_4

    ripple_triggered_zscore_values = Recording1.ripple_triggered_zscore_values

    # align & save aligned epochs
    aligned_rip_bp, aligned_zs = plot_aligned_ripple_save_FIXED(
        save_path, LFP_channel, recordingName,
        ripple_triggered_LFP_values, ripple_triggered_zscore_values, Fs=10000
    )

    # quick timing plot per recording
    Ripple.plot_ripple_zscore(save_path, aligned_rip_bp, aligned_zs)

    # --- NEW: per-session (trial) RMI + null figure + PKL -------------------
    trial_payload = compute_and_save_trial_rmi(
        aligned_zscores=aligned_zs,
        out_dir=save_path,
        recording_name=recordingName,
        fs=10_000, ripple_dur=0.06, thresh_factor=3.0, distance=20,
        n_shuffle=2000, seed=0
    )
    print(f"[{recordingName}] session pooled RMI = {trial_payload['pooled_RMI']:.3f}, "
          f"perm p = {trial_payload['pooled_perm_p']:.2f} "
          f"(epochs={len(trial_payload['per_epoch_RMI'])})")

    # Append a summary row (state-level CSV in the parent of SyncRecordingX)
    state_dir = rec_dir.parent  # e.g., .../AwakeStationary or .../ASleepNonREM/DayX
    summary_csv = state_dir / "RippleSession_RMI_summary.csv"
    row = {
        "animal": _extract_animal_id(str(rec_dir)),
        "recording": recordingName,
        "save_dir": save_path,
        "pooled_RMI": trial_payload["pooled_RMI"],
        "perm_p": trial_payload["pooled_perm_p"],
        "n_epochs": int(len(trial_payload["per_epoch_RMI"]))
    }
    df_row = pd.DataFrame([row])
    if summary_csv.exists():
        df_row.to_csv(summary_csv, mode='a', header=False, index=False)
    else:
        df_row.to_csv(summary_csv, index=False)


# ---------- Batch over a parent folder --------------------------------------
def process_parent_folder(parent_dir, LFP_channel="LFP_1", savename="RippleSave", theta_cutoff=0.5):
    """
    parent_dir contains many SyncRecording* folders (possibly nested).
    """
    parent = Path(parent_dir)
    recs = sorted([p for p in parent.rglob("SyncRecording*") if p.is_dir()])
    if not recs:
        print("No SyncRecording* folders found.")
        return
    for rec in recs:
        try:
            print(f"Processing: {rec}")
            run_one_recording_folder(rec, LFP_channel=LFP_channel, savename=savename, theta_cutoff=theta_cutoff)
        except Exception as e:
            print(f"!! Skipped {rec}: {e}")

# ---------- Pool all saved epochs & analyse timing/phase --------------------
def load_all_aligned_epochs(parent_dir, savename_glob="RippleSave*", LFP_channel="LFP_1"):
    parent = Path(parent_dir)
    save_dirs = sorted([p for p in parent.rglob(f"{savename_glob}_{LFP_channel}") if p.is_dir()])
    if not save_dirs:
        raise RuntimeError("No saved RippleSave_* folders found.")

    bp_all, bb_all, zs_all = [], [], []
    for sd in save_dirs:
        try:
            bp = _safe_load_first([str(sd / n) for n in ALIGNED_FILES["lfp_bp"]])
            bb = _safe_load_first([str(sd / n) for n in ALIGNED_FILES["lfp_bb"]])
            zs = _safe_load_first([str(sd / n) for n in ALIGNED_FILES["zscore"]])
            bp_all.append(bp); bb_all.append(bb); zs_all.append(zs)
        except Exception as e:
            print(f"Load error in {sd}: {e}")

    bp_all = np.vstack(bp_all)
    bb_all = np.vstack(bb_all)
    zs_all = np.vstack(zs_all)
    return bp_all, bb_all, zs_all

# --- small compatibility shim for SciPy's binomial test ---------------------
try:
    from scipy.stats import binomtest
    def _binom_p(k, n, p):
        return binomtest(k, n, p, alternative='greater').pvalue
except Exception:
    from scipy.stats import binom_test
    def _binom_p(k, n, p):
        return binom_test(k, n, p, alternative='greater')

def peak_enrichment_test(zscores_aligned,
                         fs=10_000,
                         time_window=(-0.1, 0.1),
                         center_width=0.02,        # total width (e.g. ±10 ms)
                         peak_thr=3.0,
                         distance=20,
                         n_perm=2000,
                         seed=0):
    """
    Test whether optical events are enriched near t=0 (ripple peak).

    Binomial test: compares observed proportion in the central window to the
    uniform null p0 = center_width / (time_window[1] - time_window[0]).
    Permutation test: epoch-wise circular shift of z-score rows with re-detect.
    """
    n_ep, n_samp = zscores_aligned.shape
    mid = n_samp // 2
    t = (np.arange(n_samp) - mid) / fs
    mask = (t >= time_window[0]) & (t <= time_window[1])
    t_win = t[mask]
    zs    = zscores_aligned[:, mask]

    # detect peaks and collect pooled times
    all_times = []
    for tr in zs:
        thr = np.median(tr) + peak_thr * median_abs_deviation(tr)
        idx, _ = find_peaks(tr, height=thr, distance=distance)
        all_times.extend(t_win[idx])
    all_times = np.asarray(all_times)

    # observed counts
    half = center_width / 2.0
    in_mask = (all_times >= -half) & (all_times <= half)
    k_in = int(np.sum(in_mask))
    N   = int(all_times.size)
    p0  = center_width / (time_window[1] - time_window[0])
    p_binom = _binom_p(k_in, N, p0) if N > 0 else 1.0

    # permutation (circular shifts within the window)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm, dtype=int)
    L = zs.shape[1]
    centre_idx = np.argwhere((t_win >= -half) & (t_win <= half)).ravel()
    centre_mask = np.zeros(L, bool); centre_mask[centre_idx] = True

    for i in range(n_perm):
        k = 0
        for row in zs:
            roll = rng.integers(L)
            shuf = np.roll(row, roll)
            thr  = np.median(shuf) + peak_thr * median_abs_deviation(shuf)
            idx, _ = find_peaks(shuf, height=thr, distance=distance)
            if idx.size:
                k += int(np.sum(centre_mask[idx]))
        null[i] = k
    p_perm = float(np.mean(null >= k_in)) if n_perm > 0 else np.nan

    return {
        "N_events": N,
        "k_in": k_in,
        "p0": p0,
        "prop_in": k_in / max(N, 1),
        "p_binom": p_binom,
        "p_perm": p_perm,
        "null_counts": null,
    }, all_times

#plot helper
def _five_ticks(lo, hi):
    ticks = np.linspace(lo, hi, 5)
    ticks[np.isclose(ticks, 0)] = 0.0   # avoid -0.00
    return ticks

# SciPy binomial (compat shim)
try:
    from scipy.stats import binomtest
    def _binom_one_sided(k, n, p=0.5): return binomtest(k, n, p, alternative="greater").pvalue
except Exception:
    from scipy.stats import binom_test
    def _binom_one_sided(k, n, p=0.5): return binom_test(k, n, p, alternative="greater")

def pre_post_asymmetry_from_zscores(ripple_band_aligned, zscores_aligned,
                                    fs=10_000, time_window=(-0.1, 0.1),
                                    analysis_width=0.04,
                                    peak_thr=3.0, distance=20,
                                    n_perm=5000, seed=0, out_png=None,
                                    label_fs=18, tick_fs=18, text_fs=18,
                                    event_times_per_epoch=None):   # NEW
    """
    Tests whether events are more frequent *before* than *after* the ripple peak
    inside a symmetric analysis window, and draws a 3-panel figure (ripple on top).
    Also saves a paired violin/dot/line stats figure for pre vs post.
    """

    # --- time crop
    n_ep, n_samp = zscores_aligned.shape
    mid = n_samp // 2
    t = (np.arange(n_samp) - mid) / fs
    mask = (t >= time_window[0]) & (t <= time_window[1])
    t_win = t[mask]
    zs    = zscores_aligned[:, mask]
    rb    = ripple_band_aligned[:, mask]

    # --- mean ripple ±95% CI
    rb_mean = rb.mean(0)
    rb_ci   = sem(rb, axis=0) * 1.96

    # --- detect events and gather times + per-epoch counts in analysis window
    half = analysis_width / 2.0

    # We will fill these consistently in either branch:
    per_epoch_times = []      # events (times in seconds) that pass the ±half selection
    peak_times_list = []      # pooled for raster/hist
    raster_rows   = []        # matching epoch indices for scatter

    if event_times_per_epoch is None:
        # Fallback: detect within the display time_window (original behaviour)
        for e, tr in enumerate(zs):
            thr = np.median(tr) + peak_thr * median_abs_deviation(tr)
            idx, _ = find_peaks(tr, height=thr, distance=distance)
            if idx.size:
                times_e_full = t_win[idx]
                # keep only within ±half and not exactly zero
                sel = (times_e_full >= -half) & (times_e_full <= half) & (times_e_full != 0)
                times_sel = times_e_full[sel]
            else:
                times_sel = np.array([], dtype=float)

            per_epoch_times.append(times_sel)
            if times_sel.size:
                peak_times_list.append(times_sel)
                raster_rows.append(np.full(times_sel.size, e))

    else:
        # Reuse shared detections and ONLY subset to ±half window here
        for e, times_all in enumerate(event_times_per_epoch):
            times_sel = times_all[(times_all >= -half) & (times_all <= half) & (times_all != 0)]
            per_epoch_times.append(times_sel)
            if times_sel.size:
                peak_times_list.append(times_sel)
                raster_rows.append(np.full(times_sel.size, e))

    # Concatenate pooled arrays for plotting
    if peak_times_list:
        peak_times = np.concatenate(peak_times_list, axis=0)
        raster_y   = np.concatenate(raster_rows,   axis=0)
    else:
        peak_times = np.array([], dtype=float)
        raster_y   = np.array([], dtype=float)

    # Per-epoch pre/post counts (paired)
    pre_counts  = np.array([np.sum(t < 0) for t in per_epoch_times], dtype=float)
    post_counts = np.array([np.sum(t > 0) for t in per_epoch_times], dtype=float)

    # Pooled counts
    k_pre  = int(np.sum(pre_counts))
    k_post = int(np.sum(post_counts))
    N_c    = k_pre + k_post

    # Sanity: ensure x/y lengths match for scatter even when empty
    assert peak_times.size == raster_y.size, \
        f"Raster mismatch: x={peak_times.size}, y={raster_y.size}"

    # --- stats (unchanged) ---------------------------------------------------
    p_binom = _binom_one_sided(k_pre, N_c, p=0.5) if N_c > 0 else 1.0

    diffs = pre_counts - post_counts
    nz = diffs != 0
    if nz.sum() >= 5:
        stat_w, p_wil = wilcoxon(diffs[nz], alternative='greater', zero_method='wilcox')
    else:
        pos = np.sum(diffs[nz] > 0); tot = int(nz.sum())
        p_wil = _binom_one_sided(pos, tot, p=0.5) if tot > 0 else 1.0

    rng = np.random.default_rng(seed)
    if n_perm and diffs.size:
        null = np.empty(n_perm)
        for i in range(n_perm):
            signs = rng.choice([-1, 1], size=diffs.size)
            null[i] = np.sum(diffs * signs)
        p_perm = float(np.mean(null >= diffs.sum()))
    else:
        p_perm = np.nan

    prop_pre = k_pre / max(N_c, 1)
    prop_post= k_post/ max(N_c, 1)
    diff_prop= prop_pre - prop_post

    stats = {
        "k_pre": k_pre, "k_post": k_post, "N_central": N_c,
        "prop_pre": prop_pre, "prop_post": prop_post, "diff_prop": diff_prop,
        "p_binom": float(p_binom), "p_wilcoxon": float(p_wil), "p_perm": float(p_perm),
        "per_epoch_pre": pre_counts, "per_epoch_post": post_counts
    }

    # --- figure (existing 3-panel) ------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 9),
                             gridspec_kw={'height_ratios':[1, 2, 1]},
                             sharex=True)

    # 1) ripple-band
    ax0 = axes[0]
    ax0.plot(t_win, rb_mean, color='black')
    ax0.fill_between(t_win, rb_mean - rb_ci, rb_mean + rb_ci, color='gray', alpha=0.3)
    ax0.set_ylabel("LFP (µV)", fontsize=label_fs)
    ax0.tick_params(axis='both', labelsize=tick_fs)
    ax0.spines[['top','right']].set_visible(False)

    # highlight analysis window + zero line on all axes
    for ax in axes:
        ax.axvspan(-half, half, color='k', alpha=0.06, zorder=0)
        ax.axvline(0.0, color='k', lw=1, alpha=0.6)

    # 2) raster
    ax1 = axes[1]
    ax1.scatter(peak_times, raster_y, s=8)
    ax1.set_ylabel("Epoch", fontsize=label_fs)
    ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.spines[['top','right']].set_visible(False)

    # 3) histogram
    ax2 = axes[2]
    ax2.hist(peak_times, bins=40)
    ax2.set_xlabel("Time from ripple peak (s)", fontsize=label_fs)
    ax2.set_ylabel("Event count", fontsize=label_fs)
    ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.spines[['top','right']].set_visible(False)
    
    for ax in axes:
        ax.set_xlim(time_window[0], time_window[1])
    ticks = _five_ticks(time_window[0], time_window[1])
    axes[-1].set_xticks(ticks)
    axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # annotation (print kept the same below)
    txt = (f"Analysis window = ±{half*1000:.0f} ms  (grey band)\n"
           f"Pre/Post = {k_pre}/{k_post}  (Δprop = {diff_prop*100:.1f}%)\n"
           f"Binomial p = {p_binom:.2e} | Wilcoxon p = {p_wil:.2e} | Perm p = {p_perm:.2e}")

    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, transparent=True)

    # ---- keep the print EXACTLY as before ----------------------------------
    print(txt)

    # ---- NEW: paired violin + dots + lines for pre vs post -----------------
    # (equal window widths on both sides ⇒ counts are comparable)
    if out_png:
        stats_path = os.path.join(os.path.dirname(out_png), "pre_vs_post_stats.png")
    else:
        # default to current working directory
        stats_path = "pre_vs_post_stats.png"

    fig2, ax = plt.subplots(figsize=(6.8, 5.2), dpi=150)

    data_stack = [pre_counts.astype(float), post_counts.astype(float)]
    parts = ax.violinplot(data_stack, positions=[1, 2], showmeans=False, showextrema=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_alpha(0.4)

    # Paired dots + lines
    x1 = np.full_like(pre_counts, 1.0, dtype=float)
    x2 = np.full_like(post_counts, 2.0, dtype=float)
    ax.plot(np.c_[x1, x2].T, np.c_[pre_counts, post_counts].T, alpha=0.3)
    ax.scatter(x1, pre_counts, s=18, zorder=3)
    ax.scatter(x2, post_counts, s=18, zorder=3)

    # Means ± SEM as bars on top (optional)
    m1, m2 = np.nanmean(pre_counts), np.nanmean(post_counts)
    se1 = sem(pre_counts, nan_policy='omit')
    se2 = sem(post_counts, nan_policy='omit')
    ax.errorbar([1,2], [m1,m2], yerr=[se1,se2], fmt='s-', lw=2, capsize=4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Pre\n(−{half*1000:.0f}→0 ms)",
                        f"Post\n(0→+{half*1000:.0f} ms)"],
                       fontsize=tick_fs)
    ax.set_ylabel("Event count (per epoch)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.spines[['top','right']].set_visible(False)

    # p-values
    ax.set_title(f"Pre vs post (per-epoch counts)\n"
                 f"Wilcoxon p={p_wil:.2f}, Perm p={p_perm:.2f}",
                 fontsize=label_fs, pad=8)

    fig2.tight_layout()
    fig2.savefig(stats_path, dpi=300, transparent=True)

    return stats, fig



def _per_epoch_rmi_events(aligned_zscores,
                          fs=10_000, ripple_dur=0.06,
                          thresh_factor=3.0, distance=80):
    """
    RMI per *epoch* (row) using peak counts inside vs outside the central ripple window.
    Returns arrays (one value per epoch) and the masks used.
    """
    n_ep, n_samp = aligned_zscores.shape
    mid = n_samp // 2
    half_w = int((ripple_dur/2) * fs)

    in_mask = np.zeros(n_samp, bool)
    in_mask[mid-half_w:mid+half_w+1] = True
    out_mask = ~in_mask
    T_out   = (n_samp/fs) - ripple_dur

    per_epoch_rmi  = np.empty(n_ep); per_epoch_rmi[:]  = np.nan
    per_epoch_in   = np.zeros(n_ep, dtype=int)
    per_epoch_out  = np.zeros(n_ep, dtype=int)

    for i, row in enumerate(aligned_zscores):
        thr = np.median(row) + thresh_factor * median_abs_deviation(row)
        peaks, _ = find_peaks(row, height=thr, distance=distance)
        n_in  = int(np.sum(in_mask[peaks]))
        n_out = int(np.sum(out_mask[peaks]))
        per_epoch_in[i], per_epoch_out[i] = n_in, n_out

        # convert to rates to remove window-length bias
        in_rate  = n_in  / max(ripple_dur, 1e-12)
        out_rate = n_out / max(T_out,      1e-12)
        denom = in_rate + out_rate
        per_epoch_rmi[i] = 0.0 if denom == 0 else (in_rate - out_rate) / denom

    return per_epoch_rmi, per_epoch_in, per_epoch_out, in_mask

def plot_rmi_null_distribution(rmi_null, rmi_obs,
                               out_png=None, bins=40,
                               title=None,
                               title_fs=24, label_fs=22, tick_fs=20,
                               legend_fs=18):
    """
    Standalone histogram of the shuffled RMI (null) with the observed RMI
    marked as a red vertical line.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if title is None:
        title = "Ripple Modulation Index (RMI) – null distribution"

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax.hist(rmi_null, bins=bins, alpha=0.9,color='gray', edgecolor='dimgray')
    ax.axvline(rmi_obs, color='r', lw=3, label=f"Observed RMI = {rmi_obs:.3f}")

    ax.set_title(title, fontsize=title_fs, pad=10)
    ax.set_xlabel("RMI (shuffled)", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.legend(frameon=False, fontsize=legend_fs)

    fig.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, transparent=True)
    return fig


def _annotate_pooled_rmi(ax_hist,
                         aligned_zscores,
                         fs=10_000,
                         ripple_dur=0.06,
                         thresh_factor=3.0,
                         distance=80,
                         n_shuffle=2000,
                         seed=0,
                         save_pkl=None,
                         print_summary=True):
    """
    Compute pooled RMI (event-based) + p_perm and (optionally) save pooled
    and per-epoch RMI to a .pkl. No inset is drawn here.
    """
    rmi_real, p_perm, rmi_null = Ripple.ripple_modulation_index_events(
        aligned_zscores, fs=fs, ripple_dur=ripple_dur,
        thresh_factor=thresh_factor, distance=distance,
        n_shuffle=n_shuffle, seed=seed
    )

    # per-epoch RMIs
    per_epoch_rmi, per_epoch_in, per_epoch_out, _ = _per_epoch_rmi_events(
        aligned_zscores, fs=fs, ripple_dur=ripple_dur,
        thresh_factor=thresh_factor, distance=distance
    )

    # annotate text on the main histogram axis
    # ax_hist.text(0.98, 0.02,
    #              f"RMI={rmi_real:.3f}\nperm p={p_perm:.2f}",
    #              ha='right', va='bottom', transform=ax_hist.transAxes)

    if print_summary:
        import numpy as np
        print(f"[RMI] pooled={rmi_real:.3f}, perm p={p_perm:.3g}, "
              f"per-epoch mean={np.nanmean(per_epoch_rmi):.2e}, "
              f"median={np.nanmedian(per_epoch_rmi):.2e}, "
              f"n_epochs={len(per_epoch_rmi)}")

    results = {
        "pooled_RMI": float(rmi_real),
        "pooled_perm_p": float(p_perm),
        "null_RMI": rmi_null,
        "per_epoch_RMI": per_epoch_rmi,
        "per_epoch_in_counts": per_epoch_in,
        "per_epoch_out_counts": per_epoch_out,
        "fs": fs, "ripple_dur": ripple_dur,
        "thresh_factor": thresh_factor, "distance": distance,
        "n_shuffle": n_shuffle, "seed": seed
    }
    if save_pkl:
        import os, pickle
        os.makedirs(os.path.dirname(save_pkl), exist_ok=True)
        with open(save_pkl, "wb") as f:
            pickle.dump(results, f)

    return results


def pooled_timing_hist(ripple_band_aligned, zscores_aligned, fs=10_000,
                       time_window=(-0.1, 0.1),
                       center_width=0.02, outer_width=0.04,
                       peak_thr=3.0, distance=20,
                       n_bins=40, n_perm=2000,
                       OUTPUT_DIR=None, out_png=None,
                       label_fs=18, tick_fs=18, text_fs=18,
                       event_times_per_epoch=None):    # optional shared detections
    """
    Pooled raster + histogram with filtered ripple (mean ±95% CI) on top.
    Tests whether the centre band (±center_width/2) has more optical events
    than the two flanks within the outer band ([-outer/2,-center/2] ∪ [center/2,outer/2]).
    If `event_times_per_epoch` is provided, those events are reused (subsetted to
    `time_window`) so N will match other panels that use the same list.
    """
    from scipy.stats import wilcoxon, binomtest

    # --- time crop + data slices ------------------------------------------------
    n_ep, n_samp = zscores_aligned.shape
    mid = n_samp // 2
    t_full = (np.arange(n_samp) - mid) / fs
    mask = (t_full >= time_window[0]) & (t_full <= time_window[1])
    t_win = t_full[mask]
    zs    = zscores_aligned[:, mask]
    rb    = ripple_band_aligned[:, mask]

    # mean ripple ±95% CI
    rb_mean = rb.mean(0)
    rb_ci   = sem(rb, axis=0) * 1.96

    # --- events per epoch (seconds, relative to t=0) ----------------------------
    inner_h = center_width / 2.0
    outer_h = outer_width  / 2.0

    per_epoch_times = []
    peak_times_list, raster_rows = [], []

    if event_times_per_epoch is None:
        # Detect once within the display window (backward-compatible path)
        for e, tr in enumerate(zs):
            thr = np.median(tr) + median_abs_deviation(tr)
            thr = np.median(tr) + peak_thr * median_abs_deviation(tr)
            idx, _ = find_peaks(tr, height=thr, distance=distance)
            times_e = t_win[idx] if idx.size else np.array([], dtype=float)
            per_epoch_times.append(times_e)
            if times_e.size:
                peak_times_list.append(times_e)
                raster_rows.append(np.full(times_e.size, e))
    else:
        # Reuse shared detections; keep only those within the displayed window
        for e, times_all in enumerate(event_times_per_epoch):
            times_e = times_all[(times_all >= time_window[0]) & (times_all <= time_window[1])]
            per_epoch_times.append(times_e)
            if times_e.size:
                peak_times_list.append(times_e)
                raster_rows.append(np.full(times_e.size, e))

    if peak_times_list:
        peak_times = np.concatenate(peak_times_list, axis=0)
        raster_y   = np.concatenate(raster_rows,   axis=0)
    else:
        peak_times = np.array([], dtype=float)
        raster_y   = np.array([], dtype=float)

    # --- main pooled figure ------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(8, 9),
                             gridspec_kw={'height_ratios':[1, 2, 1]},
                             sharex=True)

    # 1) ripple-band
    ax0 = axes[0]
    ax0.plot(t_win, rb_mean, color='black')
    ax0.fill_between(t_win, rb_mean - rb_ci, rb_mean + rb_ci, color='gray', alpha=0.3)
    ax0.set_ylabel("LFP (µV)", fontsize=label_fs)
    ax0.tick_params(axis='both', labelsize=tick_fs)
    ax0.spines[['top','right']].set_visible(False)

    # shaded bands + zero line
    for ax in axes:
        ax.axvspan(-outer_h,  outer_h,  color='k', alpha=0.06, zorder=0)  # outer
        ax.axvspan(-inner_h,  inner_h,  color='k', alpha=0.12, zorder=1)  # inner
        ax.axvline(0.0, color='k', lw=1, alpha=0.6)

    # 2) raster
    ax1 = axes[1]
    ax1.scatter(peak_times, raster_y, s=8)
    ax1.set_ylabel("Epoch", fontsize=label_fs)
    ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.spines[['top','right']].set_visible(False)

    # 3) histogram
    ax2 = axes[2]
    ax2.hist(peak_times, bins=n_bins)
    ax2.set_xlabel("Time from ripple peak (s)", fontsize=label_fs)
    ax2.set_ylabel("Event count", fontsize=label_fs)
    ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.spines[['top','right']].set_visible(False)

    for ax in axes:
        ax.set_xlim(time_window[0], time_window[1])
    axes[-1].set_xticks(_five_ticks(time_window[0], time_window[1]))
    axes[-1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=300, transparent=True)

    # --- centre vs flanks: per-epoch COUNTS + pooled counts ---------------------
    per_count_c, per_count_f = [], []
    k_c = 0; k_f = 0
    for times_e in per_epoch_times:
        if times_e.size == 0:
            per_count_c.append(0.0); per_count_f.append(0.0); continue
        n_c = np.sum((times_e >= -inner_h) & (times_e <=  inner_h))
        n_l = np.sum((times_e >= -outer_h) & (times_e <  -inner_h))
        n_r = np.sum((times_e >   inner_h) & (times_e <=  outer_h))
        n_f = n_l + n_r
        k_c += int(n_c); k_f += int(n_f)
        per_count_c.append(float(n_c)); per_count_f.append(float(n_f))

    per_count_c = np.asarray(per_count_c, float)
    per_count_f = np.asarray(per_count_f, float)

    # stats (one-sided: centre > flanks)
    diffs = per_count_c - per_count_f
    if np.any(diffs != 0):
        p_wil = wilcoxon(per_count_c, per_count_f, alternative='greater',
                         zero_method='wilcox').pvalue
    else:
        p_wil = 1.0

    rng = np.random.default_rng(0)
    null = np.empty(n_perm)
    s_obs = np.sum(diffs)
    for i in range(n_perm):
        null[i] = np.sum(diffs * rng.choice([-1, 1], size=diffs.size))
    p_perm_pair = float(np.mean(null >= s_obs))

    N_cf = int(k_c + k_f)
    p_binom = binomtest(k_c, N_cf, p=0.5, alternative='greater').pvalue if N_cf > 0 else 1.0

    print(f"[3a] Centre (±{inner_h*1000:.0f} ms) vs flanks (total { (outer_width-center_width)*1000:.0f} ms): "
          f"k_c={k_c}, k_f={k_f}, N={N_cf}")
    print(f"[3a] Wilcoxon p={p_wil:.2e}, Perm p={p_perm_pair:.3e}, Binomial p={p_binom:.2e}")

    # --- separate stats figures (counts violin + pooled bar) --------------------
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # (A) Violin + paired dots/lines of per-epoch counts
        fig_stat, ax = plt.subplots(figsize=(6.8, 5.2), dpi=150)
        data_stack = [per_count_c, per_count_f]
        parts = ax.violinplot(data_stack, positions=[1, 2],
                              showmeans=False, showextrema=False, widths=0.8)
        for pc in parts['bodies']:
            pc.set_alpha(0.4)

        x1 = np.full_like(per_count_c, 1.0, dtype=float)
        x2 = np.full_like(per_count_f, 2.0, dtype=float)
        ax.plot(np.c_[x1, x2].T, np.c_[per_count_c, per_count_f].T, alpha=0.3)
        ax.scatter(x1, per_count_c, s=18, zorder=3)
        ax.scatter(x2, per_count_f, s=18, zorder=3)

        m1, m2 = np.nanmean(per_count_c), np.nanmean(per_count_f)
        se1 = sem(per_count_c, nan_policy='omit')
        se2 = sem(per_count_f, nan_policy='omit')
        ax.errorbar([1, 2], [m1, m2], yerr=[se1, se2], fmt='s-', lw=2, capsize=4)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Centre\n(±{inner_h*1000:.0f} ms)",
                            f"Flanks\n(total {(outer_width-center_width)*1000:.0f} ms)"],
                           fontsize=tick_fs)
        ax.set_ylabel("Event count (per epoch)", fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(f"Centre vs flanks (per-epoch counts)\n"
                     f"Wilcoxon p={p_wil:.2f}, Perm p={p_perm_pair:.2f}, Binom p={p_binom:.2f}",
                     fontsize=label_fs, pad=8)

        fig_stat.tight_layout()
        fig_stat.savefig(os.path.join(OUTPUT_DIR, "centre_vs_flanks_stats.png"),
                         dpi=300, transparent=True)

        # (B) Pooled counts bar chart
        fig_cnt, axc = plt.subplots(figsize=(4.4, 3.8), dpi=150)
        axc.bar([0, 1], [k_c, k_f])
        axc.set_xticks([0, 1]); axc.set_xticklabels(["Centre", "Flanks"])
        axc.set_ylabel("Count", fontsize=label_fs)
        axc.tick_params(labelsize=tick_fs)
        axc.spines[['top','right']].set_visible(False)
        axc.set_title("Pooled counts", fontsize=label_fs, pad=6)
        fig_cnt.tight_layout()
        fig_cnt.savefig(os.path.join(OUTPUT_DIR, "centre_vs_flanks_counts.png"),
                        dpi=300, transparent=True)

    # --- existing enrichment + RMI parts (kept as in your pipeline) --------------
    rmi_payload = _annotate_pooled_rmi(
        ax_hist=None,
        aligned_zscores=zscores_aligned,
        fs=fs, ripple_dur=0.06,
        thresh_factor=3.0, distance=20,
        n_shuffle=2000, seed=0,
        save_pkl=os.path.join(OUTPUT_DIR, "pooled_ripple_RMI.pkl") if OUTPUT_DIR else None,
        print_summary=True
    )

    plot_rmi_null_distribution(
        rmi_payload["null_RMI"],
        rmi_payload["pooled_RMI"],
        out_png=os.path.join(OUTPUT_DIR, "pooled_ripple_RMI_null.png") if OUTPUT_DIR else None,
        title=f"RMI null (perm p={rmi_payload['pooled_perm_p']:.2f})",
        title_fs=26, label_fs=22, tick_fs=20, legend_fs=20
    )

    stats, all_times = peak_enrichment_test(
        zscores_aligned, fs=fs, time_window=time_window, center_width=center_width,
        peak_thr=peak_thr, distance=distance, n_perm=n_perm
    )
    print(f"Central-window enrichment: k_in={stats['k_in']}/{stats['N_events']} "
          f"(p_binom={stats['p_binom']:.2e}, p_perm={stats['p_perm']:.3e})")

    return all_times, stats, fig



def pooled_phase_preference(ripple_band_aligned, zscores_aligned,
                            fs=10_000,
                            core_window=0.04,              # visual LFP core, e.g. ±20 ms
                            bins=18, peak_thr=3.0, distance=15,
                            out_png=None, title_fs=20, tick_fs=18,
                            bar_color="#1b9e77", edge_color="k",
                            event_times_per_epoch=None,     # REQUIRED for consistent n
                            selection_window=(-0.02, 0.02)  # events to include in Rayleigh
                            ):
    from scipy.signal import hilbert

    assert event_times_per_epoch is not None, \
        "Pass event_times_per_epoch so 3a/3b/3c use the SAME events."

    n_ep, n_samp = ripple_band_aligned.shape
    mid = n_samp // 2
    t_full = (np.arange(n_samp) - mid) / fs

    # --- Build LFP CORE for phase (visual context)
    core_lo, core_hi = -core_window/2, core_window/2
    core_mask = (t_full >= core_lo) & (t_full <= core_hi)
    lfp_core = ripple_band_aligned[:, core_mask]

    # bandpass + Hilbert on the core
    lfp_filt = _butter_bandpass(lfp_core, fs, 130, 250, order=4, axis=1)
    phase = np.angle(hilbert(lfp_filt, axis=1)) % (2*np.pi)

    # align so trough = 0 phase
    trough_idx = np.argmin(lfp_filt, axis=1)
    trough_phase = phase[np.arange(phase.shape[0]), trough_idx]
    phase = (phase.T - trough_phase).T % (2*np.pi)

    # --- Select EXACT events to analyse
    sel_lo, sel_hi = selection_window
    core_start_s = core_lo  # start of core in seconds
    core_len = lfp_core.shape[1]

    peak_phases = []
    selected_count = 0
    for e, times_all in enumerate(event_times_per_epoch):
        # keep only the events inside the SELECTION window
        sel_times = times_all[(times_all >= sel_lo) & (times_all <= sel_hi)]
        if sel_times.size == 0:
            continue
        selected_count += sel_times.size

        # map times (s) -> indices within the CORE array
        idx_core = np.round((sel_times - core_start_s) * fs).astype(int)
        # keep only indices that fall inside the core (safe guard)
        keep = (idx_core >= 0) & (idx_core < core_len)
        if np.any(keep):
            peak_phases.extend(phase[e, idx_core[keep]])

    peak_phases = np.asarray(peak_phases)
    print(f"[3c] Phase-mapped events within selection_window {selection_window}: {peak_phases.size}")

    # sanity: the Rayleigh n should equal the number of accepted phases
    assert peak_phases.size == selected_count or peak_phases.size <= selected_count, \
        f"Lost events after indexing: have phases={peak_phases.size}, selected={selected_count}"

    # --- Rayleigh test
    n = peak_phases.size
    if n == 0:
        raise RuntimeError("No events available in selection_window to compute phase.")

    C = np.sum(np.cos(peak_phases))
    S = np.sum(np.sin(peak_phases))
    R = np.hypot(C, S)
    z = (R**2) / n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n))  # small-angle approx

    # --- Polar histogram
    bin_edges = np.linspace(0, 2*np.pi, bins+1)
    counts, _ = np.histogram(peak_phases, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(centers, counts, width=2*np.pi/bins, alpha=0.75,
           color=bar_color, edgecolor=edge_color)
    ax.set_title(f"Optical event phase (n={n})\nRayleigh z={z:.2f}, p={p:.3g}",
                 va="bottom", fontsize=title_fs, pad=12)
    ax.tick_params(axis='both', labelsize=tick_fs)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(tick_fs)
    ax.spines['polar'].set_linewidth(2)

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=300, transparent=True)

    return {"z": float(z), "p": float(p), "n_events": int(n),
            "counts": counts, "centers": centers, "figure": fig}




# ---------- High-level batch entrypoints ------------------------------------
def main_batch():
    """
    1) Process every SyncRecording* under `parent_dir`.
    2) Pool all saved aligned epochs.
    3) Make pooled timing histogram & phase preference plot.
    """
    parent_dir  = r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\AwakeStationary'  # <-- set your parent
    LFP_channel = "LFP_3"
    savename    = "RippleSave"

    '# Step 1: run all recordings'
    process_parent_folder(parent_dir, LFP_channel=LFP_channel, savename=savename, theta_cutoff=0.1)

    '# Step 2: pool'
    bp_all, bb_all, zs_all = load_all_aligned_epochs(parent_dir, savename_glob=savename, LFP_channel=LFP_channel)

    '# Step 3a: pooled timing preferences (relative to peak)'
    pooled_dir = os.path.join(parent_dir, "RipplePooled")
    os.makedirs(pooled_dir, exist_ok=True)
    # Detect once on ±0.1 s (same for all)
    evt_times = detect_peaks_times(zs_all, fs=10_000, detect_window=(-0.1, 0.1),
                                   peak_thr=3.0, distance=15)
    
    #3a: centre vs flanks (uses only |t|<=0.02 via centre+flanks union)
    peak_times, stats, fig = pooled_timing_hist(
        ripple_band_aligned=bp_all,
        zscores_aligned=zs_all,
        fs=10_000,
        time_window=(-0.1, 0.1),
        center_width=0.02,
        peak_thr=3.0,
        distance=15,
        n_bins=40,
        n_perm=2000,
        OUTPUT_DIR=pooled_dir,
        out_png=os.path.join(pooled_dir, "pooled_optical_timing_with_ripple.png"),
        event_times_per_epoch=evt_times,            # NEW
    )
    
    # '# 3b: pre vs post within ±20 ms'
    # stats_pp, fig_pp = pre_post_asymmetry_from_zscores(
    #     ripple_band_aligned=bp_all,
    #     zscores_aligned=zs_all,
    #     fs=10_000,
    #     time_window=(-0.1, 0.1),
    #     analysis_width=0.04,   # ±20 ms
    #     peak_thr=3.0,
    #     distance=15,
    #     n_perm=5000,
    #     out_png=os.path.join(pooled_dir, "pre_vs_post_events.png"),
    #     event_times_per_epoch=evt_times,            # NEW
    # )
    
    
    # '# 3c using EXACTLY the same ±20 ms events used in 3b'
    # stats_phase = pooled_phase_preference(
    #     bp_all, zs_all, fs=10_000,
    #     core_window=0.06,                     # visual context (±30 ms) if you like
    #     bins=18,
    #     event_times_per_epoch=evt_times,      # reuse
    #     selection_window=(-0.02, 0.02),       # SAME window as 3b
    #     out_png=os.path.join(pooled_dir, "pooled_optical_phase.png")
    # )
    # print(f"Phase preference: Rayleigh z={stats_phase['z']:.2f}, "
    #       f"p={stats_phase['p']:.3g}, n_events={stats_phase['n_events']}")
 
if __name__ == "__main__":
    main_batch()
