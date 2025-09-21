# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:04:10 2024

@author: Yifang
"""
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import sem
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm

def align_ripples (lfps,zscores,start_idx,end_idx,midpoint,Fs=10000):
    aligned_ripple_band_lfps = np.zeros_like(lfps)
    aligned_lfps=np.zeros_like(lfps)
    aligned_zscores=np.zeros_like(lfps)
    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 18))
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 18))
    for i in range(lfps.shape[0]):
        lfp_i=lfps[i]
        zscore_i=zscores[i]
        LFP_ripple_band_i=OE.band_pass_filter(lfps[i], 130, 250, Fs)
        # Find the index of the maximum value in the segment [1000:3000]
        local_max_idx = np.argmax(LFP_ripple_band_i[start_idx:end_idx]) + start_idx
        # Calculate the shift needed to align the max value to the midpoint
        shift = midpoint - local_max_idx  
        # Roll the trace to align the max value to the center
        aligned_ripple_lfp_i = np.roll(LFP_ripple_band_i, shift)   
        aligned_lfp_i=np.roll(lfp_i, shift)   
        aligned_zscore_i=np.roll(zscore_i, shift)
        # Store the aligned trace
        aligned_ripple_band_lfps[i] = aligned_ripple_lfp_i
        aligned_lfps[i]=aligned_lfp_i
        aligned_zscores[i]=aligned_zscore_i
        ax1[0].plot(lfp_i)
        ax1[1].plot(zscore_i)
        ax1[2].plot(LFP_ripple_band_i)
        ax2[0].plot(aligned_lfp_i)
        ax2[1].plot(aligned_zscore_i)
        ax2[2].plot(aligned_ripple_lfp_i)
    return aligned_ripple_band_lfps,aligned_lfps,aligned_zscores
    
def plot_ripple_heatmap_noColorbar(ripple_band_lfps,lfps,zscores,Fs=10000):
    ripple_band_lfps_mean,ripple_band_lfps_std, ripple_band_lfps_CI=OE.calculateStatisticNumpy (ripple_band_lfps)
    lfps_mean,lfps_std, lfps_CI=OE.calculateStatisticNumpy (lfps)
    zscores_mean,zscores_std, zscores_CI=OE.calculateStatisticNumpy (zscores)
    
    time = np.linspace((-len(lfps_mean)/2)/Fs, (len(lfps_mean)/2)/Fs, len(lfps_mean))  
    
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]}, figsize=(8, 16))
    axs[0].plot(time, ripple_band_lfps_mean, color='#404040', label='Ripple Band Mean')
    axs[0].fill_between(time, ripple_band_lfps_CI[0], ripple_band_lfps_CI[1], color='#404040', alpha=0.2, label='0.95 CI')
    axs[1].plot(time, lfps_mean, color='dodgerblue', label='Ripple LFP Mean')
    axs[1].fill_between(time, lfps_CI[0], lfps_CI[1], color='dodgerblue', alpha=0.2, label='0.95 CI')
    axs[2].plot(time, zscores_mean, color='limegreen', label='Ripple Zscore Mean')#tomato
    axs[2].fill_between(time, zscores_CI[0], zscores_CI[1], color='limegreen', alpha=0.2, label='0.95 CI')
    axs[0].set_title('Averaged Ripple Epoch',fontsize=18)
    for i in range(3):
        axs[i].set_xlim(time[0], time[-1])
        axs[i].margins(x=0)  # Remove any additional margins on x-axis
        #axs[i].legend()
        # Remove the frame (spines) from the first three plots
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].get_yaxis().set_visible(False)  # Opt
    axs[0].tick_params(labelbottom=False, bottom=False)  # Remove x-ticks and labels for axs[0]
    axs[1].tick_params(labelbottom=False, bottom=False)  # Remove x-ticks and labels for axs[1]
              
    sns.heatmap(lfps, cmap="viridis", ax=axs[3], cbar=False)
    #axs[3].set_title('Heatmap of LFPs',fontsize=24)
    axs[3].set_ylabel('Epoch Number',fontsize=20)
    
    sns.heatmap(zscores, cmap="viridis", ax=axs[4], cbar=False)
    #axs[4].set_title('Heatmap of Zscores',fontsize=24)
    axs[4].set_ylabel('Epoch Number',fontsize=20)
    axs[3].tick_params(axis='both', which='major', labelsize=16, rotation=0)  # Adjust the size as needed
    axs[4].tick_params(axis='both', which='major', labelsize=16, rotation=0)  # Adjust the size as needed
    axs[3].tick_params(labelbottom=False, bottom=False)
    axs[4].tick_params(labelbottom=False, bottom=False)
    plt.tight_layout()
    plt.show()
    return fig

def plot_ripple_heatmap(ripple_band_lfps, lfps, zscores, Fs=10000):
    ripple_band_lfps_mean, ripple_band_lfps_std, ripple_band_lfps_CI = OE.calculateStatisticNumpy(ripple_band_lfps)
    lfps_mean, lfps_std, lfps_CI = OE.calculateStatisticNumpy(lfps)
    zscores_mean, zscores_std, zscores_CI = OE.calculateStatisticNumpy(zscores)
    
    time = np.linspace((-len(lfps_mean)/2)/Fs, (len(lfps_mean)/2)/Fs, len(lfps_mean))  
    
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]}, figsize=(8, 16))
    
    # Plot Ripple band LFP
    axs[0].plot(time, ripple_band_lfps_mean, color='#404040')
    axs[0].fill_between(time, ripple_band_lfps_CI[0], ripple_band_lfps_CI[1], color='#404040', alpha=0.2)
    axs[0].set_ylabel('LFP (μV)', fontsize=16)
    
    # Plot broadband LFP
    axs[1].plot(time, lfps_mean, color='dodgerblue') # dodgerblue
    axs[1].fill_between(time, lfps_CI[0], lfps_CI[1], color='dodgerblue', alpha=0.2)
    axs[1].set_ylabel('LFP (μV)', fontsize=16)
    
    # Plot z-score
    axs[2].plot(time, zscores_mean, color='limegreen') #limegreen
    axs[2].fill_between(time, zscores_CI[0], zscores_CI[1], color='limegreen', alpha=0.2) #
    axs[2].set_ylabel('Zscore', fontsize=16)

    for i in range(3):
        axs[i].set_xlim(time[0], time[-1])
        axs[i].margins(x=0)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].yaxis.tick_right()
        axs[i].yaxis.set_label_position("right")
        axs[i].tick_params(axis='y', labelsize=16)

    axs[0].tick_params(labelbottom=False, bottom=False)
    axs[1].tick_params(labelbottom=False, bottom=False)
    
    # Heatmap of LFPs
    sns.heatmap(lfps, cmap="viridis", ax=axs[3], cbar=False)
    axs[3].set_ylabel('Epoch Number', fontsize=16)
    axs[2].tick_params(axis='x', labelsize=14)
    axs[3].tick_params(axis='both', labelsize=12)
    axs[3].tick_params(labelbottom=False, bottom=False)
    axs[3].tick_params(axis='y', labelrotation=0)        # <- keep y-labels horizontal
    # Heatmap of z-scores
    sns.heatmap(zscores, cmap="viridis", ax=axs[4], cbar=False)
    axs[4].set_ylabel('Epoch Number', fontsize=16)
    axs[4].tick_params(axis='both', labelsize=12)
    axs[4].tick_params(labelbottom=False, bottom=False)
    axs[4].tick_params(axis='y', labelrotation=0)  

    # LFP colourbar
    cbar_ax1 = inset_axes(axs[3],
                          width="2%", height="100%",
                          bbox_to_anchor=(1.05, 0., 1, 1),
                          bbox_transform=axs[3].transAxes,
                          loc='upper left', borderpad=0)
    norm1 = plt.Normalize(np.min(lfps), np.max(lfps))
    sm1 = cm.ScalarMappable(cmap="viridis", norm=norm1)
    cbar1 = plt.colorbar(sm1, cax=cbar_ax1, orientation='vertical')
    cbar1.set_label('LFP (μV)', fontsize=16)
    cbar1.ax.tick_params(labelsize=16)

    # Z-score colourbar
    cbar_ax2 = inset_axes(axs[4],
                          width="2%", height="100%",
                          bbox_to_anchor=(1.05, 0., 1, 1),
                          bbox_transform=axs[4].transAxes,
                          loc='upper left', borderpad=0)
    norm2 = plt.Normalize(np.min(zscores), np.max(zscores))
    sm2 = cm.ScalarMappable(cmap="viridis", norm=norm2)
    cbar2 = plt.colorbar(sm2, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('Z-Score', fontsize=16)
    cbar2.ax.tick_params(labelsize=16)

    plt.tight_layout()
    return fig
def plot_aligned_ripple_save (save_path,LFP_channel,recordingName,ripple_triggered_lfps,ripple_triggered_zscores,Fs=10000):
    os.makedirs(save_path, exist_ok=True)
    'Assume my ripple PETH are all process by OEC ripple detection, Fs=10000, length=4000'
    ripple_sample_numbers=len(ripple_triggered_lfps[0])
    midpoint=ripple_sample_numbers//2
    'align ripple in a 200ms window '
    start_idx=int(midpoint-0.08*Fs)
    end_idx=int(midpoint+0.08*Fs)
    print (midpoint,start_idx,end_idx)
    aligned_ripple_band_lfps,aligned_lfps,aligned_zscores=align_ripples (ripple_triggered_lfps,
                                                                         ripple_triggered_zscores,start_idx,end_idx,midpoint,Fs)
    fig=plot_ripple_heatmap(aligned_ripple_band_lfps,aligned_lfps,aligned_zscores,Fs)
    fig_path = os.path.join(save_path, recordingName+LFP_channel+'Ripple_aligned_heatmap_400ms.png')
    fig.savefig(fig_path, transparent=True)
    
    fig=plot_ripple_heatmap(aligned_ripple_band_lfps[:,start_idx:end_idx],
                            aligned_lfps[:,start_idx:end_idx],aligned_zscores[:,start_idx:end_idx],Fs)
    fig_path = os.path.join(save_path, recordingName+LFP_channel+'Ripple_aligned_heatmap_200ms.png')
    fig.savefig(fig_path, transparent=True)

    
    save_file_path = os.path.join(save_path,'ailgned_ripple_LFP.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(aligned_lfps, file)
    save_file_path = os.path.join(save_path,'ailgned_ripple_bandpass_LFP.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(aligned_ripple_band_lfps, file)
    save_file_path = os.path.join(save_path,'ailgned_ripple_Zscore.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(aligned_zscores, file)
        
    return aligned_ripple_band_lfps,aligned_zscores

def plot_ripple_zscore(savepath,
                       lfp_ripple,          # 2-D  [epochs × samples]
                       zscore,              # 2-D  [epochs × samples]
                       fs=10_000,
                       time_window=(-0.05, 0.05),
                       peak_thr=3,        # median + peak_thr × MAD
                       distance=20):        # min samples between peaks
    """Scatter-raster + histogram for optical peaks during aligned ripples."""

    # --- time axis: crop to window around ripple peak -----------------------
    n_samp  = lfp_ripple.shape[1]
    t_full  = np.arange(n_samp) / fs                    # seconds
    mid_idx = n_samp // 2
    mask    = (t_full - t_full[mid_idx] >= time_window[0]) & \
              (t_full - t_full[mid_idx] <= time_window[1])

    t       = t_full[mask] - t_full[mid_idx]
    lfp_rip = lfp_ripple[:, mask]
    zs      = zscore[:, mask]

    # --- mean ± 95 % CI of ripple-band LFP ----------------------------------
    mean_rip = lfp_rip.mean(0)
    ci_rip   = sem(lfp_rip, axis=0) * 1.96              # 95 % CI

    # --- peak detection for each epoch --------------------------------------
    peak_times, raster_y = [], []
    for epoch, trace in enumerate(zs):
        thr   = np.median(trace) + peak_thr * median_abs_deviation(trace)
        idx,_ = find_peaks(trace, height=thr, distance=distance)
        peak_times.extend(t[idx])
        raster_y.extend([epoch] * len(idx))

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(6, 8),
                             gridspec_kw={'height_ratios':[1,2,1]},
                             sharex=True)

    # 1) mean ripple band
    ax0 = axes[0]
    ax0.plot(t, mean_rip, color='black')
    ax0.fill_between(t, mean_rip-ci_rip, mean_rip+ci_rip,
                     color='gray', alpha=0.3)
    ax0.set_ylabel('LFP (µV)', fontsize=14)
    ax0.tick_params(axis='both', labelsize=12)
    ax0.spines[['top','right']].set_visible(False)

    # 2) raster of optical peaks
    ax1 = axes[1]
    ax1.scatter(peak_times, raster_y, c='red', s=8)
    ax1.set_ylabel('Epoch', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlim(t[0], t[-1])
    ax1.spines[['top','right']].set_visible(False)

    # 3) histogram
    ax2 = axes[2]
    ax2.hist(peak_times, bins=40, color='#377eb8', alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Peak count', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(os.path.join(savepath, 'ripple_optical_raster.png'),
                    dpi=300, transparent=True)
    plt.show()
    return fig

from scipy.signal import find_peaks, hilbert
from scipy.stats  import median_abs_deviation
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from scipy.stats  import median_abs_deviation

# ------------ helpers (unchanged) ------------------------------------------
def butter_bandpass(data, fs, low=130, high=250, order=4, axis=-1):
    b, a = butter(order, [low/(fs*0.5), high/(fs*0.5)], btype='band')
    return filtfilt(b, a, data, axis=axis)

def rayleigh_test(phases):
    n = len(phases)
    R = np.sqrt(np.sum(np.cos(phases))**2 + np.sum(np.sin(phases))**2)
    z = R**2 / n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n))
    return z, p

def event_MI(phases, bins=12, n_perm=2000):
    bin_edges = np.linspace(0, 2*np.pi, bins+1)
    counts, _ = np.histogram(phases, bins=bin_edges)
    prob  = counts / counts.sum()
    mi    = np.sum(prob * np.log((prob+1e-10)/(1/bins))) / np.log(bins)
    null  = np.empty(n_perm)
    for i in range(n_perm):
        shift = np.random.randint(len(phases))
        shuff = (phases + shift) % (2*np.pi)
        c,_   = np.histogram(shuff, bins=bin_edges)
        pnull = c / c.sum()
        null[i] = np.sum(pnull * np.log((pnull+1e-10)/(1/bins))) / np.log(bins)
    p_val = (null >= mi).mean()
    return mi, p_val, null

# ------------ main function -------------------------------------------------
def _crop_core_window(arr2d, fs, core_window):
    """
    Keep only the central ±core_window seconds from each epoch (row).
    """
    n_samp = arr2d.shape[1]
    mid    = n_samp // 2                     # ripple peak index
    half_w = int(core_window * fs)           # samples for 0.05 s
    idx    = slice(mid - half_w, mid + half_w + 1)  # +1 to keep symmetry
    return arr2d[:, idx]                     # same rows, shorter cols

def ripple_phase_stats_polar(aligned_ripple_band_lfps,
                             aligned_zscores,
                             fs=10_000,
                             core_window=0.1,   # ±50 ms => 100 ms total
                             bins=12,
                             n_perm=2000,
                             distance=None,
                             prominence=None):
    """
    Phase-locking stats using only the central ±core_window of each epoch
    and a 130–250 Hz Hilbert phase.
    """

    # 1) take centre 100 ms window ------------------------------------------
    lfp_core = _crop_core_window(aligned_ripple_band_lfps, fs, core_window)
    zs_core  = _crop_core_window(aligned_zscores,         fs, core_window)

    # 2) ripple-band phase ---------------------------------------------------
    lfp_filt = butter_bandpass(lfp_core, fs, 130, 250, order=4, axis=1)
    phase    = np.angle(hilbert(lfp_filt, axis=1)) % (2*np.pi)
    
    # ------- shift phase so that trough = 0 rad ----------------------------
    # Find the index of the negative LFP peak (trough) in the core window
    trough_idx = np.argmin(lfp_filt, axis=1)           # one per epoch
    # Phase at each trough
    trough_phase = phase[np.arange(phase.shape[0]), trough_idx]
    # Subtract trough phase from every sample of that epoch
    phase = (phase.T - trough_phase).T % (2*np.pi)
    # -----------------------------------------------------------------------

    # 3) flatten
    phase_vec = phase.ravel()
    z_vec     = zs_core.ravel()

    # 4) peak detection (median + 2.5×MAD)
    thr = np.median(z_vec) + 2 * median_abs_deviation(z_vec, scale=1.0)
    peaks, _ = find_peaks(z_vec, height=thr,
                          distance=distance, prominence=prominence)
    if len(peaks) < 10:
        raise ValueError("Fewer than 10 peaks detected.")

    peak_phases = phase_vec[peaks]

    # 5) stats ---------------------------------------------------------------
    z_ray, p_ray = rayleigh_test(peak_phases)
    mi, p_mi, _  = event_MI(peak_phases, bins=bins, n_perm=n_perm)

    # 6) polar plot ----------------------------------------------------------
    bin_edges = np.linspace(0, 2*np.pi, bins+1)
    counts, _ = np.histogram(peak_phases, bins=bin_edges)
    centers   = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='polar')

    ax.bar(centers, counts, width=2*np.pi/bins,
           color="#1b9e77", alpha=0.7, edgecolor='black', align='center')

    circ_counts  = np.append(counts, counts[0])
    circ_centres = np.append(centers, centers[0])
    ax.plot(circ_centres, circ_counts, lw=3, color='k')

    ax.plot(0.5, 0.5, marker='o', markersize=6,
            color='black', transform=ax.transAxes, zorder=10)

    ax.set_title(f"Event Phase Histogram (MI={mi:.3f})",
                 va="bottom", fontsize=14, fontweight="bold")

    r_max   = counts.max()
    r_ticks = np.linspace(0, r_max, 4)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{t:.0f}" for t in r_ticks], fontsize=14)
    ax.tick_params(axis='x', labelsize=16)
    ax.grid(False)
    ax.spines['polar'].set_linewidth(3)
    plt.tight_layout()

    return {'Rayleigh_z': z_ray,
            'Rayleigh_p': p_ray,
            'MI': mi,
            'MI_p': p_mi,
            'n_peaks': len(peaks),
            'figure': fig}
from scipy.stats  import bootstrap

def ripple_zscore_polar(aligned_ripple_band_lfps,
                        aligned_zscores,
                        fs=10_000,
                        core_window=0.1,   # ±50 ms
                        bins=30,
                        ci_level=0.95):
    """
    Polar plot of mean z-score vs. ripple phase, using only the central
    ±core_window seconds of each aligned epoch and a 130–250 Hz phase.
    """

    # 1) crop to central window ---------------------------------------------
    lfp_core = _crop_core_window(aligned_ripple_band_lfps, fs, core_window)
    zs_core  = _crop_core_window(aligned_zscores,fs, core_window)

    # 2) ripple-band phase ---------------------------------------------------
    lfp_filt = butter_bandpass(lfp_core, fs, 130, 250, order=4, axis=1)
    phase    = np.angle(hilbert(lfp_filt, axis=1)) % (2*np.pi)
    
    # ------- shift phase so that trough = 0 rad ----------------------------
    # Find the index of the negative LFP peak (trough) in the core window
    trough_idx = np.argmin(lfp_filt, axis=1)           # one per epoch
    # Phase at each trough
    trough_phase = phase[np.arange(phase.shape[0]), trough_idx]
    # Subtract trough phase from every sample of that epoch
    phase = (phase.T - trough_phase).T % (2*np.pi)
    # ------- shift phase so that trough = 0 rad ----------------------------
    # Find the index of the negative LFP peak (trough) in the core window
    trough_idx = np.argmin(lfp_filt, axis=1)           # one per epoch
    # Phase at each trough
    trough_phase = phase[np.arange(phase.shape[0]), trough_idx]
    # Subtract trough phase from every sample of that epoch
    phase = (phase.T - trough_phase).T % (2*np.pi)
    # -----------------------------------------------------------------------

    # 3) flatten -------------------------------------------------------------
    phase_vec = phase.ravel()
    z_vec     = zs_core.ravel()

    # 4) bin statistics ------------------------------------------------------
    bin_edges   = np.linspace(0, 2*np.pi, bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_z, ci_low, ci_high = [], [], []

    for k in range(bins):
        mask = (phase_vec >= bin_edges[k]) & (phase_vec < bin_edges[k+1])
        data = z_vec[mask]
        if len(data) == 0:
            mean_z.extend([0]); ci_low.extend([0]); ci_high.extend([0])
            continue
        mean_z.append(data.mean())
        bs = bootstrap((data,), np.mean, confidence_level=ci_level,
                       n_resamples=2000, method='basic')
        ci_low.append(bs.confidence_interval.low)
        ci_high.append(bs.confidence_interval.high)

    # close loop
    mean_z.append(mean_z[0])
    ci_low.append(ci_low[0])
    ci_high.append(ci_high[0])
    circ_centers = np.append(bin_centers, bin_centers[0])

    # 5) plot ---------------------------------------------------------------
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='polar')

    # centre dot
    ax.plot(0.5, 0.5, marker='o', markersize=6,
            color='black', transform=ax.transAxes, zorder=10)

    # mean + CI
    ax.plot(circ_centers, mean_z, lw=3, color="#1b9e77")
    theta_fill  = np.concatenate([circ_centers, circ_centers[::-1]])
    radius_fill = np.concatenate([ci_high, ci_low[::-1]])
    ax.fill(theta_fill, radius_fill, color="#1b9e77", alpha=0.3)

    # ticks & title
    ax.set_title("Z-score vs. Ripple Phase",
                 va="bottom", fontsize=14, fontweight="bold")

    r_min, r_max = min(ci_low), max(ci_high)
    r_ticks      = np.linspace(r_min, r_max, 4)
    ax.set_rlim(r_min, r_max)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in r_ticks],
                       fontsize=14, color='darkred')

    ax.tick_params(axis='x', labelsize=16)
    ax.grid(False)
    ax.spines['polar'].set_linewidth(3)
    plt.tight_layout()

    return fig
def ripple_modulation_index(aligned_zscores,
                            fs          = 10_000,
                            ripple_dur  = 0.05,      # width (s) regarded as “in-ripple”
                            n_shuffle   = 1000,
                            random_seed = 0):
    """
    Ripple Modulation Index (RMI) for optical data aligned to ripple peaks.

    Parameters
    ----------
    aligned_zscores : 2-D array  [epochs × samples]
        Each row is an optical trace centred on a ripple peak (e.g. 400 ms).
    fs : int
        Sampling rate (Hz).
    ripple_dur : float
        Total duration (s) to treat as the in-ripple window, centred on peak.
        Default 60 ms (±30 ms).
    n_shuffle : int
        Number of circular shuffles for the permutation test.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    rmi_real : float
        Observed Ripple-Modulation Index.
    p_perm : float
        Permutation p-value (fraction of shuffled RMI ≥ real RMI).
    rmi_null : np.ndarray
        Null distribution of shuffled RMIs (length n_shuffle).
    """

    rng = np.random.default_rng(random_seed)
    n_ep, n_samp = aligned_zscores.shape
    mid = n_samp // 2
    half_w = int((ripple_dur/2) * fs)           # samples on either side
    in_mask  = np.zeros(n_samp, dtype=bool)
    in_mask[mid-half_w : mid+half_w+1] = True   # inclusive of centre
    out_mask = ~in_mask

    # -- real RMI ------------------------------------------------------------
    in_rate  = aligned_zscores[:, in_mask ].mean()
    out_rate = aligned_zscores[:, out_mask].mean()
    rmi_real = (in_rate - out_rate) / (in_rate + out_rate + 1e-10)

    # -- permutation ---------------------------------------------------------
    rmi_null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        rolled = np.array([np.roll(row, rng.integers(n_samp)) for row in aligned_zscores])
        in_r   = rolled[:, in_mask ].mean()
        out_r  = rolled[:, out_mask].mean()
        rmi_null[i] = (in_r - out_r) / (in_r + out_r + 1e-10)

    p_perm = (rmi_null >= rmi_real).mean()

    return rmi_real, p_perm, rmi_null

def ripple_modulation_index_events(aligned_zscores,
                                   fs=10_000,
                                   ripple_dur=0.06,
                                   thresh_factor=3.0,
                                   distance=80,
                                   n_shuffle=1000,
                                   seed=0):
    rng = np.random.default_rng(seed)
    n_ep, n_samp = aligned_zscores.shape
    mid = n_samp // 2
    half_w = int((ripple_dur/2) * fs)
    in_mask = np.zeros(n_samp, bool)
    in_mask[mid-half_w:mid+half_w+1] = True

    # --- peak counts per epoch --------------------------------------------
    in_counts, out_counts = [], []
    for row in aligned_zscores:
        thr = np.median(row) + thresh_factor * median_abs_deviation(row)
        peaks, _ = find_peaks(row, height=thr, distance=distance)
        in_counts .append(np.sum(in_mask[peaks]))
        out_counts.append(np.sum(~in_mask[peaks]))

    in_rate  = np.mean(in_counts)  / (ripple_dur)               # peaks/s
    out_rate = np.mean(out_counts) / ((n_samp/fs) - ripple_dur)

    rmi_real = (in_rate - out_rate) / (in_rate + out_rate + 1e-10)

    # -------- permutation --------------------------------------------------
    rmi_null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        roll = rng.integers(n_samp)
        in_c, out_c = [], []
        for row in aligned_zscores:
            row = np.roll(row, roll)
            thr = np.median(row) + thresh_factor * median_abs_deviation(row)
            peaks, _ = find_peaks(row, height=thr, distance=distance)
            in_c .append(np.sum(in_mask[peaks]))
            out_c.append(np.sum(~in_mask[peaks]))
        in_r  = np.mean(in_c)  / ripple_dur
        out_r = np.mean(out_c) / ((n_samp/fs) - ripple_dur)
        rmi_null[i] = (in_r - out_r) / (in_r + out_r + 1e-10)

    p_val = (rmi_null >= rmi_real).mean()
    return rmi_real, p_val, rmi_null

'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_ripple_plot (dpath,LFP_channel,recordingName,savename,theta_cutoff=0.5):
    save_path = os.path.join(dpath,savename)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GEVI') 

    '''separate the theta and non-theta parts.'''
    Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_cutoff,High_thres=8,save=False,plot_theta=True)
    '''RIPPLE DETECTION
    For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
    rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,
                                                ep_start=0,ep_end=20,
                                                Low_thres=1.5,High_thres=10,
                                                plot_segment=False,plot_ripple_ep=False,excludeTheta=True)
    'GEVI has a negative'
    index = LFP_channel.split('_')[-1] 
    if index=='1':
        ripple_triggered_LFP_values=Recording1.ripple_triggered_LFP_values_1
    elif index=='2':
        ripple_triggered_LFP_values=Recording1.ripple_triggered_LFP_values_2
    elif index=='3':
        ripple_triggered_LFP_values=Recording1.ripple_triggered_LFP_values_3
    else:
        ripple_triggered_LFP_values=Recording1.ripple_triggered_LFP_values_4

    ripple_triggered_zscore_values=Recording1.ripple_triggered_zscore_values
    aligned_ripple_band_lfps,aligned_zscores=plot_aligned_ripple_save (save_path,LFP_channel,recordingName,
                                                                       ripple_triggered_LFP_values,
                                                                       ripple_triggered_zscore_values,Fs=10000)
    plot_ripple_zscore(save_path, aligned_ripple_band_lfps, aligned_zscores)
    
    rmi, p, null=ripple_modulation_index_events(aligned_zscores,
                                       fs=10_000,
                                       ripple_dur=0.06,
                                       thresh_factor=3,
                                       distance=20,
                                       n_shuffle=2000,
                                       seed=0)
    
    print(f"RMI = {rmi:.3f},  permutation p = {p:.4f}")
    #trough_index,peak_index =Recording1.plot_ripple_correlation(LFP_channel,save_path)
    # stats = ripple_phase_stats_polar(ripple_triggered_LFP_values,
    #                              ripple_triggered_zscore_values,
    #                              bins=12,
    #                              n_perm=3000,
    #                              distance=20)   # e.g. ≥20 samples apart
    # print(stats)
    
    # ripple_zscore_polar(aligned_ripple_band_lfps,
    #                       aligned_zscores,
    #                       bins=30)
    
 
    return -1

def run_ripple_plot_main():
    'This is to process a single or concatenated rial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
    dpath=r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ASleepNonREM'
    recordingName='Saved1to9Trials'

    savename='RippleSave'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'
    run_ripple_plot (dpath,LFP_channel,recordingName,savename,theta_cutoff=0.1) 

def main():    
    run_ripple_plot_main()
    
if __name__ == "__main__":
    main()

