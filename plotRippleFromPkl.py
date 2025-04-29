# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 12:02:06 2025

@author: yifan
"""

import os
import numpy as np
import pandas as pd
import pickle
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import plotRipple
import plotTheta
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import sem

    
def plot_ripple_heatmap(ripple_band_lfps,lfps,zscores,Fs=10000):
    ripple_band_lfps_mean,ripple_band_lfps_std, ripple_band_lfps_CI=OE.calculateStatisticNumpy (ripple_band_lfps)
    lfps_mean,lfps_std, lfps_CI=OE.calculateStatisticNumpy (lfps)
    zscores_mean,zscores_std, zscores_CI=OE.calculateStatisticNumpy (zscores)
    
    time = np.linspace((-len(lfps_mean)/2)/Fs, (len(lfps_mean)/2)/Fs, len(lfps_mean))  
    
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]}, figsize=(8, 16))
    axs[0].plot(time, ripple_band_lfps_mean, color='#404040', label='Ripple Band Mean')
    axs[0].fill_between(time, ripple_band_lfps_CI[0], ripple_band_lfps_CI[1], color='#404040', alpha=0.2, label='0.95 CI')
    axs[1].plot(time, lfps_mean, color='dodgerblue', label='Ripple LFP Mean')
    axs[1].fill_between(time, lfps_CI[0], lfps_CI[1], color='dodgerblue', alpha=0.2, label='0.95 CI')
    axs[2].plot(time, zscores_mean, color='limegreen', label='Ripple Zscore Mean')
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

def plot_ripple_heatmap_two(savepath,ripple_bandpass_LFP,ripple_LFP,ripple_Zscore,Fs=10000):
    'Assume my ripple PETH are all process by OEC ripple detection, Fs=10000, length=4000'
    ripple_sample_numbers=len(ripple_LFP[0])
    midpoint=ripple_sample_numbers//2
    'align ripple in a 200ms window '
    start_idx=int(midpoint-0.1*Fs)
    end_idx=int(midpoint+0.1*Fs)
    
    fig=plot_ripple_heatmap(ripple_bandpass_LFP,ripple_LFP,ripple_Zscore,Fs)
    fig_path = os.path.join(savepath,'Ripple_aligned_heatmap_400ms.png')
    fig.savefig(fig_path, transparent=True)
    
    fig=plot_ripple_heatmap(ripple_bandpass_LFP[:,start_idx:end_idx],
                            ripple_LFP[:,start_idx:end_idx],ripple_Zscore[:,start_idx:end_idx],Fs)
    fig_path = os.path.join(savepath, 'Ripple_aligned_heatmap_200ms.png')
    fig.savefig(fig_path, transparent=True)

    return -1

def plot_ripple_zscore(savepath, lfp_ripple, zscore,ripple_bandpass_LFP,fs):
    """
    Plot ripple band LFP with confidence interval, optical signal troughs as a raster plot, 
    and a histogram of optical trough counts.

    Parameters:
        savepath (str): Path to save the figure.
        lfp_ripple (numpy.ndarray): 2D array (epochs × timepoints) of ripple band LFPs.
        zscore (numpy.ndarray): 2D array (epochs × timepoints) of optical signal z-scores.
    """
    time = np.arange(lfp_ripple.shape[1]) / fs  # Convert indices to seconds
    
    # **Find midpoint and crop to -0.1s to 0.1s**
    midpoint = len(time) // 2  # Find center index
    time_window = (-0.05, 0.05)  # Define time range
    idx_range = np.where((time - time[midpoint] >= time_window[0]) & (time - time[midpoint] <= time_window[1]))[0]
    
    # Crop time and data
    time = time[idx_range] - time[midpoint]  # Centered time
    ripple_bandpass_LFP = ripple_bandpass_LFP[:, idx_range]
    zscore = zscore[:, idx_range]
    
    # Compute mean and 95% confidence interval for ripple band
    mean_ripple = np.mean(ripple_bandpass_LFP, axis=0)
    ci_ripple = sem(ripple_bandpass_LFP, axis=0) * 1.96  # 95% CI using standard error

    # Identify troughs in z-score signal for raster plot
    troughs = [np.argmin(epoch) for epoch in zscore]  # Find min per epoch
    trough_times = time[troughs]  # Convert to time

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 2, 1]})

    # Plot ripple band LFPs with confidence interval
    ax0 = axes[0]
    ax0.plot(time, mean_ripple, color='black', label='Ripple Band Mean')
    ax0.fill_between(time, mean_ripple - ci_ripple, mean_ripple + ci_ripple, color='gray', alpha=0.3, label='95% CI')
    ax0.set_ylabel("Amplitude", fontsize=14)
    ax0.tick_params(axis='both', labelsize=12)
    ax0.set_xticklabels([])
    ax0.legend(frameon=False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    
    # Raster plot of z-score troughs
    ax1 = axes[1]
    for i, t in enumerate(trough_times):
        ax1.plot([t, t], [i - 0.4, i + 0.4], color='red', lw=2)  # Small vertical lines as raster marks
    ax1.set_ylabel("Epoch", fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xticklabels([])
    ax1.set_xlim(time[0], time[-1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Histogram of trough times
    ax2 = axes[2]
    ax2.hist(trough_times, bins=60, color='#377eb8',alpha=0.8)  # Black edges added
    ax2.set_xlabel("Time (s)",fontsize=14)
    ax2.set_ylabel("Firing Count", fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', labelsize=12)

    # Save and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'ripple_optical_raster.png'), transparent=True)
    plt.show()

def load_concatenated_ripple_data(dpath):
    """
    Loads the three concatenated theta-related pkl files from the given directory.

    Returns:
        aligned_theta_LFP, aligned_theta_bandpass_LFP, aligned_theta_Zscore: numpy arrays
    """
    files = {
        'ailgned_Ripple_LFP.pkl': None,
        'ailgned_Ripple_bandpass_LFP.pkl': None,
        'ailgned_Ripple_Zscore.pkl': None
    }

    for fname in files:
        full_path = os.path.join(dpath, fname)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                files[fname] = pickle.load(f)
                print(f"Loaded {fname}, shape: {files[fname].shape}")
        else:
            print(f"Error: {fname} not found in {dpath}")

    return files['ailgned_Ripple_LFP.pkl'], files['ailgned_Ripple_bandpass_LFP.pkl'], files['ailgned_Ripple_Zscore.pkl']



# Example usage
dpath = "F:/2025_ATLAS_SPAD/Figure3_Pyr_ripple/RippleData3/"
ripple_LFP, ripple_bandpass_LFP,ripple_Zscore = load_concatenated_ripple_data(dpath)
plot_ripple_heatmap_two(dpath,ripple_bandpass_LFP,ripple_LFP,ripple_Zscore,Fs=10000)
plot_ripple_zscore(dpath,ripple_LFP, ripple_Zscore,ripple_bandpass_LFP, fs=10000)