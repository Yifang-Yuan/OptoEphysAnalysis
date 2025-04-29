# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:17:05 2025

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
    
def plot_theta_heatmap(theta_band_lfps,lfps,zscores,Fs=10000):
    theta_band_lfps_mean,theta_band_lfps_std, theta_band_lfps_CI=OE.calculateStatisticNumpy (theta_band_lfps)
    lfps_mean,lfps_std, lfps_CI=OE.calculateStatisticNumpy (lfps)
    zscores_mean,zscores_std, zscores_CI=OE.calculateStatisticNumpy (zscores)
    
    time = np.linspace((-len(lfps_mean)/2)/Fs, (len(lfps_mean)/2)/Fs, len(lfps_mean))  
    
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]}, figsize=(8, 16))
    axs[0].plot(time, theta_band_lfps_mean, color='#404040', label='theta Band Mean')
    axs[0].fill_between(time, theta_band_lfps_CI[0], theta_band_lfps_CI[1], color='#404040', alpha=0.2, label='0.95 CI')
    axs[1].plot(time, lfps_mean, color='dodgerblue', label='theta LFP Mean')
    axs[1].fill_between(time, lfps_CI[0], lfps_CI[1], color='dodgerblue', alpha=0.2, label='0.95 CI')
    axs[2].plot(time, zscores_mean, color='limegreen', label='theta Zscore Mean')
    axs[2].fill_between(time, zscores_CI[0], zscores_CI[1], color='limegreen', alpha=0.2, label='0.95 CI')
    axs[0].set_title('Averaged Theta Epoch',fontsize=18)
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

    plt.tight_layout()
    plt.show()
    return fig


def plot_raster_histogram_theta_phase(save_path,lfps, zscores, Fs=10000):
    """
    Generates three plots:
    - Top: Averaged theta band LFP waveform with confidence intervals.
    - Middle: Raster plot of firing events (minimum z-score in each epoch).
    - Bottom: Histogram of firing counts across theta phase.

    Parameters:
    - lfps: 2D array (epochs x time points) of local field potentials.
    - zscores: 2D array (epochs x time points) of z-scored neural activity.
    - Fs: Sampling frequency (Hz).
    """

    # Time window cropping
    half_window=0.065
    theta_sample_numbers = len(lfps[0])
    midpoint = theta_sample_numbers // 2
    start_idx = int(midpoint - half_window * Fs)  # 0.065 s before midpoint
    end_idx = int(midpoint + half_window * Fs)    # 0.065 s after midpoint

    # Filter LFPs into theta band
    theta_band_lfps_by_phase = np.array([OE.band_pass_filter(lfp, 5, 12, Fs) for lfp in lfps])

    # Crop signals
    lfps = lfps[:, start_idx:end_idx]
    zscores = zscores[:, start_idx:end_idx]
    theta_band_lfps_by_phase = theta_band_lfps_by_phase[:, start_idx:end_idx]

    # Compute average and confidence intervals
    theta_band_lfps_mean, theta_band_lfps_std, theta_band_lfps_CI = OE.calculateStatisticNumpy(theta_band_lfps_by_phase)
    time = np.linspace(-half_window, half_window, len(theta_band_lfps_mean))

    # Create figure with 3 subplots
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [1,3]})

    # --- Averaged Theta Band LFP ---
    axs[0].plot(time, theta_band_lfps_mean, color='#4a4a4a', linewidth=2.5, label="LFP Theta Band")
    axs[0].fill_between(time, theta_band_lfps_CI[0], theta_band_lfps_CI[1], color='#4a4a4a', alpha=0.3, label="95% CI")
    axs[0].set_ylabel("Amplitude", fontsize=16)
    #axs[0].set_title("Averaged LFP Theta Band", fontsize=18, fontweight="bold")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].legend().set_visible(False)
    
    # --- Raster Plot of Firing Events ---
    zscore_fire_times = np.argmin(zscores, axis=1) / Fs - half_window  # Convert to seconds relative to midpoint
    # --- Histogram of Theta Phase ---
    cycle_duration = half_window*2  # 130 ms = 0.065 s * 2
    theta_phase = (zscore_fire_times / (cycle_duration / 2)) * 180  # Convert to degrees

    sns.histplot(theta_phase, bins=80, kde=True, color='#1f78b4', edgecolor="black", alpha=0.8, ax=axs[1])
    axs[1].set_xlabel("Theta Phase (°)", fontsize=16)
    axs[1].set_ylabel("Count", fontsize=16)
    axs[1].set_title("Z-Score Troughs by Theta Phase", fontsize=18, fontweight="bold")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(save_path, 'Rasterplot.png')
    fig.savefig(fig_path, transparent=True, bbox_inches='tight')
    plt.show()
'''recordingMode: use py, Atlas, SPAD for different systems'''

def load_concatenated_theta_data(dpath):
    """
    Loads the three concatenated theta-related pkl files from the given directory.

    Returns:
        aligned_theta_LFP, aligned_theta_bandpass_LFP, aligned_theta_Zscore: numpy arrays
    """
    files = {
        'ailgned_theta_LFP.pkl': None,
        'ailgned_theta_bandpass_LFP.pkl': None,
        'ailgned_theta_Zscore.pkl': None
    }

    for fname in files:
        full_path = os.path.join(dpath, fname)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                files[fname] = pickle.load(f)
                print(f"Loaded {fname}, shape: {files[fname].shape}")
        else:
            print(f"Error: {fname} not found in {dpath}")

    return files['ailgned_theta_LFP.pkl'], files['ailgned_theta_bandpass_LFP.pkl'], files['ailgned_theta_Zscore.pkl']



# Example usage
dpath = "F:/2025_ATLAS_SPAD/Figure4_PV_thata/MultiAnimals/ThetaDataAll"
theta_LFP, theta_bandpass_LFP,theta_Zscore = load_concatenated_theta_data(dpath)
#plot_theta_heatmap(theta_bandpass_LFP,theta_LFP,theta_Zscore,Fs=10000)
plot_raster_histogram_theta_phase(dpath,theta_LFP, theta_Zscore, Fs=10000)