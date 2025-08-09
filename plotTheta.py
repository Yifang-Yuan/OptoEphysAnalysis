# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:16:51 2024

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm


def plot_theta_heatmap_noColorbar(theta_band_lfps,lfps,zscores,Fs=10000):
    theta_band_lfps_mean,theta_band_lfps_std, theta_band_lfps_CI=OE.calculateStatisticNumpy (theta_band_lfps)
    lfps_mean,lfps_std, lfps_CI=OE.calculateStatisticNumpy (lfps)
    zscores_mean,zscores_std, zscores_CI=OE.calculateStatisticNumpy (zscores)
    
    time = np.linspace((-len(lfps_mean)/2)/Fs, (len(lfps_mean)/2)/Fs, len(lfps_mean))  
    
    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]}, figsize=(8, 16))
    axs[0].plot(time, theta_band_lfps_mean, color='#404040', label='theta Band Mean') #  #404040
    axs[0].fill_between(time, theta_band_lfps_CI[0], theta_band_lfps_CI[1], color='#404040', alpha=0.2, label='0.95 CI')
    axs[1].plot(time, lfps_mean, color='dodgerblue', label='theta LFP Mean')
    axs[1].fill_between(time, lfps_CI[0], lfps_CI[1], color='dodgerblue', alpha=0.2, label='0.95 CI')
    axs[2].plot(time, zscores_mean, color='limegreen', label='theta Zscore Mean')  #limegreen, tomato
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

def plot_theta_heatmap(theta_band_lfps, lfps, zscores, Fs=10000):
    theta_band_lfps_mean, theta_band_lfps_std, theta_band_lfps_CI = OE.calculateStatisticNumpy(theta_band_lfps)
    lfps_mean, lfps_std, lfps_CI = OE.calculateStatisticNumpy(lfps)
    zscores_mean, zscores_std, zscores_CI = OE.calculateStatisticNumpy(zscores)

    time = np.linspace((-len(lfps_mean)/2)/Fs, (len(lfps_mean)/2)/Fs, len(lfps_mean))  

    fig, axs = plt.subplots(5, 1, gridspec_kw={'height_ratios': [1, 1, 1, 2, 2]}, figsize=(8, 16))
    
    # Plotting theta-band LFP
    axs[0].plot(time, theta_band_lfps_mean, color='#404040')
    axs[0].fill_between(time, theta_band_lfps_CI[0], theta_band_lfps_CI[1], color='#404040', alpha=0.2)
    #axs[0].set_title('Averaged Theta Epoch', fontsize=18)
    axs[0].set_ylabel('LFP (μV)', fontsize=16)
    
    # Plotting LFP mean
    axs[1].plot(time, lfps_mean, color='dodgerblue')
    axs[1].fill_between(time, lfps_CI[0], lfps_CI[1], color='dodgerblue', alpha=0.2)
    axs[1].set_ylabel('LFP (μV)', fontsize=16)
    # Plotting z-score mean
    axs[2].plot(time, zscores_mean, color='tomato')
    axs[2].fill_between(time, zscores_CI[0], zscores_CI[1], color='tomato', alpha=0.2)
    axs[2].set_ylabel('ΔF/F', fontsize=16)
    for i in range(3):
        axs[i].set_xlim(time[0], time[-1])
        axs[i].margins(x=0)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)  # Hide left spine
        axs[i].yaxis.tick_right()  # Show ticks on the right
        axs[i].yaxis.set_label_position("right")  # Move label to the right
        axs[i].tick_params(axis='y', labelsize=16)
    
    axs[0].tick_params(labelbottom=False, bottom=False)
    axs[1].tick_params(labelbottom=False, bottom=False)
    
    # Add LFP heatmap
    sns.heatmap(lfps, cmap="viridis", ax=axs[3], cbar=False)
    axs[3].set_ylabel('Epoch Number', fontsize=16)
    axs[3].tick_params(axis='both', labelsize=12)
    axs[3].tick_params(labelbottom=False, bottom=False)
    
    # Add Z-score heatmap
    sns.heatmap(zscores, cmap="viridis", ax=axs[4], cbar=False)
    axs[4].set_ylabel('Epoch Number', fontsize=16)
    axs[4].tick_params(axis='both', labelsize=12)
    axs[4].tick_params(labelbottom=False, bottom=False)
    
    # Add colorbars to right of heatmaps using inset_axes
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
    
    # Z-score heatmap colourbar
    cbar_ax2 = inset_axes(axs[4],
                          width="2%", height="100%",
                          bbox_to_anchor=(1.05, 0., 1, 1),
                          bbox_transform=axs[4].transAxes,
                          loc='upper left', borderpad=0)
    norm2 = plt.Normalize(np.min(zscores), np.max(zscores))
    sm2 = cm.ScalarMappable(cmap="viridis", norm=norm2)
    cbar2 = plt.colorbar(sm2, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('ΔF/F', fontsize=16)
    cbar2.ax.tick_params(labelsize=16)
    
    return fig

def plot_aligned_theta_phase (save_path,LFP_channel,recordingName,theta_triggered_lfps,theta_triggered_zscores,Fs=10000):
    os.makedirs(save_path, exist_ok=True)
    'Assume my theta PETH are all process by OEC theta detection, Fs=10000, length=4000'
    theta_sample_numbers=len(theta_triggered_lfps[0])
    midpoint=theta_sample_numbers//2
    'align theta in a 200ms window '
    start_idx=int(midpoint-0.25*Fs) #
    end_idx=int(midpoint+0.25*Fs)  #0.25 or 0.15
    print (midpoint,start_idx,end_idx)
    
    '''Align by phase'''
    theta_band_lfps_by_phase=np.zeros_like(theta_triggered_lfps)
    for i in range(theta_triggered_lfps.shape[0]):
        LFP_theta_band_i=OE.band_pass_filter(theta_triggered_lfps[i], 5, 12, Fs)
        theta_band_lfps_by_phase[i]=LFP_theta_band_i
    
    save_file_path = os.path.join(save_path,'ailgned_theta_LFP.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(theta_triggered_lfps, file)
    save_file_path = os.path.join(save_path,'ailgned_theta_bandpass_LFP.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(theta_band_lfps_by_phase, file)
    save_file_path = os.path.join(save_path,'ailgned_theta_Zscore.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(theta_triggered_zscores, file)
        
    'PLOT BEFORE ALIGN, originally align by phase trough'
    fig=plot_theta_heatmap(theta_band_lfps_by_phase,theta_triggered_lfps,theta_triggered_zscores,Fs)
    fig_path = os.path.join(save_path, recordingName+LFP_channel+'Theta_aligned_heatmap_1s.png')
    fig.savefig(fig_path, transparent=True)
    
    fig=plot_theta_heatmap(theta_band_lfps_by_phase[:,start_idx:end_idx],
                            theta_triggered_lfps[:,start_idx:end_idx],
                            theta_triggered_zscores[:,start_idx:end_idx],Fs)
    fig_path = os.path.join(save_path, recordingName+LFP_channel+'Theta_aligned_heatmap_300ms.png')
    fig.savefig(fig_path, transparent=True)
    
    cross_corr_values = []
    for i in range(len(theta_triggered_lfps)):
        # segment_z_score=aligned_zscores[i,start_idx:end_idx]
        # segment_LFP=aligned_lfps[i,start_idx:end_idx]
        segment_z_score=theta_triggered_zscores[i,int(midpoint-0.4*Fs):int(midpoint+0.4*Fs)]
        segment_LFP=theta_triggered_lfps[i,int(midpoint-0.4*Fs):int(midpoint+0.4*Fs)]
        lags,cross_corr =OE.calculate_correlation_with_detrend (segment_z_score,segment_LFP)
        cross_corr_values.append(cross_corr)
    cross_corr_values = np.array(cross_corr_values,dtype=float)

    event_corr_array=cross_corr_values
    mean_cross_corr,std_cross_corr, CI_cross_corr=OE.calculateStatisticNumpy (event_corr_array)
    
    # Find index of peak (maximum) mean correlation
    # peak_idx = np.argmax(mean_cross_corr)
    # peak_idx = np.argmin(mean_cross_corr)
    peak_idx = np.argmax(np.abs(mean_cross_corr))
    # Get values at that index
    peak_value = mean_cross_corr[peak_idx]
    
    # If CI_cross_corr is a tuple/list of arrays: (lower_bound, upper_bound)
    ci_lower = CI_cross_corr[0][peak_idx]
    ci_upper = CI_cross_corr[1][peak_idx]
    
    print(f"Peak correlation = {peak_value:.3f}")
    print(f"95% CI at peak = [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    x = np.linspace((-len(mean_cross_corr)/2)/Fs, (len(mean_cross_corr)/2)/Fs, len(mean_cross_corr))  
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # High resolution for publication quality
    
    # Plot the mean cross-correlation
    ax.plot(x, mean_cross_corr, color='#404040', linewidth=2, label='Mean Cross Correlation')
    
    # Fill between for the confidence interval
    ax.fill_between(x, CI_cross_corr[0], CI_cross_corr[1], color='#404040', alpha=0.2, label='0.95 CI')
    
    # Set labels with larger font sizes
    ax.set_xlabel('Lags (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cross-Correlation', fontsize=14, fontweight='bold')
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Remove spines for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save the figure with a transparent background
    fig_path = os.path.join(save_path, recordingName + LFP_channel + 'CrossCorrelation.png')
    fig.savefig(fig_path, transparent=True, bbox_inches='tight')
    
    plt.show()

    return -1

# Nature-style figure settings

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
    theta_sample_numbers = len(lfps[0])
    midpoint = theta_sample_numbers // 2
    start_idx = int(midpoint - 0.065 * Fs)  # 0.065 s before midpoint
    end_idx = int(midpoint + 0.065 * Fs)    # 0.065 s after midpoint

    # Filter LFPs into theta band
    theta_band_lfps_by_phase = np.array([OE.band_pass_filter(lfp, 5, 12, Fs) for lfp in lfps])

    # Crop signals
    lfps = lfps[:, start_idx:end_idx]
    zscores = zscores[:, start_idx:end_idx]
    theta_band_lfps_by_phase = theta_band_lfps_by_phase[:, start_idx:end_idx]

    # Compute average and confidence intervals
    theta_band_lfps_mean, theta_band_lfps_std, theta_band_lfps_CI = OE.calculateStatisticNumpy(theta_band_lfps_by_phase)
    time = np.linspace(-0.065, 0.065, len(theta_band_lfps_mean))

    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 2, 1]})

    # --- Averaged Theta Band LFP ---
    axs[0].plot(time, theta_band_lfps_mean, color='#4a4a4a', linewidth=2.5, label="LFP Theta Band")
    axs[0].fill_between(time, theta_band_lfps_CI[0], theta_band_lfps_CI[1], color='#4a4a4a', alpha=0.3, label="95% CI")
    axs[0].set_ylabel("Amplitude", fontsize=16)
    #axs[0].set_title("Averaged LFP Theta Band", fontsize=18, fontweight="bold")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].legend().set_visible(False)
    
    # --- Raster Plot of Firing Events ---
    zscore_fire_times = np.argmax(zscores, axis=1) / Fs - 0.065  # Convert to seconds relative to midpoint
    epoch_numbers = np.arange(len(zscore_fire_times))
    
    axs[1].scatter(zscore_fire_times, epoch_numbers, color='#c40030', marker='|', s=100)
    axs[1].set_xlim(time[0], time[-1])
    axs[1].set_ylabel("Epoch", fontsize=16)
    axs[1].set_title("Event Timings (Raster)", fontsize=18, fontweight="bold")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    # --- Histogram of Theta Phase ---
    cycle_duration = 0.13  # 130 ms = 0.065 s * 2
    theta_phase = (zscore_fire_times / (cycle_duration / 2)) * 180  # Convert to degrees

    sns.histplot(theta_phase, bins=80, kde=True, color='#1f78b4', edgecolor="black", alpha=0.8, ax=axs[2])
    axs[2].set_xlabel("Theta Phase (°)", fontsize=16)
    axs[2].set_ylabel("Count", fontsize=16)
    axs[2].set_title("Event numbers by Theta Phase", fontsize=18, fontweight="bold")
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(save_path, 'Rasterplot.png')
    fig.savefig(fig_path, transparent=True, bbox_inches='tight')
    plt.show()


'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_theta_plot_all_cycle (dpath,LFP_channel,recordingName,savename,theta_low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    os.makedirs(save_path, exist_ok=True)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GEVI') 
    
    Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_low_thres,High_thres=10,save=False,plot_theta=True)
    trough_index,peak_index =Recording1.plot_theta_correlation(LFP_channel,save_path)
    theta_part=Recording1.theta_part
    #theta_part=Recording1.Ephys_tracking_spad_aligned
    theta_zscores_np,theta_lfps_np=OE.get_theta_cycle_value(theta_part, LFP_channel, trough_index, half_window=0.5, fs=Recording1.fs)
    plot_aligned_theta_phase (save_path,LFP_channel,recordingName,theta_lfps_np,theta_zscores_np,Fs=10000)
    #plot_raster_histogram_theta_phase (save_path,theta_lfps_np,theta_zscores_np,Fs=10000)
    
    return -1

def run_theta_plot_main():
    'This is to process a single or concatenated trial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
   
    dpath=r'C:\SPAD\Data\OEC\1765508_Jedi2p_Atlas\Day3'
    recordingName='SyncRecording13'


    savename='ThetaSave_Move'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'
    run_theta_plot_all_cycle (dpath,LFP_channel,recordingName,savename,theta_low_thres=-0.7) #-0.3

def main():    
    run_theta_plot_main()
    
if __name__ == "__main__":
    main()