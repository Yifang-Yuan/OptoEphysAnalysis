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

def align_theta_by_trough (lfps,zscores,start_idx,end_idx,midpoint,Fs=10000):
    ripple_band_lfps_by_phase=np.zeros_like(lfps)
    aligned_ripple_band_lfps = np.zeros_like(lfps)
    aligned_lfps=np.zeros_like(lfps)
    aligned_zscores=np.zeros_like(lfps)
    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 18))
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 18))
    for i in range(lfps.shape[0]):
        lfp_i=lfps[i]
        zscore_i=zscores[i]
        LFP_ripple_band_i=OE.band_pass_filter(lfps[i], 5, 12, Fs)
        # # Find the index of the maximum value in the segment [1000:3000]
        # local_max_idx = np.argmax(LFP_ripple_band_i[start_idx:end_idx]) + start_idx
        # # Calculate the shift needed to align the max value to the midpoint
        # shift = midpoint - local_max_idx  
        # Find the index of the maximum value in the segment [1000:3000]
        local_min_idx = np.argmin(LFP_ripple_band_i[start_idx:end_idx]) + start_idx
        # Calculate the shift needed to align the max value to the midpoint
        shift = midpoint - local_min_idx  
        
        #Roll the trace to align the max value to the center
        aligned_ripple_lfp_i = np.roll(LFP_ripple_band_i, shift)   
        aligned_lfp_i=np.roll(lfp_i, shift)   
        aligned_zscore_i=np.roll(zscore_i, shift)
        #Store the ripple band and aligned trace
        ripple_band_lfps_by_phase[i]=LFP_ripple_band_i
        aligned_lfp_i=lfp_i 
        aligned_zscore_i=zscore_i
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
    
def plot_theta_aligned_by_trough (save_path,LFP_channel,recordingName,ripple_triggered_lfps,ripple_triggered_zscores,Fs=10000):
    
    os.makedirs(save_path, exist_ok=True)
    'Assume my ripple PETH are all process by OEC ripple detection, Fs=10000, length=4000'
    ripple_sample_numbers=len(ripple_triggered_lfps[0])
    midpoint=ripple_sample_numbers//2
    'align ripple in a 200ms window '
    start_idx=int(midpoint-0.15*Fs) #
    end_idx=int(midpoint+0.15*Fs)  #0,25
    print (midpoint,start_idx,end_idx)
    
    '''Align by peak'''
    aligned_ripple_band_lfps,aligned_lfps,aligned_zscores = align_theta_by_trough (ripple_triggered_lfps,
                                                                         ripple_triggered_zscores,start_idx,end_idx,midpoint,Fs)
    
    '''PLOT HEATMAP after aligning by peak'''
    fig=plot_ripple_heatmap(aligned_ripple_band_lfps,aligned_lfps,aligned_zscores,Fs)
    fig_path = os.path.join(save_path, 'Theta_aligned_heatmap_1s.png')
    fig.savefig(fig_path, transparent=True)
    
    fig=plot_ripple_heatmap(aligned_ripple_band_lfps[:,start_idx:end_idx],
                            aligned_lfps[:,start_idx:end_idx],aligned_zscores[:,start_idx:end_idx],Fs)
    fig_path = os.path.join(save_path, 'Theta_aligned_heatmap_500ms.png')
    fig.savefig(fig_path, transparent=True)
    save_file_path = os.path.join(save_path,'ailgned_theta_LFP.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(aligned_lfps, file)
    save_file_path = os.path.join(save_path,'ailgned_theta_bandpass_LFP.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(aligned_ripple_band_lfps, file)
    save_file_path = os.path.join(save_path,'ailgned_theta_Zscore.pkl')
    with open(save_file_path, "wb") as file:
        pickle.dump(aligned_zscores, file)
        
        
    cross_corr_values = []
    for i in range(len(aligned_lfps)):
        # segment_z_score=aligned_zscores[i,start_idx:end_idx]
        # segment_LFP=aligned_lfps[i,start_idx:end_idx]
        segment_z_score=aligned_zscores[i,int(midpoint-0.4*Fs):int(midpoint+0.4*Fs)]
        segment_LFP=aligned_lfps[i,int(midpoint-0.4*Fs):int(midpoint+0.4*Fs)]
        lags,cross_corr =OE.calculate_correlation_with_detrend (segment_z_score,segment_LFP)
        cross_corr_values.append(cross_corr)
    cross_corr_values = np.array(cross_corr_values,dtype=float)

    event_corr_array=cross_corr_values
    mean_cross_corr,std_cross_corr, CI_cross_corr=OE.calculateStatisticNumpy (event_corr_array)
    
    x = np.linspace((-len(mean_cross_corr)/2)/Fs, (len(mean_cross_corr)/2)/Fs, len(mean_cross_corr))  
    fig, ax = plt.subplots(figsize=(5, 3))
    # Plot the mean cross-correlation
    ax.plot(x, mean_cross_corr, color='#404040', label='Mean Cross Correlation')
    # Fill between for the confidence interval
    ax.fill_between(x, CI_cross_corr[0], CI_cross_corr[1], color='#404040', alpha=0.2, label='0.95 CI')
    # Set labels and title
    ax.set_xlabel('Lags (seconds)')
    ax.set_ylabel('Cross-Correlation')
    ax.set_title('Mean Cross-Correlation (1-Second Window)')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xlim([-0.5,0.5])
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    # Optionally remove ticks and labels
    #ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend().set_visible(False)
    #plt.grid()
    fig_path = os.path.join(save_path, 'Theta aligned correlation 200ms.png')
    fig.savefig(fig_path, transparent=True)
    plt.show()
    return -1

def plot_aligned_theta_phase (save_path,LFP_channel,recordingName,ripple_triggered_lfps,ripple_triggered_zscores,Fs=10000):
    os.makedirs(save_path, exist_ok=True)
    'Assume my ripple PETH are all process by OEC ripple detection, Fs=10000, length=4000'
    ripple_sample_numbers=len(ripple_triggered_lfps[0])
    midpoint=ripple_sample_numbers//2
    'align ripple in a 200ms window '
    start_idx=int(midpoint-0.15*Fs) #
    end_idx=int(midpoint+0.15*Fs)  #0,25
    print (midpoint,start_idx,end_idx)
    
    '''Align by phase'''
    ripple_band_lfps_by_phase=np.zeros_like(ripple_triggered_lfps)
    for i in range(ripple_triggered_lfps.shape[0]):
        LFP_ripple_band_i=OE.band_pass_filter(ripple_triggered_lfps[i], 5, 12, Fs)
        ripple_band_lfps_by_phase[i]=LFP_ripple_band_i
    
    'PLOT BEFORE ALIGN, originally align by phase trough'
    fig=plot_ripple_heatmap(ripple_band_lfps_by_phase,ripple_triggered_lfps,ripple_triggered_zscores,Fs)
    fig_path = os.path.join(save_path, recordingName+LFP_channel+'Theta_aligned_heatmap_1s.png')
    fig.savefig(fig_path, transparent=True)
    
    fig=plot_ripple_heatmap(ripple_band_lfps_by_phase[:,start_idx:end_idx],
                            ripple_triggered_lfps[:,start_idx:end_idx],
                            ripple_triggered_zscores[:,start_idx:end_idx],Fs)
    fig_path = os.path.join(save_path, recordingName+LFP_channel+'Theta_aligned_heatmap_300ms.png')
    fig.savefig(fig_path, transparent=True)
    
    cross_corr_values = []
    for i in range(len(ripple_triggered_lfps)):
        # segment_z_score=aligned_zscores[i,start_idx:end_idx]
        # segment_LFP=aligned_lfps[i,start_idx:end_idx]
        segment_z_score=ripple_triggered_zscores[i,int(midpoint-0.4*Fs):int(midpoint+0.4*Fs)]
        segment_LFP=ripple_triggered_lfps[i,int(midpoint-0.4*Fs):int(midpoint+0.4*Fs)]
        lags,cross_corr =OE.calculate_correlation_with_detrend (segment_z_score,segment_LFP)
        cross_corr_values.append(cross_corr)
    cross_corr_values = np.array(cross_corr_values,dtype=float)

    event_corr_array=cross_corr_values
    mean_cross_corr,std_cross_corr, CI_cross_corr=OE.calculateStatisticNumpy (event_corr_array)
    
    x = np.linspace((-len(mean_cross_corr)/2)/Fs, (len(mean_cross_corr)/2)/Fs, len(mean_cross_corr))  
    fig, ax = plt.subplots(figsize=(5, 3))
    # Plot the mean cross-correlation
    ax.plot(x, mean_cross_corr, color='#404040', label='Mean Cross Correlation')
    # Fill between for the confidence interval
    ax.fill_between(x, CI_cross_corr[0], CI_cross_corr[1], color='#404040', alpha=0.2, label='0.95 CI')
    # Set labels and title
    ax.set_xlabel('Lags (seconds)')
    ax.set_ylabel('Cross-Correlation')
    ax.set_title('Mean Cross-Correlation (1-Second Window)')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    
    return -1

def plot_raster_histogram_theta_phase(lfps, zscores, Fs=10000):
    
    # Determine window indices for cropping around the cycle midpoint
    ripple_sample_numbers = len(lfps[0])
    midpoint = ripple_sample_numbers // 2
    start_idx = int(midpoint - 0.065 * Fs)  # 0.07 s before midpoint
    end_idx = int(midpoint + 0.065 * Fs)    # 0.07 s after midpoint
    
    # Filter LFPs into the theta band for phase alignment
    theta_band_lfps_by_phase = np.zeros_like(lfps)
    for i in range(lfps.shape[0]):
        LFP_theta_band_i = OE.band_pass_filter(lfps[i], 5, 12, Fs)
        theta_band_lfps_by_phase[i] = LFP_theta_band_i
    
    # Crop the signals to the window
    lfps = lfps[:, start_idx:end_idx]
    zscores = zscores[:, start_idx:end_idx]
    theta_band_lfps_by_phase = theta_band_lfps_by_phase[:, start_idx:end_idx]
    
    # Calculate average and confidence intervals for the theta band LFPs
    theta_band_lfps_mean, theta_band_lfps_std, theta_band_lfps_CI = OE.calculateStatisticNumpy(theta_band_lfps_by_phase)
    time = np.linspace((-len(theta_band_lfps_mean)/2)/Fs, (len(theta_band_lfps_mean)/2)/Fs, len(theta_band_lfps_mean))
    
    # Create figure with three subplots: 
    #  - Top: averaged theta LFP waveform
    #  - Middle: raster plot of zscore firing times
    #  - Bottom: histogram of firing counts by theta phase
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 2, 1]}, figsize=(8, 12))
    
    # Plot the averaged theta band LFP
    axs[0].plot(time, theta_band_lfps_mean, color='#404040', label='Ripple Band Mean')
    axs[0].fill_between(time, theta_band_lfps_CI[0], theta_band_lfps_CI[1], color='#404040', alpha=0.2, label='0.95 CI')
    axs[0].set_title('Averaged Theta Band LFP', fontsize=18)
    axs[0].set_xlabel('Time (s)', fontsize=16)
    axs[0].set_ylabel('Amplitude', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    
    # Compute the minimum values in each epoch of the zscore array (i.e. the firing times)
    zscore_fire_times = np.argmin(zscores, axis=1) / Fs - (zscores.shape[1] / (2 * Fs))
    epoch_numbers = np.arange(len(zscore_fire_times))
    
    # Raster plot: each epoch's firing time marked as a vertical line
    axs[1].scatter(zscore_fire_times, epoch_numbers, color='red', marker='|', s=100)
    axs[1].set_xlim(time[0], time[-1])
    axs[1].set_ylim(len(epoch_numbers) - 1, 0)  # Reverse the y-axis to have first epoch on top
    axs[1].set_ylabel('Epoch Number', fontsize=20)
    axs[1].set_xlabel('Time (s)', fontsize=16)
    axs[1].set_title('Minimum Zscore Timing Raster Plot', fontsize=18)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    
    # Compute theta phase for each firing event.
    # Cycle duration based on cropped window:
    cycle_duration = (end_idx - start_idx) / Fs  # in seconds
    # Convert firing time (relative to cycle center) to phase:
    # When firing time = 0, phase = 0 (LFP minimum)
    # When firing time = ±(cycle_duration/2), phase = ±180 degrees
    theta_phase = (zscore_fire_times / (cycle_duration / 2)) * 180 + 180   # in degrees
    
    # Histogram of firing counts binned by theta phase
    axs[2].hist(theta_phase, bins=40, color='blue', alpha=0.7)
    axs[2].set_xlabel('Theta Phase (degrees)', fontsize=16)
    axs[2].set_ylabel('Firing Count', fontsize=16)
    axs[2].set_title('Histogram of Zscore Firing by Theta Phase', fontsize=18)
    #axs[2].set_xlim(-200, 200)
    axs[2].tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.show()
    return -1
'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_theta_plot_selectpeak (dpath,LFP_channel,recordingName,savename,theta_low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GECI') 

    '''RIPPLE DETECTION
    For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
    Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,
                                                ep_start=0,ep_end=10,
                                                Low_thres=theta_low_thres,High_thres=8,
                                                plot_segment=False,plot_ripple_ep=False)

    'GEVI has a negative'
    index = LFP_channel.split('_')[-1] 
    if index=='1':
        theta_triggered_LFP_values=Recording1.theta_triggered_LFP_values_1
    elif index=='2':
        theta_triggered_LFP_values=Recording1.theta_triggered_LFP_values_2
    elif index=='3':
        theta_triggered_LFP_values=Recording1.theta_triggered_LFP_values_3
    else:
        theta_triggered_LFP_values=Recording1.theta_triggered_LFP_values_4

    theta_triggered_zscore_values=Recording1.theta_triggered_zscore_values
    
    plot_theta_aligned_by_trough (save_path,LFP_channel,recordingName,theta_triggered_LFP_values,theta_triggered_zscore_values,Fs=10000)
    return -1

'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_theta_plot_all_cycle (dpath,LFP_channel,recordingName,savename,theta_low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GECI') 
    
    Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_low_thres,High_thres=10,save=False,plot_theta=True)

    
    trough_index=Recording1.plot_theta_correlation(LFP_channel)
    theta_part=Recording1.theta_part
    #theta_part=Recording1.Ephys_tracking_spad_aligned
    theta_zscores_np,theta_lfps_np=OE.get_theta_cycle_value(theta_part, LFP_channel, trough_index, half_window=0.5, fs=Recording1.fs)
    plot_aligned_theta_phase (save_path,LFP_channel,recordingName,theta_lfps_np,theta_zscores_np,Fs=10000)
    plot_raster_histogram_theta_phase (theta_lfps_np,theta_zscores_np,Fs=10000)
    return -1

def run_theta_plot_main():
    'This is to process a single or concatenated trial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
    dpath='D:/2025_ATLAS_SPAD/1842515_PV_mNeon/Day7/' #sync3
    #dpath='D:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day2/' #sync1
    recordingName='SyncRecording3'
    savename='ThetaSave_Move'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'
    run_theta_plot_all_cycle (dpath,LFP_channel,recordingName,savename,theta_low_thres=0)
    #run_theta_plot_selectpeak (dpath,LFP_channel,recordingName,savename,theta_low_thres=-0.3)
    
    

def main():    
    run_theta_plot_main()
    
if __name__ == "__main__":
    main()