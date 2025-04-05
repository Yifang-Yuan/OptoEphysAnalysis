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
    
def plot_aligned_ripple_save (save_path,LFP_channel,recordingName,ripple_triggered_lfps,ripple_triggered_zscores,Fs=10000):
    os.makedirs(save_path, exist_ok=True)
    'Assume my ripple PETH are all process by OEC ripple detection, Fs=10000, length=4000'
    ripple_sample_numbers=len(ripple_triggered_lfps[0])
    midpoint=ripple_sample_numbers//2
    'align ripple in a 200ms window '
    start_idx=int(midpoint-0.1*Fs)
    end_idx=int(midpoint+0.1*Fs)
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

def plot_ripple_zscore(savepath, lfp_ripple, zscore):
    """
    Plot ripple band LFP with confidence interval, optical signal troughs as a raster plot, 
    and a histogram of optical trough counts.

    Parameters:
        savepath (str): Path to save the figure.
        lfp_ripple (numpy.ndarray): 2D array (epochs × timepoints) of ripple band LFPs.
        zscore (numpy.ndarray): 2D array (epochs × timepoints) of optical signal z-scores.
    """
    fs = 10000  # Sampling rate in Hz
    time = np.arange(lfp_ripple.shape[1]) / fs  # Convert indices to seconds
    
    # **Find midpoint and crop to -0.1s to 0.1s**
    midpoint = len(time) // 2  # Find center index
    time_window = (-0.05, 0.05)  # Define time range
    idx_range = np.where((time - time[midpoint] >= time_window[0]) & (time - time[midpoint] <= time_window[1]))[0]
    
    # Crop time and data
    time = time[idx_range] - time[midpoint]  # Centered time
    lfp_ripple = lfp_ripple[:, idx_range]
    zscore = zscore[:, idx_range]
    
    # Compute mean and 95% confidence interval for ripple band
    mean_ripple = np.mean(lfp_ripple, axis=0)
    ci_ripple = sem(lfp_ripple, axis=0) * 1.96  # 95% CI using standard error

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


'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_ripple_plot (dpath,LFP_channel,recordingName,savename,theta_cutoff=0.5):
    save_path = os.path.join(dpath,savename)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GEVI') 

    '''separate the theta and non-theta parts.
    theta_thres: the theta band power should be bigger than 80% to be defined theta period.
    nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
    theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_cutoff,High_thres=8,save=False,plot_theta=True)

    '''RIPPLE DETECTION
    For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
    rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,
                                                ep_start=0,ep_end=20,
                                                Low_thres=1.8,High_thres=10,
                                                plot_segment=False,plot_ripple_ep=False,excludeTheta=False)

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
    aligned_ripple_band_lfps,aligned_zscores=plot_aligned_ripple_save (save_path,LFP_channel,recordingName,ripple_triggered_LFP_values,ripple_triggered_zscore_values,Fs=10000)
    plot_ripple_zscore(save_path, aligned_ripple_band_lfps, aligned_zscores)
    return -1

def run_ripple_plot_main():
    'This is to process a single or concatenated rial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
    dpath='F:/2025_ATLAS_SPAD/1842516_PV_Jedi2p/Day6/'
    
    recordingName='Saved4to7Trials'
    savename='RippleSave_Sleep'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_4'
    run_ripple_plot (dpath,LFP_channel,recordingName,savename,theta_cutoff=0.5)

def main():    
    run_ripple_plot_main()
    
if __name__ == "__main__":
    main()
#%%
'concatenate ripple'
# parent_path='F:/2024_OEC_Atlas/1765508_Jedi2p_Atlas/'
# save_path='F:/2024_OEC_Atlas/1765508_Jedi2p_Atlas/RippleConcatenatedSave/'
# pattern = os.path.join(parent_path, 'Day*/','RippleSave_*/')
# Fs=10000
# start_idx=1000
# # Get a list of all matching files
# file_list = glob.glob(pattern)
# # Loop through the file list and read each file
# aligned_ripple_bandpass_LFP_list = []
# aligned_ripple_LFP_list= []
# aligned_ripple_Zscore_list=[]

# for path in file_list:
#     try:
#         ripple_bandpass_LFP = pd.read_pickle(os.path.join(path, 'ailgned_ripple_bandpass_LFP.pkl'))
#         ripple_LFP=pd.read_pickle(os.path.join(path, 'ailgned_ripple_LFP.pkl'))
#         ripple_zscore=pd.read_pickle(os.path.join(path, 'ailgned_ripple_Zscore.pkl'))
#         # Append each DataFrame to the corresponding list
#         aligned_ripple_bandpass_LFP_list.append(ripple_bandpass_LFP)
#         aligned_ripple_LFP_list.append(ripple_LFP)
#         aligned_ripple_Zscore_list.append(ripple_zscore)
#     except Exception as e:
#         print(f"Error reading {path}: {e}")
        
# concatenated_ripple_bandpass_LFP = np.vstack(aligned_ripple_bandpass_LFP_list) 
# concatenated_ripple_LFP = np.vstack(aligned_ripple_LFP_list)
# concatenated_ripple_Zscore = -np.vstack(aligned_ripple_Zscore_list)

# plot_aligned_ripple_save (save_path,concatenated_ripple_LFP,concatenated_ripple_Zscore,Fs=10000)