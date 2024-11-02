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

def align_ripples (lfps,zscores,start_idx,end_idx,midpoint,Fs=10000):
    aligned_ripple_band_lfps = np.zeros_like(lfps)
    aligned_lfps=np.zeros_like(lfps)
    aligned_zscores=np.zeros_like(lfps)
    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 18))
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 18))
    for i in range(lfps.shape[0]):
        lfp_i=lfps[i]
        zscore_i=zscores[i]
        LFP_ripple_band_i=OE.band_pass_filter(lfps[i], 5, 9, Fs)
        # Find the index of the maximum value in the segment [1000:3000]
        local_max_idx = np.argmax(LFP_ripple_band_i[start_idx:end_idx]) + start_idx
        # Calculate the shift needed to align the max value to the midpoint
        shift = midpoint - local_max_idx  
        # # Find the index of the maximum value in the segment [1000:3000]
        # local_min_idx = np.argmin(LFP_ripple_band_i[start_idx:end_idx]) + start_idx
        # # Calculate the shift needed to align the max value to the midpoint
        # shift = midpoint - local_min_idx  
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
    axs[0].set_title('Averaged Theta Epoch')
    for i in range(3):
        axs[i].set_xlim(time[0], time[-1])
        axs[i].margins(x=0)  # Remove any additional margins on x-axis
        axs[i].legend()
        # Remove the frame (spines) from the first three plots
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].get_yaxis().set_visible(False)  # Opt
    axs[0].tick_params(labelbottom=False, bottom=False)  # Remove x-ticks and labels for axs[0]
    axs[1].tick_params(labelbottom=False, bottom=False)  # Remove x-ticks and labels for axs[1]
              
    sns.heatmap(lfps, cmap="viridis", ax=axs[3], cbar=False)
    axs[3].set_title('Heatmap of LFPs')
    axs[3].set_ylabel('Epoch Number')
    
    sns.heatmap(zscores, cmap="viridis", ax=axs[4], cbar=False)
    axs[4].set_title('Heatmap of Zscores')
    axs[4].set_ylabel('Epoch Number')
    axs[3].tick_params(labelbottom=False, bottom=False)
    plt.tight_layout()
    plt.show()
    return fig
    
def plot_aligned_ripple_save (save_path,ripple_triggered_lfps,ripple_triggered_zscores,Fs=10000):
    os.makedirs(save_path, exist_ok=True)
    'Assume my ripple PETH are all process by OEC ripple detection, Fs=10000, length=4000'
    ripple_sample_numbers=len(ripple_triggered_lfps[0])
    midpoint=ripple_sample_numbers//2
    'align ripple in a 200ms window '
    start_idx=int(midpoint-0.25*Fs)
    end_idx=int(midpoint+0.25*Fs)
    print (midpoint,start_idx,end_idx)
    aligned_ripple_band_lfps,aligned_lfps,aligned_zscores=align_ripples (ripple_triggered_lfps,
                                                                         ripple_triggered_zscores,start_idx,end_idx,midpoint,Fs)
    fig=plot_ripple_heatmap(aligned_ripple_band_lfps,aligned_lfps,aligned_zscores,Fs)
    fig_path = os.path.join(save_path, 'Ripple_aligned_heatmap_400ms.png')
    fig.savefig(fig_path, transparent=True)
    
    fig=plot_ripple_heatmap(aligned_ripple_band_lfps[:,start_idx:end_idx],
                            aligned_lfps[:,start_idx:end_idx],aligned_zscores[:,start_idx:end_idx],Fs)
    fig_path = os.path.join(save_path, 'Ripple_aligned_heatmap_200ms.png')
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
    return -1

'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_theta_plot (dpath,LFP_channel,recordingName,savename,Low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='py',indicator='GECI') 

    '''RIPPLE DETECTION
    For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
    rip_ep,rip_tsd=Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,
                                                ep_start=0,ep_end=10,
                                                Low_thres=Low_thres,High_thres=10,
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
    plot_aligned_ripple_save (save_path,theta_triggered_LFP_values,theta_triggered_zscore_values,Fs=10000)
    return -1

def run_theta_plot_main():
    'This is to process a single or concatenated trial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
    dpath='E:/ATLAS_SPAD/1820061_PVcre/Day4/'
    recordingName='SavedMovingGoodTrials'
    savename='ThetaSave_Move'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'
    run_theta_plot (dpath,LFP_channel,recordingName,savename,Low_thres=0.7)
    
run_theta_plot_main()
