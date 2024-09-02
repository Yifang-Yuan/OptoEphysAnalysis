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
#%%
'''recordingMode: use py, Atlas, SPAD for different systems
'''
#dpath='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/'
dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day1/'
#dpath='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240501_Day3/'
recordingName='SavedPostSleepTrials'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
#%%
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_3'
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0.5,High_thres=8,save=False,plot_theta=True)
#%%
'''RIPPLE DETECTION
For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,ep_start=10,ep_end=40,
                                                                          Low_thres=1,High_thres=10,plot_segment=False,
                                                                          plot_ripple_ep=False,excludeTheta=True)
#%%
save_path = os.path.join(dpath, recordingName,LFP_channel+'_Class.pkl')
with open(save_path, "wb") as file:
    # Serialize and write the instance to the file
    pickle.dump(Recording1, file)

#%%
'GEVI has a negative'
ripple_triggered_zscore_values=Recording1.ripple_triggered_zscore_values
ripple_triggered_LFP_values=Recording1.ripple_triggered_LFP_values_3
def plot_compare_align (ripple_triggered_LFP_values):
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    for i in range(len(ripple_triggered_LFP_values)):
        ripple_LFP_band_i=OE.band_pass_filter(ripple_triggered_LFP_values[i], 130, 250, 10000)
        ax[0].plot(ripple_LFP_band_i[1000:3000])
        local_max_idx = np.argmax(ripple_LFP_band_i[1000:3000]) + 1000
        shift = 2000 - local_max_idx
        # Roll the trace to align the max value to the center
        aligned_trace = np.roll(ripple_LFP_band_i, shift)
        ax[1].plot(aligned_trace[1000:3000])
    return -1
#%%
start_idx = 1000
end_idx = 3000
midpoint = 2000  # The middle of the 4000 sample trace
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
    
aligned_ripple_band_lfps,aligned_lfps,aligned_zscores=align_ripples (ripple_triggered_LFP_values,ripple_triggered_zscore_values,start_idx,end_idx,midpoint,Fs=10000)
average_ripple_band = np.mean(aligned_ripple_band_lfps, axis=0)
average_lfp = np.mean(aligned_lfps, axis=0)
average_zscore= np.mean(aligned_zscores, axis=0)
#%%
# Create a figure and subplots with shared x-axis
Fs=10000
time = np.arange(0, len(average_ripple_band)) / Fs * 1000  # Time array in milliseconds

def plot_ripple_heatmap(trace,heatmap_values):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot the average trace in the first subplot
    axs[0].plot(trace)
    axs[0].set_title('Average Trace')
    axs[0].set_ylabel('Average Value')
    
    heatmap = sns.heatmap(heatmap_values, cmap='viridis', ax=axs[1], cbar=False)

    # Set the title and labels for the heatmap
    axs[1].set_title('Heatmap of the Array')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Row')
    plt.show()
    
plot_ripple_heatmap(average_ripple_band,aligned_lfps)
plot_ripple_heatmap(average_ripple_band,aligned_zscores)
plot_ripple_heatmap(average_ripple_band[1000:3000],aligned_lfps[:, 1000:3000])
plot_ripple_heatmap(average_ripple_band[1000:3000],aligned_zscores[:, 1000:3000])
#%%
plot_ripple_heatmap(average_lfp,aligned_lfps)
plot_ripple_heatmap(average_lfp,aligned_zscores)
plot_ripple_heatmap(average_lfp[1000:3000],aligned_lfps[:, 1000:3000])
plot_ripple_heatmap(average_lfp[1000:3000],aligned_zscores[:, 1000:3000])

plot_ripple_heatmap(average_zscore,aligned_lfps)
plot_ripple_heatmap(average_zscore,aligned_zscores)
plot_ripple_heatmap(average_zscore[1000:3000],aligned_lfps[:, 1000:3000])
plot_ripple_heatmap(average_zscore[1000:3000],aligned_zscores[:, 1000:3000])
#%%
#%%
'''RIPPLE CURATION'''
def Ripple_manual_select ():
    dpath='D:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/'
    recordingName='SavedPostSleepTrials'
    # dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day3/'
    # recordingName='SavedPostSleepTrials'
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'

    '''separate the theta and non-theta parts.
    theta_thres: the theta band power should be bigger than 80% to be defined theta period.
    nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
    theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0.5,High_thres=8,save=False,plot_theta=True)
    'Manually input and select ripple'
    Recording1.ManualSelectRipple (lfp_channel=LFP_channel,ep_start=10,ep_end=40,
                                                                              Low_thres=1,High_thres=10,plot_segment=False,
                                                                              plot_ripple_ep=True,excludeTheta=True)
    Recording1.Oscillation_triggered_Optical_transient_raw (mode='ripple',lfp_channel=LFP_channel, half_window=0.2,plot_single_trace=False,plotShade='CI')
    'save Class as pickle'
    save_path = os.path.join(dpath, recordingName,LFP_channel+'_Class.pkl')
    with open(save_path, "wb") as file:
        # Serialize and write the instance to the file
        pickle.dump(Recording1, file)
    return Recording1