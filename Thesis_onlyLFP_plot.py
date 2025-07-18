# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 11:51:34 2025

@author: yifan
"""

import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
import seaborn as sns

def plot_segment_lfp_feature(recording, LFP_channel, start_time, end_time, lfp_cutoff):
    fs=recording.fs

    silced_recording = recording.slicing_pd_data(recording.Ephys_tracking_spad_aligned, start_time=start_time, end_time=end_time)

    # 2Hz high pass
    lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='low', cutoff=lfp_cutoff, fs=fs, order=5)
    # lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=2, fs=fs, order=3)
    lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)

    fig, ax = plt.subplots(3, 1, figsize=(12, 5))

    OE.plot_trace_in_seconds_ax(ax[0], lfp_low, fs, label='LFP lowpass', color=sns.color_palette("dark", 8)[7], ylabel='mV', xlabel=False)
    _, frequency, power, _ = OE.Calculate_wavelet(lfp_low, lowpassCutoff=500, Fs=fs, scale=80)
    OE.plot_wavelet(ax[1], lfp_low, frequency, power, Fs=fs, colorBar=True, logbase=False)
    ax[1].set_ylim(0, 20)

    # theta: 4-10 Hz
    lfp_theta = OE.butter_filter(silced_recording[LFP_channel], btype='high', cutoff=4, fs=fs, order=5)
    lfp_theta = OE.butter_filter(lfp_theta, btype='low', cutoff=10, fs=fs, order=5)
    lfp_theta = pd.Series(lfp_theta, index=silced_recording[LFP_channel].index)
    OE.plot_trace_in_seconds_ax(ax[2], lfp_theta, fs, label='LFP θ', color=sns.color_palette("dark", 8)[5], ylabel='μV', xlabel=True)

    for axis in ax:
        axis.legend().set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[1].set_xlabel('')
    ax[2].set_xlabel('Time (seconds)')

    makefigure_path = os.path.join(recording.savepath, 'makefigure')
    os.makedirs(makefigure_path, exist_ok=True)
    output_path = os.path.join(makefigure_path, 'example_lfp_trace.png')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

    return -1

def plot_segment_lfp_featur_ripple(recording, LFP_channel, start_time, end_time, lfp_cutoff):
    fs=recording.fs

    silced_recording = recording.slicing_pd_data(recording.Ephys_tracking_spad_aligned, start_time=start_time, end_time=end_time)

    # 2Hz high pass
    lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='low', cutoff=lfp_cutoff, fs=fs, order=5)
    lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=80, fs=fs, order=3)
    lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)

    fig, ax = plt.subplots(3, 1, figsize=(12, 5))

    OE.plot_trace_in_seconds_ax(ax[0], lfp_low, fs, label='LFP lowpass', color=sns.color_palette("dark", 8)[7], ylabel='mV', xlabel=False)
    _, frequency, power, _ = OE.Calculate_wavelet(lfp_low, lowpassCutoff=500, Fs=fs, scale=40)
    OE.plot_wavelet(ax[1], lfp_low, frequency, power, Fs=fs, colorBar=True, logbase=False)
    ax[1].set_ylim(80, 250)

    # theta: 4-10 Hz
    lfp_theta = OE.butter_filter(silced_recording[LFP_channel], btype='high', cutoff=4, fs=fs, order=5)
    lfp_theta = OE.butter_filter(lfp_theta, btype='low', cutoff=10, fs=fs, order=5)
    lfp_theta = pd.Series(lfp_theta, index=silced_recording[LFP_channel].index)
    OE.plot_trace_in_seconds_ax(ax[2], lfp_theta, fs, label='LFP θ', color=sns.color_palette("dark", 8)[5], ylabel='μV', xlabel=True)

    for axis in ax:
        axis.legend().set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[1].set_xlabel('')
    ax[2].set_xlabel('Time (seconds)')

    makefigure_path = os.path.join(recording.savepath, 'makefigure')
    os.makedirs(makefigure_path, exist_ok=True)
    output_path = os.path.join(makefigure_path, 'example_lfp_trace.png')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

    return -1

'''recordingMode: use py, Atlas, SPAD for different systems
'''
# dpath='F:/2025_ATLAS_SPAD/1887930_PV_mNeon_mCherry/Day4/'
# recordingName='SyncRecording3'
# Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
#                                      read_aligned_data_from_file=True,
#                                      recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_4'

dpath= 'F:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day4/'
#dpath='E:/ATLAS_SPAD/1825507_mCherry/Day1/'
recordingName='SyncRecording8'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                     read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=3 #the duration of the segment, in seconds
viewNum=10 #the number of segments
for i in range(viewNum):    #Recording1.plot_segment_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=100,lfp_cutoff=500)
    'This is to plot two optical traces from two ROIs, i.e. one signal and one reference'
    plot_segment_lfp_feature (Recording1,LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),lfp_cutoff=100)
#%%
'''To plot the feature of a part of the signal'''
start_time=7.5
end_time=9.5

plot_segment_lfp_feature (Recording1,LFP_channel,start_time,end_time,lfp_cutoff=500)
plot_segment_lfp_featur_ripple(Recording1,LFP_channel,start_time,end_time,lfp_cutoff=1500)
#%%

dpath= 'F:/2025_ATLAS_SPAD/1887930_PV_mNeon_mCherry/Day4_Sleep/'
recordingName='SyncRecording1'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_4'

Fs=10000
# theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.3,High_thres=8,save=False,plot_theta=True)
# LFP_theta=Recording1.theta_part[LFP_channel]

LFP_theta=Recording1.Ephys_tracking_spad_aligned[LFP_channel]


fig, ax1 = plt.subplots(1, 1, figsize=(3, 5))

OpticalAnlaysis.PSD_plot(LFP_theta, Fs, method="welch", color='black', xlim=[70, 200], linewidth=2, linestyle='-', label='LFP  ', ax=ax1)
#OpticalAnlaysis.PSD_plot(ref_theta, Fs, method="welch", color='red', xlim=[0.1, 49],linewidth=2, linestyle='-', label='Ref', ax=ax1)
ax1.set_ylabel('LFP PSD [dB/Hz]', color='black')
ax1.tick_params(axis='y', labelcolor='black')
#ax1.set_ylim(2,20)
plt.title('LFP PSD')

legend1 = ax1.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
