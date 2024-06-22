# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:39:52 2023

@author: Yang
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import signal
from open_ephys.analysis import Session

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5):
#def butter_filter(data, btype='high', cutoff=3, fs=130, order=5): # for photometry data  
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    # fig, ax = plt.subplots(figsize=(15, 3))
    # ax=plot_trace(y,ax, label="trace_10Hz_low_pass")
    return y

def plotSingleTrace (ax, signal, SamplingRate=30000,color='tab:blue'):
    ax.plot(signal,color,linewidth=1,alpha=0.8)
    ax.set_xticklabels(map(float, ax.get_xticks()/SamplingRate),fontsize=10)
    ax.set_yticklabels(ax.get_yticks(),fontsize=10)
    ax.set_xlabel("seconds",fontsize=10)
    return ax 
#%%
directory = "G:/YY/New/Jedi_test/2024-05-24_12-56-22/"
#directory = "G:/SPAD/SPADData/20230409_OEC_Ephys/2023-04-05_17-02-58_9820-noisy" #Indeed noisy
#directory = "G:/SPAD/SPADData/20230409_OEC_Ephys/2023-04-05_15-25-32_9819"
session = Session(directory)
recording= session.recordnodes[0].recordings[0]
continuous=recording.continuous
continuous0=continuous[0]
samples=continuous0.samples
events=recording.events

#%%
for i in range(24):
	fig, ax = plt.subplots(figsize=(12, 2.5))
	ax.plot(samples[:,i],linewidth=1,alpha=0.8)
#%%
'''Recording nodes that are effective'''
LFP1=samples[:,8]
LFP2=samples[:,9]
LFP3=samples[:,10]
LFP4=samples[:,11]
LFP5=samples[:,13]
LFP_NA=samples[:,1]
'''ADC lines that recorded the analog input from SPAD PCB X10 pin'''
Sync1=samples[:,16] #Full pulsed aligned with X10 input
Sync2=samples[:,17]
Sync3=samples[:,18]
Sync4=samples[:,19]
#%% 
'''begin:232585; end:3251250'''
fig, ax = plt.subplots(figsize=(12, 2.5))
ax.plot(LFP2[1528000:1535000],linewidth=1,alpha=0.8)
ax.set_xlabel("Open Ephys sample numbers",fontsize=10)
#%%
LFP_SPADsync=LFP4[232585:3251250]
LFP=LFP_SPADsync[210000:300000]

LFP_high=butter_filter(LFP, btype='high', cutoff=0.5,fs=30000, order=1)
LFP_lowpass=butter_filter(LFP_high, btype='low', cutoff=300, fs=30000, order=1)
#%%
fig, ax = plt.subplots(figsize=(12, 2.5))
plotSingleTrace (ax, LFP_lowpass, SamplingRate=30000,color='tab:blue')

#%%
recording_highFreq= session.recordnodes[1].recordings[0]
spikes=recording_highFreq.spikes[3] # Different index indicate different channels
waveform=spikes.waveforms
'''Spike waveforms'''
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(68):
	ax.plot(waveform[i,0,:],linewidth=1,alpha=0.8)
#%%
