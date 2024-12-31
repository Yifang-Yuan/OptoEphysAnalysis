# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:30:19 2023
This is the main file to perform a comparison analysis between LFP and optical signal.
This code will create a Class for the recording session you want to analyse.
@author: Yifang
"""
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
#%%
'''recordingMode: use py, Atlas, SPAD for different systems
'''
dpath='E:/2025_ATLAS_SPAD/1842514_Jedi2p/Day1/'
recordingName='SavedOpenFieldTrials'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 
#%%
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_3'
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0.5,High_thres=10,save=False,plot_theta=True)
#%%
#This is to calculate and plot the trace around theta trough
Recording1.plot_theta_correlation(LFP_channel)
#%% Detect theta event
'''THETA PEAK DETECTION
For a rigid threshold to get larger amplitude theta events: Low_thres=1, for more ripple events, Low_thres=0.5'''
data_segment,timestamps=Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,ep_start=10,ep_end=30,
                                                                         Low_thres=0,High_thres=10,plot_segment=True,plot_ripple_ep=True)
#time_duration=transient_trace.index[-1].total_seconds()

 #%% Detect ripple event
'''RIPPLE DETECTION
For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=10,
                                                                          Low_thres=1,High_thres=10,plot_segment=True,
                                                                          plot_ripple_ep=False,excludeTheta=False)
#%% Detect ripple event
'''Gamma band plot
For a rigid threshold to get larger amplitude Gamma events: Low_thres=1, for more ripple events, Low_thres=0'''
rip_ep,rip_tsd=Recording1.pynappleGammaAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=10,
                                                                          Low_thres=1,High_thres=8,plot_segment=True,
                                                                          plot_ripple_ep=True,excludeTheta=False,excludeNonTheta=False)
#%% Detect theta nested gamma event
'''Theta nested Gamma plot
For a rigid threshold to get larger amplitude Gamma events: Low_thres=1, for more ripple events, Low_thres=0'''
rip_ep,rip_tsd=Recording1.PlotThetaNestedGamma (lfp_channel=LFP_channel,Low_thres=0.5,High_thres=10,plot_segment=False, plot_ripple_ep=False)

#%%
'plot feature can be LFP or SPAD to show the power spectrum of LFP or SPAD'
Recording1.plot_gamma_power_on_theta_cycle(LFP_channel=LFP_channel)

#%%
'''To plot the feature of a part of the signal'''
start_time=11
end_time=14
#%%
Recording1.plot_segment_feature (LFP_channel,start_time,end_time,SPAD_cutoff=50,lfp_cutoff=100)
#%%
'To plot the feature of theta-band and ripple-band of the segment signal'
Recording1.plot_band_power_feature (LFP_channel,start_time,end_time,LFP=True)

#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=5 #the duration of the segment, in seconds
viewNum=6 #the number of segments
for i in range(viewNum):
    Recording1.plot_segment_feature (LFP_channel=LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=20,lfp_cutoff=100)
    #Recording1.plot_band_power_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),LFP=True)
#%%
'''sliced_recording:choose a segment or a part of your recording, this can be defined with start and end time,
or just by theta_part, non_theta_part'''
silced_recording=Recording1.slicing_pd_data (Recording1.Ephys_tracking_spad_aligned,start_time=45, end_time=48)
#silced_recording=theta_part
'''Calculate the cross correlation between two power spectrun over time at a specific frequency'''
sst_spad,frequency_spad,power_spad,global_ws_spad=OE.Calculate_wavelet(silced_recording['zscore_raw'],lowpassCutoff=500,Fs=10000)
sst_lfp,frequency_lfp,power_lfp,global_ws_lfp=OE.Calculate_wavelet(silced_recording[LFP_channel],lowpassCutoff=500,Fs=10000)
# Calculate the correlation coefficient
# lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (power_spad[18],power_lfp[18],corr_window=1)
# lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (power_spad[19],power_lfp[19],corr_window=1)
lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (power_spad[20],power_lfp[20],corr_window=1)
#%%
'''Calculate the cross correlation between LFP and optical signal for a specific segment'''
silced_recording=Recording1.slicing_pd_data (Recording1.Ephys_tracking_spad_aligned,start_time=0, end_time=90)
#silced_recording=theta_part
lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (silced_recording['zscore_raw'],silced_recording[LFP_channel],corr_window=0.5)
#%%
'''Calculate the cross correlation between LFP and optical signal for a specific segment, with low-pass filter'''
spad_lowpass= OE.smooth_signal(silced_recording['zscore_raw'],Fs=10000,cutoff=50)
lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='low', cutoff=50, fs=Recording1.fs, order=5)
spad_low = pd.Series(spad_lowpass, index=silced_recording['zscore_raw'].index)
lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)
lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (spad_low,lfp_low,corr_window=0.5)
#%%
Fs=10000

#theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
LFP_theta=Recording1.theta_part[LFP_channel]
LFP_nontheta=Recording1.non_theta_part[LFP_channel]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))

OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)

optical_theta=Recording1.theta_part['zscore_raw']
optical_nontheta=Recording1.non_theta_part['zscore_raw']

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
OpticalAnlaysis.PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='rest',ax=ax)
OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='move',ax=ax)

#%%
LFP_channel='LFP_1'
LFP=Recording1.Ephys_tracking_spad_aligned[LFP_channel]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
OpticalAnlaysis.PSD_plot (LFP,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='LFP',ax=ax)
#OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='move',ax=ax)