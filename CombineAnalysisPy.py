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
#%%
'''
recordingMode: use py, Atlas, SPAD for different systems
'''
dpath='D:/YY/2024MScR_NORtask/1756735_PVCre_Jedi2p_Compare/Day1Atlas_Sleep/'
#dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day2/'
#dpath='G:/YY/2024MScR_NORtask/1765508_Jedi2p_CompareSystem/Day3_SPC_1765508/'
#dpath="F:/2024MScR_NORtask/1732333_pyPhotometry/20240214_Day3/" 
recordingName='SavedPostSleepTrials'
#dpath="G:/SPAD/SPADData/20230722_SPADOE/SyncRecording0/"
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
#%%
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_4'
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=1,High_thres=10,save=False,plot_theta=True)
#%% Detect ripple event
'''Gamma plot
For a rigid threshold to get larger amplitude Gamma events: Low_thres=1, for more ripple events, Low_thres=0'''
rip_ep,rip_tsd=Recording1.pynappleGammaAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=10,
                                                                          Low_thres=0,High_thres=8,plot_segment=True,
                                                                          plot_ripple_ep=True,excludeTheta=False,excludeNonTheta=True)
#%% Detect ripple event
'''RIPPLE DETECTION
For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=25,
                                                                          Low_thres=1,High_thres=5,plot_segment=True,
                                                                          plot_ripple_ep=True,excludeTheta=True)
#%% Detect theta nested gamma event
'''Theta nested Gamma plot
For a rigid threshold to get larger amplitude Gamma events: Low_thres=1, for more ripple events, Low_thres=0'''
rip_ep,rip_tsd=Recording1.PlotThetaNestedGamma (lfp_channel=LFP_channel,Low_thres=0.5,High_thres=10,plot_segment=False, plot_ripple_ep=True)
#%%
Recording1.plot_average_theta_nested_gamma(LFP_channel=LFP_channel,plotFeature='SPAD')
#%% This is to calculate and plot the trace around theta trough
Recording1.plot_theta_correlation(LFP_channel)
#%% Detect theta event
'''THETA PEAK DETECTION
For a rigid threshold to get larger amplitude theta events: Low_thres=1, for more ripple events, Low_thres=0.5'''
theta_ep,theta_tsd=Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=30,
                                                                         Low_thres=-0.5,High_thres=8,plot_segment=True,plot_ripple_ep=False)
#time_duration=transient_trace.index[-1].total_seconds()
#%%
'''To plot the feature of a part of the signal'''
start_time=16
end_time=20
#%%
Recording1.plot_segment_feature (LFP_channel,start_time,end_time,SPAD_cutoff=20,lfp_cutoff=100)
#%%
'To plot the feature of theta-band and ripple-band of the segment signal'
Recording1.plot_band_power_feature (LFP_channel,start_time,end_time,LFP=True)

#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=2 #the duration of the segment, in seconds
viewNum=30 #the number of segments
for i in range(viewNum):
    Recording1.plot_segment_feature (LFP_channel=LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=50,lfp_cutoff=500)
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


