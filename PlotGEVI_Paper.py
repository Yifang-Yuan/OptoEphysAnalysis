# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:36:48 2024

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
dpath='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240501_Day3/'
recordingName='SavedOpenFieldTrials'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0,High_thres=8,save=False,plot_theta=True)
#%%
'''To plot the feature of a part of the signal'''
start_time=19
end_time=24
#%%
Recording1.plot_segment_band_feature (LFP_channel,start_time,end_time,SPAD_cutoff=50,lfp_cutoff=100)
#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=5 #the duration of the segment, in seconds
viewNum=11 #the number of segments
for i in range(viewNum):
    Recording1.plot_segment_band_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=50,lfp_cutoff=100)
#%%


#%%
def plot_PSD_for_rest_run ():
    'plot PSD'
    dpath='F:/2024MScR_NORtask//1765507_iGlu_Atlas/20240501_Day3/'
    recordingName='SavedOpenFieldTrials'
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    LFP_channel='LFP_1'
    theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
    
    recordingName='SavedPostAwakeTrials'
    Recording2=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    LFP_channel='LFP_1'
    theta_part,non_theta_part=Recording2.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
    
    recordingName='SavedPostSleepTrials'
    Recording3=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    LFP_channel='LFP_1'
    theta_part,non_theta_part=Recording3.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
    
    LFP_theta=Recording1.theta_part['LFP_1']
    LFP_nontheta=Recording2.non_theta_part['LFP_1']
    LFP_sleep=Recording3.non_theta_part['LFP_1']
    fig, ax = plt.subplots(1, 1, figsize=(3, 6))
    OpticalAnlaysis.PSD_plot (LFP_sleep/1000,fs=10000,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle=':',label='LFP-sleep',ax=ax)
    OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=10000,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
    OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=10000,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)

    optical_theta=Recording1.theta_part['zscore_raw']
    optical_nontheta=Recording2.non_theta_part['zscore_raw']
    optical_sleep=Recording3.non_theta_part['zscore_raw']
    fig, ax = plt.subplots(1, 1, figsize=(3, 6))
    OpticalAnlaysis.PSD_plot (optical_sleep,fs=10000,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle=':',label='iGlu-sleep',ax=ax)
    OpticalAnlaysis.PSD_plot (optical_nontheta,fs=10000,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='iGlu-rest',ax=ax)
    OpticalAnlaysis.PSD_plot (optical_theta,fs=10000,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='iGlu-move',ax=ax)
    return -1


plot_PSD_for_rest_run ()
#%%