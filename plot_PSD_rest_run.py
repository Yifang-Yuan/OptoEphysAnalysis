# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:41:42 2024

@author: Yifang
"""

import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt


def plot_PSD_for_rest_run ():
    'plot PSD'
    
    Fs=10000
    dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day2/'
    recordingName='SavedOpenFieldTrials'
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    LFP_channel='LFP_3'
    theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
    LFP_theta=Recording1.theta_part[LFP_channel]
    
    dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day3/'
    recordingName='SavedPostAwakeTrials'
    Recording2=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    LFP_channel='LFP_3'
    theta_part,non_theta_part=Recording2.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
    LFP_nontheta=Recording2.non_theta_part[LFP_channel]
    
    recordingName='SavedPostSleepTrials'
    Recording3=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
    LFP_channel='LFP_3'
    theta_part,non_theta_part=Recording3.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
    LFP_sleep=Recording3.non_theta_part[LFP_channel]
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 6))
    OpticalAnlaysis.PSD_plot (LFP_sleep/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle=':',label='LFP-sleep',ax=ax)
    OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
    OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)

    optical_theta=Recording1.theta_part['zscore_raw']
    optical_nontheta=Recording2.non_theta_part['zscore_raw']
    optical_sleep=Recording3.non_theta_part['zscore_raw']
    fig, ax = plt.subplots(1, 1, figsize=(3, 6))
    OpticalAnlaysis.PSD_plot (optical_sleep,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle=':',label='PV-GECI-sleep',ax=ax)
    OpticalAnlaysis.PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='PV-GECI-rest',ax=ax)
    OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='PV-GECI-move',ax=ax)
    return -1

plot_PSD_for_rest_run ()