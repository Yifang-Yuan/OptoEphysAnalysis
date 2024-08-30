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
dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day1/'
recordingName='SavedOpenFieldTrials'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
LFP_channel='LFP_1'
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
#%%
'''To plot the feature of a part of the signal'''
start_time=38
end_time=43
coherence=Recording1.plot_freq_power_coherence (LFP_channel,start_time,end_time,SPAD_cutoff=100,lfp_cutoff=100)
#%%
Recording1.plot_segment_band_feature (LFP_channel,start_time,end_time,SPAD_cutoff=50,lfp_cutoff=100)
#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=5 #the duration of the segment, in seconds
viewNum=11 #the number of segments
for i in range(viewNum):
    Recording1.plot_segment_band_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=50,lfp_cutoff=100)

#%%
'''To plot the spectrum coherence for LFP and optical signal'''
start_time=0
end_time=50
coherence=Recording1.plot_freq_power_coherence (LFP_channel,start_time,end_time,SPAD_cutoff=100,lfp_cutoff=100)
#%%