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
#%%
'''recordingMode: use py, Atlas, SPAD for different systems
'''
dpath='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/'
recordingName='SavedPostSleepTrials'
#dpath="G:/SPAD/SPADData/20230722_SPADOE/SyncRecording0/"
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
#%%
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_1'
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
                                                                          plot_ripple_ep=True,excludeTheta=True)
#%%
save_path = os.path.join(dpath, recordingName,LFP_channel+'_Class.pkl')
with open(save_path, "wb") as file:
    # Serialize and write the instance to the file
    pickle.dump(Recording1, file)
#%%
ripple_triggered_zscore_values=Recording1.ripple_triggered_zscore_values
ripple_triggered_LFP_values_1=Recording1.ripple_triggered_LFP_values_1