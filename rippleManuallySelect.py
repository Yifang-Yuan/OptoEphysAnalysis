# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:50:02 2024

@author: Yifang
"""

from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import pickle


'''RIPPLE CURATION'''
def Ripple_manual_select ():
    dpath='D:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/'
    recordingName='SavedPostSleepTrials'
    # dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day3/'
    # recordingName='SavedPostSleepTrials'
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
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