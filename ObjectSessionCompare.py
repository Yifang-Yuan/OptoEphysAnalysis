# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 22:04:38 2024

@author: Yifang
"""
import os
import numpy as np
import pandas as pd
import pickle
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE

def ReadOneDaySession (parent_folder,folderName='SyncRecording', IsTracking=False,
                       read_aligned_data_from_file=False,read_class_data_from_file=False):
    
    # Create an empty DataFrame with specified columns
    columns = ['TrialName', 'Timestamp', 'RippleNumLFP1','RippleFreqLFP1','RippleNumLFP2','RippleFreqLFP2',
               'RippleNumLFP3','RippleFreqLFP3','RippleNumLFP4','RippleFreqLFP4']
    SessionRippleInfo = pd.DataFrame(columns=columns)
    
    
    # List all files and directories in the parent folder
    all_contents = os.listdir(parent_folder)
    # Filter out directories containing the target string
    sync_recording_folders = [folder for folder in all_contents if "SyncRecording" in folder]
    # Define a custom sorting key function to sort folders in numeric order
    def numeric_sort_key(folder_name):
        return int(folder_name.lstrip("SyncRecording"))
    # Sort the folders in numeric order
    sync_recording_folders.sort(key=numeric_sort_key)
    # Iterate over each sync recording folder
    for SyncRecordingName in sync_recording_folders:
        
        # Now you can perform operations on each folder, such as reading files inside it
        print("Now processing folder:", SyncRecordingName)
        Recording1=SyncOEpyPhotometrySession(parent_folder,SyncRecordingName,IsTracking,read_aligned_data_from_file,read_class_data_from_file) 
        
        for i in range (4):
            LFP_channel='LFP_'+str(i+1)
            
            theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0.5,High_thres=10)
            
            '''RIPPLE DETECTION
            For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
            ripple_band_filtered,rip_ep,rip_tsd,cross_corr_values=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=80,
                                                                                      Low_thres=2,High_thres=10,plot_segment=False,plot_ripple_ep=True,excludeTheta=True)
            #This is to plot the average calcium transient around a ripple peak
            figName=SyncRecordingName+'_RipplePeakOpticalSignal_'+LFP_channel+'.png'
            fig_save_path=os.path.join(parent_folder,'Results',figName)
            transient_trace=Recording1.Ephys_tracking_spad_aligned['zscore_raw']
            mean_z_score,std_z_score=OE.Transient_during_LFP_event (fig_save_path,rip_tsd,transient_trace,half_window=0.2,fs=10000)
            
            '''THETA PEAK DETECTION
            For a rigid threshold to get larger amplitude theta events: Low_thres=1, for more ripple events, Low_thres=0.5'''
            theta_band_filtered,rip_ep,rip_tsd,cross_corr_values=Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,ep_start=0,ep_end=100,
                                                                                     Low_thres=0.5,High_thres=10,plot_ripple_ep=True)
            figName=SyncRecordingName+'_ThetaPeakOpticalSignal_'+LFP_channel+'.png'
            fig_save_path=os.path.join(parent_folder,'Results',figName)
            #This is to plot the average calcium transient around a ripple peak
            transient_trace=Recording1.Ephys_tracking_spad_aligned['zscore_raw']
            mean_z_score,std_z_score=OE.Transient_during_LFP_event (fig_save_path,rip_tsd,transient_trace,half_window=0.5,fs=10000)
            
        'Save Current Recording Class to pickle'
        current_trial_folder_path = os.path.join(parent_folder, SyncRecordingName)
        Trial_save_path = os.path.join(current_trial_folder_path, SyncRecordingName+'Class.pkl')
        with open(Trial_save_path, "wb") as file:
            # Serialize and write the instance to the file
            pickle.dump(Recording1, file)
        
        'Add ripple information of Current Recording Trial to the Session DataFrame '
        Row_to_add = {'TrialName': SyncRecordingName,'Timestamp':Recording1.Ephys_tracking_spad_aligned['timestamps'][0],
                      'RippleNumLFP1':Recording1.ripple_numbers['LFP_1'],'RippleFreqLFP1':Recording1.ripple_freq['LFP_1'], 
                      'RippleNumLFP2':Recording1.ripple_numbers['LFP_2'],'RippleFreqLFP2':Recording1.ripple_freq['LFP_2'], 
                      'RippleNumLFP3':Recording1.ripple_numbers['LFP_3'],'RippleFreqLFP3':Recording1.ripple_freq['LFP_3'], 
                      'RippleNumLFP4':Recording1.ripple_numbers['LFP_4'],'RippleFreqLFP4':Recording1.ripple_freq['LFP_4'] }
        SessionRippleInfo.loc[len(SessionRippleInfo)] = Row_to_add
        Session_save_path = os.path.join(parent_folder, 'Results','SessionRippleInfo.pkl')
        SessionRippleInfo.to_pickle(Session_save_path)

    return SessionRippleInfo                                                                   


parent_folder='E:/YYFstudy/20240214_Day3'

SessionRippleInfo=ReadOneDaySession (parent_folder,folderName='SyncRecording', IsTracking=False,read_aligned_data_from_file=False,read_class_data_from_file=False)

#%%
import matplotlib.pyplot as plt
#SessionRippleInfo.set_index('TrialName', inplace=True)
# Plot multiple columns as a bar plot
columns_to_plot = ['RippleFreqLFP1', 'RippleFreqLFP2','RippleFreqLFP3','RippleFreqLFP4']
plt.figure(figsize=(20, 12))
ax=SessionRippleInfo[columns_to_plot].plot(kind='line')
plt.xticks(rotation='vertical')
# Show the plot
plt.show()
#%%
SessionRippleInfo['tag'] = ['pre','pre','pre','pre','pre','pre','pre','OF','post','post','post','post','post','post','post']
SessionRippleInfo['sleep'] = ['awake','awake','awake','awake','awake','awake','awake','awake','sleep','sleep','sleep','sleep','sleep','sleep','sleep']