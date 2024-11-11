# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 22:04:38 2024
@author: Yifang
Use this code to read all trials and save SyncOEC**SessionClass as pickle files with information of each single Recording.
Note:I named it as SyncOECSessionClass but it is actually a single recording trial. 
"""
import os
import numpy as np
import pandas as pd
import pickle
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE

def ReadOneDaySession (parent_folder,TargetfolderName='SyncRecording', IsTracking=False,
                       read_aligned_data_from_file=False,recordingMode='SPAD',indicator='GEVI'):
    
    # List all files and directories in the parent folder
    all_contents = os.listdir(parent_folder)
    # Filter out directories containing the target string
    sync_recording_folders = [folder for folder in all_contents if TargetfolderName in folder]
    # Define a custom sorting key function to sort folders in numeric order
    def numeric_sort_key(folder_name):
        return int(folder_name.lstrip(TargetfolderName))
    # Sort the folders in numeric order
    sync_recording_folders.sort(key=numeric_sort_key)
    # Iterate over each sync recording folder
    for SyncRecordingName in sync_recording_folders:
        # Now you can perform operations on each folder, such as reading files inside it
        print("----Now processing folder:", SyncRecordingName)
        Recording1=SyncOEpyPhotometrySession(parent_folder,SyncRecordingName,IsTracking=IsTracking,
                                             read_aligned_data_from_file=read_aligned_data_from_file,
                                             recordingMode=recordingMode,indicator=indicator) 
        for i in range (4):
            LFP_channel='LFP_'+str(i+1)
            #LFP_channel='LFP_4'
            theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0.5,High_thres=10)
            '''RIPPLE DETECTION
            For a rigid threshold to get larger amplitude ripple events: Low_thres=3'''
            rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,
                                                        ep_start=0,ep_end=80,Low_thres=2,High_thres=10, 
                                                        plot_segment=False,plot_ripple_ep=False,excludeTheta=True)
            '''THETA PEAK DETECTION
            For a rigid threshold to get larger amplitude theta events: Low_thres=1, for more ripple events, Low_thres=0.5'''
            rip_ep,rip_tsd=Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,
                                                             ep_start=0,ep_end=100,Low_thres=0.5,High_thres=10,
                                                             plot_segment=False,plot_ripple_ep=False)
            Recording1.plot_theta_correlation(LFP_channel)
            'Save Current Recording Class for this LFP channel to pickle'
            # current_trial_folder_path = os.path.join(parent_folder, SyncRecordingName)
            # Trial_save_path = os.path.join(current_trial_folder_path, SyncRecordingName+LFP_channel+'_Class.pkl')
            # with open(Trial_save_path, "wb") as file:
            #     # Serialize and write the instance to the file
            #     pickle.dump(Recording1, file)
    return -1                                                                   

def main():    
    '''
    Put all your parent folders here for batch processing.
    recordingMode: 'py' for pyPhotometry recording, 'SPAD' for SPAD-SPC recording
    '''
    parent_folder='E:/ATLAS_SPAD/1825505_SimCre/Day2/'
    ReadOneDaySession (parent_folder,TargetfolderName='SyncRecording', 
                                          IsTracking=True,read_aligned_data_from_file=False,recordingMode='Atlas',indicator='GEVI')
    
if __name__ == "__main__":
    main()
