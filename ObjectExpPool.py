# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:44:44 2024

@author: Yifang
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import OpenEphysTools as OE
import pynapple as nap
import pickle

def PoolTrialsByState (parent_folder,LFP_channel='LFP_1'):
    all_contents = os.listdir(parent_folder)
    # Filter out directories containing the target string
    day_recording_folders = [folder for folder in all_contents if 'Day' in folder]
    # Define a custom sorting key function to sort folders in numeric order
    sorted_folders = sorted(day_recording_folders, key=lambda x: int(x.split('Day')[-1]))
    # Iterate over each sync recording folder
    ripple_triggered_optical_peak_std={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_triggered_optical_peak_time={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_triggered_zscore={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}

    for DayRecordingFolder in sorted_folders:
        # Now you can perform operations on each folder, such as reading files inside it
        print("Day folder:", DayRecordingFolder)
        day_folder_path= os.path.join(parent_folder, DayRecordingFolder)
        all_contents = os.listdir(day_folder_path)
        sync_recording_folders = [folder for folder in all_contents if 'SyncRecording' in folder]
        # Define a custom sorting key function to sort folders in numeric order
        def numeric_sort_key(folder_name):
            return int(folder_name.lstrip('SyncRecording'))
        sync_recording_folders.sort(key=numeric_sort_key)
        for SyncRecordingName in sync_recording_folders:
            print("Now processing folder:", SyncRecordingName)
            Classfilepath=os.path.join(day_folder_path, SyncRecordingName,SyncRecordingName+'Class.pkl')
            print ("Class file name:", Classfilepath)
            with open(Classfilepath, 'rb') as file:
                Recording1 = pickle.load(file)
                column_name=Recording1.TrainingState+'_'+Recording1.sleepState

                ripple_triggered_optical_peak_std [column_name].append(Recording1.ripple_triggered_optical_peak_std_array)
                ripple_triggered_optical_peak_time [column_name].append(Recording1.ripple_triggered_optical_peak_time_array)
                ripple_triggered_zscore [column_name].append(Recording1.ripple_triggered_zscore_values_arrary)
    
    return ripple_triggered_optical_peak_std,ripple_triggered_optical_peak_time,ripple_triggered_zscore


parent_folder='E:/YYFstudy/Exp1'
ripple_triggered_optical_peak_std,ripple_triggered_optical_peak_time,ripple_triggered_zscore=PoolTrialsByState (parent_folder,LFP_channel='LFP_4')
#%%
