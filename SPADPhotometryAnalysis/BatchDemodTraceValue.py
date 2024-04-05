# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:03:58 2024

@author: Yifang
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import SPADreadBin
import os
import shutil

def plot_section_for_threshold(fs,dpath):
    '''Time division mode with one ROI, GCamp and isosbestic'''
    '''Read files'''
    filename=Analysis.Set_filename (dpath, csv_filename="traceValueAll.csv")
    Trace_raw=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=100)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    Analysis.plot_trace(Trace_raw,ax, fs=9938.4, label="Full raw data trace")
    fig, ax = plt.subplots(figsize=(12, 2.5))
    Analysis.plot_trace(Trace_raw[0:200],ax, fs=9938.4, label="Part raw data trace")
    
    return -1

def check_ROI (dpath):
    '''Show images'''
    filename = os.path.join(dpath, "spc_data1.bin")
    Bindata=SPADreadBin.SPADreadBin(filename,pyGUI=False)
    SPADreadBin.ShowImage(Bindata,dpath) 
    return -1

def replace_outliers(arr, threshold=5):
    # Calculate the mean and standard deviation of the array
    mean = np.mean(arr)
    std = np.std(arr)
    # Create a mask to identify outliers
    mask = np.abs(arr - mean) > threshold * std
    # Replace outliers with the mean of neighboring values
    neighbor_mean = np.convolve(arr, np.ones(3) / 3, mode='same')  # Calculate the mean of neighboring values
    cleaned_arr = np.where(mask, neighbor_mean, arr)  # Replace outliers with the neighboring mean
    return cleaned_arr

def demod_multiple_SPAD_folders_save_zscore(parent_folder,high_thd=12000,low_thd=6000):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    def sort_by_trial_index(file_name):
        trial_index = int(file_name.split("Trial")[-1])
        return trial_index
    folder_names = [folder for folder in os.listdir(parent_folder) if 'Trial' in folder]
    sorted_folders = sorted(folder_names, key=sort_by_trial_index)
    # Read the directories in sorted order
    for folder_name in sorted_folders:
        directory=os.path.join(parent_folder, folder_name)
        print("Folder name:", directory)
        'Demodulate single ROI time division recodings'
        filename=Analysis.Set_filename (directory, csv_filename="traceValueAll.csv")
        Trace_raw=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=100)
        Green,Red= Analysis.getTimeDivisionTrace_fromMask (directory, Trace_raw, high_thd=high_thd,low_thd=low_thd)
        Green=replace_outliers(Green, threshold=3)
        Red=replace_outliers(Red, threshold=3)
        z_sig,smooth_sig,corrected_sig=Analysis.photometry_smooth_plot (Red,Green,
                                                                                  sampling_rate=9938.4,smooth_win =20)
        zscorefname = os.path.join(directory, "Zscore_traceAll.csv")
        np.savetxt(zscorefname, z_sig, delimiter=",")
    return -1

def copy_file(file_to_copy,source_dir,destination_dir):
    if file_to_copy in os.listdir(source_dir):
        # Construct the source and destination file paths
        source_file = os.path.join(source_dir, file_to_copy)
        destination_file = os.path.join(destination_dir, file_to_copy)
        # Copy the file to the destination directory
        shutil.copy(source_file, destination_file)
    else:
        print(f"File '{file_to_copy}' does not exist in the source directory.")
    return -1

def copy_results_to_SyncRecording (day_parent_folder,SPAD_parent_folder,new_folder_name='SyncRecording'):
    # Sort the directories by creation time
    def sort_by_trial_index(file_name):
        trial_index = int(file_name.split("Trial")[-1])
        return trial_index
    folder_names = [folder for folder in os.listdir(SPAD_parent_folder) if 'Trial' in folder]
    sorted_folders = sorted(folder_names, key=sort_by_trial_index)
    # Read the directories in sorted order
    i=0
    for folder in sorted_folders:
        i=i+1
        source_dir=os.path.join(SPAD_parent_folder, folder)
        folder_name = f'{new_folder_name}{i}'
        destination_dir = os.path.join(day_parent_folder, folder_name)
        print ('source_dir is', source_dir)
        print ('destination_dir is', destination_dir)
        # Create the folder if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        file_to_copy = 'Zscore_traceAll.csv'
        copy_file(file_to_copy,source_dir,destination_dir)
        file_to_copy = 'Green_traceAll.csv'
        copy_file(file_to_copy,source_dir,destination_dir)
        file_to_copy = 'Red_traceAll.csv'
        copy_file(file_to_copy,source_dir,destination_dir)
    return -1
#%%
fs   = 9938.4
day_parent_folder="F:/2024MScR_NORtask/1732333_SPAD/20240305_Day2/"
parent_folder=os.path.join(day_parent_folder, 'SPAD')
#%%
# Step 1. check threshold can signal quality
'Plot the 1st trial and the last trial to check whether the signal is stable'
dpath1=os.path.join(parent_folder, '2024_3_5_14_43_11_Trial1')
plot_section_for_threshold(fs,dpath1)
dpath2=os.path.join(parent_folder, '2024_3_5_15_44_34_Trial6')
plot_section_for_threshold(fs,dpath2)
#%%
# Step 2. Do the batch demodulation.
'Set the high_thd to a value larger than the maximun in the above plot, and low_thd to a value lower than the signal minimun value. '
demod_multiple_SPAD_folders_save_zscore(parent_folder,high_thd=1200,low_thd=600)

#%%
# Step 3. Copy the files to the SyncRecording folder.
copy_results_to_SyncRecording (day_parent_folder,SPAD_parent_folder=parent_folder,new_folder_name='SyncRecording')
#%%
'Check ROI plot if the photon count is too small'
check_ROI (dpath1)