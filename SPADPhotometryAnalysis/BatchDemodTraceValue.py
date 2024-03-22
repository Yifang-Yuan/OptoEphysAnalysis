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
    # Get a list of all directories in the parent folder
    directories = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    # Sort the directories by creation time
    directories.sort(key=lambda x: os.path.getctime(x))
    # Read the directories in sorted order
    for directory in directories:
        print("Folder:", directory)
        'Demodulate single ROI time division recodings'
        filename=Analysis.Set_filename (directory, csv_filename="traceValueAll.csv")
        Trace_raw=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=100)
        Green,Red= Analysis.getTimeDivisionTrace_fromMask (directory, Trace_raw, high_thd=high_thd,low_thd=low_thd)
        Green=replace_outliers(Green, threshold=5)
        Red=replace_outliers(Red, threshold=5)
        z_sig,smooth_sig,corrected_sig=Analysis.photometry_smooth_plot (Red,Green,
                                                                                  sampling_rate=9938.4,smooth_win =20)
        zscorefname = os.path.join(directory, "Zscore_traceAll.csv")
        np.savetxt(zscorefname, z_sig, delimiter=",")
    return -1
    
#%%
# Sampling Frequency
fs   = 9938.4
'''Read binary files for single ROI'''
parent_folder="F:/2024MScR_NORtask/1732333_SPAD/20240304_Day1/SPAD"

'Plot the 1st trial and the last trial to check whether the signal is stable'
dpath1="F:/2024MScR_NORtask/1732333_SPAD/20240304_Day1/SPAD/2024_3_4_15_33_57_Trial1/"
plot_section_for_threshold(fs,dpath1)
dpath2="F:/2024MScR_NORtask/1732333_SPAD/20240304_Day1/SPAD/2024_3_4_18_17_51_Trial11/"
plot_section_for_threshold(fs,dpath2)

#%%
'Set the high_thd to a value larger than the maximun in the above plot, and low_thd to a value lower than the signal minimun value. '
demod_multiple_SPAD_folders_save_zscore(parent_folder,high_thd=12000,low_thd=7000)
#%%
'check ROI plot if the photon count is too small'
check_ROI (dpath1)