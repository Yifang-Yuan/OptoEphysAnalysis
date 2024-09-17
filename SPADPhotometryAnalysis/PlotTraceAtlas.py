# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:12:54 2024

@author: Yifang
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import SPADreadBin
from SPADPhotometryAnalysis import photometry_functions as fp
import os
from scipy import signal

def plot_trace(trace,ax, fs=9938.4, label="trace",color='tab:blue'):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    ax.plot(taxis,trace,linewidth=0.5,label=label,color=color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.xaxis.set_visible(False)  # Hide x-axis
    #ax.yaxis.set_visible(False)  # Hide x-axis
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax

from scipy.ndimage import uniform_filter1d

def replace_outliers_with_nearest_avg(data, window_size=25000, z_thresh=3):
    # Calculate the mean and standard deviation of the moving window
    mean = uniform_filter1d(data, window_size, mode='reflect')
    std = uniform_filter1d(data**2, window_size, mode='reflect')
    std = np.sqrt(std - mean**2)

    # Identify the outliers
    outliers = (np.abs(data - mean) > z_thresh * std)

    # Replace outliers with the average of their nearest non-outlier neighbors
    for i in np.where(outliers)[0]:
        j = i - 1
        while j >= 0 and outliers[j]:
            j -= 1
        k = i + 1
        while k < len(data) and outliers[k]:
            k += 1
        if j >= 0 and k < len(data):
            data[i] = (data[j] + data[k]) / 2
        elif j >= 0:
            data[i] = data[j]
        elif k < len(data):
            data[i] = data[k]

    return data

def replace_outliers_with_avg(data, threshold):
    # Identify the outliers
    outliers = np.abs(data) > threshold

    # Get the indices of the outliers
    outlier_indices = np.where(outliers)[0]

    # Iterate over the outlier indices
    for index in outlier_indices:
        # Find the nearest non-outlier value
        left = index
        while left > 0 and outliers[left - 1]:
            left -= 1

        right = index
        while right < len(data) - 1 and outliers[right + 1]:
            right += 1

        # If the outlier is not at the boundaries of the array, replace it with the average of its neighbors
        if left != 0 and right != len(data) - 1:
            data[index] = (data[left - 1] + data[right + 1]) / 2
        # If the outlier is at the boundaries of the array, replace it with the nearest non-outlier value
        else:
            data[index] = data[left - 1] if left != 0 else data[right + 1]

    return data

def notchfilter (data,f0=100,bw=10,fs=840):
    # Bandwidth of the notch filter (in Hz)   
    Q = f0/bw # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    for _ in range(4):
        data = signal.filtfilt(b, a, data)
    return data
# Function to find the average of nearby valid values
def get_nearby_average(arr, idx, threshold):
    # Initialize left and right pointers
    left = idx - 1
    right = idx + 1
    valid_neighbors = []
    # Move the left pointer until we find a valid neighbor or reach the start
    while left >= 0:
        if arr[left] <= threshold:
            valid_neighbors.append(arr[left])
            break
        left -= 1
    # Move the right pointer until we find a valid neighbor or reach the end
    while right < len(arr):
        if arr[right] <= threshold:
            valid_neighbors.append(arr[right])
            break
        right += 1
    # Calculate the average of the valid neighbors
    if valid_neighbors:
        return np.mean(valid_neighbors)
    else:
        return arr[idx]  # If no valid neighbors, return the original value

def get_clean_data(data, threshold):
    data_cleaned = data.copy()
    above_threshold_mask = data > threshold
    outlier_indices = np.where(above_threshold_mask)[0]
    # Replace outliers with the average of nearby valid values
    for idx in outlier_indices:
        data_cleaned[idx] = get_nearby_average(data, idx, threshold)

    # Filter out consecutive indices, keeping only the first index of continuous peaks
    filtered_indices = []
    for i in range(len(outlier_indices)):
        if i == 0 or outlier_indices[i] > outlier_indices[i-1] + 1:
            filtered_indices.append(outlier_indices[i])
    return filtered_indices, data_cleaned

def get_average_opto_response_by_sync (data,threshold,half_window,sampling_rate):
    outlier_indices,data_cleaned=get_clean_data (data,threshold)
    half_win_size = int(half_window * sampling_rate)  # convert to samples

    windows = []
    for index in outlier_indices:
        start_index = index
        end_index = int(index+sampling_rate*half_window)
        if end_index <= len(data_cleaned) and start_index>=0:
            window_data = data_cleaned[start_index:end_index]
            windows.append(window_data)
    if windows:
        windows_array = np.array(windows)
        average_signals = np.mean(windows_array, axis=0)
        std_signals = np.std(windows_array, axis=0)
        # Plot the average signals and standard deviation as shaded area
        time_axis = np.arange(half_win_size) / sampling_rate  # Convert to time in seconds
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, average_signals, label='Average Signal')
        plt.fill_between(time_axis, average_signals - std_signals, average_signals + std_signals, color='b', alpha=0.2, label='Standard Deviation')
        plt.axvline(x=0, color='red', linestyle='--', label='Event Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title('Average Signal with Standard Deviation')
        plt.legend()
        plt.show()
    else:
        print("No windows to analyze. Adjust your threshold or window size.")
    return windows
        
#%%
'''Read binary files for single ROI'''
fs=840
dpath='D:/ATLAS_SPAD/1769566_PVcre_opto/SyncRecording4_1Hz2ms/'
# fs=1000
# dpath='G:/YY/New/1765508_Jedi2p_CompareSystem/Day2_pyPhotometry/SyncRecording4'
csv_filename='Green_traceAll.csv'
filepath=Analysis.Set_filename (dpath, csv_filename)
#filepath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/smallROI_100Hznoise.csv'
Trace_raw=Analysis.getSignalTrace (filepath, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)
Trace_raw=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(Trace_raw,ax, fs,label='840Hz')

#%% Sample data (replace this with your actual signal data)
data = Trace_raw
sampling_rate = 840  # samples per second
threshold = 12
window_duration =0.9# in seconds
windows=get_average_opto_response_by_sync (data,threshold,window_duration,sampling_rate)
#%%
outlier_indices,data_cleaned=get_clean_data (data,threshold)
#%%
for i in range (10):
    fig, ax = plt.subplots(figsize=(8,2))
    plot_trace(Trace_raw[fs*i:fs*(i+1)],ax, fs,label='840Hz')
    fig, ax = plt.subplots(figsize=(8,2))
    plot_trace(data_cleaned[fs*i:fs*(i+1)],ax, fs,label='840Hz')
#%%
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(data)), data, label='Data Above Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.legend()
plt.show()
# plt.figure(figsize=(12, 3))
# plt.plot(np.arange(len(data_cleaned)), data_cleaned, label='Data Above Threshold')
# plt.xlabel('Sample Index')
# plt.ylabel('Signal Value')
# plt.legend()
# plt.show()

trace_binned=Analysis.get_bin_trace (data_cleaned,bin_window=10,color='tab:blue',Fs=sampling_rate)

#%%
bin_window=2
Signal_bin=Analysis.get_bin_trace(Trace_raw,bin_window=bin_window,Fs=840)

bin_window=20
Signal_bin=Analysis.get_bin_trace(Trace_raw[10*840:30*840],bin_window=bin_window,Fs=840)

#SNR=Analysis.calculate_SNR(Trace_raw[0:9000])
#ATLAS 840

#%% Wavelet analysis
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from waveletFunctions import wave_signif, wavelet
import OpenEphysTools as OE

data=Trace_raw
signal_smooth= OE.butter_filter(data, btype='high', cutoff=5, fs=fs, order=2)
signal_smooth= OE.butter_filter(signal_smooth, btype='low', cutoff=200, fs=fs, order=3)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(signal_smooth[5*840:8*840],ax, fs,label='840Hz')
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal_smooth[5*840:8*840],lowpassCutoff=20,Fs=fs,scale=5)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)

#%%
data=Trace_raw
signal_smooth= OE.butter_filter(data, btype='high', cutoff=30, fs=fs, order=2)
signal_smooth= OE.butter_filter(signal_smooth, btype='low', cutoff=60, fs=fs, order=3)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(signal_smooth[22*840:23*840],ax, fs,label='840Hz')
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal_smooth[22*840:23*840],lowpassCutoff=60,Fs=fs,scale=5)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)
#%%
data=Trace_raw
signal_smooth= OE.butter_filter(data, btype='high', cutoff=130, fs=fs, order=2)
signal_smooth= OE.butter_filter(signal_smooth, btype='low', cutoff=180, fs=fs, order=3)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(signal_smooth[22*840:23*840],ax, fs,label='840Hz')
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal_smooth[22*840:23*840],lowpassCutoff=180,Fs=fs,scale=2)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)
