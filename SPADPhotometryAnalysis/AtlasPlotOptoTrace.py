# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:12:54 2024
This is to plot Atlas traces after decoding
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
    ax.set_ylabel('df/f')
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

def get_nearby_value(data, idx, threshold):
    # Find nearby values that are above the threshold
    left = idx - 1
    right = idx + 1
    while left >= 0 or right < len(data):
        # Check left neighbor
        if left >= 0 and data[left] > threshold:
            return data[left]
        # Check right neighbor
        if right < len(data) and data[right] > threshold:
            return data[right]
        left -= 1
        right += 1
    # If no valid neighbor found, return the original value (or some other fallback value)
    return data[idx]

def get_clean_data(data, outlier_thres,opto_thre):
    above_opto_threshold_mask = data > opto_thre
    opto_indices = np.where(above_opto_threshold_mask)[0]
    filtered_indices = []
    for i in range(len(opto_indices)):
        if i == 0 or opto_indices[i] > opto_indices[i-1] + 1:
            filtered_indices.append(opto_indices[i])
    
    data_cleaned = data.copy()
    above_threshold_mask = data > outlier_thres
    outlier_indices = np.where(above_threshold_mask)[0]
    # Replace outliers with the average of nearby valid values
    for idx in outlier_indices:
        data_cleaned[idx] = get_nearby_average(data, idx, outlier_thres)
    return filtered_indices, data_cleaned

def get_clean_data_below(data, outlier_length, opto_thre):

    # Find indices where data > threshold
    indices_above_threshold = np.where(data > opto_thre)[0]
    
    # Group consecutive indices
    pulse_indices = []
    current_group = []
    for i in indices_above_threshold:
        if current_group and i != current_group[-1] + 1:
            if len(current_group) >= outlier_length:
                pulse_indices.append(current_group)
            current_group = []
        current_group.append(i)
    if len(current_group) >= outlier_length:
        pulse_indices.append(current_group)
    
    # Flatten the list of indices
    flattened_pulse_indices = [i for group in pulse_indices for i in group]
    filtered_indices = []
    for i in range(len(flattened_pulse_indices)):
        if i == 0 or flattened_pulse_indices[i] > flattened_pulse_indices[i - 1] + 1:  # Ensure indices are not consecutive
            filtered_indices.append(flattened_pulse_indices[i])
    # Create a copy of the data and replace pulse values
    cleaned_data = data.copy()
    cleaned_data[flattened_pulse_indices] = np.nan  # Replace with NaN
    
    # Optionally interpolate the NaN values
    cleaned_data = np.interp(
        np.arange(len(cleaned_data)),
        np.arange(len(cleaned_data))[~np.isnan(cleaned_data)],
        cleaned_data[~np.isnan(cleaned_data)],
    )
    return filtered_indices,cleaned_data

# def get_clean_data_below(data, outlier_thres, opto_thre):
#     # Identify values below the opto threshold
#     below_opto_threshold_mask = data < opto_thre
#     opto_indices = np.where(below_opto_threshold_mask)[0]

#     # Filter out consecutive indices, keeping only the first index of continuous low values
#     filtered_indices = []
#     for i in range(len(opto_indices)):
#         if i == 0 or opto_indices[i] > opto_indices[i - 1] + 1:  # Ensure indices are not consecutive
#             filtered_indices.append(opto_indices[i])
#     # Create a copy of the data for cleaning
#     data_cleaned = data.copy()
#     # Identify outliers (values below the outlier threshold)
#     below_outlier_threshold_mask = data < outlier_thres
#     outlier_indices = np.where(below_outlier_threshold_mask)
#     # Replace outliers with the average of nearby valid values
#     for idx in outlier_indices:
#         data_cleaned[idx] = get_nearby_value(data, idx, outlier_thres)
#     return filtered_indices, data_cleaned

def get_average_opto_response_by_sync (data,outlier_thres,opto_thre,half_window,sampling_rate):
    opto_indices,data_cleaned=get_clean_data_below (data,outlier_thres,opto_thre)
    #opto_indices,data_cleaned=get_clean_data (data,outlier_thres,opto_thre)
    half_win_size = int(half_window * sampling_rate)  # convert to samples

    windows = []
    for index in opto_indices:
        start_index = index-84
        end_index = int(index+sampling_rate*half_window)
        if end_index <= len(data_cleaned) and start_index>=0:
            window_data = data_cleaned[start_index:end_index]
            windows.append(window_data)
    if windows:
        windows_array = np.array(windows)
        average_signals = np.mean(windows_array, axis=0)
        std_signals = np.std(windows_array, axis=0)
        # Plot the average signals and standard deviation as shaded area
        time_axis = np.arange(84+half_win_size) / sampling_rate  # Convert to time in seconds
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, average_signals, label='Average Signal')
        plt.fill_between(time_axis, average_signals - std_signals, average_signals + std_signals, color='b', alpha=0.2, label='Standard Deviation')
        plt.axvline(x=0.1, color='red', linestyle='--', label='Event Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title('Average Signal with Standard Deviation')
        plt.legend()
        plt.show()
    else:
        print("No windows to analyze. Adjust your threshold or window size.")
    return windows
def plot_segment_avg(data,segment_length):
    segment_length = segment_length  # Number of samples per 1-second segment
    num_segments = len(data) // segment_length  # Total number of segments (10)
    # Reshape data into segments of 1 second each (10 segments, each with 840 samples)
    segments = np.reshape(data[:num_segments * segment_length], (num_segments, segment_length))
    
    # Calculate mean and std across the segments for each sample point
    mean_across_segments = np.mean(segments, axis=0)
    std_across_segments = np.std(segments, axis=0)
    
    # Plot
    time = np.arange(segment_length) / fs  # Time in seconds for each sample within the segment
    
    plt.figure(figsize=(10, 5))
    plt.plot(time, mean_across_segments, label='Mean', color='blue')
    plt.fill_between(time, mean_across_segments - std_across_segments, mean_across_segments + std_across_segments, 
                     color='blue', alpha=0.3, label='Standard Deviation')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Average Signal with Standard Deviation Over 1-second Segments')
    plt.legend()
    plt.show()
#%%
'''Read binary files for single ROI'''
fs=840
dpath='E:/ATLAS_SPAD/1769566_PVcre_opto_ipslateral/Day2/SyncRecording1/'
# fs=1000
# dpath='G:/YY/New/1765508_Jedi2p_CompareSystem/Day2_pyPhotometry/SyncRecording4'
csv_filename='Zscore_traceAll.csv'
filepath=Analysis.Set_filename (dpath, csv_filename)
#filepath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/smallROI_100Hznoise.csv'
Trace_raw=Analysis.getSignalTrace (filepath, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)
data=Trace_raw
#Trace_raw=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(data,ax, fs,label='840Hz')

# data=Trace_raw[8400:16800]
# #Trace_raw=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
# fig, ax = plt.subplots(figsize=(8,2))
# plot_trace(data,ax, fs,label='840Hz')

# data=Trace_raw[16800:25200]
# #Trace_raw=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
# fig, ax = plt.subplots(figsize=(8,2))
# plot_trace(data,ax, fs,label='840Hz')
#%%
data=Trace_raw
plot_segment_avg(data,840)

#%% Wavelet analysis
import matplotlib.pylab as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from waveletFunctions import wave_signif, wavelet
import OpenEphysTools as OE

data=Trace_raw[0:16800]
signal= OE.butter_filter(data, btype='high', cutoff=5, fs=fs, order=5)
signal_smooth= OE.butter_filter(signal, btype='low', cutoff=200, fs=fs, order=5)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(signal_smooth,ax, fs,label='840Hz')
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal,lowpassCutoff=200,Fs=fs,scale=10)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)

#%%
data=Trace_raw[0:8400]
signal_smooth= OE.butter_filter(data, btype='high', cutoff=30, fs=fs, order=2)
signal_smooth= OE.butter_filter(signal_smooth, btype='low', cutoff=60, fs=fs, order=3)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(signal_smooth,ax, fs,label='840Hz')
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal_smooth,lowpassCutoff=60,Fs=fs,scale=5)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)
#%%
data=Trace_raw
signal_smooth= OE.butter_filter(data, btype='high', cutoff=130, fs=fs, order=2)
signal_smooth= OE.butter_filter(signal_smooth, btype='low', cutoff=180, fs=fs, order=3)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(signal_smooth,ax, fs,label='840Hz')
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal_smooth,lowpassCutoff=180,Fs=fs,scale=2)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)
#%%
#%%
'Find sync by opto peaks '
data = Trace_raw[0:8400]
outlier_length=9
opto_thre = 5
window_duration =0.9# in seconds
windows=get_average_opto_response_by_sync (data,outlier_length,opto_thre,window_duration,fs)
opto_indices,data_cleaned=get_clean_data_below (data,outlier_length,opto_thre)
#%%
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(data_cleaned,ax, fs,label='840Hz')
#%%
for i in range (30):
    fig, ax = plt.subplots(2,1,figsize=(8,4))
    plot_trace(data[opto_indices[i]:opto_indices[i]+840],ax[0], fs,label='840Hz')
    plot_trace(data_cleaned[opto_indices[i]:opto_indices[i]+840],ax[1], fs,label='840Hz')
    ax[1].axvline(x=0, color='red', linestyle='--', label='Event Time')
#%%
data_cleaned=pd.Series(data_cleaned[840*0:840*10])
Signal_bin=fp.smooth_signal(data_cleaned,10,'flat')
fig, ax = plt.subplots(figsize=(8,2))
ax=plot_trace(Signal_bin,ax, fs,label='840Hz')
for i in range(10):
    ax.axvline(x=opto_indices[i]/fs, color='red', linestyle='--', label='Event Time')
#%%
Signal_bin=fp.smooth_signal(data_cleaned,2,'flat')
fig, ax = plt.subplots(figsize=(8,2))
ax=plot_trace(Signal_bin[0:840*2],ax, fs,label='840Hz')
for i in range(2):
    ax.axvline(x=opto_indices[i]/fs, color='red', linestyle='--', label='Event Time')
