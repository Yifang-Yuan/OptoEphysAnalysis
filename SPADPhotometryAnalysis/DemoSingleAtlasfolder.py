# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:05:21 2024

@author: Yifang
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
from SPADPhotometryAnalysis import AtlasDecode
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
#%% Workable code, above is testin
dpath='G:/GEVItest/1819287_mNeon/Burst-RS-25200frames-840Hz_2024-08-23_11-52/'
#dpath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/Burst-RS-1017frames-1017Hz_4uW/'
hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
xxrange = [45, 85]
yyrange = [50, 90]

Trace_raw,z_score=AtlasDecode.get_zscore_from_atlas_continuous (dpath,hotpixel_path,xxrange=xxrange,yyrange=yyrange,fs=840,photoncount_thre=500)
#%%
data=Trace_raw
#%%
sampling_rate=840
'''Wavelet spectrum ananlysis'''
Analysis.plot_wavelet_data(data,sampling_rate,cutoff=300,xlim = ([6,30]))

#%%
#Plot the image of the pixel array
fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(Trace_raw[800:1600],ax, fs=840, label="raw_data")
'''Read binary files for single ROI'''
# Display the grayscale image
#pixel_array_all_frames,sum_pixel_array,avg_pixel_array=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=500)
#%%
#show_image_with_pixel_array(avg_pixel_array,showPixel_label=True)
#%%
# sum_values_over_time,mean_values_over_time,region_pixel_array=get_trace_from_3d_pixel_array(pixel_array_all_frames,avg_pixel_array,xxrange,yyrange)
# fig, ax = plt.subplots(figsize=(8, 2))
# plot_trace(sum_values_over_time[1:],ax, fs=840, label="raw_data")
#%%
# for i in range(21):
#     show_image_with_pixel_array(pixel_array_all_frames[:,:,187+i],showPixel_label=True)
#pixel_array=pixel_array_all_frames[:,:,800]
#pixel_array_plot_hist(pixel_array_all_frames[:,:,1000], plot_min_thre=100)
#%%
# Sample data (replace this with your actual signal data)
data = Trace_raw
sampling_rate = 840  # samples per second
threshold = -2
window_duration = 0.2 # in seconds
# Function to find the average of nearby valid values
def get_nearby_average(arr, idx):
    # Initialize left and right pointers
    left = idx - 1
    right = idx + 1
    valid_neighbors = []
    # Move the left pointer until we find a valid neighbor or reach the start
    while left >= 0:
        if arr[left] >= threshold:
            valid_neighbors.append(arr[left])
            break
        left -= 1
    # Move the right pointer until we find a valid neighbor or reach the end
    while right < len(arr):
        if arr[right] >= threshold:
            valid_neighbors.append(arr[right])
            break
        right += 1
    # Calculate the average of the valid neighbors
    if valid_neighbors:
        return np.mean(valid_neighbors)
    else:
        return arr[idx]  # If no valid neighbors, return the original value

def get_clean_data (data,threshold):
    data_cleaned = data.copy()
    below_threshold_mask = data < threshold
    outlier_indices = np.where(below_threshold_mask)[0]
    # Replace outliers with the average of nearby valid values
    for idx in outlier_indices:
        data_cleaned[idx] = get_nearby_average(data, idx)
    return data_cleaned

def get_average_opto_response_by_sync (data,threshold,window_duration,sampling_rate):
    data_cleaned=get_clean_data (data,threshold)
    window_size = int(window_duration * sampling_rate)  # convert to samples
    # Step 2: Detect rising edges
    below_threshold_mask = data < threshold
    falling_edges = np.where(np.diff(below_threshold_mask.astype(int)) == -1)[0]
    # Step 3: Define windows starting at rising edges
    windows = []
    for index in falling_edges:
        start_index = int(index-84)
        end_index = start_index + window_size
        if end_index <= len(data_cleaned) and start_index>=0:
            window_data = data_cleaned[start_index:end_index]
            windows.append(window_data)
    # Step 4: Calculate average signal and standard deviation for each window
    if windows:
        windows_array = np.array(windows)
        average_signals = np.mean(windows_array, axis=0)
        std_signals = np.std(windows_array, axis=0)
        # Plot the average signals and standard deviation as shaded area
        time_axis = np.arange(window_size) / sampling_rate  # Convert to time in seconds
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
        
def get_average_opto_response_by_fixed_freq (data,window_duration,sampling_rate,num_windows):
    window_size = int(window_duration * sampling_rate)  # convert to samples
    windows = []
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size
        window_data = data[start_index:end_index]
        windows.append(window_data)
    # Step 4: Calculate average signal and standard deviation for each window
    if windows:
        windows_array = np.array(windows)
        average_signals = np.mean(windows_array, axis=0)
        std_signals = np.std(windows_array, axis=0)
        # Plot the average signals and standard deviation as shaded area
        time_axis = np.arange(window_size) / sampling_rate  # Convert to time in seconds
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, average_signals, label='Average Signal')
        plt.fill_between(time_axis, average_signals - std_signals, average_signals + std_signals, color='b', alpha=0.2, label='Standard Deviation')
        #plt.axvline(x=0.1, color='red', linestyle='--', label='Event Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        plt.title('Average Signal with Standard Deviation')
        plt.legend()
        plt.show()
    else:
        print("No windows to analyze. Adjust your threshold or window size.")
    return windows
#windows=get_average_opto_response_by_sync (data,threshold,window_duration,sampling_rate)  
#%%
get_average_opto_response_by_sync (data,threshold,window_duration,sampling_rate)
#%%
window_duration = 1.0 # in seconds
windows=get_average_opto_response_by_fixed_freq (data,window_duration,sampling_rate,num_windows=10)
#%%
data_cleaned=get_clean_data (data,threshold)
windows=get_average_opto_response_by_fixed_freq (data_cleaned,window_duration,sampling_rate,num_windows=10)
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
