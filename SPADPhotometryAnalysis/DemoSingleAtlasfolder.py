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
#%% Workable code, above is testin
dpath='E:/YYFstudy/1769567_1Hz10ms/1769567_1Hz10ms/Burst-RS-8400frames-840Hz_2024-08-01_16-49/'
#dpath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/Burst-RS-1017frames-1017Hz_4uW/'
hotpixel_path='F:/SPADdata/Altas_hotpixel.csv'
xxrange = [45, 85]
yyrange = [50, 90]

Trace_raw,z_score=AtlasDecode.get_zscore_from_atlas_continuous (dpath,hotpixel_path,xxrange=xxrange,yyrange=yyrange,fs=840,photoncount_thre=500)
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
# Parameters
sampling_rate = 840  # samples per second
threshold = 8.4
window_duration = 0.5  # in seconds
window_size = int(window_duration * sampling_rate)  # convert to samples

# Step 1: Identify data points below threshold
below_threshold_mask = data < threshold

# Step 2: Detect rising edges
falling_edges = np.where(np.diff(below_threshold_mask.astype(int)) == -1)[0]

# Step 3: Define windows starting at rising edges
windows = []
for index in falling_edges:
    start_index = index + 1
    end_index = start_index + window_size
    if end_index <= len(data):
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
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.title('Average Signal with Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No windows to analyze. Adjust your threshold or window size.")
    
#%%
# Step 2: Remove data points below threshold
data_cleaned = data[~below_threshold_mask]

# Step 3: Plot the original and new traces
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(data_cleaned)), data_cleaned, label='Data Above Threshold')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.legend()
plt.show()