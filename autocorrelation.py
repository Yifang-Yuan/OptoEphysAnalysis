# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:36:14 2025

@author: Yifang
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
def compute_autocorrelation(signal_data, fs, max_lag=1.0):
    """ Compute and plot the normalized autocorrelation of a signal """
    signal_data = signal_data - np.mean(signal_data)  # Remove DC component
    autocorr = np.correlate(signal_data, signal_data, mode='full')  # Compute autocorrelation
    autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags
    
    autocorr /= autocorr[0]  # Normalize by the value at lag=0

    time_lags = np.arange(len(autocorr)) / fs  # Convert lags to time (seconds)

    # Find index corresponding to max_lag seconds
    max_lag_idx = int(max_lag * fs)
    
    # Plot only up to max_lag
    plt.figure(figsize=(6, 4))
    plt.plot(time_lags[:max_lag_idx], autocorr[:max_lag_idx], color='black')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Normalized Autocorrelation')
    plt.title('Autocorrelation of Signal')

    # Set y-axis limits
    plt.ylim(-1, 1)  
    plt.xlim(0, max_lag)  

    plt.show()
    
    return autocorr[:max_lag_idx], time_lags[:max_lag_idx]

def compute_segmented_autocorrelation(signal_data, fs, max_lag=1.0):
    """ Compute and plot the averaged autocorrelation with confidence interval """
    
    # Remove DC component
    signal_data = signal_data - np.mean(signal_data)
    
    # Convert max_lag (in seconds) to samples
    max_lag_samples = int(max_lag * fs)
    
    # Split signal into non-overlapping segments
    num_segments = len(signal_data) // max_lag_samples
    segments = np.array_split(signal_data[:num_segments * max_lag_samples], num_segments)

    # Compute autocorrelation for each segment
    autocorr_list = []
    for segment in segments:
        autocorr = np.correlate(segment, segment, mode='full')  # Full autocorrelation
        autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags
        autocorr /= autocorr[0]  # Normalize
        autocorr_list.append(autocorr[:max_lag_samples])  # Truncate at max_lag

    # Convert to numpy array
    autocorr_array = np.array(autocorr_list)

    # Compute mean and confidence intervals
    mean_autocorr = np.mean(autocorr_array, axis=0)
    std_autocorr = np.std(autocorr_array, axis=0)
    ci_upper = mean_autocorr + 1.96 * std_autocorr / np.sqrt(len(segments))
    ci_lower = mean_autocorr - 1.96 * std_autocorr / np.sqrt(len(segments))

    # Time lags for plotting
    time_lags = np.arange(max_lag_samples) / fs

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(time_lags, mean_autocorr, color='black', label='Mean Autocorrelation')
    plt.fill_between(time_lags, ci_lower, ci_upper, color='gray', alpha=0.3, label='95% CI')
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Normalized Autocorrelation')
    plt.title('Segmented & Averaged Autocorrelation')
    plt.legend()
    plt.ylim(-1, 1)
    plt.xlim(0, max_lag)

    plt.show()
    
    return mean_autocorr, ci_lower, ci_upper, time_lags


'Autocorrelation for LFP signals'
# file_path_ephys=os.path.join(dpath, "open_ephys_read_pd.pkl")
# EphysData = pd.read_pickle(file_path_ephys)
# signal_data = EphysData['LFP_1'].values
# fs=30000
# autocorr, time_lags = compute_autocorrelation(signal_data, fs,max_lag=1)
#mean_autocorr, ci_lower, ci_upper, time_lags = compute_segmented_autocorrelation(signal_data, fs, max_lag=1)

'Autocorrelation for optical signals'
dpath = 'D:/2025_ATLAS_SPAD/1842515_PV_mNeon/Day7/SyncRecording3'
#dpath='D:/2024_OEC_Atlas_main/1765010_PVGCaMP8f_Atlas/Day1/SyncRecording1/'
#dpath='D:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day2/SyncRecording1/'
filename = "Zscore_traceAll.csv"
filepath = os.path.join(dpath, filename)
trace = pd.read_csv(filepath, header=None).squeeze("columns")

#Select a portion of the signal
fs = 841.68  # in Hz
signal_data = trace # Select 10-second segment
# Filter the signal
# cutoff_frequency = 50  # in Hz
# signal_data = butter_lowpass_filter(trace, cutoff_frequency, fs)

autocorr, time_lags = compute_autocorrelation(signal_data, fs,max_lag=1)
mean_autocorr, ci_lower, ci_upper, time_lags = compute_segmented_autocorrelation(signal_data, fs, max_lag=0.5)
#%%

cutoff_frequency = 100  # in Hz
signal_data = butter_lowpass_filter(trace, cutoff_frequency, fs)
#filtered_signal=signal_data
# Parameters for segmentation
segment_length = int(1* fs)  # 0.5 seconds = 420 samples
num_segments = len(signal_data) // segment_length

# Initialize storage for autocorrelations
all_autocorrs = []

# Compute autocorrelation for each segment
for i in range(num_segments):
    segment = signal_data[i * segment_length:(i + 1) * segment_length]
    autocorr = np.correlate(segment, segment, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Take only positive lags
    autocorr /= autocorr[0]  # Normalize
    all_autocorrs.append(autocorr)

# Average the autocorrelations
average_autocorr = np.mean(all_autocorrs, axis=0)

# Compute time lags
lags = np.arange(len(average_autocorr))
time_lags = lags / fs  # Convert lags to seconds

# Plot the averaged autocorrelation
plt.figure(figsize=(10, 6))
#plt.plot(time_lags, average_autocorr)
plt.plot(time_lags, average_autocorr)
plt.title("Averaged Autocorrelation of the Filtered Signal")
plt.xlabel("Time Lag (s)")
plt.ylabel("Autocorrelation")
plt.grid()
plt.show()