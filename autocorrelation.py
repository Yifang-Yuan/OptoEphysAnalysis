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

# Load the signal
dpath = 'E:/2025_ATLAS_SPAD/1842515_PV_mNeon/Day2Theta/SyncRecording1/'
filename = "Zscore_traceAll.csv"
filepath = os.path.join(dpath, filename)
trace = pd.read_csv(filepath, header=None).squeeze("columns")

# Select a portion of the signal
frame_rate = 840  # in Hz
signal_data = trace[840 * 10:840 * 20]  # Select 10-second segment

# Filter the signal
cutoff_frequency = 50  # in Hz
filtered_signal = butter_lowpass_filter(signal_data, cutoff_frequency, frame_rate)

# Parameters for segmentation
segment_length = int(0.2 * frame_rate)  # 0.5 seconds = 420 samples
num_segments = len(filtered_signal) // segment_length

# Initialize storage for autocorrelations
all_autocorrs = []

# Compute autocorrelation for each segment
for i in range(num_segments):
    segment = filtered_signal[i * segment_length:(i + 1) * segment_length]
    autocorr = np.correlate(segment, segment, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # Take only positive lags
    autocorr /= autocorr[0]  # Normalize
    all_autocorrs.append(autocorr)

# Average the autocorrelations
average_autocorr = np.mean(all_autocorrs, axis=0)

# Compute time lags
lags = np.arange(len(average_autocorr))
time_lags = lags / frame_rate  # Convert lags to seconds

# Plot the averaged autocorrelation
plt.figure(figsize=(10, 6))
#plt.plot(time_lags, average_autocorr)
plt.plot(time_lags[10:], average_autocorr[10:])
plt.title("Averaged Autocorrelation of the Filtered Signal")
plt.xlabel("Time Lag (s)")
plt.ylabel("Autocorrelation")
plt.grid()
plt.show()