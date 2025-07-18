# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 22:01:38 2025

@author: yifan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from SPADPhotometryAnalysis import AtlasDecode
def notchfilter (data,f0=50,bw=10,fs=30000):
    # Bandwidth of the notch filter (in Hz)   
    Q = f0/bw # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    data=signal.filtfilt(b, a, data)
    return data
#%%
'''pyPhotometry imaging'''
# Folder with your files
# Modify it depending on where your file is located
folder ="G:/2025_ATLAS_SPAD/ChirpSignal/pyPhotometry/"
# File name
file_name = '30mVsweep-2025-07-05-135626.csv'
sampling_rate=1000
'''Read csv file and calculate zscore of the fluorescent signal'''
PhotometryData = pd.read_csv(folder+file_name,index_col=False) 
raw_signal = PhotometryData['Analog1'][1:]
#%%
'''ATLAS imaging'''

sampling_rate=1682.92
dpath=r'G:\2025_ATLAS_SPAD\ChirpSignal\ATLAS\30mV'
#dpath='F:/2025_ATLAS_SPAD/1887933_Jedi2P_Multi/Day1/Test/'
#hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'

pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (dpath,
                                                                                   hotpixel_path,photoncount_thre=180000)
AtlasDecode.show_image_with_pixel_array(avg_pixel_array,showPixel_label=True)
center_x, center_y,radius=AtlasDecode.find_circle_mask(avg_pixel_array,radius=10,threh=0.2)

#%%
sampling_rate=1682.92
trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,
                                                         avg_pixel_array,hotpixel_path,center_x, center_y,radius,
                                                         fs=1682.92,snr_thresh=30)
#%%
raw_signal=notchfilter (trace_raw,f0=100,bw=10,fs=1682.92)
raw_signal=notchfilter (raw_signal,f0=200,bw=5,fs=1682.92)
raw_signal = pd.Series(raw_signal)
#%%
#raw_signal = pd.Series(trace_raw)
#%%
# Generate time vector
time_vector = raw_signal.index / sampling_rate

# Plot the full raw signal
plt.figure(figsize=(10, 4))
plt.plot(time_vector, raw_signal, color='blue', linewidth=1)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Raw Signal', fontsize=12)
plt.title('Full Raw Signal', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.signal import spectrogram
# Compute spectrogram
frequencies, times, Sxx = spectrogram(
    raw_signal,
    fs=sampling_rate,
    nperseg=256,
    noverlap=128,
    scaling='density',
    mode='psd'
)
# Convert times to absolute time in seconds
absolute_times = times

# Plot spectrogram as heatmap
plt.figure(figsize=(10, 4))
plt.pcolormesh(absolute_times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
plt.colorbar(label='Power [dB]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Power Spectrogram of Cut Signal')
plt.ylim(0, 250)  # ← Fixed here
plt.tight_layout()
plt.show()
#%%
# Ask user for start and end times
start_time = float(input("Enter start time (in seconds): "))
end_time = float(input("Enter end time (in seconds): "))
# Convert time to index
start_idx = int(start_time * sampling_rate)
end_idx = int(end_time * sampling_rate)
# Slice the signal
cut_signal1 = raw_signal[start_idx:end_idx]
cut_time1 = time_vector[start_idx:end_idx]

start_time = float(input("Enter start time (in seconds): "))
end_time = float(input("Enter end time (in seconds): "))
# Convert time to index
start_idx = int(start_time * sampling_rate)
end_idx = int(end_time * sampling_rate)
# Slice the signal
cut_signal2 = raw_signal[start_idx:end_idx]
cut_time2 = time_vector[start_idx:end_idx]


# Concatenate signals
cut_signal = np.concatenate((cut_signal2, cut_signal1))
# Concatenate time vectors (optional, only if you want a continuous time axis)
total_length = len(cut_signal)
cut_time = np.arange(total_length) / sampling_rate
# Plot the cut signal
plt.figure(figsize=(20, 4))
plt.plot(cut_time, cut_signal, color='green', linewidth=1.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Raw Signal', fontsize=12)
plt.title(f'Signal from {start_time}s to {end_time}s', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
from scipy.signal import spectrogram
# Compute spectrogram
frequencies, times, Sxx = spectrogram(
    cut_signal,
    fs=sampling_rate,
    nperseg=256,
    noverlap=128,
    scaling='density',
    mode='psd'
)

# Convert times to absolute time in seconds
absolute_times = times

# Plot spectrogram as heatmap
plt.figure(figsize=(10, 4))
plt.pcolormesh(absolute_times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')

# Add colorbar with larger label font
cbar = plt.colorbar()
cbar.set_label('Power [dB]', fontsize=18)
cbar.ax.tick_params(labelsize=16)

# Axis labels with larger font
plt.ylabel('Frequency [Hz]', fontsize=18)
plt.xlabel('Time [s]', fontsize=16)

# Title with larger font
plt.title('Power Spectrogram', fontsize=18)

# Tick labels larger
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.ylim(0, 250)
plt.tight_layout()
plt.show()