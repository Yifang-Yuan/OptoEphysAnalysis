# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:06:13 2025
@author: yifan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import OpenEphysTools as OE

def PSD_plot(data, fs, method="welch", color='tab:blue', xlim=[0,100], linewidth=1, linestyle='-',label='PSD',ax=None,resolution=4096):
    '''Three methods to plot PSD: welch, periodogram, plotlib based on a given ax'''
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure and axis if none provided
    else:
        fig = ax.figure  # Reference the figure from the provided ax
       
    if method == "welch":
        #f, Pxx_den = signal.welch(data, fs=fs, nperseg=resolution,detrend=False)
        f, Pxx_den = signal.welch(data, fs=fs, nperseg=resolution)
    elif method == "periodogram":
        f, Pxx_den = signal.periodogram(data, fs=fs,window="hann",nfft=resolution,detrend=False)
    # Convert to dB/Hz
    Pxx_den_dB = 10 * np.log10(Pxx_den)
    
    # Filter the data for the x-axis range [xlim[0], xlim[1]] Hz
    idx = (f >= xlim[0]) & (f <= xlim[1])
    f_filtered = f[idx]
    Pxx_den_dB_filtered = Pxx_den_dB[idx]
    # Plot the filtered data on the given ax with specified linestyle
    ax.plot(f_filtered, Pxx_den_dB_filtered, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    #ax.plot(f, Pxx_den_dB, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    ax.set_xlim(xlim)  # Limit x-axis to the specified range
 
    ax.set_ylim([np.min(Pxx_den_dB_filtered) - 1, np.max(Pxx_den_dB_filtered) + 1])
    
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [dB/Hz]')

    legend = ax.legend(fontsize=12, markerscale=1.5)
    legend.get_frame().set_facecolor('none')  # Remove the background color
    legend.get_frame().set_edgecolor('none')  # Remove the border
        
    return fig, ax
#%%
#dpath = "D:/2025_ATLAS_SPAD/1836686_PV_mNeon_F/Day7/"
# dpath = "D:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day2/"
# dpath = "D:/2024_OEC_Atlas_main/1765010_PVGCaMP8f_Atlas/Day1/"
#dpath = "D:/2025_ATLAS_SPAD/1844609_WT_Jedi2p/Day4/"
#dpath = "D:/2025_ATLAS_SPAD/1844608_WT_mNeon/Day5/"
dpath = 'D:/2025_ATLAS_SPAD/1851547_WT_mNeon/Day3/'
dpath = "D:/2025_ATLAS_SPAD/1842516_PV_Jedi2p/Day1/"


recordingNum=7
LFP_channel='LFP_3'
sync_recording_str = f"SyncRecording{recordingNum}"


recording_path=file_path_optical=os.path.join(dpath,sync_recording_str)
file_path_optical=os.path.join(dpath,sync_recording_str, "Green_traceAll.csv")
data = pd.read_csv(file_path_optical, header=None)  # Adjust if there's a header
fs=841.68
# Convert to NumPy array (assume signal is in the first column)
optical_data = data.iloc[:, 0].values

'To read the ephys data from pkl'
# file_path_ephys=os.path.join(recording_path, "open_ephys_read_pd.pkl")
# EphysData = pd.read_pickle(file_path_ephys)
'To decode the ephys data from binary data'

ephys_folder = os.path.join(dpath, "Ephys")
subfolders = [f for f in os.listdir(ephys_folder) if os.path.isdir(os.path.join(ephys_folder, f))]
if len(subfolders) == 1:
    Ephys_folder_path = os.path.join(ephys_folder, subfolders[0])
    print("Full path:", Ephys_folder_path)
    
#DEFINE RecordingNum and LFP channel here.
EphysData=OE.readEphysChannel (Ephys_folder_path,recordingNum=recordingNum-1,Fs=30000) 

LFP_data = EphysData[LFP_channel].values
fs_ephys=30000




fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))

# Plot optical PSD on the left y-axis
PSD_plot(optical_data, fs, method="welch", color='green', xlim=[0, 50], linewidth=2, linestyle='-', label='GEVI', ax=ax1, resolution=2048)
ax1.set_ylabel('Optical PSD [dB/Hz]', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Create a second y-axis for LFP
ax2 = ax1.twinx()

# Plot LFP PSD on the right y-axis
PSD_plot(LFP_data, fs_ephys, method="welch", color='black', xlim=[0, 50], linewidth=2, linestyle='-', label='LFP  ', ax=ax2, resolution=65536)
ax2.set_ylabel('LFP PSD [dB/Hz]', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Common x-axis and title
ax1.set_xlabel('Frequency [Hz]')
plt.title('Optical and LFP PSD')

legend1 = ax1.legend(loc='upper right', frameon=False)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.95))

plt.tight_layout()
plt.show()