# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:13:41 2022
This file is to read and analyse pyPhotometry recordings. 
Also include a demostration of how the analysis is performed
It is modified from:
https://github.com/katemartian/Photometry_data_processing

@author: Yifang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
import os
# Folder with your files
# Modify it depending on where your file is located
folder ="G:/pyPhotometryJY/1746062_chemoInhibit_Rbp4_EC5a/"
# File name
file_name = '1746062inh-2024-05-31-163322.csv'
sampling_rate=130
#%%
'''Read csv file and calculate zscore of the fluorescent signal'''
raw_signal,raw_reference,Cam_Sync=fp.read_photometry_data (folder, file_name, readCamSync=True,plot=True)
plt.tight_layout()
#%%
'''Get zdFF directly'''
zdFF = fp.get_zdFF(raw_reference,raw_signal,smooth_win=2,remove=0,lambd=5e4,porder=1,itermax=50)
# fig = plt.figure(figsize=(16, 5))
# ax1 = fig.add_subplot(111)
# ax1 = fp.plotSingleTrace (ax1, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')
'''Save signal'''
# greenfname = os.path.join(folder, "Green_traceAll.csv")
# np.savetxt(greenfname, raw_signal, delimiter=",")
# redfname = os.path.join(folder, "Red_traceAll.csv")
# np.savetxt(redfname, raw_reference, delimiter=",")
# zscorefname = os.path.join(folder, "Zscore_traceAll.csv")
# np.savetxt(zscorefname, zdFF, delimiter=",")
# CamSyncfname = os.path.join(folder, "CamSync_photometry.csv")
# np.savetxt(CamSyncfname, Cam_Sync, fmt='%d',delimiter=",")
#%%
'''Define the segments you want to zoom in, in seconds'''
def get_part_trace(data,start_time,end_time,fs):
    start_time=start_time
    end_time=end_time
    sliced_data=data[fs*start_time:fs*end_time]
    return sliced_data

'''!!!Skip this part or comment these four lines if you dont want to cut your data'''
start_time=50
end_time=120

raw_signal=get_part_trace(raw_signal,start_time=start_time,end_time=end_time,fs=sampling_rate)
raw_reference=get_part_trace(raw_reference,start_time=start_time,end_time=end_time,fs=sampling_rate)
Cam_Sync=get_part_trace(Cam_Sync,start_time=start_time,end_time=end_time,fs=sampling_rate).to_numpy()
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1 = fp.plotSingleTrace (ax1, raw_signal, SamplingRate=sampling_rate,color='green',Label='Signal')
ax2 = fig.add_subplot(312)
ax2 = fp.plotSingleTrace (ax2, raw_reference, SamplingRate=sampling_rate,color='purple',Label='Reference')
ax3 = fig.add_subplot(313)
ax3 = fp.plotSingleTrace (ax3, Cam_Sync, SamplingRate=sampling_rate,color='orange',Label='Digital_Sync')
plt.tight_layout()
#%%
'''
You can get zdFF directly by calling the function fp.get_zdFF()
TO CHECK THE SIGNAL STEP BY STEP:
YOU CAN USE THE FOLLOWING CODES TO GET MORE PLOTS
These will give you plots for 
smoothed signal, corrected signal, normalised signal and the final zsocre
'''
'''Step 1, plot smoothed traces'''
smooth_win = 10
smooth_reference,smooth_signal,r_base,s_base = fp.photometry_smooth_plot (
    raw_reference,raw_signal,sampling_rate=sampling_rate, smooth_win = smooth_win)
#%%
'''Step 2, plot corrected traces, removing the baseline (detrend)'''
remove=0
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:])  

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(311)
ax1 = fp.plotSingleTrace (ax1, signal, SamplingRate=sampling_rate,color='blue',Label='corrected_signal')
ax2 = fig.add_subplot(312)
ax2 = fp.plotSingleTrace (ax2, reference, SamplingRate=sampling_rate,color='purple',Label='corrected_reference')
ax3 = fig.add_subplot(313)
ax3 = fp.plotSingleTrace (ax3, Cam_Sync, SamplingRate=sampling_rate,color='orange',Label='Digital_Sync')
plt.tight_layout()
#%%
'''Step 3, plot normalised traces'''
z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
ax2 = fig.add_subplot(212)
ax2 = fp.plotSingleTrace (ax2, z_reference, SamplingRate=sampling_rate,color='purple',Label='normalised_reference')

#%%
'''Step 4, plot fitted reference trace and signal'''
from sklearn.linear_model import Lasso
lin = Lasso(alpha=0.001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
n = len(z_reference)
'''Need to change to numpy if previous smooth window is 1'''
# z_signal=z_signal.to_numpy()
# z_reference=z_reference.to_numpy()
lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(111)
ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
ax1 = fp.plotSingleTrace (ax1, z_reference_fitted, SamplingRate=sampling_rate,color='purple',Label='fitted_reference')
#%%
'''Step 5, plot zscore'''
zdFF = (z_signal - z_reference_fitted)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax1 = fp.plotSingleTrace (ax1, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')
# ax2 = fig.add_subplot(212)
# ax2 = fp.plotSingleTrace (ax2, Cam_Sync, SamplingRate=sampling_rate,color='orange',Label='Digital_Sync')
# plt.tight_layout()
#%%
zdFF=pd.Series(zdFF)
Cam_Sync=pd.Series(Cam_Sync)
indices = Cam_Sync[Cam_Sync == 1].index

'''plot optical triggered average signal'''
segments = []
half_window_seconds=0.5
for idx in indices:
    start_idx = max(0, idx - int(half_window_seconds*sampling_rate))
    end_idx = min(len(zdFF), idx + int(half_window_seconds*sampling_rate) + 1)
    segment_data = zdFF.iloc[start_idx:end_idx]
    if start_idx>0 and end_idx<len(zdFF):
        segments.append(segment_data)
        
segments_array = np.vstack(segments)

import matplotlib.pyplot as plt

# Assuming you have already calculated the mean and std for your segments_array
mean_values = np.mean(segments_array, axis=0)
std_values = np.std(segments_array, axis=0)

# Calculate the midpoint index
midpoint_index = len(mean_values) // 2

# Create an x-axis centered around the midpoint
x_values = np.arange(-midpoint_index, midpoint_index + 1) / sampling_rate  # Convert to seconds

# Plot the mean as a solid line
plt.plot(x_values, mean_values, label='Mean', color='black')

# Plot the standard deviation as shaded areas
plt.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=0.3, color='black', label='Std')

# Add a vertical line at x=0
plt.axvline(x=0, color='red', linestyle='--', label='Time Zero')

# Customize the plot (add labels, title, etc.)
plt.xlabel('Time (seconds)')
plt.ylabel('zscore')
plt.title('Mean and Standard Deviation')
plt.legend()

# Remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.show()