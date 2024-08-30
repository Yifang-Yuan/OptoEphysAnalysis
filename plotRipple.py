# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:04:10 2024

@author: Yifang
"""
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
#%%
'''recordingMode: use py, Atlas, SPAD for different systems
'''
dpath='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/'
recordingName='SavedPostSleepTrials'
# dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day3/'
# recordingName='SavedPostSleepTrials'

Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
#%%
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_1'
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=0.5,High_thres=8,save=False,plot_theta=True)
#%%
'''RIPPLE DETECTION
For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,ep_start=10,ep_end=40,
                                                                          Low_thres=1,High_thres=10,plot_segment=False,
                                                                          plot_ripple_ep=False,excludeTheta=True)
#%%
'''RIPPLE CURATION'''
rip_timestamp=Recording1.ManualSelectRipple (lfp_channel=LFP_channel,ep_start=10,ep_end=40,
                                                                          Low_thres=1,High_thres=10,plot_segment=False,
                                                                          plot_ripple_ep=True,excludeTheta=True)
#%%
save_path = os.path.join(dpath, recordingName,LFP_channel+'_Class.pkl')
with open(save_path, "wb") as file:
    # Serialize and write the instance to the file
    pickle.dump(Recording1, file)
#%%
Recording1.Oscillation_triggered_Optical_transient_raw (mode='ripple',lfp_channel=LFP_channel, half_window=0.2,plot_single_trace=False,plotShade='CI')
#%%
ripple_triggered_zscore_values=-Recording1.ripple_triggered_zscore_values
ripple_triggered_LFP_values_1=Recording1.ripple_triggered_LFP_values_1
ripple_LFP_band=[]
for i in range(len(ripple_triggered_LFP_values_1)):
    ripple_LFP_band_i=OE.band_pass_filter(ripple_triggered_LFP_values_1[i], 130, 250, 10000)
    ripple_LFP_band.append(ripple_LFP_band_i)
ripple_LFP_band=np.vstack(ripple_LFP_band)

average_trace = np.mean(ripple_LFP_band, axis=0)
#%%
# Create a figure and subplots with shared x-axis
Fs=10000
time = np.arange(0, len(average_trace)) / Fs * 1000  # Time array in milliseconds

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot the average trace in the first subplot
axs[0].plot(average_trace)
axs[0].set_title('Average Trace')
axs[0].set_ylabel('Average Value')

# Plot the heatmap in the second subplot
heatmap_values=ripple_triggered_zscore_values
heatmap = sns.heatmap(heatmap_values, cmap='viridis', ax=axs[1], cbar_kws={"orientation": "horizontal"})
#axs[1].set_xticks(np.linspace(0, ripple_triggered_LFP_values_1.shape[1], 5))  # Set 5 evenly spaced ticks
#axs[1].set_xticklabels(np.linspace(time[0], time[-1], 5).astype(int))  # Set time labels

# Set the title and labels for the heatmap
axs[1].set_title('Heatmap of the Array')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Row')

# Move the color bar to below the figure
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.set_xlabel('Color Bar', fontsize=12)
cbar.ax.set_position([0.2, -0.2, 0.6, 0.02])  # Adjust the position

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Show the plot
plt.show()
#%%
# Create a figure and subplots with shared x-axis
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot the average trace in the first subplot
axs[0].plot(average_trace[1000:3000])
axs[0].set_title('Average Trace')
axs[0].set_ylabel('Average Value')

# Plot the heatmap in the second subplot
heatmap_values=ripple_triggered_zscore_values[:, 1000:3000]
heatmap = sns.heatmap(heatmap_values, cmap='viridis', ax=axs[1], cbar_kws={"orientation": "horizontal"})

# Set the title and labels for the heatmap
axs[1].set_title('Heatmap of the Array')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Row')

# Move the color bar to below the figure
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.set_xlabel('Color Bar', fontsize=12)
cbar.ax.set_position([0.2, -0.2, 0.6, 0.02])  # Adjust the position

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Show the plot
plt.show()