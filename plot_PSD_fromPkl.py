# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:34:21 2025

@author: yifang
"""
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
Fs=10000
dpath=r'G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\Day4'
recordingName='SyncRecording4'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_2'
# theta_part,non_theta_part=Recording1.pynacollada_label_theta (
#     LFP_channel,Low_thres=-0.3,High_thres=8,save=False,plot_theta=True)
# LFP_theta=Recording1.theta_part[LFP_channel]
# sig_theta=Recording1.theta_part['sig_raw']
# ref_theta=Recording1.theta_part['ref_raw']

LFP_theta=Recording1.Ephys_tracking_spad_aligned[LFP_channel]
sig_theta=Recording1.Ephys_tracking_spad_aligned['sig_raw']
#ref_theta=Recording1.Ephys_tracking_spad_aligned['ref_raw']
#%%
fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

# Plot optical signal
OpticalAnlaysis.PSD_plot(sig_theta, Fs, method="welch", color='green', xlim=[0.1, 40],
                         linewidth=2, linestyle='-', label='Sig', ax=ax1)
# OpticalAnlaysis.PSD_plot(ref_theta, Fs, method="welch", color='red', xlim=[0.1, 40],
#                          linewidth=2, linestyle='-', label='Ref', ax=ax1)
ax1.set_ylabel('Optical PSD [dB/Hz]', color='green', fontsize=14)
ax1.tick_params(axis='y', labelcolor='green', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

# Create second y-axis for LFP
ax2 = ax1.twinx()
OpticalAnlaysis.PSD_plot(LFP_theta, Fs, method="welch", color='black', xlim=[0.1, 40],
                         linewidth=2, linestyle='-', label='LFP', ax=ax2)
ax2.set_ylabel('LFP PSD [dB/Hz]', color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)

# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=12)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.89), fontsize=12)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()

#%%
'''For Thesis:
    Compare signal channel alone and with reference removed zscore trace'''
    
sig_theta=Recording1.Ephys_tracking_spad_aligned['sig_raw']
zscore_theta=Recording1.Ephys_tracking_spad_aligned['zscore_raw']

fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

# Plot optical signal
OpticalAnlaysis.PSD_plot(sig_theta, Fs, method="welch", color='green', xlim=[0.1, 50],
                         linewidth=2, linestyle='-', label='Sig', ax=ax1)

ax1.set_ylabel('GEVI PSD [dB/Hz]', color='green', fontsize=14)
ax1.tick_params(axis='y', labelcolor='green', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_ylim(-30, -5)

# Create second y-axis for LFP
ax2 = ax1.twinx()
OpticalAnlaysis.PSD_plot(zscore_theta, Fs, method="welch", color='blue', xlim=[0.1, 50],
                         linewidth=2, linestyle='-', label='zscore', ax=ax2)
ax2.set_ylabel('Zscore PSD [dB/Hz]', color='blue', fontsize=14)
ax2.tick_params(axis='y', labelcolor='blue', labelsize=14)
ax2.set_ylim(-30, -5)
# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)

# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=12)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.99, 0.89), fontsize=12)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()

