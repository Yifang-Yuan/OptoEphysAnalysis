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
dpath='F:/2025_ATLAS_SPAD/1887930_PV_mNeon_mCherry/Day5/'
recordingName='SyncRecording1'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
LFP_channel='LFP_4'


theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.3,High_thres=8,save=False,plot_theta=True)
# LFP_theta=Recording1.theta_part[LFP_channel]
# sig_theta=Recording1.theta_part['sig_raw']
# ref_theta=Recording1.theta_part['ref_raw']


LFP_theta=Recording1.Ephys_tracking_spad_aligned[LFP_channel]
sig_theta=Recording1.Ephys_tracking_spad_aligned['sig_raw']
#ref_theta=Recording1.Ephys_tracking_spad_aligned['ref_raw']

fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))

OpticalAnlaysis.PSD_plot(sig_theta, Fs, method="welch", color='green', xlim=[0.1, 100], linewidth=2, linestyle='-', label='GEVI', ax=ax1)
#OpticalAnlaysis.PSD_plot(ref_theta, Fs, method="welch", color='red', xlim=[0.1, 49],linewidth=2, linestyle='-', label='Ref', ax=ax1)
ax1.set_ylabel('Optical PSD [dB/Hz]', color='green')
ax1.tick_params(axis='y', labelcolor='green')
#ax1.set_ylim(2,20)
# Create a second y-axis for LFP
ax2 = ax1.twinx()
# Plot LFP PSD on the right y-axis
OpticalAnlaysis.PSD_plot(LFP_theta, Fs, method="welch", color='black', xlim=[0.1, 100], linewidth=2, linestyle='-', label='LFP  ', ax=ax2)
ax2.set_ylabel('LFP PSD [dB/Hz]', color='black')
ax2.tick_params(axis='y', labelcolor='black')
# Common x-axis and title
ax1.set_xlabel('Frequency [Hz]')
plt.title('Optical and LFP PSD')

legend1 = ax1.legend(loc='upper right', frameon=False)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.92))

plt.tight_layout()
plt.show()

