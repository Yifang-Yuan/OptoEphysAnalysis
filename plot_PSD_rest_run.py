# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:41:42 2024

@author: Yifang
"""

import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def PSD_plot(data, fs=9938.4, method="welch", color='tab:blue', xlim=[0,100], linewidth=1, linestyle='-',label='PSD',ax=None):
    '''Three methods to plot PSD: welch, periodogram, plotlib based on a given ax'''
    if ax is None:
        fig, ax = plt.subplots()  # Create a new figure and axis if none provided
    else:
        fig = ax.figure  # Reference the figure from the provided ax
    
    if method == "welch":
        f, Pxx_den = signal.welch(data, fs=fs, nperseg=8192)
    elif method == "periodogram":
        f, Pxx_den = signal.periodogram(data, fs=fs, nfft=8192, window='hann')
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
Fs=10000
dpath='F:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day3/'
recordingName='SyncRecording3'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
LFP_theta=Recording1.theta_part[LFP_channel]
optical_theta=Recording1.theta_part['zscore_raw']

dpath='F:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day3/'
recordingName='SyncRecording11'
Recording2=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
theta_part,non_theta_part=Recording2.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
LFP_nontheta=Recording2.non_theta_part[LFP_channel]
optical_nontheta=Recording2.non_theta_part['zscore_raw']

# dpath='F:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day3/'
# recordingName='SavedPostSleepTrials'
# Recording3=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording3.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
# LFP_sleep=Recording3.non_theta_part[LFP_channel]
#optical_sleep=Recording3.non_theta_part['zscore_raw']



fig, ax = plt.subplots(1, 1, figsize=(3, 6))
#OpticalAnlaysis.PSD_plot (LFP_sleep/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle=':',label='LFP-sleep',ax=ax)
PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)


fig, ax = plt.subplots(1, 1, figsize=(3, 6))
PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='GEVI-rest',ax=ax)
PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='GEVI-move',ax=ax)
#OpticalAnlaysis.PSD_plot (optical_sleep,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle=':',label='GEVI-sleep',ax=ax)

#%%
'plot PSD----iGlu'

Fs=10000
# dpath='F:/2024_OEC_Atlas/1765507_iGlu_Atlas/Day3/'
# recordingName='SavedOpenFieldTrials'
# Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
# LFP_theta=Recording1.theta_part[LFP_channel]

# dpath='F:/2024_OEC_Atlas/1765507_iGlu_Atlas/Day1/'
# recordingName='SavedPreAwakeTrials'
# Recording2=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording2.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
# LFP_nontheta=Recording2.non_theta_part[LFP_channel]

# dpath='F:/2024_OEC_Atlas/1765507_iGlu_Atlas/Day3/'
# recordingName='SavedPostSleepTrials'
# Recording3=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording3.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
# LFP_sleep=Recording3.non_theta_part[LFP_channel]

# fig, ax = plt.subplots(1, 1, figsize=(3, 6))

# OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
# OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)
# OpticalAnlaysis.PSD_plot (LFP_sleep/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle=':',label='LFP-sleep',ax=ax)

# optical_theta=Recording1.theta_part['zscore_raw']
# optical_nontheta=Recording2.non_theta_part['zscore_raw']
# optical_sleep=Recording3.non_theta_part['zscore_raw']
# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# OpticalAnlaysis.PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='iGlu-rest',ax=ax)
# OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='iGlu-move',ax=ax)
# OpticalAnlaysis.PSD_plot (optical_sleep,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle=':',label='iGlu-sleep',ax=ax)
#%%
'plot PSD----GECI-PVIN'
# Fs=10000
# dpath='F:/2024_OEC_Atlas/1765010_PVGCaMP8f_Atlas/Day1/'
# recordingName='SavedOpenFieldTrials'
# Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
# LFP_theta=Recording1.theta_part[LFP_channel]

# dpath='F:/2024_OEC_Atlas/1765010_PVGCaMP8f_Atlas/Day1/'
# recordingName='SavedPreAwakeTrials'
# Recording2=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording2.pynacollada_label_theta (LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
# LFP_nontheta=Recording2.non_theta_part[LFP_channel]

# dpath='F:/2024_OEC_Atlas/1765010_PVGCaMP8f_Atlas/Day1/'
# recordingName='SavedPostSleepTrials'
# Recording3=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GECI') 
# LFP_channel='LFP_1'
# theta_part,non_theta_part=Recording3.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
# LFP_sleep=Recording3.non_theta_part[LFP_channel]

# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# OpticalAnlaysis.PSD_plot (LFP_sleep/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle=':',label='LFP-sleep',ax=ax)
# OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
# OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)


# optical_theta=Recording1.theta_part['zscore_raw']
# optical_nontheta=Recording2.non_theta_part['zscore_raw']
# optical_sleep=Recording3.non_theta_part['zscore_raw']
# fig, ax = plt.subplots(1, 1, figsize=(3, 6))
# OpticalAnlaysis.PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='PV-GECI-rest',ax=ax)
# OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='PV-GECI-move',ax=ax)
# OpticalAnlaysis.PSD_plot (optical_sleep,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle=':',label='PV-GECI-sleep',ax=ax)

#%%

#%%
Fs=10000
dpath='E:/ATLAS_SPAD/1820061_PVcre/Day4/SNR_methods/'
recordingName='SavedMovingTrials'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
LFP_theta=Recording1.theta_part[LFP_channel]

dpath='E:/ATLAS_SPAD/1820061_PVcre/Day4/SNR_methods/'
recordingName='SavedRestTrials'
Recording2=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
theta_part,non_theta_part=Recording2.pynacollada_label_theta (LFP_channel,Low_thres=0,High_thres=8,save=False,plot_theta=True)
LFP_nontheta=Recording2.non_theta_part[LFP_channel]

optical_theta=Recording1.theta_part['zscore_raw']
optical_nontheta=Recording2.non_theta_part['zscore_raw']
#%%
fig, ax = plt.subplots(1, 1, figsize=(3, 6))
OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='--',label='LFP-rest',ax=ax)
OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,100],linewidth=2,linestyle='-',label='LFP-move',ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
OpticalAnlaysis.PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='--',label='GEVI-rest',ax=ax)
OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='GEVI-move',ax=ax)
