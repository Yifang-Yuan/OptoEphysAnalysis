# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:34:21 2025

@author: yifang
"""
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
Fs=10000
#%%
'For thesis--multifibre'

dpath= r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day2'
recordingName='SyncRecording4_forPSD'
LFP_channel='LFP_2'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 

# Recording1.pynacollada_label_theta (
#     LFP_channel,Low_thres=-1,High_thres=8,save=False,plot_theta=True)
# LFP_theta=Recording1.theta_part[LFP_channel]
# sig_theta=Recording1.theta_part['sig_raw']
# ref_theta=Recording1.theta_part['ref_raw']

LFP_theta=Recording1.Ephys_tracking_spad_aligned[LFP_channel]
sig_theta=Recording1.Ephys_tracking_spad_aligned['sig_raw']
ref_theta=Recording1.Ephys_tracking_spad_aligned['ref_raw']
z_theta=Recording1.Ephys_tracking_spad_aligned['zscore_raw']

# from scipy.stats import zscore
# sig_theta = zscore(sig_theta, ddof=0, nan_policy='omit')
# ref_theta = zscore(ref_theta, ddof=0, nan_policy='omit')

fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

# Plot optical signal
OpticalAnlaysis.PSD_plot(ref_theta, Fs, method="welch", color='red', xlim=[0.1, 40],
                         linewidth=2, linestyle='-', label='Ref', ax=ax1)
OpticalAnlaysis.PSD_plot(sig_theta, Fs, method="welch", color='green', xlim=[0.1, 40],
                         linewidth=2, linestyle='-', label='GEVI', ax=ax1)
OpticalAnlaysis.PSD_plot(z_theta, Fs, method="welch", color='blue', xlim=[0.1, 40],
                         linewidth=2, linestyle='-', label='CA3', ax=ax1)

ax1.set_ylabel('Optical PSD [dB/Hz]', color='green', fontsize=14)
ax1.tick_params(axis='y', labelcolor='green', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
#ax1.set_ylim(-40, -10)

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
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.93), fontsize=12)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()
#%%
'For thesis, plot multiROI PSD'
dpath= r'G:\2025_ATLAS_SPAD\MultiFibre\1887933_Jedi2P_Multi\Day2'
recordingName='SyncRecording4_forPSD'
LFP_channel='LFP_2'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 

LFP_theta=Recording1.Ephys_tracking_spad_aligned[LFP_channel]
LFP_theta2=Recording1.Ephys_tracking_spad_aligned['LFP_3']
sig_theta=Recording1.Ephys_tracking_spad_aligned['sig_raw']
ref_theta=Recording1.Ephys_tracking_spad_aligned['ref_raw']
z_theta=Recording1.Ephys_tracking_spad_aligned['zscore_raw']
from scipy.stats import zscore
sig_theta = zscore(sig_theta, ddof=0, nan_policy='omit')
ref_theta = zscore(ref_theta, ddof=0, nan_policy='omit')
z_theta = zscore(z_theta, ddof=0, nan_policy='omit')

fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

# Plot optical signal
OpticalAnlaysis.PSD_plot(ref_theta, Fs, method="welch", color='hotpink', xlim=[0.1, 45],
                         linewidth=2, linestyle='-', label='R-CA1', ax=ax1)
OpticalAnlaysis.PSD_plot(sig_theta, Fs, method="welch", color='limegreen', xlim=[0.1, 45],
                         linewidth=2, linestyle='-', label='L-CA1', ax=ax1)
OpticalAnlaysis.PSD_plot(z_theta, Fs, method="welch", color='royalblue', xlim=[0.1, 45],
                         linewidth=2, linestyle='-', label='L-CA3', ax=ax1)
ax1.set_ylabel('Optical PSD [dB/Hz]', color='green', fontsize=14)
ax1.tick_params(axis='y', labelcolor='green', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.set_ylim(-45, -10)

# Create second y-axis for LFP
ax2 = ax1.twinx()
OpticalAnlaysis.PSD_plot(LFP_theta, Fs, method="welch", color='red', xlim=[0.1, 45],
                         linewidth=2, linestyle='-', label='LFP2', ax=ax2)
OpticalAnlaysis.PSD_plot(LFP_theta2, Fs, method="welch", color='black', xlim=[0.1, 45],
                         linewidth=2, linestyle='-', label='LFP3', ax=ax2)
ax2.set_ylabel('LFP PSD [dB/Hz]', color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)
# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=12)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.92, 0.84), fontsize=12)
# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)
plt.show()
#%%
'For thesis, LFP PSD during theta, non theta'
Fs=10000
'For REM and non-REM sleep'
dpath= r'G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\Day4_Sleep'
recordingName='SyncRecording1'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_4'
LFP_theta1=Recording1.Ephys_tracking_spad_aligned[LFP_channel]

dpath= r'G:\2025_ATLAS_SPAD\PVCre\1887930_PV_mNeon_mCherry\Day4_Sleep\non_REM'
recordingName='SavednonREMTrials'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_4'
theta_part,non_theta_part=Recording1.pynacollada_label_theta (
    LFP_channel,Low_thres=0.2,High_thres=8,save=False,plot_theta=True)
LFP_theta2=Recording1.theta_part[LFP_channel]
#LFP_theta2=Recording1.Ephys_tracking_spad_aligned[LFP_channel]
#%%
fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size
# Plot optical signal

OpticalAnlaysis.PSD_plot(LFP_theta1, Fs, method="welch", color='black', xlim=[0.5, 80],
                         linewidth=2, linestyle='-', label='non-REM', ax=ax1)
OpticalAnlaysis.PSD_plot(LFP_theta2, Fs, method="welch", color='grey', xlim=[0.5, 80],
                         linewidth=2, linestyle='-', label='REM', ax=ax1)
ax1.set_ylabel('PSD [dB/Hz]', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)
ax1.set_ylim(25, 59)
# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=14)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.89), fontsize=14)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

# Plot optical signal
OpticalAnlaysis.PSD_plot(LFP_theta1, Fs, method="welch", color='black', xlim=[80, 200],
                         linewidth=2, linestyle='-', label='non-REM', ax=ax1)
OpticalAnlaysis.PSD_plot(LFP_theta2, Fs, method="welch", color='grey', xlim=[80, 200],
                         linewidth=2, linestyle='-', label='REM', ax=ax1)


ax1.set_ylabel('PSD [dB/Hz]', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)
ax1.set_ylim(8, 30)
# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=14)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(1, 0.89), fontsize=14)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()
#%%
'Used in Thesis do not delete---For moving and rest '
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- config ---
LFP_channel = "LFP_1"

# use ONE recording for both conditions (recommended)
dpath = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\Day3'
recordingName = 'SyncRecording4'
Recording1 = SyncOEpyPhotometrySession(
    dpath, recordingName, IsTracking=False, read_aligned_data_from_file=True,
    recordingMode='Atlas', indicator='GEVI'
)
Fs = Recording1.fs  # sampling rate

df = Recording1.Ephys_tracking_spad_aligned

def extract_1d(df, col, movement_value):
    """Return clean 1-D array for PSD (only the requested column, drop NaNs, contiguous)."""
    s = df.loc[df['movement'] == movement_value, col].astype(float)
    s = s.dropna()
    return np.ascontiguousarray(s.values)

x_move = extract_1d(df, LFP_channel, 'moving')
x_rest = extract_1d(df, LFP_channel, 'notmoving')

# x_move = extract_1d(df, 'zscore_raw', 'moving')
# x_rest = extract_1d(df, 'zscore_raw', 'notmoving')

# Optionally high/lowpass like you do elsewhere
# from scipy.signal import butter, filtfilt
# def butter_filter(x, btype, cutoff, fs, order=3):
#     nyq = fs/2; wn = np.asarray(cutoff)/nyq if np.iterable(cutoff) else cutoff/nyq
#     b,a = butter(order, wn, btype=btype); return filtfilt(b,a,x)
# x_move = butter_filter(butter_filter(x_move, 'low', 100, Fs, 5), 'high', 4, Fs, 3)
# x_rest = butter_filter(butter_filter(x_rest, 'low', 100, Fs, 5), 'high', 4, Fs, 3)

# Make nperseg safe
def safe_welch(x, fs, nperseg=16384, **kwargs):
    if x.size == 0:
        raise ValueError("Selected segment is empty after filtering/masking.")
    nps = min(nperseg, len(x))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nps, **kwargs)
    return f, Pxx

f1, P1 = safe_welch(x_move, Fs, window='hann')
f2, P2 = safe_welch(x_rest, Fs, window='hann')

# Plot
fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))
ax1.plot(f1, 10*np.log10(P1), color='black', lw=2, ls='-',  label='Moving')
ax1.plot(f2, 10*np.log10(P2), color='black', lw=2, ls='--', label='Rest')
ax1.set_xlim(0.5, 80)
#ax1.set_ylim(-42,-24)
ax1.set_ylim(25,50)
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_ylabel('PSD [dB/Hz]', fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax1.legend(loc='upper right', frameon=False, fontsize=14)
fig.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.12)
plt.show()

#%%
'For moving and rest '
dpath =r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\Day3'
recordingName='SyncRecording1'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
df=Recording1.Ephys_tracking_spad_aligned

#%%
fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

OpticalAnlaysis.PSD_plot(df[LFP_channel], Fs, method="welch", color='black', xlim=[0.5, 80],
                         linewidth=2, linestyle='-', label='LFP', ax=ax1)
ax1.set_ylabel('PSD [dB/Hz]', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)

# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=14)

# Create second y-axis for LFP
ax2 = ax1.twinx()
OpticalAnlaysis.PSD_plot(df['sig_raw'], Fs, method="welch", color='green', xlim=[0.5, 80],
                         linewidth=2, linestyle='-', label='GEVI', ax=ax2)
ax2.set_ylabel('GEVI PSD [dB/Hz]', color='green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='green', labelsize=14)
#ax2.set_ylim(-30, -5)
# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)

# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=12)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.99, 0.89), fontsize=12)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()


#%%
'''For Thesis:
Compare signal channel alone and with reference removed zscore trace'''

Fs=10000
dpath= r'G:\2025_ATLAS_SPAD\PyramidalWT\1881365_Jedi2p_mCherry\Day9_Cont'
recordingName='SyncRecording4'
Recording1=SyncOEpyPhotometrySession(
    dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
    recordingMode='Atlas',indicator='GEVI') 

sig_theta=Recording1.Ephys_tracking_spad_aligned['sig_raw']
zscore_theta=Recording1.Ephys_tracking_spad_aligned['zscore_raw']

sig_theta = zscore(sig_theta, ddof=0, nan_policy='omit')

# sig = (zscore_theta - min(zscore_theta))  
# ref_theta=sig / min(zscore_theta)
fig, ax1 = plt.subplots(1, 1, figsize=(4, 6))  # Fixed figure size

# Plot optical signal
OpticalAnlaysis.PSD_plot(sig_theta, Fs, method="welch", color='green', xlim=[0.1, 50],
                         linewidth=2, linestyle='-', label='Sig', ax=ax1)

ax1.set_ylabel('GEVI PSD [dB/Hz]', color='green', fontsize=14)
ax1.tick_params(axis='y', labelcolor='green', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
# ax1.set_ylim(20, 30)

# Create second y-axis for LFP
ax2 = ax1.twinx()
OpticalAnlaysis.PSD_plot(zscore_theta, Fs, method="welch", color='blue', xlim=[0.1, 50],
                         linewidth=2, linestyle='-', label='zscore', ax=ax2)
ax2.set_ylabel('Zscore PSD [dB/Hz]', color='blue', fontsize=14)
ax2.tick_params(axis='y', labelcolor='blue', labelsize=14)
#ax2.set_ylim(-30, -5)
# Set labels and title
ax1.set_xlabel('Frequency [Hz]', fontsize=14)
ax1.set_title('', fontsize=16)

# Legends
legend1 = ax1.legend(loc='upper right', frameon=False, fontsize=12)
legend2 = ax2.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.99, 0.89), fontsize=12)

# Manually adjust layout (instead of tight_layout)
fig.subplots_adjust(left=0.18, right=0.82, top=0.90, bottom=0.12)

plt.show()

