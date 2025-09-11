# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 21:50:24 2025

@author: yifan
"""

import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import scipy.signal as signal

def bandpass_filter(signal_data, fs, lowcut=5, highcut=11, order=4):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, signal_data)

def calculate_theta_trough_index(df,angle_source, Fs=10000):
    # Detect local minima in theta phase: trough = 0 rad
    troughs = (
        (df[angle_source] < df[angle_source].shift(-1)) &
        (df[angle_source] < df[angle_source].shift(1)) &
        ((df[angle_source] < 0.2) | (df[angle_source] > (2 * np.pi - 0.2)))  # Around 0
    )
    trough_index = df.index[troughs]

    # Detect peaks: now at π radians (≈3.14)
    peaks = (
        (df[angle_source] > (np.pi - 0.1)) &
        (df[angle_source] < (np.pi + 0.1))
    )
    peak_index = df.index[peaks]

    return trough_index, peak_index

def filter_close_peaks(peak_indices, min_distance_samples):
    """Keep only peaks that are at least `min_distance_samples` apart."""
    if len(peak_indices) == 0:
        return np.array([])
    filtered = [peak_indices[0]]
    for idx in peak_indices[1:]:
        if idx - filtered[-1] >= min_distance_samples:
            filtered.append(idx)
    return np.array(filtered)

def plot_theta_traces(theta_part, LFP_channel, start_time, end_time, fs=10000):
    # Convert time to index
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    time_vector = np.arange(len(theta_part)) / fs

    # Subset and time axis
    segment = theta_part.iloc[start_idx:end_idx].copy()
    t = time_vector[start_idx:end_idx]

    # Apply theta bandpass filter
    filtered_LFP = bandpass_filter(segment[LFP_channel].values, fs)
    filtered_zscore = bandpass_filter(segment['zscore_raw'].values, fs)

    # Detect LFP theta peaks (phase near ±π)
    segment_peak_LFP = segment[
        (segment['LFP_theta_angle'] > 2.9) &
        (segment['LFP_theta_angle'] < 3.2)
    ].index

    # Identify optical theta peaks: same idea
    segment_peak_optical = segment[
        (segment['optical_theta_angle'] > 2.9) &
        (segment['optical_theta_angle'] < 3.2)
    ].index

    # Enforce 80 ms minimum distance between peaks
    min_dist_samples = int(0.08 * fs)  # 80 ms
    peak_idx_LFP = filter_close_peaks(segment_peak_LFP.to_numpy(), min_dist_samples)
    peak_idx_optical = filter_close_peaks(segment_peak_optical.to_numpy(), min_dist_samples)

    # Convert to time
    peak_t_LFP = peak_idx_LFP / fs
    peak_t_optical = peak_idx_optical / fs

    # Start plotting
    fig, axes = plt.subplots(4, 1, figsize=(22, 8), sharex=True)

    # Remove all subplot frames & set tick label size
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=18)  # enlarge tick labels

    # 1. Raw LFP trace
    axes[0].plot(t, segment[LFP_channel], color='black')
    axes[0].set_ylabel(LFP_channel, fontsize=20)
    axes[0].set_title('LFP trace', fontsize=20)

    # 2. zscore_raw
    zscore_lowpass = OE.smooth_signal(segment['zscore_raw'], fs, 100, window='flat')
    axes[1].plot(t, zscore_lowpass, color='green')
    axes[1].set_ylabel('zscore', fontsize=20)
    axes[1].set_title('GEVI Signal', fontsize=20)

    # 3. Filtered LFP + LFP peaks + zscore peaks as dots
    axes[2].plot(t, filtered_LFP, color='black', label='Filtered LFP')
    for pt in peak_t_LFP:
        if start_time <= pt <= end_time:
            axes[2].axvline(x=pt, color='red', linestyle='--', alpha=0.6)
    # Overlay optical theta peaks as dots
    dot_y = np.interp(peak_t_optical, t, filtered_LFP)
    axes[2].scatter(peak_t_optical, dot_y, color='green', marker='o', s=40, label='zscore peaks')
    axes[2].set_ylabel('LFP theta', fontsize=20)
    axes[2].set_title('LFP Theta band + Optical Peaks', fontsize=20)

    # 4. Filtered zscore + optical peaks
    axes[3].plot(t, filtered_zscore, color='green')
    for pt in peak_t_optical:
        if start_time <= pt <= end_time:
            axes[3].axvline(x=pt, color='red', linestyle='--', alpha=0.6)
    axes[3].set_ylabel('zscore theta', fontsize=20)
    axes[3].set_title('GEVI theta band', fontsize=20)
    axes[3].set_xlabel('Time (s)', fontsize=20)

    plt.tight_layout()
    plt.show()
    
def plot_zscore_peaks_on_LFP_phase(theta_part, fs=10000, wrap_cycles=2):
    """
    Plot a scatter plot of optical theta peaks (zscore) on the corresponding LFP theta phase.
    
    Args:
        theta_part (pd.DataFrame): DataFrame containing 'optical_theta_angle' and 'LFP_theta_angle' columns.
        fs (int): Sampling rate in Hz.
        wrap_cycles (int): How many theta cycles to wrap the LFP phase (1 for 0–360, 2 for 0–720).
    """
    # Detect optical theta peaks (where optical phase ~ π)
    peak_idx_optical = theta_part[
        (theta_part['optical_theta_angle'] > 2.9) & (theta_part['optical_theta_angle'] < 3.2)
    ].index.to_numpy()

    # Enforce 80 ms minimum distance
    min_dist_samples = int(0.08 * fs)
    peak_idx_optical = filter_close_peaks(peak_idx_optical, min_dist_samples)

    # Get LFP theta phase at each optical peak
    lfp_phases = theta_part.loc[peak_idx_optical, 'LFP_theta_angle'].values

    # Convert radians to degrees and wrap across cycles if needed
    lfp_degrees = np.rad2deg(lfp_phases) % 360
    if wrap_cycles == 2:
        # Track which cycle (0 or 1) each peak comes from, and offset accordingly
        cycle_index = np.arange(len(lfp_degrees)) % 2
        lfp_degrees += 360 * cycle_index

    # Plot
    plt.figure(figsize=(4, 6))
    plt.scatter(np.arange(len(lfp_degrees)), lfp_degrees, color='purple', alpha=0.7)
    plt.ylim(0, 360 * wrap_cycles)
    plt.xlabel('Optical Theta Peak Index', fontsize=14)
    plt.ylabel('LFP Theta Phase (degrees)', fontsize=14)
    plt.title('LFP Theta Phase at Optical Theta Peaks', fontsize=14)
    plt.yticks(np.arange(0, 360 * wrap_cycles + 1, 90))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()    
    
def run_theta_cycle_plot (dpath,LFP_channel,recordingName,savename,theta_low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    os.makedirs(save_path, exist_ok=True)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GEVI') 
    
    # Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_low_thres,High_thres=10,save=False,plot_theta=True)
    # theta_part=Recording1.theta_part
    
    theta_part=Recording1.Ephys_tracking_spad_aligned
    # theta_part =  Recording1.Ephys_tracking_spad_aligned[
    # Recording1.Ephys_tracking_spad_aligned['speed'] <2]
    
    theta_part=theta_part.reset_index(drop=True)
    theta_part['LFP_theta_angle']=OE.calculate_theta_phase_angle(theta_part[LFP_channel], theta_low=5, theta_high=12) 
    theta_part['optical_theta_angle']=OE.calculate_theta_phase_angle(theta_part['zscore_raw'], theta_low=5, theta_high=12) 
    trough_index_LFP,peak_index_LFP = calculate_theta_trough_index(theta_part,LFP_channel,Fs=10000)
    trough_index_optical,peak_index_optical = calculate_theta_trough_index(theta_part,'zscore_raw',Fs=10000)
    
    # for i in range(7):
    #     plot_theta_traces(theta_part, LFP_channel, start_time=i*3, end_time=i*3+3, fs=10000)
    plot_theta_traces(theta_part, LFP_channel, start_time=6, end_time=12, fs=10000)  
    plot_zscore_peaks_on_LFP_phase(theta_part, fs=10000, wrap_cycles=2)
    return theta_part

#%%
'This is to process a single or concatenated trial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
   
dpath=r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_Atlas\ALocomotion'
recordingName='SyncRecording7'


savename='ThetaSave_Move'
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_1'
theta_part=run_theta_cycle_plot (dpath,LFP_channel,recordingName,savename,theta_low_thres=-0.7) #-0.3
