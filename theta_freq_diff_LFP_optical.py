# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 21:49:11 2025

@author: yifan
"""

import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def compare_PSDs_and_theta_peaks(
    LFP_data, 
    optical_data, 
    fs=10000, 
    method="welch", 
    xlim=[0, 100], 
    theta_band=(4, 12)
):
    """
    Calculate and plot PSDs for LFP and optical signals. 
    Find peak theta frequency for each, and compute difference.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3,6))

    def compute_psd(data, label, color):
        if method == "welch":
            f, Pxx_den = signal.welch(data, fs=fs, nperseg=8192)
        elif method == "periodogram":
            f, Pxx_den = signal.periodogram(data, fs=fs, nfft=8192, window='hann')
        else:
            raise ValueError("Unsupported method. Use 'welch' or 'periodogram'.")
        
        # Convert to dB
        Pxx_dB = 10 * np.log10(Pxx_den)
        
        # Plot
        idx_plot = (f >= xlim[0]) & (f <= xlim[1])
        ax.plot(f[idx_plot], Pxx_dB[idx_plot], label=label, linewidth=2, color=color)
        
        # Find peak frequency in theta band
        idx_theta = (f >= theta_band[0]) & (f <= theta_band[1])
        f_theta = f[idx_theta]
        Pxx_theta = Pxx_dB[idx_theta]
        peak_freq = f_theta[np.argmax(Pxx_theta)]
        
        return peak_freq

    # Compute and plot both PSDs
    peak_lfp = compute_psd(LFP_data, label='LFP PSD', color='black')
    peak_opt = compute_psd(optical_data, label='Optical PSD', color='tab:green')

    # Plot formatting
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB/Hz)")
    ax.set_xlim(xlim)
    ax.legend(frameon=False)
    ax.set_title("LFP vs Optical PSD")

    # Print or return frequency difference
    freq_diff = peak_lfp - peak_opt
    print(f"LFP theta peak: {peak_lfp:.2f} Hz")
    print(f"Optical theta peak: {peak_opt:.2f} Hz")
    print(f"Peak frequency difference: {freq_diff:.2f} Hz ({'LFP faster' if freq_diff > 0 else 'Optical faster'})")

    return fig, ax, peak_lfp, peak_opt, freq_diff



Fs=10000
dpath='G:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day3/'
recordingName='SyncRecording1'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
LFP_theta=Recording1.theta_part[LFP_channel]
optical_theta=Recording1.theta_part['zscore_raw']


LFP_theta = Recording1.theta_part[LFP_channel] / 1000  # Scale if needed
optical_theta = Recording1.theta_part['zscore_raw']

fig, ax, peak_lfp, peak_opt, freq_diff = compare_PSDs_and_theta_peaks(
    LFP_theta,
    optical_theta,
    fs=Fs,
    method="welch",
    xlim=[0, 100],
    theta_band=(4, 12)
)