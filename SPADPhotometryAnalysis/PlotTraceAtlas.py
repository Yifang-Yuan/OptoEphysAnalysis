# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:12:54 2024

@author: Yifang
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import SPADreadBin
from SPADPhotometryAnalysis import photometry_functions as fp
import os

def plot_trace(trace,ax, fs=9938.4, label="trace",color='tab:blue'):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    ax.plot(taxis,trace,linewidth=0.5,label=label,color=color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.xaxis.set_visible(False)  # Hide x-axis
    #ax.yaxis.set_visible(False)  # Hide x-axis
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax

from scipy.ndimage import uniform_filter1d

def replace_outliers_with_nearest_avg(data, window_size=25000, z_thresh=3):
    # Calculate the mean and standard deviation of the moving window
    mean = uniform_filter1d(data, window_size, mode='reflect')
    std = uniform_filter1d(data**2, window_size, mode='reflect')
    std = np.sqrt(std - mean**2)

    # Identify the outliers
    outliers = (np.abs(data - mean) > z_thresh * std)

    # Replace outliers with the average of their nearest non-outlier neighbors
    for i in np.where(outliers)[0]:
        j = i - 1
        while j >= 0 and outliers[j]:
            j -= 1
        k = i + 1
        while k < len(data) and outliers[k]:
            k += 1
        if j >= 0 and k < len(data):
            data[i] = (data[j] + data[k]) / 2
        elif j >= 0:
            data[i] = data[j]
        elif k < len(data):
            data[i] = data[k]

    return data


def replace_outliers_with_avg(data, threshold):
    # Identify the outliers
    outliers = np.abs(data) > threshold

    # Get the indices of the outliers
    outlier_indices = np.where(outliers)[0]

    # Iterate over the outlier indices
    for index in outlier_indices:
        # Find the nearest non-outlier value
        left = index
        while left > 0 and outliers[left - 1]:
            left -= 1

        right = index
        while right < len(data) - 1 and outliers[right + 1]:
            right += 1

        # If the outlier is not at the boundaries of the array, replace it with the average of its neighbors
        if left != 0 and right != len(data) - 1:
            data[index] = (data[left - 1] + data[right + 1]) / 2
        # If the outlier is at the boundaries of the array, replace it with the nearest non-outlier value
        else:
            data[index] = data[left - 1] if left != 0 else data[right + 1]

    return data


#%%
# Sampling Frequency
'''Read binary files for single ROI'''
fs=840
dpath='G:/GEVItest/1804114_JediCA1_2/'
# fs=1000
# dpath='G:/YY/New/1765508_Jedi2p_CompareSystem/Day2_pyPhotometry/SyncRecording4'
csv_filename='Green_traceAll.csv'
filepath=Analysis.Set_filename (dpath, csv_filename)
#filepath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/smallROI_100Hznoise.csv'
Trace_raw=Analysis.getSignalTrace (filepath, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)

#%%
fs=840
Trace_raw=Trace_raw[fs*10:fs*12]
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(Trace_raw,ax, fs,label='840Hz')
#%%
lambd = 10e3 # Adjust lambda to get the best fit
porder = 1
itermax = 15
sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
signal = (Trace_raw - sig_base)  
z_score=(signal - np.median(signal)) / np.std(signal)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(z_score,ax, fs,label='z_score')

#%%
bin_window=2
Signal_bin=Analysis.get_bin_trace(Trace_raw,bin_window=bin_window,Fs=840)

bin_window=5
Signal_bin=Analysis.get_bin_trace(Trace_raw,bin_window=bin_window,Fs=840)

#SNR=Analysis.calculate_SNR(Trace_raw[0:9000])
#ATLAS 840

#%% Wavelet analysis
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from waveletFunctions import wave_signif, wavelet
import OpenEphysTools as OE

signal=Trace_raw
signal_smooth= OE.butter_filter(signal, btype='high', cutoff=4, fs=fs, order=2)
signal_smooth= OE.butter_filter(signal_smooth, btype='low', cutoff=20, fs=fs, order=3)
'scale also change the frequency range you can get'
sst,frequency,power,global_ws=OE.Calculate_wavelet(signal_smooth,lowpassCutoff=100,Fs=fs,scale=5)

fig, ax = plt.subplots(figsize=(8,2))
OE.plot_wavelet(ax,sst,frequency,power,Fs=fs,colorBar=False,logbase=True)

