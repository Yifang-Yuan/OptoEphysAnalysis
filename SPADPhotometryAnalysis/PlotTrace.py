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
dpath='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/Day4/SyncRecording16'
csv_filename='Green_traceAll.csv'
filepath=Analysis.Set_filename (dpath, csv_filename)
#filepath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/smallROI_100Hznoise.csv'
Trace_raw=Analysis.getSignalTrace (filepath, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)
Trace_raw=Trace_raw
#%%
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(Trace_raw,ax, fs,label='840Hz')
Trace_raw=replace_outliers_with_nearest_avg(Trace_raw, window_size=25000, z_thresh=4)
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
z_score=replace_outliers_with_avg(z_score, threshold=4)

fig, ax = plt.subplots(figsize=(8,2))
plot_trace(z_score,ax, fs,label='cleaned')
#%%
fig, ax = plt.subplots(figsize=(4,2))
plot_trace(Trace_raw[1:],ax, fs,label='Trace')
#%%
bin_window=2
Signal_bin=Analysis.get_bin_trace(z_score,bin_window=bin_window,Fs=840)

bin_window=5
Signal_bin=Analysis.get_bin_trace(z_score,bin_window=bin_window,Fs=840)

#SNR=Analysis.calculate_SNR(Trace_raw[0:9000])
#ATLAS 840

#%% Wavelet analysis
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from waveletFunctions import wave_signif, wavelet

signal=Trace_raw[10*840:]
sst = Analysis.butter_filter(signal, btype='low', cutoff=300, fs=fs, order=4)
sst = Analysis.butter_filter(signal, btype='high', cutoff=20, fs=fs, order=4)
#sst = OE.butter_filter(signal, btype='high', cutoff=30, fs=Recording1.fs, order=5)
sst = sst - np.mean(sst)
variance = np.std(sst, ddof=1) ** 2
print("variance = ", variance)
# ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------
if 0:
    variance = 1.0
    sst = sst / np.std(sst, ddof=1)
n = len(sst)
dt = 1/fs
time = np.arange(len(sst)) * dt   # construct time array

pad = 1  # pad the time series with zeroes (recommended), do not need to change
dj = 0.25 # this will do 4 sub-octaves per octave, 
s0 = 2 * dt  # 
j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
lag1 = 0.1  # lag-1 autocorrelation for red noise background
print("lag1 = ", lag1)
mother = 'MORLET'

# Wavelet transform:
wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
frequency=1/period

xlim = ([0,20])  # plotting range
fig, plt3 = plt.subplots(figsize=(15,5))

levels = [0, 4,20, 100, 200,300]
# *** or use 'contour'
CS = plt.contourf(time, frequency, power, len(levels))

plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Power Spectrum')
plt.xlim(xlim[:])
#plt3.set_yscale('log', base=2, subs=None)
plt.ylim([np.min(frequency), np.max(frequency)])
plt.ylim([0, 300])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
#plt3.ticklabel_format(axis='y', style='plain')
#plt3.invert_yaxis()
# set up the size and location of the colorbar
position=fig.add_axes([0.2,0.01,0.4,0.02])
plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
plt.subplots_adjust(right=0.7, top=0.9)





