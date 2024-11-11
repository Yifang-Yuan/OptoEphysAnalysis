# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:35:10 2024

@author: Yifang
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import SPADreadBin
from SPADPhotometryAnalysis import photometry_functions as fp
import os
from scipy import signal


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

def notchfilter (data,f0=100,bw=10,fs=840):
    # Bandwidth of the notch filter (in Hz)   
    Q = f0/bw # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    for _ in range(4):
        data = signal.filtfilt(b, a, data)
    return data

'''Read binary files for single ROI'''
fs=840
dpath='E:/ATLAS_SPAD/1825504_SimCre_GCamp8f_taper/Day2/SyncRecording4'
# fs=1000
# dpath='G:/YY/New/1765508_Jedi2p_CompareSystem/Day2_pyPhotometry/SyncRecording4'
csv_filename='Zscore_traceAll.csv'
filepath=Analysis.Set_filename (dpath, csv_filename)
#filepath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/smallROI_100Hznoise.csv'
Trace_raw=Analysis.getSignalTrace (filepath, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)

Trace_raw=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
#Trace_raw=Analysis.butter_filter(Trace_raw, btype='low', cutoff=80, fs=840, order=5)


Trace_raw = pd.Series(Trace_raw)

trace_smooth=fp.smooth_signal(Trace_raw[0:8400],4,'flat')


fig, ax = plt.subplots(figsize=(8,2))
plot_trace(trace_smooth,ax, fs,label='840Hz',color='tab:blue')

trace_smooth=fp.smooth_signal(Trace_raw[8400:16800],4,'flat')

#trace_smooth=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(trace_smooth,ax, fs,label='840Hz')

trace_smooth=fp.smooth_signal(Trace_raw[16800:],4,'flat')

#trace_smooth=notchfilter (Trace_raw,f0=100,bw=10,fs=840)
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(trace_smooth,ax, fs,label='840Hz')

