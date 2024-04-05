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
import os

def plot_trace(trace,ax, fs=9938.4, label="trace",color='tab:blue'):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    ax.plot(taxis,trace,linewidth=0.5,color=color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax

# Sampling Frequency
'''Read binary files for two ROIs'''
#dpath="E:/SPAD/SPADData/20231027_GCamp8f_pyr_OECSync/2023_10_27_17_5_43_Cage_td_g30Iso20_recording1/"
#Green,Red=SPADreadBin.readMultipleBinfiles_twoROIs(dpath,9,xxrange_g=[105,210],yyrange_g=[125,235],xxrange_r=[105,210],yyrange_r=[25,105]) 
'''Read binary files for single ROI'''
dpath="D:/SPADdata/SqaureWave_10_500Hz/Atlas/"
csv_filename='1_ROI_trace_10Hz.csv'

filepath=Analysis.Set_filename (dpath, csv_filename)

Trace_raw=Analysis.getSignalTrace (filepath, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)
fig, ax = plt.subplots(figsize=(8,2))
plot_trace(Trace_raw,ax, fs=1017)











