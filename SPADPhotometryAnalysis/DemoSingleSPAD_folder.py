# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:50:05 2022
This is the main file to process the binary data recorded by the SPC imager. 

@author: Yifang
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import SPADreadBin
import os
#%%
# Sampling Frequency
fs   = 9938.4
'''Read binary files for two ROIs'''
#dpath="E:/SPAD/SPADData/20231027_GCamp8f_pyr_OECSync/2023_10_27_17_5_43_Cage_td_g30Iso20_recording1/"
#Green,Red=SPADreadBin.readMultipleBinfiles_twoROIs(dpath,9,xxrange_g=[105,210],yyrange_g=[125,235],xxrange_r=[105,210],yyrange_r=[25,105]) 
'''Read binary files for single ROI'''
dpath="D:/SPADdata/SNR_test_5uW/SPCimager/2024_4_1_17_0_35_025uW/"
'To read raw trace'
#TraceRaw=SPADreadBin.readMultipleBinfiles(dpath,1,xxRange=[100,250],yyRange=[40,190])
# Set the path to the parent folder
'''Show images'''
filename = os.path.join(dpath, "spc_data1.bin")
Bindata=SPADreadBin.SPADreadBin(filename,pyGUI=False)
#SPADreadBin.ShowImage(Bindata,dpath,xxRange=[100,250],yyRange=[40,190]) #squara wave
SPADreadBin.ShowImage(Bindata,dpath,xxRange=[10,170],yyRange=[70,230]) #SNR calculation
#%%
'''Time division mode with one ROI, GCamp and isosbestic'''
'''Read files'''
dpath="D:/SPADdata/SNR_test_5uW/SPCimager/2024_4_1_17_5_0_5uW/"

filename=Analysis.Set_filename (dpath, csv_filename="traceValue1.csv")
Trace_raw=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)
fig, ax = plt.subplots(figsize=(12, 2.5))
Analysis.plot_trace(Trace_raw,ax, fs=9938.4, label="Full raw data trace")
fig, ax = plt.subplots(figsize=(12, 2.5))
Analysis.plot_trace(Trace_raw[0:200],ax, fs=9938.4, label="Part raw data trace")

#%% Single ROIs
bin_window=10
Signal_bin=Analysis.get_bin_trace(Trace_raw,bin_window=bin_window)
csv_savename = os.path.join(dpath, '8_traceValue_bin1000Hz_5uW.csv')
np.savetxt(csv_savename, Signal_bin, delimiter=',')
bin_window=20
Signal_bin=Analysis.get_bin_trace(Trace_raw,bin_window=bin_window)
csv_savename = os.path.join(dpath, '8_traceValue_bin500Hz_5uW.csv')
np.savetxt(csv_savename, Signal_bin, delimiter=',')
#SNR=Analysis.calculate_SNR(Trace_raw[0:9000])
#%%
'''Demodulate using peak value'''
#Green,Red=Analysis.getTimeDivisionTrace (dpath, Trace_raw, sig_highlim=2300,sig_lowlim=1900,ref_highlim=400,ref_lowlim=50)
#%%
'Demodulate single ROI time division recodings'
Green,Red= Analysis.getTimeDivisionTrace_fromMask (dpath, Trace_raw, high_thd=12000,low_thd=6000)
#%%
'''Two ROIs for GEVI, one is green another is red for reference'''
# Two ROIs
#Green_raw,Red_raw = Analysis.ReadTwoROItrace (dpath,plot_xrange=200)
#%%
#Green, Red=Analysis.DemodTwoTraces (dpath,Green_raw, Red_raw,high_g=600,low_g=380,high_r=1200,low_r=800)
#%%
z_sig,smooth_sig,corrected_sig=Analysis.photometry_smooth_plot (Red,Green,
                                                                          sampling_rate=9938.4,smooth_win =500)

#%%
z_sig,smooth_sig,corrected_sig=Analysis.photometry_smooth_plot (Red,Green,
                                                                          sampling_rate=9938.4,smooth_win =20)
zscorefname = os.path.join(dpath, "Zscore_traceAll.csv")
np.savetxt(zscorefname, z_sig, delimiter=",")
#%%
signal1, signal2=Analysis.getICA (Red,Green)
z_sig,smooth_sig,corrected_sig=Analysis.photometry_smooth_plot (signal1,signal2,
                                                                          sampling_rate=9938.4,smooth_win =500)

#%% Wavelet analysis
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from waveletFunctions import wave_signif, wavelet

signal=z_sig
sst = Analysis.butter_filter(signal, btype='low', cutoff=20, fs=fs, order=5)
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

pad = 1  # pad the time series with zeroes (recommended)
dj = 0.25  # this will do 4 sub-octaves per octave
s0 = 25 * dt  # this says start at a scale of 6 months
j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
lag1 = 0.1  # lag-1 autocorrelation for red noise background
print("lag1 = ", lag1)
mother = 'MORLET'

# Wavelet transform:
wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
frequency=1/period

xlim = ([0,85])  # plotting range
fig, plt3 = plt.subplots(figsize=(15,5))

levels = [0, 4,20, 100, 200,300]
# *** or use 'contour'
CS = plt.contourf(time, frequency, power, len(levels))

plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Power Spectrum')
plt.xlim(xlim[:])
plt3.set_yscale('log', base=2, subs=None)
#plt.ylim([np.min(frequency), np.max(frequency)])
plt.ylim([0, 20])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt3.ticklabel_format(axis='y', style='plain')
#plt3.invert_yaxis()
# set up the size and location of the colorbar
position=fig.add_axes([0.2,0.01,0.4,0.02])
plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
plt.subplots_adjust(right=0.7, top=0.9)
