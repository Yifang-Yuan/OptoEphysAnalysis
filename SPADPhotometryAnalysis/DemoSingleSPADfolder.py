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
dpath='H:\ThesisData\HardwareTest\Atlas_linearity\AtlasData\Burst-RS-4200frames-840Hz_2024-11-04_16-05_2uW'
'To read raw trace'
#TraceRaw=SPADreadBin.readMultipleBinfiles(dpath,9,xxRange=[0,160],yyRange=[80,240])
# Set the path to the parent folder
'''Show images'''
filename = os.path.join(dpath, "spc_data1.bin")
Bindata=SPADreadBin.SPADreadBin(filename,pyGUI=False)
#%%
SPADreadBin.ShowImage(Bindata,dpath,xxRange=[135,270],yyRange=[70,205]) #squara wave
#SPADreadBin.ShowImage(Bindata,dpath,xxRange=[10,170],yyRange=[70,230]) #SNR calculation
#SPADreadBin.ShowImage(Bindata,dpath,xxRange=[10,170],yyRange=[70,230]) #sin wave
#%%
'''Time division mode with one ROI, GCamp and isosbestic'''
'''Read files'''
#dpath="F:/SPADdata/SineWave_10_500Hz/SPC/2024_4_12_16_45_47_200Hz/"

filename=Analysis.Set_filename (dpath, csv_filename="traceValue1.csv")
Trace_raw=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=10)
fig, ax = plt.subplots(figsize=(8, 2))
Analysis.plot_trace(Trace_raw[0:10000],ax, fs=9938.4, label="Full raw data trace")
fig, ax = plt.subplots(figsize=(8, 2))
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

