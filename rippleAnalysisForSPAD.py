# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:49:06 2023

@author: Yifang
"""
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pylab as plt
import pynapple as nap
import pynacollada as pyna
from scipy.signal import filtfilt
import OpenEphysTools as OE
import seaborn as sns


dpath='E:/SPAD/SPADData/20230818_SPADOEC/SyncRecording0_1681633_g8m/'
Spad_fs=9938.4


sig_csv_filename=os.path.join(dpath, "Green_traceAll.csv")
ref_csv_filename=os.path.join(dpath, "Red_traceAll.csv")
zscore_csv_filename=os.path.join(dpath, "Zscore_traceAll.csv")
sig_data = np.genfromtxt(sig_csv_filename, delimiter=',')
ref_data = np.genfromtxt(ref_csv_filename, delimiter=',')
zscore_data = np.genfromtxt(zscore_csv_filename, delimiter=',')
time_interval = 1.0 / Spad_fs
total_duration = len(sig_data) * time_interval
timestamps = np.arange(0, total_duration, time_interval)
timestamps_time = pd.to_timedelta(timestamps, unit='s')
sig_raw = pd.Series(sig_data, index=timestamps_time)
ref_raw = pd.Series(ref_data, index=timestamps_time)
zscore_raw = pd.Series(zscore_data, index=timestamps_time)

#%%
'To plot the LFP data using the pynapple method'
LFP=nap.Tsd(t = timestamps, d = zscore_raw.to_numpy(), time_units = 's')
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(LFP.as_units('s'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Time (s)")
ax.set_title("LFP_raw")

#%%
LFP_lowpass= OE.butter_filter(LFP, btype='low', cutoff=20, fs=Spad_fs, order=5)
LFP_lowpass=nap.Tsd(t = timestamps, d = LFP_lowpass, time_units = 's')

plt.figure(figsize=(15,5))
plt.plot(LFP_lowpass.as_units('s'))
plt.xlabel("Time (s)")
plt.title("theta band")
plt.show()
#%%
'''This is to set the short interval you want to look at '''
ex_ep = nap.IntervalSet(start = 0, end = 90, time_units = 's') 
'''detect theta wave '''
lfp_theta = OE.band_pass_filter(LFP,4,15,Spad_fs)
lfp_theta=nap.Tsd(t = timestamps, d = lfp_theta, time_units = 's')

plt.figure(figsize=(15,5))
plt.plot(lfp_theta.restrict(ex_ep).as_units('s'))
plt.xlabel("Time (s)")
plt.title("theta band")
plt.show()

#%%
Low_thres=1
ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getRippleEvents (LFP,Spad_fs,windowlen=1000,Low_thres=1,High_thres=10)

#%%
fig, ax = plt.subplots(4, 1, figsize=(15, 8))
OE.plotRippleSpectrogram (ax, LFP, ripple_band_filtered, rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres, y_lim=30, Fs=Spad_fs)

#%%
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from waveletFunctions import wave_signif, wavelet
sst = OE.butter_filter(sig_raw.to_numpy(), btype='low', cutoff=500, fs=Spad_fs, order=5)
#sst = OE.butter_filter(signal, btype='high', cutoff=30, fs=Recording1.fs, order=5)

sst = sst - np.mean(sst)
variance = np.std(sst, ddof=1) ** 2
print("variance = ", variance)
# ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------
if 0:
    variance = 1.0
    sst = sst / np.std(sst, ddof=1)
n = len(sst)
dt = 0.0001
time = np.arange(len(sst)) * dt   # construct time array
#%%
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
#%%
xlim = ([40,45])  # plotting range
fig, plt3 = plt.subplots(figsize=(15,5))

levels = [0, 4, 20, 80,100, 200, 999]
# *** or use 'contour'
CS = plt.contourf(time, frequency, power, len(levels))

plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Wavelet Power Spectrum')
plt.xlim(xlim[:])
plt3.set_yscale('log', base=2, subs=None)
#plt.ylim([np.min(frequency), np.max(frequency)])
plt.ylim([0, 300])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt3.ticklabel_format(axis='y', style='plain')
#plt3.invert_yaxis()
# set up the size and location of the colorbar
position=fig.add_axes([0.2,0.01,0.4,0.02])
plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)

plt.subplots_adjust(right=0.7, top=0.9)

#%%