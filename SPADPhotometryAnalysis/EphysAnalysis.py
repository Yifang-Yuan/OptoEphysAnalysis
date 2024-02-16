# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:11:42 2023

@author: Yifang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:39:52 2023

@author: Yang
"""
import os
import os.path as op
import numpy as np
import pandas as pd
from scipy import signal
from open_ephys.analysis import Session
import pynapple as nap
import matplotlib.pylab as plt
from matplotlib.pyplot import *
#%%

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5):
#def butter_filter(data, btype='high', cutoff=3, fs=130, order=5): # for photometry data  
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    # fig, ax = plt.subplots(figsize=(15, 3))
    # ax=plot_trace(y,ax, label="trace_10Hz_low_pass")
    return y

def plotSingleTrace (ax, signal, SamplingRate=30000,color='tab:blue'):
    ax.plot(signal,color,linewidth=1,alpha=0.8)
    xtick=ax.get_xticks()/SamplingRate
    xtick=np.round(xtick,2)
    ax.set_xticklabels(xtick,fontsize=10)
    ax.set_yticklabels(ax.get_yticks(),fontsize=10)
    ax.set_xlabel("seconds",fontsize=10)
    return ax 

def plotTwoTrace (ax, Ehpys_signal,SPAD_signal, Fs_SPAD=9938.4, Fs_EPhys=30000,color_SPAD='k',color_Ephys='tab:blue'):
    t_SPAD=np.arange(0, len(SPAD_signal)/9938.4, 1/9938.4)
    t_Ephys=np.arange(0, len(Ehpys_signal)/30000, 1/30000)
    ax.plot(t_SPAD,SPAD_signal,color_SPAD,linewidth=1,alpha=0.8)
    ax.plot(t_Ephys,Ehpys_signal,color_Ephys,linewidth=1,alpha=0.8)
    ax.set_xticklabels(np.round(ax.get_xticks(),2),fontsize=10)
    ax.set_yticklabels(ax.get_yticks(),fontsize=10)
    ax.set_xlabel("seconds",fontsize=10)
    ax.set_ylabel("Amplitude")
    return ax 

def get_detrend(sub):
     sub_detrend = signal.detrend(sub)
     return sub_detrend
 
def calculate_correlation (data1,data2):
    '''normalize'''
    s1 = (data1 - np.mean(data1)) / (np.std(data1))
    s2 = (data2 - np.mean(data2)) / (np.std(data2))
    lags=signal.correlation_lags(len(data1), len(data2), mode='full') 
    corr=signal.correlate(s1, s2, mode='full', method='auto')/len(data1)
    return lags,corr
def calculate_correlation_sub (sub1,sub2):
    trace1=get_detrend(sub1)
    trace2=get_detrend(sub2)
    lags,corr=calculate_correlation (trace1,trace2)
    return lags,corr

def band_pass_filter(data,low_freq,high_freq,Fs):
    data_high=butter_filter(data, btype='high', cutoff=low_freq,fs=Fs, order=1)
    data_low=butter_filter(data_high, btype='low', cutoff=high_freq, fs=Fs, order=5)
    return data_low

def notchfilter (data,f0=50,Q=20):
    f0 = 50 # Center frequency of the notch (Hz)
    Q = 20 # Quality factor
    b, a = signal.iirnotch(f0, Q, 30000)
    data=signal.filtfilt(b, a, data)
    return data
#%%
directory = "E:/SPAD/SPADData/20230424_Ephys_sleep_ASAPpyPhoto/2023-04-24_18-04-55_9819_sleep"
#directory = "G:/SPAD/SPADData/20230409_OEC_Ephys/2023-04-05_17-02-58_9820-noisy" #Indeed noisy
#directory = "E:/SPAD/SPADData/20230409_OEC_Ephys/2023-04-05_15-25-32_9819"

#%%
session = Session(directory)
recording= session.recordnodes[0].recordings[0]
continuous=recording.continuous
continuous0=continuous[0]
samples=continuous0.samples
timestamps=continuous0.timestamps
events=recording.events
#%%
'''Recording nodes that are effective'''
LFP1=samples[:,8]
LFP2=samples[:,9]
LFP3=samples[:,10]
LFP4=samples[:,11]
LFP5=samples[:,13]
LFP_NA=samples[:,1]
'''ADC lines that recorded the analog input from SPAD PCB X10 pin'''
Sync1=samples[:,16] #Full pulsed aligned with X10 input
Sync2=samples[:,17]
Sync3=samples[:,18]
Sync4=samples[:,19]

#%%
LFPdata= notchfilter (LFP2,f0=50,Q=20)
#%%
LFP=nap.Tsd(t = timestamps, d = LFPdata, time_units = 's')
# And how the software transform you timetamps in second in timestamps in microsecond
# You can plot your data
plt.plot(LFP, '-')
plt.show()
#%%
import pynacollada as pyna
frequency=30000
lfpsleep=LFP[0:9000000]
signal = pyna.eeg_processing.bandpass_filter(lfpsleep, 100, 300, frequency)

figure(figsize=(15,5))
plot(lfpsleep.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
show()
#%%
windowLength = 51

from scipy.signal import filtfilt

squared_signal = np.square(signal.values)
window = np.ones(windowLength)/windowLength
nSS = filtfilt(window, 1, squared_signal)
nSS = (nSS - np.mean(nSS))/np.std(nSS)
nSS = nap.Tsd(t = signal.index.values, d = nSS, time_support = signal.time_support)

# Round1 : Detecting Ripple Periods by thresholding normalized signal
low_thres = 1
high_thres = 10

nSS2 = nSS.threshold(low_thres, method='above')
nSS3 = nSS2.threshold(high_thres, method='below')

# Round 2 : Excluding ripples whose length < minRipLen and greater than Maximum Ripple Length
minRipLen = 20 # ms
maxRipLen = 200 # ms

rip_ep = nSS3.time_support
rip_ep = rip_ep.drop_short_intervals(minRipLen, time_units = 'ms')
rip_ep = rip_ep.drop_long_intervals(maxRipLen, time_units = 'ms')

# Round 3 : Merging ripples if inter-ripple period is too short
minInterRippleInterval = 20 # ms


rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
rip_ep = rip_ep.reset_index(drop=True)

# Extracting Ripple peak
rip_max = []
rip_tsd = []
for s, e in rip_ep.values:
    tmp = nSS.loc[s:e]
    rip_tsd.append(tmp.idxmax())
    rip_max.append(tmp.max())

rip_max = np.array(rip_max)
rip_tsd = np.array(rip_tsd)

rip_tsd = nap.Tsd(t = rip_tsd, d = rip_max, time_support = sleep_ep)

# Writing for neuroscope the Intervals
data.write_neuroscope_intervals(extension='.rip.evt', isets=rip_ep, name='Ripples')

# Saving ripples time and epochs
data.save_nwb_intervals(rip_ep, 'sleep_ripples')
data.save_nwb_timeseries(rip_tsd, 'sleep_ripples')

# Load ripples times
rip_ep = data.load_nwb_intervals('sleep_ripples')
rip_tsd = data.load_nwb_timeseries('sleep_ripples')



rip_ep1, rip_tsd1 = pyna.eeg_processing.detect_oscillatory_events(
                                            lfp = lfp,
                                            epoch = sleep_ep,
                                            freq_band = (100,300),
                                            thres_band = (1, 10),
                                            duration_band = (0.02,0.2),
                                            min_inter_duration = 0.02
                                            )