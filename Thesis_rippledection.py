# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:03:25 2023

@author: Yifang
"""

# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-17 15:50:57
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-11 17:46:22

import numpy as np
import pynapple as nap
from matplotlib.pyplot import *
import pynacollada as pyna
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession


dpath= 'F:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day4/'
#dpath='E:/ATLAS_SPAD/1825507_mCherry/Day1/'
recordingName='SyncRecording8'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                     read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 
LFP_channel='LFP_1'
#%%
LFP=Recording1.Ephys_tracking_spad_aligned[LFP_channel]
timestamps=Recording1.Ephys_tracking_spad_aligned['timestamps'].copy().to_numpy()
LFP_data=nap.Tsd(t = timestamps, d = LFP.to_numpy(), time_units = 's')
#%%
ep_start=7.5
ep_end=9.5

ex_ep = nap.IntervalSet(start = ep_start+timestamps[0], end = ep_end+timestamps[0], time_units = 's') 
#%%
frequency=10000
signal = pyna.eeg_processing.bandpass_filter(LFP_data, 100, 300, frequency)

figure(figsize=(15,5))
subplot(211)
plot(LFP_data.restrict(ex_ep).as_units('s'))
subplot(212)
plot(signal.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
show()

#%% Second step is to look at the enveloppe of the filtered signal.
windowLength = 500

from scipy.signal import filtfilt

squared_signal = np.square(signal.values)
window = np.ones(windowLength)/windowLength
nSS = filtfilt(window, 1, squared_signal)
nSS = (nSS - np.mean(nSS))/np.std(nSS)
nSS = nap.Tsd(t = signal.index.values, d = nSS, time_support = signal.time_support)


#%%
low_thres = 1
high_thres = 10

nSS2 = nSS.threshold(low_thres, method='above')
nSS3 = nSS2.threshold(high_thres, method='below')

# Get time index from LFP_data (assuming index is in seconds or convertible)
time_values = LFP_data.restrict(ex_ep).as_units('s').index
xlim_start = time_values[0]
xlim_end = time_values[-1]

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

# Subplot 1: LFP
axes[0].plot(LFP_data.restrict(ex_ep).as_units('s'))
axes[0].set_xlim(xlim_start, xlim_end)
axes[0].tick_params(labelbottom=False)
axes[0].set_ylabel("Amplitude (μV)")
# Subplot 2: signal
axes[1].plot(signal.restrict(ex_ep).as_units('s'))
axes[1].set_xlim(xlim_start, xlim_end)
axes[1].tick_params(labelbottom=False)
axes[1].set_ylabel("Amplitude (μV)")
# Subplot 3: nSS and thresholded signal
axes[2].plot(nSS.restrict(ex_ep).as_units('s'))
axes[2].plot(nSS3.restrict(ex_ep).as_units('s'), '.')
axes[2].axhline(low_thres, color='grey', linestyle='--')
axes[2].set_xlim(xlim_start, xlim_end)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Amplitude (μV)")
# Adjust layout
plt.tight_layout()
plt.show()
#%%