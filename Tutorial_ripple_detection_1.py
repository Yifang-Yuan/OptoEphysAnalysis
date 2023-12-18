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

data_directory = 'C:/Users/Yang/Downloads/A2929-200711/A2929-200711'

data = nap.load_session(data_directory, 'neurosuite')

sleep_ep = data.epochs['sleep']

frequency = 1250.0

lfp = data.load_lfp(channel=15,extension='.eeg',frequency=frequency)

lfpsleep = lfp.restrict(sleep_ep)

ex_ep = nap.IntervalSet(start = 0, end = 10.0, time_units = 's') 
figure(figsize=(15,5))
plot(lfpsleep.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
show()
#%%
signal = pyna.eeg_processing.bandpass_filter(lfpsleep, 100, 300, frequency)

figure(figsize=(15,5))
subplot(211)
plot(lfpsleep.restrict(ex_ep).as_units('s'))
subplot(212)
plot(signal.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
show()

#%% Second step is to look at the enveloppe of the filtered signal.
windowLength = 51

from scipy.signal import filtfilt

squared_signal = np.square(signal.values)
window = np.ones(windowLength)/windowLength
nSS = filtfilt(window, 1, squared_signal)
nSS = (nSS - np.mean(nSS))/np.std(nSS)
nSS = nap.Tsd(t = signal.index.values, d = nSS, time_support = signal.time_support)
#%%
figure(figsize=(15,5))
subplot(311)
plot(lfpsleep.restrict(ex_ep).as_units('s'))
subplot(312)
plot(signal.restrict(ex_ep).as_units('s'))
subplot(313)
plot(nSS.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
tight_layout()
show()
#%%

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
#%%
figure(figsize=(15,5))
plot(lfpsleep.restrict(rip_tsd1).as_units('s'))
xlabel("Time (s)")
show()
#%%
ex_ep = nap.IntervalSet(start = 0.5, end = 2.0, time_units = 's') 
frequency = 1250.0

lfp = data.load_lfp(channel=15,extension='.eeg',frequency=frequency)

lfpsleep = lfp.restrict(sleep_ep)

figure(figsize=(15,5))
plot(lfpsleep.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
show()
#%%
signal = pyna.eeg_processing.bandpass_filter(lfpsleep, 100, 300, frequency)

figure(figsize=(15,5))
subplot(211)
plot(lfpsleep.restrict(ex_ep).as_units('s'))
subplot(212)
plot(signal.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
show()
#%%
windowLength = 51

from scipy.signal import filtfilt

squared_signal = np.square(signal.values)
window = np.ones(windowLength)/windowLength
nSS = filtfilt(window, 1, squared_signal)
nSS = (nSS - np.mean(nSS))/np.std(nSS)
nSS = nap.Tsd(t=signal.index.values, 
              d=nSS, 
              time_support=signal.time_support)
              

figure(figsize=(15,5))
subplot(311)
plot(lfpsleep.restrict(ex_ep).as_units('s'))
subplot(312)
plot(signal.restrict(ex_ep).as_units('s'))
subplot(313)
plot(nSS.restrict(ex_ep).as_units('s'))
xlabel("Time (s)")
tight_layout()
show()

##%%
low_thres = 1
high_thres = 10

nSS2 = nSS.threshold(low_thres, method='above')
nSS3 = nSS2.threshold(high_thres, method='below')

figure(figsize=(15,5))
subplot(311)
plot(lfpsleep.restrict(ex_ep).as_units('s'))
subplot(312)
plot(signal.restrict(ex_ep).as_units('s'))
subplot(313)
plot(nSS.restrict(ex_ep).as_units('s'))
plot(nSS3.restrict(ex_ep).as_units('s'), '.')
axhline(low_thres)
xlabel("Time (s)")
tight_layout()
show()