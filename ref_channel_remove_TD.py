# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 22:45:37 2025

@author: yifan
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
import os
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis

fs=1682.92
dpath = r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_mCherry\Day9_TD\SyncRecording3'  # Replace with your actual path

# Read the CSV files (no header)
raw_signal = pd.read_csv(os.path.join(dpath, 'Green_raw.csv'), header=None)
raw_reference = pd.read_csv(os.path.join(dpath, 'Red_raw.csv'), header=None)
raw_signal = raw_signal.values
raw_reference = raw_reference.values
raw_signal = raw_signal.flatten()
raw_reference = raw_reference.flatten()

fig, ax = plt.subplots(figsize=(8, 2))
Analysis.plot_trace(raw_signal[0:100],ax, fs, label="raw_signal trace")
fig, ax = plt.subplots(figsize=(8, 2))
Analysis.plot_trace(raw_reference[0:100],ax, fs, label="raw_ref trace")
#%%
Green,_= Analysis.getTimeDivisionTrace (dpath,raw_signal,35000,19000,3000,1000)
fname = os.path.join(dpath, "Green_trace_demod.csv")
np.savetxt(fname, Green, delimiter=",")
#%%
_,Red= Analysis.getTimeDivisionTrace (dpath, raw_reference, 20000,15000, 15000,13000)
fname = os.path.join(dpath, "Red_trace_demod.csv")
np.savetxt(fname, Red, delimiter=",")
#%%

dpath = r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_mCherry\Day9_TD\SyncRecording3'  # Replace with your actual path

# Read the CSV files (no header)
raw_signal = pd.read_csv(os.path.join(dpath, 'Green_trace_demod.csv'), header=None)
raw_reference = pd.read_csv(os.path.join(dpath, 'Red_trace_demod.csv'), header=None)
raw_signal = raw_signal[0]        # Extract column 0 as a pandas Series
raw_reference = raw_reference[0]
sampling_rate=1682.92
'''Read csv file and calculate zscore of the fluorescent signal'''
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(311)
ax1 = fp.plotSingleTrace (ax1, raw_signal, SamplingRate=sampling_rate,color='green',Label='Signal')
ax2 = fig.add_subplot(312)
ax2 = fp.plotSingleTrace (ax2, raw_reference, SamplingRate=sampling_rate,color='purple',Label='Reference')
# ax3 = fig.add_subplot(313)
# ax3 = fp.plotSingleTrace (ax3, Cam_Sync, SamplingRate=sampling_rate,color='orange',Label='Digital_Sync')
plt.tight_layout()
#%%
'''
You can get zdFF directly by calling the function fp.get_zdFF()
TO CHECK THE SIGNAL STEP BY STEP:
YOU CAN USE THE FOLLOWING CODES TO GET MORE PLOTS
These will give you plots for 
smoothed signal, corrected signal, normalised signal and the final zsocre
'''
'''Step 1, plot smoothed traces'''
smooth_win = 10

smooth_reference,smooth_signal,r_base,s_base = fp.photometry_smooth_plot (raw_reference,raw_signal,sampling_rate=sampling_rate, smooth_win = smooth_win)
#%%
'''Step 2, plot corrected traces, removing the baseline (detrend)'''
remove=0
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:])  

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(311)
ax1 = fp.plotSingleTrace (ax1, signal, SamplingRate=sampling_rate,color='blue',Label='corrected_signal')
ax2 = fig.add_subplot(312)
ax2 = fp.plotSingleTrace (ax2, reference, SamplingRate=sampling_rate,color='purple',Label='corrected_reference')
# ax3 = fig.add_subplot(313)
# ax3 = fp.plotSingleTrace (ax3, Cam_Sync, SamplingRate=sampling_rate,color='orange',Label='Digital_Sync')
# plt.tight_layout()

fname = os.path.join(dpath, "Green_traceAll.csv")
np.savetxt(fname, signal, delimiter=",")
fname = os.path.join(dpath, "Red_traceAll.csv")
np.savetxt(fname, reference, delimiter=",")

#%%
'''Step 3, plot normalised traces'''
z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
ax2 = fig.add_subplot(212)
ax2 = fp.plotSingleTrace (ax2, z_reference, SamplingRate=sampling_rate,color='purple',Label='normalised_reference')

#%%
'''Step 4, plot fitted reference trace and signal'''
from sklearn.linear_model import Lasso
lin = Lasso(alpha=0.001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
n = len(z_reference)
'''Need to change to numpy if previous smooth window is 1'''
# z_signal=z_signal.to_numpy()
# z_reference=z_reference.to_numpy()
lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)

fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(111)
ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
ax1 = fp.plotSingleTrace (ax1, z_reference_fitted, SamplingRate=sampling_rate,color='purple',Label='fitted_reference')
#%%
'''Step 5, plot zscore'''
zdFF = (z_signal - z_reference_fitted)
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax1 = fp.plotSingleTrace (ax1, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')

fname = os.path.join(dpath, "Zscore_traceAll.csv")
np.savetxt(fname, zdFF, delimiter=",")