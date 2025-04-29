# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:30:19 2023
This is the main file to perform a comparison analysis between LFP and optical signal.
This code will create a Class for the recording session you want to analyse.
@author: Yifang
"""
import pandas as pd
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE
import numpy as np
import os
import pickle
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
import matplotlib.pyplot as plt
#%%
'''recordingMode: use py, Atlas, SPAD for different systems
'''
# dpath='E:/ATLAS_SPAD/1825507_mCherry/Day1/'
# recordingName='SavedMovingTrials'

dpath='F:/2025_ATLAS_SPAD/PVCre/1842515_PV_mNeon/RippleTrials/'
recordingName='Day9SyncRecording19'
'''You can try LFP1,2,3,4 and plot theta to find the best channel'''
LFP_channel='LFP_2'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 
#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
Recording1.pynacollada_label_theta (LFP_channel,Low_thres=1,High_thres=10,save=False,plot_theta=True)
#%%
#This is to calculate and plot the trace around theta trough
Recording1.plot_theta_correlation(LFP_channel)
#%%
'plot feature can be LFP or SPAD to show the power spectrum of LFP or SPAD'
Recording1.plot_gamma_power_on_theta_cycle(LFP_channel=LFP_channel)
#%% Detect theta event
'''THETA PEAK DETECTION
For a rigid threshold to get larger amplitude theta events: Low_thres=1, for more ripple events, Low_thres=0.5'''
data_segment,timestamps=Recording1.pynappleThetaAnalysis (lfp_channel=LFP_channel,ep_start=2,ep_end=5,
                                                                         Low_thres=-0.3,High_thres=10,plot_segment=True,plot_ripple_ep=True)
#time_duration=transient_trace.index[-1].total_seconds()

 #%% Detect ripple event
'''RIPPLE DETECTION
For a rigid threshold to get larger amplitude ripple events: Low_thres=3, for more ripple events, Low_thres=1'''
rip_ep,rip_tsd=Recording1.pynappleAnalysis (lfp_channel=LFP_channel,ep_start=10,ep_end=30,
                                                                          Low_thres=1.2,High_thres=10,plot_segment=True,
                                                                          plot_ripple_ep=True,excludeTheta=True)
#%% Detect ripple event
'''GAMMA DETECTION
For a rigid threshold to get larger amplitude Gamma events: Low_thres=1, for more ripple events, Low_thres=0'''
rip_ep,rip_tsd=Recording1.pynappleGammaAnalysis (lfp_channel=LFP_channel,ep_start=16,ep_end=18,
                                                                          Low_thres=0.2,High_thres=8,plot_segment=True,
                                                                          plot_ripple_ep=True,excludeTheta=False,excludeNonTheta=False)
#%% Detect theta nested gamma event
'''GAMMA- Theta nested Gamma plot
For a rigid threshold to get larger amplitude Gamma events: Low_thres=1, for more ripple events, Low_thres=0'''
rip_ep,rip_tsd=Recording1.PlotThetaNestedGamma (lfp_channel=LFP_channel,Low_thres=-0.7,High_thres=10,plot_segment=False, plot_ripple_ep=True)

#%%
'''To plot the feature of a part of the signal'''
start_time=6
end_time=9
#%%
Recording1.plot_segment_feature (LFP_channel,start_time,end_time,SPAD_cutoff=50,lfp_cutoff=500)
#%%
'To plot the feature of theta-band and ripple-band of the segment signal'
Recording1.plot_band_power_feature (LFP_channel,start_time,end_time,LFP=True)

#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=3 #the duration of the segment, in seconds
viewNum=10 #the number of segments
for i in range(viewNum):
    Recording1.plot_segment_feature (LFP_channel=LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=20,lfp_cutoff=200)
    #Recording1.plot_band_power_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),LFP=True)
#%%
'''sliced_recording:choose a segment or a part of your recording, this can be defined with start and end time,
or just by theta_part, non_theta_part'''
sliced_recording=Recording1.slicing_pd_data (Recording1.Ephys_tracking_spad_aligned,start_time=0, end_time=90)
#silced_recording=theta_part
'''Calculate the cross correlation between two power spectrun over time at a specific frequency'''
sst_spad,frequency_spad,power_spad,global_ws_spad=OE.Calculate_wavelet(sliced_recording['zscore_raw'],lowpassCutoff=500,Fs=10000)
sst_lfp,frequency_lfp,power_lfp,global_ws_lfp=OE.Calculate_wavelet(sliced_recording[LFP_channel],lowpassCutoff=500,Fs=10000)
# Calculate the correlation coefficient
# lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (power_spad[18],power_lfp[18],corr_window=1)
# lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (power_spad[19],power_lfp[19],corr_window=1)
lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (power_spad[20],power_lfp[20],corr_window=1)
#%%
'''Calculate the cross correlation between LFP and optical signal for a specific segment'''
silced_recording=Recording1.slicing_pd_data (Recording1.Ephys_tracking_spad_aligned,start_time=0, end_time=90)
#silced_recording=theta_part
lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (silced_recording['zscore_raw'],silced_recording[LFP_channel],corr_window=0.5)
#%%
'''Calculate the cross correlation between LFP and optical signal for a specific segment, with low-pass filter'''
spad_lowpass= OE.smooth_signal(silced_recording['zscore_raw'],Fs=10000,cutoff=50)
lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='low', cutoff=50, fs=Recording1.fs, order=5)
spad_low = pd.Series(spad_lowpass, index=silced_recording['zscore_raw'].index)
lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)
lags,Corr_mean,Corr_std=Recording1.get_mean_corr_two_traces (spad_low,lfp_low,corr_window=0.5)
#%%
Fs=10000
# save_path = 'D:/2024_OEC_Atlas_main/1765508_Jedi2p_Atlas/Day2/Results'  # Change this to your desired directory
# os.makedirs(save_path, exist_ok=True)

#theta_part,non_theta_part=Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
LFP_theta=Recording1.theta_part[LFP_channel]
LFP_nontheta=Recording1.non_theta_part[LFP_channel]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
#fig.patch.set_alpha(0)
OpticalAnlaysis.PSD_plot (LFP_nontheta/1000,fs=Fs,method="welch",color='black', xlim=[0,40],linewidth=2,linestyle=':',label='LFP-rest',ax=ax)
OpticalAnlaysis.PSD_plot (LFP_theta/1000,fs=Fs,method="welch",color='black', xlim=[0,40],linewidth=2,linestyle='-',label='LFP-move',ax=ax)
#fig.savefig(os.path.join(save_path, "LFP_theta_PSD.png"), dpi=300, bbox_inches='tight', transparent=True)

optical_theta=Recording1.theta_part['zscore_raw']
optical_nontheta=Recording1.non_theta_part['zscore_raw']

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
#fig.patch.set_alpha(0)
OpticalAnlaysis.PSD_plot (optical_nontheta,fs=Fs,method="welch",color='tab:green', xlim=[0,40],linewidth=2,linestyle=':',label='optical-rest',ax=ax)
OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,40],linewidth=2,linestyle='-',label='optical-move',ax=ax)
#fig.savefig(os.path.join(save_path, "optical_theta_PSD.png"), dpi=300, bbox_inches='tight', transparent=True)
#%%
sliced_recording=Recording1.slicing_pd_data (Recording1.Ephys_tracking_spad_aligned,start_time=0, end_time=200)
LFP_sliced=sliced_recording[LFP_channel]
fig, ax = plt.subplots(1, 1, figsize=(3, 6))
OpticalAnlaysis.PSD_plot (LFP_sliced/1000,fs=Fs,method="welch",color='black', xlim=[0,40],linewidth=2,linestyle='-',label='LFP',ax=ax)
#%%
LFP_channel='LFP_1'
LFP=Recording1.Ephys_tracking_spad_aligned[LFP_channel]

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
OpticalAnlaysis.PSD_plot (LFP,fs=Fs,method="welch",color='tab:green', xlim=[0,40],linewidth=2,linestyle='--',label='LFP',ax=ax)
#OpticalAnlaysis.PSD_plot (optical_theta,fs=Fs,method="welch",color='tab:green', xlim=[0,100],linewidth=2,linestyle='-',label='move',ax=ax)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from tensorpac import Pac


# Example data (replace these with your actual data)
fs = 10000  # Sampling frequency in Hz

LFP = Recording1.theta_part[LFP_channel] # Example LFP signal
SPAD=Recording1.theta_part['zscore_raw'] 
LFP=LFP.to_numpy()
SPAD=SPAD.to_numpy()
# Define a Pac object
p_obj = Pac(idpac=(6,0,0),f_pha=(2, 20, 2, 0.4), f_amp=(30, 100, 10, 2))
#%%
# Filter the data and extract pac
xpac = p_obj.filterfit(fs, SPAD)

# plot your Phase-Amplitude Coupling :
p_obj.comodulogram(xpac.mean(-1), cmap='Spectral_r', plotas='contour', ncontours=5,
               title=r'theta phase$\Leftrightarrow$Gamma amplitude coupling',
               fz_title=14, fz_labels=13)

p_obj.show()
#%%
xpac = p_obj.filterfit(fs, LFP)

# plot your Phase-Amplitude Coupling :
p_obj.comodulogram(xpac.mean(-1), cmap='Spectral_r', plotas='contour', ncontours=5,
               title=r'theta phase$\Leftrightarrow$Gamma amplitude coupling',
               fz_title=14, fz_labels=13)

p_obj.show()
#%%
# extract all of the phases and amplitudes
pha_LFP = p_obj.filter(fs, LFP, ftype='phase')
amp_LFP = p_obj.filter(fs, LFP, ftype='amplitude')

pha_SPAD = p_obj.filter(fs, SPAD, ftype='phase')
amp_SPAD = p_obj.filter(fs, SPAD, ftype='amplitude')

pac_LFP = p_obj.fit(pha_LFP, amp_LFP).mean(-1)
pac_SPAD = p_obj.fit(pha_SPAD, amp_SPAD).mean(-1)
pac_LFPtheta_SPADgamma =p_obj.fit(pha_LFP, amp_SPAD).mean(-1)

vmax = np.min([pac_LFP.max(), pac_SPAD.max(), pac_LFPtheta_SPADgamma.max()])
kw = dict(vmax=vmax, vmin=0, cmap='viridis')
plt.figure(figsize=(14, 4))
plt.subplot(131)
p_obj.comodulogram(pac_LFP, title="PAC LFP", **kw)
plt.subplot(132)
p_obj.comodulogram(pac_SPAD, title="PAC SPAD", **kw)
plt.ylabel('')
plt.subplot(133)
p_obj.comodulogram(pac_LFPtheta_SPADgamma, title="PAC LFP-SPAD", **kw)
plt.ylabel('')
plt.tight_layout()
plt.show()