# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:36:48 2024
@author: Yifang
"""
from SyncOECPySessionClass import SyncOEpyPhotometrySession
import OpenEphysTools as OE

'''recordingMode: use py, Atlas, SPAD for different systems
'''
#dpath= 'F:/2025_ATLAS_SPAD/1887930_PV_mNeon_mCherry/Day4/'
dpath=r'G:\2025_ATLAS_SPAD\PyramidalWT\1881363_Jedi2p_mCherry\Day1'
recordingName='SyncRecording5'
LFP_channel='LFP_1'
Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,read_aligned_data_from_file=True,
                                     recordingMode='Atlas',indicator='GEVI') 

#%%
'''separate the theta and non-theta parts.
theta_thres: the theta band power should be bigger than 80% to be defined theta period.
nonthetha_thres: the theta band power should be smaller than 50% to be defined as theta period.'''
Recording1.pynacollada_label_theta (LFP_channel,Low_thres=-0.5,High_thres=8,save=False,plot_theta=True)
#plot_GEVI_theta_correlation(Recording1)
#%%
'''Here for the spectrum, I used a 0.5Hz high pass filter to process both signals'''
timewindow=3 #the duration of the segment, in seconds
viewNum=9 #the number of segments
for i in range(viewNum):
    #Recording1.plot_segment_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=100,lfp_cutoff=500)
    'This is to plot two optical traces from two ROIs, i.e. one signal and one reference'
    #Recording1.plot_segment_feature_multiROI (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=100,lfp_cutoff=500)
    #Recording1.plot_segment_band_feature_twoROIs (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=50,lfp_cutoff=500)
    Recording1.plot_segment_feature (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=50,lfp_cutoff=100)
    #Recording1.plot_freq_power_coherence (LFP_channel,start_time=timewindow*i,end_time=timewindow*(i+1),SPAD_cutoff=50,lfp_cutoff=200)

#%%
'''To plot the feature of a part of the signal'''
start_time=70 #Day4 and day6
end_time=75

#sig_smooth,ref_smooth,sig_raw=Recording1.plot_segment_band_feature_twoROIs (LFP_channel,start_time,end_time,SPAD_cutoff=100,lfp_cutoff=500)
#Recording1.plot_segment_feature_multiROI (LFP_channel,start_time=start_time,end_time=end_time,SPAD_cutoff=50,lfp_cutoff=500)

#Recording1.plot_segment_band_feature (LFP_channel,start_time,end_time,SPAD_cutoff=50,lfp_cutoff=500)
# coherence=Recording1.plot_freq_power_coherence (LFP_channel,start_time,end_time,SPAD_cutoff=50,lfp_cutoff=500)
Recording1.plot_segment_feature (LFP_channel,start_time,end_time,SPAD_cutoff=100,lfp_cutoff=100)

#%%
'''To plot the spectrum coherence for LFP and optical signal'''
start_time=13
end_time=18
coherence=Recording1.plot_freq_power_coherence (LFP_channel,start_time,end_time,SPAD_cutoff=100,lfp_cutoff=100)
#%%

def plot_GEVI_theta_correlation(SyncRecordingObject):
    df=SyncRecordingObject.Ephys_tracking_spad_aligned
    df=df.reset_index(drop=True)
    #print (silced_recording.index)
    df['theta_angle']=OE.calculate_theta_phase_angle(df['zscore_raw'], theta_low=5, theta_high=12) #range 5 to 9
    OE.plot_trace_in_seconds(df['theta_angle'],Fs=10000,title='theta angle')
    trough_index,peak_index = OE.calculate_theta_trough_index(df,Fs=10000)
    #print (trough_index)
    OE.plot_theta_cycle (df, LFP_channel,trough_index,half_window=0.15,fs=10000,plotmode='two')
    OE.plot_zscore_to_theta_phase (df['theta_angle'],df['zscore_raw'])
    return trough_index