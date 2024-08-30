# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:00:55 2023
@author:Yifang
PACKAGE THAT NEED FOR THIS ANALYSIS
https://github.com/open-ephys/open-ephys-python-tools
https://github.com/pynapple-org/pynapple
https://github.com/PeyracheLab/pynacollada#getting-started

These are functions that I will call in main analysis.
"""
import os
import numpy as np
import pandas as pd
from scipy import signal
from open_ephys.analysis import Session
import matplotlib.pylab as plt
import pynapple as nap
import pynacollada as pyna
from scipy.signal import filtfilt
import pickle
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from matplotlib.ticker import MaxNLocator

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5): 
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def band_pass_filter(data,low_freq,high_freq,Fs):
    data_high=butter_filter(data, btype='high', cutoff=low_freq,fs=Fs, order=5)
    data_low=butter_filter(data_high, btype='low', cutoff=high_freq, fs=Fs, order=5)
    return data_low


def notchfilter (data,f0=50,bw=10,fs=30000):
    # Bandwidth of the notch filter (in Hz)   
    Q = f0/bw # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    data=signal.filtfilt(b, a, data)
    return data

def smooth_signal(data,Fs,cutoff,window='flat'):

    """smooth the data using a window with requested size.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.
    output:
        the smoothed signal        
    """
    window_len=int(Fs/cutoff)
    x=data.reset_index(drop=True)
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(int(window_len/2)-1):-int(window_len/2)]

def readEphysChannel (Directory,recordingNum,Fs=30000):
    '''Read a single recording in a specific session or folder'''
    session = Session(Directory)
    recording= session.recordnodes[0].recordings[recordingNum]
    continuous=recording.continuous
    continuous0=continuous[0]
    samples=continuous0.samples
    timestamps=continuous0.timestamps
    events=recording.events
    
    '''Recording nodes that are effective'''
    LFP1=samples[:,8]
    LFP2=samples[:,9]
    LFP3=samples[:,10]
    LFP4=samples[:,11]
    LFP5=samples[:,13]
    '''ADC lines that recorded the analog input from SPAD PCB X10 pin'''
    Sync1=samples[:,16] #Full pulsed aligned with X10 input
    Sync2=samples[:,17]
    Sync3=samples[:,18]
    Sync4=samples[:,19]
    
    LFP_clean1= butter_filter(LFP1, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean2= butter_filter(LFP2, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean3= butter_filter(LFP3, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean4= butter_filter(LFP4, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean1= notchfilter (LFP_clean1,f0=50,bw=5)
    LFP_clean2= notchfilter (LFP_clean2,f0=50,bw=5)
    LFP_clean3= notchfilter (LFP_clean3,f0=50,bw=5)
    LFP_clean4= notchfilter (LFP_clean4,f0=50,bw=5)
    
    EphysData = pd.DataFrame({
        'timestamps': timestamps,
        'CamSync': Sync1,
        'SPADSync': Sync2,
        'AtlasSync': Sync3,
        'LFP_1': LFP_clean1,
        'LFP_2': LFP_clean2,
        'LFP_3': LFP_clean3,
        'LFP_4': LFP_clean4,
    })
    
    return EphysData

def readEphysChannel_withSessionInput (session,recordingNum,Fs=30000):
    '''Same as the above function but used for batch processing when we already read a session'''
    recording= session.recordnodes[0].recordings[recordingNum]
    continuous=recording.continuous
    continuous0=continuous[0]
    samples=continuous0.samples
    timestamps=continuous0.timestamps
    events=recording.events
    
    '''Recording nodes that are effective'''
    LFP1=samples[:,8]
    LFP2=samples[:,9]
    LFP3=samples[:,10]
    LFP4=samples[:,11]
    LFP5=samples[:,13]
    '''ADC lines that recorded the analog input from SPAD PCB X10 pin'''
    Sync1=samples[:,16] #Full pulsed aligned with X10 input
    Sync2=samples[:,17]
    Sync3=samples[:,18]
    Sync4=samples[:,19]
    
    LFP_clean1= butter_filter(LFP1, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean2= butter_filter(LFP2, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean3= butter_filter(LFP3, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean4= butter_filter(LFP4, btype='low', cutoff=2000, fs=Fs, order=5)
    LFP_clean1= notchfilter (LFP_clean1,f0=50,bw=5)
    LFP_clean2= notchfilter (LFP_clean2,f0=50,bw=5)
    LFP_clean3= notchfilter (LFP_clean3,f0=50,bw=5)
    LFP_clean4= notchfilter (LFP_clean4,f0=50,bw=5)
    
    EphysData = pd.DataFrame({
        'timestamps': timestamps,
        'CamSync': Sync1,
        'SPADSync': Sync2,
        'AtlasSync': Sync3,
        'LFP_1': LFP_clean1,
        'LFP_2': LFP_clean2,
        'LFP_3': LFP_clean3,
        'LFP_4': LFP_clean4,
    })
    
    return EphysData

def SPAD_sync_mask (SPAD_Sync, start_lim, end_lim):
    '''
       	SPAD_Sync : numpy array
       		This is SPAD X10 output to the Open Ephys acquisition board. Each recorded frame will output a pulse.
       	start_lim : frame number
       	end_lim : frame number
       	SPAD_Sync usually have output during live mode and when the GUI is stopped. 
       	start and end lim will roughly limit the time duration for the real acquistion time.
       	Returns: SPAD_mask : numpy list
       		0 and 1 mask, 1 means SPAD is recording during this time.
    '''
    SPAD_mask=np.zeros(len(SPAD_Sync),dtype=np.int)
    SPAD_mask[np.where(SPAD_Sync <5000)[0]]=1
    SPAD_mask[0:start_lim]=0
    SPAD_mask[end_lim:]=0
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(SPAD_mask)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(len(SPAD_Sync)-4):
        if ((SPAD_mask[i]==0)&(SPAD_mask[i+1]==0)&(SPAD_mask[i+2]==0)&(SPAD_mask[i+3]==0)&(SPAD_mask[i+4]==0))==False:
        #if ((SPAD_mask[i]==0)&(SPAD_mask[i+1]==0))==False:
           	SPAD_mask[i]=1

    plot_trace_in_seconds(SPAD_mask,30000)
    mask_array_bool = np.array(SPAD_mask, dtype=bool)
    return mask_array_bool

def Atlas_sync_mask (Atlas_Sync, start_lim, end_lim,recordingTime=30):
    
    Atlas_mask=np.zeros(len(Atlas_Sync),dtype=np.int)
    #peak_index = np.argmax(Atlas_Sync > 25000)
    peak_indices = np.argwhere(Atlas_Sync > 25000).flatten()
    peak_index=peak_indices[-1]
    #print ('peak_indices', peak_indices)
    print ('peak_index', peak_index)
    duration=int (recordingTime*30000)
    Atlas_mask[peak_index:peak_index+duration]=1
    Atlas_mask[0:start_lim]=0
    Atlas_mask[end_lim:]=0
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(Atlas_mask)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plot_trace_in_seconds(Atlas_mask,30000)
    mask_array_bool = np.array(Atlas_mask, dtype=bool)
    return mask_array_bool

def py_sync_mask (Sync_line, start_lim, end_lim):
    py_mask=np.zeros(len(Sync_line),dtype=np.int)
    py_mask[np.where(Sync_line >15000)[0]]=1
    rising_edge_index = None
    falling_edge_index = None
    py_mask[0:start_lim]=0
    py_mask[end_lim:]=0
    # Iterate through the data array
    for i in range(len(py_mask) - 1):
        # Check for rising edge (transition from low to high)
        if py_mask[i] == 0 and py_mask[i + 1] == 1:
            rising_edge_index = i
            break  # Exit loop once the first rising edge is found

    # Iterate through the data array in reverse to find the falling edge
    for i in range(len(py_mask) - 1, 0, -1):
        # Check for falling edge (transition from high to low)
        if py_mask[i] == 1 and py_mask[i - 1] == 0:
            falling_edge_index = i
            break  # Exit loop once the last falling edge is found
    py_mask_final=np.zeros(len(Sync_line),dtype=np.int)
    print ('The py_mask 1st index is: ',rising_edge_index)
    py_mask_final[rising_edge_index:falling_edge_index]=1
    mask_array_bool = np.array(py_mask_final, dtype=bool)
    return mask_array_bool

def check_Optical_mask_length(data):
    filtered_series = data[data == 1]
    length_of_filtered_series = len(filtered_series)
    print('Mask length as Sample number in EphysSync:', length_of_filtered_series)
    Length_in_second=length_of_filtered_series/30000
    print('Mask length in Second:', Length_in_second)
    spad_sample_num=Length_in_second*9938.4
    print('Total optical sample number (if SPAD):', spad_sample_num)
    return -1

def save_SPAD_mask (dpath,mask_data_array):
    savefilename=os.path.join(dpath, "SPAD_mask.pkl")
    with open(savefilename, 'wb') as pickle_file:
        pickle.dump(mask_data_array, pickle_file) 
        return -1
    
def save_open_ephys_data (dpath, data):
    filepath=os.path.join(dpath, "open_ephys_read_pd.pkl")
    data.to_pickle(filepath)
    return -1

def getRippleEvents (lfp_raw,Fs,windowlen=200,Low_thres=1,High_thres=10,low_freq=130,high_freq=250):
    ripple_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, low_freq, high_freq, Fs) #for ripple:130Hz-250Hz
    squared_signal = np.square(ripple_band_filtered.values)
    window = np.ones(windowlen)/windowlen
    nSS = filtfilt(window, 1, squared_signal)
    nSS = (nSS - np.mean(nSS))/np.std(nSS)
    nSS = nap.Tsd(t=ripple_band_filtered.index.values, 
                  d=nSS, 
                  time_support=ripple_band_filtered.time_support)
                  
    nSS2 = nSS.threshold(Low_thres, method='above')
    nSS3 = nSS2.threshold(High_thres, method='below')
    # Round 2 : Excluding ripples whose length < minRipLen and greater than Maximum Ripple Length
    minRipLen = 10 # ms
    maxRipLen = 500 # ms   //200ms 
    rip_ep = nSS3.time_support
    rip_ep = rip_ep.drop_short_intervals(minRipLen, time_units = 'ms')
    rip_ep = rip_ep.drop_long_intervals(maxRipLen, time_units = 'ms')
    
    # Round 3 : Merging ripples if inter-ripple period is too short
    # minInterRippleInterval = 20 # ms   
    # rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
    # rip_ep = rip_ep.reset_index(drop=True)
    
    # Extracting Ripple peak
    rip_max = []
    rip_tsd = []
    for s, e in rip_ep.values:
        tmp = nSS.loc[s:e]
        rip_tsd.append(tmp.idxmax())
        rip_max.append(tmp.max())
    
    rip_max = np.array(rip_max)
    rip_tsd = np.array(rip_tsd)
    
    rip_tsd = nap.Tsd(t = rip_tsd, d = rip_max)

    return ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd

def getThetaEvents (lfp_raw,Fs,windowlen=1000,Low_thres=2,High_thres=10):
    theta_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, 5, 9, Fs,order=2)
    squared_signal = np.square(theta_band_filtered.values)
    window = np.ones(windowlen)/windowlen
    nSS = filtfilt(window, 1, squared_signal)
    nSS = (nSS - np.mean(nSS))/np.std(nSS)
    nSS = nap.Tsd(t=theta_band_filtered.index.values, 
                  d=nSS, 
                  time_support=theta_band_filtered.time_support)            
    nSS2 = nSS.threshold(Low_thres, method='above')
    nSS3 = nSS2.threshold(High_thres, method='below')
    # Round 2 : Excluding ripples whose length < minRipLen and greater than Maximum Ripple Length
    minThetaLen = 80 # ms
    maxThetaLen = 10000 # ms    
    rip_ep = nSS3.time_support
    rip_ep = rip_ep.drop_short_intervals(minThetaLen, time_units = 'ms')
    rip_ep = rip_ep.drop_long_intervals(maxThetaLen, time_units = 'ms')
    
    # Round 3 : Merging ripples if inter-ripple period is too short
    # minInterRippleInterval = 20 # ms   
    # rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
    # rip_ep = rip_ep.reset_index(drop=True)
    
    # Extracting Ripple peak
    rip_max = []
    rip_tsd = []
    for s, e in rip_ep.values:
        tmp = nSS.loc[s:e]
        rip_tsd.append(tmp.idxmax())
        rip_max.append(tmp.max())
    rip_max = np.array(rip_max)
    rip_tsd = np.array(rip_tsd)
    rip_tsd = nap.Tsd(t = rip_tsd, d = rip_max)
    return theta_band_filtered,nSS,nSS3,rip_ep,rip_tsd

def getThetaDeltaRatio (lfp_raw,Fs,windowlen=1000):
    theta_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, 5, 9, Fs,order=2)
    squared_signal = np.square(theta_band_filtered.values)
    window = np.ones(windowlen)/windowlen
    nSS_theta = filtfilt(window, 1, squared_signal)
    nSS_theta = (nSS_theta - np.mean(nSS_theta))/np.std(nSS_theta)
    # nSS_theta = nap.Tsd(t=theta_band_filtered.index.values, d=nSS_theta, time_support=theta_band_filtered.time_support)      
    
    delta_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, 1, 4, Fs,order=2)
    squared_signal = np.square(delta_band_filtered.values)
    window = np.ones(windowlen)/windowlen
    nSS_delta = filtfilt(window, 1, squared_signal)
    nSS_delta = (nSS_delta - np.mean(nSS_delta))/np.std(nSS_delta)
    # nSS_delta = nap.Tsd(t=delta_band_filtered.index.values, d=nSS_delta, time_support=delta_band_filtered.time_support) 
    ThetaDeltaRatio=np.abs(nSS_theta/nSS_delta)
    ThetaDeltaRatio[ThetaDeltaRatio > 2] = 2
    return ThetaDeltaRatio

def get_detrend(data):
     data_detrend = signal.detrend(data)
     return data_detrend
 
def calculate_correlation (data1,data2):
    '''normalize'''
    s1 = (data1 - np.mean(data1)) / (np.std(data1))
    s2 = (data2 - np.mean(data2)) / (np.std(data2))
    lags=signal.correlation_lags(len(data1), len(data2), mode='full') 
    corr=signal.correlate(s1, s2, mode='full', method='auto')/len(data1)
    return lags,corr

def calculate_correlation_with_detrend (spad_data,lfp_data):
    if isinstance(spad_data, (pd.DataFrame, pd.Series)):
        spad_np=spad_data.values
    else:
        spad_np=spad_data
    if isinstance(lfp_data, (pd.DataFrame, pd.Series)):
        lfp_np=lfp_data.values
    else:
        lfp_np=lfp_data
    #spad_np=get_detrend(spad_np)
    lags,corr=calculate_correlation (spad_np,lfp_np)
    return lags,corr

def plot_trace_nap (ax, pynapple_data,restrict_interval, color, title='LFP raw Trace'):
    ax.plot(pynapple_data.restrict(restrict_interval).as_units('s'),color=color)
    ax.set_title(title)
    ax.margins(0, 0)
    return ax

def plot_ripple_event (ax, rip_ep, rip_tsd, restrict_interval, nSS, nSS3, Low_thres):
    ax.plot(nSS.restrict(restrict_interval).as_units('s'))
    ax.plot(nSS3.restrict(rip_ep.intersect(restrict_interval)).as_units('s'), '.')
    [ax.axvline(t, color='green') for t in rip_tsd.restrict(restrict_interval).as_units('s').index.values]
    ax.margins(0, 0)
    ax.axhline(Low_thres)
    ax.set_title('Oscillation envelope')
    return ax

def plot_ripple_spectrum (ax, pynapple_data, restrict_interval,y_lim=250, Fs=30000,vmax_percentile=100):
    nperseg = 4096
    noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(pynapple_data.restrict(restrict_interval), fs=Fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    v_max = np.percentile(Sxx, vmax_percentile) 
    pcm = ax.pcolormesh(t, f, Sxx, cmap='nipy_spectral', vmin=0, vmax=v_max)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim([0, y_lim])
    ax.set_title('Spectrogram')
    return ax

def plotRippleSpectrogram(ax, lfp_raw, ripple_band_filtered, rip_ep, rip_tsd, restrict_interval, nSS, nSS3, Low_thres, y_lim=250, Fs=30000):
    ax[0].plot(lfp_raw.restrict(restrict_interval).as_units('s'))
    ax[0].set_title('LFP Trace')
    ax[0].margins(0, 0)
    
    ax[1].plot(ripple_band_filtered.restrict(restrict_interval).as_units('s'))
    ax[1].set_title('Ripple Band')
    ax[1].margins(0, 0)
    
    ax[2].plot(nSS.restrict(restrict_interval).as_units('s'))
    ax[2].plot(nSS3.restrict(rip_ep.intersect(restrict_interval)).as_units('s'), '.')
    [ax[2].axvline(t, color='green') for t in rip_tsd.restrict(restrict_interval).as_units('s').index.values]
    ax[2].margins(0, 0)
    ax[2].axhline(Low_thres)
    ax[2].set_title('Ripple envelope')
    
    nperseg = 4096
    noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(lfp_raw.restrict(restrict_interval), fs=Fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    vmax_percentile = 99  # You can adjust this percentile as needed
    v_max = np.percentile(Sxx, vmax_percentile)
    v_max = np.max(Sxx)    
    ax[3].pcolormesh(t, f, Sxx, cmap='nipy_spectral', vmin=0, vmax=v_max)
    ax[3].set_ylabel('Frequency (Hz)')
    ax[3].set_ylim([0, y_lim])
    ax[3].set_title('Spectrogram')
    #plt.colorbar(pcm, ax=ax[3])
    plt.tight_layout()
    return ax

def plotSpectrogram(ax, lfp_raw, plot_unit='WHz', nperseg=2048, y_lim=300, vmax_percentile=100, Fs=30000,showCbar=True):
    noverlap = nperseg // 2    # Overlap between adjacent segments
    f, t, Sxx = signal.spectrogram(lfp_raw, Fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    v_max = np.percentile(Sxx, vmax_percentile) # You can adjust this percentile as needed
    #v_max = np.max(Sxx)

    if (plot_unit == 'WHz'):
        pcm = ax.pcolormesh(t, f, Sxx, cmap='nipy_spectral', vmin=0, vmax=v_max)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([0, y_lim])
    else:
        pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap='nipy_spectral')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim([0, y_lim])

    ax.set_xlabel("")  # Hide x-label
    ax.set_xticks([])  # Hide x-axis tick marks
    ax.figure.tight_layout()
    if showCbar:        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)  # Adjust 'pad' to control the distance between plot and colorbar
        cbar = plt.colorbar(pcm, cax=cax)
        if plot_unit == 'WHz':
            cbar.set_label('W/Hz')
        else:
            cbar.set_label('dB')
        cbar.ax.tick_params(labelsize=8)  # Adjust color bar tick label size
    return pcm

def plotRippleEvent (lfp_raw, ripple_band_filtered, restrict_interval,nSS,nSS3,Low_thres):
	
	plt.figure(figsize=(15,5))
	plt.subplot(311)
	plt.plot(lfp_raw.restrict(restrict_interval).as_units('s'))
	plt.subplot(312)
	plt.plot(ripple_band_filtered.restrict(restrict_interval).as_units('s'))
	plt.subplot(313)
	plt.plot(nSS.restrict(restrict_interval).as_units('s'))
	plt.plot(nSS3.restrict(restrict_interval).as_units('s'), '.')
	plt.axhline(Low_thres)
	plt.xlabel("Time (s)")
	plt.tight_layout()
	plt.show()
	return -1

def plot_trace_in_seconds(data,Fs,title='Trace in seconds'):
    fig, ax = plt.subplots(figsize=(15,5))
    num_samples = len(data)
    time_seconds = np.arange(num_samples) / Fs
    ax.plot(time_seconds,data)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    plt.show()
    return -1

def plot_trace_in_seconds_ax (ax,data, Fs, label='data',color='b',ylabel='z-score',xlabel=True):
    num_samples = len(data)
    time_seconds = np.arange(num_samples) / Fs
    sns.lineplot(x=time_seconds, y=data.values, ax=ax, label=label, linewidth=2, color=color)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)    # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['left'].set_visible(False)   # Hide the left spine
    #ax.spines['bottom'].set_visible(True)  # Show the bottom spine
    if xlabel==False:
        ax.set_xticks([])  # Hide x-axis tick marks
        ax.set_xlabel([])
        ax.set_xlabel('')  # Hide x-axis label
        ax.spines['bottom'].set_visible(False)  # Show the bottom spine
    ax.set_xlim(time_seconds.min(), time_seconds.max())  # Set x-limits
    ax.legend(loc='lower right')
    return ax

def plot_timedelta_trace_in_seconds (data,ax,label='data',color='b',ylabel='z-score',xlabel=True):
    sns.lineplot(x=data.index.total_seconds(), y=data.values, ax=ax, label=label, linewidth=1, color=color)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)    # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    ax.spines['left'].set_visible(False)   # Hide the left spine
    #ax.spines['bottom'].set_visible(True)  # Show the bottom spine
    if xlabel==False:
        ax.set_xticks([])  # Hide x-axis tick marks
        ax.set_xlabel([])
        ax.set_xlabel('')  # Hide x-axis label
        ax.spines['bottom'].set_visible(False)  # Show the bottom spine
    ax.set_xlim(data.index.total_seconds().min(), data.index.total_seconds().max())  # Set x-limits
    ax.legend(loc='upper right')
    return ax

def plot_animal_tracking (trackingdata):
    # Calculate the ratio of units on X and Y axes
    x_range = trackingdata['X'].max() - trackingdata['X'].min()
    y_range = trackingdata['Y'].max() - trackingdata['Y'].min()
    aspect_ratio = y_range / x_range    
    # Set the figure size based on the aspect ratio
    fig, ax = plt.subplots(figsize=(16, 16*aspect_ratio))  # Adjust the '5' based on your preference
    # Creating an x-y scatter plot
    trackingdata.plot.scatter(x='X', y='Y', color='blue', marker='o', s=2, ax=ax)    
    # Adding labels and title
    plt.xlabel('X')
    plt.title('Animal tracking Plot')
    plt.show()
    return -1

def plot_two_traces_in_seconds (data1,Fs1, data2, Fs2, label1='optical',label2='LFP'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    num_samples_1 = len(data1)
    time_seconds_1 = np.arange(num_samples_1) / Fs1
    num_samples_2 = len(data2)
    time_seconds_2 = np.arange(num_samples_2) / Fs2
    
    sns.lineplot(x=time_seconds_1, y=data1.values, ax=ax1, label=label1, linewidth=1, color=sns.color_palette("husl", 8)[3])
    #ax1.plot(spad_resampled, label='spad')
    #ax1.set_ylabel('PhotonCount')
    ax1.set_ylabel('z-score')
    ax1.legend()
    sns.lineplot(x=time_seconds_2, y=data2.values, ax=ax2, label=label2, linewidth=1, color=sns.color_palette("husl", 8)[5])
    num_ticks = 20  # Adjust the number of ticks as needed
    ax1.xaxis.set_major_locator(MaxNLocator(num_ticks))
    ax2.set_ylabel('Amplitude')
    #ax2.set_title('LFP')
    num_ticks = 20  # Adjust the number of ticks as needed
    ax2.xaxis.set_major_locator(MaxNLocator(num_ticks))
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_two_raw_traces (data1,data2, spad_label='spad',lfp_label='LFP'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    sns.lineplot(x=data1.index, y=data1.values, ax=ax1, label=spad_label, linewidth=1, color=sns.color_palette("husl", 8)[3])
    #ax1.plot(spad_resampled, label='spad')
    #ax1.set_ylabel('PhotonCount')
    ax1.set_ylabel('z-score')
    ax1.legend()
    sns.lineplot(x=data2.index, y=data2.values, ax=ax2, label=lfp_label, linewidth=1, color=sns.color_palette("husl", 8)[5])
    ax2.set_ylabel('Amplitude')
    #ax2.set_title('LFP')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    #plt.tight_layout()
    plt.show()
    return fig
    
def plot_speed_heatmap(ax, speed_series,cbar=False,annot=False):
    speed_series = speed_series.to_frame()
    #heatmap = sns.heatmap(speed_series.transpose(), annot=annot, cmap='YlGnBu', ax=ax, cbar=cbar, yticklabels=[])
    ax.set_title("Heatmap of Animal speed over time")
    # Format x-axis labels to show seconds
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.Timedelta(x*100000).seconds))
    ax.set_xticks([])  # Hide x-axis tick marks
    ax.set_xlabel('')  # Hide x-axis label
    ax.set_ylabel('Speed')

    return -1

def plot_moving_state_heatmap(ax, speed_series,cbar=False,annot=False):
    speed_series = speed_series.to_frame()
    heatmap = sns.heatmap(speed_series.transpose(), annot=annot, cmap='YlGnBu', vmax=1, ax=ax,cbar=cbar, yticklabels=[])
    ax.set_title("Heatmap of Animal speed over time")
    # Format x-axis labels to show seconds
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.Timedelta(x*100000).seconds))
    ax.set_xticks([])  # Hide x-axis tick marks
    ax.set_xlabel('')  # Hide x-axis label
    ax.set_ylabel('Speed')

    return -1
        

def Calculate_wavelet(signal_pd,lowpassCutoff=1500,Fs=10000,scale=40):
    from waveletFunctions import wavelet
    if isinstance(signal_pd, np.ndarray)==False:
        signal=signal_pd.to_numpy()
    else:
        signal=signal_pd
    sst = butter_filter(signal, btype='low', cutoff=lowpassCutoff, fs=Fs, order=5)
    sst = sst - np.mean(sst)
    variance = np.std(sst, ddof=1) ** 2
    #print("variance = ", variance)
    # ----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E---------------
    if 0:
        variance = 1.0
        sst = sst / np.std(sst, ddof=1)
    n = len(sst)
    dt = 1/Fs

    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.25  # this will do 4 sub-octaves per octave
    s0 = scale * dt  # this says start at a scale of 10ms, use shorter scale will give you wavelet at high frequecny
    j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.1  # lag-1 autocorrelation for red noise background
    #print("lag1 = ", lag1)
    mother = 'MORLET'
    # Wavelet transform:
    wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
    frequency=1/period
    return sst,frequency,power,global_ws

def plot_wavelet(ax,sst,frequency,power,Fs=10000,colorBar=False,logbase=False):
    import matplotlib.ticker as ticker
    time = np.arange(len(sst)) /Fs   # construct time array
    level=8 #level is how many contour levels you want
    CS = ax.contourf(time, frequency, power, level)
    #ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    #ax.set_title('Wavelet Power Spectrum')
    #ax.set_xlim(xlim[:])
    if logbase:
        ax.set_yscale('log', base=2, subs=None)
    ax.set_ylim([np.min(frequency), np.max(frequency)])
    yax = plt.gca().yaxis
    yax.set_major_formatter(ticker.ScalarFormatter())
    if colorBar: 
        fig = plt.gcf()  # Get the current figure
        position = fig.add_axes([0.2, 0.01, 0.2, 0.01])
        #position = fig.add_axes()
        cbar=plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
        cbar.set_label('Power (mV$^2$)', fontsize=12) 
        #plt.subplots_adjust(right=0.7, top=0.9)              
    return -1

def plot_wavelet_feature(sst,frequency,power,global_ws,time,sst_filtered,powerband='(4-15Hz)'):
        import matplotlib.ticker as ticker
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(9, 10))
        gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                            wspace=0, hspace=0)
        plt.subplot(gs[0, 0:3])
        plt.plot(time, sst, 'k')
        #plt.xlim(xlim[:])
        plt.xlabel('Time (second)')
        plt.ylabel('Amplitude (mV)')
        plt.title('a) Local field potental (0-500Hz)')

        # --- Contour plot wavelet power spectrum
        plt3 = plt.subplot(gs[1, 0:3])
        levels = 6
        # *** or use 'contour'
        CS = plt.contourf(time, frequency, power, levels)
        #im = plt.contourf(CS, levels=levels,colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
        plt.xlabel('Time (second)')
        plt.ylabel('Frequency (Hz)')
        plt.title('b) Wavelet Power Spectrum')
        #plt.xlim(xlim[:])
        # format y-scale
        plt3.set_yscale('log', base=2, subs=None)
        plt.ylim([np.min(frequency), np.max(frequency)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt3.ticklabel_format(axis='y', style='plain')
        # set up the size and location of the colorbar
        position = fig.add_axes([0.7, 0.3, 0.3, 0.01])
        cbar = plt.colorbar(CS, cax=position, orientation='horizontal')
        cbar.set_label('Power (mV$^2$)', fontsize=10)  # Add a label to the colorbar

        # --- Plot global wavelet spectrum
        plt4 = plt.subplot(gs[1, -1])
        plt.plot(global_ws, frequency)
        #plt.plot(global_signif, frequency, '--')
        plt.xlabel('Power (mV$^2$)')
        plt.title('c) Wavelet Spectrum')
        plt.xlim([0, 1.25 * np.max(global_ws)])
        # format y-scale
        plt4.set_yscale('log', base=2, subs=None)
        plt.ylim([np.min(frequency), np.max(frequency)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt4.ticklabel_format(axis='y', style='plain')

        # --- Plot 2--8 yr scale-average time series
        plt.subplot(gs[2, 0:3])
        plt.plot(time, sst_filtered, 'k')
        #plt.xlim(xlim[:])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.title('d) Local field potental '+ powerband)

        plt.show()
        return -1
    
def plot_wavelet_feature_ripple(sst,frequency,power,global_ws,time,sst_filtered):
        import matplotlib.ticker as ticker
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(9, 10))
        gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                            wspace=0, hspace=0)
        plt.subplot(gs[0, 0:3])
        plt.plot(time, sst_filtered, 'k')
        #plt.xlim(xlim[:])
        plt.xlabel('Time (second)')
        plt.ylabel('Amplitude (mV)')
        plt.title('a) Local field potental (0-500Hz)')

        # --- Contour plot wavelet power spectrum
        plt3 = plt.subplot(gs[1, 0:3])
        levels = 6
        # *** or use 'contour'
        CS = plt.contourf(time, frequency, power, levels)
        #im = plt.contourf(CS, levels=levels,colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
        plt.xlabel('Time (second)')
        plt.ylabel('Frequency (Hz)')
        plt.title('b) Wavelet Power Spectrum')
        #plt.xlim(xlim[:])
        # format y-scale
        plt3.set_yscale('log', base=2, subs=None)
        plt.ylim([np.min(frequency), np.max(frequency)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt3.ticklabel_format(axis='y', style='plain')
        # set up the size and location of the colorbar
        position = fig.add_axes([0.7, 0.3, 0.3, 0.01])
        cbar = plt.colorbar(CS, cax=position, orientation='horizontal')
        cbar.set_label('Power (mV$^2$)', fontsize=10)  # Add a label to the colorbar

        # --- Plot global wavelet spectrum
        plt4 = plt.subplot(gs[1, -1])
        plt.plot(global_ws, frequency)
        #plt.plot(global_signif, frequency, '--')
        plt.xlabel('Power (mV$^2$)')
        plt.title('c) Wavelet Spectrum')
        plt.xlim([0, 1.25 * np.max(global_ws)])
        # format y-scale
        plt4.set_yscale('log', base=2, subs=None)
        plt.ylim([np.min(frequency), np.max(frequency)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt4.ticklabel_format(axis='y', style='plain')

        # --- Plot 2--8 yr scale-average time series
        plt.subplot(gs[2, 0:3])
        plt.plot(time, sst, 'k')
        #plt.xlim(xlim[:])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (mV)')
        plt.title('d) Local field potental (100-300Hz)')

        plt.show()
        return -1
def plot_ripple_trace(ax,time,trace_ep,color='k'):
    time=time*1000
    ax.plot(time, trace_ep, color,linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.margins(x=0)
    #ax.set_xlabel('Time (ms)')
    return -1

def plot_power_spectrum (ax,time,frequency, power,colorbar=True):
    time=time*1000
    levels = 6
    CS = ax.contourf(time, frequency, power, levels)
    if colorbar:
        cbar = plt.colorbar(CS, ax=ax)
        cbar.set_label('Power/Frequency (mV2/Hz)')  
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim([np.min(frequency), np.max(frequency)])
    return -1

def plot_two_trace_overlay(ax, time,trace1,trace2, title='Wavelet Power Spectrum',color1='black', color2='lime'):
    # Remove the plot frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    normalized_trace1 = (trace1 - trace1.min()) / (trace1.max() - trace1.min()) 
    normalized_trace2 = (trace2 - trace2.min()) / (trace2.max() - trace2.min()) 
    ax.plot(time, normalized_trace1, color1,linewidth=2)
    ax.plot(time, normalized_trace2, color2,linewidth=2)
    ax.margins(x=0)

    return ax

def plot_ripple_overlay(ax, sst, SPAD_ep, frequency, power, time, sst_filtered, title='Wavelet Power Spectrum',plotLFP=True,plotSPAD=False,plotRipple=False,plotColorMap=True):
    time=time*1000
    if plotColorMap:
        levels = 6
        CS = ax.contourf(time, frequency, power, levels)
        cbar = plt.colorbar(CS, ax=ax)
        cbar.set_label('Power/Frequency (mV2/Hz)')    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    normalized_sst = (sst - sst.min()) / (sst.max() - sst.min()) * (frequency.max() - frequency.min()) + frequency.min()
    normalized_sst_filtered = (sst_filtered - sst_filtered.min()) / (sst_filtered.max() - sst_filtered.min()) * (frequency.max() - frequency.min()) + frequency.min()
    normalized_SPAD_ep=(SPAD_ep - SPAD_ep.min()) / (SPAD_ep.max() - SPAD_ep.min()) * (frequency.max() - frequency.min()) + frequency.min()
    if plotLFP:
        ax.plot(time, normalized_sst, 'white',linewidth=2)
    if plotSPAD:
        ax.plot(time, normalized_SPAD_ep, 'lime',linewidth=2)
    if plotRipple:
        ax.plot(time, normalized_sst_filtered, 'white', linewidth=2)
    return ax

def plot_theta_overlay(ax, sst, SPAD_ep, frequency, power, time, sst_filtered, title='Wavelet Power Spectrum',plotLFP=True,plotSPAD=False,plotTheta=False):
    'sst is LFP, sst filtered is filtered LFP'
    levels = 6
    CS = ax.contourf(time, frequency, power, levels)
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    y_max=25
    ax.set_ylim([np.min(frequency), y_max])
    cbar = plt.colorbar(CS, ax=ax)
    cbar.set_label('Power/Frequency (mV2/Hz)')    

    # Remove the plot frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    normalized_sst = (sst - sst.min()) / (sst.max() - sst.min()) * (y_max - frequency.min()) + frequency.min()
    normalized_sst_filtered = (sst_filtered - sst_filtered.min()) / (sst_filtered.max() - sst_filtered.min()) * (y_max - frequency.min()) + frequency.min()
    normalized_SPAD_ep=(SPAD_ep - SPAD_ep.min()) / (SPAD_ep.max() - SPAD_ep.min()) * (y_max - frequency.min()) + frequency.min()
    if plotLFP:
        ax.plot(time, normalized_sst, 'white',linewidth=2)
    if plotSPAD:
        ax.plot(time, normalized_SPAD_ep, 'lime',linewidth=2)
    if plotTheta:
        ax.plot(time, normalized_sst_filtered, 'white', linewidth=2.5)
    return ax

def plot_theta_nested_gamma_overlay(ax, LFP_ep, SPAD_ep, frequency, power, time, sst_filtered,y_max=100,
                                    title='Wavelet Power Spectrum',plotLFP=True,plotSPAD=False,plotTheta=False,plotSpectrum=True):
    'sst is LFP, sst filtered is filtered LFP'
    y_max=y_max
    if plotSpectrum:
        levels = 6
        CS = ax.contourf(time, frequency, power, levels)
        ax.set_xlabel('Time (second)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        ax.set_ylim([np.min(frequency), y_max])
        # cbar = plt.colorbar(CS, ax=ax)

        # Remove the plot frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    normalized_sst = (LFP_ep - LFP_ep.min()) / (LFP_ep.max() - LFP_ep.min()) * (y_max - frequency.min()) + frequency.min()
    normalized_sst_filtered = (sst_filtered - sst_filtered.min()) / (sst_filtered.max() - sst_filtered.min()) * (y_max - frequency.min()) + frequency.min()
    normalized_SPAD_ep=(SPAD_ep - SPAD_ep.min()) / (SPAD_ep.max() - SPAD_ep.min()) * (y_max - frequency.min()) + frequency.min()
    if plotLFP:
        ax.plot(time, normalized_sst, 'white',linewidth=2)
    if plotSPAD:
        ax.plot(time, normalized_SPAD_ep, 'lime',linewidth=2)
    if plotTheta:
        ax.plot(time, normalized_sst_filtered, 'k', linewidth=2)
    return ax

def calculate_theta_phase_angle(channel_data, theta_low=5, theta_high=9):
    filtered_data = band_pass_filter(channel_data, low_freq=theta_low, high_freq=theta_high,Fs=10000)  # filtered in theta range
    analytic_signal = signal.hilbert(filtered_data)  # hilbert transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    angle = np.angle(analytic_signal)  # this is the theta angle (radians)
    return angle

def calculate_theta_trough_index(df,Fs=10000):
    troughs = (df['theta_angle'] < df['theta_angle'].shift(-1)) & (df['theta_angle'] < df['theta_angle'].shift(1)) & (df['theta_angle']<-3.13)
    trough_index = df.index[troughs]
    #trough_time=trough_index/Fs
    return trough_index

def plot_theta_cycle(df, LFP_channel, trough_index, half_window, fs=10000,plotmode='one'):
    half_window = half_window  # second
    # Initialize lists to store cycle data
    cycle_data_values_zscore = []
    cycle_data_values_lfp = []
    #half_cycle_time = pd.to_timedelta(half_window, unit='s')
    half_cycle_time=half_window
    # Extract A values for each cycle and calculate mean and std
    # zscore_filtered=band_pass_filter(df['zscore_raw'],4,20,fs)
    # df['zscore_raw']=zscore_filtered
    for i in range(len(trough_index)):
        start = int(trough_index[i] - half_cycle_time*fs)
        end = int(trough_index[i] + half_cycle_time*fs)
        cycle_zscore = df['zscore_raw'].loc[start:end]
        print ('length of the cycle',len(cycle_zscore))
        cycle_zscore=smooth_signal(cycle_zscore,10000,cutoff=50,window='flat')
        cycle_lfp = df[LFP_channel].loc[start:end]
        #cycle_zscore_np = cycle_zscore.to_numpy()
        cycle_zscore_np = cycle_zscore
        cycle_lfp_np = cycle_lfp.to_numpy()
        if len(cycle_lfp_np) > half_window * fs * 2:
            cycle_data_values_zscore.append(cycle_zscore_np)
            cycle_data_values_lfp.append(cycle_lfp_np)
    cycle_data_values_zscore_np = np.vstack(cycle_data_values_zscore)
    cycle_data_values_lfp_np = np.vstack(cycle_data_values_lfp)
    
    mean_zscore,std_zscore, CI_zscore=calculateStatisticNumpy (cycle_data_values_zscore_np)
    mean_lfp,std_lfp, CI_LFP=calculateStatisticNumpy (cycle_data_values_lfp_np)

    x = np.linspace(-half_window, half_window, len(mean_zscore))
    if plotmode=='two':
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 8))
        # Plot mean z-score on the first y-axis
        ax1.plot(x, mean_zscore, color=sns.color_palette("husl", 8)[3], label='Mean zscore')
        ax1.fill_between(x, CI_zscore[0], CI_zscore[1], color=sns.color_palette("husl", 8)[3], alpha=0.3,label='0.95 CI')
        ax1.axvline(x=0, color='k',linestyle='--')
        
        ax1.set_ylabel('Zscore', color='k',fontsize=16)
        ax1.set_title('Mean z-score and Mean LFP during a theta cycle')
        ax1.legend(loc='upper right',frameon=False)
        # Create a second y-axis and plot mean LFP on it
        ax2.plot(x, mean_lfp, color=sns.color_palette("husl", 8)[5], label='Mean LFP')
        ax2.fill_between(x, CI_LFP[0], CI_LFP[1], color=sns.color_palette("husl", 8)[5], alpha=0.3,label='0.95 CI')
        ax2.set_ylabel('Amplitude (μV)', color='k',fontsize=16)
        ax2.axvline(x=0, color='k',linestyle='--')
        ax2.legend(loc='upper right',frameon=False)
        ax2.set_xlabel('Time (seconds)',fontsize=14)
        # ax1.legend().set_visible(False)
        # ax2.legend().set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        #ax1.spines['bottom'].set_visible(False)
        #ax1.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.set_xlim([-0.15,0.15])
        ax2.set_xlim([-0.15,0.15])
        #ax2.spines['bottom'].set_visible(False)
        #ax2.spines['left'].set_visible(False)
        plt.show()
    return -1

def plot_gamma_power_on_theta(Fs,df, LFP_channel, trough_index, half_window):
    'plot low gamma'
    half_window = half_window  # second
    # Initialize lists to store cycle data
    cycle_data_values_zscore = []
    cycle_data_values_lfp = []
    theta_power_values=[]
    gamma_power_values=[]
    gamma_power_values_spad=[]
    theta_power_max_values=[]
    gamma_power_max_lfp_values=[]
    gamma_power_max_spad_values=[]
    sst_theta_values=[]
    sst_gamma_lfp_values=[]
    sst_gamma_spad_values=[]
    #half_cycle_time = pd.to_timedelta(half_window, unit='s')
    half_cycle_time=half_window
    # Extract A values for each cycle and calculate mean and std
    for i in range(len(trough_index)):
        start = int(trough_index[i] - half_window*Fs)
        end = int(trough_index[i] + half_window*Fs)
        cycle_zscore = df['zscore_raw'].loc[start:end]
        #print ('length of the cycle',len(cycle_zscore))
        cycle_zscore=smooth_signal(cycle_zscore,Fs,cutoff=100,window='flat')
        cycle_lfp = df[LFP_channel].loc[start:end]
        #cycle_zscore_np = cycle_zscore.to_numpy()
        cycle_zscore_np = cycle_zscore
        cycle_lfp_np = cycle_lfp.to_numpy()
        theta_band_filtered_cycle_lfp=band_pass_filter(cycle_lfp_np,low_freq=5,high_freq=10,Fs=Fs)
        sst_theta,theta_frequency,theta_power_cycle,_=Calculate_wavelet(theta_band_filtered_cycle_lfp,lowpassCutoff=50,Fs=Fs,scale=80) 
        theta_power_max=theta_power_cycle[16]
        
        gamma_band_filtered_cycle_lfp=band_pass_filter(cycle_lfp_np,low_freq=20,high_freq=50,Fs=Fs)
        sst_gamma_lfp,frequency,power_cycle,_=Calculate_wavelet(gamma_band_filtered_cycle_lfp,lowpassCutoff=100,Fs=Fs,scale=80) 
        gamma_power_max_lfp=power_cycle[8]
        
        gamma_band_filtered_cycle_spad=band_pass_filter(cycle_zscore_np,low_freq=20,high_freq=50,Fs=Fs)
        sst_gamma_spad,frequency_spad,power_cycle_spad,_=Calculate_wavelet(gamma_band_filtered_cycle_spad,lowpassCutoff=100,Fs=Fs,scale=80) 
        gamma_power_max_spad=power_cycle_spad[8]
        
        if len(cycle_lfp_np) > half_window * Fs * 2:
            cycle_data_values_zscore.append(cycle_zscore_np)
            cycle_data_values_lfp.append(cycle_lfp_np)
            theta_power_values.append(theta_power_cycle)
            gamma_power_values.append(power_cycle)
            gamma_power_values_spad.append(power_cycle_spad)
            theta_power_max_values.append(theta_power_max)
            gamma_power_max_lfp_values.append(gamma_power_max_lfp)
            gamma_power_max_spad_values.append(gamma_power_max_spad)
            sst_theta_values.append(sst_theta)
            sst_gamma_lfp_values.append(sst_gamma_lfp)
            sst_gamma_spad_values.append(sst_gamma_spad)
            
    cycle_data_values_zscore_np = np.vstack(cycle_data_values_zscore)
    cycle_data_values_lfp_np = np.vstack(cycle_data_values_lfp)
    theta_power_max_values_np=np.vstack(theta_power_max_values)
    gamma_power_max_lfp_values_np=np.vstack(gamma_power_max_lfp_values)
    gamma_power_max_spad_values_np=np.vstack(gamma_power_max_spad_values)
    sst_theta_values_np=np.vstack(sst_theta_values)
    sst_gamma_lfp_values_np=np.vstack(sst_gamma_lfp_values)
    sst_gamma_spad_values_np=np.vstack(sst_gamma_spad_values)

    mean_zscore,std_zscore, CI_zscore=calculateStatisticNumpy (cycle_data_values_zscore_np)
    mean_lfp,std_lfp, CI_LFP=calculateStatisticNumpy (cycle_data_values_lfp_np)
    average_gamma_powerSpectrum = np.mean(gamma_power_values, axis=0)
    average_gamma_powerSpectrum_spad = np.mean(gamma_power_values_spad, axis=0)
    
    mean_theta_power,std_theta_power, CI_theta_power=calculateStatisticNumpy (theta_power_max_values_np)
    mean_gamma_lfp_power,std_gamma_lfp_power, CI_gamma_lfp_power=calculateStatisticNumpy (gamma_power_max_lfp_values_np)
    mean_gamma_spad_power,std_gamma_spad_power, CI_gamma_spad_power=calculateStatisticNumpy (gamma_power_max_spad_values_np)
    mean_sst_theta,std_sst_theta, CI_sst_theta=calculateStatisticNumpy (sst_theta_values_np)
    mean_sst_gamma_lfp,std_sst_gamma_lfp, CI_sst_gamma_lfp=calculateStatisticNumpy (sst_gamma_lfp_values_np)
    mean_sst_gamma_spad,std_sst_gamma_spad, CI_sst_gamma_spad=calculateStatisticNumpy (sst_gamma_spad_values_np)
                                                             
    'plot low gamma power spectrum'             
    time = np.linspace(-half_window, half_window, len(mean_zscore))
    fig, ax = plt.subplots(2,1,figsize=(8, 10))

    plot_title='Theta nested slow gamma (LFP power 20-50Hz)'
    plot_theta_nested_gamma_overlay (ax[0],mean_lfp,mean_zscore,frequency,average_gamma_powerSpectrum,time,
                            mean_lfp,70,plot_title,plotLFP=True,plotSPAD=False,plotTheta=False)  
    
    plot_title='Theta nested slow gamma (Optical power 20-50Hz)'
    plot_theta_nested_gamma_overlay (ax[1],mean_lfp,mean_zscore,frequency_spad,average_gamma_powerSpectrum_spad,time,
                            mean_lfp,70,plot_title,plotLFP=False,plotSPAD=True,plotTheta=False)  
    
    fig, ax = plt.subplots(1,1,figsize=(8, 4))
    plot_title='Theta average for LFP and optical'
    plot_theta_nested_gamma_overlay (ax,mean_lfp,mean_zscore,frequency,average_gamma_powerSpectrum,time,
                            mean_lfp,80,plot_title,plotLFP=True,plotSPAD=True,plotTheta=True,plotSpectrum=False)  
    plt.show()
    
    'plot power amplitude trace'
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 8))
    # Plot mean z-score on the first y-axis
    ax2.plot(time, mean_gamma_spad_power, color=sns.color_palette("husl", 8)[3], label='Mean optical gamma power')
    ax2.fill_between(time, CI_gamma_spad_power[0], CI_gamma_spad_power[1], color=sns.color_palette("husl", 8)[3], alpha=0.3,label='0.95 CI')
    ax2.axvline(x=0, color='k',linestyle='--')
    
    ax2.set_ylabel('Power Amplitutde', color='k',fontsize=16)
    ax1.set_title('Mean gamma amplitude')
    ax2.legend(loc='upper right',frameon=False)
    # Create a second y-axis and plot mean LFP on it
    ax1.plot(time, mean_gamma_lfp_power, color=sns.color_palette("husl", 8)[5], label='Mean LFP gamma power')
    ax1.fill_between(time, CI_gamma_lfp_power[0], CI_gamma_lfp_power[1], color=sns.color_palette("husl", 8)[5], alpha=0.3,label='0.95 CI')
    ax1.set_ylabel('Power Amplitude', color='k',fontsize=16)
    ax1.axvline(x=0, color='k',linestyle='--')
    ax1.legend(loc='upper right',frameon=False)
    ax2.set_xlabel('Time (seconds)',fontsize=14)
    # ax1.legend().set_visible(False)
    # ax2.legend().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.show()


    'plot sst gamma band amplitude trace'
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 8))
    # Plot mean z-score on the first y-axis
    ax2.plot(time, mean_sst_gamma_spad, color=sns.color_palette("husl", 8)[3], label='Mean optical gamma power')
    ax2.fill_between(time, CI_sst_gamma_spad[0], CI_sst_gamma_spad[1], color=sns.color_palette("husl", 8)[3], alpha=0.3,label='0.95 CI')
    ax2.axvline(x=0, color='k',linestyle='--')
    
    ax2.set_ylabel('Gamma band Amplitutde', color='k',fontsize=16)
    ax1.set_title('Mean gamma amplitude (μV)')
    ax2.legend(loc='upper right',frameon=False)
    # Create a second y-axis and plot mean LFP on it
    ax1.plot(time, mean_sst_gamma_lfp, color=sns.color_palette("husl", 8)[5], label='Mean LFP gamma power')
    ax1.fill_between(time, CI_sst_gamma_lfp[0], CI_sst_gamma_lfp[1], color=sns.color_palette("husl", 8)[5], alpha=0.3,label='0.95 CI')
    ax1.set_ylabel('Gamma band Amplitude (μV)', color='k',fontsize=16)
    ax1.axvline(x=0, color='k',linestyle='--')
    ax1.legend(loc='upper right',frameon=False)
    ax2.set_xlabel('Time (seconds)',fontsize=14)
    # ax1.legend().set_visible(False)
    # ax2.legend().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    
    'plot mid gamma'
    half_window = half_window  # second
    # Initialize lists to store cycle data
    cycle_data_values_zscore = []
    cycle_data_values_lfp = []
    theta_power_values=[]
    gamma_power_values=[]
    gamma_power_values_spad=[]
    theta_power_max_values=[]
    gamma_power_max_lfp_values=[]
    gamma_power_max_spad_values=[]
    sst_theta_values=[]
    sst_gamma_lfp_values=[]
    sst_gamma_spad_values=[]
    #half_cycle_time = pd.to_timedelta(half_window, unit='s')
    half_cycle_time=half_window
    # Extract A values for each cycle and calculate mean and std
    for i in range(len(trough_index)):
        start = int(trough_index[i] - half_cycle_time*Fs)
        end = int(trough_index[i] + half_cycle_time*Fs)
        cycle_zscore = df['zscore_raw'].loc[start:end]
        #print ('length of the cycle',len(cycle_zscore))
        cycle_zscore=smooth_signal(cycle_zscore,Fs,cutoff=100,window='flat')
        cycle_lfp = df[LFP_channel].loc[start:end]
        #cycle_zscore_np = cycle_zscore.to_numpy()
        cycle_zscore_np = cycle_zscore
        cycle_lfp_np = cycle_lfp.to_numpy()
        theta_band_filtered_cycle_lfp=band_pass_filter(cycle_lfp_np,low_freq=5,high_freq=10,Fs=Fs)
        sst_theta,theta_frequency,theta_power_cycle,_=Calculate_wavelet(theta_band_filtered_cycle_lfp,lowpassCutoff=50,Fs=Fs,scale=80) 
        theta_power_max=theta_power_cycle[16]
        
        gamma_band_filtered_cycle_lfp=band_pass_filter(cycle_lfp_np,low_freq=50,high_freq=90,Fs=Fs)
        sst_gamma_lfp,frequency,power_cycle,_=Calculate_wavelet(gamma_band_filtered_cycle_lfp,lowpassCutoff=200,Fs=Fs,scale=80) 
        gamma_power_max_lfp=power_cycle[3]
        
        gamma_band_filtered_cycle_spad=band_pass_filter(cycle_zscore_np,low_freq=50,high_freq=90,Fs=Fs)
        sst_gamma_spad,frequency_spad,power_cycle_spad,_=Calculate_wavelet(gamma_band_filtered_cycle_spad,lowpassCutoff=200,Fs=Fs,scale=80) 
        gamma_power_max_spad=power_cycle_spad[3]
        
        if len(cycle_lfp_np) > half_window * Fs * 2:
            cycle_data_values_zscore.append(cycle_zscore_np)
            cycle_data_values_lfp.append(cycle_lfp_np)
            theta_power_values.append(theta_power_cycle)
            gamma_power_values.append(power_cycle)
            gamma_power_values_spad.append(power_cycle_spad)
            theta_power_max_values.append(theta_power_max)
            gamma_power_max_lfp_values.append(gamma_power_max_lfp)
            gamma_power_max_spad_values.append(gamma_power_max_spad)
            sst_theta_values.append(sst_theta)
            sst_gamma_lfp_values.append(sst_gamma_lfp)
            sst_gamma_spad_values.append(sst_gamma_spad)
            
    cycle_data_values_zscore_np = np.vstack(cycle_data_values_zscore)
    cycle_data_values_lfp_np = np.vstack(cycle_data_values_lfp)
    theta_power_max_values_np=np.vstack(theta_power_max_values)
    gamma_power_max_lfp_values_np=np.vstack(gamma_power_max_lfp_values)
    gamma_power_max_spad_values_np=np.vstack(gamma_power_max_spad_values)
    sst_theta_values_np=np.vstack(sst_theta_values)
    sst_gamma_lfp_values_np=np.vstack(sst_gamma_lfp_values)
    sst_gamma_spad_values_np=np.vstack(sst_gamma_spad_values)

    mean_zscore,std_zscore, CI_zscore=calculateStatisticNumpy (cycle_data_values_zscore_np)
    mean_lfp,std_lfp, CI_LFP=calculateStatisticNumpy (cycle_data_values_lfp_np)
    average_gamma_powerSpectrum = np.mean(gamma_power_values, axis=0)
    average_gamma_powerSpectrum_spad = np.mean(gamma_power_values_spad, axis=0)
    
    mean_theta_power,std_theta_power, CI_theta_power=calculateStatisticNumpy (theta_power_max_values_np)
    mean_gamma_lfp_power,std_gamma_lfp_power, CI_gamma_lfp_power=calculateStatisticNumpy (gamma_power_max_lfp_values_np)
    mean_gamma_spad_power,std_gamma_spad_power, CI_gamma_spad_power=calculateStatisticNumpy (gamma_power_max_spad_values_np)
    mean_sst_theta,std_sst_theta, CI_sst_theta=calculateStatisticNumpy (sst_theta_values_np)
    mean_sst_gamma_lfp,std_sst_gamma_lfp, CI_sst_gamma_lfp=calculateStatisticNumpy (sst_gamma_lfp_values_np)
    mean_sst_gamma_spad,std_sst_gamma_spad, CI_sst_gamma_spad=calculateStatisticNumpy (sst_gamma_spad_values_np)
                                                             
    'plot low gamma power spectrum'             
    time = np.linspace(-half_window, half_window, len(mean_zscore))
    fig, ax = plt.subplots(2,1,figsize=(8, 10))
    #time = np.arange(-half_cycle_time*fs,half_cycle_time*fs) *(1/Fs)
    plot_title='Theta nested mid gamma (LFP power 50-90Hz)'
    plot_theta_nested_gamma_overlay (ax[0],mean_lfp,mean_zscore,frequency,average_gamma_powerSpectrum,time,
                            mean_lfp,100,plot_title,plotLFP=True,plotSPAD=False,plotTheta=False)  
    
    plot_title='Theta nested mid gamma (Optical power 50-90Hz)'
    plot_theta_nested_gamma_overlay (ax[1],mean_lfp,mean_zscore,frequency_spad,average_gamma_powerSpectrum_spad,time,
                            mean_lfp,100,plot_title,plotLFP=False,plotSPAD=True,plotTheta=False)  
    
    fig, ax = plt.subplots(1,1,figsize=(8, 4))
    plot_title='Theta nested mid gamma (LFP 50-90Hz)'
    plot_theta_nested_gamma_overlay (ax,mean_lfp,mean_zscore,frequency,average_gamma_powerSpectrum,time,
                            mean_lfp,80,plot_title,plotLFP=True,plotSPAD=True,plotTheta=True,plotSpectrum=False)  
    plt.show()
    
    'plot power amplitude trace'
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 8))
    # Plot mean z-score on the first y-axis
    ax2.plot(time, mean_gamma_spad_power, color=sns.color_palette("husl", 8)[3], label='Mean optical gamma power')
    ax2.fill_between(time, CI_gamma_spad_power[0], CI_gamma_spad_power[1], color=sns.color_palette("husl", 8)[3], alpha=0.3,label='0.95 CI')
    ax2.axvline(x=0, color='k',linestyle='--')
    
    ax2.set_ylabel('Power Amplitutde', color='k',fontsize=16)
    ax1.set_title('Mean gamma amplitude')
    ax2.legend(loc='upper right',frameon=False)
    # Create a second y-axis and plot mean LFP on it
    ax1.plot(time, mean_gamma_lfp_power, color=sns.color_palette("husl", 8)[5], label='Mean LFP gamma power')
    ax1.fill_between(time, CI_gamma_lfp_power[0], CI_gamma_lfp_power[1], color=sns.color_palette("husl", 8)[5], alpha=0.3,label='0.95 CI')
    ax1.set_ylabel('Power Amplitude', color='k',fontsize=16)
    ax1.axvline(x=0, color='k',linestyle='--')
    ax1.legend(loc='upper right',frameon=False)
    ax2.set_xlabel('Time (seconds)',fontsize=14)
    # ax1.legend().set_visible(False)
    # ax2.legend().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.show()

    'plot sst gamma band amplitude trace'
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 8))
    # Plot mean z-score on the first y-axis
    ax2.plot(time, mean_sst_gamma_spad, color=sns.color_palette("husl", 8)[3], label='Mean optical gamma power')
    ax2.fill_between(time, CI_sst_gamma_spad[0], CI_sst_gamma_spad[1], color=sns.color_palette("husl", 8)[3], alpha=0.3,label='0.95 CI')
    ax2.axvline(x=0, color='k',linestyle='--')
    
    ax2.set_ylabel('Gamma Amplitutde (μV)', color='k',fontsize=16)
    ax1.set_title('Mean gamma amplitude')
    ax2.legend(loc='upper right',frameon=False)
    # Create a second y-axis and plot mean LFP on it
    ax1.plot(time, mean_sst_gamma_lfp, color=sns.color_palette("husl", 8)[5], label='Mean LFP gamma power')
    ax1.fill_between(time, CI_sst_gamma_lfp[0], CI_sst_gamma_lfp[1], color=sns.color_palette("husl", 8)[5], alpha=0.3,label='0.95 CI')
    ax1.set_ylabel('Gamma Amplitude (μV)', color='k',fontsize=16)
    ax1.axvline(x=0, color='k',linestyle='--')
    ax1.legend(loc='upper right',frameon=False)
    ax2.set_xlabel('Time (seconds)',fontsize=14)
    # ax1.legend().set_visible(False)
    # ax2.legend().set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax1.spines['bottom'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    return -1

def find_peak_and_std(data,half_win_len,mode='max'):
    if isinstance(data, pd.Series):
        # If data is a pandas Series
        if mode=='max':
            peak_index = data.idxmax()
        else:
            peak_index = data.idxmin()
        print ('peak_index',peak_index)
        peak_value = data.iloc[peak_index]
        window_data = data.iloc[max(0, peak_index - half_win_len):min(len(data) - 1, peak_index + half_win_len) + 1]
        peak_std = peak_value/window_data.std()
    elif isinstance(data, np.ndarray):
        # If data is a numpy array
        if mode=='max':
            peak_index = np.argmax(data)
        else:
            peak_index = np.argmin(data)
        peak_value = data[peak_index]
        window_start = max(0, peak_index - half_win_len)
        window_end = min(len(data) - 1, peak_index + half_win_len)
        window_data = data[window_start:window_end + 1]
        peak_std = peak_value/np.std(window_data)
    else:
        raise TypeError("Data type not recognized. Please provide either a pandas Series or a numpy array.") 
    return peak_value, peak_index, peak_std

def align_numpy_array_to_same_length (data):
    max_common_length = max(len(column) for column in data)
    new_data = []
    for column in data:
        if len(column) >= 0.9 * max_common_length:
            # If the length is sufficient, append the column to the new data list
            new_data.append(column)
    filtered_data = np.array(new_data,dtype=object)         

    common_length = min(len(column) for column in filtered_data)
    filtered_data = np.array([column[1:common_length-1] for column in filtered_data])
    filtered_data = filtered_data.astype(float)
    return filtered_data

def calculateStatisticNumpy (data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    sem = stats.sem(data)
    df = len(data) - 1
    moe = stats.t.ppf(0.975, df) * sem  # 0.975 for 95% confidence level (two-tailed)
    # Calculate the confidence interval
    confidence_interval = mean - moe, mean + moe
    return mean,std, confidence_interval

def getNormalised (data):
    mean = np.mean(data)
    std= np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data