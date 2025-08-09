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
#from tensorpac import Pac
from scipy.stats import pearsonr, spearmanr
from scipy.signal import hilbert
from scipy.signal import resample_poly
from pandas.tseries.frequencies import to_offset
from SPADPhotometryAnalysis import photometry_functions as fp
from scipy.stats import zscore

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5): 
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def band_pass_filter(data,low_freq,high_freq,Fs):
    data_high=butter_filter(data, btype='high', cutoff=low_freq,fs=Fs, order=4)
    data_low=butter_filter(data_high, btype='low', cutoff=high_freq, fs=Fs, order=4)
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
    
    # LFP1= butter_filter(LFP1, btype='low', cutoff=2000, fs=Fs, order=5)
    # LFP2= butter_filter(LFP2, btype='low', cutoff=2000, fs=Fs, order=5)
    # LFP3= butter_filter(LFP3, btype='low', cutoff=2000, fs=Fs, order=5)
    # LFP4= butter_filter(LFP4, btype='low', cutoff=2000, fs=Fs, order=5)
    # LFP_clean1= notchfilter (LFP_clean1,f0=50,bw=5)
    # LFP_clean2= notchfilter (LFP_clean2,f0=50,bw=5)
    # LFP_clean3= notchfilter (LFP_clean3,f0=50,bw=5)
    # LFP_clean4= notchfilter (LFP_clean4,f0=50,bw=5)
    
    EphysData = pd.DataFrame({
        'timestamps': timestamps,
        'CamSync': Sync1,
        'SPADSync': Sync2,
        'AtlasSync': Sync3,
        'LFP_1': LFP1,
        'LFP_2': LFP2,
        'LFP_3': LFP3,
        'LFP_4': LFP4,
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
    # LFP_clean1= notchfilter (LFP_clean1,f0=50,bw=5)
    # LFP_clean2= notchfilter (LFP_clean2,f0=50,bw=5)
    # LFP_clean3= notchfilter (LFP_clean3,f0=50,bw=5)
    # LFP_clean4= notchfilter (LFP_clean4,f0=50,bw=5)
    
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
    SPAD_mask=np.zeros(len(SPAD_Sync),dtype=int)
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
    
    Atlas_mask=np.zeros(len(Atlas_Sync),dtype=int)
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
    py_mask=np.zeros(len(Sync_line),dtype=int)
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
    py_mask_final=np.zeros(len(Sync_line),dtype=int)
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

def resample_signal(signal_df, original_fs, target_fs):
    """
    Resample a timestamp-indexed DataFrame using polyphase filtering.
    Handles both DatetimeIndex and TimedeltaIndex.

    Parameters:
        signal_df (pd.DataFrame): Timestamp-indexed signal.
        original_fs (float): Original sampling rate in Hz.
        target_fs (float): Target sampling rate in Hz.

    Returns:
        pd.DataFrame: Resampled DataFrame with timestamp index.
    """
    up = int(target_fs)
    down = int(original_fs)

    # Resample each column
    resampled_data = {
        col: resample_poly(signal_df[col].astype(float).values, up, down)
        for col in signal_df.columns
    }

    n_samples = len(next(iter(resampled_data.values())))

    # Determine proper start_time
    index = signal_df.index
    if isinstance(index, pd.TimedeltaIndex):
        start_time = pd.Timestamp(0) + index[0]
    elif isinstance(index, pd.DatetimeIndex):
        start_time = index[0]
    else:
        raise TypeError("Index must be DatetimeIndex or TimedeltaIndex.")

    # Generate new timestamp index
    freq = to_offset(pd.Timedelta(seconds=1 / target_fs))
    new_index = pd.date_range(start=start_time, periods=n_samples, freq=freq)

    return pd.DataFrame(resampled_data, index=new_index)

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
    minRipLen = 20 # ms
    maxRipLen = 200 # ms   //200ms 
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
    theta_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, 5, 9, Fs,order=2) #range 5 to 9
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
    minThetaLen = 200 # ms
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
        position = fig.add_axes([0.2, 0.02, 0.4, 0.02])
        #position = fig.add_axes()
        cbar=plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.2, pad=0.5)
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
    ax.plot(time, trace_ep, color=color,linewidth=2)
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

'Using this, phase 0 is set at theta peaks'
# def calculate_theta_phase_angle(channel_data, theta_low=5, theta_high=9):
#     filtered_data = band_pass_filter(channel_data, low_freq=theta_low, high_freq=theta_high,Fs=10000)  # filtered in theta range
#     analytic_signal = signal.hilbert(filtered_data)  # hilbert transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
#     angle = np.angle(analytic_signal)  # this is the theta angle (radians)
#     return angle

'Using this, phase 0 is set at theta troughs'
def calculate_theta_phase_angle(channel_data, theta_low=5, theta_high=9):
    filtered_data = band_pass_filter(channel_data, low_freq=theta_low, high_freq=theta_high, Fs=10000)  # Filter in theta range
    analytic_signal = signal.hilbert(filtered_data)  # Hilbert transform
    angle_default = np.angle(analytic_signal)  # Default: 0 at peak
    angle = (angle_default + np.pi) % (2 * np.pi)  # Shift so that 0 = trough
    return angle


# def calculate_theta_trough_index(df,Fs=10000):
#     troughs = (df['theta_angle'] < df['theta_angle'].shift(-1)) & (df['theta_angle'] < df['theta_angle'].shift(1)) & (df['theta_angle']<-3.12)
#     #troughs = (df['theta_angle'] < df['theta_angle'].shift(-1)) & (df['theta_angle'] < df['theta_angle'].shift(1))
#     trough_index = df.index[troughs]
#     peaks = (df['theta_angle']<0.01) & (df['theta_angle']>-0.01)
#     peak_index = df.index[peaks]
#     #trough_time=trough_index/Fs
#     return trough_index,peak_index

def calculate_theta_trough_index(df, Fs=10000):
    # Detect local minima in theta phase: trough = 0 rad
    troughs = (
        (df['theta_angle'] < df['theta_angle'].shift(-1)) &
        (df['theta_angle'] < df['theta_angle'].shift(1)) &
        ((df['theta_angle'] < 0.2) | (df['theta_angle'] > (2 * np.pi - 0.2)))  # Around 0
    )
    trough_index = df.index[troughs]

    # Detect peaks: now at π radians (≈3.14)
    peaks = (
        (df['theta_angle'] > (np.pi - 0.1)) &
        (df['theta_angle'] < (np.pi + 0.1))
    )
    peak_index = df.index[peaks]

    return trough_index, peak_index

def calculate_gamma_trough_index(df,Fs=10000):
    troughs = (df['gamma_angle'] < df['gamma_angle'].shift(-1)) & (df['gamma_angle'] < df['gamma_angle'].shift(1)) & (df['gamma_angle']<-3.12)
    trough_index = df.index[troughs]
    peaks = (df['gamma_angle']<0.01) & (df['gamma_angle']>-0.01)
    peak_index = df.index[peaks]
    #trough_time=trough_index/Fs
    return trough_index,peak_index

def calculate_ripple_trough_index(df,Fs=10000):
    troughs = (df['ripple_angle'] < df['ripple_angle'].shift(-1)) & (df['ripple_angle'] < df['ripple_angle'].shift(1)) & (df['ripple_angle']<-3.12)
    trough_index = df.index[troughs]
    peaks = (df['ripple_angle']<0.01) & (df['ripple_angle']>-0.01)
    peak_index = df.index[peaks]
    #trough_time=trough_index/Fs
    return trough_index,peak_index

# def plot_zscore_to_theta_phase (theta_angle,zscore_data):
#     # Create a polar plot of ΔF/F against theta phase
#     zscore_data=butter_filter(zscore_data, btype='low', cutoff=50, fs=10000, order=5)
#     #zscore_data=smooth_signal(zscore_data, Fs=10000, cutoff=50)
#     bins=30
#     plt.figure(figsize=(6, 6))
#     ax = plt.subplot(111, polar=True)
#     # Create a histogram of theta phases weighted by zscore_data
#     ax.hist(theta_angle, bins=bins, weights=zscore_data, color='green', alpha=0.6, edgecolor='black')
#     ax.set_title("ΔF/F vs Theta Phase (Histogram)", va='bottom')
#     plt.show()
    
#     bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     zscore_means = []

#     for i in range(len(bin_edges) - 1):
#         indices = (theta_angle >= bin_edges[i]) & (theta_angle < bin_edges[i + 1])
#         zscore_means.append(np.mean(zscore_data[indices]))

#     zscore_means = np.array(zscore_means)

#     # Close the circular data
#     zscore_means = np.append(zscore_means, zscore_means[0])
#     bin_centers = np.append(bin_centers, bin_centers[0])

#     # Create the polar plot
#     plt.figure(figsize=(6, 6))
#     ax = plt.subplot(111, polar=True)
#     ax.plot(bin_centers, zscore_means, color='green', linewidth=2)
#     #ax.fill(bin_centers, zscore_means, color='blue', alpha=0.3)
#     ax.set_title("ΔF/F vs Theta Phase (Star Plot)", va='bottom')
#     plt.show()
    

def set_polar_labels_vertical(ax):
    """ Rotates polar tick labels to be vertical to the radius. """
    angles = ax.get_xticks()
    labels = [label.get_text() for label in ax.get_xticklabels()]

    for angle, label in zip(angles, ax.get_xticklabels()):
        label.set_rotation(np.degrees(angle) + 90)  # Rotate by angle + 90° for vertical
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")

def set_sparse_polar_labels(ax, angular_ticks=[0, 180], angular_labels=["0", "π"],
                            radial_ticks=[0.5, 1.0], fontsize=14):
    """
    Set sparse angular (theta) and radial (r) tick labels on a polar axis.
    """
    # Set angular (theta) tick locations and labels
    ax.set_thetagrids(angular_ticks, labels=angular_labels)
    
    # Set radial (r) gridline positions
    ax.set_rgrids(radial_ticks, angle=0, fontsize=fontsize)

    # Set general tick label font size
    ax.tick_params(labelsize=fontsize)
    
def bootstrap_ci(data, n_boot=1000, ci=95):
    """Returns mean, lower, and upper CI bounds using bootstrapping."""
    if len(data) == 0:
        return 0, 0, 0
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

def plot_zscore_to_theta_phase(theta_angle, zscore_data):
    zscore_data = butter_filter(zscore_data, btype="low", cutoff=250, fs=10000, order=5)
    bins = 30

    # --- FIGURE 1: Histogram ---
    fig1, ax1 = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax1.hist(theta_angle, bins=bins, weights=zscore_data, color="#1b9e77", alpha=0.7, edgecolor="black")
    ax1.set_title("ΔF/F vs Phase (Histogram)", va="bottom", fontsize=14, fontweight="bold")
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(False)
    set_polar_labels_vertical(ax1)
    ax1.spines['polar'].set_linewidth(2)
    ax1.spines['polar'].set_alpha(0.5)

    # Compute means and bootstrapped 95% CIs per bin
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    zscore_means, ci_lowers, ci_uppers = [], [], []
    

    for i in range(len(bin_edges) - 1):
        mask = (theta_angle >= bin_edges[i]) & (theta_angle < bin_edges[i + 1])
        bin_data = zscore_data[mask]
        mean, lower, upper = bootstrap_ci(bin_data)
        zscore_means.append(mean)
        ci_lowers.append(lower)
        ci_uppers.append(upper)
    print ('CI_LOWER:',ci_lowers)

    # Close the loop
    bin_centers = np.append(bin_centers, bin_centers[0])
    zscore_means.append(zscore_means[0])
    ci_lowers.append(ci_lowers[0])
    ci_uppers.append(ci_uppers[0])

    # --- FIGURE 2: Line plot with CI ---
    fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    ax2.plot(bin_centers, zscore_means, color="#1b9e77", linewidth=3)
    #ax2.plot(bin_centers, zscore_means, color="tomato", linewidth=3)
    ax2.plot(0.5, 0.5, marker='o', markersize=6, color='black', transform=ax2.transAxes, zorder=10)
    theta_fill = np.concatenate([bin_centers, bin_centers[::-1]])
    radius_fill = np.concatenate([ci_uppers, ci_lowers[::-1]])
    ax2.fill(theta_fill, radius_fill, color="#1b9e77", alpha=0.3)
    #ax2.fill(theta_fill, radius_fill, color="tomato", alpha=0.3)
    
    # Define radius limits
    r_min = np.min(ci_lowers)
    r_max = np.max(ci_uppers)
    n_ticks = 4  # number of radius ticks
    
    r_ticks = np.linspace(r_min, r_max, n_ticks)
    ax2.set_yticks(r_ticks)
    ax2.set_yticklabels([f"{t:.2f}" for t in r_ticks], fontsize=14, color='darkred')  # <- change label colour here
    ax2.set_rlim(r_min, r_max)
    
    # Set angular tick label size and colour
    ax2.tick_params(axis='x', labelsize=18, colors='black')  # for angular (theta) labels
    
    ax2.set_title("-ZScore vs Phase", va="bottom", fontsize=14, fontweight="bold")
    ax2.grid(False)
    
    set_polar_labels_vertical(ax2)  # Assuming this sets angular label orientation
    ax2.spines['polar'].set_linewidth(4)
    ax2.spines['polar'].set_alpha(0.5)

    return fig1, fig2

def compute_phase_modulation_index(theta_phase,
                                   signal,
                                   Fs=None,
                                   bins: int = 30,
                                   envelope_mode: str = "zclip",
                                   min_count: int = 1,
                                   plot: bool = True):
    """
    Modulation Index (MI) between a signal’s amplitude envelope and theta phase.

    Parameters
    ----------
    theta_phase : array-like
        LFP-theta phase (radians 0–2π) for every sample, same length as `signal`.
    signal : array-like
        Optical or neural signal.
    Fs : float, optional
        Sampling rate (not used here—kept for API compatibility).
    bins : int
        Number of phase bins (20–30 typical).
    envelope_mode : {"raw", "z", "zclip"}, default "zclip"
        * "raw"   – use |Hilbert(signal)| (all positive).
        * "z"     – z-score the envelope, keep negatives **as is**.
        * "zclip" – z-score then half-wave rectify (clip <0 to 0). **Matches your old code**.
    min_count : int
        Minimum samples required in a bin. Bins with fewer are left at 0.
    plot : bool
        Draw a polar plot if True.

    Returns
    -------
    MI : float
        Modulation Index (0 = uniform, >0 = modulation).
    bin_centers : np.ndarray
        Phase bin centres (radians).
    norm_amp : np.ndarray
        Normalised amplitude distribution (probability mass) over bins.
    fig : matplotlib.figure.Figure or None
        Figure handle (None if plot=False).
    """

    theta_phase = np.asarray(theta_phase) % (2 * np.pi)
    signal      = np.asarray(signal)
    assert len(theta_phase) == len(signal), "theta_phase and signal must be same length"

    # --- 1. amplitude envelope ------------------------------------------------
    env = np.abs(hilbert(signal))

    if envelope_mode.lower() in ("z", "zclip"):
        env = zscore(env)          # centre & scale
    if envelope_mode.lower() == "zclip":
        env = np.clip(env, 0, None)

    if np.nanstd(env) == 0:
        raise ValueError("Amplitude envelope has zero variance; cannot compute MI.")

    # --- 2. phase binning -----------------------------------------------------
    bin_edges   = np.linspace(0, 2 * np.pi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    amp_binned = np.zeros(bins)
    for k in range(bins):
        mask = (theta_phase >= bin_edges[k]) & (theta_phase < bin_edges[k + 1])
        if np.sum(mask) >= min_count:
            amp_binned[k] = np.mean(env[mask])
        # else leave at 0

    # --- 3. normalise & MI ----------------------------------------------------
    if amp_binned.sum() == 0:
        raise ValueError("All bins are empty after clipping / min_count filter.")

    norm_amp = amp_binned / (amp_binned.sum() + 1e-10)
    uniform  = np.ones(bins) / bins
    kl_div   = np.nansum(norm_amp * np.log((norm_amp + 1e-10) / uniform))
    MI       = kl_div / np.log(bins)

    # --- 4. optional polar plot ----------------------------------------------
    fig = None
    if plot:
        fig = plt.figure(figsize=(6, 6))
        ax  = fig.add_subplot(111, projection='polar')
        circ_amp = np.append(norm_amp, norm_amp[0])
        circ_ctr = np.append(bin_centers, bin_centers[0])

        ax.plot(circ_ctr, circ_amp, lw=3, color="#1b9e77")
        ax.fill(circ_ctr, circ_amp, alpha=0.3, color="#1b9e77")
        ax.set_title(f"Phase Modulation (MI = {MI:.3f})",
                     va="bottom", fontsize=14, fontweight="bold")
        ax.set_yticklabels([])
        ax.tick_params(axis='x', labelsize=16)
        ax.grid(False)
        ax.spines['polar'].set_linewidth(3)
        plt.tight_layout()

    return fig, MI, bin_centers, norm_amp

def compute_optical_event_on_phase(theta_phase,
                                         signal,
                                         Fs=None,
                                         bins=30,
                                         height_factor=3.0,
                                         distance=20,
                                         prominence=None,
                                         plot=True,**kw):
    """
    Event-count Modulation Index (MI) between optical transients and theta phase.

    Parameters
    ----------
    theta_phase : array-like
        LFP-theta phase (radians 0–2π) for every sample; same length as `signal`.
    signal : array-like
        Optical trace (dF/F) sample-aligned with `theta_phase`.
    Fs : float, optional
        Sampling rate.  Only needed if you express `distance` in seconds.
    bins : int, default 30
        Number of equally spaced phase bins.
    height_factor : float, default 3.0
        Peak threshold = median + height_factor × MAD.
    distance : int or None
        Minimum number of *samples* between peaks (passed to `find_peaks`).
    prominence : float or None
        Prominence threshold for `find_peaks`.
    plot : bool, default True
        Draw a polar histogram of event probability.

    Returns
    -------
    MI : float
        Modulation Index (KL / log(bins)).
    bin_centers : np.ndarray
        Phase-bin centres (radians).
    prob : np.ndarray
        Normalised event probability per bin.
    peak_idx : np.ndarray
        Indices of detected peaks.
    fig : matplotlib.figure.Figure or None
        Figure handle (None if plot=False).
    """

    # ---- sanity -------------------------------------------------------------
    theta_phase = np.asarray(theta_phase) % (2 * np.pi)
    signal      = np.asarray(signal)

    if theta_phase.shape != signal.shape:
        raise ValueError("theta_phase and signal must be the same length")

    # ---- 1. detect peaks ----------------------------------------------------
    baseline = np.median(signal)
    mad      = median_abs_deviation(signal, scale=1.0)
    thresh   = baseline + height_factor * mad

    peak_idx, _ = find_peaks(signal,
                             height=thresh,
                             distance=distance,
                             prominence=prominence)
    #peak_idx, _ = find_peaks(signal, height=kw.get('th', np.std(signal)))

    if len(peak_idx) < 10:
        raise ValueError("Fewer than 10 peaks detected; MI not meaningful")

    # ---- 2. bin event counts -----------------------------------------------
    events_phase = theta_phase[peak_idx]
    bin_edges    = np.linspace(0, 2*np.pi, bins + 1)
    counts, _    = np.histogram(events_phase, bins=bin_edges)
    prob         = counts / counts.sum()

    # ---- 3. compute MI ------------------------------------------------------
    uniform  = np.ones(bins) / bins
    kl_div   = np.sum(prob * np.log((prob + 1e-10) / uniform))
    MI       = kl_div / np.log(bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # ---- 4. optional plot ---------------------------------------------------
    fig = None
    if plot:
        fig = plt.figure(figsize=(6, 6))
        ax  = fig.add_subplot(111, projection='polar')
        circ_prob = np.append(prob, prob[0])
        circ_ctr  = np.append(bin_centers, bin_centers[0])

        #bar histogram
        ax.bar(bin_centers, counts,
               width=(2*np.pi/bins), color="#1b9e77",
               alpha=0.7, edgecolor='black', align='center')
        
        # ax.bar(bin_centers, counts,
        #        width=(2*np.pi/bins), color="tomato",
        #        alpha=0.7, edgecolor='black', align='center')
        

         # overlay outline at full bar height
        # circ_counts = np.append(counts, counts[0])     # raw bar heights
        # ax.plot(circ_ctr, circ_counts, lw=3, color='k')

        ax.set_title(f"Event Phase Histogram (MI={MI:.3f})",
                     va="bottom", fontsize=14, fontweight="bold")
        r_max = counts.max()
        r_ticks = [0, r_max * 0.25, r_max * 0.5,r_max * 0.75, r_max]            # three ticks: 0, mid, max
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{t:.0f}" for t in r_ticks],
                           fontsize=14)          
        ax.tick_params(axis='x', labelsize=16)
        ax.grid(False)
        ax.spines['polar'].set_linewidth(3)
        plt.tight_layout()

    return fig, MI, bin_centers, prob

def rayleigh_test(phases):
    """
    Rayleigh test for non-uniformity of circular data.
    Returns: (z, p) where z = test statistic, p = p-value
    """
    n = len(phases)
    R = np.sqrt(np.sum(np.cos(phases))**2 + np.sum(np.sin(phases))**2)
    z = R**2 / n
    p = np.exp(-z) * (1 + (2*z - z**2) / (4*n) -
                      (24*z - 132*z**2 + 76*z**3 - 9*z**4) / (288*n**2))
    return z, p

def weighted_rayleigh(phases, weights):
    """weights ≥0, same length as phases"""
    C = np.sum(weights * np.cos(phases))
    S = np.sum(weights * np.sin(phases))
    R = np.sqrt(C**2 + S**2)
    W = np.sum(weights)
    z = (R**2) / W
    # p-value approximation identical to unweighted Rayleigh
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*W))
    return z, p

from scipy.signal import find_peaks
# def rayleigh_for_optical(theta_phase, optical_signal, mode='peaks', **kw):
#     if mode == 'peaks':
#         # detect transients
#         idx, _ = find_peaks(optical_signal, height=kw.get('th', np.std(optical_signal)))
#         phases = theta_phase[idx]
#         z, p = rayleigh_test(phases)
#     else:                      # 'weighted'
#         amp_env = np.abs(hilbert(optical_signal))
#         z, p = weighted_rayleigh(theta_phase, amp_env)
#     return z, p

def rayleigh_for_optical(theta_phase,
                         optical_signal,
                         mode='peaks',
                         distance=None,
                         prominence=None):
    """
    Rayleigh test for phase locking of an optical signal to LFP theta.

    Parameters
    ----------
    theta_phase : array-like
        Theta phase (radians, 0–2π) for every sample.
    optical_signal : array-like
        dF/F (or similar) trace, sample-aligned with theta_phase.
    mode : {'peaks', 'weighted'}
        'peaks'    – use peaks above median + 3×MAD (default).  
        'weighted' – use amplitude envelope as weights.
    distance : int, optional
        Minimum samples between detected peaks (passed to find_peaks).
    prominence : float, optional
        Prominence threshold for find_peaks.

    Returns
    -------
    z : float
        Rayleigh test statistic.
    p : float
        Rayleigh p-value.
    """

    theta_phase = np.asarray(theta_phase) % (2 * np.pi)
    optical_signal = np.asarray(optical_signal)

    # ---------- peak-based Rayleigh ----------------------------------------
    if mode.lower() == 'peaks':
        baseline = np.median(optical_signal)
        mad      = median_abs_deviation(optical_signal, scale=1.0)
        thresh   = baseline + 3 * mad

        peak_idx, _ = find_peaks(optical_signal,
                                 height=thresh,
                                 distance=distance,
                                 prominence=prominence)

        if len(peak_idx) == 0:
            raise ValueError("No peaks above 3×MAD detected.")

        phases = theta_phase[peak_idx]
        z, p = rayleigh_test(phases)

    # ---------- amplitude-weighted Rayleigh --------------------------------
    else:  # mode == 'weighted'
        amp_env = np.abs(hilbert(optical_signal))
        z, p = weighted_rayleigh(theta_phase, amp_env)

    return z, p
# -----------------------
# 2. Permutation test for MI
# -----------------------
from typing import Optional
from scipy.stats import median_abs_deviation 
def event_MI(theta_phase,
             optical_signal,
             bins: int = 30,
             n_perm: int = 2000,
             height_factor: float = 3.0,
             distance: Optional[int] = None,
             prominence: Optional[float] = None):
    
    """
    Modulation Index (MI) based on *event counts* per phase bin.

    Parameters
    ----------
    theta_phase : 1-D array (len N)
        LFP-theta phase in radians [0, 2π] for every sample.
    optical_signal : 1-D array (len N)
        dF/F (or other) optical trace aligned sample-for-sample with `theta_phase`.
    bins : int
        Number of equally sized phase bins (default 30 → 12° each).
    n_perm : int
        Number of circular shuffles for the permutation test (default 2 000).
    height_factor : float
        Peak threshold = median + height_factor × MAD (default 3).
    distance : int, optional
        Minimum number of samples between peaks (passed to `find_peaks`).
    prominence : float, optional
        Prominence threshold for `find_peaks`.

    Returns
    -------
    MI : float
        Modulation Index (KL / log(k)).
    p_value : float
        Permutation p-value (fraction of null MIs ≥ real MI).
    null_MI : np.ndarray
        Null-distribution (length n_perm).
    peak_idx : np.ndarray
        Indices of detected peaks in `optical_signal`.
    """
    # --- sanity checks -------------------------------------------------------
    theta_phase = np.asarray(theta_phase) % (2 * np.pi)
    optical_signal = np.asarray(optical_signal)
    assert theta_phase.shape == optical_signal.shape, "Signals must be same length"

    # --- 1. peak detection ---------------------------------------------------
    baseline = np.median(optical_signal)
    mad = median_abs_deviation(optical_signal, scale=1.0)          # raw MAD
    threshold = baseline + height_factor * mad

    peak_idx, _ = find_peaks(optical_signal,
                             height=threshold,
                             distance=distance,
                             prominence=prominence)

    if len(peak_idx) < 10:                         # too few events → no MI
        return 0.0, 1.0, np.zeros(n_perm), peak_idx

    # --- 2. bin counts and real MI ------------------------------------------
    events_phase = theta_phase[peak_idx]
    bin_edges = np.linspace(0, 2*np.pi, bins + 1)
    counts, _ = np.histogram(events_phase, bins=bin_edges)

    prob = counts / counts.sum()
    uniform = np.ones(bins) / bins
    kl_real = np.sum(prob * np.log((prob + 1e-10) / uniform))
    real_mi = kl_real / np.log(bins)

    # --- 3. permutation null -------------------------------------------------
    null_MI = np.empty(n_perm)
    N = len(theta_phase)

    for i in range(n_perm):
        shift = np.random.randint(N)          # circular shuffle
        shuff_idx = (peak_idx + shift) % N
        shuff_counts, _ = np.histogram(theta_phase[shuff_idx], bins=bin_edges)
        prob_null = shuff_counts / shuff_counts.sum()
        kl_null = np.sum(prob_null * np.log((prob_null + 1e-10) / uniform))
        null_MI[i] = kl_null / np.log(bins)

    p_value = (null_MI >= real_mi).mean()

    return real_mi, p_value, null_MI, peak_idx


# -----------------------
# 3. Wrapper function: run both tests & plot
# -----------------------
def run_phase_modulation_analysis(theta_phase, raw_data, bins=30, n_perm=1000,distance=20):
    """
    Runs Rayleigh test, MI permutation test, and plots phase vs signal scatter.
    """
    # Scatter plot for visual check
    plt.figure(figsize=(6, 4))
    plt.scatter(theta_phase, raw_data, alpha=0.4, s=10)
    plt.xlabel('Theta Phase (radians)', fontsize=12)
    plt.ylabel('Raw Signal', fontsize=12)
    plt.title('Raw Signal vs Theta Phase', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Run Rayleigh test
    rayleigh_z, rayleigh_p = rayleigh_for_optical(theta_phase, raw_data, mode='peaks')
    
    real_mi, p_value, null_mis,peaks = event_MI(theta_phase,
                               raw_data,
                               bins=bins,
                               n_perm=n_perm,
                               distance=distance)   # e.g. ≥20 ms apart

    results = {
        "rayleigh_z": rayleigh_z,
        "rayleigh_p": rayleigh_p,
        "Modulation Index": real_mi,
        "mi_p_value": p_value,
        "null_mis": null_mis[:10]
    }
    print("Rayleigh Z:", results['rayleigh_z'])
    print("Rayleigh P:  ", results['rayleigh_p'])
    print("MI p-value:  ", results['mi_p_value'])
    print("real_mi:  ", results['Modulation Index'])

    return results


def compute_optical_phase_preference(theta_phase,
                                     signal,
                                     bins=30,
                                     height_factor=3.0,
                                     distance=20,
                                     prominence=None,
                                     min_events=50,
                                     alpha=0.01,
                                     use_event_indices=None,
                                     plot=True):
    """
    Preferred theta phase & modulation depth of optical events.
    Assumes `theta_phase` spans [0, 2π) with 0 at the trough (your convention).
    """
    theta_phase = np.asarray(theta_phase) % (2*np.pi)
    signal      = np.asarray(signal)

    if theta_phase.shape != signal.shape:
        raise ValueError("theta_phase and signal must be the same length")

    # --- Detect optical events or use provided indices
    if use_event_indices is None:
        baseline = np.median(signal)
        mad      = median_abs_deviation(signal, scale=1.0)
        thresh   = baseline + height_factor * mad
        event_idx, _ = find_peaks(signal,
                                  height=thresh,
                                  distance=distance,
                                  prominence=prominence)
    else:
        event_idx = np.asarray(use_event_indices, dtype=int)

    if event_idx.size == 0:
        raise ValueError("No optical events detected/provided.")

    # --- Event phases
    event_phases = theta_phase[event_idx]

    # --- Preferred phase (mean direction) and modulation depth (R)
    C = np.mean(np.cos(event_phases))
    S = np.mean(np.sin(event_phases))
    preferred_phase = (np.arctan2(S, C)) % (2*np.pi)  # radians
    R = np.sqrt(C**2 + S**2)                          # mean resultant length

    # --- Rayleigh test
    Z, p = rayleigh_test(event_phases)

    # --- Histogram for the contour
    bin_edges   = np.linspace(0, 2*np.pi, bins + 1)
    counts, _   = np.histogram(event_phases, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if counts.max() > 0:
        r = counts / counts.max()
    else:
        r = counts.astype(float)

    theta_circ = np.append(bin_centers, bin_centers[0])
    r_circ     = np.append(r, r[0])

    # --- Plot (contour + arrow only)
    fig = None
    if plot:
        fig = plt.figure(figsize=(5.2, 5.2))
        ax  = fig.add_subplot(111, projection='polar')

        # 0° at right, anti-clockwise
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)  # positive direction anticlockwise

        # Unit circle, thicker border
        ax.spines['polar'].set_linewidth(2.5)
        ax.grid(True, linewidth=0.8, alpha=0.4)

        # Angle labels with larger font
        ax.set_thetagrids([0, 90, 180, 270],
                          labels=['0', '90', '180', '270'],
                          fontsize=14, weight='bold')

        # Radial ticks at 0.5 and 1.0
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['0.5', '1'])
        ax.tick_params(axis='y', labelsize=11)

        # Solid contour
        ax.plot(theta_circ, r_circ, linewidth=2.5)

        # Preferred-phase arrow
        arrow_len = 0.9 * R
        ax.annotate(
            "",
            xy=(preferred_phase, arrow_len),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=4)
        )

        ax.set_title(
            f"pref={np.degrees(preferred_phase):.1f}°, R={R:.3f}, p={p:.3g} (n={event_idx.size})",
            va="bottom", pad=12, fontsize=12
        )

        plt.tight_layout()

    significant = (event_idx.size >= min_events) and (p < alpha)

    return {
        'preferred_phase_rad': float(preferred_phase),
        'preferred_phase_deg': float(np.degrees(preferred_phase) % 360.0),
        'modulation_depth_R' : float(R),
        'Z'                  : float(Z),
        'p'                  : float(p),
        'significant'        : bool(significant),
        'n_events'           : int(event_idx.size),
        'event_indices'      : event_idx,
        'bin_centers'        : bin_centers,
        'counts'             : counts,
        'fig'                : fig
    }

from scipy.stats import circmean, circstd
def calculate_perferred_phase(theta_angle, zscore_data):
    """
    Finds the minimum z-score in each theta cycle, calculates the preferred theta phase,
    and plots the preferred phase with error bars (95% CI).

    Parameters:
    - theta_angle: 1D numpy array of theta phase angles (radians).
    - zscore_data: 1D numpy array of z-score values.
    """

    # Identify cycle boundaries (theta crosses from π to -π)
    cycle_starts = np.where(np.diff(theta_angle) < -np.pi)[0] + 1  # +1 to move to the next cycle start

    # Add first and last indices to ensure full cycles
    cycle_starts = np.insert(cycle_starts, 0, 0)  # First point
    cycle_starts = np.append(cycle_starts, len(theta_angle))  # Last point

    # Store minimum z-score phase in each cycle
    peak_phases = []

    for i in range(len(cycle_starts) - 1):
        start, end = cycle_starts[i], cycle_starts[i + 1]
        cycle_zscores = zscore_data[start:end]
        cycle_thetas = np.array(theta_angle[start:end])  # Convert to NumPy array

        if len(cycle_zscores) > 0:
            max_idx = np.argmax(cycle_zscores)  # Index of minimum z-score in this cycle
            peak_phases.append(cycle_thetas[max_idx])  # Store corresponding theta phase

    peak_phases = np.array(peak_phases)  # Convert to numpy array


    # Compute circular mean (preferred phase) in radians & degrees
    preferred_phase_rad = circmean(peak_phases, high=np.pi, low=-np.pi)
    preferred_phase_deg = np.degrees(preferred_phase_rad)

    # Compute circular standard deviation
    circ_std_rad = circstd(peak_phases, high=np.pi, low=-np.pi)
    circ_std_deg = np.degrees(circ_std_rad)

    # Compute standard error of the mean (SEM)
    sem_rad = circ_std_rad / np.sqrt(len(peak_phases))
    sem_deg = np.degrees(sem_rad)

    # Compute 95% confidence interval
    ci_lower = preferred_phase_deg - 1.96 * sem_deg
    ci_upper = preferred_phase_deg + 1.96 * sem_deg
    # Print results
    print(f"Preferred Phase (Degrees): {preferred_phase_deg:.2f}")
    print(f"Circular Standard Deviation (Degrees): {circ_std_deg:.2f}")
    print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    # Plot preferred phase with error bars
    plt.figure(figsize=(4, 4))
    plt.errorbar([0], [preferred_phase_deg], yerr=[[preferred_phase_deg - ci_lower], [ci_upper - preferred_phase_deg]], 
                 fmt='o', color='red', capsize=5, label=f"Preferred Phase: {preferred_phase_deg:.1f}°")
    # Formatting
    plt.axhline(0, color='grey', linestyle='--', alpha=0.7)
    plt.yticks(np.arange(-180, 181, 45))
    plt.ylabel("Theta Phase (Degrees)")
    plt.xticks([])
    plt.title("Preferred Theta Phase with 95% CI")
    plt.legend()
    plt.show()
    return preferred_phase_deg,circ_std_deg,ci_lower,ci_upper

def get_theta_cycle_value(df, LFP_channel, trough_index, half_window, fs=10000):
    df=df.reset_index(drop=True)
    half_window = half_window  # second
    # Initialize lists to store cycle data
    cycle_data_values_zscore = []
    cycle_data_values_lfp = []
    #half_cycle_time = pd.to_timedelta(half_window, unit='s')
    half_cycle_time=half_window
    'use df/f'
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(df['sig_raw'],lambda_=lambd,porder=porder,itermax=itermax) 
    sig = (df['sig_raw'] - sig_base)  
    dff_sig=100*sig / sig_base
    zscore_raw_smoothed=smooth_signal(df['zscore_raw'],fs,cutoff=50,window='flat')
    'use zscore'
    #zscore_raw_smoothed=smooth_signal(df['zscore_raw'],fs,cutoff=50,window='flat')
    if len(zscore_raw_smoothed) == len(df['zscore_raw']):
        zscore_raw_smoothed_series = pd.Series(zscore_raw_smoothed, index=df['zscore_raw'].index)
    else:
        print("Error: Mismatched lengths between the smoothed data and the DataFrame.")
    # Extract A values for each cycle and calculate mean and std
    # zscore_filtered=band_pass_filter(df['zscore_raw'],4,20,fs)
    # df['zscore_raw']=zscore_filtered
    for i in range(len(trough_index)):
        start = int(trough_index[i] - half_cycle_time*fs)
        end = int(trough_index[i] + half_cycle_time*fs)
        cycle_zscore = zscore_raw_smoothed_series.loc[start:end]

        cycle_lfp = df[LFP_channel].loc[start:end]
        cycle_zscore_np = cycle_zscore.to_numpy()
        #cycle_zscore_np = cycle_zscore
        cycle_lfp_np = cycle_lfp.to_numpy()
        if len(cycle_lfp_np) > half_window * fs * 2:
            cycle_data_values_zscore.append(cycle_zscore_np)
            cycle_data_values_lfp.append(cycle_lfp_np)
    #print(cycle_data_values_zscore)
    cycle_data_values_zscore_np = np.vstack(cycle_data_values_zscore)
    cycle_data_values_lfp_np = np.vstack(cycle_data_values_lfp)
    return cycle_data_values_zscore_np,cycle_data_values_lfp_np
    
def plot_theta_cycle(df, LFP_channel, trough_index, half_window, fs=10000,plotmode='two'):
    half_window = half_window  # second
    # Initialize lists to store cycle data
    cycle_data_values_zscore = []
    cycle_data_values_lfp = []
    #half_cycle_time = pd.to_timedelta(half_window, unit='s')
    half_cycle_time=half_window
    # Extract A values for each cycle and calculate mean and std
    # zscore_filtered=band_pass_filter(df['zscore_raw'],4,20,fs)
    # df['zscore_raw']=zscore_filtered
    #print ('trough_index[i]',trough_index)
    for i in range(len(trough_index)):
        start = int(trough_index[i] - half_cycle_time*fs)
        end = int(trough_index[i] + half_cycle_time*fs)
        cycle_zscore = df['zscore_raw'].loc[start:end]
        #print ('length of the cycle',len(cycle_zscore))
        cycle_zscore=smooth_signal(cycle_zscore,fs,cutoff=250,window='flat')
        cycle_lfp = df[LFP_channel].loc[start:end]
        #cycle_zscore_np = cycle_zscore.to_numpy()
        cycle_zscore_np = cycle_zscore
        cycle_lfp_np = cycle_lfp.to_numpy()
        if len(cycle_lfp_np) > half_window * fs * 2:
            cycle_data_values_zscore.append(cycle_zscore_np)
            cycle_data_values_lfp.append(cycle_lfp_np)
    #print(cycle_data_values_zscore)
    cycle_data_values_zscore_np = np.vstack(cycle_data_values_zscore)
    cycle_data_values_lfp_np = np.vstack(cycle_data_values_lfp)
    
    mean_zscore,std_zscore, CI_zscore=calculateStatisticNumpy (cycle_data_values_zscore_np)
    mean_lfp,std_lfp, CI_LFP=calculateStatisticNumpy (cycle_data_values_lfp_np)

    x = np.linspace(-half_window, half_window, len(mean_zscore))
    if plotmode == 'two':
        fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True, dpi=300)
        fig.patch.set_alpha(0)  # Transparent background
    
        # Plot z-score
        axs[0].plot(x, mean_zscore, color=sns.color_palette("husl", 8)[3], linewidth=2)
        axs[0].fill_between(x, CI_zscore[0], CI_zscore[1], color=sns.color_palette("husl", 8)[3], alpha=0.3)
        axs[0].axvline(x=0, color='k', linestyle='--', linewidth=1)
    
        # Plot LFP
        axs[1].plot(x, mean_lfp, color=sns.color_palette("husl", 8)[5], linewidth=2)
        axs[1].fill_between(x, CI_LFP[0], CI_LFP[1], color=sns.color_palette("husl", 8)[5], alpha=0.3)
        axs[1].axvline(x=0, color='k', linestyle='--', linewidth=1)
    
        # Formatting
        axs[0].set_title('Mean -zscore and LFP in theta cycles', fontsize=14, fontweight='bold', pad=10)
        axs[0].set_ylabel('Z-score', fontsize=12)
        axs[1].set_ylabel('Amplitude (μV)', fontsize=12)
        axs[1].set_xlabel('Time (s)', fontsize=12)
    
        # Axis limits & clean styling
        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)  # Remove top and right spines
            ax.tick_params(labelsize=10)
            ax.set_xlim([-0.15, 0.15])  # Focus on main time window
    
        axs[0].tick_params(labelbottom=False, bottom=False)  # Hide x-label on top plot
        
        plt.tight_layout()
        plt.show()
    return fig


def compute_and_plot_gamma_correlation(zscore, gamma_band, fs):
    # Compute the envelope (power) of gamma-band-filtered LFP
    gamma_power = np.abs(signal.hilbert(gamma_band))

    # Downsample zscore and gamma_power if necessary to match sampling rates
    if len(zscore) != len(gamma_power):
        min_len = min(len(zscore), len(gamma_power))
        zscore = zscore[:min_len]
        gamma_power = gamma_power[:min_len]

    # Compute Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(zscore, gamma_power)
    spearman_corr, _ = spearmanr(zscore, gamma_power)

    # Print correlation values
    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"Spearman Correlation: {spearman_corr:.3f}")

    # Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    
    # Create a hexbin plot to manage dense scatter points
    hb = plt.hexbin(gamma_power, zscore, gridsize=50, cmap='Blues', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('Count')

    # Regression line (least squares fit)
    m, b = np.polyfit(gamma_power, zscore, 1)
    x = np.linspace(np.min(gamma_power), np.max(gamma_power), 100)
    plt.plot(x, m * x + b, color='red', label=f'Regression Line (R={pearson_corr:.2f})')

    # Plot details
    plt.xlabel("Gamma Power (Envelope)")
    plt.ylabel("ΔF/F (Z-score)")
    plt.title("Correlation Between Gamma Power and ΔF/F")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
def compute_and_plot_gamma_power_crosscorr(zscore, LFP, gamma_band=(30,80), fs=10000, max_lag_s=0.5):
    # Filter for gamma band (30-80 Hz)
    zscore_gamma_band = band_pass_filter(zscore, gamma_band[0], gamma_band[1], fs)
    LFP_gamma_band = band_pass_filter(LFP, gamma_band[0], gamma_band[1], fs)

    # Compute the envelope (gamma power)
    zscore_gamma_power = np.abs(signal.hilbert(zscore_gamma_band))
    LFP_gamma_power = np.abs(signal.hilbert(LFP_gamma_band))

    # Downsample if necessary
    min_len = min(len(zscore_gamma_power), len(LFP_gamma_power))
    zscore_gamma_power = zscore_gamma_power[:min_len]
    LFP_gamma_power = LFP_gamma_power[:min_len]

    # Compute cross-correlation
    max_lag = int(max_lag_s * fs)  # convert seconds to samples
    lags = np.arange(-max_lag, max_lag + 1)
    cross_corr = signal.correlate(zscore_gamma_power - np.mean(zscore_gamma_power),
                                  LFP_gamma_power - np.mean(LFP_gamma_power),
                                  mode='full')
    cross_corr = cross_corr[len(cross_corr)//2 - max_lag : len(cross_corr)//2 + max_lag + 1]

    # Normalise
    cross_corr /= (np.std(zscore_gamma_power) * np.std(LFP_gamma_power) * min_len)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(lags / fs * 1000, cross_corr, color='purple')  # lags in milliseconds
    plt.axvline(0, color='k', linestyle='--')
    plt.title('Cross-correlation Between LFP and ΔF/F Gamma Power')
    plt.xlabel('Lag (ms)')
    plt.ylabel('Cross-correlation (r)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Find and print the lag of maximum correlation
    max_corr_idx = np.argmax(cross_corr)
    max_corr = cross_corr[max_corr_idx]
    lag_at_max_corr = lags[max_corr_idx] / fs * 1000  # convert to ms
    print(f"Max cross-correlation: {max_corr:.3f} at lag {lag_at_max_corr:.1f} ms")
    
def compute_and_plot_gamma_power_correlation(zscore, LFP,gamma_band=(30,80), fs=10000):
    # Filter for gamma band (30-80 Hz)
    zscore_gamma_band = band_pass_filter(zscore, gamma_band[0], gamma_band[1], fs)
    LFP_gamma_band = band_pass_filter(LFP, gamma_band[0], gamma_band[1], fs)

    # Compute the envelope (power) of gamma-band-filtered signals
    zscore_gamma_power = np.abs(signal.hilbert(zscore_gamma_band))
    LFP_gamma_power = np.abs(signal.hilbert(LFP_gamma_band))#
    
    # Downsample if necessary to match sampling rates
    if len(zscore_gamma_power) != len(LFP_gamma_power):
        min_len = min(len(zscore_gamma_power), len(LFP_gamma_power))
        zscore_gamma_power = zscore_gamma_power[:min_len]
        LFP_gamma_power = LFP_gamma_power[:min_len]

    # Compute Pearson and Spearman correlations
    pearson_corr, _ = pearsonr(LFP_gamma_power, zscore_gamma_power)
    spearman_corr, _ = spearmanr(LFP_gamma_power, zscore_gamma_power)

    # Print correlation values
    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"Spearman Correlation: {spearman_corr:.3f}")

    # Scatter plot with regression line
    plt.figure(figsize=(8, 6))
    
    # Create a hexbin plot to manage dense scatter points
    hb = plt.hexbin(LFP_gamma_power, zscore_gamma_power, gridsize=50, cmap='Blues', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('Count')

    # Regression line (least squares fit)
    m, b = np.polyfit(LFP_gamma_power, zscore_gamma_power, 1)
    x = np.linspace(np.min(LFP_gamma_power), np.max(LFP_gamma_power), 100)
    plt.plot(x, m * x + b, color='red', label=f'Regression Line (R={pearson_corr:.2f})')

    # Plot details
    plt.xlabel("LFP Gamma Power (Envelope)")
    plt.ylabel("ΔF/F Gamma Power (Envelope)")
    plt.title("Correlation Between LFP Gamma Power and ΔF/F Gamma Power")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
def plot_gamma_amplitude_on_theta_phase(LFP, zscore, fs, theta_band=(4, 12), gamma_band=(30, 80), bins=30):
    """Plot gamma amplitude averaged over theta phases with a polar plot, with radial axis scaled to data range."""
    # Filter LFP for theta and gamma bands
    theta_filtered = band_pass_filter(LFP, theta_band[0], theta_band[1], fs)
    gamma_filtered_LFP = band_pass_filter(LFP, gamma_band[0], gamma_band[1], fs)
    gamma_filtered_zscore = band_pass_filter(zscore, gamma_band[0], gamma_band[1], fs)

    # Compute theta phase and gamma amplitude
    theta_phase = np.angle(signal.hilbert(theta_filtered))
    gamma_amplitude_LFP = np.abs(signal.hilbert(gamma_filtered_LFP))
    gamma_amplitude_zscore = np.abs(signal.hilbert(gamma_filtered_zscore))

    # Bin the theta phase and average gamma amplitude within each bin
    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    gamma_amplitude_avg_LFP = []
    gamma_amplitude_avg_zscore = []

    for i in range(len(bin_edges) - 1):
        indices = (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
        gamma_amplitude_avg_LFP.append(np.mean(gamma_amplitude_LFP[indices]))
        gamma_amplitude_avg_zscore.append(np.mean(gamma_amplitude_zscore[indices]))

    gamma_amplitude_avg_LFP = np.array(gamma_amplitude_avg_LFP)
    gamma_amplitude_avg_zscore = np.array(gamma_amplitude_avg_zscore)

    # Close the circular data for smooth plotting
    gamma_amplitude_avg_LFP = np.append(gamma_amplitude_avg_LFP, gamma_amplitude_avg_LFP[0])
    gamma_amplitude_avg_zscore = np.append(gamma_amplitude_avg_zscore, gamma_amplitude_avg_zscore[0])
    bin_centers = np.append(bin_centers, bin_centers[0])

    # Create polar plot for LFP gamma amplitude
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(bin_centers, gamma_amplitude_avg_LFP, color='blue', linewidth=2, label='LFP Gamma Amplitude')
    ax.set_ylim(np.min(gamma_amplitude_avg_LFP), np.max(gamma_amplitude_avg_LFP))  # Dynamic range
    ax.set_title("LFP Gamma Amplitude on Theta Phase", va='bottom')
    ax.legend(loc='upper right')

    # Create polar plot for ΔF/F gamma amplitude
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(bin_centers, gamma_amplitude_avg_zscore, color='green', linewidth=2, label='ΔF/F Gamma Amplitude')
    ax.set_ylim(np.min(gamma_amplitude_avg_zscore), np.max(gamma_amplitude_avg_zscore))  # Dynamic range
    ax.set_title("ΔF/F Gamma Amplitude on Theta Phase", va='bottom')
    ax.legend(loc='upper right')
    plt.show()

def compute_gamma_MI_on_theta_phase(LFP, zscore, theta_phase, fs, theta_band=(4, 12), gamma_band=(30, 80), bins=30):
    """
    Compute and plot gamma amplitude modulation by theta phase for LFP and optical signals.
    Returns modulation indices (MI) for each.

    Parameters:
        LFP: Raw LFP signal.
        zscore: Raw ΔF/F signal (not the gamma envelope).
        fs: Sampling rate in Hz.
        theta_band: Tuple defining the theta frequency range.
        gamma_band: Tuple defining the gamma frequency range.
        bins: Number of bins for theta phase.

    Returns:
        MI_LFP, MI_DFF: Modulation indices for LFP and ΔF/F signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert, butter, filtfilt

    def band_pass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def compute_MI(theta_phase, amplitude_envelope, bins=30):
        bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        amp_binned = np.zeros(bins)

        for i in range(bins):
            mask = (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
            if np.any(mask):
                amp_binned[i] = np.mean(amplitude_envelope[mask])
            else:
                amp_binned[i] = 0

        amp_binned = np.clip(amp_binned, a_min=0, a_max=None)
        norm_amp = amp_binned / np.sum(amp_binned + 1e-10)

        # Compute Modulation Index (KL divergence from uniform)
        uniform_dist = np.ones(bins) / bins
        kl_divergence = np.sum(norm_amp * np.log((norm_amp + 1e-10) / uniform_dist))
        MI = kl_divergence / np.log(bins)
        return MI, bin_centers, norm_amp

    # === Step 1: Extract phase and envelope signals ===
    gamma_filtered_LFP = band_pass_filter(LFP, gamma_band[0], gamma_band[1], fs)
    gamma_filtered_DFF = band_pass_filter(zscore, gamma_band[0], gamma_band[1], fs)

    gamma_envelope_LFP = np.abs(hilbert(gamma_filtered_LFP))
    gamma_envelope_DFF = np.abs(hilbert(gamma_filtered_DFF))

    # === Step 2: Compute MI ===
    MI_LFP, bin_centers, norm_amp_LFP = compute_MI(theta_phase, gamma_envelope_LFP, bins)
    MI_DFF, _, norm_amp_DFF = compute_MI(theta_phase, gamma_envelope_DFF, bins)

    # === Step 3: Plot LFP polar plot ===
    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(111, polar=True)
    ax1.plot(np.append(bin_centers, bin_centers[0]),
             np.append(norm_amp_LFP, norm_amp_LFP[0]),
             linewidth=3, color="#1b9e77")
    ax1.fill(np.append(bin_centers, bin_centers[0]),
             np.append(norm_amp_LFP, norm_amp_LFP[0]),
             alpha=0.3, color="#1b9e77")
    ax1.set_title(f"LFP Gamma Modulation\nMI = {MI_LFP:.3f}", fontsize=14, fontweight="bold", va='bottom')
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=14)
    ax1.spines['polar'].set_linewidth(4)
    ax1.spines['polar'].set_alpha(0.5)

    # === Step 4: Plot ΔF/F polar plot ===
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111, polar=True)
    ax2.plot(np.append(bin_centers, bin_centers[0]),
             np.append(norm_amp_DFF, norm_amp_DFF[0]),
             linewidth=3, color="#7570b3")
    ax2.fill(np.append(bin_centers, bin_centers[0]),
             np.append(norm_amp_DFF, norm_amp_DFF[0]),
             alpha=0.3, color="#7570b3")
    ax2.set_title(f"ΔF/F Gamma Modulation\nMI = {MI_DFF:.3f}", fontsize=14, fontweight="bold", va='bottom')
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=14)
    ax2.spines['polar'].set_linewidth(4)
    ax2.spines['polar'].set_alpha(0.5)

    plt.show()

    return MI_LFP, MI_DFF

def plot_gamma_power_on_theta(Fs, df, LFP_channel, theta_peak_index, half_window, gamma_band=(30, 80), bins=18):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.signal import hilbert
    from scipy.stats import entropy

    # --- Helper function to compute MI ---
    def compute_MI(theta_phase, amplitude, bins):
        amp = amplitude.flatten()
        phase = theta_phase.flatten()

        # Bin phase
        bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_amp = np.zeros(bins)

        for i in range(bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
            if np.any(mask):
                binned_amp[i] = np.mean(amp[mask])
            else:
                binned_amp[i] = 0

        # Normalise
        p = binned_amp / np.sum(binned_amp + 1e-12)
        uniform_p = np.ones(bins) / bins
        mi = entropy(p, uniform_p) / np.log(bins)

        return mi, bin_centres, p

    # 1. Pre-compute continuous wavelet power for the entire trace
    lfp_trace = df[LFP_channel].to_numpy()
    spad_trace = df['zscore_raw'].to_numpy()

    theta_band_filtered_lfp = band_pass_filter(lfp_trace, 4, 12, Fs)
    gamma_band_filtered_lfp = band_pass_filter(lfp_trace, gamma_band[0], gamma_band[1], Fs)
    gamma_band_filtered_spad = band_pass_filter(spad_trace, gamma_band[0], gamma_band[1], Fs)

    sst_theta, theta_freqs, theta_power, _ = Calculate_wavelet(theta_band_filtered_lfp, lowpassCutoff=50, Fs=Fs, scale=80)
    sst_gamma_lfp, gamma_freqs, gamma_power_lfp, _ = Calculate_wavelet(gamma_band_filtered_lfp, lowpassCutoff=100, Fs=Fs, scale=40)
    sst_gamma_spad, gamma_freqs_spad, gamma_power_spad, _ = Calculate_wavelet(gamma_band_filtered_spad, lowpassCutoff=100, Fs=Fs, scale=40)

    gamma_amp_lfp = np.abs(hilbert(gamma_band_filtered_lfp))
    gamma_amp_spad = np.abs(hilbert(gamma_band_filtered_spad))

    # 2. Align data around theta peaks
    samples_window = int(half_window * Fs)
    time = np.linspace(-half_window, half_window, 2*samples_window)

    aligned_theta = []
    aligned_gamma_power_lfp = []
    aligned_gamma_power_spad = []
    aligned_gamma_amp_lfp = []
    aligned_gamma_amp_spad = []

    for idx in theta_peak_index:
        if idx-samples_window < 0 or idx+samples_window > len(lfp_trace):
            continue
        
        aligned_theta.append(theta_band_filtered_lfp[idx-samples_window:idx+samples_window])
        aligned_gamma_power_lfp.append(gamma_power_lfp[11, idx-samples_window:idx+samples_window])
        aligned_gamma_power_spad.append(gamma_power_spad[11, idx-samples_window:idx+samples_window])
        aligned_gamma_amp_lfp.append(gamma_amp_lfp[idx-samples_window:idx+samples_window])
        aligned_gamma_amp_spad.append(gamma_amp_spad[idx-samples_window:idx+samples_window])

    aligned_theta = np.vstack(aligned_theta)
    aligned_gamma_power_lfp = np.vstack(aligned_gamma_power_lfp)
    aligned_gamma_power_spad = np.vstack(aligned_gamma_power_spad)
    aligned_gamma_amp_lfp = np.vstack(aligned_gamma_amp_lfp)
    aligned_gamma_amp_spad = np.vstack(aligned_gamma_amp_spad)

    # 3. Compute mean + CI
    mean_theta, _, _ = calculateStatisticNumpy(aligned_theta)
    mean_gamma_power_lfp, _, CI_gamma_power_lfp = calculateStatisticNumpy(aligned_gamma_power_lfp)
    mean_gamma_power_spad, _, CI_gamma_power_spad = calculateStatisticNumpy(aligned_gamma_power_spad)

    # 4. Plot gamma power and theta (compact, frameless)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    
    color_gamma = sns.color_palette("husl", 8)[5]
    color_theta = 'black'
    color_gamma_spad = sns.color_palette("husl", 8)[3]
    
    # --- LFP gamma power ---
    ax1.plot(time, mean_gamma_power_lfp, color=color_gamma)
    ax1.fill_between(time, CI_gamma_power_lfp[0], CI_gamma_power_lfp[1], color=color_gamma, alpha=0.3)
    ax1.set_ylabel('Gamma Power (μV²)', fontsize=14, color=color_gamma)
    ax1.axvline(x=0, color='k', linestyle='--')
    
    ax1b = ax1.twinx()
    ax1b.plot(time, mean_theta, color=color_theta, linewidth=2)
    ax1b.set_ylabel('Theta-filtered LFP (μV)', fontsize=14, color=color_theta)
    
    # --- SPAD gamma power ---
    ax2.plot(time, mean_gamma_power_spad, color=color_gamma_spad)
    ax2.fill_between(time, CI_gamma_power_spad[0], CI_gamma_power_spad[1], color=color_gamma_spad, alpha=0.3)
    ax2.set_ylabel('Gamma Power (a.u.)', fontsize=14, color=color_gamma_spad)
    ax2.axvline(x=0, color='k', linestyle='--')
    
    ax2b = ax2.twinx()
    ax2b.plot(time, mean_theta, color=color_theta, linewidth=2)
    ax2b.set_ylabel('Theta-filtered LFP (μV)', fontsize=14, color=color_theta)
    ax2.set_xlabel('Time (s)', fontsize=14)

    # --- Remove frames and enlarge tick labels ---
    for ax in [ax1, ax2, ax1b, ax2b]:
        ax.tick_params(axis='both', labelsize=13)
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # 5. Compute MI
    theta_phase = np.angle(hilbert(aligned_theta, axis=1))

    mi_lfp, bin_centres, dist_lfp = compute_MI(theta_phase, aligned_gamma_power_lfp, bins)
    mi_spad, _, dist_spad = compute_MI(theta_phase, aligned_gamma_power_spad, bins)

    print(f"Modulation Index (LFP gamma): {mi_lfp:.4f}")
    print(f"Modulation Index (SPAD gamma): {mi_spad:.4f}")

    # 6. Plot polar histograms
    fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 5))
    width = (2 * np.pi) / bins
    ax_polar.bar(bin_centres, dist_lfp, width=width, alpha=0.6, label='LFP gamma power')
    ax_polar.bar(bin_centres, dist_spad, width=width, alpha=0.6, label='SPAD gamma power')
    ax_polar.set_title('Theta Phase–Gamma Power Coupling')
    ax_polar.legend(loc='upper right')
    plt.show()

    return mi_lfp, mi_spad,fig

def compute_gamma_MI_with_wavelet_power(theta_phase, gamma_power_lfp, gamma_power_spad, bins=30):
    """
    Compute and plot gamma power modulation by theta phase using wavelet-derived gamma power.
    
    Parameters:
        theta_phase: 1D array of instantaneous theta phase (0 to 2π).
        gamma_power_lfp: 1D array of wavelet gamma power for LFP.
        gamma_power_spad: 1D array of wavelet gamma power for SPAD/ΔF/F.
        bins: Number of phase bins.
    
    Returns:
        MI_LFP, MI_SPAD: Modulation indices for LFP and SPAD signals.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def compute_MI(theta_phase, amplitude, bins):
        bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        binned_amp = np.zeros(bins)

        for i in range(bins):
            mask = (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
            if np.any(mask):
                binned_amp[i] = np.mean(amplitude[mask])
            else:
                binned_amp[i] = 0

        # Normalise and compute MI using KL divergence
        binned_amp = np.clip(binned_amp, a_min=0, a_max=None)
        prob_dist = binned_amp / np.sum(binned_amp + 1e-10)
        uniform_dist = np.ones_like(prob_dist) / bins
        kl = np.sum(prob_dist * np.log((prob_dist + 1e-10) / uniform_dist))
        MI = kl / np.log(bins)
        return MI, bin_centres, prob_dist

    # Compute modulation index
    MI_LFP, bin_centres, dist_lfp = compute_MI(theta_phase, gamma_power_lfp, bins)
    MI_SPAD, _, dist_spad = compute_MI(theta_phase, gamma_power_spad, bins)

    # Plot polar plot for LFP
    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(111, polar=True)
    ax1.plot(np.append(bin_centres, bin_centres[0]),
             np.append(dist_lfp, dist_lfp[0]),
             color="#1b9e77", linewidth=2)
    ax1.fill(np.append(bin_centres, bin_centres[0]),
             np.append(dist_lfp, dist_lfp[0]),
             color="#1b9e77", alpha=0.3)
    ax1.set_title(f"LFP Gamma Power Modulation\nMI = {MI_LFP:.3f}", fontsize=13)
    ax1.set_yticklabels([])

    # Plot polar plot for SPAD
    fig2 = plt.figure(figsize=(5, 5))
    ax2 = fig2.add_subplot(111, polar=True)
    ax2.plot(np.append(bin_centres, bin_centres[0]),
             np.append(dist_spad, dist_spad[0]),
             color="#7570b3", linewidth=2)
    ax2.fill(np.append(bin_centres, bin_centres[0]),
             np.append(dist_spad, dist_spad[0]),
             color="#7570b3", alpha=0.3)
    ax2.set_title(f"SPAD Gamma Power Modulation\nMI = {MI_SPAD:.3f}", fontsize=13)
    ax2.set_yticklabels([])

    plt.show()
    return MI_LFP, MI_SPAD
def plot_gamma_power_heatmap_on_theta(Fs,df, LFP_channel, peak_index, half_window,gamma_band=(30, 80)):
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

    # Extract A values for each cycle and calculate mean and std
    for i in range(len(peak_index)):
        start = int(peak_index[i] - half_window*Fs)
        end = int(peak_index[i] + half_window*Fs)
        cycle_zscore_raw = df['zscore_raw'].loc[start:end]
        #print ('length of the cycle',len(cycle_zscore))
        cycle_zscore=smooth_signal(cycle_zscore_raw,Fs,cutoff=200,window='flat')
        cycle_zscore_theta=smooth_signal(cycle_zscore_raw,Fs,cutoff=100,window='flat')
        cycle_lfp = df[LFP_channel].loc[start:end]
        #cycle_zscore_np = cycle_zscore.to_numpy()
        cycle_zscore_np = cycle_zscore
        cycle_lfp_np = cycle_lfp.to_numpy()
        theta_band_filtered_cycle_lfp=band_pass_filter(cycle_lfp_np,low_freq=4,high_freq=12,Fs=Fs)
        sst_theta,theta_frequency,theta_power_cycle,_=Calculate_wavelet(theta_band_filtered_cycle_lfp,lowpassCutoff=50,Fs=Fs,scale=80) 
        theta_power_max=theta_power_cycle[16]
        
        gamma_band_filtered_cycle_lfp=band_pass_filter(cycle_lfp_np,low_freq=gamma_band[0],high_freq=gamma_band[1],Fs=Fs)
        sst_gamma_lfp,frequency,power_cycle,_=Calculate_wavelet(gamma_band_filtered_cycle_lfp,lowpassCutoff=100,Fs=Fs,scale=40) 
        gamma_power_max_lfp=power_cycle[8]
        
        gamma_band_filtered_cycle_spad=band_pass_filter(cycle_zscore_np,low_freq=gamma_band[0],high_freq=gamma_band[1],Fs=Fs)
        sst_gamma_spad,frequency_spad,power_cycle_spad,_=Calculate_wavelet(gamma_band_filtered_cycle_spad,lowpassCutoff=100,Fs=Fs,scale=40) 
        gamma_power_max_spad=power_cycle_spad[8]
        
        if len(cycle_lfp_np) > half_window * Fs * 2:
            cycle_data_values_zscore.append(cycle_zscore_theta)
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

    mean_zscore,std_zscore, CI_zscore=calculateStatisticNumpy (cycle_data_values_zscore_np)
    mean_lfp,std_lfp, CI_LFP=calculateStatisticNumpy (cycle_data_values_lfp_np)
    average_gamma_powerSpectrum = np.mean(gamma_power_values, axis=0)
    average_gamma_powerSpectrum_spad = np.mean(gamma_power_values_spad, axis=0)
    
                                                 
    'plot low gamma power spectrum'             
    time = np.linspace(-half_window, half_window, len(mean_zscore))
    fig, ax = plt.subplots(2,1,figsize=(8, 10))

    plot_title='Theta nested gamma(LFP)'
    plot_theta_nested_gamma_overlay (ax[0],mean_lfp,mean_zscore,frequency,average_gamma_powerSpectrum,time,
                            mean_lfp,100,plot_title,plotLFP=True,plotSPAD=False,plotTheta=False)  
    
    plot_title='Theta nested gamma(optical)'
    plot_theta_nested_gamma_overlay (ax[1],mean_lfp,mean_zscore,frequency_spad,average_gamma_powerSpectrum_spad,time,
                            mean_lfp,100,plot_title,plotLFP=False,plotSPAD=True,plotTheta=False)  
    
    fig, ax = plt.subplots(1,1,figsize=(8, 4))
    plot_title='Theta average for LFP and optical'
    plot_theta_nested_gamma_overlay (ax,mean_lfp,mean_zscore,frequency,average_gamma_powerSpectrum,time,
                            mean_lfp,100,plot_title,plotLFP=True,plotSPAD=True,plotTheta=True,plotSpectrum=False)  
    plt.show()
    
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