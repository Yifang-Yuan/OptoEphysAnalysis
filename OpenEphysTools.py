# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 10:00:55 2023
@author:Yifang
PACKAGE THAT NEED FOR THIS ANALYSIS
https://github.com/open-ephys/open-ephys-python-tools
https://github.com/pynapple-org/pynapple
https://github.com/PeyracheLab/pynacollada#getting-started
https://github.com/PeyracheLab/pynacollada/blob/main/pynacollada/PETH/Tutorial_PETH_Ripples.ipynb

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

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5): 
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def band_pass_filter(data,low_freq,high_freq,Fs):
    data_high=butter_filter(data, btype='high', cutoff=low_freq,fs=Fs, order=1)
    data_low=butter_filter(data_high, btype='low', cutoff=high_freq, fs=Fs, order=5)
    return data_low


def notchfilter (data,f0=50,bw=10,fs=30000):
    f0 = 50 # Center frequency of the notch (Hz)
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

def check_SPAD_mask_length(data):
    filtered_series = data[data == 1]
    length_of_filtered_series = len(filtered_series)
    print('length in EphysSync:', length_of_filtered_series)
    Length_in_second=length_of_filtered_series/30000
    print('length in Second:', Length_in_second)
    spad_sample_num=Length_in_second*9938.4
    print('SPAD sample numbers:', spad_sample_num)
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

def getRippleEvents (lfp_raw,Fs,windowlen=300,Low_thres=1,High_thres=10):
    ripple_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, 120, 250, Fs)
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
    maxRipLen = 200 # ms    
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
    theta_band_filtered = pyna.eeg_processing.bandpass_filter(lfp_raw, 4, 15, Fs,order=1)
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
    maxThetaLen = 2000 # ms    
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
        
    spad_1=get_detrend(spad_np)
    lags,corr=calculate_correlation (spad_1,lfp_np)
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

def plot_trace_in_seconds(data,Fs):
    fig, ax = plt.subplots(figsize=(15,5))
    num_samples = len(data)
    time_seconds = np.arange(num_samples) / Fs
    ax.plot(time_seconds,data)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (s)')
    return -1

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
    plt.yticks([])
    plt.title('Animal tracking Plot')
    plt.show()
    return -1

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
    plt.tight_layout()
    plt.show()
    return fig
    
def plot_speed_heatmap(ax, speed_series,cbar=False,annot=False):
    speed_series = speed_series.to_frame()
    heatmap = sns.heatmap(speed_series.transpose(), annot=annot, cmap='YlGnBu', ax=ax, cbar=cbar, yticklabels=[])
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

def Transient_during_LFP_event (rip_tsd,transient_trace,half_window,fs):
    event_peak_times=rip_tsd.index.to_numpy()
    transient_trace = transient_trace.reset_index(drop=True)
    half_window_len=int(half_window*fs)
    z_score_values = []
    for i in range(len(event_peak_times)):
        peak_time_index=int(event_peak_times[i]*fs)
        if peak_time_index>half_window_len and peak_time_index<len(transient_trace)-half_window_len:
            #print ('peak time is', peak_time_index)
            start_idx=peak_time_index-half_window_len
            end_idx=peak_time_index+half_window_len
            silced_recording=transient_trace[start_idx:end_idx]
            #z_score=butter_filter(silced_recording.values, btype='low', cutoff=50, fs=fs, order=5)
            z_score=smooth_signal(silced_recording,Fs=10000,cutoff=50)
            z_score_values.append(z_score)
    #print (z_score_values)
    z_score_values = np.array(z_score_values)
    mean_z_score = np.mean(z_score_values, axis=0)
    std_z_score = np.std(z_score_values, axis=0)
    x = np.linspace(-half_window, half_window, len(mean_z_score))
    plt.figure(figsize=(10, 5))
    plt.plot(x,mean_z_score, color='b', label='Mean z-score during ripple event')
    plt.fill_between(x,mean_z_score - std_z_score, mean_z_score + std_z_score, color='gray', alpha=0.3, label='Standard Deviation')
    [plt.axvline(x=0, color='green')]
    plt.xlabel('Time(seconds)')
    plt.ylabel('z-score')
    plt.title('Mean z-score during ripple event')
    plt.legend()
    plt.grid()
    plt.show()
    return mean_z_score,std_z_score
        

def Calculate_wavelet(signal_pd,lowpassCutoff=1500,Fs=10000,scale=40):
    from waveletFunctions import wavelet
    if isinstance(signal_pd, np.ndarray)==False:
        signal=signal_pd.to_numpy()
    else:
        signal=signal_pd
    sst = butter_filter(signal, btype='low', cutoff=lowpassCutoff, fs=Fs, order=5)
    sst = sst - np.mean(sst)
    variance = np.std(sst, ddof=1) ** 2
    print("variance = ", variance)
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
    print("lag1 = ", lag1)
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
    ax.set_title('Wavelet Power Spectrum')
    #ax.set_xlim(xlim[:])
    if logbase:
        ax.set_yscale('log', base=2, subs=None)
    ax.set_ylim([np.min(frequency), np.max(frequency)])
    yax = plt.gca().yaxis
    yax.set_major_formatter(ticker.ScalarFormatter())
    if colorBar: 
        fig = plt.gcf()  # Get the current figure
        position = fig.add_axes([0.2, 0.01, 0.4, 0.02])
        #position = fig.add_axes()
        cbar=plt.colorbar(CS, cax=position, orientation='horizontal', fraction=0.05, pad=0.5)
        cbar.set_label('Power (mV$^2$)', fontsize=12) 
        #plt.subplots_adjust(right=0.7, top=0.9)              
    return -1

def plot_wavelet_feature(sst,frequency,power,global_ws,time,sst_filtered):
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
        plt.title('d) Local field potental (4-15 Hz)')

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
    
def plot_ripple_overlay(ax, sst, SPAD_ep, frequency, power, time, sst_filtered, title='Wavelet Power Spectrum',plotLFP=True,plotSPAD=False,plotRipple=False):
    levels = 6
    CS = ax.contourf(time, frequency, power, levels)
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    cbar = plt.colorbar(CS, ax=ax)
    cbar.set_label('Power/Frequency (mV2/Hz)')    
    normalized_sst = (sst - sst.min()) / (sst.max() - sst.min()) * (frequency.max() - frequency.min()) + frequency.min()
    normalized_sst_filtered = (sst_filtered - sst_filtered.min()) / (sst_filtered.max() - sst_filtered.min()) * (frequency.max() - frequency.min()) + frequency.min()
    normalized_SPAD_ep=(SPAD_ep - SPAD_ep.min()) / (SPAD_ep.max() - SPAD_ep.min()) * (frequency.max() - frequency.min()) + frequency.min()
    if plotLFP:
        ax.plot(time, normalized_sst, 'white')
    if plotSPAD:
        ax.plot(time, normalized_SPAD_ep, 'lime')
    if plotRipple:
        ax.plot(time, normalized_sst_filtered, 'k', linewidth=2)
    return ax

def plot_theta_overlay(ax, sst, SPAD_ep, frequency, power, time, sst_filtered, title='Wavelet Power Spectrum',plotLFP=True,plotSPAD=False,plotTheta=False):
    levels = 6
    CS = ax.contourf(time, frequency, power, levels)
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    y_max=25
    ax.set_ylim([np.min(frequency), y_max])
    cbar = plt.colorbar(CS, ax=ax)
    cbar.set_label('Power/Frequency (mV2/Hz)')    
    normalized_sst = (sst - sst.min()) / (sst.max() - sst.min()) * (y_max - frequency.min()) + frequency.min()
    normalized_sst_filtered = (sst_filtered - sst_filtered.min()) / (sst_filtered.max() - sst_filtered.min()) * (y_max - frequency.min()) + frequency.min()
    normalized_SPAD_ep=(SPAD_ep - SPAD_ep.min()) / (SPAD_ep.max() - SPAD_ep.min()) * (y_max - frequency.min()) + frequency.min()
    if plotLFP:
        ax.plot(time, normalized_sst, 'white')
    if plotSPAD:
        ax.plot(time, normalized_SPAD_ep, 'lime')
    if plotTheta:
        ax.plot(time, normalized_sst_filtered, 'k', linewidth=2)
    return ax

def calculate_theta_phase_angle(channel_data, theta_low=5, theta_high=9):
    filtered_data = band_pass_filter(channel_data, low_freq=theta_low, high_freq=theta_high,Fs=10000)  # filtered in theta range
    analytic_signal = signal.hilbert(filtered_data)  # hilbert transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    angle = np.angle(analytic_signal)  # this is the theta angle (radians)
    return angle

def calculate_theta_trough_index(df):
    troughs = (df['theta_angle'] < df['theta_angle'].shift(-1)) & (df['theta_angle'] < df['theta_angle'].shift(1)) & (df['theta_angle']<-3.13)
    trough_index = df.index[troughs]
    return trough_index

def plot_theta_cycle(df, LFP_channel, trough_index, half_window, fs=10000,plotmode='one'):
    half_window = half_window  # second
    # Initialize lists to store cycle data
    cycle_data_values_zscore = []
    cycle_data_values_lfp = []
    half_cycle_time = pd.to_timedelta(half_window, unit='s')
    # Extract A values for each cycle and calculate mean and std
    for i in range(len(trough_index)):
        start = trough_index[i] - half_cycle_time
        end = trough_index[i] + half_cycle_time
        cycle_zscore = df['zscore_raw'].loc[start:end]
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

    mean_z_score = np.mean(cycle_data_values_zscore_np, axis=0)
    std_z_score = np.std(cycle_data_values_zscore_np, axis=0)
    mean_lfp = np.mean(cycle_data_values_lfp_np, axis=0)
    std_lfp = np.std(cycle_data_values_lfp_np, axis=0)

    x = np.linspace(-half_window, half_window, len(mean_z_score))
    if plotmode=='one':
        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 5))
    
        # Plot mean z-score on the first y-axis
        ax1.plot(x, mean_z_score, color='g', label='Mean z-score')
        ax1.fill_between(x, mean_z_score - std_z_score, mean_z_score + std_z_score, color='gray', alpha=0.3,
                         label='Standard Deviation')
        ax1.axvline(x=0, color='green')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('z-score', color='g')
        ax1.set_title('Mean z-score and Mean LFP during a theta cycle')
        ax1.legend(loc='upper left')
        ax1.grid()
    
        # Create a second y-axis and plot mean LFP on it
        ax2 = ax1.twinx()
        ax2.plot(x, mean_lfp, color='b', label='Mean LFP')
        ax2.fill_between(x, mean_lfp - std_lfp, mean_lfp + std_lfp, color='lightblue', alpha=0.3,
                         label='Standard Deviation')
        ax2.set_ylabel('Amplitude (uV)', color='b')
        ax2.legend(loc='upper right')
        plt.show()
    if plotmode=='two':
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10, 10))
        # Plot mean z-score on the first y-axis
        ax1.plot(x, mean_z_score, color='g', label='Mean z-score')
        ax1.fill_between(x, mean_z_score - std_z_score, mean_z_score + std_z_score, color='gray', alpha=0.3,                         label='Standard Deviation')
        ax1.axvline(x=0, color='green')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('z-score', color='g')
        ax1.set_title('Mean z-score and Mean LFP during a theta cycle')
        ax1.legend(loc='upper left')
        # Create a second y-axis and plot mean LFP on it
        ax2.plot(x, mean_lfp, color='b', label='Mean LFP')
        ax2.fill_between(x, mean_lfp - std_lfp, mean_lfp + std_lfp, color='lightblue', alpha=0.3,
                         label='Standard Deviation')
        ax2.set_ylabel('Amplitude (uV)', color='b')
        ax2.axvline(x=0, color='green')
        ax2.legend(loc='upper right')
        plt.show()
    return -1