# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:36:16 2022

@author: Yifang
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq
import scipy
from scipy import interpolate
from scipy.signal import find_peaks

def findMask(trace,high_thd,low_thd=0):
    mask=trace.copy()
    mask[mask>high_thd]=0
    mask[mask<low_thd]=0
    mask[mask!=0]=1
    return mask

def preserve_more_than_five_ones(mask_array):
    preserved_array = mask_array.copy()
    consecutive_ones = 0

    for i in range(len(mask_array)):
        if mask_array[i] == 1:
            consecutive_ones += 1
        else:
            if consecutive_ones < 5:
                preserved_array[i - consecutive_ones:i] = 0
            consecutive_ones = 0

    # Handle the last sequence if it ends with 1s
    if consecutive_ones > 3:
        preserved_array[len(mask_array) - consecutive_ones:] = 1

    return preserved_array

def preserve_fewer_than_four_ones(mask_array):
    preserved_array = mask_array.copy()
    consecutive_ones = 0

    for i in range(len(mask_array)):
        if mask_array[i] == 1:
            consecutive_ones += 1
        else:
            if consecutive_ones > 0 and consecutive_ones <= 4:
                preserved_array[i - consecutive_ones:i] = 1
            else:
                preserved_array[i - consecutive_ones:i] = 0
            consecutive_ones = 0

    # Handle the last sequence if it ends with 1s
    if consecutive_ones > 0 and consecutive_ones < 3:
        preserved_array[len(mask_array) - consecutive_ones:] = 1
    else:
        preserved_array[len(mask_array) - consecutive_ones:] = 0

    return preserved_array


def findTraceFromMask(trace,mask):
    non_signal_index=np.where(mask==0)[0]
    trace_temp=trace.copy()
    trace_temp[non_signal_index]=0
    peaks, _ = find_peaks(trace_temp)
    envelope = np.zeros_like(trace_temp)
    envelope[peaks] = trace[peaks]
    return peaks,envelope


def ShowRFFT(count_value,fs=9938.4):
    sample_number=len(count_value)
    print ('sample_number is', sample_number)
    
    yf = rfft(count_value)
    xf = rfftfreq(sample_number, 1 / fs)
    #plt.plot(xf, yf)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xf, np.abs(yf))
    return fig, ax

def hl_envelopes_max(s, dmin=1, dmax=1, split=True):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal     
        # pre-sorting of locals min based on relative position with respect to s_mid 
        #lmin = lmin[s[lmin]<800]
        #lmin = lmin[s[lmin]>100]
        lmin = lmax[s[lmax]<1000]
        lmin = lmin[s[lmin]>200]
        # pre-sorting of localbased on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>2700]
        lmax = lmax[s[lmax]<4000]

    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def Find_targetPeaks(s, dmin=1, dmax=1, high_limit=4000, low_limit=2700):
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    lmax = lmax[s[lmax]>low_limit]
    lmax = lmax[s[lmax]<high_limit]
    
    lmin = lmax[s[lmax]<500]
    lmin = lmin[s[lmin]>100]
 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    return lmin, lmax

def Interpolate_timeDiv (Index,trace):
    x=Index
    y=trace[Index]
    if Index[0]!=0:
        x = np.concatenate( ([0], x ))
        y = np.concatenate( ([y[0]], y ) )
    if Index[-1]!=len(trace)-1:
        x = np.concatenate( (x,[len(trace)-1]) )
        y = np.concatenate( (y ,[y[-1]]) )
    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, len(trace), 1)
    ynew = f(xnew)   # use interpolation function returned by `interp1d`
    return xnew, ynew

def plotDemodFreq (mixedTrace,envelope,xf,yfMixed,yfEnvelope,color):
    '''
    Plot 
    1.Separated single channel mixed signal.
    2.Single channel spectrum.
    3.Zoomed window of modulated signal and envelope
    4.Spectrum of recovered signal
    '''
    sample_points=np.arange(len(mixedTrace)) #timeline in sample points    
    
    fig, ((ax0, ax2), (ax1,ax3)) = plt.subplots(2,2)
    ax0.plot(sample_points, mixedTrace, label='Modulated',linewidth=1)
    ax0.plot(sample_points, envelope, color=color,label='Envelope',linewidth=1)
    ax0.legend(loc='best')
    
    ax1.plot(sample_points, mixedTrace, label='Modulated',linewidth=1)
    ax1.plot(sample_points, envelope,color=color,label='Envelope',linewidth=1)
    ax1.set_xlim(2000,2200)
    ax1.set_xlabel("Time in frames")
    ax1.legend(loc='best')
    
    ax2.plot(xf, np.abs(yfMixed), label='Modulated spectrum',linewidth=1)
    ax2.legend(loc='best')
    
    ax3.plot(xf, np.abs(yfEnvelope), label='Envelope spectrum',color=color,linewidth=1)
    ax3.set_xlabel("Frequency")
    ax3.legend(loc='best')
    ax3.set_xlim(-10,100)    
    fig.tight_layout()
    return fig

def plotTwoChannel (mixed_red,envelope_red,mixed_green,envelope_green,zoomWindw=[2000,2200]):
    '''
    Plot Two Color
    '''
    sample_points=np.arange(len(mixed_red)) #timeline in sample points    
    
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=4)
    ax0.plot(sample_points, envelope_red, color='r',label='Envelope',linewidth=1)
    #ax0.legend(loc='best')
    
    ax1.plot(sample_points, envelope_red,color='r',label='Zoomed Envelope',linewidth=1)
    ax1.set_xlim(zoomWindw)
    ax1.legend(loc='upper right')
    
    ax2.plot(sample_points, envelope_green,color='g',label='Envelope',linewidth=1)    
    
    ax3.plot(sample_points, envelope_green,color='g',label='Zoomed Envelope',linewidth=1)
    ax3.legend(loc='upper right')
    ax3.set_xlim(zoomWindw)   
    ax3.set_xlabel("Time in frames")
    
    fig.tight_layout()
    return fig


def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=10):   
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = scipy.signal.filtfilt(b, a, data, axis=0)
    return y

def DemodFreqShift (count_value,fc_g,fc_r,fs=9938.4):
    sample_number=len(count_value)
    #sample_points=np.arange(sample_number) #timeline in sample points
    print ('sample_number is', sample_number)
    print ('sampling rage is', fs)
    
    yf = rfft(count_value)
    xf = rfftfreq(sample_number, 1 / fs)
    
    '''Remove unwanted low/high frequencies and demodulation'''
    points_per_freq = len(xf) / (fs/2)
    #yf[0:int(10*points_per_freq)] = 0 
    #yf[int(3060*points_per_freq):]=0
    #mix = irfft(yf)
    
    yf_g = np.copy(yf)
    yf_r=np.copy(yf)
    
    '''The maximum frequency is half the sample rate'''
    fc_g_idx = int(points_per_freq * fc_g)
    sideBand=int(points_per_freq *250) 
    fc_r_idx = int(points_per_freq * fc_r) 
    print ('fc_green is',fc_g)
    print ('fc_red is',fc_r)
    print ('fc_g_idx is',fc_g_idx)
    print ('sideBand_idx is',sideBand)
    print ('fc_r_idx is',fc_r_idx)
    
    '''For red---preserve wanted band'''
    yf_r[0: fc_r_idx - sideBand] = 0 
    yf_r[fc_r_idx + sideBand : ] = 0 
    signal_r = irfft(yf_r)    
    print ('For red channal, keep band:',fc_r_idx - sideBand,'to',fc_r_idx + sideBand)   
    
    '''Envelope--red'''
    lmin,lmax=hl_envelopes_max(signal_r, dmin=1, dmax=1, split=False)
    xnew, red_recovered=Interpolate_timeDiv (lmax,signal_r)
        
    '''For green'''
    yf_g[0: fc_g_idx - sideBand] = 0 
    yf_g[fc_g_idx + sideBand : ] = 0 
    signal_g = irfft(yf_g)
    print ('For green channal, keep band:',fc_g_idx - sideBand,'to',fc_g_idx + sideBand)   
    '''Hilbert transform--green'''
    lmin,lmax=hl_envelopes_max(signal_g, dmin=1, dmax=1, split=False)
    xnew, green_recovered=Interpolate_timeDiv (lmax,signal_g)          
    
    '''PLOT TO COMPARE'''    
    fig=plotTwoChannel (mixed_red=signal_r,envelope_red=red_recovered,
                        mixed_green=signal_g,envelope_green=green_recovered,
                        zoomWindw=[0,500])
    
    return red_recovered,green_recovered

def DemodFreqShift_bandpass (count_value,fc_g,fc_r,fs=9938.4):
    mix_g=butter_filter(count_value, btype='low', cutoff=fc_g+250, fs=fs, order=5)
    mix_g=butter_filter(mix_g, btype='high', cutoff=fc_g-250, fs=fs, order=5)
    xcorr = lambda i: np.correlate(np.roll(mix_g,i), mix_g)
    lags = np.arange(int(-0.5*fs/fc_g),int(0.5*fs/fc_g))
    lag = lags[np.argmax([xcorr(s) for s in lags])]
    # Demodulate signal by multiplication with lagged modulation then lowpass filtering.
    mixed_green = np.roll(mix_g,lag)*mix_g

    mix_r=butter_filter(count_value, btype='low', cutoff=fc_r+250, fs=fs, order=5)
    mix_r=butter_filter(mix_r, btype='high', cutoff=fc_r-250, fs=fs, order=5)
    xcorr = lambda i: np.correlate(np.roll(mix_r,i), mix_r)
    lags = np.arange(int(-0.5*fs/fc_r),int(0.5*fs/fc_r))
    lag = lags[np.argmax([xcorr(s) for s in lags])]
    # Demodulate signal by multiplication with lagged modulation then lowpass filtering.
    mixed_red = np.roll(mix_r,lag)*mix_r

    green_recovered=butter_filter(mixed_green, btype='low', cutoff=200, fs=fs, order=5)
    red_recovered=butter_filter(mixed_red, btype='low', cutoff=200, fs=fs, order=5)
  
    
    '''PLOT TO COMPARE'''    
    fig=plotTwoChannel (mixed_red=mixed_red,envelope_red=red_recovered,
                        mixed_green=mixed_green,envelope_green=green_recovered,
                        zoomWindw=[0,10000])
    return red_recovered,green_recovered

'''Time Division'''
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]#
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax



def main():
    dpath="C:/SPAD/SPADData/20220423/1454214_g1r2_2022_4_23_13_48_56"
    filename = os.path.join(dpath, "traceValue1.csv")  #csv file is the file contain values for each frame
    count_value = np.genfromtxt(filename, delimiter=',')
    '''PLOT the trace'''
    plt.figure(figsize=(15, 4))
    plt.plot(count_value,linewidth=1)
    plt.title("trace")
    '''PLOT the details of the trace'''
    plt.figure()
    plt.plot(count_value,linewidth=1)
    #plt.xlim(2000,2200)
    
    '''Using freq shift modulation'''
    red_recovered,green_recovered=DemodFreqShift (count_value,fc_g=1000,fc_r=2000,fs=9938.4)
    '''Using phase shift modulation'''
    #red_recovered,green_recovered=DemodPhaseShift (count_value,fc=1000,ini_phase=(-0.848* np.pi))
    
    return -1

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
        
        