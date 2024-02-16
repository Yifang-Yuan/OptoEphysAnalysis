# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:36:42 2022

OLD FUNCTIONS THAT NOT USING

@author: Yifang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import hilbert
import scipy
from scipy import interpolate


'''MOVED to SPADdemod'''

# request higher and lower envelope
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
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

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

def DemodFreqShift_Bandpass (count_value,fc_g,fc_r,fs=9938.4):
    sample_number=len(count_value)
    sample_points=np.arange(sample_number) #timeline in sample points
    print ('sample_number is', sample_number)
    print ('sampling rage is', fs)
    
    trace1=butter_filter(count_value, btype='low', cutoff=2500, fs=fs, order=10)
    print ('High frequency noise above 3kHz removed')
    # trace2=butter_filter(trace1, btype='high', cutoff=10, fs=10000, order=10)
    # print ('Low frequency noise below 10Hz removed')
    
    yf = rfft(trace1)
    xf = rfftfreq(sample_number, 1 / fs)
    
    fig, (ax1,ax2) = plt.subplots(nrows=2)
    ax1.plot(sample_points, trace1,color='b',label='Mixed')
    ax1.set_xlabel("time in frames")
    ax1.legend(loc='upper right')
    #ax1.set_xlim(0,100)
    ax2.plot(xf, np.abs(yf),color='b',label='Freq Peaks (rfft)')
    ax2.set_xlabel("frequency in Hz")
    ax2.legend(loc='upper right')
    #ax2.set_ylim(-10,1e6)
    fig.tight_layout() 

    
    '''The maximum frequency is half the sample rate'''
    points_per_freq = len(xf) / (fs/2)
    fc_g_idx = int(points_per_freq * fc_g)
    sideBand=int(points_per_freq * 300) 
    fc_r_idx = int(points_per_freq * fc_r) 
    print ('fc_green is',fc_g)
    print ('fc_red is',fc_r)
    print ('fc_g_idx is',fc_g_idx)
    print ('sideBand_idx is',sideBand)
    print ('fc_r_idx is',fc_r_idx)
  
    '''For red---bandpass filter'''
    signal_r = butter_filter(trace1, btype='high', cutoff=1800, fs=fs, order=10) 
    yf_r = rfft(signal_r)
    print ('For red channal, keep 2kHz band')    
    
    '''Hilbert transform--red'''
    analytic_signal = hilbert(signal_r)
    red_recovered = np.abs(analytic_signal)
    yf_rre = rfft(red_recovered)
    
    '''For green---bandpass filter'''
    signal_g = butter_filter(trace1, btype='high', cutoff=800, fs=fs, order=10) -signal_r
    #signal_g=trace1-signal_r
    yf_g = rfft(signal_g)
    print ('For green channal, keep 1kHz band')   
    
    '''Hilbert transform--green'''
    analytic_signal = hilbert(signal_g)
    green_recovered = np.abs(analytic_signal)
    yf_gre = rfft(green_recovered)          
    
    '''PLOT TO COMPARE'''
    fig=plotDemodFreq (mixedTrace=signal_r,envelope=red_recovered,
                    xf=xf,yfMixed=yf_r,yfEnvelope=yf_rre,color='r') 
    
    fig=plotDemodFreq (mixedTrace=signal_g,envelope=green_recovered,
                    xf=xf,yfMixed=yf_g,yfEnvelope=yf_gre,color='g')  
    
    fig=plotTwoChannel (mixed_red=signal_r,envelope_red=red_recovered,
                        mixed_green=signal_g,envelope_green=green_recovered,
                        zoomWindw=[4000,5000]) 
    
    return red_recovered,green_recovered

def DemodFreqShift (count_value,fc_g,fc_r,fs=9938.4):
    sample_number=len(count_value)
    sample_points=np.arange(sample_number) #timeline in sample points
    print ('sample_number is', sample_number)
    print ('sampling rage is', fs)
    
    yf = rfft(count_value)
    xf = rfftfreq(sample_number, 1 / fs)
    
    # fig, (ax1,ax2) = plt.subplots(nrows=2)
    # ax1.plot(sample_points, count_value,color='b',label='Mixed',linewidth=1)
    # ax1.set_xlabel("time in frames")
    # ax1.legend(loc='upper right')
    # #ax1.set_xlim(0,100)
    # ax2.plot(xf, np.abs(yf),color='b',label='Freq Peaks (rfft)',linewidth=1)
    # ax2.set_xlabel("frequency in Hz")
    # ax2.legend(loc='upper right')
    # #ax2.set_xlim(0,200)
    # #ax2.set_ylim(0,1e6)
    #fig.tight_layout() 
    '''Remove unwanted low/high frequencies and demodulation'''
    points_per_freq = len(xf) / (fs/2)
    #yf[0:int(10*points_per_freq)] = 0 
    #yf[int(3060*points_per_freq):]=0
    
    #mix = irfft(yf)
    
    # fig, (ax1,ax2) = plt.subplots(nrows=2)
    # ax1.plot(sample_points, mix,color='b',label='Cleaned Mixed signal',linewidth=1)
    # ax1.set_xlabel("time in frames")
    # ax1.legend(loc='upper right')
    # #ax1.set_xlim(0,100)
    # ax2.plot(xf, np.abs(yf),color='b',label='Freq Peaks (rfft)',linewidth=1)
    # ax2.set_xlabel("frequency in Hz")
    # ax2.legend(loc='upper right')
    # ax2.set_xlim(-10,2200)
    # fig.tight_layout() 
    # print ('High frequency noise around 4kHz removed')
    # print ('Low frequency noise removed (Hz):','[0:20]')
    yf_g = np.copy(yf)
    yf_r=np.copy(yf)
    
    '''The maximum frequency is half the sample rate'''
    fc_g_idx = int(points_per_freq * fc_g)
    sideBand=int(points_per_freq *300) 
    fc_r_idx = int(points_per_freq * fc_r) 
    print ('fc_green is',fc_g)
    print ('fc_red is',fc_r)
    print ('fc_g_idx is',fc_g_idx)
    print ('sideBand_idx is',sideBand)
    print ('fc_r_idx is',fc_r_idx)
    
    '''For red---remove unwanted band'''
    # yf_r[fc_g_idx - sideBand : fc_g_idx + sideBand] = 0 
    # signal_r = irfft(yf_r)
    # print ('For red channal, remove band:',fc_g_idx - sideBand,'to',fc_g_idx + sideBand)
    '''For red---preserve wanted band'''
    yf_r[0: fc_r_idx - sideBand] = 0 
    yf_r[fc_r_idx + sideBand : ] = 0 
    signal_r = irfft(yf_r)    
    print ('For red channal, keep band:',fc_r_idx - sideBand,'to',fc_r_idx + sideBand)    
    '''Hilbert transform--red'''
    analytic_signal = hilbert(signal_r)
    red_recovered = np.abs(analytic_signal)
    yf_rre = rfft(red_recovered)
        
    '''For green'''
    # yf_g[fc_r_idx - sideBand : fc_r_idx + sideBand] = 0 
    # signal_g = irfft(yf_g)
    # print ('For green channal, remove band:',fc_r_idx - sideBand,'to',fc_r_idx + sideBand)   
    '''For green'''
    yf_g[0: fc_g_idx - sideBand] = 0 
    yf_g[fc_g_idx + sideBand : ] = 0 
    signal_g = irfft(yf_g)
    print ('For green channal, keep band:',fc_g_idx - sideBand,'to',fc_g_idx + sideBand)   
    '''Hilbert transform--green'''
    analytic_signal = hilbert(signal_g)
    green_recovered = np.abs(analytic_signal)
    yf_gre = rfft(green_recovered)          
    
    '''PLOT TO COMPARE'''
    fig=plotDemodFreq (mixedTrace=signal_r,envelope=red_recovered,
                    xf=xf,yfMixed=yf_r,yfEnvelope=yf_rre,color='r') 
    
    fig=plotDemodFreq (mixedTrace=signal_g,envelope=green_recovered,
                    xf=xf,yfMixed=yf_g,yfEnvelope=yf_gre,color='g')  
    
    fig=plotTwoChannel (mixed_red=signal_r,envelope_red=red_recovered,
                        mixed_green=signal_g,envelope_green=green_recovered,
                        zoomWindw=[0,2000])
    return red_recovered,green_recovered

def DemodPhaseShift (count_value,fc,ini_phase=0,fs=9938.4):
    sample_number=len(count_value)
    sample_points=np.arange(sample_number) #timeline in sample points
    print ('sample_number is', sample_number)
    print ('sampling rage is', fs)
    
    yf = rfft(count_value)
    xf = rfftfreq(sample_number, 1 / fs)
    
    fig, (ax1,ax2) = plt.subplots(nrows=2)
    ax1.plot(sample_points, count_value,color='b',label='Mixed')
    ax1.set_xlabel("time in frames")
    ax1.legend(loc='upper right')
    #ax1.set_xlim(0,100)
    ax2.plot(xf, np.abs(yf),color='b',label='Freq Peaks (rfft)')
    ax2.set_xlabel("frequency in Hz")
    ax2.legend(loc='upper right')
    #ax2.set_ylim(-10,1e8)
    fig.tight_layout() 
    
    
    '''Remove unwanted low/high frequencies and demodulation'''
    yf[0:100] = 0 
    print ('Low frequency noise removed (Hz):','[0:100]')
    
    mix = irfft(yf)
    
    fig, (ax1,ax2) = plt.subplots(nrows=2)
    ax1.plot(sample_points, mix,color='b',label='Mixed')
    ax1.set_xlabel("time in frames")
    ax1.legend(loc='upper right')
    #ax1.set_xlim(0,100)
    ax2.plot(xf, np.abs(yf),color='b',label='Freq Peaks (rfft)')
    ax2.set_xlabel("frequency in Hz")
    ax2.legend(loc='upper right')
    ax2.set_xlim(995,1005)
    fig.tight_layout()
    
    '''Simulate the carrier signals'''
    t = np.arange(sample_number)/fs #timeline in seconds
    pi=np.pi 
    
    green_c=np.sin(2*pi*(fc)*t+ini_phase) #Amplitude of green is lower
    red_c=np.sin(2*pi*fc*t+pi/2+ini_phase) #red carrier is pi/2 phase shifted
    
    print ('Demodulate signals')
    green_d=mix*green_c
    red_d=mix*red_c    
    
    '''Remove unwanted frequencies and demodulation'''
    '''For red'''
    yf = rfft(red_d)
    xf = rfftfreq(sample_number, 1 / fs)
    
    yf_r=np.copy(yf)
    # #remove high frequecy sprectrum
    yf_r[250:] = 0
    #Inverse rfft
    red_recovered = irfft(yf_r)
    
    fig, (ax0, ax1,ax2) = plt.subplots(nrows=3)
    ax0.plot(sample_points, red_recovered,color='r',label='red_recovered')
    ax0.set_xlabel("time in frames")
    ax0.legend(loc='upper right')
    ax1.plot(xf, np.abs(yf), label='red mixed spectrum')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-10,1e7)

    ax2.plot(xf, np.abs(yf_r), label='red recovered spectrum')
    ax2.set_xlabel("Frequency")
    #ax2.set_xlim(-10,50)
    ax2.legend(loc='upper right')
    fig.tight_layout()
        
    '''For green'''
    yf = rfft(green_d)
    xf = rfftfreq(sample_number, 1 / fs)
     
    yf_g=np.copy(yf)
    yf_g[250:] = 0

    #Inverse rfft
    green_recovered = irfft(yf_g)
    
    fig, (ax0, ax1,ax2) = plt.subplots(nrows=3)
    ax0.plot(sample_points, green_recovered,color='g',label='green_recovered')
    ax0.set_xlabel("time in frames")
    ax0.legend(loc='upper right')
    ax1.plot(xf, np.abs(yf), label='green mixed spectrum')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-10,1e7)
    ax2.plot(xf, np.abs(yf_g), label='green recovered spectrum')
    ax2.set_xlabel("Frequency")
    #ax2.set_xlim(-2,20)
    ax2.legend(loc='upper right')
    fig.tight_layout()
    
    '''PLOT TO COMPARE'''
    # fig=plotDemodFreq (mixedTrace=signal_r,envelope=red_recovered,
    #                 xf=xf,yfMixed=yf_r,yfEnvelope=yf_rre,color='r') 
    
    # fig=plotDemodFreq (mixedTrace=signal_g,envelope=green_recovered,
    #                 xf=xf,yfMixed=yf_g,yfEnvelope=yf_gre,color='g')  
    
    # fig=plotTwoChannel (mixed_red=signal_r,envelope_red=red_recovered,
    #                     mixed_green=signal_g,envelope_green=green_recovered,
    #                     zoomWindw=[4000,6000])
    return red_recovered,green_recovered