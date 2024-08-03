# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:12:47 2021

@author: Yifang
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy import signal
from SPADPhotometryAnalysis import SPADdemod
from SPADPhotometryAnalysis import photometry_functions as fp
from scipy.fft import fft

'''Set basic parameter'''
def Set_filename (dpath, csv_filename="traceValue.csv"):
    #dpath=set_dpath()
    filename = os.path.join(dpath, csv_filename) #csv file is the file contain values for each frame
    return filename
    
def Read_trace (filename,mode="SPAD",dtype="numpy"):
    '''mode can be SPAD or photometry'''
    '''dtype can be numpy or pandas---in the future'''
    if mode =="SPAD":
        trace = np.genfromtxt(filename, delimiter=',')
        return trace
    elif mode =="photometry":
        Two_traces=pd.read_csv(filename)
        Green=Two_traces['Analog1']
        Red=Two_traces[' Analog2']
        return Green,Red
    
def getSignalTrace (filename, traceType='Constant',HighFreqRemoval=True,getBinTrace=False,bin_window=20):
    '''TraceType:Freq, Constant, TimeDiv'''
    trace=Read_trace (filename,mode="SPAD")
    if HighFreqRemoval==True:
            trace=butter_filter(trace, btype='low', cutoff=2000, fs=9938.4, order=10)  
    if traceType=='Constant':
        if getBinTrace==True:
            trace_binned=get_bin_trace(trace,bin_window=bin_window,color='m')
            #trace_binned=Ananlysis.get_bin_trace(trace,bin_window=bin_window)
            return trace_binned
        else:
            return trace
    if traceType=='Freq':
        #Red,Green= SPADdemod.DemodFreqShift (trace,fc_g=1000,fc_r=2000,fs=9938.4)
        Red,Green= SPADdemod.DemodFreqShift_bandpass (trace,fc_g=1009,fc_r=1609,fs=9938.4)
        #Red=Ananlysis.butter_filter(Red, btype='low', cutoff=200, fs=9938.4, order=10)
        #Green=Ananlysis.butter_filter(Green, btype='low', cutoff=200, fs=9938.4, order=10)
        Signal=getSignal_subtract(Red,Green,fs=9938.4)
        return Red,Green,Signal
    
def getTimeDivisionTrace (dpath, trace, sig_highlim,sig_lowlim, ref_highlim,ref_lowlim):
    '''
    This method is suitable when the time-division two signals are with very different amplitude,
    I use two different thresholds to detect peak values for two channels
    '''
    lmin,lmax=SPADdemod.Find_targetPeaks(trace, dmin=1, dmax=1,high_limit=sig_highlim, low_limit=sig_lowlim)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(lmax,trace[lmax], color='g')
    x_green, Green=SPADdemod.Interpolate_timeDiv (lmax,trace)
    fname = os.path.join(dpath, "Green_traceAll.csv")
    np.savetxt(fname, Green, delimiter=",")
    
    lmin,lmax=SPADdemod.Find_targetPeaks(trace, dmin=1, dmax=1,high_limit=ref_highlim, low_limit=ref_lowlim)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(lmax,trace[lmax], color='r')
    x_red, Red=SPADdemod.Interpolate_timeDiv (lmax,trace)
    fname = os.path.join(dpath, "Red_traceAll.csv")
    np.savetxt(fname, Red, delimiter=",")
    return Green,Red

def getTimeDivisionTrace_fromMask (dpath, Trace_raw, high_thd,low_thd):
    '''
   This method can be used when the two channels are with similar amplitudes.
   Usually, I use 500Hz square wave for time division
   Signal channel is modulated by a 30% duty cycle(5-6 samples)sqaure wave, 
   while reference channel is modulated by a 20% duty cycle wave(2-4 samples).
   high_thd and low_thd are for detecting all square wave peaks.
   Then I can use the width of the square wave to sparate the two channels
    '''
    mask=SPADdemod.findMask(Trace_raw,high_thd=high_thd,low_thd=low_thd)
    mask_green=SPADdemod.preserve_more_than_five_ones(mask)
    mask_red=SPADdemod.preserve_fewer_than_four_ones(mask)
    Green_peakIdx,Green_raw= SPADdemod.findTraceFromMask(Trace_raw,mask_green)
    Red_peakIdx,Red_raw= SPADdemod.findTraceFromMask(Trace_raw,mask_red)
    
    x_green, Green=SPADdemod.Interpolate_timeDiv (Green_peakIdx,Green_raw)
    x_red, Red=SPADdemod.Interpolate_timeDiv (Red_peakIdx,Red_raw)
    
    fname = os.path.join(dpath, "Green_traceAll.csv")
    np.savetxt(fname, Green, delimiter=",")
    
    fname = os.path.join(dpath, "Red_traceAll.csv")
    np.savetxt(fname, Red, delimiter=",")
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x_green,Green, color='g')
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x_red,Red, color='r')

    return Green,Red

def ReadTwoROItrace (dpath,plot_xrange=500):
    '''This is to plot the raw trace from two ROIs that recorded under time division mode.
    From the plotting, I can find the threshold for demodulate the green and red trace with DemodTwoTraces function'''
    filename_g=Set_filename (dpath,"traceGreenAll.csv")
    filename_r=Set_filename (dpath,"traceRedAll.csv")
    Green_raw=getSignalTrace (filename_g,traceType='Constant',HighFreqRemoval=False,getBinTrace=False)
    Red_raw=getSignalTrace (filename_r,traceType='Constant',HighFreqRemoval=False,getBinTrace=False)
    fig, ax = plt.subplots(figsize=(12, 2.5))
    plot_trace(Green_raw[100:plot_xrange],ax, fs=9938.4, label="Green data trace")
    fig, ax = plt.subplots(figsize=(12, 2.5))
    plot_trace(Red_raw[100:plot_xrange],ax, fs=9938.4, label="Red data trace")
    return Green_raw,Red_raw

def DemodTwoTraces (dpath,Green_raw, Red_raw,high_g,low_g,high_r,low_r):
    lmin,lmax=SPADdemod.Find_targetPeaks(Green_raw, dmin=1, dmax=1,high_limit=high_g, low_limit=low_g)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(lmax,Green_raw[lmax], color='g')
    #ax.plot(lmin,Green_raw[lmin], color='k')
    x_green, Green=SPADdemod.Interpolate_timeDiv (lmax,Green_raw)
    #x_dark, Dark=SPADdemod.Interpolate_timeDiv (lmin,Green_raw)
    
    lmin,lmax=SPADdemod.Find_targetPeaks(Red_raw, dmin=1, dmax=1,high_limit=high_r, low_limit=low_r)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(lmax,Red_raw[lmax], color='r')
    #ax.plot(lmin,Red_raw[lmin], color='k')
    x_red, Red=SPADdemod.Interpolate_timeDiv (lmax,Red_raw)
    #x_dark, Dark=SPADdemod.Interpolate_timeDiv (lmin,Red_raw)  
    fname = os.path.join(dpath, "Green_traceAll.csv")
    np.savetxt(fname, Green, delimiter=",")
    
    fname = os.path.join(dpath, "Red_traceAll.csv")
    np.savetxt(fname, Red, delimiter=",")
    return Green, Red

def get_bin_trace (trace,bin_window=10,color='tab:blue',Fs=9938.4):
    '''Basic filter and smooth'''
    trace = trace.astype(np.float64)
    '''reverse the trace (voltron and ASAP3 is reversed)''' 
    # trace_reverse=np.negative(trace_raw)
    # plot_trace(trace_reverse, name='raw_trace_reverse')     
    trace_binned=np.array(trace).reshape(-1, bin_window).mean(axis=1)
    fig, ax = plt.subplots(figsize=(12,4))
    ax=plot_trace(trace_binned,ax, fs=Fs/bin_window,label="Trace_binned to "+str(int(Fs/bin_window))+"Hz",color=color)
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return trace_binned

def get_detrend (trace):
    trace_detrend= signal.detrend(trace) 
    fig, ax = plt.subplots(figsize=(15, 3))
    ax=plot_trace(trace_detrend,ax, title="Trace_detrend")
    return trace_detrend

def butter_filter(data, btype='low', cutoff=10, fs=9938.4, order=5):
#def butter_filter(data, btype='high', cutoff=3, fs=130, order=5): # for photometry data  
    # cutoff and fs in Hz
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
    y = signal.filtfilt(b, a, data, axis=0)
    # fig, ax = plt.subplots(figsize=(15, 3))
    # ax=plot_trace(y,ax, label="trace_10Hz_low_pass")
    return y

'''Use python Scipy to plot PSD'''
def PSD_plot (data,fs=9938.4,method="welch",color='tab:blue',linewidth=1):
    '''Three methods to plot PSD: welch,periodogram,plotlib'''
    fig=plt.figure()
    if method == "welch":
        f, Pxx_den = signal.welch(data, fs=fs, nperseg=4096)       
        plt.semilogy(f, Pxx_den,linewidth=linewidth)
        #plt.ylim([0.5e-3, 1])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
    elif method == "periodogram":
        f, Pxx_den = signal.periodogram(data, fs=fs)
        plt.semilogy(f, Pxx_den,linewidth=linewidth)
    elif method == "plotlib":
        plt.psd(data,Fs=fs,linewidth=linewidth)
    return fig

def combineTraces (dpath,fileNum):
    for i in range(fileNum):
        filename = os.path.join(dpath, "traceValue"+str(i+1)+".csv")  #csv file is the file contain values for each frame
        print(filename)
        if i==0:
            trace_raw = np.genfromtxt(filename, delimiter=',')
        else:
            trace_add = np.genfromtxt(filename, delimiter=',')
            trace_raw=np.hstack((trace_raw,trace_add))
    filename = os.path.join(dpath, "traceValueAll.csv")
    np.savetxt(filename, trace_raw, delimiter=",")
    return trace_raw

def getSignal_subtract_freq(trace,fc_g=1000,fc_r=2000,fs=9938.4):
    #Red,Green= SPADdemod.DemodFreqShift (trace,fc_g=fc_g,fc_r=fc_r,fs=9938.4)
    Red,Green= SPADdemod.DemodFreqShift_bandpass (trace,fc_g=fc_g,fc_r=fc_r,fs=9938.4)
    from sklearn import preprocessing
    RedNorm=preprocessing.normalize([Red])
    GreenNorm=preprocessing.normalize([Green])
    Signal=GreenNorm-RedNorm
    return Signal[0]

def getSignal_subtract(Red,Green,fs=9938.4):
    from sklearn import preprocessing
    RedNorm=preprocessing.normalize([Red])
    GreenNorm=preprocessing.normalize([Green])
    Signal=GreenNorm-RedNorm
    return Signal[0]

def getICA (Red,Green):
    channel1=Green
    channel2=Red
    X = np.c_[channel1,channel2]
    # Compute ICA
    ica = FastICA(n_components=2)
    S = ica.fit_transform(X)  # Reconstruct signals
    A = ica.mixing_  # Get estimated mixing matrix
    '''Plot ICA'''
    plt.figure()
    models = [X, S]
    names = [
        "Observations (mixed signal)",
        "ICA recovered signals",
    ]
    colors = ["green", "red"]
    
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(2, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color,alpha=0.5)
    plt.tight_layout()
    plt.show()
    '''get two separated signals'''
    signal1=S[:,0]
    signal2=S[:,1]
    
    return signal1, signal2

'''PSD analysis after subtracting mean'''
def plot_PSD_bands (trace,fs=9938.4):
    from numpy.fft import fft
    t = np.arange(len(trace)) / fs
    
    x = trace                               # Relabel the data variable
    dt = t[1] - t[0]                      # Define the sampling interval
    N = x.shape[0]                        # Define the total number of data points
    T = N * dt                            # Define the total duration of the data
    
    xf = fft(x - x.mean())                # Compute Fourier transform of x
    Sxx = 2 * dt ** 2 / T * (xf * xf.conj())  # Compute spectrum
    Sxx = Sxx[:int(len(x) / 2)]           # Ignore negative frequencies
    
    df = 1 / T.max()                      # Determine frequency resolution
    fNQ = 1 / dt / 2                      # Determine Nyquist frequency
    faxis = np.arange(0,fNQ,df)              # Construct frequency axis

    fig, ax = plt.subplots(2,2,sharey=False)
    
    ax[0,0].plot(faxis, Sxx.real)                 # Plot spectrum vs frequency
    ax[0,0].set_xlim([0, 1])
    #ax[0,0].set_ylim([0, 2000])                    # Select frequency range
    ax[0,0].set_title("Slow Wave band",fontsize=8)
    ax[0,0].xaxis.set_tick_params(labelsize=8)
    ax[0,0].yaxis.set_tick_params(labelsize=8)
    
    ax[0,1].plot(faxis, Sxx.real)                 # Plot spectrum vs frequency
    ax[0,1].set_xlim([0, 28])
    #ax[0,1].set_ylim([0, 20000])                    # Select frequency range
    ax[0,1].set_title("Theta band",fontsize=8)
    ax[0,1].xaxis.set_tick_params(labelsize=8)
    ax[0,1].yaxis.set_tick_params(labelsize=8)
    
    ax[1,0].plot(faxis, Sxx.real)                 # Plot spectrum vs frequency
    ax[1,0].set_xlim([30, 80])
    ax[1,0].set_ylim([0, 5000])                    # Select frequency range
    ax[1,0].set_title("Gamma band",fontsize=8)
    ax[1,0].xaxis.set_tick_params(labelsize=8)
    ax[1,0].yaxis.set_tick_params(labelsize=8)
    
    ax[1,1].plot(faxis, Sxx.real)                 # Plot spectrum vs frequency
    ax[1,1].set_xlim([100, 220])
    ax[1,1].set_ylim([0, 3000])                    # Select frequency range
    ax[1,1].set_title("Ripple band",fontsize=8)
    ax[1,1].xaxis.set_tick_params(labelsize=8)
    ax[1,1].yaxis.set_tick_params(labelsize=8)
    
    '''How to add a common label'''
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Frequency [Hz]",fontsize=8)
    plt.ylabel("Power [$\mu V^2$/Hz]",fontsize=8)
    
    fig.tight_layout()
    
    return fig

def plot_PSD_bands_full (trace,fs=9938.4):
    from numpy.fft import fft
    t = np.arange(len(trace)) / fs
    
    x = trace                               # Relabel the data variable
    dt = t[1] - t[0]                      # Define the sampling interval
    N = x.shape[0]                        # Define the total number of data points
    T = N * dt                            # Define the total duration of the data
    
    xf = fft(x - x.mean())                # Compute Fourier transform of x
    Sxx = 2 * dt ** 2 / T * (xf * xf.conj())  # Compute spectrum
    Sxx = Sxx[:int(len(x) / 2)]           # Ignore negative frequencies
    
    df = 1 / T.max()                      # Determine frequency resolution
    fNQ = 1 / dt / 2                      # Determine Nyquist frequency
    faxis = np.arange(0,fNQ,df)              # Construct frequency axis

    fig, ax = plt.subplots(1,1)
    
    ax.plot(faxis, Sxx.real)                 # Plot spectrum vs frequency
    ax.set_xlim([0,5000])
    #ax.set_ylim([0, 1e-7])                    # Select frequency range
    ax.set_title("Full band",fontsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    return fig
    
def plot_trace(trace,ax, fs=9938.4, label="trace",color='tab:blue'):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    #ax.plot(taxis,trace,linewidth=0.5,label=label,color=color)
    ax.plot(taxis,trace,linewidth=1,label=label,color=color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.xaxis.set_visible(False)  # Hide x-axis
    #ax.yaxis.set_visible(False)  # Hide x-axis
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax

def plotSingleTrace (ax, signal, SamplingRate,color='blue'):
    t=(len(signal)) / SamplingRate
    taxis = np.arange(len(signal)) / SamplingRate
    ax.plot(taxis,signal,color,linewidth=1,alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.set_xlabel('Time(second)',fontsize=20)
    ax.set_ylabel('Photon Count')
    return ax 

def plotSpectrum (signal,Fs=9938.4):
    
    t = np.linspace(0, 1, len(signal))
    # Apply Fourier transform
    frequencies = np.fft.fftfreq(len(signal), t[1]-t[0])
    spectrum = np.abs(np.fft.fft(signal))**2
    # Reshape the spectrum as a 2D array
    time_bins = 10
    freq_bins = 2
    spectrum_2d = spectrum.reshape(time_bins, freq_bins)
    
    # Plot the power spectrum heatmap over time
    fig, ax = plt.subplots()
    im = ax.imshow(spectrum_2d.T, cmap='inferno', aspect='auto',
                   extent=[t.min(), t.max(), frequencies.min(), frequencies.max()])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Power Spectrum Heatmap')
    plt.colorbar(im)
    plt.show()

def photometry_smooth_plot (raw_reference,raw_signal,sampling_rate=500, smooth_win = 10):
    smooth_reference = fp.smooth_signal(raw_reference, smooth_win)
    smooth_signal = fp.smooth_signal(raw_signal, smooth_win)
    
    lambd = 10e8 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    
    r_base=fp.airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=fp.airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)
    
    # fig = plt.figure(figsize=(16, 10))
    # ax1 = fig.add_subplot(211)
    # ax1 = fp.plotSingleTrace (ax1, smooth_signal, SamplingRate=sampling_rate,color='blue',Label='Smoothed signal')
    # #ax1.plot(s_base,'black',linewidth=1)
    # ax2 = fig.add_subplot(212)
    # ax2 = fp.plotSingleTrace (ax2, smooth_reference, SamplingRate=sampling_rate,color='purple',Label='Smoothed reference')
    # #ax2.plot(r_base,'black',linewidth=1)
    
    remove=0
    reference = (smooth_reference[remove:] - r_base[remove:])
    signal = (smooth_signal[remove:] - s_base[remove:])  
    
    # fig = plt.figure(figsize=(16, 10))
    # ax1 = fig.add_subplot(211)
    # ax1 = fp.plotSingleTrace (ax1, signal, SamplingRate=sampling_rate,color='blue',Label='corrected_signal')
    # ax2 = fig.add_subplot(212)
    # ax2 = fp.plotSingleTrace (ax2, reference, SamplingRate=sampling_rate,color='purple',Label='corrected_reference')
    
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)
    
    # fig = plt.figure(figsize=(16, 10))
    # ax1 = fig.add_subplot(211)
    # ax1 = fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue',Label='normalised_signal')
    # ax2 = fig.add_subplot(212)
    # ax2 = fp.plotSingleTrace (ax2, z_reference, SamplingRate=sampling_rate,color='purple',Label='normalised_reference')
    
    from sklearn.linear_model import Lasso
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(z_reference)
    lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
    z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)
    
    # fig = plt.figure(figsize=(16, 10))
    # ax1 = fig.add_subplot(111)
    # ax1=fp.plotSingleTrace (ax1, z_signal, SamplingRate=sampling_rate,color='blue')
    # ax1=fp.plotSingleTrace (ax1, z_reference_fitted, SamplingRate=sampling_rate,color='purple')
    # #ax1.plot(z_signal,'blue',linewidth=1)
    # #ax1.plot(z_reference_fitted,'purple',linewidth=1)
    
    zdFF = (z_signal - z_reference_fitted)
    fig = plt.figure(figsize=(16, 15))
    ax1 = fig.add_subplot(311)
    ax1 = fp.plotSingleTrace (ax1, smooth_signal, SamplingRate=sampling_rate,color='blue',Label='Smoothed signal')
    ax2 = fig.add_subplot(312)
    ax2 = fp.plotSingleTrace (ax2, smooth_reference, SamplingRate=sampling_rate,color='purple',Label='Smoothed reference')
    ax3 = fig.add_subplot(313)
    ax3=fp.plotSingleTrace (ax3, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')
    
    return z_signal,smooth_signal,signal



