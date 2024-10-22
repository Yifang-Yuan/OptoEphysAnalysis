# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:13:10 2022
@author: Yifang
The photometry processing uses Katemartian pipeline
https://github.com/katemartian/Photometry_data_processing
Other plotting and I/O functions are written by Yifang Yuan
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Lasso
import pandas as pd
import os
'''
get_zdFF.py calculates standardized dF/F signal based on calcium-idependent 
and calcium-dependent signals commonly recorded using fiber photometry calcium imaging
Ocober 2019 Ekaterina Martianova ekaterina.martianova.1@ulaval.ca 
Reference:
  (1) Martianova, E., Aronson, S., Proulx, C.D. Multi-Fiber Photometry 
      to Record Neural Activity in Freely Moving Animal. J. Vis. Exp. 
      (152), e60278, doi:10.3791/60278 (2019)
      https://www.jove.com/video/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving
'''

def get_zdFF(reference,signal,smooth_win=10,remove=0,lambd=5e4,porder=1,itermax=50): 
  '''
  Calculates z-score dF/F signal based on fiber photometry calcium-idependent 
  and calcium-dependent signals
  
  Input
      reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
      signal: calcium-dependent signal (usually 465-490 nm excitation for 
                   green fluorescent proteins, or ~560 nm for red), 1D array
      smooth_win: window for moving average smooth, integer
      remove: the beginning of the traces with a big slope one would like to remove, integer
      Inputs for airPLS:
      lambd: parameter that can be adjusted by user. The larger lambda is,  
              the smoother the resulting background, z
      porder: adaptive iteratively reweighted penalized least squares for baseline fitting
      itermax: maximum iteration times
  Output
      zdFF - z-score dF/F, 1D numpy array
  '''
 # Smooth signal
  reference = smooth_signal(reference, smooth_win)
  signal = smooth_signal(signal, smooth_win)
  
 # Remove slope using airPLS algorithm
  r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
  s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax) 

 # Remove baseline and the begining of recording
  reference = (reference[remove:] - r_base[remove:])
  signal = (signal[remove:] - s_base[remove:])   

 # Standardize signals    
  reference = (reference - np.median(reference)) / np.std(reference)
  signal = (signal - np.median(signal)) / np.std(signal)
  
 # Align reference signal to calcium signal using non-negative robust linear regression
  lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
              positive=True, random_state=9999, selection='random')
  n = len(reference)
  if isinstance(signal, pd.Series):
      signal=signal.to_numpy()
  if isinstance(reference, pd.Series):
      reference=reference.to_numpy()
      
  lin.fit(reference.reshape(n,1), signal.reshape(n,1))
  reference = lin.predict(reference.reshape(n,1)).reshape(n,)

 # z dFF    
  zdFF = (signal - reference)
 
  return zdFF


def smooth_signal(x,window_len=10,window='flat'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.
    output:
        the smoothed signal        
    """
    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x.loc[window_len-1:0:-1],x,x.loc[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]


'''
airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares
This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls

'''
def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is, 
                 the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

'''
FROM HERE ARE CUSTOMISED CODES FOR ANALYSE, PLOT AND SYNC
'''

def plotSingleTrace (ax, signal, SamplingRate,color='blue', Label=None,linewidth=1):
    t=(len(signal)) / SamplingRate
    time = np.arange(len(signal)) / SamplingRate
    ax.plot(time,signal,color,linewidth=linewidth,label=Label,alpha=0.7)
    if Label is not None:
        ax.legend(loc="upper right", frameon=False,fontsize=20)  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.set_xlabel('Time(second)',fontsize=20)
    ax.set_ylabel('Photon Count')
    return ax 

def photometry_smooth_plot (raw_reference,raw_signal,sampling_rate=500, smooth_win = 10):
    smooth_reference = smooth_signal(raw_reference, smooth_win)
    smooth_Signal = smooth_signal(raw_signal, smooth_win)
    
    lambd = 10e4 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    
    r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=airPLS(smooth_Signal,lambda_=lambd,porder=porder,itermax=itermax)
    
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(211)
    ax1 = plotSingleTrace (ax1, smooth_Signal, SamplingRate=sampling_rate,color='blue',Label='smooth_signal') 
    ax1 = plotSingleTrace (ax1, s_base, SamplingRate=sampling_rate,color='black',Label='baseline_signal',linewidth=2) 
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2 = fig.add_subplot(212)
    ax2 = plotSingleTrace (ax2, smooth_reference, SamplingRate=sampling_rate,color='purple',Label='smooth_reference')
    ax2 = plotSingleTrace (ax2, r_base, SamplingRate=sampling_rate,color='black',Label='baseline_reference',linewidth=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    return smooth_reference,smooth_Signal,r_base,s_base

def read_photometry_data (folder, file_name, readCamSync='True',plot=False,sampling_rate=130):
    PhotometryData = pd.read_csv(folder+file_name,index_col=False) # Adjust this line depending on your data file
    raw_reference = PhotometryData[' Analog2'][1:]
    raw_signal = PhotometryData['Analog1'][1:]
    if readCamSync:
        Cam_Sync=PhotometryData[' Digital1'][1:]
        if plot:
            fig = plt.figure(figsize=(16, 10))
            ax1 = fig.add_subplot(311)
            ax1 = plotSingleTrace (ax1, raw_signal, SamplingRate=sampling_rate,color='blue',Label='raw_Signal')
            ax2 = fig.add_subplot(312)
            ax2 = plotSingleTrace (ax2, raw_reference, SamplingRate=sampling_rate,color='purple',Label='raw_Reference')
            ax3 = fig.add_subplot(313)
            ax3 = plotSingleTrace (ax3, Cam_Sync, SamplingRate=sampling_rate,color='orange',Label='Digital_Sync')
        return raw_signal,raw_reference,Cam_Sync
    else:
        if plot:
            fig = plt.figure(figsize=(16, 10))
            ax1 = fig.add_subplot(211)
            ax1 = plotSingleTrace (ax1, raw_signal, SamplingRate=sampling_rate,color='blue',Label='raw_Signal')
            ax2 = fig.add_subplot(212)
            ax2 = plotSingleTrace (ax2, raw_reference, SamplingRate=sampling_rate,color='purple',Label='raw_Reference')
        return raw_signal,raw_reference

def read_Bonsai_Sync (folder, sync_filename,plot=False):
    CamSync_LED = pd.read_csv(folder+sync_filename,index_col=False)
    CamSync_LED['LEDSync'] = CamSync_LED['Value.X'].apply(lambda x: 1 if pd.isna(x) else 0)
    if plot:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        ax = plotSingleTrace (ax, CamSync_LED['LEDSync'], SamplingRate=24,color='blue',Label='Cam_sync')
    return CamSync_LED['LEDSync']

def plot_sync (raw_signal,raw_reference,Cam_Sync,CamSync_LED,pyFs=130,CamFs=24):
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(411)
    ax1 = plotSingleTrace (ax1, raw_signal, SamplingRate=pyFs,color='blue',Label='raw_Signal')
    ax2 = fig.add_subplot(412)
    ax2 = plotSingleTrace (ax2, raw_reference, SamplingRate=pyFs,color='purple',Label='raw_Reference')
    ax3 = fig.add_subplot(413)
    ax3 = plotSingleTrace (ax3, Cam_Sync, SamplingRate=pyFs,color='orange',Label='Camera_Sync')
    ax4 = fig.add_subplot(414)
    ax4 = plotSingleTrace (ax4, CamSync_LED, SamplingRate=CamFs,color='green',Label='Camera_LEDSync')
    return -1

def Cut_photometry_data (data, CamSync):
    '''This code cut the photometry data from the first syncing pulse, 
    so the data or zscore start from the syncing frame'''
    first_1_index = CamSync.idxmax()
    df=pd.Series(data)
    data_sync = df[first_1_index:]
    return data_sync

def read_cheeseboard_from_COLD (folder, COLD_filename):
    '''This code read the cheeseboard timing data from the COLD pipeline results'''
    filename=os.path.join(folder,COLD_filename)
    cheeseboard_data = pd.read_excel(filename,index_col=False)
    return cheeseboard_data

def sync_photometry_Cam(zdFF,pyCam_Sync,CamSync_LED,CamFs):
    '''This code returns the synchronised z-score from the photometry recording, 
    and the sync start frame index/time of the LED recorded by the behavoral camera'''
    first_sync_idx_py = pyCam_Sync.idxmax()
    last_sync_idx_py = pyCam_Sync[::-1].idxmax()
    sync_length_py=(last_sync_idx_py-first_sync_idx_py)/130
    print ('Sync pulses in pyPhotometry Digital: ', sync_length_py)
    zscore_sync=Cut_photometry_data (zdFF, pyCam_Sync)
    first_sync_idx_inCam = CamSync_LED.idxmax()
    last_sync_idx_inCam= CamSync_LED[::-1].idxmax()
    sync_length_Cam=(last_sync_idx_inCam-first_sync_idx_inCam)/CamFs
    print ('Sync pulses in LED recorded by Camera ',sync_length_Cam)
    Sync_Start_time=first_sync_idx_inCam/CamFs
    return zscore_sync,first_sync_idx_inCam,Sync_Start_time

def adjust_time_to_photometry(cheeaseboard_session_data,trial_index,Sync_Start_time):
    '''This code adjust the cheeseboard timing from the COLD to sync with the photometry trace
    The returned time points are the time in the photometry trace'''
    startingtime_COLD=cheeaseboard_session_data['startingtime_s'][trial_index]
    well1time_COLD=cheeaseboard_session_data['well1time_s'][trial_index]
    well2time_COLD=cheeaseboard_session_data['well2time_s'][trial_index]
    entertime=startingtime_COLD-Sync_Start_time
    well1time=(well1time_COLD+startingtime_COLD)-Sync_Start_time
    well2time=(well2time_COLD+startingtime_COLD)-Sync_Start_time
    print ('startingtime_py', entertime)
    print ('well1time_py', well1time)
    print ('well2time_py', well2time)
    return entertime, well1time, well2time

def PETH_plot_zscore(ax, zscore_sync,centre_time, half_timewindow, fs,color,Label='zscore'):
    start_idx=int((centre_time-half_timewindow)*fs)
    end_idx=int((centre_time+half_timewindow)*fs)
    num_samples = len(zscore_sync[start_idx:end_idx])
    #time_in_seconds = np.arange(num_samples) / fs
    time_in_seconds = np.linspace(-half_timewindow, half_timewindow, num_samples)
    #ax = plotSingleTrace (ax, zscore_sync[start_idx:end_idx], SamplingRate=fs,color=color,Label=None)
    ax.plot(time_in_seconds,zscore_sync[start_idx:end_idx], label=Label, color=color,alpha=0.5)
    return ax

def PETH_plot_zscore_diff_window(ax, zscore_sync,centre_time, before_window,after_window, fs,color,Label='zscore'):
    start_idx=int((centre_time-before_window)*fs)
    end_idx=int((centre_time+after_window)*fs)
    num_samples = len(zscore_sync[start_idx:end_idx])
    #time_in_seconds = np.arange(num_samples) / fs
    time_in_seconds = np.linspace(-before_window, after_window, num_samples)
    #ax = plotSingleTrace (ax, zscore_sync[start_idx:end_idx], SamplingRate=fs,color=color,Label=None)
    ax.plot(time_in_seconds,zscore_sync[start_idx:end_idx], label=Label, color=color,alpha=0.5)
    return ax


def Plot_mean_With_Std_PSTH(event_window_traces, before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=None):
    mean_signal = event_window_traces.mean(axis=1)
    std_deviation = event_window_traces.std(axis=1)
    event_time = 0
    num_samples = len(mean_signal)
    time_in_seconds = np.linspace(-before_window, after_window, num_samples)

    # If an 'ax' is provided, use it for plotting; otherwise, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_in_seconds, mean_signal, label='Mean Signal', color=meancolor,linewidth=0.5)
    ax.fill_between(time_in_seconds, mean_signal - std_deviation, mean_signal + std_deviation, color=stdcolor, alpha=0.5, label='Standard Deviation')
    ax.axvline(x=event_time, color='red', linestyle='--', label='Event Time')
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Value')
    ax.set_title('Mean Signal with Standard Deviation ' + animalID)
    #ax.legend()

    # If 'ax' was provided, do not call plt.show() to allow the caller to display or save the figure as needed
    if ax is None:
        plt.show()
    return ax

def Plot_single_trial_PSTH(event_window_traces, trialIdx,timediff,before_window, after_window, animalID, meancolor='blue', stdcolor='lightblue', ax=None):
    signal = event_window_traces.filter(regex=str(trialIdx)+'_1$', axis=1)
    event_time1 = 0
    event_time2 =timediff
    num_samples = len(signal)
    time_in_seconds = np.linspace(-before_window, after_window, num_samples)

    # If an 'ax' is provided, use it for plotting; otherwise, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_in_seconds, signal, label='Mean Signal', color=meancolor,linewidth=0.5)
    ax.axvline(x=event_time1, color='green', linestyle='--', label='Well1 Time')
    ax.axvline(x=event_time2, color='red', linestyle='--', label='Well2 Time')
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Value')
    ax.set_title('Mean Signal with Standard Deviation ' + animalID)
    # If 'ax' was provided, do not call plt.show() to allow the caller to display or save the figure as needed
    if ax is None:
        plt.show()
    return ax

def read_all_photometry_files(folder_path, py_target_string,sync_target_string,CamFs,pyFs,COLD_folder,COLD_filename):
    '''This is to read multiple photometry files for a cheeseboard session and sync with behaviour data,
    py_target_string: a specific string in pyPhotometry data .csv file name
    sync_target_string: a specific string in Bonsai sync file name .csv file name
    you need to delete other non-csv files in the same folder.'''
    files = os.listdir(folder_path)
    filtered_files = [file for file in files if py_target_string in file]  
    cheeaseboard_session_data=read_cheeseboard_from_COLD (COLD_folder, COLD_filename)
    photometry_df = pd.DataFrame([])
    for file in filtered_files:
        # Extract the last number from the file name
        last_underscore_index = file.rfind('_')
        target_index_str = file[last_underscore_index + 1:file.rfind('.csv')]
        target_index = int(target_index_str)
        #target_index = str(''.join(filter(str.isdigit, file)))[-1]
        print(file+target_index_str)
        # Read the CSV file with photometry read
        raw_signal,raw_reference,Cam_Sync=read_photometry_data (folder_path, file, readCamSync='True',plot=False,sampling_rate=130)
        zdFF = get_zdFF(raw_reference,raw_signal,smooth_win=10,remove=0,lambd=10e4,porder=1,itermax=50)
        filtered_files_sync = [file for file in files if sync_target_string in file]
        for Sync_file in filtered_files_sync:
            syncfile_last_underscore_index = Sync_file.rfind('_')
            syncfile_target_index_str = Sync_file[syncfile_last_underscore_index + 1:Sync_file.rfind('.csv')]
            syncfile_target_index=int(syncfile_target_index_str)
            #last_number = str(''.join(filter(str.isdigit, Sync_file)))[-1]            
            if syncfile_target_index == target_index:
                print(Sync_file+syncfile_target_index_str)
                CamSync_LED=read_Bonsai_Sync (folder_path,Sync_file,plot=False)
                zscore_sync,Sync_index_inCam,Sync_Start_time=sync_photometry_Cam(zdFF,Cam_Sync,CamSync_LED,CamFs=CamFs)
                zscore_series=pd.Series(zscore_sync)
                zscore_series=zscore_series.reset_index(drop=True)
                Sync_Start_time_series=pd.Series(Sync_Start_time)
                entertime, well1time,well2time=adjust_time_to_photometry(cheeaseboard_session_data,int(target_index),Sync_Start_time)
                if np.isnan(well1time):
                    real_well1time = well2time
                    real_well2time =well1time
                elif np.isnan(well2time):
                    real_well1time = well1time
                    real_well2time=well2time
                else:
                    # Both numbers are real, so find the larger one
                    real_well1time = np.minimum(well1time, well2time)
                    real_well2time=np.maximum(well1time, well2time)
                photometry_df['pyData'+target_index_str]=zscore_series
                photometry_df['SyncStartTimeInVideo'+target_index_str]=Sync_Start_time_series
                photometry_df['entertime'+target_index_str]=pd.Series(entertime)
                photometry_df['well1time'+target_index_str]=pd.Series(real_well1time)
                photometry_df['well2time'+target_index_str]=pd.Series(real_well2time)
    return photometry_df

def Plot_multiple_PETH(df_py_cheese,half_timewindow,fs=130, animalID='(Mouse 100)'):
    event_window_traces = pd.DataFrame([])
    selected_columns = [col_name for col_name in df_py_cheese.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
    column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
    event_time = 0 
    '''This is to make a figure and plot all PETH traces on the same figure'''
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for col_num in column_numbers:
        column_name = f'pyData{col_num}'
        column_photometry_data = df_py_cheese[column_name]
        column_well1time=df_py_cheese[f'well1time{col_num}'][0]
        if not np.isnan(column_well1time).any():
            start_idx=int((column_well1time-half_timewindow)*fs)
            end_idx=int((column_well1time+half_timewindow)*fs)
            para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
            event_window_traces[f'pyData{col_num}'+'_1']=para_event_photometry_data
            PETH_plot_zscore(ax, df_py_cheese[f'pyData{col_num}'],centre_time=
                             df_py_cheese[f'well1time{col_num}'][0], half_timewindow=half_timewindow, fs=fs,color='green',Label=f'Trace{col_num+1} Well1time')
        column_well2time=df_py_cheese[f'well2time{col_num}'][0]
        if not np.isnan(column_well2time).any():
            start_idx=int((column_well2time-half_timewindow)*fs)
            end_idx=int((column_well2time+half_timewindow)*fs)
            para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
            event_window_traces[f'pyData{col_num}'+'_2']=para_event_photometry_data
            PETH_plot_zscore(ax, df_py_cheese[f'pyData{col_num}'],centre_time=
                             df_py_cheese[f'well2time{col_num}'][0], half_timewindow=half_timewindow, fs=fs,color='red',Label=f'Trace{col_num+1} Well2time')
    plt.axvline(x=0, color='red', linestyle='--', label='Event Time') 
    plt.xlabel('Time (second)')
    plt.ylabel('Value')
    plt.title('Single Calcium traces while reaching well1 (green) and well2 (red) '+animalID)
    #plt.legend()
    main_signal = event_window_traces.mean(axis=1)
    std_deviation = event_window_traces.std(axis=1)
     
    # Create the plot
    num_samples = len(main_signal)
    #time_in_seconds = np.arange(num_samples) / fs
    time_in_seconds = np.linspace(-half_timewindow, half_timewindow, num_samples)
    # Set the x-axis tick positions and labels
    plt.figure(figsize=(10, 6))
    plt.plot(time_in_seconds,main_signal, label='Mean Signal', color='blue')
    plt.fill_between(time_in_seconds, main_signal - std_deviation, main_signal + std_deviation, color='lightblue', alpha=0.5, label='Standard Deviation')
    plt.axvline(x=event_time, color='red', linestyle='--', label='Event Time')
    plt.xlabel('Time (second)')
    plt.ylabel('Value')
    plt.title('Mean Signal with Standard Deviation '+animalID)
    plt.legend()
    plt.show()
    return event_window_traces

def Plot_multiple_PETH_different_window(df_py_cheese,before_window,after_window,fs=130, animalID='(Mouse 100)'):
    event_window_traces = pd.DataFrame([])
    selected_columns = [col_name for col_name in df_py_cheese.columns if col_name.startswith('pyData') and col_name[6:].isdigit()]
    column_numbers = [int(col_name.replace('pyData', '')) for col_name in selected_columns]  
    event_time = 0 
    '''This is to make a figure and plot all PETH traces on the same figure'''
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for col_num in column_numbers:
        column_name = f'pyData{col_num}'
        column_photometry_data = df_py_cheese[column_name]
        column_well1time=df_py_cheese[f'well1time{col_num}'][0]
        if not np.isnan(column_well1time).any():
            start_idx=int((column_well1time-before_window)*fs)
            end_idx=int((column_well1time+after_window)*fs)
            para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
            event_window_traces[f'pyData{col_num}'+'_1']=para_event_photometry_data
            PETH_plot_zscore_diff_window(ax, df_py_cheese[f'pyData{col_num}'],centre_time=
                             df_py_cheese[f'well1time{col_num}'][0], before_window=before_window,after_window=after_window, fs=fs,color='green',Label=f'Trace{col_num+1} Well1time')
        column_well2time=df_py_cheese[f'well2time{col_num}'][0]
        if not np.isnan(column_well2time).any():
            start_idx=int((column_well2time-before_window)*fs)
            end_idx=int((column_well2time+after_window)*fs)
            para_event_photometry_data=column_photometry_data[start_idx:end_idx].reset_index(drop=True)
            event_window_traces[f'pyData{col_num}'+'_2']=para_event_photometry_data
            PETH_plot_zscore_diff_window(ax, df_py_cheese[f'pyData{col_num}'],centre_time=
                             df_py_cheese[f'well2time{col_num}'][0], before_window=before_window,after_window=after_window, fs=fs,color='red',Label=f'Trace{col_num+1} Well2time')
    plt.axvline(x=0, color='red', linestyle='--', label='Event Time') 
    plt.xlabel('Time (second)')
    plt.ylabel('Value')
    plt.title('Single Calcium traces while reaching well1 (green) and well2 (red) '+animalID)
    #plt.legend()
    main_signal = event_window_traces.mean(axis=1)
    std_deviation = event_window_traces.std(axis=1)
     
    # Create the plot
    num_samples = len(main_signal)
    #time_in_seconds = np.arange(num_samples) / fs
    time_in_seconds = np.linspace(-before_window, after_window, num_samples)
    # Set the x-axis tick positions and labels
    plt.figure(figsize=(6, 6))
    plt.plot(time_in_seconds,main_signal, label='Mean Signal', color='blue',linewidth=1)
    plt.fill_between(time_in_seconds, main_signal - std_deviation, main_signal + std_deviation, color='lightblue', alpha=0.5, label='Standard Deviation')
    plt.axvline(x=event_time, color='red', linestyle='--', label='Event Time')
    plt.xlabel('Time (second)')
    plt.ylabel('Value')
    plt.title('Mean Signal with Standard Deviation '+animalID)
    plt.legend()
    plt.show()
    return event_window_traces
def Read_Concat_pkl_files (folder, IndexNumFromFilename=-4):
    dfs = []
    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):
            # Read the DataFrame from the .pkl file
            df = pd.read_pickle(os.path.join(folder, filename))            
            # Extract an index from the file name (you may need to customize this)
            index = filename.split('.')[0] [IndexNumFromFilename:] # Assuming the file names are like "1.pkl", "2.pkl", etc.            
            # Rename the columns with the extracted index
            df.columns = [f'{index}_{col}' for col in df.columns]            
            # Append the DataFrame to the list
            dfs.append(df)   
    # Concatenate the DataFrames column-wise
    result_df = pd.concat(dfs, axis=1)
    return result_df
