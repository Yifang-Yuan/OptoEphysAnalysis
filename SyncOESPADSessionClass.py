# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:03:35 2023

@author: Yifang
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy import signal
from scipy.fft import fft
import seaborn as sns
import OpenEphysTools as OE
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pynapple as nap
import pynacollada as pyna

class SyncOESPADSession:
    def __init__(self, dpath,IsTracking,read_aligned_data_from_file):
        '''
        Parameters
        ----------
        dpath : TYPE
            DESCRIPTION.
        EphysData : TYPE:pandas, readout from open-ephys recording, with SPAD mask

        '''
        self.Spad_fs = 9938.4
        self.ephys_fs = 30000
        self.tracking_fs = 15
        self.fs = 10000.0
        self.dpath=dpath
        self.IsTracking=IsTracking

        self.Ephys_data=self.read_open_ephys_data() #read ephys data that we pre-processed from dpath
        if IsTracking:
            self.read_tracking_data() #read tracking raw data   
            self.trackingdata_extent,frame_count, frame_indices= self.extent_tracking_to_ephys_pd ()    
        
        # #self.Format_tracking_data_index () ## this is to change the index to timedelta index, for resampling
        if (read_aligned_data_from_file):
            filepath=os.path.join(self.dpath, "Ephys_tracking_spad_aligned.pkl")
            self.Ephys_tracking_spad_aligned = pd.read_pickle(filepath)
        else:
            self.Sync_ephys_with_spad() #output self.spad_align, self.ephys_align
               		
    def Sync_ephys_with_spad(self):
        if self.IsTracking:
           self.Ephys_data = pd.concat([self.Ephys_data, self.trackingdata_extent], axis=1)
           
        self.form_ephys_spad_sync_data() # find the spad sync part
        self.Format_ephys_data_index () # this is to change the index to timedelta index, for resampling
        self.SPADdata=self.Read_SPAD_data() #read spad data
        self.resample_spad()
        self.resample_ephys()
        self.slice_ephys_to_align_with_spad()
        self.Ephys_tracking_spad_aligned=pd.concat([self.ephys_align, self.spad_align], axis=1)  
        self.save_data(self.Ephys_tracking_spad_aligned, 'Ephys_tracking_spad_aligned.pkl')
        return self.Ephys_tracking_spad_aligned
        
    def remove_noise(self,start_time,end_time):
        time_interval = 1 / self.fs  # Time interval in seconds
        slicing_index = pd.timedelta_range(start=f'{start_time:.9f}S', end=f'{end_time:.9f}S', freq=f'{time_interval:.9f}S')[:-1]
        # Use boolean indexing to filter out rows within the specified time range
        self.Ephys_tracking_spad_aligned = self.Ephys_tracking_spad_aligned.drop(slicing_index)
        return self.Ephys_tracking_spad_aligned
    
    def reset_index_data(self):
        time_interval = 1 / self.fs  # Time interval in seconds
        total_duration=len(self.Ephys_tracking_spad_aligned)*time_interval
        # Reindex the DataFrame with a continuous timestamp index
        new_index = pd.timedelta_range(start=f'{0:.9f}S', end=f'{total_duration:.9f}S', freq=f'{time_interval:.9f}S')[:-1]
        self.Ephys_tracking_spad_aligned = self.Ephys_tracking_spad_aligned.reset_index(drop=True)
        self.Ephys_tracking_spad_aligned = self.Ephys_tracking_spad_aligned.set_index(new_index,drop=True)
        self.Ephys_tracking_spad_aligned['timestamps']=np.arange(0, total_duration, time_interval)
        # # Save the data
        self.save_data(self.Ephys_tracking_spad_aligned, 'Ephys_tracking_spad_aligned.pkl')
        return self.Ephys_tracking_spad_aligned
    
    def Read_SPAD_data (self):
        '''
        SPAD has sampling rate of 9938.4 Hz.
        But if we use 500Hz time division photometry recording, the effective sampling rate for sig_raw and ref_raw is 500Hz.
        In the pre-processing for SPAD data, I usually smooth it to 200Hz to obtain the z-score.
        '''
        self.sig_csv_filename=os.path.join(self.dpath, "Green_traceAll.csv")
        self.ref_csv_filename=os.path.join(self.dpath, "Red_traceAll.csv")
        self.zscore_csv_filename=os.path.join(self.dpath, "Zscore_traceAll.csv")
        sig_data = np.genfromtxt(self.sig_csv_filename, delimiter=',')
        ref_data = np.genfromtxt(self.ref_csv_filename, delimiter=',')
        zscore_data = np.genfromtxt(self.zscore_csv_filename, delimiter=',')
        time_interval = 1.0 / self.Spad_fs
        total_duration = len(sig_data) * time_interval
        timestamps = np.arange(0, total_duration, time_interval)
        timestamps_time = pd.to_timedelta(timestamps, unit='s')
        sig_raw = pd.Series(sig_data, index=timestamps_time)
        ref_raw = pd.Series(ref_data, index=timestamps_time)
        zscore_raw = pd.Series(zscore_data, index=timestamps_time)
        'Zscore data is obtained by Kate Martian method, smoothed to 250Hz effective sampling rate'
        self.SPADdata = pd.DataFrame({
            'sig_raw': sig_raw,
            'ref_raw': ref_raw,
            'zscore_raw': zscore_raw,
        })
        return self.SPADdata
    
    def read_open_ephys_data (self):
        filepath=os.path.join(self.dpath, "open_ephys_read_pd.pkl")
        self.Ephys_data = pd.read_pickle(filepath)  
        return self.Ephys_data
    
    def form_ephys_spad_sync_data (self):
        mask = self.Ephys_data['SPAD_mask'] 
        self.Ehpys_sync_data=self.Ephys_data[mask]
        OE.plot_two_raw_traces (mask,self.Ehpys_sync_data['LFP_2'], spad_label='spad_mask',lfp_label='LFP_raw') 
        return self.Ehpys_sync_data   
       

    def Format_ephys_data_index (self):
        time_interval = 1.0 / self.ephys_fs
        total_duration = len(self.Ehpys_sync_data) * time_interval
        timestamps = np.arange(0, total_duration, time_interval)
        timedeltas_index = pd.to_timedelta(timestamps, unit='s')            
        self.Ehpys_sync_data.index = timedeltas_index
        return self.Ehpys_sync_data
    
    def resample_spad (self):
        time_interval_common = 1.0 / self.fs
        self.spad_resampled = self.SPADdata.resample(f'{time_interval_common:.9f}S').mean()
        self.spad_resampled = self.spad_resampled.fillna(method='ffill')
        return self.spad_resampled
    
    def resample_ephys (self):
        time_interval_common = 1.0 / self.fs
        self.ephys_resampled = self.Ehpys_sync_data.resample(f'{time_interval_common:.9f}S').mean()
        self.ephys_resampled = self.ephys_resampled.fillna(method='ffill')
        return self.ephys_resampled                     
    
    def slice_ephys_to_align_with_spad (self):
        '''
        This is important because sometimes the effective SPAD recording is shorter than the real recording time due to deadtime. 
        E.g, I recorded 10 blocks 10s data, should be about 100s recording, but in most cases, there's no data in the last block.
        '''
        self.ephys_align = self.ephys_resampled[:len(self.spad_resampled)]
        self.spad_align=self.spad_resampled
        # Create the plot 
        return self.spad_align, self.ephys_align
    
    def read_tracking_data (self, correctTrackingFrameRate=True):
        keyword='AnimalTracking'
        files_in_directory = os.listdir(self.dpath)
        matching_files = [filename for filename in files_in_directory if keyword in filename]
        if matching_files:
            csv_file_path = os.path.join(self.dpath, matching_files[0])
            print (csv_file_path)
            self.trackingdata = pd.read_csv(csv_file_path)
            self.trackingdata=self.trackingdata.fillna(method='ffill')
            self.trackingdata=self.trackingdata/20      

            #This is to calculate the speed per frame
            #self.trackingdata['speed']=self.trackingdata.X.diff() 
            df_temp = np.sqrt(np.diff(self.trackingdata['X'])**2 + np.diff(self.trackingdata['Y'])**2)
            self.trackingdata['speed'] = [np.nan] + df_temp.tolist()
            self.trackingdata['speed']=self.trackingdata['speed']*self.tracking_fs # cm per second
            self.trackingdata['speed_abs']=self.trackingdata.speed.abs()
            self.trackingdata['speed_abs'][self.trackingdata['speed_abs'] > 20] = np.nan #If the speed is too fast, maybe the tracking position is wrong, delete it.
            self.trackingdata['speed_abs'] = self.trackingdata['speed_abs'].fillna(method='bfill')
            OE.plot_animal_tracking (self.trackingdata)
            if correctTrackingFrameRate:
                self.trackingdata = self.trackingdata.reindex(self.trackingdata.index.repeat(2)).reset_index(drop=True)
        else:
            print ('No available Tracking data in the folder!')
        return self.trackingdata
    
    # def Format_tracking_data_index (self):
    #     time_interval = 1.0 / self.tracking_fs
    #     total_duration = len(self.trackingdata) * time_interval
    #     timestamps = np.arange(0, total_duration, time_interval)
    #     timedeltas_index = pd.to_timedelta(timestamps, unit='s')            
    #     self.trackingdata.index = timedeltas_index
    #     return self.trackingdata
    
    def resample_tracking_to_ephys (self):
        time_interval_common = 1.0 / self.ephys_fs
        tracking_resampled_to_ephys = self.trackingdata.resample(f'{time_interval_common:.9f}S').mean()
        tracking_resampled_to_ephys = tracking_resampled_to_ephys.fillna(method='ffill')
        return tracking_resampled_to_ephys    
    
    def count_frames_and_indices(self, threshold=29000):
        frame_count = 0
        frame_indices = []
        prev_value = self.Ephys_data['CamSync'][0] > threshold
        
        for i, value in enumerate(self.Ephys_data['CamSync']):
            current_value = value > threshold
            if current_value != prev_value:
                frame_count += 1
                frame_indices.append(i)
            prev_value = current_value   
        print ('frame count is', frame_count)
        return frame_count, frame_indices
    
    def extent_tracking_to_ephys_pd (self):
        frame_count, frame_indices=self.count_frames_and_indices()
        if len(self.trackingdata)>frame_count:            
            self.trackingdata=self.trackingdata[1:frame_count+1]
        if len(self.trackingdata)<frame_count:
            frame_indices=frame_indices[0:len(self.trackingdata)]
        trackingdata_extent = pd.DataFrame(index=range(len(self.Ephys_data)), columns=self.trackingdata.columns)
        trackingdata_extent.loc[frame_indices,:] = self.trackingdata.values
        trackingdata_extent=trackingdata_extent.fillna(method='bfill')
        trackingdata_extent=trackingdata_extent.fillna(method='ffill')  
        return trackingdata_extent,frame_count, frame_indices
    
    def save_data (self, data,filename):
        filepath=os.path.join(self.dpath, filename)
        data.to_pickle(filepath)
        return -1
        
    
    def slicing_pd_data (self, data,start_time, end_time):
        # Start time in seconds
        # End time in seconds
        time_interval = 1 / self.fs # Time interval in seconds      
        slicing_index = pd.timedelta_range(start=f'{start_time:.9f}S', end=f'{end_time:.9f}S', freq=f'{time_interval:.9f}S')[:-1]   
        silced_data=data.loc[slicing_index]
        return silced_data
    
    def slicing_np_data (self, data,start_time, end_time):
        # Start time in seconds
        # End time in seconds
        start_idx =int( start_time * self.fs)# Time interval in seconds   
        end_idx = int(end_time * self.fs)# Time interval in seconds
        silced_data=data[start_idx:end_idx]
        return silced_data
    
        
    def plot_two_traces_heatmapSpeed (self, spad_data,lfp_data, speed_series, spad_label='spad',lfp_label='LFP',Spectro_ylim=30,AddColorbar=False):
        fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(15, 8))
        OE.plot_timedelta_trace_in_seconds (spad_data,ax1,label=spad_label,color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)
        OE.plot_timedelta_trace_in_seconds (lfp_data,ax2,label=lfp_label,color=sns.color_palette("husl", 8)[5],ylabel='uV')
        # You can adjust this percentile as needed
        OE.plotSpectrogram (ax3,lfp_data,plot_unit='WHz',nperseg=4096,y_lim=Spectro_ylim,vmax_percentile=100,Fs=self.fs,showCbar=AddColorbar)
        OE.plot_speed_heatmap(ax4, speed_series,cbar=AddColorbar)
        plt.subplots_adjust(hspace=0.2)
        #plt.tight_layout()
        #plt.show()
        return -1
    
    def plot_two_traces_lineSpeed (self, spad_data,lfp_data, speed_series, spad_label='spad',lfp_label='LFP',Spectro_ylim=20,AddColorbar=False):
        lfp_data=lfp_data/1000
        
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        OE.plot_timedelta_trace_in_seconds (spad_data,ax[0],label=spad_label,color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)
        OE.plot_timedelta_trace_in_seconds (lfp_data,ax[1],label=lfp_label,color=sns.color_palette("husl", 8)[5],ylabel='mV')
        # You can adjust this percentile as needed
        # OE.plotSpectrogram (ax[2],lfp_data,plot_unit='WHz',nperseg=8192,y_lim=Spectro_ylim,vmax_percentile=100,Fs=self.fs,showCbar=AddColorbar)
        
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_data,lowpassCutoff=1500,Fs=self.fs)
        OE.plot_wavelet(ax[2],sst,frequency,power,Fs=self.fs,colorBar=AddColorbar,logbase=False)
        OE.plot_timedelta_trace_in_seconds(speed_series,ax[3],label='speed',color=sns.color_palette("husl", 8)[6],ylabel='speed')
        ax[3].set_ylim(0,20)
        ax[3].set_xlabel('Time (seconds)')
        plt.subplots_adjust(hspace=0.5)
        #plt.tight_layout()
        #plt.show()
        return -1
    
    def plot_two_traces_noSpeed (self, spad_data,lfp_data, spad_label='spad',lfp_label='LFP',Spectro_ylim=20,AddColorbar=False):
        '''This will plot both SPAD and LFP signal with their wavelet spectrum'''
        lfp_data=lfp_data/1000
        
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        OE.plot_timedelta_trace_in_seconds (spad_data,ax[0],label=spad_label,color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(spad_data,lowpassCutoff=500,Fs=self.fs,scale=40)
        OE.plot_wavelet(ax[1],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=False)
        OE.plot_timedelta_trace_in_seconds (lfp_data,ax[2],label=lfp_label,color=sns.color_palette("husl", 8)[5],ylabel='mV')
        # You can adjust this percentile as needed
        # OE.plotSpectrogram (ax[2],lfp_data,plot_unit='WHz',nperseg=8192,y_lim=Spectro_ylim,vmax_percentile=100,Fs=self.fs,showCbar=AddColorbar)
        
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_data,lowpassCutoff=1500,Fs=self.fs,scale=40)
        OE.plot_wavelet(ax[3],sst,frequency,power,Fs=self.fs,colorBar=AddColorbar,logbase=False)
        ax[1].set_ylim(0,20)
        ax[3].set_ylim(0,20)
        ax[3].set_xlabel('Time (seconds)')
        plt.subplots_adjust(hspace=0.5)
        #plt.tight_layout()
        #plt.show()
        return -1
    
    def plot_segment_feature (self,LFP_channel,start_time,end_time,SPAD_cutoff,lfp_cutoff):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        self.plot_lowpass_two_trace (silced_recording,LFP_channel, SPAD_cutoff,lfp_cutoff)

        return -1
    
    def plot_lowpass_two_trace (self,data, LFP_channel,SPAD_cutoff,lfp_cutoff):
        #SPAD_smooth= OE.butter_filter(data['zscore_raw'], btype='high', cutoff=0.5, fs=self.fs, order=5)
        SPAD_smooth= OE.smooth_signal(data['zscore_raw'],Fs=self.fs,cutoff=SPAD_cutoff)
        SPAD_smooth= OE.butter_filter(SPAD_smooth, btype='high', cutoff=0.5, fs=self.fs, order=1)
        lfp_lowpass = OE.butter_filter(data[LFP_channel], btype='high', cutoff=0.5, fs=self.fs, order=1)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        spad_low = pd.Series(SPAD_smooth, index=data['zscore_raw'].index)
        lfp_low = pd.Series(lfp_lowpass, index=data[LFP_channel].index)
        if self.IsTracking:
            self.plot_two_traces_lineSpeed (spad_low,lfp_low,data['speed_abs'],Spectro_ylim=20,AddColorbar=True)
        else:
            self.plot_two_traces_noSpeed (spad_low,lfp_low,Spectro_ylim=20,AddColorbar=True)
        return -1
    
    
    def plot_theta_feature (self,LFP_channel,start_time,end_time,LFP=True):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        if LFP:
            lfp_data=silced_recording[LFP_channel]/1000
        else:
            lfp_data=silced_recording[LFP_channel]
        data=lfp_data.to_numpy()
        #lfp_thetaband=OE.butter_filter(signal, btype='low', cutoff=15, fs=self.fs, order=5)
        lfp_thetaband=OE.band_pass_filter(data,5,9,Fs=self.fs)
        #lfp_thetaband = lfp_thetaband - np.mean(lfp_thetaband)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_data,lowpassCutoff=500,Fs=self.fs)
        time = np.arange(len(sst)) *(1/self.fs)
        OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,lfp_thetaband)
        return -1

    
    def plot_ripple_feature (self,LFP_channel,start_time,end_time,LFP=True):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        if LFP:
            lfp_data=silced_recording[LFP_channel]/1000
        else:
            lfp_data=silced_recording[LFP_channel]
        data=lfp_data.to_numpy()
        #lfp_thetaband=OE.butter_filter(signal, btype='low', cutoff=15, fs=self.fs, order=5)
        lfp_rippleband=OE.band_pass_filter(data,150,250,Fs=self.fs)
        #lfp_thetaband = lfp_thetaband - np.mean(lfp_thetaband)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_rippleband,lowpassCutoff=500,Fs=self.fs,scale=20)
        time = np.arange(len(sst)) *(1/self.fs)
        OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,lfp_data)
        return -1
   
    def separate_theta (self,LFP_channel,theta_thres,nonthetha_thres):
        lfp_data=self.Ephys_tracking_spad_aligned[LFP_channel]/1000
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_data/1000,lowpassCutoff=500,Fs=self.fs)
        #set bound for theta band
        lower_bound= 5
        upper_bound = 10
        indices_between_range = np.where((frequency >= lower_bound) & (frequency <= upper_bound))
        
        power_band=power[indices_between_range[0]]
        power_band_mean=np.max(power_band,axis=0)
        percentile_thres_theta = np.percentile(power_band_mean, theta_thres)
        percentile_thres_nontheta = np.percentile(power_band_mean, nonthetha_thres)
        indices_above_percentile = np.where(power_band_mean > percentile_thres_theta)
        indices_below_percentile = np.where(power_band_mean < percentile_thres_nontheta)
        #Separate theta and non-theta
        self.theta_part=self.Ephys_tracking_spad_aligned.iloc[indices_above_percentile[0]]
        self.non_theta_part=self.Ephys_tracking_spad_aligned.iloc[indices_below_percentile[0]] 
        # Save the theta part with real indices
        theta_path=os.path.join(self.dpath, "theta_part_with_index.pkl")
        self.theta_part.to_pickle(theta_path) 
        non_theta_path=os.path.join(self.dpath, "non_theta_part_with_index.pkl")
        self.non_theta_part.to_pickle(non_theta_path)
        
        #From here,I reset the index, concatenate the theta part and non-theta part just for plotting and show the features
        self.theta_part=self.theta_part.reset_index(drop=True)
        self.non_theta_part=self.non_theta_part.reset_index(drop=True)
        
        time_interval = 1.0 / self.fs
        total_duration = len(self.non_theta_part) * time_interval
        self.non_theta_part['timestamps'] = np.arange(0, total_duration, time_interval)
        total_duration = len(self.theta_part) * time_interval
        # Convert the 'time_column' to timedelta if it's not already
        self.non_theta_part['time_column'] = pd.to_timedelta(self.non_theta_part['timestamps'],unit='s') 
        # Set the index to the 'time_column'
        self.non_theta_part.set_index('time_column', inplace=True)
        
        self.theta_part['timestamps'] = np.arange(0, total_duration, time_interval)
        self.theta_part['time_column'] = pd.to_timedelta(self.theta_part['timestamps'], unit='s') 
        self.theta_part.set_index('time_column', inplace=True)
        
        #plot theta
        sst,frequency,power,global_ws=OE.Calculate_wavelet(self.theta_part[LFP_channel]/1000,lowpassCutoff=500,Fs=self.fs)
        time = np.arange(len(sst)) *(1/self.fs)
        OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,self.theta_part[LFP_channel])
        self.plot_lowpass_two_trace (self.theta_part, LFP_channel,SPAD_cutoff=20,lfp_cutoff=20)
        #plot non-theta
        sst,frequency,power,global_ws=OE.Calculate_wavelet(self.non_theta_part[LFP_channel]/1000,lowpassCutoff=500,Fs=self.fs)
        time = np.arange(len(sst)) *(1/self.fs)
        OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,self.non_theta_part[LFP_channel])
        self.plot_lowpass_two_trace (self.non_theta_part, LFP_channel,SPAD_cutoff=20,lfp_cutoff=20)
        
        return self.theta_part,self.non_theta_part
         
    def get_mean_corr_two_traces (self, spad_data,lfp_data,corr_window):
        # corr_window as second
        total_seconds=len(spad_data)/self.fs
        print('total_second:',total_seconds)
        overlap=1
        total_num=int((total_seconds-corr_window)/overlap)+1
        print('total_num:',total_num)
        if type(spad_data) is pd.Series:
            spad_data_np=spad_data.to_numpy()
        elif type(spad_data) is np.ndarray:
                spad_data_np=spad_data
        if type(lfp_data) is pd.Series:
            lfp_data_np=lfp_data.to_numpy()
        elif type(lfp_data) is np.ndarray:
                lfp_data_np=lfp_data
        
        cross_corr_values = []
        for i in range(total_num):
            start_time = i * overlap
            end_time = start_time + corr_window  
            spad_1=self.slicing_np_data (spad_data_np,start_time=start_time, end_time=end_time)
            lfp_1=self.slicing_np_data (lfp_data_np,start_time=start_time, end_time=end_time)
            lags,cross_corr =OE.calculate_correlation_with_detrend (spad_1,lfp_1)
            #cross_corr = signal.correlate(spad_1, lfp_1, mode='full')
            cross_corr_values.append(cross_corr)
            
        cross_corr_values = np.array(cross_corr_values)
        mean_cross_corr = np.mean(cross_corr_values, axis=0)
        std_cross_corr = np.std(cross_corr_values, axis=0)
        
        x = lags/self.fs
        
        plt.figure(figsize=(10, 5))
        plt.plot(x, mean_cross_corr, color='b', label='Mean Cross-Correlation')
        plt.fill_between(x, mean_cross_corr - std_cross_corr, mean_cross_corr + std_cross_corr, color='gray', alpha=0.3, label='Standard Deviation')
        plt.xlabel('Lags(seconds)')
        plt.ylabel('Cross-Correlation')
        plt.title('Mean Cross-Correlation with Standard Deviation (1-Second Window)')
        plt.legend()
        plt.grid()
        plt.show()
        return lags,mean_cross_corr,std_cross_corr    
    
    def pynappleAnalysis (self,lfp_channel='LFP_2',ep_start=0,ep_end=10,Low_thres=1,High_thres=10,plot_ripple_ep=True):
        'This is the LFP data that need to be saved for the sync ananlysis'
        data_segment=self.Ephys_tracking_spad_aligned
        #data_segment=self.non_theta_part
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        timestamps=timestamps-timestamps[0]
        #Use non-theta part to detect ripple
        lfp_data=data_segment[lfp_channel]
        spad_data=data_segment['zscore_raw']
        lfp_data=lfp_data/1000 #change the unit from uV to mV
        SPAD_cutoff=50
        SPAD_smooth_np = OE.smooth_signal(spad_data,Fs=self.fs,cutoff=SPAD_cutoff)
        'To align LFP and SPAD raw data to pynapple format'
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        SPAD=nap.Tsd(t = timestamps, d = spad_data.to_numpy(), time_units = 's')
        SPAD_smooth=nap.Tsd(t = timestamps, d = SPAD_smooth_np, time_units = 's')
        if self.IsTracking:
            speed=nap.Tsd(t = timestamps, d = data_segment['speed_abs'].to_numpy(), time_units = 's')
        
        'To detect ripple'
        ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getRippleEvents (LFP,self.fs,windowlen=500,Low_thres=Low_thres,High_thres=High_thres)    
        'To plot the choosen segment'
        ex_ep = nap.IntervalSet(start = ep_start, end = ep_end, time_units = 's') 
        fig, ax = plt.subplots(6, 1, figsize=(10, 12))
        OE.plot_trace_nap (ax[0], LFP,ex_ep,color=sns.color_palette("husl", 8)[5],title='LFP raw Trace')
        OE.plot_trace_nap (ax[1], ripple_band_filtered,ex_ep,color=sns.color_palette("husl", 8)[5],title='Ripple band')
        OE.plot_ripple_event (ax[2], rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres=Low_thres) 
                
        print('LFP length:',len(LFP))
        print('SPAD length:',len(SPAD))
        print('SPAD_smooth length:',len(SPAD_smooth_np))
        
        OE.plot_trace_nap (ax[3], SPAD_smooth,ex_ep,color='green',title='calcium recording (z-score)')
        LFP_rippleband=OE.band_pass_filter(LFP.restrict(ex_ep),150,250,Fs=self.fs)        
        sst,frequency,power,global_ws=OE.Calculate_wavelet(LFP_rippleband,lowpassCutoff=1500,Fs=self.fs,scale=40)
        OE.plot_wavelet(ax[4],sst,frequency,power,Fs=self.fs,colorBar=False)
        
        if self.IsTracking:
            OE.plot_trace_nap (ax[5], speed,ex_ep,color='grey',title='speed (cm/second)')
            ax[5].set_ylim(0,10)
        #OE.plot_ripple_spectrum (ax[4], LFP, ex_ep,y_lim=30,Fs=self.fs,vmax_percentile=100)
        plt.subplots_adjust(hspace=0.5)

        print('LFP_rippleband length:',len(LFP_rippleband))
        '''To calculate cross-correlation'''
        cross_corr_values = []
        if plot_ripple_ep:
            event_peak_times=rip_tsd.index.to_numpy()
            for i in range(1,len(rip_ep)-1):
                fig, ax = plt.subplots(3, 1, figsize=(10, 18))
                ripple_std=rip_tsd.iloc[i]
                ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  
                if event_peak_times[i]>0.25:
                    start_time=event_peak_times[i]-0.25
                    end_time=event_peak_times[i]+0.25
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_ep=SPAD.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_long_ep)                    
                    #Set the title of ripple feature
                    plot_title = "Optical signal from SPAD" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(SPAD_ep,lowpassCutoff=200,Fs=self.fs,scale=40)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    OE.plot_ripple_overlay (ax[0],LFP_ep,SPAD_smooth_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=True,plotRipple=False)                                   
                    #Set the title of ripple feature
                    plot_title = "Local Field Potential with Spectrogram" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=1500,Fs=self.fs,scale=40) 
                    OE.plot_ripple_overlay (ax[1],LFP_ep,SPAD_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=False)
                    lags,cross_corr =OE.calculate_correlation_with_detrend (SPAD_ep,LFP_ep)
                    #cross_corr = signal.correlate(spad_1, lfp_1, mode='full')
                    cross_corr_values.append(cross_corr)
                    print('LFP_ep length:',len(LFP_ep))
                    print('SPAD_ep length:',len(SPAD_ep))
                    print('Cross corr:',len(cross_corr))
                    
                if event_peak_times[i]>0.05:
                    start_time=event_peak_times[i]-0.05
                    end_time=event_peak_times[i]+0.05
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_ep=SPAD.restrict(rip_long_ep)
                    ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_long_ep)
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=1500,Fs=self.fs,scale=40)                
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    #Set the title of ripple feature
                    plot_title = f"Ripple Peak std:{ripple_std:.2f}, Ripple Duration:{ripple_duration:.2f} ms" 
                    OE.plot_ripple_overlay (ax[2],LFP_ep,SPAD_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=True)
                ax[0].axvline(0, color='white',linewidth=2)
                ax[1].axvline(0, color='white',linewidth=2)
                ax[2].axvline(0, color='white',linewidth=2) 
                '''Using the following codes, I only plot the ripple epoch'''
                # LFP_ep=LFP.restrict(rip_ep.iloc[[i]])
                # SPAD_ep=SPAD_smooth.restrict(rip_ep.iloc[[i]])
                # ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_ep.iloc[[i]])
                # sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=1500,Fs=self.fs,scale=40)              
                # time = np.arange(len(sst_ep)) *(1/self.fs)
                # #Set the title of ripple feature
                # ripple_std=rip_tsd.iloc[i]
                # ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]
                # plot_title = "Filtered ripple"
                # OE.plot_ripple_overlay (ax[2],LFP_ep,SPAD_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=True)
                # t=event_peak_times[i]-rip_ep['start'][i]
                # ax[2].axvline(t, color='white',linewidth=2)
            cross_corr_values = np.array(cross_corr_values)
            # Truncate all columns to the common length
            common_length = min(len(column) for column in cross_corr_values)
            truncated_corr_array = np.array([column[1:common_length-1] for column in cross_corr_values])
            mean_cross_corr = np.mean(truncated_corr_array, axis=0)
            std_cross_corr = np.std(truncated_corr_array, axis=0)
            
            x = lags[1:common_length-1]/self.fs
            
            plt.figure(figsize=(10, 5))
            plt.plot(x, mean_cross_corr, color='b', label='Mean Cross-Correlation')
            plt.fill_between(x, mean_cross_corr - std_cross_corr, mean_cross_corr + std_cross_corr, color='gray', alpha=0.3, label='Standard Deviation')
            plt.xlabel('Lags(seconds)')
            plt.ylabel('Cross-Correlation')
            plt.title('Mean Cross-Correlation with Standard Deviation (1-Second Window)')
            plt.legend()
            plt.grid()
            plt.show()
            
        return ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd,cross_corr_values

    def pynappleThetaAnalysis (self,lfp_channel='LFP_2',ep_start=0,ep_end=10,Low_thres=1,High_thres=10,plot_ripple_ep=True):
        'This is the LFP data that need to be saved for the sync ananlysis'
        data_segment=self.Ephys_tracking_spad_aligned
        #data_segment=self.non_theta_part
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        timestamps=timestamps-timestamps[0]
        #Use non-theta part to detect ripple
        lfp_data=data_segment[lfp_channel]
        spad_data=data_segment['zscore_raw']
        lfp_data=lfp_data/1000 #change the unit from uV to mV
        SPAD_cutoff=20
        SPAD_smooth_np = OE.smooth_signal(spad_data,Fs=self.fs,cutoff=SPAD_cutoff)
        'To align LFP and SPAD raw data to pynapple format'
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        SPAD=nap.Tsd(t = timestamps, d = spad_data.to_numpy(), time_units = 's')
        SPAD_smooth=nap.Tsd(t = timestamps, d = SPAD_smooth_np, time_units = 's')
        if self.IsTracking:
            speed=nap.Tsd(t = timestamps, d = data_segment['speed_abs'].to_numpy(), time_units = 's')
        
        'To detect theta'
        theta_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getThetaEvents (LFP,self.fs,windowlen=2000,Low_thres=Low_thres,High_thres=High_thres)    
        'To plot the choosen segment'
        ex_ep = nap.IntervalSet(start = ep_start, end = ep_end, time_units = 's') 
        fig, ax = plt.subplots(6, 1, figsize=(10, 12))
        OE.plot_trace_nap (ax[0], LFP,ex_ep,color=sns.color_palette("husl", 8)[5],title='LFP raw Trace')
        OE.plot_trace_nap (ax[1], theta_band_filtered,ex_ep,color=sns.color_palette("husl", 8)[5],title='Theta band')
        OE.plot_ripple_event (ax[2], rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres=Low_thres) 
                
        print('LFP length:',len(LFP))
        print('SPAD length:',len(SPAD))
        print('SPAD_smooth length:',len(SPAD_smooth_np))
        print('theta_band_filtered length:',len(theta_band_filtered))
        
        OE.plot_trace_nap (ax[3], SPAD_smooth,ex_ep,color='green',title='calcium recording (z-score)')
        LFP_thetaband=OE.band_pass_filter(LFP.restrict(ex_ep),4,20,Fs=self.fs)        
        sst,frequency,power,global_ws=OE.Calculate_wavelet(LFP_thetaband,lowpassCutoff=50,Fs=self.fs,scale=200)
        OE.plot_wavelet(ax[4],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=True)
        
        if self.IsTracking:
            OE.plot_trace_nap (ax[5], speed,ex_ep,color='grey',title='speed (cm/second)')
            ax[5].set_ylim(0,10)
        #OE.plot_ripple_spectrum (ax[4], LFP, ex_ep,y_lim=30,Fs=self.fs,vmax_percentile=100)
        plt.subplots_adjust(hspace=0.5)
    
        print('LFP_thetaband length:',len(LFP_thetaband))
        '''To calculate cross-correlation'''
        cross_corr_values = []
        if plot_ripple_ep:
            event_peak_times=rip_tsd.index.to_numpy()
            for i in range(1,len(rip_ep)-1):
                fig, ax = plt.subplots(3, 1, figsize=(10, 18))
                ripple_std=rip_tsd.iloc[i]
                ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  
                if event_peak_times[i]>0.5:
                    start_time=event_peak_times[i]-0.5
                    end_time=event_peak_times[i]+0.5
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_ep=SPAD.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    ripple_band_filtered_ep=theta_band_filtered.restrict(rip_long_ep)                    
                    #Set the title of ripple feature
                    plot_title = "Optical signal from SPAD" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(SPAD_ep,lowpassCutoff=500,Fs=self.fs,scale=400)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    OE.plot_theta_overlay (ax[0],LFP_ep,SPAD_smooth_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=True,plotTheta=False)                                   
                    #Set the title of ripple feature
                    plot_title = "Local Field Potential with Spectrogram" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=100,Fs=self.fs,scale=400) 
                    OE.plot_theta_overlay (ax[1],LFP_ep,SPAD_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotTheta=False)
                    lags,cross_corr =OE.calculate_correlation_with_detrend (SPAD_ep,LFP_ep)
                    #cross_corr = signal.correlate(spad_1, lfp_1, mode='full')
                    cross_corr_values.append(cross_corr)
                    print('LFP_ep length:',len(LFP_ep))
                    print('SPAD_ep length:',len(SPAD_ep))
                    print('Cross corr:',len(cross_corr))
                    
                if event_peak_times[i]>0.5:
                    start_time=event_peak_times[i]-0.5
                    end_time=event_peak_times[i]+0.5
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_ep=SPAD.restrict(rip_long_ep)
                    ripple_band_filtered_ep=theta_band_filtered.restrict(rip_long_ep)
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=1500,Fs=self.fs,scale=200)                
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    #Set the title of ripple feature
                    plot_title = f"Ripple Peak std:{ripple_std:.2f}, Ripple Duration:{ripple_duration:.2f} ms" 
                    OE.plot_theta_overlay (ax[2],LFP_ep,SPAD_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=False,plotTheta=True)
                ax[0].axvline(0, color='white',linewidth=2)
                ax[1].axvline(0, color='white',linewidth=2)
                ax[2].axvline(0, color='white',linewidth=2) 
                '''Using the following codes, I only plot the ripple epoch'''
                # LFP_ep=LFP.restrict(rip_ep.iloc[[i]])
                # SPAD_ep=SPAD_smooth.restrict(rip_ep.iloc[[i]])
                # ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_ep.iloc[[i]])
                # sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=1500,Fs=self.fs,scale=40)              
                # time = np.arange(len(sst_ep)) *(1/self.fs)
                # #Set the title of ripple feature
                # ripple_std=rip_tsd.iloc[i]
                # ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]
                # plot_title = "Filtered ripple"
                # OE.plot_ripple_overlay (ax[2],LFP_ep,SPAD_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=True)
                # t=event_peak_times[i]-rip_ep['start'][i]
                # ax[2].axvline(t, color='white',linewidth=2)
            cross_corr_values = np.array(cross_corr_values)
            # Truncate all columns to the common length
            common_length = min(len(column) for column in cross_corr_values)
            truncated_corr_array = np.array([column[1:common_length-1] for column in cross_corr_values])
            mean_cross_corr = np.mean(truncated_corr_array, axis=0)
            std_cross_corr = np.std(truncated_corr_array, axis=0)
            
            x = lags[1:common_length-1]/self.fs
            
            plt.figure(figsize=(10, 5))
            plt.plot(x, mean_cross_corr, color='b', label='Mean Cross-Correlation')
            plt.fill_between(x, mean_cross_corr - std_cross_corr, mean_cross_corr + std_cross_corr, color='gray', alpha=0.3, label='Standard Deviation')
            plt.xlabel('Lags(seconds)')
            plt.ylabel('Cross-Correlation')
            plt.title('Mean Cross-Correlation with Standard Deviation (1-Second Window)')
            plt.legend()
            plt.grid()
            plt.show()
            
        return theta_band_filtered,nSS,nSS3,rip_ep,rip_tsd,cross_corr_values

