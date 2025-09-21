# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:03:35 2023
@author: Yifang

This is the Class that used to form a synchronised dataset including LFP channel signals, pyPhotometry recorded optical signal as zscore, and animal position.
Note:
Some of the functions are named as SPAD*** just because I made this for SPAD-ephys analysis first, then modified it for pyPhotometry recordings.
I named it as SyncOECSessionClass but it is actually a single recording trial. 
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import OpenEphysTools as OE
import pynapple as nap
import MakePlots
import pynacollada as pyna
import pickle
from SPADPhotometryAnalysis import photometry_functions as fp
import re

class SyncOEpyPhotometrySession:
    def __init__(
        self,
        SessionPath,
        recordingName,
        IsTracking=False,
        read_aligned_data_from_file=False,
        recordingMode='Atlas',
        indicator='GEVI',
        tracking_source='Bonsai',              # NEW: 'Bonsai' or 'DLC'
        dlc_bodyparts=('head','shoulder','bottom'),  # NEW: which DLC parts to load
    ):
        '''
        Parameters
        ----------
        SessionPath : path to save data for a single recording trial with ephys and optical data. 
        IsTracking: 
            whether to include animal position tracking data, not neccessary for sleeping sessions.
        read_aligned_data_from_file: 
            False if it is the first time you analyse this trial of data, 
            once you have removed noises, run the single-trial analysis, saved the .pkl file, 
            set it to true to read aligned data directly.
        '''
        self.recordingMode=recordingMode
        self.indicator=indicator
        'Define photometry recording sampling rate by recording mode'
        if self.recordingMode=='py':
            self.pyPhotometry_fs = 130
        if self.recordingMode=='SPAD':
            self.Spad_fs = 9938.4
        if self.recordingMode=='Atlas':
            self.Spad_fs = 841.68
            #self.Spad_fs = 1682.92
            
        self.ephys_fs = 30000
        self.tracking_fs = 10
        self.fs = 10000
        
        self.recordingName=recordingName
        self.dpath=os.path.join(SessionPath, self.recordingName) #Recording pre-processed data path:'SyncRecording*' folder
        self.IsTracking=IsTracking
        self.tracking_source = tracking_source
        self.dlc_bodyparts = tuple([bp.lower() for bp in dlc_bodyparts])
        
        # --- NEW: fixed speed/movement parameters ---
        self.cm_per_px_x = 0.0525
        self.cm_per_px_y = 0.0525
        self.speed_threshold_cm_s = 3.0
        self.min_moving_duration_s = 0
        self.speed_bodypart_preference = ('shoulder', 'head', 'bottom')
        
        # if self.IsTracking:
        #     self.ReadTrialAnimalState(SessionPath)
        
        if (read_aligned_data_from_file):
            filepath=os.path.join(self.dpath, "Ephys_tracking_photometry_aligned.pkl")
            self.Ephys_tracking_spad_aligned = pd.read_pickle(filepath)
            duration= len(self.Ephys_tracking_spad_aligned['timestamps'])/self.fs
            self.Ephys_tracking_spad_aligned['timestamps']= np.linspace(0, duration, len(self.Ephys_tracking_spad_aligned['timestamps']), endpoint=False)
        else:
            self.Ephys_data=self.read_open_ephys_data() #read ephys data that we pre-processed from dpath
            if self.recordingMode=='py':
                self.Sync_ephys_with_pyPhotometry() #output self.spad_align, self.ephys_align
            elif self.recordingMode=='SPAD' or self.recordingMode=='Atlas':
                self.Sync_ephys_with_spad()
                if self.indicator == 'GEVI':
                    self.Ephys_tracking_spad_aligned['zscore_raw']=-self.Ephys_tracking_spad_aligned['zscore_raw']  
                    self.Ephys_tracking_spad_aligned['sig_raw']=-self.Ephys_tracking_spad_aligned['sig_raw'] 
                    self.Ephys_tracking_spad_aligned['ref_raw']=-self.Ephys_tracking_spad_aligned['ref_raw'] 
            'Label REM and nonREM sleep'
            self.Label_REM_sleep ('LFP_1')
            'Speed calculation was done before resample'
            self.save_data(self.Ephys_tracking_spad_aligned, 'Ephys_tracking_photometry_aligned.pkl')
        
        self.savepath = os.path.join(SessionPath, "Results")
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath) 
          
        if not read_aligned_data_from_file:
            del self.Ephys_data
            del self.Ephys_sync_data
            del self.PhotometryData
            del self.ephys_resampled
            del self.ephys_align
            del self.photometry_sync_data
            del self.photometry_align
            del self.py_resampled
            if self.IsTracking:
                del self.trackingdata
                del self.trackingdata_resampled
                del self.trackingdata_align

    def ReadTrialAnimalState(self,SessionPath):
        SessionLabelFile=os.path.join(SessionPath,'TrailLabel.csv')
        self.SessionLabelData = pd.read_csv(SessionLabelFile)
        index = self.recordingName.find('SyncRecording')
        if index != -1:
            # Extract the number part of the file name
            number = self.recordingName[index + len('SyncRecording'):]
            recording_number = int(number)
            print("Recording number:", recording_number)
        else:
            print("File name does not contain 'SyncRecording'") 
        self.sleepState=self.SessionLabelData['sleepState'][recording_number-1]
        self.movingState=self.SessionLabelData['movingState'][recording_number-1]
        self.TrainingState=self.SessionLabelData['TrainingState'][recording_number-1]
        return -1
    		
    def Sync_ephys_with_pyPhotometry(self):
        '''
        Main function to read all saved decoded ephys and optical data and save them as a signal Pandas dataFrame.
        The algorithm is (1) to read Ephys data, photometry data and camera tracking data respectively;
        (2) using the camSync as a mask to cut the period within our sync pulse;
        (3) format the index to timedelta index and resample all the three data sources;
        (4) concatenate them to a single DataFrame.
        '''
        self.form_ephys_sync_data() # find the spad sync part
        self.Format_ephys_data_index () # this is to change the index to timedelta index, for resampling
        self.PhotometryData=self.Read_photometry_data() #read pyPhotometry data
        self.photometry_sync_data=self.form_photometry_sync_data () # find the photometry sync part,i.e. the same part with sync pulses
        self.resample_photometry() #resampling
        self.resample_ephys()
        if self.IsTracking:
            self.read_tracking_data() #read tracking raw data   
            self.resample_tracking_to_ephys()    
            self.slice_to_align_with_min_len() #since two data sources are with different sampling rate, the resampled data length are not the same, but similiar.
            self.photometry_align = self.photometry_align.set_index(self.ephys_align.index)
            self.trackingdata_align = self.trackingdata_align.set_index(self.ephys_align.index)
            self.Ephys_tracking_spad_aligned=pd.concat([self.ephys_align, self.photometry_align, self.trackingdata_align], axis=1)  #define a new DataFrame to save both LPF and optical signal
        if not self.IsTracking:
            self.slice_to_align_with_min_len() #since two data sources are with different sampling rate, the resampled data length are not the same, but similiar.
            self.photometry_align = self.photometry_align.set_index(self.ephys_align.index)
            self.Ephys_tracking_spad_aligned=pd.concat([self.ephys_align, self.photometry_align], axis=1)
        self.Ephys_tracking_spad_aligned.reset_index(drop=True, inplace=True) 
                
        while True: #remove noise by cutting part of the synchronised the data
            OE.plot_two_traces_in_seconds (self.Ephys_tracking_spad_aligned['zscore_raw'],self.fs, 
                                       self.Ephys_tracking_spad_aligned['LFP_4'], self.fs, label1='zscore_raw',label2='LFP_4')
            start_time = input("Enter the start time to move noise (or 'q' to quit): ")
            if start_time.lower() == 'q':
                break
            end_time = input("Enter the end time to move noise (or 'q' to quit): ")
            if end_time.lower() == 'q':
                break 
            print(f"Start time: {start_time}, End time: {end_time}")
            self.remove_noise(start_time=int(start_time),end_time=int(end_time))
        return -1
    
    def Sync_ephys_with_spad(self):
        'This is to read SPAD photometry data, the only difference is that SPAD recording does not have a CamSync line'
        'So we just read SPAD post-processed data from .csv, format the index and resample. '            
        self.form_ephys_spad_sync_data() # find the spad sync part
        self.Format_ephys_data_index () # this is to change the index to timedelta index, for resampling
        self.PhotometryData=self.Read_photometry_data() #read spad data
        self.Format_SPAD_data_index ()
        self.resample_photometry()
        self.resample_ephys()
        #self.slice_ephys_to_align_with_spad()
        
        if self.IsTracking:
            self.trackingdata=self.read_tracking_data() #read tracking raw data
            print ('Total length of tracking (seconds):', len(self.trackingdata)/self.tracking_fs)
            self.form_tracking_spad_sync_data() 
            self.resample_tracking_to_ephys() 
            self.slice_to_align_with_min_len()
            self.photometry_align = self.photometry_align.set_index(self.ephys_align.index)
            self.trackingdata_align = self.trackingdata_align.set_index(self.ephys_align.index)
            self.Ephys_tracking_spad_aligned=pd.concat([self.ephys_align, self.photometry_align, self.trackingdata_align], axis=1)        
        if not self.IsTracking:
            'slice_to_align_with_min_len() is important because sometimes the effective SPAD recording is shorter than the real recording time due to deadtime. '
            'E.g, I recorded 10 blocks 10s data, should be about 100s recording, but in most cases, there is no data in the last block.'
            self.slice_to_align_with_min_len()
            self.photometry_align = self.photometry_align.set_index(self.ephys_align.index)
            self.Ephys_tracking_spad_aligned=pd.concat([self.ephys_align, self.photometry_align], axis=1)
            
        self.Ephys_tracking_spad_aligned.reset_index(drop=True, inplace=True)  
        
        OE.plot_two_traces_in_seconds (self.Ephys_tracking_spad_aligned['zscore_raw'],self.fs, 
                                       self.Ephys_tracking_spad_aligned['LFP_2'], self.fs, label1='zscore_raw',label2='LFP_2')
        
        while True: #remove noise by cutting part of the synchronised the data
            start_time = input("Enter the start time to move noise (or 'q' to quit): ")
            if start_time.lower() == 'q':
                break
            end_time = input("Enter the end time to move noise (or 'q' to quit): ")
            if end_time.lower() == 'q':
                break 
            print(f"Start time: {start_time}, End time: {end_time}")
            self.remove_noise(start_time=int(start_time),end_time=int(end_time))
        
            OE.plot_two_traces_in_seconds (self.Ephys_tracking_spad_aligned['zscore_raw'],self.fs, 
                                       self.Ephys_tracking_spad_aligned['LFP_2'], self.fs, label1='zscore_raw',label2='LFP_2')
        return -1
    
    def read_open_ephys_data (self):
        filepath=os.path.join(self.dpath, "open_ephys_read_pd.pkl")
        self.Ephys_data = pd.read_pickle(filepath)  
        return self.Ephys_data       
    
    def form_ephys_sync_data (self):
        mask = self.Ephys_data['py_mask'] 
        self.Ephys_sync_data=self.Ephys_data[mask]
        OE.plot_two_traces_in_seconds (mask,self.ephys_fs,self.Ephys_data['LFP_1'],self.ephys_fs,
                                       label1='Cam_mask',label2='LFP_raw_data') 
        print ('Ephys data length', len(self.Ephys_data)/self.ephys_fs)
        print ('Ephys synced part data length', len(self.Ephys_sync_data)/self.ephys_fs)
        return -1     
    def form_ephys_spad_sync_data (self):
        
        mask = self.Ephys_data['SPAD_mask'] 
        self.Ephys_sync_data=self.Ephys_data[mask]
        # OE.plot_two_raw_traces (mask,self.Ephys_sync_data['LFP_1'], spad_label='spad_mask',lfp_label='LFP_raw')
        # OE.plot_trace_in_seconds(self.Ephys_sync_data['LFP_1'], self.ephys_fs,title='Ephys sync part data')
        print ('Ephys data length', len(self.Ephys_data)/self.ephys_fs)
        print ('Ephys synced part data length', len(self.Ephys_sync_data)/self.ephys_fs)
        return -1  

    def Format_ephys_data_index (self):
        time_interval = 1.0 / self.ephys_fs
        total_duration = len(self.Ephys_sync_data) * time_interval
        timestamps = np.arange(0, total_duration, time_interval)
        timedeltas_index = pd.to_timedelta(timestamps, unit='s')            
        self.Ephys_sync_data.index = timedeltas_index
        return -1
    
    def Format_SPAD_data_index (self):
        print ('SPAD data length', len(self.PhotometryData)/self.Spad_fs)
        time_interval = 1.0 / self.Spad_fs
        total_duration = len(self.PhotometryData) * time_interval
        timestamps = np.arange(0, total_duration, time_interval)
        timedeltas_index = pd.to_timedelta(timestamps, unit='s')
        self.photometry_sync_data=self.PhotometryData
        self.photometry_sync_data.index = timedeltas_index
        return self.photometry_sync_data
        
    def remove_noise(self,start_time,end_time):
        start_idx =int(start_time * self.fs)# Time interval in seconds   
        end_idx = int(end_time * self.fs)# Time interval in seconds
        mask = ~self.Ephys_tracking_spad_aligned.index.isin(range(start_idx, end_idx + 1))
        self.Ephys_tracking_spad_aligned = self.Ephys_tracking_spad_aligned[mask]
        self.Ephys_tracking_spad_aligned = self.Ephys_tracking_spad_aligned.reset_index(drop=True)
        return self.Ephys_tracking_spad_aligned

    def Read_photometry_data (self):
        '''pyPhotometr sampling rate is 130 Hz.'''
        self.sig_csv_filename=os.path.join(self.dpath, "Green_traceAll.csv")
        self.ref_csv_filename=os.path.join(self.dpath, "Red_traceAll.csv")
        self.zscore_csv_filename=os.path.join(self.dpath, "Zscore_traceAll.csv")
        self.CamSync_photometry_filename=os.path.join(self.dpath, "CamSync_photometry.csv")
        sig_data = np.genfromtxt(self.sig_csv_filename, delimiter=',')
        ref_data = np.genfromtxt(self.ref_csv_filename, delimiter=',')
        zscore_data = np.genfromtxt(self.zscore_csv_filename, delimiter=',')
        if self.recordingMode=='Atlas':
            zscore_data=OE.notchfilter (zscore_data,f0=100,bw=2,fs=840)
            sig_data=OE.notchfilter (sig_data,f0=100,bw=2,fs=840)
            ref_data=OE.notchfilter (ref_data,f0=100,bw=2,fs=840)
        sig_raw = pd.Series(sig_data)
        ref_raw = pd.Series(ref_data)
        zscore_raw = pd.Series(zscore_data)
        if self.recordingMode=='py':
            CamSync_photometry_data=np.genfromtxt(self.CamSync_photometry_filename, delimiter=',')
            CamSync_photometry=pd.Series(CamSync_photometry_data)
            'Zscore data is obtained by Kate Martian method, smoothed to 250Hz effective sampling rate'
            self.PhotometryData = pd.DataFrame({
                'sig_raw': sig_raw,
                'ref_raw': ref_raw,
                'zscore_raw': zscore_raw,
                'Cam_Sync':CamSync_photometry
            })
        else:
            self.PhotometryData = pd.DataFrame({
                'sig_raw': sig_raw,
                'ref_raw': ref_raw,
                'zscore_raw': zscore_raw,
            }) 
        return self.PhotometryData
    
    def form_photometry_sync_data (self):
        CamSync=self.PhotometryData['Cam_Sync']
        py_mask=np.zeros(len(CamSync),dtype=int)
        #indices = np.where(CamSync > 0.5)[0]
        py_mask[np.where(CamSync >0.5)[0]]=1
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
        py_mask_final=np.zeros(len(CamSync),dtype=int)        
        py_mask_final[rising_edge_index:falling_edge_index]=1

        mask_array_bool = np.array(py_mask_final, dtype=bool)
        self.PhotometryData['mask']=mask_array_bool
        self.photometry_sync_data=self.PhotometryData[mask_array_bool]  
        time_interval = 1.0 / self.pyPhotometry_fs
        total_duration = len(self.photometry_sync_data) * time_interval
        timestamps = np.arange(0, total_duration, time_interval)
        timestamps_time = pd.to_timedelta(timestamps, unit='s')
        #To make the timestamp always the same length with the data, otherwise one might be 1 data longer than the other
        if len(self.photometry_sync_data)>len(timestamps_time):
            self.photometry_sync_data=self.photometry_sync_data[:len(timestamps_time)]
        elif len(self.photometry_sync_data)<len(timestamps_time):
            timestamps_time=timestamps_time[:len(self.photometry_sync_data)]    
        self.photometry_sync_data.index = timestamps_time
        #OE.plot_two_raw_traces (self.PhotometryData['mask'],self.photometry_sync_data['zscore_raw'], spad_label='Cam_mask',lfp_label='zScore_raw_within_mask')
        OE.plot_two_traces_in_seconds (self.PhotometryData['mask'],self.pyPhotometry_fs,
                                       self.PhotometryData['zscore_raw'],self.pyPhotometry_fs,
                                       label1='Cam_mask',label2='zScore_raw_within_mask') 
        
        print ('pyPhotometry data length', len(self.PhotometryData)/self.pyPhotometry_fs)
        print ('pyPhotometry synced part data length', len(self.photometry_sync_data)/self.pyPhotometry_fs)
        absolute_difference = abs(len(self.photometry_sync_data)/self.pyPhotometry_fs - len(self.Ephys_sync_data)/self.ephys_fs)
        if absolute_difference>0.05:
            print ('NOTE!!!Synchronised mask matched wrong!')
            print('Detected a difference of the sync mask durations larger than 50ms.Please make sure pyPhotometry data and ephys data are from the same trial.')
            print ('The pipeline will cut the longer recording automatically')
        else:
            print ('Yay~~Sync mask matched! Synchronising LFP and Optical signal finished.')          
        return self.photometry_sync_data
    
    def resample_photometry(self):
        if self.recordingMode=='py':
            original_fs = self.pyPhotometry_fs
        if self.recordingMode=='SPAD':
            original_fs= self.Spad_fs
        if self.recordingMode=='Atlas':
            original_fs= self.Spad_fs
        self.py_resampled = OE.resample_signal(self.photometry_sync_data, original_fs, self.fs)
        return self.py_resampled

    def resample_ephys(self):
        original_fs = self.ephys_fs  # use the known original sampling rate
        self.ephys_resampled = OE.resample_signal(self.Ephys_sync_data, original_fs, self.fs)
        return self.ephys_resampled

    # def resample_photometry (self):
    #     time_interval_common = 1.0 / self.fs
    #     self.py_resampled = self.photometry_sync_data.resample(f'{time_interval_common:.9f}S').mean()
    #     self.py_resampled = self.py_resampled.fillna(method='bfill')
    #     return self.py_resampled
    
    # def resample_ephys (self):
    #     time_interval_common = 1.0 / self.fs
    #     self.ephys_resampled = self.Ephys_sync_data.resample(f'{time_interval_common:.9f}S').mean()
    #     self.ephys_resampled = self.ephys_resampled.interpolate()
    #     return self.ephys_resampled                     
    
    def slice_to_align_with_min_len (self):
        'This is important with different sampling rate, the calculated durations might have a ~10ms difference.'
        if len(self.py_resampled)<len(self.ephys_resampled):
            self.ephys_align = self.ephys_resampled[:len(self.py_resampled)]
            self.photometry_align=self.py_resampled
        else:
            self.photometry_align = self.py_resampled[:len(self.ephys_resampled)]
            self.ephys_align=self.ephys_resampled
        if self.IsTracking:
            if len(self.trackingdata_resampled)>=len(self.photometry_align):
                self.trackingdata_align = self.trackingdata_resampled[:len(self.photometry_align)]
            else:
                #concatenate dummy frames at the begining
                add_to_start = pd.concat([self.trackingdata_resampled.iloc[[0]]] * (len(self.photometry_align) - len(self.trackingdata_resampled)) )
                self.trackingdata_align = pd.concat([add_to_start, self.trackingdata_resampled])
        return -1
    
    # def read_tracking_data (self):
    #     keyword='AnimalTracking'
    #     'Only for Bonsai tracking'
    #     files_in_directory = os.listdir(self.dpath)
    #     matching_files = [filename for filename in files_in_directory if keyword in filename]
    #     if matching_files:
    #         behaviour_file_path = os.path.join(self.dpath, matching_files[0])
    #         print ('---Reading behavioural tracking data----')
    #         self.trackingdata = pd.read_pickle(behaviour_file_path)
    #         OE.plot_animal_tracking (self.trackingdata)
    #     else:
    #         print ('No available Tracking data in the folder!')
    #     return self.trackingdata
    
    def _find_file(self, keyword, ext=None):
        """Find first file in self.dpath containing keyword (and optional extension)."""
        files = os.listdir(self.dpath)
        for fn in files:
            if keyword in fn and (ext is None or fn.lower().endswith(ext.lower())):
                return os.path.join(self.dpath, fn)
        return None

    def _read_tracking_bonsai(self):
        """
        Read Bonsai tracking from a pickled DataFrame.
        Accepts any file named like 'AnimalTracking_<index>.pkl' (index can vary).
        Returns the DataFrame as-is (e.g., with columns such as 'x','y' if present).
        """
        files_in_directory = os.listdir(self.dpath)
        matching = [fn for fn in files_in_directory
                    if fn.startswith("AnimalTracking_") and fn.lower().endswith(".pkl")]
        if not matching:
            raise FileNotFoundError("Bonsai PKL not found (expected 'AnimalTracking_<index>.pkl').")
        # If multiple, choose the first in sorted order (customise if you prefer mtime)
        matching.sort()
        behaviour_file_path = os.path.join(self.dpath, matching[0])
        print(f"---Reading Bonsai tracking (PKL): {os.path.basename(behaviour_file_path)}")
        df = pd.read_pickle(behaviour_file_path)
        return df
    


    def _read_tracking_dlc_csv(self, bodyparts=('head','shoulder','bottom')):
        """
        Read a DLC CSV with a 3-row MultiIndex header and extract x/y for specified bodyparts.
        Matches any filename containing 'AnimalVideo_<number>DLC_Resnet50_' (case-insensitive).
        """
        pattern = re.compile(r"AnimalVideo_\d+DLC_Resnet50_", re.IGNORECASE)
    
        candidates = [
            fn for fn in os.listdir(self.dpath)
            if fn.lower().endswith(".csv") and pattern.search(fn)
        ]
        if not candidates:
            raise FileNotFoundError(
                "DLC CSV not found. Expected a filename containing 'AnimalVideo_<n>DLC_Resnet50_'."
            )
    
        candidates.sort()
        path = os.path.join(self.dpath, candidates[0])
        print(f"---Reading DLC tracking (CSV): {os.path.basename(path)}")
    
        # Read with MultiIndex header
        df_mi = pd.read_csv(path, header=[0, 1, 2])
    
        # Collect requested bodyparts (x/y)
        want = {bp.lower() for bp in bodyparts}
        cols = {}
        for col in df_mi.columns:
            scorer, bp, coord = col
            bp_l = str(bp).strip().lower()
            coord_l = str(coord).strip().lower()
            if bp_l in want and coord_l in ("x", "y"):
                cols[f"{bp_l}_{coord_l}"] = col
    
        if not cols:
            raise ValueError(f"No requested DLC bodyparts found. Asked for: {bodyparts}")
    
        # Build output DataFrame
        out = pd.DataFrame(index=df_mi.index)
        for k, col in sorted(cols.items()):
            out[k] = pd.to_numeric(df_mi[col], errors="coerce")
    
        return out
    
    def read_tracking_data(self):
        """
        Unified tracking reader controlled by self.tracking_source.
        - 'Bonsai': returns ['x','y']
        - 'DLC': returns ['head_x','head_y','shoulder_x','shoulder_y','bottom_x','bottom_y'] (subset if missing)
        Also triggers a quick plot (if desired) via OE.plot_animal_tracking.
        """
        print('---Reading behavioural tracking data----')
        if self.tracking_source.lower() == 'bonsai':
            self.trackingdata = self._read_tracking_bonsai()
        elif self.tracking_source.lower() == 'dlc':
            self.trackingdata = self._read_tracking_dlc_csv(self.dlc_bodyparts)
        else:
            raise ValueError("tracking_source must be 'Bonsai' or 'DLC'.")
        # Optional: quick sanity plot if your OE helper accepts multi-column
        try:
            OE.plot_animal_tracking(self.trackingdata)
        except Exception:
            pass
    
        return self.trackingdata
    
    def form_tracking_spad_sync_data (self):
        spad_mask = self.Ephys_data['SPAD_mask'] 
        cam_mask=self.Ephys_data['cam_mask']
        OE.plot_trace_in_seconds(spad_mask,self.ephys_fs,title='SPAD mask')
        OE.plot_trace_in_seconds(cam_mask,self.ephys_fs,title='Cam mask')
        spad_start_idx=spad_mask.idxmax()
        cam_start_idx=cam_mask.idxmax()
        # print ('spad_start_index in Ephys---', spad_start_idx)
        # print ('cam_start_index in Ephys---', cam_start_idx)
        time_diff=(spad_start_idx-cam_start_idx)/self.ephys_fs
        mask_start_idx=int(time_diff*self.tracking_fs) #spad mask in tracking data
        time_duration=spad_mask.sum()/self.ephys_fs
        #print ('time_duration of SPAD mask---', time_duration)
        mask_end_idx=int(mask_start_idx+time_duration*self.tracking_fs)
        # print ('spad_mask_in_tracking_start_idx---', mask_start_idx)
        # print ('spad_mask_in tracking_end_idx---', mask_end_idx)
        self.trackingdata=self.trackingdata[mask_start_idx:mask_end_idx]
        print ('Tracking data length during SPAD mask (seconds)---', len(self.trackingdata)/self.tracking_fs)
        return -1  
    
    def resample_tracking_to_ephys(self):
        # ----- build tracking time index at the native tracking Fs -----
        time_interval = 1.0 / self.tracking_fs
        total_duration = len(self.trackingdata) * time_interval
        timestamps = np.arange(0, total_duration, time_interval)
        timestamps_time = pd.to_timedelta(timestamps, unit='s')
        if len(self.trackingdata) > len(timestamps_time):
            self.trackingdata = self.trackingdata[:len(timestamps_time)]
        elif len(self.trackingdata) < len(timestamps_time):
            timestamps_time = timestamps_time[:len(self.trackingdata)]
        self.trackingdata.index = timestamps_time
    
        print(f'Behavioural data length {total_duration:.3f} seconds')
        absolute_difference = abs(total_duration - len(self.Ephys_sync_data)/self.ephys_fs)
        if absolute_difference > 0.3:
            print('NOTE!!! Behavioural tracking data length might be wrong!')
            print('Detected a difference between Behavioural Cam and Recording > 300 ms.')
            print('If not changing, the pipeline will align the length automatically')
        else:
            print('Yay~~Camera time mask matched! Synchronising LFP and Optical signal finished.')
    
        # ----- NEW: compute speed & movement at tracking Fs BEFORE resampling -----
        speed_tracking = self._compute_speed_from_tracking_pre_resample(self.trackingdata)
        move_tracking  = self._classify_movement_at_tracking_fs(speed_tracking)
        # attach to the tracking dataframe (still at tracking Fs)
        self.trackingdata['speed']    = speed_tracking
        self.trackingdata['movement'] = move_tracking
    
        # ----- resample everything to the common Fs (e.g., 10 kHz) -----
        time_interval_common = 1.0 / self.fs
    
        # For categorical 'movement', resample as 0/1 then map back
        move_map = {'moving': 1.0, 'notmoving': 0.0}
        df_to_resample = self.trackingdata.copy()
        df_to_resample['movement'] = df_to_resample['movement'].map(move_map)
    
        # pandas Resampler may not have .pad(); use .ffill() instead
        self.trackingdata_resampled = df_to_resample.resample(f'{time_interval_common:.9f}S').ffill()
    
        # convert movement back to labels
        self.trackingdata_resampled['movement'] = (
            self.trackingdata_resampled['movement']
            .round()
            .map({1.0: 'moving', 0.0: 'notmoving'})
            .astype('object')
        )
    
        # ensure no leading NaNs after resample
        self.trackingdata_resampled = self.trackingdata_resampled.fillna(method='bfill')
    
        return self.trackingdata_resampled
    
    def save_data (self, data,filename):
        filepath=os.path.join(self.dpath, filename)
        data.to_pickle(filepath)
        return -1      
    
    def slicing_pd_data (self, data,start_time, end_time):
        # Start time in seconds
        # End time in seconds
        start_idx =int( start_time * self.fs)# Time interval in seconds   
        end_idx = int(end_time * self.fs)# Time interval in seconds
        silced_data=data.iloc[start_idx:end_idx]
        return silced_data
    
    def slicing_np_data (self, data,start_time, end_time):
        # Start time in seconds
        # End time in seconds
        start_idx =int( start_time * self.fs)# Time interval in seconds   
        end_idx = int(end_time * self.fs)# Time interval in seconds
        silced_data=data[start_idx:end_idx]
        return silced_data    
        
    def _select_xy_columns_for_speed(self, df):
        """Pick (x, y) columns. Prefer shoulder_x/shoulder_y; fallback to head/bottom; else Bonsai x,y."""
        cols = {c.lower(): c for c in df.columns}
        # DLC priority list
        for bp in self.speed_bodypart_preference:
            xk, yk = f"{bp}_x", f"{bp}_y"
            if xk in cols and yk in cols:
                return cols[xk], cols[yk]
        # Bonsai default
        if 'x' in cols and 'y' in cols:
            return cols['x'], cols['y']
        # DLC fallback: any prefix with _x/_y
        prefixes = {n[:-2] for n in cols if n.endswith('_x')}
        for pre in prefixes:
            if f"{pre}_y" in cols:
                return cols[f"{pre}_x"], cols[f"{pre}_y"]
        raise ValueError("No suitable (x,y) columns found for speed.")

# --- NEW: compute speed (cm/s) on the aligned dataframe ---
    def _compute_speed_from_tracking_pre_resample(self, tracking_df):
        """
        Compute speed (cm/s) at the *tracking* sampling rate (e.g. 10 Hz), BEFORE upsampling.
        """
        x_col, y_col = self._select_xy_columns_for_speed(tracking_df)
        x = pd.to_numeric(tracking_df[x_col], errors='coerce').interpolate(limit_direction='both')
        y = pd.to_numeric(tracking_df[y_col], errors='coerce').interpolate(limit_direction='both')
    
        # Per-frame displacement in cm (anisotropic scaling supported)
        dx_cm = x.diff() * self.cm_per_px_x
        dy_cm = y.diff() * self.cm_per_px_y
        ds_cm = np.sqrt(dx_cm**2 + dy_cm**2)
    
        # Frames → seconds using the *tracking* fps
        speed = ds_cm * float(self.tracking_fs)      # (cm/frame) * (frames/s) = cm/s
        speed.iloc[0] = 0.0
    
        # Optional: light smoothing at 10 Hz (e.g., 3 frames ≈ 0.3 s)
        speed = speed.rolling(3, min_periods=1, center=True).median()
    
        speed.name = 'speed'
        return speed
    
    # --- NEW: classify movement from speed ---
    def _classify_movement_at_tracking_fs(self, speed_series):
        """
        Classify at tracking Fs: 'moving' if speed > threshold for >= min_moving_duration_s consecutively.
        """
        arr = (speed_series > self.speed_threshold_cm_s).to_numpy()
        win = int(round(self.min_moving_duration_s * self.tracking_fs))
        if win < 1: win = 1
    
        # Run-length segmentation
        # pad with 0 at both ends to detect edges
        z = np.r_[0, arr.astype(int), 0]
        edges = np.diff(z)
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        keep = np.zeros_like(arr, dtype=bool)
        for s, e in zip(starts, ends):
            if (e - s) >= win:
                keep[s:e] = True
    
        lab = np.where(keep, 'moving', 'notmoving')
        return pd.Series(lab, index=speed_series.index, name='movement')
    
    def plot_freq_power_coherence (self,LFP_channel,start_time,end_time,SPAD_cutoff,lfp_cutoff):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        #SPAD_smooth= OE.butter_filter(data['zscore_raw'], btype='high', cutoff=0.5, fs=self.fs, order=5)
        SPAD_smooth= OE.smooth_signal(silced_recording['zscore_raw'],Fs=self.fs,cutoff=SPAD_cutoff)
        SPAD_smooth= OE.butter_filter(SPAD_smooth, btype='high', cutoff=2, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='high', cutoff=2, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        spad_low = pd.Series(SPAD_smooth, index=silced_recording['zscore_raw'].index)
        lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)
        
        fig, ax = plt.subplots(6, 1, figsize=(16, 8))
        OE.plot_trace_in_seconds_ax (ax[0],spad_low,self.fs,label='GEVI',color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)
        #spad_filtered=OE.band_pass_filter(spad_data,120,300,self.fs)
        sst_spad,frequency,power_spad,global_ws=OE.Calculate_wavelet(spad_low,lowpassCutoff=100,Fs=self.fs,scale=40)
        OE.plot_wavelet(ax[1],sst_spad,frequency,power_spad,Fs=self.fs,colorBar=False,logbase=False)
        lfp_low=lfp_low
        OE.plot_trace_in_seconds_ax (ax[2],lfp_low,self.fs,label='LFP',color=sns.color_palette("dark", 8)[7],ylabel='mV',xlabel=False)
        #lfp_data_filtered=OE.band_pass_filter(lfp_data,120,300,self.fs)
        sst_lfp,frequency,power_lfp,global_ws=OE.Calculate_wavelet(lfp_low,lowpassCutoff=500,Fs=self.fs,scale=40)
        OE.plot_wavelet(ax[3],sst_lfp,frequency,power_lfp,Fs=self.fs,colorBar=False,logbase=False)
        
        cross_power = np.abs(power_spad * np.conj(power_lfp))**2
        coherence = cross_power / (np.abs(power_spad) * np.abs(power_lfp))
        
        ax[1].set_ylim(0,20)
        ax[3].set_ylim(0,20)
        ax[3].set_xlabel('Time (seconds)')
        ax[0].legend().set_visible(False)
        ax[2].legend().set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['bottom'].set_visible(False)
        ax[3].spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[1].set_xticks([])  # Hide x-axis tick marks
        ax[1].set_xlabel([])
        ax[1].set_xlabel('')  # Hide x-axis label
        ax[2].set_xticks([])  # Hide x-axis tick marks
        ax[2].set_xlabel([])
        ax[2].set_xlabel('')  # Hide x-axis label
        ax[3].set_xticks([])  # Hide x-axis tick marks
        ax[3].set_xlabel([])
        ax[3].set_xlabel('')  # Hide x-axis label
        
        time = np.arange(len(sst_spad)) / self.fs  # Construct time array
        level = 8  # Number of contour levels you want
        # Custom color limits
        vmin = coherence.min()*1  # Replace with your desired minimum value
        vmax = coherence.max()*0.9  # Replace with your desired maximum value
        CS = ax[4].contourf(time, frequency, coherence, level, vmin=vmin, vmax=vmax)
        #OE.plot_wavelet(ax[4],sst_spad,frequency,coherence,Fs=self.fs,colorBar=True,logbase=True)

        #ax[4].set_yscale('log', base=2, subs=None)
        ax[4].set_ylim([np.min(frequency), np.max(frequency)])
        #yax = plt.gca().yaxis
        #yax.set_major_formatter(ticker.ScalarFormatter())

        ax[4].set_ylim(0,20)
        ax[4].set_ylabel('Frequency [Hz]')
        ax[4].set_xlabel('Time [s]')
        ax[4].set_title('Coherence between SPAD and LFP')
        
        SPAD_smooth= OE.smooth_signal(silced_recording['zscore_raw'],Fs=self.fs,cutoff=SPAD_cutoff)
        SPAD_smooth= OE.butter_filter(SPAD_smooth, btype='high', cutoff=15, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='high', cutoff=15, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        spad_low = pd.Series(SPAD_smooth, index=silced_recording['zscore_raw'].index)
        lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)
        
        sst_spad,frequency,power_spad,global_ws=OE.Calculate_wavelet(spad_low,lowpassCutoff=100,Fs=self.fs,scale=40)
        
        sst_lfp,frequency,power_lfp,global_ws=OE.Calculate_wavelet(lfp_low,lowpassCutoff=500,Fs=self.fs,scale=40)
        
        cross_power = np.abs(power_spad * np.conj(power_lfp))**2
        coherence = cross_power / (np.abs(power_spad) * np.abs(power_lfp))
        
        vmin = coherence.min()*1  # Replace with your desired minimum value
        vmax = coherence.max()*0.9  # Replace with your desired maximum value
        CS = ax[5].contourf(time, frequency, coherence, level, vmin=vmin, vmax=vmax)
        #OE.plot_wavelet(ax[4],sst_spad,frequency,coherence,Fs=self.fs,colorBar=True,logbase=True)

        #ax[5].set_yscale('log', base=2, subs=None)
        ax[5].set_ylim([np.min(frequency), np.max(frequency)])
        #yax = plt.gca().yaxis
        #yax.set_major_formatter(ticker.ScalarFormatter())

        ax[5].set_ylim(0,50)
        ax[5].set_ylabel('Frequency [Hz]')
        ax[5].set_xlabel('Time [s]')
        ax[5].set_title('Coherence above 15Hz')
        #plt.tight_layout()
        output_path=os.path.join(self.savepath,'makefigure','example_coherence.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        
        return -1
    
    def plot_segment_band_feature (self,LFP_channel,start_time,end_time,SPAD_cutoff,lfp_cutoff):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
       
        SPAD_smooth= OE.smooth_signal(silced_recording['zscore_raw'],Fs=self.fs,cutoff=SPAD_cutoff)
        SPAD_smooth= OE.butter_filter(SPAD_smooth, btype='high', cutoff=2, fs=self.fs, order=3)
        
        lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=2, fs=self.fs, order=3)
        spad_low = pd.Series(SPAD_smooth, index=silced_recording['zscore_raw'].index)
        lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)
        
        fig, ax = plt.subplots(8, 1, figsize=(24, 20))
        OE.plot_trace_in_seconds_ax (ax[0],spad_low,self.fs,label='GEVI',color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)
        #spad_filtered=OE.band_pass_filter(spad_data,120,300,self.fs)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(spad_low,lowpassCutoff=100,Fs=self.fs,scale=20)
        OE.plot_wavelet(ax[1],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=False)

        OE.plot_trace_in_seconds_ax (ax[2],lfp_low,self.fs,label='LFP',color=sns.color_palette("dark", 8)[7],ylabel='mV',xlabel=False)
        #lfp_data_filtered=OE.band_pass_filter(lfp_data,120,300,self.fs)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_low,lowpassCutoff=500,Fs=self.fs,scale=20)
        OE.plot_wavelet(ax[3],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=False)
        ax[1].set_ylim(0,20)
        ax[3].set_ylim(0,20)
        ax[3].set_xlabel('Time (seconds)')
        ax[0].legend().set_visible(False)
        ax[2].legend().set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['bottom'].set_visible(False)
        ax[3].spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[1].set_xticks([])  # Hide x-axis tick marks
        ax[1].set_xlabel([])
        ax[1].set_xlabel('')  # Hide x-axis label
        ax[2].set_xticks([])  # Hide x-axis tick marks
        ax[2].set_xlabel([])
        ax[2].set_xlabel('')  # Hide x-axis label
        ax[3].set_xticks([])  # Hide x-axis tick marks
        ax[3].set_xlabel([])
        ax[3].set_xlabel('')  # Hide x-axis label
        
        SPAD_theta= OE.butter_filter(silced_recording['zscore_raw'], btype='high', cutoff=4, fs=self.fs, order=5)
        SPAD_theta= OE.butter_filter(SPAD_theta, btype='low', cutoff=10, fs=self.fs, order=5)
        lfp_theta = OE.butter_filter(silced_recording[LFP_channel], btype='high', cutoff=4, fs=self.fs, order=5)
        lfp_theta = OE.butter_filter(lfp_theta, btype='low', cutoff=10, fs=self.fs, order=5)
        SPAD_theta = pd.Series(SPAD_theta, index=silced_recording['zscore_raw'].index)
        lfp_theta = pd.Series(lfp_theta, index=silced_recording[LFP_channel].index)
        OE.plot_trace_in_seconds_ax (ax[4],SPAD_theta,self.fs,label='GEVI',color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)

        OE.plot_trace_in_seconds_ax (ax[5],lfp_theta,self.fs,label='LFP',color=sns.color_palette("dark", 8)[7],
                               ylabel='μV',xlabel=True)
        
        ax[4].legend().set_visible(False)
        ax[5].legend().set_visible(False)
        #Beta: 15-30Hz
        SPAD_beta= OE.butter_filter(silced_recording['zscore_raw'], btype='high', cutoff=20, fs=self.fs, order=5)
        SPAD_beta= OE.butter_filter(SPAD_beta, btype='low', cutoff=40, fs=self.fs, order=5)
        lfp_beta = OE.butter_filter(silced_recording[LFP_channel], btype='high', cutoff=20, fs=self.fs, order=5)
        lfp_beta = OE.butter_filter(lfp_beta, btype='low', cutoff=40, fs=self.fs, order=5)
        SPAD_beta = pd.Series(SPAD_beta, index=silced_recording['zscore_raw'].index)
        lfp_beta = pd.Series(lfp_beta, index=silced_recording[LFP_channel].index)
        OE.plot_trace_in_seconds_ax (ax[6],SPAD_beta,self.fs,label='GEVI',color=sns.color_palette("husl", 8)[3],
                               ylabel='z-score',xlabel=False)

        OE.plot_trace_in_seconds_ax (ax[7],lfp_beta,self.fs,label='LFP',color=sns.color_palette("dark", 8)[7],
                               ylabel='mV',xlabel=True)
        
        ax[6].legend().set_visible(False)
        ax[7].legend().set_visible(False)
        #plt.tight_layout()
        makefigure_path = os.path.join(self.savepath, 'makefigure')
        if not os.path.exists(makefigure_path):
            os.makedirs(makefigure_path)
            
        output_path=os.path.join(makefigure_path,'example_theta_trace.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        return -1
    
    def plot_segment_band_feature_twoROIs (self,LFP_channel,start_time,end_time,SPAD_cutoff,lfp_cutoff):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        
        '''
        You can get zdFF directly by calling the function fp.get_zdFF()
        TO CHECK THE SIGNAL STEP BY STEP:
        YOU CAN USE THE FOLLOWING CODES TO GET MORE PLOTS
        These will give you plots for 
        smoothed signal, corrected signal, normalised signal and the final zsocre
        '''
        '''Step 1, plot smoothed traces'''
        smooth_win = 2
        smooth_reference,smooth_signal,r_base,s_base = fp.photometry_smooth_plot (
            silced_recording['ref_raw'],silced_recording['sig_raw'],sampling_rate=self.fs, smooth_win = smooth_win)
        
        reference = (smooth_reference- r_base)
        signal = (smooth_signal - s_base) 
        '''Step 3, plot df/f traces'''
        dff_ref = reference / r_base
        dff_sig = signal / s_base
        
        # z_reference = pd.Series(z_reference)
        # z_signal = pd.Series(z_signal)
        # zdff=z_signal-z_reference
        
        'change here to change what to plot'
        sig_smooth= OE.smooth_signal(dff_sig,Fs=self.fs,cutoff=SPAD_cutoff)
        sig_smooth= OE.butter_filter(sig_smooth, btype='high', cutoff=2, fs=self.fs, order=3)

        ref_smooth= OE.smooth_signal(dff_ref,Fs=self.fs,cutoff=SPAD_cutoff)
        #ref_smooth= OE.butter_filter(ref_smooth, btype='high', cutoff=2, fs=self.fs, order=3)
        
        print (np.max(ref_smooth))
        print (np.max(sig_smooth))
        
        lfp_lowpass = OE.butter_filter(silced_recording[LFP_channel], btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=2, fs=self.fs, order=3)
        
        sig_low = pd.Series(sig_smooth, index=silced_recording['sig_raw'].index)
        ref_low = pd.Series(ref_smooth, index=silced_recording['ref_raw'].index)
        lfp_low = pd.Series(lfp_lowpass, index=silced_recording[LFP_channel].index)
        
        sig_theta= OE.butter_filter(sig_low, btype='high', cutoff=4, fs=self.fs, order=5)
        sig_theta= OE.butter_filter(sig_theta, btype='low', cutoff=9, fs=self.fs, order=5)
        ref_theta= OE.butter_filter(ref_low, btype='high', cutoff=4, fs=self.fs, order=5)
        ref_theta= OE.butter_filter(ref_theta, btype='low', cutoff=9, fs=self.fs, order=5)
        lfp_theta = OE.butter_filter(lfp_low, btype='high', cutoff=4, fs=self.fs, order=5)
        lfp_theta = OE.butter_filter(lfp_theta, btype='low', cutoff=9, fs=self.fs, order=5)
        
        sig_theta = pd.Series(sig_theta, index=silced_recording['sig_raw'].index)
        ref_theta = pd.Series(ref_theta, index=silced_recording['ref_raw'].index)
        lfp_theta = pd.Series(lfp_theta, index=silced_recording[LFP_channel].index)
        
        fig, ax = plt.subplots(8, 1, figsize=(24, 20))
        OE.plot_trace_in_seconds_ax (ax[0],sig_low,self.fs,label='GEVI',color=sns.color_palette("husl", 8)[3],
                               ylabel='GEVI',xlabel=False)
        #spad_filtered=OE.band_pass_filter(spad_data,120,300,self.fs)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(sig_low,lowpassCutoff=100,Fs=self.fs,scale=20)
        OE.plot_wavelet(ax[1],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=False)
        
        OE.plot_trace_in_seconds_ax (ax[2],ref_low,self.fs,label='Ref',color=sns.color_palette("husl", 8)[0],
                               ylabel='Ref',xlabel=False)
        #spad_filtered=OE.band_pass_filter(spad_data,120,300,self.fs)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(ref_low,lowpassCutoff=100,Fs=self.fs,scale=20)
        OE.plot_wavelet(ax[3],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=False)

        OE.plot_trace_in_seconds_ax (ax[4],lfp_low,self.fs,label='LFP',color=sns.color_palette("dark", 8)[7],ylabel='mV',xlabel=False)
        #lfp_data_filtered=OE.band_pass_filter(lfp_data,120,300,self.fs)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_low,lowpassCutoff=500,Fs=self.fs,scale=20)
        OE.plot_wavelet(ax[5],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=False)
        
        ax[1].set_ylim(0,15)
        ax[3].set_ylim(0,15)
        ax[5].set_ylim(0,15)
        # ax[0].set_ylim(-37,37)
        # ax[2].set_ylim(-37,37)

        ax[5].set_xlabel('Time (seconds)')
        ax[0].legend().set_visible(False)
        ax[2].legend().set_visible(False)
        ax[4].legend().set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['bottom'].set_visible(False)
        ax[3].spines['left'].set_visible(False)
        ax[5].spines['top'].set_visible(False)
        ax[5].spines['right'].set_visible(False)
        ax[5].spines['bottom'].set_visible(False)
        ax[5].spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[1].set_xticks([])  # Hide x-axis tick marks
        ax[1].set_xlabel([])
        ax[1].set_xlabel('')  # Hide x-axis label
        ax[2].set_xticks([])  # Hide x-axis tick marks
        ax[2].set_xlabel([])
        ax[2].set_xlabel('')  # Hide x-axis label
        ax[3].set_xticks([])  # Hide x-axis tick marks
        ax[3].set_xlabel([])
        ax[3].set_xlabel('')  # Hide x-axis label
        
        
        OE.plot_trace_in_seconds_ax (ax[6],sig_theta,self.fs,label='GEVI',color=sns.color_palette("husl", 8)[3],
                               ylabel='GEVI',xlabel=False)
        # OE.plot_trace_in_seconds_ax (ax[6],ref_theta,self.fs,label='Ref',color=sns.color_palette("husl", 8)[0],
        #                        ylabel='Ref',xlabel=False)

        OE.plot_trace_in_seconds_ax (ax[7],lfp_theta,self.fs,label='LFP',color=sns.color_palette("dark", 8)[7],
                               ylabel='μV',xlabel=True)
        
        ax[6].legend().set_visible(False)
        ax[7].legend().set_visible(False)
       
        #plt.tight_layout()
        makefigure_path = os.path.join(self.savepath, 'makefigure')
        if not os.path.exists(makefigure_path):
            os.makedirs(makefigure_path)
            
        output_path=os.path.join(makefigure_path,'example_theta_trace.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        return sig_smooth,ref_smooth,silced_recording['sig_raw']
    
    def plot_band_power_feature (self,LFP_channel,start_time,end_time,LFP=True):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        if LFP:
            lfp_data=silced_recording[LFP_channel]/1000
        else:
            lfp_data=silced_recording[LFP_channel]
        data=lfp_data.to_numpy()
        lfp_thetaband=OE.band_pass_filter(data,5,9,Fs=self.fs)
        #lfp_thetaband = lfp_thetaband - np.mean(lfp_thetaband)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(data,lowpassCutoff=500,Fs=self.fs)
        time = np.arange(len(sst)) *(1/self.fs)
        OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,lfp_thetaband,powerband='(5-9Hz)')
        
        lfp_rippleband=OE.band_pass_filter(data,150,250,Fs=self.fs)
        #lfp_thetaband = lfp_thetaband - np.mean(lfp_thetaband)
        sst,frequency,power,global_ws=OE.Calculate_wavelet(lfp_rippleband,lowpassCutoff=500,Fs=self.fs,scale=20)
        time = np.arange(len(sst)) *(1/self.fs)
        OE.plot_wavelet_feature(data,frequency,power,global_ws,time,lfp_rippleband,powerband='(150-250Hz)')     
        return -1
    
    def plot_segment_feature (self,LFP_channel,start_time,end_time,SPAD_cutoff,lfp_cutoff):
        silced_recording=self.slicing_pd_data (self.Ephys_tracking_spad_aligned,start_time=start_time, end_time=end_time)
        self.plot_lowpass_two_trace (silced_recording,LFP_channel, SPAD_cutoff,lfp_cutoff)
        return -1
    
    def plot_lowpass_two_trace (self,data, LFP_channel,SPAD_cutoff,lfp_cutoff):
        
        lambd = 5e4 # Adjust lambda to get the best fit
        porder = 1
        itermax = 50
        sig_base = fp.airPLS(data['sig_raw'], lambda_=lambd, porder=porder, itermax=itermax)
        sig = data['sig_raw'] - sig_base
        eps = 1e-12
        dff_sig = 100.0 * sig / (sig_base + eps)
        SPAD_smooth= OE.smooth_signal(dff_sig,Fs=self.fs,cutoff=SPAD_cutoff)
        
        #SPAD_smooth= OE.smooth_signal(data['zscore_raw'],Fs=self.fs,cutoff=SPAD_cutoff)
        SPAD_smooth= OE.butter_filter(SPAD_smooth, btype='high', cutoff=2.5, fs=self.fs, order=3)
        lfp_lowpass = OE.butter_filter(data[LFP_channel],btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=4, fs=self.fs, order=3)

        spad_low = pd.Series(SPAD_smooth, index=data['zscore_raw'].index)
        lfp_low = pd.Series(lfp_lowpass, index=data[LFP_channel].index)
        self.plot_two_traces (spad_low,lfp_low,Spectro_ylim=20,AddColorbar=True)
        return -1

    def plot_two_traces(self, spad_data, lfp_data,
                    spad_label='photometry', lfp_label='LFP',
                    Spectro_ylim=20, AddColorbar=False,
                    # --- NEW: font controls ---
                    label_fs=18, tick_fs=16,
                    five_xticks=False):
        """Plot SPAD + LFP with wavelets; bigger labels/ticks; optional 5 x-ticks."""
        fig, ax = plt.subplots(4, 1, figsize=(12, 10))
    
        # SPAD trace + wavelet
        OE.plot_trace_in_seconds_ax(ax[0], spad_data, self.fs, label=spad_label,
                                    color=sns.color_palette("husl", 8)[3],
                                    ylabel='z-score', xlabel=False)
        sst, frequency, power, global_ws = OE.Calculate_wavelet(spad_data, lowpassCutoff=100, Fs=self.fs, scale=40)
        OE.plot_wavelet(ax[1], sst, frequency, power, Fs=self.fs, colorBar=False, logbase=False)
    
        # LFP (mV) + wavelet
        lfp_data = lfp_data / 1000
        OE.plot_trace_in_seconds_ax(ax[2], lfp_data, self.fs, label=lfp_label,
                                    color=sns.color_palette("husl", 8)[5],
                                    ylabel='mV', xlabel=False)
        sst, frequency, power, global_ws = OE.Calculate_wavelet(lfp_data, lowpassCutoff=500, Fs=self.fs, scale=40)
        OE.plot_wavelet(ax[3], sst, frequency, power, Fs=self.fs, colorBar=AddColorbar, logbase=False)
    
        # Spectrogram y-lims
        for a in (ax[1], ax[3]):
            a.set_ylim(0, Spectro_ylim)
    
        # Axis labels (bigger)
        ax[3].set_xlabel('Time (seconds)', fontsize=label_fs)
        for i in (0, 2):
            if ax[i].get_ylabel():
                ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=label_fs)
    
        # Ticks (bigger)
        for a in ax:
            a.tick_params(axis='both', labelsize=tick_fs, width=1.2)
    
        # Optional: exactly five x-ticks on bottom axis
        if five_xticks:
            lo, hi = ax[3].get_xlim()
            ticks = np.linspace(lo, hi, 5)
            ticks[np.isclose(ticks, 0.0)] = 0.0  # avoid -0.00
            ax[3].set_xticks(ticks)
            ax[3].set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=tick_fs)
    
        # Legends off (unchanged)
        ax[0].legend().set_visible(False)
        ax[2].legend().set_visible(False)
    
        # Cosmetics (unchanged)
        for a in (ax[1], ax[3]):
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)
        ax[2].spines['bottom'].set_visible(False)
        ax[1].set_xticks([]); ax[1].set_xlabel('')
        ax[2].set_xticks([]); ax[2].set_xlabel('')
    
        # Save
        makefigure_path = os.path.join(self.savepath, 'makefigure')
        os.makedirs(makefigure_path, exist_ok=True)
        output_path = os.path.join(makefigure_path, 'example_trace_powerspectral.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
        plt.show()
        return -1
    
    def plot_segment_feature_multiROI(self, LFP_channel, start_time, end_time,
                                      SPAD_cutoff, lfp_cutoff,
                                      # --- NEW: font controls ---
                                      label_fs=18, tick_fs=16, text_fs=16,
                                      five_xticks=False):
        """Bigger labels/ticks; optional 5 x-ticks on the bottom axis."""
        # slice recording
        data = self.slicing_pd_data(self.Ephys_tracking_spad_aligned,
                                    start_time=start_time, end_time=end_time)
    
        # --- Photometry smoothing (signal / control / z) -------------------------
        sig_smooth = OE.smooth_signal(data['sig_raw'], Fs=self.fs, cutoff=SPAD_cutoff)
        sig_smooth = OE.butter_filter(sig_smooth, btype='high', cutoff=2.5, fs=self.fs, order=3)
    
        ref_smooth = OE.smooth_signal(data['ref_raw'], Fs=self.fs, cutoff=SPAD_cutoff)
        ref_smooth = OE.butter_filter(ref_smooth, btype='high', cutoff=2.5, fs=self.fs, order=3)
    
        # FIXED: create z_smooth from z-score channel (not ref_raw), then high-pass
        z_smooth = OE.smooth_signal(data['zscore_raw'], Fs=self.fs, cutoff=SPAD_cutoff)
        z_smooth = OE.butter_filter(z_smooth, btype='high', cutoff=2.5, fs=self.fs, order=3)
    
        # --- LFP smoothing -------------------------------------------------------
        lfp_lowpass = OE.butter_filter(data[LFP_channel], btype='low', cutoff=lfp_cutoff, fs=self.fs, order=5)
        lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=2.5, fs=self.fs, order=3)
    
        # --- Wrap into Series to keep time index --------------------------------
        signal_data = pd.Series(sig_smooth, index=data['sig_raw'].index)
        ref_data    = pd.Series(ref_smooth, index=data['ref_raw'].index)
        z_data      = pd.Series(z_smooth,   index=data['zscore_raw'].index)
        lfp_data    = pd.Series(lfp_lowpass, index=data[LFP_channel].index)
    
        # --- Figure: now 8 rows (sig, sig-WT, ref, ref-WT, z, z-WT, LFP, LFP-WT) -
        fig, ax = plt.subplots(8, 1, figsize=(12, 17))
    
        # signal trace + wavelet
        OE.plot_trace_in_seconds_ax(ax[0], signal_data, self.fs, label='signal',
                                    color=sns.color_palette("husl", 8)[3],
                                    ylabel='z-score', xlabel=False)
        sst, frequency, power, global_ws = OE.Calculate_wavelet(signal_data, lowpassCutoff=100, Fs=self.fs, scale=40)
        OE.plot_wavelet(ax[1], sst, frequency, power, Fs=self.fs, colorBar=False, logbase=False)
    
        # control trace + wavelet
        OE.plot_trace_in_seconds_ax(ax[2], ref_data, self.fs, label='control',
                                    color=sns.color_palette("husl", 8)[0],
                                    ylabel='z-score', xlabel=False)
        sst, frequency, power, global_ws = OE.Calculate_wavelet(ref_data, lowpassCutoff=100, Fs=self.fs, scale=40)
        OE.plot_wavelet(ax[3], sst, frequency, power, Fs=self.fs, colorBar=False, logbase=False)
    
        # NEW: z_smooth trace + wavelet
        OE.plot_trace_in_seconds_ax(ax[4], z_data, self.fs, label='z_smooth',
                                    color=sns.color_palette("husl", 8)[2],
                                    ylabel='z-score', xlabel=False)
        sst, frequency, power, global_ws = OE.Calculate_wavelet(z_data, lowpassCutoff=100, Fs=self.fs, scale=40)
        OE.plot_wavelet(ax[5], sst, frequency, power, Fs=self.fs, colorBar=False, logbase=False)
    
        # LFP (to mV) + wavelet
        lfp_mv = lfp_data / 1000.0
        OE.plot_trace_in_seconds_ax(ax[6], lfp_mv, self.fs, label='LFP',
                                    color=sns.color_palette("husl", 8)[5],
                                    ylabel='mV', xlabel=False)
        sst, frequency, power, global_ws = OE.Calculate_wavelet(lfp_mv, lowpassCutoff=500, Fs=self.fs, scale=40)
        OE.plot_wavelet(ax[7], sst, frequency, power, Fs=self.fs, colorBar=False, logbase=False)
    
        # wavelet y-lims
        for wl_ax in (ax[1], ax[3], ax[5], ax[7]):
            wl_ax.set_ylim(0, 20)
    
        # axis cosmetics
        ax[7].set_xlabel('Time (seconds)', fontsize=label_fs)
    
        # legends off (keep, but scale if you ever show them)
        for i in (0, 2, 4, 6):
            leg = ax[i].legend()
            leg.set_visible(False)
    
        # remove spines on wavelets
        for wl_ax in (ax[1], ax[3], ax[5], ax[7]):
            wl_ax.spines['top'].set_visible(False)
            wl_ax.spines['right'].set_visible(False)
            wl_ax.spines['bottom'].set_visible(False)
            wl_ax.spines['left'].set_visible(False)
    
        # hide x-ticks/labels where requested (all but bottom-most wavelet)
        for a in (ax[1], ax[2], ax[3], ax[4], ax[5], ax[6]):
            a.set_xticks([])
            a.set_xlabel('')
    
        # --- scale fonts everywhere ----------------------------------------------
        for a in ax:
            if a.get_ylabel():
                a.set_ylabel(a.get_ylabel(), fontsize=label_fs)
            a.tick_params(axis='both', labelsize=tick_fs, width=1.2)
    
        # --- OPTIONAL: exactly five x-ticks on bottom axis -----------------------
        if five_xticks:
            lo, hi = ax[7].get_xlim()
            ticks = np.linspace(lo, hi, 5)
            ticks[np.isclose(ticks, 0.0)] = 0.0  # avoid -0.00
            ax[7].set_xticks(ticks)
            ax[7].set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=tick_fs)
    
        # save
        makefigure_path = os.path.join(self.savepath, 'makefigure')
        os.makedirs(makefigure_path, exist_ok=True)
        output_path = os.path.join(makefigure_path, 'example_trace_powerspectral.png')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
        plt.show()
        return -1

    def Label_REM_sleep (self,LFP_channel):
        lfp_data=self.Ephys_tracking_spad_aligned[LFP_channel]/1000
        timestamps=self.Ephys_tracking_spad_aligned['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        ThetaDeltaRatio=OE.getThetaDeltaRatio (LFP,self.fs,windowlen=1000)
        self.Ephys_tracking_spad_aligned['REMstate'] = 'nonREM'  # Initialize with 'nonREM'
        self.Ephys_tracking_spad_aligned.loc[ThetaDeltaRatio > 1.2, 'REMstate'] = 'REM'
        return -1
    
    def pynacollada_label_theta (self,LFP_channel,Low_thres=0.2,High_thres=10,save=False,plot_theta=False):
        '''NOTE: 
        I applied the pynacollada ripple-detection method, but with the theta band to extract high-theta period.
        Other period are defined as non-theta.
        '''
        lfp_data=self.Ephys_tracking_spad_aligned[LFP_channel]/1000
        timestamps=self.Ephys_tracking_spad_aligned['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        'To detect theta'
        theta_band_filtered,nSS,nSS3,theta_ep,theta_tsd = OE.getThetaEvents (LFP,self.fs,windowlen=500,Low_thres=Low_thres,High_thres=High_thres)  
        print ('Label theta part, found theta high epoch number---',len(theta_ep))
        '''METHOD: label theta_ep in the Dataframe'''
        timestamps_round=np.round(timestamps, 2)

        self.Ephys_tracking_spad_aligned['BrainState']='nontheta'
        indices_theta_epoch = []
        for i in range (len(theta_ep)):
            # print ('ep start---', theta_ep.iloc[[i]]['start'][0])
            # print ('ep end---', theta_ep.iloc[[i]]['end'][0])
            indices_theta_epoch_i = np.where((timestamps_round>=theta_ep.iloc[[i]]['start'][0]) & (timestamps_round<theta_ep.iloc[[i]]['end'][0]))[0]
            indices_theta_epoch.extend(indices_theta_epoch_i.astype(int))
            
        self.Ephys_tracking_spad_aligned.iloc[indices_theta_epoch, self.Ephys_tracking_spad_aligned.columns.get_loc('BrainState')]='theta' 
        #self.save_data(self.Ephys_tracking_spad_aligned, 'Ephys_tracking_photometry_aligned.pkl')

        print ('---Theta labelling saved, plotting theta and nontheta features---') 
        '''This will separate theta and non-theta period, but only for visualisation.
        We should not use the separated and concatenated theta/nontheta periods for other analysis,
        because it will cut and concatenate the LFP and optical signals arbitrarily'''
        self.theta_part=self.Ephys_tracking_spad_aligned.iloc[indices_theta_epoch]
        self.non_theta_part=self.Ephys_tracking_spad_aligned[self.Ephys_tracking_spad_aligned['BrainState'] == 'nontheta']
        self.theta_part=self.theta_part.reset_index(drop=True)
        self.non_theta_part=self.non_theta_part.reset_index(drop=True)
        'Save the theta part with real indices'
        if save:
            theta_path=os.path.join(self.dpath, "theta_part_with_index.pkl")
            self.theta_part.to_pickle(theta_path) 
            non_theta_path=os.path.join(self.dpath, "non_theta_part_with_index.pkl")
            self.non_theta_part.to_pickle(non_theta_path)
        #From here,I reset the index, concatenate the theta part and non-theta part just for plotting and show the features
        if plot_theta:
            time_interval = 1.0 / self.fs
            total_duration = len(self.non_theta_part) * time_interval
            timestamps = np.arange(0, total_duration, time_interval)
            if len(self.non_theta_part)>len(timestamps):
                self.non_theta_part=self.non_theta_part[:len(timestamps)]
            elif len(self.non_theta_part)<len(timestamps):
                timestamps=timestamps[:len(self.non_theta_part)]   
            # Convert the 'time_column' to timedelta if it's not already
            self.non_theta_part['time_column'] = pd.to_timedelta(timestamps,unit='s') 
            # Set the index to the 'time_column'
            self.non_theta_part.set_index('time_column', inplace=True)
            
            total_duration = len(self.theta_part) * time_interval
            timestamps = np.arange(0, total_duration, time_interval)
            if len(self.theta_part)>len(timestamps):
                self.theta_part=self.theta_part[:len(timestamps)]
            elif len(self.theta_part)<len(timestamps):
                timestamps=timestamps[:len(self.theta_part)]   
            self.theta_part['time_column'] = pd.to_timedelta(timestamps, unit='s') 
            self.theta_part.set_index('time_column', inplace=True)      
            
            #plot theta
            thetaPropotion=len(self.theta_part)/len(self.non_theta_part)
            print ('thetaPropotion is:',thetaPropotion)
            sst,frequency,power,global_ws=OE.Calculate_wavelet(self.theta_part[LFP_channel]/1000,lowpassCutoff=500,Fs=self.fs)
            time = np.arange(len(sst)) *(1/self.fs)
            OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,self.theta_part[LFP_channel])
            #self.plot_lowpass_two_trace (self.theta_part, LFP_channel,SPAD_cutoff=20,lfp_cutoff=20)
            #plot non-theta
            sst,frequency,power,global_ws=OE.Calculate_wavelet(self.non_theta_part[LFP_channel]/1000,lowpassCutoff=500,Fs=self.fs)
            time = np.arange(len(sst)) *(1/self.fs)
            OE.plot_wavelet_feature(sst,frequency,power,global_ws,time,self.non_theta_part[LFP_channel])
            #self.plot_lowpass_two_trace (self.non_theta_part, LFP_channel,SPAD_cutoff=20,lfp_cutoff=20)
        return -1
    
    def plot_theta_correlation(self,theta_part,LFP_channel,save_path=None,optical_channel='zscore_raw'):
        silced_recording=theta_part
        #silced_recording=self.Ephys_tracking_spad_aligned
        silced_recording=silced_recording.reset_index(drop=True)
        #print (silced_recording.index)
        silced_recording['theta_angle']=OE.calculate_theta_phase_angle(silced_recording[LFP_channel], theta_low=4, theta_high=12) #range 5 to 9
        OE.plot_trace_in_seconds(silced_recording['theta_angle'],Fs=10000,title='theta angle')
        trough_index,peak_index = OE.calculate_theta_trough_index(silced_recording,Fs=10000)
        #print (trough_index)
        fig=OE.plot_theta_cycle (silced_recording, LFP_channel,trough_index,half_window=0.15,fs=10000,optical_channel=optical_channel)
        #fig1,fig2=OE.plot_zscore_to_theta_phase (silced_recording['theta_angle'],silced_recording['zscore_raw'])
        #OE.calculate_perferred_phase(silced_recording['theta_angle'],silced_recording['zscore_raw'])
        
        fig4,MI, bin_centers, prop=OE.compute_optical_event_on_phase(silced_recording['theta_angle'], 
                                    silced_recording[optical_channel], bins=30, distance=10,plot=True)
        OE.compute_optical_phase_preference(silced_recording['theta_angle'],
                                     silced_recording[optical_channel],bins=30,
                                     height_factor=3.0,
                                     distance=20,
                                     prominence=None,
                                     min_events=50,
                                     alpha=0.01,
                                     use_event_indices=None,
                                     plot=True)

        # OE.run_phase_modulation_analysis(silced_recording['theta_angle'],
        #                                  silced_recording[optical_channel], bins=30, n_perm=2000)
        
        if save_path:
            fig_path = os.path.join(save_path,'LFP_GEVI_average.png')
            fig.savefig(fig_path, transparent=True)
            # fig_path = os.path.join(save_path,'zscore_theta_phase.png')
            # fig2.savefig(fig_path, transparent=True)
            fig_path = os.path.join(save_path,'Phase modulation depth.png')
            fig4.savefig(fig_path, transparent=True)
            
            
            pkl_path = os.path.join(save_path,'phase_MI.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'MI': MI,
                    'bin_centers': bin_centers,
                    'prop': prop
                }, f)
        
        return trough_index,peak_index
    
    def plot_gamma_correlation(self,LFP_channel,save_path):
        silced_recording=self.theta_part
        #silced_recording=self.Ephys_tracking_spad_aligned
        silced_recording=silced_recording.reset_index(drop=True)
        #print (silced_recording.index)
        silced_recording['gamma_angle']=OE.calculate_theta_phase_angle(silced_recording[LFP_channel], theta_low=30, theta_high=60) #range 5 to 9
        OE.plot_trace_in_seconds(silced_recording['gamma_angle'],Fs=10000,title='gamma angle')
        trough_index,peak_index = OE.calculate_gamma_trough_index(silced_recording,Fs=10000)
        #print (trough_index)
        fig=OE.plot_theta_cycle (silced_recording, LFP_channel,peak_index,half_window=0.15,fs=10000,plotmode='two')
        fig1,fig2=OE.plot_zscore_to_theta_phase (silced_recording['gamma_angle'],silced_recording['zscore_raw'])
        fig4,MI, bin_centers, norm_amp=OE.compute_optical_event_on_phase(silced_recording['gamma_angle'], silced_recording['zscore_raw'], bins=12, plot=True)
        OE.compute_optical_phase_preference(silced_recording['gamma_angle'],
                                     silced_recording['zscore_raw'],bins=30,
                                     height_factor=3.0,
                                     distance=20,
                                     prominence=None,
                                     min_events=50,
                                     alpha=0.01,
                                     use_event_indices=None,
                                     plot=True)
        
        OE.run_phase_modulation_analysis(silced_recording['gamma_angle'], silced_recording['zscore_raw'], bins=12, n_perm=3000)
        
        fig_path = os.path.join(save_path,'LFP_GEVI_gamma_average.png')
        fig.savefig(fig_path, transparent=True)
        
        fig_path = os.path.join(save_path,'zscore_gamma_phase.png')
        fig2.savefig(fig_path, transparent=True)
        
        fig_path = os.path.join(save_path,'Gamma Phase modulation depth.png')
        fig4.savefig(fig_path, transparent=True)
        
        if save_path:
            pkl_path = os.path.join(save_path,'gamma_phase_MI.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'MI': MI,
                    'bin_centers': bin_centers,
                    'norm_amp': norm_amp
                }, f)
        return trough_index,peak_index
 
    
    def pynappleAnalysis (self,lfp_channel='LFP_2',ep_start=0,ep_end=10,
                          Low_thres=1,High_thres=10,plot_segment=False,plot_ripple_ep=True,excludeTheta=True,excludeREM=False):
        'This is the LFP data that need to be saved for the sync ananlysis'
        #data_segment=self.Ephys_tracking_spad_aligned
        data_segment=self.non_theta_part
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        #timestamps=timestamps-timestamps[0]
        #Use non-theta part to detect ripple
        lfp_data=data_segment[lfp_channel]
        spad_data=data_segment['zscore_raw']
        lfp_data=lfp_data/1000 #change the unit from uV to mV
        SPAD_cutoff=200
        SPAD_smooth_np = OE.smooth_signal(spad_data,Fs=self.fs,cutoff=SPAD_cutoff)
        LFP_smooth_np = OE.smooth_signal(lfp_data,Fs=self.fs,cutoff=1000)
        'To align LFP and SPAD raw data to pynapple format'
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        SPAD=nap.Tsd(t = timestamps, d = spad_data.to_numpy(), time_units = 's')
        SPAD_smooth=nap.Tsd(t = timestamps, d = SPAD_smooth_np, time_units = 's')  
        LFP_smooth=nap.Tsd(t = timestamps, d = LFP_smooth_np, time_units = 's')  
        'Calculate theta band for optical signal'
        #SPAD_ripple_band_filtered,nSS_spad,nSS3_spad,rip_ep_spad,rip_tsd_spad = OE.getRippleEvents (SPAD_smooth,self.fs,windowlen=500,Low_thres=Low_thres,High_thres=High_thres)
        'To detect ripple'
        ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getRippleEvents (LFP,self.fs,windowlen=500,Low_thres=Low_thres,High_thres=High_thres)
        SPAD_ripple_band_filtered = pyna.eeg_processing.bandpass_filter(SPAD, 130, 250, self.fs)
        # SPAD_ripple_band_filtered = OE.band_pass_filter(SPAD,120,300,self.fs)
        # SPAD_ripple_band_filtered=nap.Tsd(t = timestamps, d = SPAD_ripple_band_filtered, time_units = 's')
        
        if excludeTheta:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'theta' in close_timestamps_df['BrainState'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near theta, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
            
        if excludeREM:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'REM' in close_timestamps_df['REMstate'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near REM state, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
        
        # Assign a value to the dynamically generated key
        self.ripple_numbers = len(rip_ep)
        'Calculate ripple frequency during non-theta periods'
        nontheta_length=len(data_segment[data_segment['BrainState'] == 'nontheta'])
        self.ripple_freq=np.round(self.ripple_numbers/(nontheta_length/self.fs),4)
        
        print('LFP length in seconds:',len(LFP)/self.fs)
        print('nontheta_LFP length in seconds:',nontheta_length/self.fs)
        print('Optical signal length in seconds:',len(SPAD)/self.fs)
        print('Found ripple event numbers:',self.ripple_numbers)
        print('Ripple event frequency during non-theta:',self.ripple_freq, 'events/seconds')
        
        'To plot the choosen segment with start time and end time'
        if plot_segment:
            ex_ep = nap.IntervalSet(start = ep_start+timestamps[0], end = ep_end+timestamps[0], time_units = 's') 
            fig, ax = plt.subplots(6, 1, figsize=(10, 12))
            OE.plot_trace_nap (ax[0], LFP,ex_ep,color=sns.color_palette("husl", 8)[5],title='LFP raw Trace')
            OE.plot_trace_nap (ax[1], ripple_band_filtered,ex_ep,color=sns.color_palette("husl", 8)[5],title='Ripple band')
            OE.plot_ripple_event (ax[2], rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres=Low_thres) 
            OE.plot_trace_nap (ax[3], SPAD_smooth,ex_ep,color='green',title='calcium recording (z-score)')
            #LFP_rippleband=OE.band_pass_filter(LFP.restrict(ex_ep),150,250,Fs=self.fs)     
            sst,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered.restrict(ex_ep),lowpassCutoff=500,Fs=self.fs,scale=40)
            OE.plot_wavelet(ax[4],sst,frequency,power,Fs=self.fs,colorBar=False)
            #OE.plot_ripple_spectrum (ax[4], LFP, ex_ep,y_lim=30,Fs=self.fs,vmax_percentile=100)
            plt.subplots_adjust(hspace=0.5)
            
        self.ripple_std_values=[]
        self.ripple_duration_values=[]
        self.ripple_optic_power_values=[]
        self.ripple_LFP_power_values=[]

        event_peak_times=rip_tsd.index.to_numpy()
        for i in range(len(rip_ep)):
            ripple_std=rip_tsd.iloc[i]
            ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  #second to ms
            self.ripple_std_values.append(ripple_std)
            self.ripple_duration_values.append(ripple_duration)
            if event_peak_times[i]-timestamps[0]>0.1 and timestamps[-1]-event_peak_times[i]>0.1:
                if plot_ripple_ep:
                    start_time=event_peak_times[i]-0.1
                    end_time=event_peak_times[i]+0.1
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    LFP_smooth_ep=LFP_smooth.restrict(rip_long_ep)
                    ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_long_ep)   
                    SPAD_ripple_band_filtered_ep=SPAD_ripple_band_filtered.restrict(rip_long_ep)
                    start_time1=event_peak_times[i]-0.1
                    end_time1=event_peak_times[i]+0.1
                    rip_short_ep = nap.IntervalSet(start = start_time1, end = end_time1, time_units = 's') 
                    LFP_short_ep=LFP.restrict(rip_short_ep)
                    SPAD_short_ep=SPAD.restrict(rip_short_ep)
                    ripple_band_filtered_short_ep=ripple_band_filtered.restrict(rip_short_ep)
                    save_ripple_path = os.path.join(self.savepath, self.recordingName+'_Ripples_'+lfp_channel)
                    if not os.path.exists(save_ripple_path):
                        os.makedirs(save_ripple_path)
                    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
                    #Set the title of ripple feature
                    plot_title = "Optical signal triggerred by ripple" 
                    
                    sst_ep,frequency_spad,power_spad,global_ws=OE.Calculate_wavelet(SPAD_ripple_band_filtered_ep,lowpassCutoff=150,Fs=self.fs,scale=40)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    OE.plot_ripple_overlay (ax[0],LFP_smooth_ep,SPAD_smooth_ep,frequency_spad,power_spad,time,ripple_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=True,plotRipple=False)                                   
                    #Set the title of ripple feature
                    plot_title = "Local Field Potential with Spectrogram" 
                    sst_ep,frequency,power_LFP,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=250,Fs=self.fs,scale=40) 
                    OE.plot_ripple_overlay (ax[1],LFP_smooth_ep,SPAD_smooth_ep,frequency,power_LFP,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=False)
                    
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_short_ep,lowpassCutoff=250,Fs=self.fs,scale=40)                
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    #Set the title of ripple feature
                    plot_title = f"Ripple Peak std:{ripple_std:.2f}, Ripple Duration:{ripple_duration:.2f} ms" 
                    OE.plot_two_trace_overlay(ax[2], time,ripple_band_filtered_ep,SPAD_ripple_band_filtered_ep, title='Wavelet Power Spectrum',color1='black', color2='lime')
                    #OE.plot_ripple_overlay (ax[2],ripple_band_filtered_ep,SPAD_ripple_band_filtered_ep,frequency,power,time,SPAD_ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=True,plotRipple=True)
                    #OE.plot_ripple_overlay (ax[2],LFP_short_ep,SPAD_short_ep,frequency,power,time,ripple_band_filtered_short_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=True)         
                    ax[0].axvline(0, color='white',linewidth=2)
                    ax[1].axvline(0, color='white',linewidth=2)
                    ax[2].axvline(0, color='white',linewidth=2) 
                    plt.tight_layout() 
                    figName=self.recordingName+'_Ripple'+str(i)+'.png'
                    fig.savefig(os.path.join(save_ripple_path,figName),transparent=True)
                    self.ripple_optic_power_values.append(power_spad)
                    self.ripple_LFP_power_values.append(power_LFP)

        if len(self.ripple_std_values) !=0:
            self.ripple_std_mean= sum(self.ripple_std_values) / len(self.ripple_std_values)
            self.ripple_duration_mean =sum(self.ripple_duration_values) / len(self.ripple_duration_values)
          
        self.rip_ep=rip_ep
        self.rip_tsd=rip_tsd
        if len(self.rip_tsd)>2:
            self.Oscillation_triggered_Optical_transient (mode='ripple',lfp_channel=lfp_channel, half_window=0.2, plot_single_trace=True,plotShade='CI')
            self.Oscillation_optical_correlation (mode='ripple',lfp_channel=lfp_channel, half_window=0.2)
        if plot_ripple_ep:
            'plot averaged power spectrum'
            # Ensure all arrays have the same shape
            expected_shape = (29, 2001)
            for i, arr in enumerate(self.ripple_LFP_power_values):
                if arr.shape != expected_shape:
                    #print(f"Array at index {i} has shape {arr.shape}, resizing to {expected_shape}")
                    self.ripple_LFP_power_values[i] = np.resize(arr, expected_shape)
            # Calculate the average of the arrays
            average_LFP_powerSpectrum = np.mean(self.ripple_LFP_power_values, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            OE.plot_power_spectrum (ax,time,frequency, average_LFP_powerSpectrum,colorbar=True)
            # Ensure all arrays have the same shape
            expected_shape = (29, 2001)
            for i, arr in enumerate(self.ripple_optic_power_values):
                if arr.shape != expected_shape:
                    #print(f"Array at index {i} has shape {arr.shape}, resizing to {expected_shape}")
                    self.ripple_optic_power_values[i] = np.resize(arr, expected_shape)
            # Calculate the average of the arrays
            average_optic_powerSpectrum = np.mean(self.ripple_optic_power_values, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            OE.plot_power_spectrum (ax,time,frequency, average_optic_powerSpectrum,colorbar=True)
        return rip_ep,rip_tsd
    
    
    def ManualSelectRipple (self,lfp_channel='LFP_2',ep_start=0,ep_end=10,
                          Low_thres=1,High_thres=10,plot_segment=False,plot_ripple_ep=True,excludeTheta=True,excludeREM=False):
        'This is the LFP data that need to be saved for the sync ananlysis'
        data_segment=self.Ephys_tracking_spad_aligned
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        lfp_data=data_segment[lfp_channel]
        spad_data=data_segment['zscore_raw']
        lfp_data=lfp_data/1000 #change the unit from uV to mV
        SPAD_cutoff=200
        SPAD_smooth_np = OE.smooth_signal(spad_data,Fs=self.fs,cutoff=SPAD_cutoff)
        LFP_smooth_np = OE.smooth_signal(lfp_data,Fs=self.fs,cutoff=1000)
        'To align LFP and SPAD raw data to pynapple format'
        LFP=nap.Tsd(t = timestamps, d = LFP_smooth_np, time_units = 's')
        SPAD=nap.Tsd(t = timestamps, d = SPAD_smooth_np, time_units = 's')
        SPAD_smooth=nap.Tsd(t = timestamps, d = spad_data.to_numpy(), time_units = 's')  
        LFP_smooth=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')  
        'Calculate theta band for optical signal'
        #SPAD_ripple_band_filtered,nSS_spad,nSS3_spad,rip_ep_spad,rip_tsd_spad = OE.getRippleEvents (SPAD_smooth,self.fs,windowlen=500,Low_thres=Low_thres,High_thres=High_thres)
        'To detect ripple'
        ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getRippleEvents (LFP,self.fs,windowlen=500,Low_thres=Low_thres,High_thres=High_thres)
        SPAD_ripple_band_filtered = pyna.eeg_processing.bandpass_filter(SPAD, 130, 250, self.fs)
        
        if excludeTheta:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'theta' in close_timestamps_df['BrainState'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near theta, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
            
        if excludeREM:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'REM' in close_timestamps_df['REMstate'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near REM state, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
        
        # Assign a value to the dynamically generated key
        self.ripple_numbers = len(rip_ep)
        'Calculate ripple frequency during non-theta periods'
        nontheta_length=len(data_segment[data_segment['BrainState'] == 'nontheta'])
        self.ripple_freq=np.round(self.ripple_numbers/(nontheta_length/self.fs),4)
        
        print('LFP length in seconds:',len(LFP)/self.fs)
        print('Optical signal length in seconds:',len(SPAD)/self.fs)
        print('Found ripple event numbers:',self.ripple_numbers)
        print('Ripple event frequency during non-theta:',self.ripple_freq, 'events/seconds')
        
        'To plot the choosen segment with start time and end time'
        if plot_segment:
            ex_ep = nap.IntervalSet(start = ep_start+timestamps[0], end = ep_end+timestamps[0], time_units = 's') 
            fig, ax = plt.subplots(6, 1, figsize=(10, 12))
            OE.plot_trace_nap (ax[0], LFP,ex_ep,color=sns.color_palette("husl", 8)[5],title='LFP raw Trace')
            OE.plot_trace_nap (ax[1], ripple_band_filtered,ex_ep,color=sns.color_palette("husl", 8)[5],title='Ripple band')
            OE.plot_ripple_event (ax[2], rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres=Low_thres) 
            OE.plot_trace_nap (ax[3], SPAD_smooth,ex_ep,color='green',title='calcium recording (z-score)')
            #LFP_rippleband=OE.band_pass_filter(LFP.restrict(ex_ep),150,250,Fs=self.fs)     
            sst,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered.restrict(ex_ep),lowpassCutoff=500,Fs=self.fs,scale=40)
            OE.plot_wavelet(ax[4],sst,frequency,power,Fs=self.fs,colorBar=False)
            #OE.plot_ripple_spectrum (ax[4], LFP, ex_ep,y_lim=30,Fs=self.fs,vmax_percentile=100)
            plt.subplots_adjust(hspace=0.5)
            
        self.ripple_std_values=[]
        self.ripple_duration_values=[]
        self.ripple_optic_power_values=[]
        self.ripple_LFP_power_values=[]
        self.ripple_time_cured=[]

        event_peak_times=rip_tsd.index.to_numpy()
        for i in range(len(rip_ep)):
            ripple_std=rip_tsd.iloc[i]
            ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  #second to ms
            self.ripple_std_values.append(ripple_std)
            self.ripple_duration_values.append(ripple_duration)
            if event_peak_times[i]-timestamps[0]>0.1 and timestamps[-1]-event_peak_times[i]>0.1:
                if plot_ripple_ep:
                    start_time=event_peak_times[i]-0.1
                    end_time=event_peak_times[i]+0.1
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    LFP_smooth_ep=LFP_smooth.restrict(rip_long_ep)
                    ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_long_ep)   
                    SPAD_ripple_band_filtered_ep=SPAD_ripple_band_filtered.restrict(rip_long_ep)
                    start_time1=event_peak_times[i]-0.1
                    end_time1=event_peak_times[i]+0.1
                    rip_short_ep = nap.IntervalSet(start = start_time1, end = end_time1, time_units = 's') 
                    LFP_short_ep=LFP.restrict(rip_short_ep)
                    SPAD_short_ep=SPAD.restrict(rip_short_ep)
                    ripple_band_filtered_short_ep=ripple_band_filtered.restrict(rip_short_ep)
                    save_ripple_path = os.path.join(self.savepath, self.recordingName+'_Ripples_'+lfp_channel)
                    if not os.path.exists(save_ripple_path):
                        os.makedirs(save_ripple_path)
                    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
                    #Set the title of ripple feature
                    plot_title = "Optical signal triggerred by ripple" 
                    
                    sst_ep,frequency_spad,power_spad,global_ws=OE.Calculate_wavelet(SPAD_ripple_band_filtered_ep,lowpassCutoff=150,Fs=self.fs,scale=40)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    OE.plot_ripple_overlay (ax[0],LFP_smooth_ep,SPAD_smooth_ep,frequency_spad,power_spad,time,ripple_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=True,plotRipple=False)                                   
                    #Set the title of ripple feature
                    plot_title = "Local Field Potential with Spectrogram" 
                    sst_ep,frequency,power_LFP,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=250,Fs=self.fs,scale=40) 
                    OE.plot_ripple_overlay (ax[1],LFP_smooth_ep,SPAD_smooth_ep,frequency,power_LFP,time,ripple_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotRipple=False)
                    
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_short_ep,lowpassCutoff=250,Fs=self.fs,scale=40)                
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    #Set the title of ripple feature
                    plot_title = f"Ripple Peak std:{ripple_std:.2f}, Ripple Duration:{ripple_duration:.2f} ms" 
                    OE.plot_two_trace_overlay(ax[2], time,ripple_band_filtered_ep,SPAD_ripple_band_filtered_ep, title='Wavelet Power Spectrum',color1='black', color2='lime') 
                    ax[0].axvline(0, color='white',linewidth=2)
                    ax[1].axvline(0, color='white',linewidth=2)
                    ax[2].axvline(0, color='white',linewidth=2) 
                    plt.tight_layout() 
                    plt.show()
                    ripple_std=rip_tsd.iloc[i]
                    ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  #second to ms
                    while True: #remove noise by cutting part of the synchronised the data
                        keepRipple = input("Enter whether to keep this ripple ('y' or 'n'): ")
                        if keepRipple.lower() == 'y':
                            self.ripple_std_values.append(ripple_std)
                            self.ripple_duration_values.append(ripple_duration)
                            self.ripple_time_cured.append(rip_tsd.index[i])
                            self.ripple_optic_power_values.append(power_spad)
                            self.ripple_LFP_power_values.append(power_LFP)
                            break
                        if keepRipple.lower() == 'n':
                            break 
        self.ripple_time_cured = pd.Series(self.ripple_time_cured)
        
        if len(self.ripple_std_values) !=0:
            self.ripple_std_mean= sum(self.ripple_std_values) / len(self.ripple_std_values)
            self.ripple_duration_mean =sum(self.ripple_duration_values) / len(self.ripple_duration_values)
           
        if len(self.ripple_time_cured)>2:
            self.Oscillation_triggered_Optical_transient_raw (mode='ripple',lfp_channel=lfp_channel, half_window=0.2, plot_single_trace=True,plotShade='CI')
            #self.Oscillation_optical_correlation (mode='ripple',lfp_channel=lfp_channel, half_window=0.2)
        if plot_ripple_ep:
            'plot averaged power spectrum'
            # Ensure all arrays have the same shape
            expected_shape = (29, 2001)
            for i, arr in enumerate(self.ripple_LFP_power_values):
                if arr.shape != expected_shape:
                    #print(f"Array at index {i} has shape {arr.shape}, resizing to {expected_shape}")
                    self.ripple_LFP_power_values[i] = np.resize(arr, expected_shape)
            # Calculate the average of the arrays
            average_LFP_powerSpectrum = np.mean(self.ripple_LFP_power_values, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            OE.plot_power_spectrum (ax,time,frequency, average_LFP_powerSpectrum,colorbar=True)
            # Ensure all arrays have the same shape
            expected_shape = (29, 2001)
            for i, arr in enumerate(self.ripple_optic_power_values):
                if arr.shape != expected_shape:
                    #print(f"Array at index {i} has shape {arr.shape}, resizing to {expected_shape}")
                    self.ripple_optic_power_values[i] = np.resize(arr, expected_shape)
            # Calculate the average of the arrays
            average_optic_powerSpectrum = np.mean(self.ripple_optic_power_values, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            OE.plot_power_spectrum (ax,time,frequency, average_optic_powerSpectrum,colorbar=True)
        return self.ripple_time_cured
    
    def pynappleGammaAnalysis (self,lfp_channel='LFP_2',ep_start=0,ep_end=10,
                          Low_thres=1,High_thres=10,plot_segment=False,plot_ripple_ep=True,excludeTheta=True,
                          excludeREM=False,excludeNonTheta=False):
        'This is the LFP data that need to be saved for the sync ananlysis'
        data_segment=self.Ephys_tracking_spad_aligned
        #data_segment=self.non_theta_part
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        #timestamps=timestamps-timestamps[0]
        #Use non-theta part to detect ripple
        lfp_data=data_segment[lfp_channel]
        spad_data=data_segment['zscore_raw']
        lfp_data=lfp_data/1000 #change the unit from uV to mV
        SPAD_cutoff=200
        SPAD_smooth_np = OE.smooth_signal(spad_data,Fs=self.fs,cutoff=SPAD_cutoff)
        'To align LFP and SPAD raw data to pynapple format'
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        SPAD=nap.Tsd(t = timestamps, d = spad_data.to_numpy(), time_units = 's')
        SPAD_smooth=nap.Tsd(t = timestamps, d = SPAD_smooth_np, time_units = 's')  
        'Calculate theta band for optical signal'
        'To detect gamma'
        ripple_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getRippleEvents (LFP,self.fs,windowlen=400,
                                                                           Low_thres=Low_thres,High_thres=High_thres,
                                                                           low_freq=30,high_freq=60)
        SPAD_ripple_band_filtered = pyna.eeg_processing.bandpass_filter(SPAD, 30, 80, self.fs)
        # SPAD_ripple_band_filtered = OE.band_pass_filter(SPAD,120,300,self.fs)
        # SPAD_ripple_band_filtered=nap.Tsd(t = timestamps, d = SPAD_ripple_band_filtered, time_units = 's')
        
        if excludeTheta:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'theta' in close_timestamps_df['BrainState'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near theta, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
            
        if excludeNonTheta:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'nontheta' in close_timestamps_df['BrainState'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near non-theta, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
            
        if excludeREM:
            'To remove detected ripples if they are during theta----meaning they are fast gamma'
            drop_index_ep=[]
            drop_index_std=[]
            for i in range (len(rip_ep)):
                ripple_std_time=rip_tsd.index[i]
                #close_timestamps_mask = np.abs(data_segment['timestamps'] -data_segment['timestamps'][0]- ripple_std_time) <= 0.01     
                close_timestamps_mask = np.abs(data_segment['timestamps'] - ripple_std_time) <= 0.01      
                close_timestamps_df = data_segment[close_timestamps_mask]
                if 'REM' in close_timestamps_df['REMstate'].values:
                    drop_index_ep.append(i)
                    drop_index_std.append(ripple_std_time)
                    print ('Romeve rip_ep near REM state, peak time is --', ripple_std_time)    
            rip_ep = rip_ep.drop(drop_index_ep)
            rip_tsd = rip_tsd.drop(drop_index_std)
        
        # Assign a value to the dynamically generated key
        self.ripple_numbers = len(rip_ep)
        'Calculate ripple frequency during non-theta periods'
        nontheta_length=len(data_segment[data_segment['BrainState'] == 'nontheta'])
        self.ripple_freq=np.round(self.ripple_numbers/(nontheta_length/self.fs),4)
        
        print('LFP length in seconds:',len(LFP)/self.fs)
        print('Optical signal length in seconds:',len(SPAD)/self.fs)
        print('Found ripple event numbers:',self.ripple_numbers)
        print('Ripple event frequency during non-theta:',self.ripple_freq, 'events/seconds')
        
        'To plot the choosen segment with start time and end time'
        if plot_segment:
            ex_ep = nap.IntervalSet(start = ep_start+timestamps[0], end = ep_end+timestamps[0], time_units = 's') 
            fig, ax = plt.subplots(6, 1, figsize=(10, 12))
            OE.plot_trace_nap (ax[0], LFP,ex_ep,color=sns.color_palette("husl", 8)[5],title='LFP raw Trace')
            OE.plot_trace_nap (ax[1], ripple_band_filtered,ex_ep,color=sns.color_palette("husl", 8)[5],title='Ripple band')
            OE.plot_ripple_event (ax[2], rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres=Low_thres) 
            OE.plot_trace_nap (ax[3], SPAD_smooth,ex_ep,color='green',title='calcium recording (z-score)')
            #LFP_rippleband=OE.band_pass_filter(LFP.restrict(ex_ep),150,250,Fs=self.fs)     
            sst,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered.restrict(ex_ep),lowpassCutoff=200,Fs=self.fs,scale=40)
            OE.plot_wavelet(ax[4],sst,frequency,power,Fs=self.fs,colorBar=False)
            #OE.plot_ripple_spectrum (ax[4], LFP, ex_ep,y_lim=30,Fs=self.fs,vmax_percentile=100)
            plt.subplots_adjust(hspace=0.5)
            for axi in ax:
                axi.set_xlabel(axi.get_xlabel(), fontsize=16)
                axi.set_ylabel(axi.get_ylabel(), fontsize=16)
                axi.title.set_fontsize(18)
                axi.tick_params(axis='both', labelsize=14)
            
        self.ripple_std_values=[]
        self.ripple_duration_values=[]

        event_peak_times=rip_tsd.index.to_numpy()
        for i in range(len(rip_ep)):
            ripple_std=rip_tsd.iloc[i]
            ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  #second to ms
            self.ripple_std_values.append(ripple_std)
            self.ripple_duration_values.append(ripple_duration)
            half_window_long=0.2
            half_window_short=0.2
            if event_peak_times[i]-timestamps[0]>half_window_long and timestamps[-1]-event_peak_times[i]>half_window_long:
                if plot_ripple_ep:
                    start_time=event_peak_times[i]-half_window_long
                    end_time=event_peak_times[i]+half_window_long
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    ripple_band_filtered_ep=ripple_band_filtered.restrict(rip_long_ep)   
                    SPAD_ripple_band_filtered_ep=SPAD_ripple_band_filtered.restrict(rip_long_ep)
                    start_time1=event_peak_times[i]-half_window_short
                    end_time1=event_peak_times[i]+half_window_short
                    rip_short_ep = nap.IntervalSet(start = start_time1, end = end_time1, time_units = 's') 
                    LFP_short_ep=LFP.restrict(rip_short_ep)
                    SPAD_short_smooth_ep=SPAD_smooth.restrict(rip_short_ep)
                    ripple_band_filtered_short_ep=ripple_band_filtered.restrict(rip_short_ep)
                    SPAD_ripple_band_filtered_short_ep=SPAD_ripple_band_filtered.restrict(rip_short_ep)
                    save_ripple_path = os.path.join(self.savepath, self.recordingName+'_Ripples_'+lfp_channel)
                    if not os.path.exists(save_ripple_path):
                        os.makedirs(save_ripple_path)
                    fig, ax = plt.subplots(6, 1, figsize=(6, 12))
                    #Set the title of ripple feature
                    plot_title = "Optical signal triggerred by gamma" 
                    sst_ep,frequency_spad,power_spad,global_ws=OE.Calculate_wavelet(SPAD_ripple_band_filtered_ep,lowpassCutoff=150,Fs=self.fs,scale=80)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    OE.plot_ripple_trace(ax[0],time,SPAD_smooth_ep,color='green')
                    OE.plot_power_spectrum (ax[1],time,frequency_spad, power_spad,colorbar=False)                                   
                    #Set the title of ripple feature
                    plot_title = "Local Field Potential with Spectrogram" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_ep,lowpassCutoff=150,Fs=self.fs,scale=80) 
                    OE.plot_ripple_trace(ax[2],time,LFP_ep,color='black')
                    OE.plot_power_spectrum (ax[3],time,frequency, power,colorbar=False)  
                    
                    # sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(ripple_band_filtered_short_ep,lowpassCutoff=500,Fs=self.fs,scale=40)                
                    # time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    #Set the title of ripple feature
                    plot_title = f"Filtered Gamma Band" 
                    OE.plot_ripple_trace(ax[4],time,SPAD_ripple_band_filtered_short_ep,color='green')
                    OE.plot_ripple_trace(ax[5],time,ripple_band_filtered_short_ep,color='black')
                 
                    # ax[0].axvline(0, color='white',linewidth=2)
                    # ax[1].axvline(0, color='white',linewidth=2)
                    # ax[2].axvline(0, color='white',linewidth=2) 
                    plt.tight_layout() 
                    figName=self.recordingName+'_Ripple'+str(i)+'.png'
                    fig.savefig(os.path.join(save_ripple_path,figName))

        if len(self.ripple_std_values) !=0:
            self.ripple_std_mean= sum(self.ripple_std_values) / len(self.ripple_std_values)
            self.ripple_duration_mean =sum(self.ripple_duration_values) / len(self.ripple_duration_values)
          
        self.rip_ep=rip_ep
        self.rip_tsd=rip_tsd
        if len(self.rip_tsd)>2:
            self.Oscillation_triggered_Optical_transient (mode='ripple',lfp_channel=lfp_channel, half_window=0.2, plot_single_trace=True,plotShade='CI')
            self.Oscillation_optical_correlation (mode='ripple',lfp_channel=lfp_channel, half_window=0.2)
        return rip_ep,rip_tsd
    
    def pynappleThetaAnalysis (self,lfp_channel='LFP_2',ep_start=0,ep_end=10,Low_thres=1,High_thres=10,plot_segment=False, plot_ripple_ep=True):
        'This is the LFP data that need to be saved for the sync ananlysis'
        data_segment=self.Ephys_tracking_spad_aligned
        #data_segment=self.theta_part
        # duration= len(data_segment['timestamps'])/self.fs
        # timestamps = np.linspace(0, duration, len(data_segment['timestamps']), endpoint=False)
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
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
        'Calculate theta band for optical signal'
        theta_band_filtered_spad,nSS_spad,nSS3_spad,rip_ep_spad,rip_tsd_spad = OE.getThetaEvents (SPAD_smooth,self.fs,windowlen=2000,Low_thres=Low_thres,High_thres=High_thres)
        'To detect theta by LFP'
        theta_band_filtered,nSS,nSS3,rip_ep,rip_tsd = OE.getThetaEvents (LFP,self.fs,windowlen=2000,Low_thres=Low_thres,High_thres=High_thres)  
        
        if plot_segment:
            'To plot the choosen segment'
            ex_ep = nap.IntervalSet(start = ep_start+timestamps[0], end = ep_end+timestamps[0], time_units = 's') 
            fig, ax = plt.subplots(5, 1, figsize=(10, 12))
            OE.plot_trace_nap (ax[0], LFP,ex_ep,color=sns.color_palette("husl", 8)[5],title='LFP raw Trace')
            OE.plot_trace_nap (ax[1], theta_band_filtered,ex_ep,color=sns.color_palette("husl", 8)[5],title='Theta band')
            OE.plot_ripple_event (ax[2], rip_ep, rip_tsd, ex_ep, nSS, nSS3, Low_thres=Low_thres) 
            
            OE.plot_trace_nap (ax[3], SPAD_smooth,ex_ep,color='green',title='calcium recording (z-score)')
            LFP_thetaband=OE.band_pass_filter(LFP.restrict(ex_ep),4,20,Fs=self.fs)        
            sst,frequency,power,global_ws=OE.Calculate_wavelet(LFP_thetaband,lowpassCutoff=50,Fs=self.fs,scale=200)
            OE.plot_wavelet(ax[4],sst,frequency,power,Fs=self.fs,colorBar=False,logbase=True)
        plt.subplots_adjust(hspace=0.5)
    
        '''To calculate cross-correlation'''
        event_peak_times=rip_tsd.index.to_numpy()
        print ('Total theta number:',len(event_peak_times))
        for i in range(len(rip_ep)):
            ripple_std=rip_tsd.iloc[i]
            ripple_duration=((rip_ep.iloc[[i]]['end']-rip_ep.iloc[[i]]['start'])*1000)[0]  
            if event_peak_times[i]-timestamps[0]>=0.5 and timestamps[-1]-event_peak_times[i]>=0.5:
                if plot_ripple_ep:
                    start_time=event_peak_times[i]-0.5
                    end_time=event_peak_times[i]+0.5
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    theta_band_filtered_spad_ep=theta_band_filtered_spad.restrict(rip_long_ep)  
                    theta_band_filtered_ep=theta_band_filtered.restrict(rip_long_ep)  
                    save_theta_path = os.path.join(self.savepath, self.recordingName+'_Thetas_'+lfp_channel)
                    if not os.path.exists(save_theta_path):
                        os.makedirs(save_theta_path)
                    fig, ax = plt.subplots(3, 1, figsize=(9, 12))
                    #Set the title of ripple feature
                    plot_title = "Optical signal triggered by theta peak" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(theta_band_filtered_spad_ep,lowpassCutoff=SPAD_cutoff,Fs=self.fs,scale=400)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    OE.plot_theta_overlay (ax[0],LFP_ep,SPAD_smooth_ep,frequency,power,time,theta_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=True,plotTheta=False)                                   
                    #Set the title of ripple feature
                    plot_title = "Local Field Potential with Spectrogram" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(theta_band_filtered_ep,lowpassCutoff=100,Fs=self.fs,scale=400) 
                    OE.plot_theta_overlay (ax[1],LFP_ep,SPAD_smooth_ep,frequency,power,time,theta_band_filtered_ep,plot_title,plotLFP=True,plotSPAD=False,plotTheta=False)                   
                    ripple_band_filtered_ep=theta_band_filtered.restrict(rip_long_ep)           
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    #Set the title of ripple feature
                    plot_title = f"Theta Peak std:{ripple_std:.2f}, Theta Duration:{ripple_duration:.2f} ms" 
                    OE.plot_theta_overlay (ax[2],LFP_ep,theta_band_filtered_spad_ep,frequency,power,time,ripple_band_filtered_ep,plot_title,plotLFP=False,plotSPAD=True,plotTheta=True)
                    ax[0].axvline(0, color='white',linewidth=2)
                    ax[1].axvline(0, color='white',linewidth=2)
                    ax[2].axvline(0, color='white',linewidth=2) 
                    # Optionally remove ticks
                    ax[0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    ax[1].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    
                    plt.tight_layout()
                    figName=self.recordingName+'Theta'+str(i)+'.png'
                    fig.savefig(os.path.join(save_theta_path,figName))
     
        self.theta_ep=rip_ep
        self.theta_tsd=rip_tsd
        if len(self.theta_tsd)>0:
            self.Oscillation_triggered_Optical_transient (mode='theta',lfp_channel=lfp_channel,half_window=0.5,plot_single_trace=True,plotShade='CI')
            self.Oscillation_optical_correlation (mode='theta',lfp_channel=lfp_channel, half_window=0.5)
        return data_segment,timestamps
    
    def PlotThetaNestedGamma (self,lfp_channel='LFP_2',Low_thres=1,High_thres=10,plot_segment=False, plot_ripple_ep=True):
        'This is the LFP data that need to be saved for the sync ananlysis'
        data_segment=self.Ephys_tracking_spad_aligned
        #data_segment=self.theta_part
        timestamps=data_segment['timestamps'].copy()
        timestamps=timestamps.to_numpy()
        #timestamps=timestamps-timestamps[0]
        #Use non-theta part to detect ripple
        lfp_data=data_segment[lfp_channel]
        spad_data=data_segment['zscore_raw']
        lfp_data=lfp_data/1000 #change the unit from uV to mV
        SPAD_cutoff=200
        SPAD_smooth_np = OE.smooth_signal(spad_data,Fs=self.fs,cutoff=SPAD_cutoff)
        'To align LFP and SPAD raw data to pynapple format'
        LFP=nap.Tsd(t = timestamps, d = lfp_data.to_numpy(), time_units = 's')
        SPAD=nap.Tsd(t = timestamps, d = spad_data.to_numpy(), time_units = 's')
        SPAD_smooth=nap.Tsd(t = timestamps, d = SPAD_smooth_np, time_units = 's')
        'Calculate theta band for optical signal'
        theta_band_filtered_spad,_,_,_,_ = OE.getThetaEvents (SPAD,self.fs,windowlen=2000,
                                                                              Low_thres=Low_thres,High_thres=High_thres)
        gamma_band_filtered_spad,_,_,_,_ = OE.getRippleEvents (SPAD,self.fs,windowlen=1000,
                                                                            Low_thres=0,High_thres=8,
                                                                            low_freq=30,high_freq=80)
        'To detect theta by LFP'
        theta_band_filtered,_,_,rip_ep,rip_tsd = OE.getThetaEvents (LFP,self.fs,windowlen=2000,
                                                                         Low_thres=Low_thres,High_thres=High_thres)  
        gamma_band_filtered,_,_,_,_ = OE.getRippleEvents (LFP,self.fs,windowlen=1000,
                                                                            Low_thres=0,High_thres=8,
                                                                            low_freq=30,high_freq=80)

        '''To calculate cross-correlation'''
        event_peak_times=rip_tsd.index.to_numpy()
        print ('Total theta number:',len(event_peak_times))
        for i in range(len(rip_ep)):
            if event_peak_times[i]-timestamps[0]>=0.5 and timestamps[-1]-event_peak_times[i]>=0.5:
                if plot_ripple_ep:
                    start_time=event_peak_times[i]-0.5
                    end_time=event_peak_times[i]+0.5
                    rip_long_ep = nap.IntervalSet(start = start_time, end = end_time, time_units = 's') 
                    LFP_ep=LFP.restrict(rip_long_ep)
                    SPAD_smooth_ep=SPAD_smooth.restrict(rip_long_ep)
                    theta_band_filtered_spad_ep=theta_band_filtered_spad.restrict(rip_long_ep)  
                    gamma_band_filtered_spad_ep=gamma_band_filtered_spad.restrict(rip_long_ep)
                    theta_band_filtered_ep=theta_band_filtered.restrict(rip_long_ep)  
                    gamma_band_filtered_ep=gamma_band_filtered.restrict(rip_long_ep)
                    save_theta_path = os.path.join(self.savepath, self.recordingName+'_Thetas_'+lfp_channel)
                    if not os.path.exists(save_theta_path):
                        os.makedirs(save_theta_path)
                   

                    fig, ax = plt.subplots(6, 1, figsize=(9, 12))
                    #Set the title of ripple feature
                    plot_title = "Theta nested gamma (Optical)" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(gamma_band_filtered_spad_ep,lowpassCutoff=200,Fs=self.fs,scale=40)
                    time = np.arange(-len(sst_ep)/2,len(sst_ep)/2) *(1/self.fs)
                    
                    #OE.plot_two_trace_overlay(ax[0], time,SPAD_smooth_ep,theta_band_filtered_spad_ep, title='Theta band optical',color1='lime', color2='black')   
                    OE.plot_ripple_trace(ax[0],time,SPAD_smooth_ep,color=sns.color_palette("husl", 8)[3])
                    OE.plot_ripple_trace(ax[1],time,gamma_band_filtered_spad_ep,color='red')
                    OE.plot_theta_nested_gamma_overlay (ax[2],LFP_ep,gamma_band_filtered_spad_ep,frequency,power,time,
                                           theta_band_filtered_spad_ep,100,plot_title,plotLFP=False,plotSPAD=False,plotTheta=True)   
                        
                    #Set the title of ripple feature
                    plot_title = "Theta nested gamma (Ephys)" 
                    sst_ep,frequency,power,global_ws=OE.Calculate_wavelet(gamma_band_filtered_ep,lowpassCutoff=200,Fs=self.fs,scale=40) 
                    #OE.plot_two_trace_overlay(ax[3], time,LFP_ep,theta_band_filtered_ep, title='Theta band LFP',color1='blue', color2='black')   
                    OE.plot_ripple_trace(ax[3],time,LFP_ep,color=sns.color_palette("husl", 8)[5])
                    OE.plot_ripple_trace(ax[4],time,gamma_band_filtered_ep,color='red')
                    OE.plot_theta_nested_gamma_overlay (ax[5],gamma_band_filtered_ep,gamma_band_filtered_ep,frequency,power,time,
                                           theta_band_filtered_ep,100,plot_title,plotLFP=False,plotSPAD=False,plotTheta=True)
                    for axi in ax:
                        axi.set_xlabel(axi.get_xlabel(), fontsize=16)
                        axi.set_ylabel(axi.get_ylabel(), fontsize=16)
                        axi.title.set_fontsize(18)
                        axi.tick_params(axis='both', labelsize=14)

                    # Optionally remove ticks
                    ax[0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    ax[1].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    ax[2].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    ax[3].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    ax[4].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
                    
                    plt.tight_layout()
                    figName=self.recordingName+'ThetaNestedGamma'+str(i)+'.png'
                    fig.savefig(os.path.join(save_theta_path,figName),transparent=True)
     
        return rip_ep,rip_tsd
    
    def plot_gamma_power_on_theta_cycle(self,LFP_channel):
        silced_recording=self.theta_part
        silced_recording=silced_recording.reset_index(drop=True)

        silced_recording['theta_angle']=OE.calculate_theta_phase_angle(silced_recording[LFP_channel], theta_low=5, theta_high=9)
        #OE.plot_trace_in_seconds(silced_recording['theta_angle'],Fs=10000,title='theta angle')
        trough_index,peak_index = OE.calculate_theta_trough_index(silced_recording,Fs=self.fs)
        #print (trough_index)
        gamma_band=(30, 60)
    
        mi_lfp, mi_spad,fig=OE.plot_gamma_power_on_theta(self.fs,silced_recording,LFP_channel,peak_index,half_window=0.15,gamma_band=gamma_band)
        #OE.plot_gamma_power_heatmap_on_theta(self.fs,silced_recording,LFP_channel,peak_index,half_window=0.15,gamma_band=gamma_band)
        # '''plot correlation using hilbert '''
        #zscore=OE.smooth_signal(silced_recording['zscore_raw'], self.fs,200,window='flat')
        #OE.compute_and_plot_gamma_power_correlation(zscore, silced_recording[LFP_channel],gamma_band, self.fs)
        #OE.compute_and_plot_gamma_power_crosscorr(zscore, silced_recording[LFP_channel],gamma_band, self.fs)
        OE.plot_gamma_amplitude_on_theta_phase(silced_recording[LFP_channel], silced_recording['zscore_raw'], 
                                             self.fs, theta_band=(4, 12), gamma_band=gamma_band, bins=30)
        # Save the figure with a transparent background
        fig_path = os.path.join(self.savepath, 'Gamma Power on Theta Phase.png')
        fig.savefig(fig_path, transparent=True, bbox_inches='tight')
        return -1

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
        plt.title('Mean Cross-Correlation with Standard Deviation')
        plt.legend()
        #plt.grid()
        plt.show()
        return lags,mean_cross_corr,std_cross_corr  
    
    def Oscillation_triggered_Optical_transient_raw (self, mode='ripple',lfp_channel='LFP_2', half_window=0.2,plot_single_trace=False,plotShade='std'):
        if mode=='ripple':
            #event_peak_times=self.rip_tsd.index.to_numpy()
            event_peak_times=self.ripple_time_cured.to_numpy()
            savename='_RipplePeak_'
            cutoff=200
            
        timestamps=self.Ephys_tracking_spad_aligned['timestamps']
        z_score_values = []
        LFP_values_1=[]
        LFP_values_2=[]
        LFP_values_3=[]
        LFP_values_4=[]
        peak_values =[]
        peak_indexs = []
        peak_stds = []
        zscore_peak_window=half_window
        half_window_len=int(zscore_peak_window*self.fs)

        if plot_single_trace:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i in range(len(event_peak_times)):
            if event_peak_times[i]>=half_window and event_peak_times[i]<=timestamps.iloc[-1]-half_window:
                self.Ephys_tracking_spad_aligned['abs_diff'] = abs(self.Ephys_tracking_spad_aligned['timestamps'] - event_peak_times[i])
                closest_index = self.Ephys_tracking_spad_aligned['abs_diff'].idxmin()
                start_idx=closest_index-int(half_window*self.fs)
                end_idx=closest_index+int(half_window*self.fs)
                segment_data = self.Ephys_tracking_spad_aligned[start_idx:end_idx]
                segment_zscore=segment_data['zscore_raw']
                z_score=OE.smooth_signal(segment_zscore,Fs=self.fs,cutoff=cutoff)
                #z_score=segment_zscore.to_numpy()
                # normalise to zscore
                normalized_z_score = OE.getNormalised (z_score)
                #normalized_z_score = z_score
                z_score_values.append(normalized_z_score) 
                
                segment_LFP_1=segment_data['LFP_1']
                segment_LFP_2=segment_data['LFP_2']
                segment_LFP_3=segment_data['LFP_3']
                segment_LFP_4=segment_data['LFP_4']
                
                LFP_normalised_1=OE.getNormalised (segment_LFP_1)
                LFP_normalised_2=OE.getNormalised (segment_LFP_2)
                LFP_normalised_3=OE.getNormalised (segment_LFP_3)
                LFP_normalised_4=OE.getNormalised (segment_LFP_4)
              
                LFP_values_1.append(LFP_normalised_1)
                LFP_values_2.append(LFP_normalised_2)
                LFP_values_3.append(LFP_normalised_3)
                LFP_values_4.append(LFP_normalised_4)
                #Calculate optical peak triggerred by ripple peak
                mididx=int(len(normalized_z_score)/2)
                if self.indicator=='GECI':
                    peak_value, peak_index, peak_std=OE.find_peak_and_std(normalized_z_score[mididx-half_window_len:mididx+half_window_len],half_window_len,mode='max')
                else:
                    peak_value, peak_index, peak_std=OE.find_peak_and_std(normalized_z_score[mididx-half_window_len:mididx+half_window_len],half_window_len,mode='min')
                    
                peak_values.append(peak_value)
                peak_indexs.append(peak_index)
                peak_stds.append(peak_std)
                if plot_single_trace:
                    x = np.linspace(-half_window, half_window, len(z_score))
                    ax.plot(x,normalized_z_score)
        if plot_single_trace:
            [plt.axvline(x=0, color='green')]
            plt.xlabel('Time(seconds)')
            plt.ylabel('z-score')
            plt.title(f'{mode} triggered optical traces')
            plt.show()
        
        z_score_values = np.array(z_score_values)
        mean_z_score,std_z_score, CI_z_score=OE.calculateStatisticNumpy (z_score_values)
        
        LFP_values_1 = np.array(LFP_values_1)
        LFP_values_2 = np.array(LFP_values_2)
        LFP_values_3 = np.array(LFP_values_3)
        LFP_values_4 = np.array(LFP_values_4)
        
        mean_LFP_1,std_LFP_1, CI_LFP_1=OE.calculateStatisticNumpy (LFP_values_1)
        mean_LFP_2,std_LFP_2, CI_LFP_2=OE.calculateStatisticNumpy (LFP_values_2)
        mean_LFP_3,std_LFP_3, CI_LFP_3=OE.calculateStatisticNumpy (LFP_values_3)
        mean_LFP_4,std_LFP_4, CI_LFP_4=OE.calculateStatisticNumpy (LFP_values_4)
        'Plot LFP and optical signal during ripple/theta events'

        x = np.linspace(-half_window, half_window, len(mean_z_score))
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        MakePlots.plot_oscillation_epoch_traces(ax[0,0],x,mean_z_score,mean_LFP_1,std_z_score,
                                                      std_LFP_1,CI_z_score,CI_LFP_1,mode=mode,plotShade=plotShade)
        ax[0,0].legend().remove()
        MakePlots.plot_oscillation_epoch_traces(ax[0,1],x,mean_z_score,mean_LFP_2,std_z_score,
                                                      std_LFP_2,CI_z_score,CI_LFP_2,mode=mode,plotShade=plotShade)
        ax[0,1].legend().remove()
        MakePlots.plot_oscillation_epoch_traces(ax[1,0],x,mean_z_score,mean_LFP_3,std_z_score,
                                                      std_LFP_3,CI_z_score,CI_LFP_3,mode=mode,plotShade=plotShade)
        ax[1,0].legend().remove()
        MakePlots.plot_oscillation_epoch_traces(ax[1,1],x,mean_z_score,mean_LFP_4,std_z_score,
                                                      std_LFP_4,CI_z_score,CI_LFP_4,mode=mode,plotShade=plotShade)
        ax[1,1].legend().remove()
        
        ax[0,0].set_title('Electrode 1')
        ax[0,1].set_title('Electrode 2')
        ax[1,0].set_title('Electrode 3')
        ax[1,1].set_title('Electrode 4')
        # Save and show plot
        fig.suptitle(f'Mean optical transient triggered by {mode} peak in {lfp_channel}')
        plt.tight_layout()
        plt.show()
        
        peak_values=np.array(peak_values)
        peak_indexs=np.array(peak_indexs)
        peak_times=peak_indexs/self.fs-zscore_peak_window
        peak_stds=np.array(peak_stds)
        if mode=='ripple':
            self.ripple_triggered_zscore_values=z_score_values
            self.ripple_triggered_LFP_values_1=LFP_values_1
            self.ripple_triggered_LFP_values_2=LFP_values_2
            self.ripple_triggered_LFP_values_3=LFP_values_3
            self.ripple_triggered_LFP_values_4=LFP_values_4
            self.ripple_triggered_optical_peak_times=peak_times
            self.ripple_triggered_optical_peak_values=peak_values
            
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[0,0],x,peak_times,peak_values,
                                                       mean_LFP_1,std_LFP_1,CI_LFP_1,half_window,mode=mode,plotShade=plotShade)
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[0,1],x,peak_times,peak_values,
                                                       mean_LFP_2,std_LFP_2,CI_LFP_2,half_window,mode=mode,plotShade=plotShade)
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[1,0],x,peak_times,peak_values,
                                                       mean_LFP_3,std_LFP_3,CI_LFP_3,half_window,mode=mode,plotShade=plotShade)
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[1,1],x,peak_times,peak_values,
                                                       mean_LFP_4,std_LFP_4,CI_LFP_4,half_window,mode=mode,plotShade=plotShade)
        ax[0,0].set_title('Electrode 1')
        ax[0,1].set_title('Electrode 2')
        ax[1,0].set_title('Electrode 3')
        ax[1,1].set_title('Electrode 4')
        ax[0,0].legend().remove()
        ax[0,1].legend().remove()
        ax[1,0].legend().remove()
        ax[1,1].legend().remove()
        fig.suptitle(f'Optical peaks during {mode} Event on {lfp_channel}')
        plt.tight_layout()
        plt.show()
        return -1

    def Oscillation_triggered_Optical_transient (self, mode='ripple',lfp_channel='LFP_2', half_window=0.2,plot_single_trace=False,plotShade='std'):
        if mode=='ripple':
            event_peak_times=self.rip_tsd.index.to_numpy()
            savename='_RipplePeak_'
            cutoff=150
        if mode=='theta':
            event_peak_times=self.theta_tsd.index.to_numpy()
            savename='_ThetaPeak_'
            cutoff=20
                 
        timestamps=self.Ephys_tracking_spad_aligned['timestamps']
        z_score_values = []
        LFP_values_1=[]
        LFP_values_2=[]
        LFP_values_3=[]
        LFP_values_4=[]
        peak_values =[]
        peak_indexs = []
        peak_stds = []
        zscore_peak_window=half_window
        half_window_len=int(zscore_peak_window*self.fs)

        if plot_single_trace:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for i in range(len(event_peak_times)):
            if event_peak_times[i]-timestamps[0]>=half_window and event_peak_times[i]<=timestamps.iloc[-1]-half_window:
                #print ('Trial:',i,'-event_peak_times is:',event_peak_times[i])
                #print ('timestamps[0]:',timestamps[0])
                self.Ephys_tracking_spad_aligned['abs_diff'] = abs(self.Ephys_tracking_spad_aligned['timestamps'] - event_peak_times[i])
                closest_index = self.Ephys_tracking_spad_aligned['abs_diff'].idxmin()
                start_idx=closest_index-int(half_window*self.fs)
                end_idx=closest_index+int(half_window*self.fs)
                segment_data = self.Ephys_tracking_spad_aligned[start_idx:end_idx]
                segment_zscore=segment_data['zscore_raw']
                z_score=OE.smooth_signal(segment_zscore,Fs=self.fs,cutoff=cutoff)
                #z_score=OE.butter_filter(segment_zscore, btype='low', cutoff=cutoff, fs=self.fs, order=5)
                #z_score=segment_zscore.to_numpy()
                # normalise to zscore
                normalized_z_score = OE.getNormalised (z_score)
                #normalized_z_score = z_score
                z_score_values.append(normalized_z_score) 
                
                segment_LFP_1=segment_data['LFP_1']
                segment_LFP_2=segment_data['LFP_2']
                segment_LFP_3=segment_data['LFP_3']
                segment_LFP_4=segment_data['LFP_4']
                
                LFP_smooth_1=OE.smooth_signal(segment_LFP_1,Fs=self.fs,cutoff=500)
                LFP_smooth_2=OE.smooth_signal(segment_LFP_2,Fs=self.fs,cutoff=500)
                LFP_smooth_3=OE.smooth_signal(segment_LFP_3,Fs=self.fs,cutoff=500)
                LFP_smooth_4=OE.smooth_signal(segment_LFP_4,Fs=self.fs,cutoff=500)
                LFP_normalised_1=OE.getNormalised (LFP_smooth_1)
                LFP_normalised_2=OE.getNormalised (LFP_smooth_2)
                LFP_normalised_3=OE.getNormalised (LFP_smooth_3)
                LFP_normalised_4=OE.getNormalised (LFP_smooth_4)
                
                LFP_values_1.append(LFP_smooth_1)
                LFP_values_2.append(LFP_smooth_2)
                LFP_values_3.append(LFP_smooth_3)
                LFP_values_4.append(LFP_smooth_4)
              
                # LFP_values_1.append(LFP_normalised_1)
                # LFP_values_2.append(LFP_normalised_2)
                # LFP_values_3.append(LFP_normalised_3)
                # LFP_values_4.append(LFP_normalised_4)
                #Calculate optical peak triggerred by ripple peak
                mididx=int(len(normalized_z_score)/2)
                if self.indicator=='GECI':
                    peak_value, peak_index, peak_std=OE.find_peak_and_std(normalized_z_score[mididx-half_window_len:mididx+half_window_len],half_window_len,mode='max')
                else:
                    peak_value, peak_index, peak_std=OE.find_peak_and_std(normalized_z_score[mididx-half_window_len:mididx+half_window_len],half_window_len,mode='min')
                    
                peak_values.append(peak_value)
                peak_indexs.append(peak_index)
                peak_stds.append(peak_std)
                if plot_single_trace:
                    x = np.linspace(-half_window, half_window, len(z_score))
                    ax.plot(x,normalized_z_score)
        if plot_single_trace:
            [plt.axvline(x=0, color='green')]
            plt.xlabel('Time(seconds)')
            plt.ylabel('z-score')
            plt.title(f'{mode} triggered optical traces')
            figName=self.recordingName+savename+'singleOptical_'+lfp_channel+'.png'
            plt.savefig(os.path.join(self.savepath,figName))
            plt.show()
        
        z_score_values = np.array(z_score_values)
        mean_z_score,std_z_score, CI_z_score=OE.calculateStatisticNumpy (z_score_values)
        
        LFP_values_1 = np.array(LFP_values_1)
        LFP_values_2 = np.array(LFP_values_2)
        LFP_values_3 = np.array(LFP_values_3)
        LFP_values_4 = np.array(LFP_values_4)
        
        mean_LFP_1,std_LFP_1, CI_LFP_1=OE.calculateStatisticNumpy (LFP_values_1)
        mean_LFP_2,std_LFP_2, CI_LFP_2=OE.calculateStatisticNumpy (LFP_values_2)
        mean_LFP_3,std_LFP_3, CI_LFP_3=OE.calculateStatisticNumpy (LFP_values_3)
        mean_LFP_4,std_LFP_4, CI_LFP_4=OE.calculateStatisticNumpy (LFP_values_4)
        'Plot LFP and optical signal during ripple/theta events'

        x = np.linspace(-half_window, half_window, len(mean_z_score))
        
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        MakePlots.plot_oscillation_epoch_traces(ax[0,0],x,mean_z_score,mean_LFP_1,std_z_score,
                                                      std_LFP_1,CI_z_score,CI_LFP_1,mode=mode,plotShade=plotShade)
        ax[0,0].legend().remove()
        MakePlots.plot_oscillation_epoch_traces(ax[0,1],x,mean_z_score,mean_LFP_2,std_z_score,
                                                      std_LFP_2,CI_z_score,CI_LFP_2,mode=mode,plotShade=plotShade)
        ax[0,1].legend().remove()
        MakePlots.plot_oscillation_epoch_traces(ax[1,0],x,mean_z_score,mean_LFP_3,std_z_score,
                                                      std_LFP_3,CI_z_score,CI_LFP_3,mode=mode,plotShade=plotShade)
        ax[1,0].legend().remove()
        MakePlots.plot_oscillation_epoch_traces(ax[1,1],x,mean_z_score,mean_LFP_4,std_z_score,
                                                      std_LFP_4,CI_z_score,CI_LFP_4,mode=mode,plotShade=plotShade)
        ax[1,1].legend().remove()
        
        ax[0,0].set_title('Electrode 1')
        ax[0,1].set_title('Electrode 2')
        ax[1,0].set_title('Electrode 3')
        ax[1,1].set_title('Electrode 4')
        # Save and show plot
        fig.suptitle(f'Mean optical transient triggered by {mode} peak in {lfp_channel}')
        plt.tight_layout()
        figName = self.recordingName + savename + 'OpticalMean_' + lfp_channel + '.png'
        fig.savefig(os.path.join(self.savepath, figName), transparent=True)
        plt.show()
        
        peak_values=np.array(peak_values)
        peak_indexs=np.array(peak_indexs)
        peak_times=peak_indexs/self.fs-zscore_peak_window
        peak_stds=np.array(peak_stds)
        if mode=='ripple':
            self.ripple_triggered_zscore_values=z_score_values
            self.ripple_triggered_LFP_values_1=LFP_values_1
            self.ripple_triggered_LFP_values_2=LFP_values_2
            self.ripple_triggered_LFP_values_3=LFP_values_3
            self.ripple_triggered_LFP_values_4=LFP_values_4
            self.ripple_triggered_optical_peak_times=peak_times
            self.ripple_triggered_optical_peak_values=peak_values
        if mode=='theta':
            self.theta_triggered_zscore_values=z_score_values
            self.theta_triggered_LFP_values_1=LFP_values_1
            self.theta_triggered_LFP_values_2=LFP_values_2
            self.theta_triggered_LFP_values_3=LFP_values_3
            self.theta_triggered_LFP_values_4=LFP_values_4
            self.theta_triggered_optical_peak_times=peak_times
            self.theta_triggered_optical_peak_values=peak_values
            
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[0,0],x,peak_times,peak_values,
                                                       mean_LFP_1,std_LFP_1,CI_LFP_1,half_window,mode=mode,plotShade=plotShade)
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[0,1],x,peak_times,peak_values,
                                                       mean_LFP_2,std_LFP_2,CI_LFP_2,half_window,mode=mode,plotShade=plotShade)
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[1,0],x,peak_times,peak_values,
                                                       mean_LFP_3,std_LFP_3,CI_LFP_3,half_window,mode=mode,plotShade=plotShade)
        MakePlots.plot_oscillation_epoch_optical_peaks(ax[1,1],x,peak_times,peak_values,
                                                       mean_LFP_4,std_LFP_4,CI_LFP_4,half_window,mode=mode,plotShade=plotShade)
        ax[0,0].set_title('Electrode 1')
        ax[0,1].set_title('Electrode 2')
        ax[1,0].set_title('Electrode 3')
        ax[1,1].set_title('Electrode 4')
        ax[0,0].legend().remove()
        ax[0,1].legend().remove()
        ax[1,0].legend().remove()
        ax[1,1].legend().remove()
        fig.suptitle(f'Optical peaks during {mode} Event on {lfp_channel}')
        plt.tight_layout()
        figName=self.recordingName+savename+'peaktime_'+lfp_channel+'.png'
        fig.savefig(os.path.join(self.savepath,figName), transparent=True)
        plt.show()
        return -1
    
    def Oscillation_optical_correlation (self, mode='ripple',lfp_channel='LFP_2', half_window=0.2):
        if mode=='ripple':
            savename='_Ripple_'
            event_peak_times=self.rip_tsd.index.to_numpy()
            cutoff=100
        if mode=='theta':
            savename='_Theta_'
            event_peak_times=self.theta_tsd.index.to_numpy()
            cutoff=20
        cross_corr_values = []
        timestamps=self.Ephys_tracking_spad_aligned['timestamps']
        for i in range(len(event_peak_times)):
            if event_peak_times[i]-timestamps[0]>=half_window and event_peak_times[i]<=timestamps.iloc[-1]-half_window:
                self.Ephys_tracking_spad_aligned['abs_diff'] = abs(self.Ephys_tracking_spad_aligned['timestamps'] - event_peak_times[i])
                closest_index = self.Ephys_tracking_spad_aligned['abs_diff'].idxmin()
                start_idx=closest_index-int(half_window*self.fs)
                end_idx=closest_index+int(half_window*self.fs)
                segment_data = self.Ephys_tracking_spad_aligned[start_idx:end_idx]
                segment_zscore=segment_data['zscore_raw']
                segment_LFP=segment_data[lfp_channel]
                z_score=OE.smooth_signal(segment_zscore,Fs=self.fs,cutoff=cutoff)
                lags,cross_corr =OE.calculate_correlation_with_detrend (z_score,segment_LFP)
                cross_corr_values.append(cross_corr)
        cross_corr_values = np.array(cross_corr_values,dtype=float)
        # Truncate all columns to the common length   
        #event_corr_array=OE.align_numpy_array_to_same_length (cross_corr_values)
        event_corr_array=cross_corr_values
        mean_cross_corr,std_cross_corr, CI_cross_corr=OE.calculateStatisticNumpy (event_corr_array)
        max_index = np.argmax(np.abs(mean_cross_corr))
        max_mean = mean_cross_corr[max_index]
        max_CI = CI_cross_corr[0][max_index],CI_cross_corr[1][max_index]
        # print("Maximum Mean Index:", max_index)
        # print("Maximum Mean Value:", max_mean)
        # print("Corresponding CI Value:", max_CI)
        if mode=='ripple':
            self.ripple_event_corr_array=event_corr_array

        if mode=='theta':
            self.theta_event_corr_array=event_corr_array
            # Assuming mean_cross_corr and CI_cross_corr have already been calculated

            # # Save mean_cross_corr to a pickle file
            # with open(os.path.join(self.dpath,'mean_cross_corr.pkl'), 'wb') as f:
            #     pickle.dump(mean_cross_corr, f)
            # # Save CI_cross_corr to a separate pickle file
            # with open(os.path.join(self.dpath,'CI_cross_corr.pkl'), 'wb') as f:
            #     pickle.dump(CI_cross_corr, f)
            
        x = np.linspace((-len(mean_cross_corr)/2)/self.fs, (len(mean_cross_corr)/2)/self.fs, len(mean_cross_corr))  
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # Plot the mean cross-correlation
        ax.plot(x, mean_cross_corr, color='#404040', label='Mean Cross Correlation')
        # Fill between for the confidence interval
        ax.fill_between(x, CI_cross_corr[0], CI_cross_corr[1], color='#404040', alpha=0.2, label='0.95 CI')
        # Set labels and title
        ax.set_xlabel('Lags (seconds)', fontsize=14)
        ax.set_ylabel('Cross-Correlation', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title('Mean Cross-Correlation (1-Second Window)')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend().set_visible(False)
        #plt.grid()
        figName=self.recordingName+savename+'CrossCorrelation_'+lfp_channel+'.png'
        plt.savefig(os.path.join(self.savepath,figName), transparent=True)
        plt.show()
        return -1
    