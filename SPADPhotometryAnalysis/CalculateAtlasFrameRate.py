# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:20:42 2025

@author: Yifang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
from SPADPhotometryAnalysis import AtlasDecode
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
#%% Workable code, above is testin
#ppath='D:/ATLAS_SPAD/1825505_SimCre/Day2/Atlas/'

center_x, center_y,best_radius=47,60,25
#hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'

def calculate_frame_rate(signal,threshold,square_wave_fs,frame_num):
    threshold = threshold  # Set a threshold to separate high and low states
    square_wave_frequency = square_wave_fs  # Known frequency of the square wave (in Hz)
    period = 1 / square_wave_frequency  # Period of the square wave (in seconds)

    rising_edges = np.where((signal[:-1] <= threshold) & (signal[1:] > threshold))[0]  # Indices of rising edges
    # Count the number of cycles
    num_cycles = len(rising_edges)

    # Calculate recording duration
    recording_duration = num_cycles * period
    print(f"Number of cycles: {num_cycles}")
    print(f"Recording duration: {recording_duration:.2f} seconds")

    if len(rising_edges) > 1:  # Ensure there are at least two cycles
        cycle_samples = np.diff(rising_edges)  # Samples between consecutive rising edges
        print(f"Samples per cycle: {cycle_samples}")
        print(f"Average samples per cycle: {np.mean(cycle_samples)}")
    else:
        print("Not enough cycles to calculate samples per cycle.")
    sampling_rate_calculated=frame_num/recording_duration
    print(f"Estimated sampling rate {sampling_rate_calculated}")
    print('-------')
    return  sampling_rate_calculated


#%%


dpath='E:/2025_ATLAS_SPAD/FrameRate/Burst-RS-1680frames-840Hz_2025-01-14_13-31/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=5000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,200,100,1680)

dpath='E:/2025_ATLAS_SPAD/FrameRate/Burst-RS-1680frames-840Hz_2025-01-14_13-34/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=5000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,200,100,1680)

dpath='E:/2025_ATLAS_SPAD/FrameRate/Burst-RS-840frames-840Hz_2025-01-14_12-41/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=5000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,200,100,840)

