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


center_x, center_y,best_radius=52,27,20
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

dpath='F:/2025_ATLAS_SPAD/FrameRate_newFOV/Burst-RS-17170frames-1717Hz_100HzSquareWave/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (dpath,hotpixel_path,photoncount_thre=200000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,10000,100,17170)
#%%
dpath='F:/2025_ATLAS_SPAD/FrameRate_newFOV/Burst-RS-17170frames-1717Hz_200HzSquareWave/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (dpath,hotpixel_path,photoncount_thre=200000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,4000,200,17170)
#%%
dpath='F:/2025_ATLAS_SPAD/FrameRate_newFOV/Burst-RS-51510frames-870Hz_2025-04-29_17-43_400Hz/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (dpath,hotpixel_path,photoncount_thre=200000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,40000,400,51510)

dpath='F:/2025_ATLAS_SPAD/FrameRate_newFOV/Burst-RS-51510frames-870Hz_2025-04-29_17-49_400Hz/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (dpath,hotpixel_path,photoncount_thre=200000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,40000,400,51510)

dpath='F:/2025_ATLAS_SPAD/FrameRate_newFOV/Burst-RS-51510frames-870Hz_2025-04-29_17-52_400Hz/'
pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (dpath,hotpixel_path,photoncount_thre=200000)
Trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=0)
calculate_frame_rate(Trace_raw,40000,400,51510)
