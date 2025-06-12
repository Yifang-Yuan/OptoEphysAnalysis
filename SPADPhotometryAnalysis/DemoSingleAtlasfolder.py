# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:05:21 2024

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

dpath='F:/2025_ATLAS_SPAD/1881363_Jedi2p_mCherry/Day5Sleep/Test/'
#hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'

pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=3500)
#%%
AtlasDecode.show_image_with_pixel_array(avg_pixel_array,showPixel_label=True)
#%%
center_x, center_y,best_radius=AtlasDecode.find_circle_mask(avg_pixel_array,radius=10,threh=0.2)
#%%
#center_x, center_y,best_radius=53, 46, 15
Trace_raw=AtlasDecode.get_dff_from_pixel_array (pixel_array_all_frames,avg_pixel_array,hotpixel_path,center_x, center_y,best_radius,fs=840,snr_thresh=2)
'GET TOTAL PHOTON COUNT'
#Trace_raw,dff=AtlasDecode.get_total_photonCount_atlas_continuous_circle_mask (dpath,hotpixel_path,center_x, center_y,best_radius,fs=840,photoncount_thre=2000)

#%%
signal = Trace_raw  # Your square wave signal as a NumPy array
threshold = 200  # Set a threshold to separate high and low states
square_wave_frequency = 100  # Known frequency of the square wave (in Hz)
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
sampling_rate_calculated=1680/recording_duration
print(f"Estimated sampling rate {sampling_rate_calculated}")
#print('-------')
#%%
# Load the square wave signal
for i in range (3):
    signal = Trace_raw[int(840*10*i):int(840*10*(i+1))]  # Your square wave signal as a NumPy array
    threshold = 30  # Set a threshold to separate high and low states
    square_wave_frequency = 10  # Known frequency of the square wave (in Hz)
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
    sampling_rate_calculated=840*10/recording_duration
    print(f"Estimated sampling rate {sampling_rate_calculated}")
    print('-------')

#%%
data=Trace_raw
sampling_rate=840
#Plot the image of the pixel array
fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(Trace_raw[0:840],ax, fs=840, label="raw_data")
#%%
'''Wavelet spectrum ananlysis'''
Analysis.plot_wavelet_data(data,sampling_rate,cutoff=300,xlim = ([6,30]))

#%%
'''Read binary files for single ROI'''
# Display the grayscale image
pixel_array_all_frames,sum_pixel_array,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=2000)
#%%

#%%
ppath='D:/ATLAS_SPAD/1825505_SimCre/Day2/Atlas/'
item_path = os.path.join(ppath,  'pixel_array_all_frames.npy')
np.save(item_path, pixel_array_all_frames)
#%%
AtlasDecode.show_image_with_pixel_array(avg_pixel_array,showPixel_label=True)
AtlasDecode.show_image_with_pixel_array(sum_pixel_array,showPixel_label=True)
#%%
# sum_values_over_time,mean_values_over_time,region_pixel_array=get_trace_from_3d_pixel_array(pixel_array_all_frames,avg_pixel_array,xxrange,yyrange)
# fig, ax = plt.subplots(figsize=(8, 2))
# plot_trace(sum_values_over_time[1:],ax, fs=840, label="raw_data")
#%%
# for i in range(21):
#     show_image_with_pixel_array(pixel_array_all_frames[:,:,187+i],showPixel_label=True)
#pixel_array=pixel_array_all_frames[:,:,800]
#pixel_array_plot_hist(pixel_array_all_frames[:,:,1000], plot_min_thre=100)
#%%
