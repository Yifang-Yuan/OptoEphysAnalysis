# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:26:13 2024

@author: Yifang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
from SPADPhotometryAnalysis import AtlasDecode
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from scipy.ndimage import uniform_filter
#%% 
dpath='E:/ATLAS_SPAD/1836687_PV_g8f_taperfibre/Day3/Burst-RS-25200frames-840Hz_2024-12-14_15-01/'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
photoncount_thre=1000
fs=840
snr_thresh=2

pixel_array_all_frames_1,_,average_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
pixel_array_all_frames_1[:, :, 0]=pixel_array_all_frames_1[:, :,1]
#%%
dpath='E:/ATLAS_SPAD/1836687_PV_g8f_taperfibre/Day3/Burst-RS-25200frames-840Hz_2024-12-14_15-04/'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
photoncount_thre=1000
fs=840
snr_thresh=2

pixel_array_all_frames_2,_,average_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
pixel_array_all_frames_2[:, :, 0]=pixel_array_all_frames_2[:, :,1]
#%%
pixel_array_all_frames = np.concatenate((pixel_array_all_frames_1, pixel_array_all_frames_2), axis=2)
#%%
center_x, center_y,radius=AtlasDecode.find_circle_mask(average_pixel_array,radius=30,threh=0.1)
#%%
ring_outer_radius=25
width = 10
ring_inner_radius = ring_outer_radius - width
middle_circle_radius = 10
#%%
mean_image, std_image, snr_image = AtlasDecode.get_snr_image(pixel_array_all_frames)
pixel_mask = AtlasDecode.mask_low_snr_pixels(snr_image, snr_thresh)
shape = pixel_array_all_frames.shape[0:2] 
y, x = np.ogrid[:shape[0], :shape[1]]

outer_circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= ring_outer_radius ** 2
inner_circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= ring_inner_radius ** 2
ring_mask = outer_circle_mask & ~inner_circle_mask

middle_circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= middle_circle_radius ** 2

fig, ax = plt.subplots(figsize=(5, 5))
pos = ax.imshow(snr_image, cmap='viridis')  # Adjust colormap if desired
ax.set_title('SNR')
fig.colorbar(pos, ax=ax)
circle1 = plt.Circle((center_x,center_y), ring_outer_radius, color='cyan', fill=False, linewidth=2, label='Ring Circle')
circle2 = plt.Circle((center_x,center_y), ring_inner_radius, color='cyan', fill=False, linewidth=2, label='Ring Circle')
circle3 = plt.Circle((center_x,center_y), middle_circle_radius, color='pink', fill=False, linewidth=2, label='Middle Circle')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.gca().add_patch(circle3)
plt.tight_layout()
plt.show()
#Inner circle trace
trace_inner = AtlasDecode.extract_trace(pixel_array_all_frames, middle_circle_mask, pixel_mask, activity = 'mean')

fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(trace_inner,ax, fs, label="trace_inner")
lambd = 10e3 # Adjust lambda to get the best fit
porder = 1
itermax = 15
sig_base=fp.airPLS(trace_inner,lambda_=lambd,porder=porder,itermax=itermax) 
signal_inner = (trace_inner - sig_base)  
dff_inner=100*signal_inner / sig_base

fig, ax = plt.subplots(figsize=(8, 2))
dff_inner_smooth=Analysis.get_bin_trace (dff_inner,bin_window=20,color='tab:blue',Fs=840)
AtlasDecode.plot_trace(dff_inner_smooth,ax, fs/20, label="inner signal df/f")
plt.show()

#outer ring trace
trace_outer = AtlasDecode.extract_trace(pixel_array_all_frames, ring_mask, pixel_mask, activity = 'mean')

fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(trace_outer,ax, fs, label="trace_outer")
sig_base=fp.airPLS(trace_outer,lambda_=lambd,porder=porder,itermax=itermax) 
signal_outer = (trace_outer - sig_base)  
dff_outer=100*signal_outer / sig_base

fig, ax = plt.subplots(figsize=(8, 2))
dff_outer_smooth=Analysis.get_bin_trace (dff_outer,bin_window=20,color='tab:blue',Fs=840)
AtlasDecode.plot_trace(dff_outer_smooth,ax, fs/20, label="outer signal df/f")
plt.show()

#%%
from scipy.stats import sem, t

bin_window=4
trace_inner_smooth=Analysis.get_bin_trace (signal_inner,bin_window=bin_window,color='tab:blue',Fs=840)
trace_outer_smooth=Analysis.get_bin_trace (signal_outer,bin_window=bin_window,color='tab:blue',Fs=840)
# Parameters
confidence_level = 0.95
segment_length=2

# Store cross-correlation results for each segment
cross_correlations = []

# Loop through each segment, calculate cross-correlation, and store it
for i in range(30):
    inner_i = trace_inner_smooth[int(i * 840*segment_length/bin_window):int((i + 1) * 840*segment_length/bin_window)]
    outer_i = trace_outer_smooth[int(i * 840*segment_length/bin_window):int((i + 1) * 840*segment_length/bin_window)]
    # Calculate the cross-correlation between inner and outer signals for this segment
    cross_corr = np.correlate(inner_i - np.mean(inner_i), outer_i - np.mean(outer_i), mode='full')
    cross_corr = cross_corr / (np.std(inner_i) * np.std(outer_i) * len(inner_i))  # Normalize cross-correlation
    lag = len(cross_corr) // 2
    cross_correlations.append(cross_corr)  # Take the defined lag range
    lag_range = np.arange(-lag, lag+1)  # Define the range of lags you want to consider
    fig, ax = plt.subplots(figsize=(12, 4))
    AtlasDecode.plot_trace(inner_i, ax, fs / bin_window, label="Inner Signal correct",unit='N/A')
    AtlasDecode.plot_trace(outer_i, ax, fs / bin_window, label="Outer Signal correct",unit='N/A')
    plt.legend()
    plt.legend(frameon=False)
    plt.show()

# Convert the list of cross-correlations to a numpy array
cross_correlations = np.array(cross_correlations)  # Shape: (30, len(lag_range))

# Calculate the average cross-correlation and confidence interval
mean_cross_corr = np.mean(cross_correlations, axis=0)
std_error = sem(cross_correlations, axis=0)
confidence_interval = t.interval(confidence_level, len(cross_correlations) - 1, loc=mean_cross_corr, scale=std_error)

# Plot the average cross-correlation with the confidence interval as a shaded area
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(lag_range, mean_cross_corr, color="blue", label="Average Cross-Correlation")
ax.fill_between(lag_range, confidence_interval[0], confidence_interval[1], color="blue", alpha=0.2, label="95% CI")
ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="Lag = 0")
ax.set_xlabel("Lag")
ax.set_ylabel("Cross-Correlation")
ax.set_title("Average Cross-Correlation with 95% Confidence Interval")
plt.legend()
plt.legend(frameon=False)
plt.show()

print("Average cross-correlation at each lag:", mean_cross_corr)
print("95% confidence interval at each lag:", confidence_interval)
#%%

