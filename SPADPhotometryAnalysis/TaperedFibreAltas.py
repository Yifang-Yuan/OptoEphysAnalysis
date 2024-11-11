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
#%% Workable code, above is testin
dpath='E:/ATLAS_SPAD/1825504_SimCre_GCamp8f_taper/Day2/Atlas/Burst-RS-25200frames-840Hz_2024-10-28_17-21/'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
photoncount_thre=800
fs=840
snr_thresh=1

pixel_array_all_frames,sum_pixel_array,_=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
#%%
#%%
center_x, center_y,radius=AtlasDecode.find_circle_mask(sum_pixel_array,radius=10)
#%%
ring_outer_radius=16
width =6 
ring_inner_radius = ring_outer_radius - width
middle_circle_radius = 6
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
trace_inner=trace_inner[1:]
trace_inner = np.append(trace_inner, trace_inner[-1])

fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(trace_inner,ax, fs, label="trace_inner")
lambd = 10e3 # Adjust lambda to get the best fit
porder = 1
itermax = 15
sig_base=fp.airPLS(trace_inner,lambda_=lambd,porder=porder,itermax=itermax) 
signal = (trace_inner - sig_base)  
dff_inner=100*signal / sig_base

fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(dff_inner,ax, fs, label="inner signal df/f")
plt.show()

#outer ring trace
trace_outer = AtlasDecode.extract_trace(pixel_array_all_frames, ring_mask, pixel_mask, activity = 'mean')
trace_outer=trace_outer[1:]
trace_outer = np.append(trace_outer, trace_outer[-1])

fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(trace_outer,ax, fs, label="trace_outer")
sig_base=fp.airPLS(trace_outer,lambda_=lambd,porder=porder,itermax=itermax) 
signal = (trace_outer - sig_base)  
dff_outer=100*signal / sig_base

fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(dff_outer,ax, fs, label="outer signal df/f")
plt.show()

#%%
from scipy.stats import sem, t
trace_inner_smooth=Analysis.get_bin_trace (dff_inner,bin_window=4,color='tab:blue',Fs=840)
trace_outer_smooth=Analysis.get_bin_trace (dff_outer,bin_window=4,color='tab:blue',Fs=840)
# Parameters
confidence_level = 0.95
lag_range = np.arange(-100, 101)  # Define the range of lags you want to consider

# Store cross-correlation results for each segment
cross_correlations = []

# Loop through each segment, calculate cross-correlation, and store it
for i in range(30):
    inner_i = trace_inner_smooth[i * 210:(i + 1) * 210]
    outer_i = trace_outer_smooth[i * 210:(i + 1) * 210]
    # Calculate the cross-correlation between inner and outer signals for this segment
    cross_corr = np.correlate(inner_i - np.mean(inner_i), outer_i - np.mean(outer_i), mode='full')
    cross_corr = cross_corr / (np.std(inner_i) * np.std(outer_i) * len(inner_i))  # Normalize cross-correlation
    mid = len(cross_corr) // 2
    cross_correlations.append(cross_corr[mid - 100:mid + 101])  # Take the defined lag range
    
    fig, ax = plt.subplots(figsize=(8, 2))
    AtlasDecode.plot_trace(inner_i, ax, fs / 4, label="Inner Signal df/f")
    AtlasDecode.plot_trace(outer_i, ax, fs / 4, label="Outer Signal df/f")
    plt.legend()
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
plt.show()

print("Average cross-correlation at each lag:", mean_cross_corr)
print("95% confidence interval at each lag:", confidence_interval)
#%%
from scipy.signal import correlate

# 1. Calculate cross-correlation and corresponding lags
lags = np.arange(-len(trace_inner_smooth) + 1, len(trace_inner_smooth))
cross_corr = correlate(trace_inner_smooth, trace_outer_smooth, mode='full')

# 2. Find the lag that maximizes the cross-correlation
optimal_lag = lags[np.argmax(cross_corr)]
print("Optimal lag (delay) in samples:", optimal_lag)

# 3. Plot cross-correlation against lags
plt.figure(figsize=(10, 5))
plt.plot(lags, cross_corr, color="purple")
plt.axvline(optimal_lag, color="red", linestyle="--", label=f"Optimal Lag = {optimal_lag} samples")
plt.xlabel("Lag")
plt.ylabel("Cross-Correlation")
plt.title("Cross-Correlation Between Inner and Outer Signals")
plt.legend()
plt.grid(True)
plt.show()

# 4. Overlay signals with optimal lag applied
plt.figure(figsize=(10, 5))
if optimal_lag > 0:
    aligned_inner = trace_inner_smooth[optimal_lag:]
    aligned_outer = trace_outer_smooth[:-optimal_lag]
elif optimal_lag < 0:
    aligned_inner = trace_inner_smooth[:optimal_lag]
    aligned_outer = trace_outer_smooth[-optimal_lag:]
else:
    aligned_inner = trace_inner_smooth
    aligned_outer = trace_outer_smooth

time_axis = np.arange(len(aligned_inner))  # Adjusted time axis after shifting

# Plot the aligned signals
plt.plot(time_axis, aligned_inner, label="Inner Signal (Shifted)", color="blue")
plt.plot(time_axis, aligned_outer, label="Outer Signal", color="orange")
plt.title("Aligned Signals with Optimal Lag")
plt.xlabel("Time (Adjusted)")
plt.ylabel("Signal Value")
plt.legend()
plt.show()
