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
dpath='E:/ATLAS_SPAD/1825504_Sim1Cre_GCamp8f_taper/Day2/Burst-RS-25200frames-840Hz_2024-10-28_17-17/'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
photoncount_thre=800
fs=840
snr_thresh=1

pixel_array_all_frames,sum_pixel_array,_=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
#%%
center_x, center_y,radius=AtlasDecode.find_circle_mask(sum_pixel_array,radius=10)
#%%
outer_radius=12
width = 8 
inner_radius = outer_radius - width

#%%
mean_image, std_image, snr_image = AtlasDecode.get_snr_image(pixel_array_all_frames)
pixel_mask = AtlasDecode.mask_low_snr_pixels(snr_image, snr_thresh)
shape = pixel_array_all_frames.shape[0:2] 
y, x = np.ogrid[:shape[0], :shape[1]]

outer_circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= outer_radius ** 2
inner_circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= inner_radius ** 2
# Subtract inner mask from outer mask to get a ring mask
ring_mask = outer_circle_mask & ~inner_circle_mask

fig, ax = plt.subplots(figsize=(5, 5))
pos = ax.imshow(snr_image, cmap='viridis')  # Adjust colormap if desired
ax.set_title('SNR')
fig.colorbar(pos, ax=ax)
circle1 = plt.Circle((center_x,center_y), outer_radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
circle2 = plt.Circle((center_x,center_y), inner_radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)
plt.tight_layout()
plt.show()
#Inner circle trace
trace_inner = AtlasDecode.extract_trace(pixel_array_all_frames, inner_circle_mask, pixel_mask, activity = 'mean')
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
zscore_smooth=Analysis.get_bin_trace (dff_inner,bin_window=20,color='tab:blue',Fs=840)
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
zscore_smooth=Analysis.get_bin_trace (dff_outer,bin_window=20,color='tab:blue',Fs=840)
