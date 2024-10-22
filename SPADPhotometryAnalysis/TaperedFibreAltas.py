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
#%% Workable code, above is testin
dpath='D:/ATLAS_SPAD/1825504_Sim1Cre_GCamp8f_taper/Burst-RS-25200frames-840Hz_2024-10-18_17-23/'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
photoncount_thre=500
fs=840

pixel_array_all_frames,sum_pixel_array,_=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
AtlasDecode.show_image_with_pixel_array(sum_pixel_array,showPixel_label=True)
#%%
xxrange = [40, 90]
yyrange = [45, 95]
Trace_raw,z_score,pixel_array_all_frames=AtlasDecode.get_zscore_from_atlas_snr_mask (dpath,hotpixel_path,xxrange,yyrange,fs=840,snr_thresh=2)
#%%
pixel_array_all_frames=AtlasDecode.decode_atlas_folder_without_hotpixel_removal (dpath)
        
snr_thresh=2
mean_image, std_image, snr_image = AtlasDecode.get_snr_image(pixel_array_all_frames)
# look at the snr_image with colorbar and set this (pixel value below thresh will be 0 in the mask) 
pixel_mask = AtlasDecode.mask_low_snr_pixels(snr_image, snr_thresh)
#%%
xxrange1 = [40, 90]
yyrange1 = [45, 95]

xxrange2 = [70, 90]
yyrange2 = [60, 80]

xxrange3 = [60, 70]
yyrange3 = [65, 75]

xxrange4 = [40, 60]
yyrange4 = [60, 80]

i=0
for i in range (4):
    if i==0:
        roi_mask,_ = AtlasDecode.construct_roi_mask(xx_1 = xxrange1[0], xx_2 = xxrange1[1], yy_1 = yyrange1[0] , yy_2 = yyrange1[1])
    elif i==1:
        roi_mask,_ = AtlasDecode.construct_roi_mask(xx_1 = xxrange2[0], xx_2 = xxrange2[1], yy_1 = yyrange2[0] , yy_2 = yyrange2[1])
    elif i==2:
        roi_mask,_ = AtlasDecode.construct_roi_mask(xx_1 = xxrange3[0], xx_2 = xxrange3[1], yy_1 = yyrange3[0] , yy_2 = yyrange3[1])
    elif i==3:
        roi_mask,_ = AtlasDecode.construct_roi_mask(xx_1 = xxrange4[0], xx_2 = xxrange4[1], yy_1 = yyrange4[0] , yy_2 = yyrange4[1])
        
    trace = AtlasDecode.extract_trace(pixel_array_all_frames, roi_mask, pixel_mask, activity = 'mean')    
    Trace_raw=trace[1:]
    
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])
    fig, ax = plt.subplots(figsize=(8, 2))
    Trace_raw=Trace_raw[0:8400]
    
    AtlasDecode.plot_trace(Trace_raw,ax, fs, label="raw_data")
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    z_score=(signal - np.median(signal)) / np.std(signal)
    
    zscore_smooth=Analysis.get_bin_trace (z_score,bin_window=42,color='tab:blue',Fs=840)
    # fig, ax = plt.subplots(figsize=(8, 2))
    # AtlasDecode.plot_trace(zscore_smooth,ax, fs, label="zscore")
    i=i+1

#%%
Trace_raw,z_score,pixel_array_all_frames=AtlasDecode.get_zscore_from_atlas_snr_mask (dpath,hotpixel_path,xxrange,yyrange,fs=840,snr_thresh=2)