# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:06:34 2025

@author: yifan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import AtlasDecode
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis

dpath=r'G:\2025_ATLAS_SPAD\1907336_Jedi2p_CB\Day2\test'
#dpath='F:/2025_ATLAS_SPAD/1887933_Jedi2P_Multi/Day1/Test/'
#hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'

pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder_smallFOV (
    dpath,hotpixel_path,photoncount_thre=50000)
#%%
AtlasDecode.show_image_with_pixel_array(avg_pixel_array,showPixel_label=True)
#%%
center_x, center_y,radius=AtlasDecode.find_circle_mask(avg_pixel_array,radius=10,threh=0.2)
#%%
'''
For Three ROI EXP
#Black
center_x, center_y,radius=47, 35,9
#Green
center_x, center_y,radius=48, 16,10
#Red
center_x, center_y,radius=64, 25,10
'''
#%%
Trace_raw=AtlasDecode.get_dff_from_pixel_array_smallFOV (pixel_array_all_frames,
                                                         avg_pixel_array,hotpixel_path,center_x, center_y,radius,
                                                         fs=1682.92,snr_thresh=4)
#%%
data=Trace_raw
sampling_rate=1682.92
#Plot the image of the pixel array
fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(Trace_raw[0:500],ax, fs=sampling_rate, label="raw_data")