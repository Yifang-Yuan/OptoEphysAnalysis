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
dpath='E:/ATLAS_SPAD/1825504_SimCre_GCamp8f_taper/Day2/Atlas/Burst-RS-25200frames-840Hz_2024-10-28_17-17/'
#dpath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/Burst-RS-1017frames-1017Hz_4uW/'
#hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'

pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=2000)
#%%
AtlasDecode.show_image_with_pixel_array(avg_pixel_array,showPixel_label=True)
#%%
center_x, center_y,radius=AtlasDecode.find_circle_mask(avg_pixel_array,radius=13)
#%%
#Trace_raw,dff=AtlasDecode.get_dff_from_atlas_snr_circle_mask (dpath,hotpixel_path,center_x, center_y,radius,fs=840,snr_thresh=2,photoncount_thre=5000)
#%%
'GET TOTAL PHOTON COUNT'
Trace_raw,dff=AtlasDecode.get_total_photonCount_atlas_continuous_circle_mask (dpath,hotpixel_path,center_x, center_y,radius,fs=840,photoncount_thre=2000)
#%%
day_parent_folder='F:/SPAD2024/SPADdata_SNRtest/5uWComparePSD/'
folder_name='5uW_Atlas'
save_folder = os.path.join(day_parent_folder, folder_name)
print ('save_folder is', save_folder)
# Create the folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), dff, delimiter=',', comments='')
np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_raw, delimiter=',', comments='')

#%%
data=Trace_raw
sampling_rate=840
#Plot the image of the pixel array
fig, ax = plt.subplots(figsize=(8, 2))
AtlasDecode.plot_trace(Trace_raw[16800:25200],ax, fs=840, label="raw_data")
#%%
'''Wavelet spectrum ananlysis'''
Analysis.plot_wavelet_data(data,sampling_rate,cutoff=300,xlim = ([6,30]))

#%%
'''Read binary files for single ROI'''
# Display the grayscale image
pixel_array_all_frames,sum_pixel_array,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=1000)
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
