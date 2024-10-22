#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:00:05 2024

@author: kurtulus
"""

import numpy as np 
import matplotlib.pyplot as plt 



def get_snr_image(data):

    fr_rate = 840
    row_size = 128
    col_size = 128
    defect_pixel_mask = np.ones((row_size,col_size))
    mean_image = data[:, :,:].mean(axis=2)
    std_image = data[:, :,:].std(axis=2)
    snr_image = mean_image/std_image
    return mean_image, std_image, snr_image


def mask_low_snr_pixels(snr_image, thresh):
    
    mask = np.ones((128, 128))
    mask[np.where(snr_image<thresh)] = 0
    
    return mask

def mask_high_snr_pixels(snr_image, thresh):
    
    mask = np.ones((128, 128))
    mask[np.where(snr_image>thresh)] = 0
    
    return mask

def construct_roi_mask(xx_1 = 50, xx_2 = 150, yy_1 = 100 , yy_2 = 150):
    
    roi_mask = np.zeros((128,128))

    roi_mask[yy_1:yy_2, xx_1:xx_2] = 1
    
    background_mask = np.logical_not(roi_mask)
    
    return roi_mask, background_mask



def extract_trace(raw_data, roi_mask, hot_pixel_mask, activity = 'sum'):
    
    no_of_data_points = raw_data.shape[2]
    
    no_of_pixels_per_roi = roi_mask.sum()
    
    trace = []
    
    for i in range(no_of_data_points):
        
        frame = (raw_data[:,:,i]*hot_pixel_mask)*roi_mask
        trace.append(frame.sum())
        
    np_trace = np.asarray(trace)
        
    if activity == 'mean':
        np_trace = np_trace/no_of_pixels_per_roi

    return np_trace



file_path ='D:/ATLAS_SPAD/1825505_SimCre/Day2/Atlas/pixel_array_all_frames_1.npy'
   
data = np.load(file_path)


#get mean, std, and snr_image from the data 
mean_image, std_image, snr_image = get_snr_image(data)

#%%

thresh = 2 # look at the snr_image with colorbar and set this (pixel value below thresh will be 0 in the mask) 
pixel_mask = mask_low_snr_pixels(snr_image, thresh)

xxrange = [30, 80]
yyrange = [35, 85]
#construct roi_mask
roi_mask,_ = construct_roi_mask(xx_1 = 30, xx_2 = 80, yy_1 = 35 , yy_2 = 85)


#extract the trace based on the roi_mask and hot_pixel_mask
trace = extract_trace(data, roi_mask, pixel_mask, activity = 'mean')
#%%
#plot the data
layout = [['mean', 'std', 'snr']]

fig, ax = plt.subplot_mosaic(layout, figsize = (10, 3))

index = 'mean'
pos = ax[index].imshow(mean_image)
ax[index].set_title(index)
fig.colorbar(pos, ax=ax[index])


index = 'std'
pos = ax[index].imshow(snr_image)
ax[index].set_title(index)
fig.colorbar(pos, ax=ax[index])

index = 'snr'
pos = ax[index].imshow(snr_image)
ax[index].set_title(index)
fig.colorbar(pos, ax=ax[index])

plt.tight_layout()  # Adjust subplots to avoid overlap
plt.show()
#%%
layout = [['mean', 'std', 'snr'],
          ['trace', 'trace', 'trace']]

fig, ax = plt.subplot_mosaic(layout, figsize = (10, 10))

index = 'mean'
pos = ax[index].imshow(mean_image)
ax[index].set_title(index)
fig.colorbar(pos, ax=ax[index])


index = 'std'
pos = ax[index].imshow(std_image)
ax[index].set_title(index)
fig.colorbar(pos, ax=ax[index])

index = 'snr'
pos = ax[index].imshow(snr_image)
ax[index].set_title(index)
fig.colorbar(pos, ax=ax[index])


index = 'trace'
ax[index].plot(trace) # plotting only second second trace 
ax[index].set_title(index)

plt.tight_layout()  # Adjust subplots to avoid overlap
plt.show()





