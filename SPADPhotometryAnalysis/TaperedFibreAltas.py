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
dpath='E:/ATLAS_SPAD/1825504_Sim1Cre_GCamp8f_taper/Day2/Test/'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
photoncount_thre=800
fs=840

pixel_array_all_frames,sum_pixel_array,_=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
AtlasDecode.show_image_with_pixel_array(sum_pixel_array,showPixel_label=True)
#%%
#sum_pixel_array=snr_image
shape = sum_pixel_array.shape
max_avg_photon_count = 0
best_center = (0, 0)
# Define ranges for center positions and radii
center_y_range = range(10, shape[0] - 10)  # Avoid edges for centers
center_x_range = range(10, shape[1] - 10)
radii = range(5, min(shape) // 2)
radius=12
# Iterate over possible centers
for center_y in center_y_range:
    for center_x in center_x_range:
        # Create a circular mask for the current center and radius
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask_area = np.sum(mask)  # Area of the circle in pixels
    
        # Calculate average photon count within the mask
        if mask_area > 0:
            total_photon_count = np.sum(sum_pixel_array[mask])
            average_photon_count = total_photon_count / mask_area
    
            # Update if we find a new maximum average photon count
            if average_photon_count > max_avg_photon_count:
                max_avg_photon_count = average_photon_count
                best_center = (center_x, center_y)

# Plot the image with the best circle overlay
plt.figure(figsize=(6, 6))
plt.imshow(sum_pixel_array, cmap='hot')
plt.colorbar(label='Photon Count')

# Draw the best detected circle
circle = plt.Circle(best_center, radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
plt.gca().add_patch(circle)
plt.title('Photon Count Image with Best Circle Overlay')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.show()

print("Best center:", best_center)
print("Radius:", radius)
print("Max average photon count within circle:", max_avg_photon_count)
#%%

best_center,radius=find_circle_mask(sum_pixel_array)
#%%
pixel_array_all_frames,sum_pixel_array=AtlasDecode.decode_atlas_folder_without_hotpixel_removal (dpath)
        
snr_thresh=2
mean_image, std_image, snr_image = AtlasDecode.get_snr_image(pixel_array_all_frames)

layout = [['mean', 'std', 'snr']]

fig, ax = plt.subplot_mosaic(layout, figsize = (10, 3))

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

plt.tight_layout()  # Adjust subplots to avoid overlap
plt.show()


# look at the snr_image with colorbar and set this (pixel value below thresh will be 0 in the mask) 
pixel_mask = AtlasDecode.mask_low_snr_pixels(snr_image, snr_thresh)
#%%
xxrange1 = [28, 71]
yyrange1 = [40, 82]

xxrange2 = [44, 54]
yyrange2 = [57, 67]

xxrange3 = [54, 64]
yyrange3 = [52, 72]

xxrange4 = [34, 44]
yyrange4 = [52, 72]

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
    Trace_raw=Trace_raw
    
    AtlasDecode.plot_trace(Trace_raw,ax, fs, label="raw_data")
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    z_score=(signal - np.median(signal)) / np.std(signal)
    
    zscore_smooth=Analysis.get_bin_trace (z_score,bin_window=20,color='tab:blue',Fs=840)
    # fig, ax = plt.subplots(figsize=(8, 2))
    # AtlasDecode.plot_trace(zscore_smooth,ax, fs, label="zscore")
    i=i+1

#%%
Trace_raw,z_score,pixel_array_all_frames=AtlasDecode.get_zscore_from_atlas_snr_mask (dpath,hotpixel_path,xxrange,yyrange,fs=840,snr_thresh=2)