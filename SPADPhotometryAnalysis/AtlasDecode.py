# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:18:25 2024

@author: Yifang
"""
import os
from scipy.io import loadmat
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import photometry_functions as fp
from scipy.ndimage import uniform_filter1d
import matplotlib.patches as patches

def loadPCFrame (readData):
    '''loads a sensor bytestream file containing a single Photon Count frame
        into a bytearray that can be used to perform operations on in python
    '''
    orig = np.repeat((np.repeat(np.arange(8).reshape((1, 8)), 128, 0)), 16, 0).reshape((128, 128))
    add = np.repeat(np.array(np.repeat(np.arange(16).reshape(1, 16), 128, 0)), 8, 1) * 1024
    add2 = np.repeat(np.array(np.arange(128).reshape(128, 1)), 128, 1) * 8
    lut = orig + add + add2
    insertionIndices = np.repeat(np.array(np.arange(1024)), 20, 0)*44

    readData = np.insert(readData, insertionIndices, 0, 0)    #insert pads to make stream 32 bit ints BEFORE unpacking and reordering
    #print("padded array shape: {}".format(readData.shape))      #16384 pixels * 4 bytes = 65536 bytes
    readData = readData.reshape((-1, 2))            #reshape to match output of ATLAS
    readData = np.roll(readData, 1, 1)              #get top and bottom half of image correct (swap top and bottom half of image)
    readData = np.unpackbits(readData, 1)
    readData = np.split(readData, 1024, 0)
    readData = np.packbits(readData, 1, 'big')
    readData = np.flip(readData, 1)
    readData = np.swapaxes(readData, 1, 2)
    readData = np.ascontiguousarray(readData)
    readData = readData.view(np.int32)
    readData = np.swapaxes(readData, 0, 1)
    readData = readData.reshape((-1, 1))            #reshape to match output of ATLAS
    readData = readData[lut]
    return readData
    
def remove_hotpixel(readData,photoncount_thre=2000):
    readData[:,:,0][readData[:,:,0] > photoncount_thre] = 0
    return readData

def find_hotpixel_idx(hotpixel_path,array_data,photoncount_thre=2000):
    index_array = np.argwhere(array_data > photoncount_thre)
    np.savetxt(hotpixel_path, index_array, fmt='%d', delimiter=',')
    return index_array

def decode_atlas_folder (folderpath,hotpixel_path,photoncount_thre=2000):
    files = os.listdir(folderpath)
    # Filter out the CSV files that match the pattern 'AnimalTracking_*.csv'
    frame_files = [file for file in files if file.startswith('frame_') and file.endswith('.mat')]
    # Sort the CSV files based on the numerical digits in their filenames
    sorted_mat_files = sorted(frame_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    #print (sorted_mat_files)
    pixel_arrays = []
    hotpixel_indices= np.loadtxt(hotpixel_path, delimiter=',', dtype=int)
    i=0
    for file in sorted_mat_files:
        #print ('framefile name', file)
        single_frame_data_path = os.path.join(folderpath, file)
        matdata = loadmat(single_frame_data_path)
        real_data=matdata['realData']
        readData=loadPCFrame(real_data) #decode data to single pixel frame
        readData=remove_hotpixel(readData,photoncount_thre) #REMOVE hotpixel by a threshold
        single_pixel_array=readData[:,:,0]
        single_pixel_array[hotpixel_indices[:, 0], hotpixel_indices[:, 1]] = 0 #REMOVE HOTPIXEL FROM MASK
        i=i+1
        if i>0:
            pixel_arrays.append(single_pixel_array)
    pixel_array_all_frames = np.stack(pixel_arrays, axis=2)
    sum_pixel_array = np.sum(pixel_array_all_frames, axis=2)
    avg_pixel_array =np.mean(pixel_array_all_frames, axis=2)
    
    return pixel_array_all_frames,sum_pixel_array,avg_pixel_array

def decode_atlas_folder_without_hotpixel_removal (folderpath):
    files = os.listdir(folderpath)
    # Filter out the CSV files that match the pattern 'AnimalTracking_*.csv'
    frame_files = [file for file in files if file.startswith('frame_') and file.endswith('.mat')]
    # Sort the CSV files based on the numerical digits in their filenames
    sorted_mat_files = sorted(frame_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    #print (sorted_mat_files)
    pixel_arrays = []
    #hotpixel_indices= np.loadtxt(hotpixel_path, delimiter=',', dtype=int)
    i=0
    for file in sorted_mat_files:
        #print ('framefile name', file)
        single_frame_data_path = os.path.join(folderpath, file)
        matdata = loadmat(single_frame_data_path)
        real_data=matdata['realData']
        readData=loadPCFrame(real_data) #decode data to single pixel frame
        #readData=remove_hotpixel(readData,photoncount_thre) #REMOVE hotpixel by a threshold
        #readData=remove_hotpixel(readData,photoncount_thre) #REMOVE hotpixel by a threshold
        single_pixel_array=readData[:,:,0]
        #single_pixel_array[hotpixel_indices[:, 0], hotpixel_indices[:, 1]] = 0 #REMOVE HOTPIXEL FROM MASK
        i=i+1
        if i>0:
            pixel_arrays.append(single_pixel_array)
    pixel_array_all_frames = np.stack(pixel_arrays, axis=2)
    sum_pixel_array = np.sum(pixel_array_all_frames, axis=2)
    
    return pixel_array_all_frames,sum_pixel_array

def show_image_with_pixel_array(pixel_array_2d,showPixel_label=True):
    # vmin = 0  # Minimum value
    # vmax = 80  # Maximum value

    #plt.imshow(pixel_array_2d, cmap='gray', vmin=vmin, vmax=vmax)
    plt.imshow(pixel_array_2d, cmap='gray')
    # Add color bar with photon count numbers
    if showPixel_label:
        # Add x and y axes with pixel IDs
        plt.xticks(np.arange(0, 128, 10), labels=np.arange(0, 128, 10))
        plt.yticks(np.arange(0, 128, 10), labels=np.arange(0, 128, 10))
        cbar = plt.colorbar()
        cbar.set_label('Photon Count')
    else:
        plt.axis('off')  # Turn off axis
    plt.show()
    return -1

def find_circle_mask(pixel_array,radius=12):
    shape = pixel_array.shape
    max_avg_photon_count = 0
    best_center = (0, 0)
    # Define ranges for center positions and radii
    center_y_range = range(10, shape[0] - 10)  # Avoid edges for centers
    center_x_range = range(10, shape[1] - 10)
    # Iterate over possible centers
    for center_y in center_y_range:
        for center_x in center_x_range:
            # Create a circular mask for the current center and radius
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            mask_area = np.sum(mask)  # Area of the circle in pixels
            # Calculate average photon count within the mask
            if mask_area > 0:
                total_photon_count = np.sum(pixel_array[mask])
                average_photon_count = total_photon_count / mask_area
        
                # Update if we find a new maximum average photon count
                if average_photon_count > max_avg_photon_count:
                    max_avg_photon_count = average_photon_count
                    best_center = (center_x, center_y)
    while True: 
        # Plot the image with the best circle overlay
        plt.figure(figsize=(4, 4))
        plt.imshow(pixel_array, cmap='hot')
        plt.colorbar(label='Photon Count')
        # Draw the best detected circle
        circle = plt.Circle(best_center, radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
        plt.gca().add_patch(circle)
        plt.title('Find ROI')
        plt.xlabel('X',fontsize=12)
        plt.ylabel('Y',fontsize=12)
        plt.legend(loc='upper right')
        plt.show()
        print("Radius:", radius)
        new_radius = input("Enter new radius (or 'q' to quit): ")
        if new_radius.lower() == 'q':
            break
        else:
            radius=int(new_radius)
            
    print("Best center:", best_center)
    print("Radius:", radius)
    print("Max average photon count within circle:", max_avg_photon_count)
    return best_center[0], best_center[1],radius

def pixel_array_plot_hist(pixel_array, plot_min_thre=100):
    photon_counts = pixel_array.flatten()
    # Plot the histogram
    plt.hist(photon_counts, bins=50, range=(plot_min_thre, photon_counts.max()), color='blue', alpha=0.7)
    #plt.hist(photon_counts, bins=50, range=(0, 500), color='blue', alpha=0.7)
    plt.xlabel('Photon Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Photon Counts')
    plt.grid(True)
    plt.show()
    return -1

def get_trace_from_3d_pixel_array(pixel_array_all_frames,pixel_array,xxrange,yyrange):
    plt.figure(figsize=(6, 6))
    plt.imshow(pixel_array, cmap='gray')
    plt.colorbar(label='Photon count')
    plt.title('Image with Selected Region')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    # Add a rectangle around the selected region
    rect = patches.Rectangle((xxrange[0], yyrange[0]), xxrange[1]-xxrange[0], yyrange[1]-yyrange[0], 
                             linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()
    
    region_pixel_array = pixel_array_all_frames[xxrange[0]:xxrange[1]+1, yyrange[0]:yyrange[1]+1, :]
    #Check hotpixels
    # Sum along the x and y axes to get the sum of photon counts within the specified range for each frame
    sum_values_over_time = np.sum(region_pixel_array, axis=(0, 1))
    mean_values_over_time = np.mean(region_pixel_array, axis=(0, 1))
    return sum_values_over_time,mean_values_over_time,region_pixel_array

def get_trace_from_3d_pixel_array_circle_mask(pixel_array_all_frames,pixel_array,center_x, center_y,radius):
    shape = pixel_array.shape
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    plt.figure(figsize=(6, 6))
    plt.imshow(pixel_array, cmap='gray')
    plt.colorbar(label='Photon count')
    # Add a rectangle around the selected region
    circle = plt.Circle((center_x,center_y), radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
    plt.gca().add_patch(circle)
    plt.title('Image with Selected Region')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend(loc='upper right')
    plt.show()
    
    masked_values = pixel_array_all_frames[mask, :]

    # Calculate the sum and mean for each frame along the length axis (frame axis)
    sum_values_over_time = np.sum(masked_values, axis=0)  # Sum photon counts within mask for each frame
    mean_values_over_time = np.mean(masked_values, axis=0)  # Mean photon count within mask for each frame
    return sum_values_over_time,mean_values_over_time


def plot_trace(trace,ax, fs=1017, label="trace"):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    #mean_trace = np.mean(trace)
    ax.plot(taxis,trace,linewidth=1,label=label)
    #ax.plot(taxis,trace,linewidth=1)
    #ax.axhline(mean_trace, color='r', linestyle='--', label='Mean Value', linewidth=1.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    #ax.xaxis.set_visible(False)  # Hide x-axis
    #ax.yaxis.set_visible(False)  # Hide x-axis
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax

def replace_outliers_with_nearest_avg(data, window_size=25000, z_thresh=3):
    # Calculate the mean and standard deviation of the moving window
    mean = uniform_filter1d(data, window_size, mode='reflect')
    std = uniform_filter1d(data**2, window_size, mode='reflect')
    std = np.sqrt(std - mean**2)

    # Identify the outliers
    outliers = (np.abs(data - mean) > z_thresh * std)

    # Replace outliers with the average of their nearest non-outlier neighbors
    for i in np.where(outliers)[0]:
        j = i - 1
        while j >= 0 and outliers[j]:
            j -= 1
        k = i + 1
        while k < len(data) and outliers[k]:
            k += 1
        if j >= 0 and k < len(data):
            data[i] = (data[j] + data[k]) / 2
        elif j >= 0:
            data[i] = data[j]
        elif k < len(data):
            data[i] = data[k]

    return data
def get_snr_image(data):
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
    print ('no_of_pixels_per_roi---', no_of_pixels_per_roi)
    
    trace = []
    for i in range(no_of_data_points):
        
        frame = (raw_data[:,:,i]*hot_pixel_mask)*roi_mask
        trace.append(frame.sum())
    np_trace = np.asarray(trace)
    if activity == 'mean':
        np_trace = np_trace/no_of_pixels_per_roi
    return np_trace

def get_dff_from_atlas_continuous_circle_mask (dpath,hotpixel_path,center_x, center_y,radius,fs=840,photoncount_thre=2000):
    pixel_array_all_frames,sum_pixel_array,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
    _,mean_values_over_time=get_trace_from_3d_pixel_array_circle_mask(pixel_array_all_frames,sum_pixel_array,center_x, center_y,radius)
    #print('original lenth: ', len(mean_values_over_time))
    Trace_raw=mean_values_over_time[1:]
    #print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])
    #print('trace_raw lenth 2: ', len(Trace_raw))
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    dff=100*signal / sig_base
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(dff,ax, fs, label="df/f")
    return Trace_raw,dff

def get_dff_from_atlas_snr_circle_mask (dpath,hotpixel_path,center_x, center_y,radius,fs=840,snr_thresh=2,photoncount_thre=2000):
    pixel_array_all_frames,_,avg_pixel_array=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
        
    mean_image, std_image, snr_image = get_snr_image(pixel_array_all_frames)
    pixel_mask = mask_low_snr_pixels(snr_image, snr_thresh)

    shape = pixel_array_all_frames.shape[0:2] 
    y, x = np.ogrid[:shape[0], :shape[1]]
    roi_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = ax.imshow(snr_image, cmap='viridis')  # Adjust colormap if desired
    ax.set_title('SNR')
    fig.colorbar(pos, ax=ax)
    circle = plt.Circle((center_x,center_y), radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
    plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = ax.imshow(avg_pixel_array, cmap='viridis')  # Adjust colormap if desired
    ax.set_title('Averaged')
    fig.colorbar(pos, ax=ax)
    circle = plt.Circle((center_x,center_y), radius, color='cyan', fill=False, linewidth=2, label='Best Circle')
    plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.show()
    #extract the trace based on the roi_mask and hot_pixel_mask
    trace = extract_trace(pixel_array_all_frames, roi_mask, pixel_mask, activity = 'sum')
    #print('original lenth: ', len(mean_values_over_time))
    Trace_raw=trace[1:]
    #print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])

    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    dff=100*signal / sig_base
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(dff,ax, fs, label="df/f")
    plt.show()
    return Trace_raw,dff

def get_total_photonCount_atlas_continuous_circle_mask (dpath,hotpixel_path,center_x, center_y,radius,fs=840,photoncount_thre=2000):
    pixel_array_all_frames,sum_pixel_array,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
    sum_values_over_time,_=get_trace_from_3d_pixel_array_circle_mask(pixel_array_all_frames,sum_pixel_array,center_x, center_y,radius)
    #print('original lenth: ', len(mean_values_over_time))
    Trace_raw=sum_values_over_time[1:]
    #print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])
    #print('trace_raw lenth 2: ', len(Trace_raw))
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    dff=100*signal / sig_base
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(dff,ax, fs, label="df/f")
    return Trace_raw,dff
# def get_zscore_from_atlas_continuous (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840,photoncount_thre=2000):
#     pixel_array_all_frames,sum_pixel_array,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
#     _,mean_values_over_time,_=get_trace_from_3d_pixel_array(pixel_array_all_frames,sum_pixel_array,xxrange,yyrange)
#     #print('original lenth: ', len(mean_values_over_time))
#     Trace_raw=mean_values_over_time[1:]
#     #print('trace_raw lenth 1: ', len(Trace_raw))
#     Trace_raw = np.append(Trace_raw, Trace_raw[-1])
#     #print('trace_raw lenth 2: ', len(Trace_raw))
#     fig, ax = plt.subplots(figsize=(8, 2))
#     plot_trace(Trace_raw,ax, fs, label="raw_data")
    
#     lambd = 10e3 # Adjust lambda to get the best fit
#     porder = 1
#     itermax = 15
#     sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
#     signal = (Trace_raw - sig_base)  
#     dff=100*signal / sig_base
#     fig, ax = plt.subplots(figsize=(8, 2))
#     plot_trace(dff,ax, fs, label="df/f")
#     return Trace_raw,dff

# def get_zscore_from_atlas_snr_mask (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840,snr_thresh=2,photoncount_thre=2000):
#     pixel_array_all_frames,_,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
        
#     mean_image, std_image, snr_image = get_snr_image(pixel_array_all_frames)
#     fig, ax = plt.subplots(figsize=(5, 5))
#     # Plot only the SNR image
#     pos = ax.imshow(snr_image, cmap='viridis')  # Adjust colormap if desired
#     ax.set_title('SNR')
#     fig.colorbar(pos, ax=ax)
#     rect = patches.Rectangle((xxrange[0], yyrange[0]), xxrange[1]-xxrange[0], yyrange[1]-yyrange[0], 
#                              linewidth=2, edgecolor='r', facecolor='none')
#     plt.gca().add_patch(rect)
#     plt.show()
#     plt.tight_layout()
#     plt.show()
#     # look at the snr_image with colorbar and set this (pixel value below thresh will be 0 in the mask) 
#     pixel_mask = mask_low_snr_pixels(snr_image, snr_thresh)
#     #construct roi_mask
#     roi_mask,_ = construct_roi_mask(xx_1 = xxrange[0], xx_2 = xxrange[1], yy_1 = yyrange[0] , yy_2 = yyrange[1])
    
#     #extract the trace based on the roi_mask and hot_pixel_mask
#     trace = extract_trace(pixel_array_all_frames, roi_mask, pixel_mask, activity = 'mean')
#     #print('original lenth: ', len(mean_values_over_time))
#     Trace_raw=trace[1:]
#     #print('trace_raw lenth 1: ', len(Trace_raw))
#     Trace_raw = np.append(Trace_raw, Trace_raw[-1])

#     fig, ax = plt.subplots(figsize=(8, 2))
#     plot_trace(Trace_raw,ax, fs, label="raw_data")
#     lambd = 10e3 # Adjust lambda to get the best fit
#     porder = 1
#     itermax = 15
#     sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
#     signal = (Trace_raw - sig_base)  
#     dff=100*signal / sig_base
    
#     fig, ax = plt.subplots(figsize=(8, 2))
#     plot_trace(dff,ax, fs, label="df/f")
#     plt.show()
#     return Trace_raw,dff