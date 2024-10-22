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
    avg_pixel_array =np.mean(pixel_array_all_frames, axis=2)
    
    return pixel_array_all_frames

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
    mean_array=np.mean(region_pixel_array, axis=2)
    std_array=np.std(region_pixel_array, axis=2)
    #Check hotpixels
    # Sum along the x and y axes to get the sum of photon counts within the specified range for each frame
    sum_values_over_time = np.sum(region_pixel_array, axis=(0, 1))
    mean_values_over_time = np.mean(region_pixel_array, axis=(0, 1))
    return sum_values_over_time,mean_values_over_time,region_pixel_array

def plot_trace(trace,ax, fs=1017, label="trace"):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    mean_trace = np.mean(trace)
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

def get_zscore_from_atlas_continuous (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840,photoncount_thre=2000):
    pixel_array_all_frames,sum_pixel_array,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
    _,mean_values_over_time,_=get_trace_from_3d_pixel_array(pixel_array_all_frames,sum_pixel_array,xxrange,yyrange)
    #print('original lenth: ', len(mean_values_over_time))
    Trace_raw=mean_values_over_time[1:]
    #print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])
    #print('trace_raw lenth 2: ', len(Trace_raw))
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")
    
    #remove outlier
    #Trace_raw=replace_outliers_with_nearest_avg(Trace_raw, window_size=25000, z_thresh=4)
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    z_score=(signal - np.median(signal)) / np.std(signal)
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(z_score,ax, fs, label="zscore")
    return Trace_raw,z_score

def get_zscore_from_atlas_snr_mask (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840,snr_thresh=2):
    pixel_array_all_frames=decode_atlas_folder_without_hotpixel_removal (dpath)
        
    mean_image, std_image, snr_image = get_snr_image(pixel_array_all_frames)
    # look at the snr_image with colorbar and set this (pixel value below thresh will be 0 in the mask) 
    pixel_mask = mask_low_snr_pixels(snr_image, snr_thresh)

    xxrange = [30, 80]
    yyrange = [35, 85]
    #construct roi_mask
    roi_mask,_ = construct_roi_mask(xx_1 = xxrange[0], xx_2 = xxrange[1], yy_1 = yyrange[0] , yy_2 = yyrange[1])
    
    #extract the trace based on the roi_mask and hot_pixel_mask
    trace = extract_trace(pixel_array_all_frames, roi_mask, pixel_mask, activity = 'mean')
    #print('original lenth: ', len(mean_values_over_time))
    Trace_raw=trace[1:]
    #print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])

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

    #print('trace_raw lenth 2: ', len(Trace_raw))
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")

    #Trace_raw=replace_outliers_with_nearest_avg(Trace_raw, window_size=25000, z_thresh=4)
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    z_score=(signal - np.median(signal)) / np.std(signal)
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(z_score,ax, fs, label="zscore")
    plt.show()
    return Trace_raw,z_score,pixel_array_all_frames

def get_total_photonCount_atlas_continuous (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840,photoncount_thre=2000):
    pixel_array_all_frames,sum_pixel_array,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=photoncount_thre)
    sum_values_over_time,mean_values_over_time,region_pixel_array=get_trace_from_3d_pixel_array(pixel_array_all_frames,sum_pixel_array,xxrange,yyrange)
    #print('original lenth: ', len(mean_values_over_time))
    Trace_raw=sum_values_over_time[1:]
    #print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])
    #print('trace_raw lenth 2: ', len(Trace_raw))
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")

    return Trace_raw