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
        #single_pixel_array[hotpixel_indices[:, 0], hotpixel_indices[:, 1]] = 0 #REMOVE HOTPIXEL FROM MASK
        i=i+1
        if i>0:
            pixel_arrays.append(single_pixel_array)
    pixel_array_all_frames = np.stack(pixel_arrays, axis=2)
    sum_pixel_array = np.sum(pixel_array_all_frames, axis=2)
    avg_pixel_array =np.mean(pixel_array_all_frames, axis=2)
    
    return pixel_array_all_frames,sum_pixel_array,avg_pixel_array

def show_image_with_pixel_array(pixel_array_2d,showPixel_label=True):
    plt.imshow(pixel_array_2d, cmap='gray')
    # Add color bar with photon count numbers
    cbar = plt.colorbar()
    cbar.set_label('Photon Count')
    if showPixel_label:
        # Add x and y axes with pixel IDs
        plt.xticks(np.arange(0, 128, 10), labels=np.arange(0, 128, 10))
        plt.yticks(np.arange(0, 128, 10), labels=np.arange(0, 128, 10))
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

def get_trace_from_3d_pixel_array(pixel_array_all_frames,sum_pixel_array,xxrange,yyrange):
    import matplotlib.patches as patches
    plt.figure(figsize=(6, 6))
    plt.imshow(sum_pixel_array, cmap='gray')
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
    #ax.plot(taxis,trace,linewidth=1,label=label)
    ax.plot(taxis,trace,linewidth=1)
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

def get_zscore_from_atlas_continuous (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840):
    pixel_array_all_frames,sum_pixel_array,_=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=1000)
    _,mean_values_over_time,_=get_trace_from_3d_pixel_array(pixel_array_all_frames,sum_pixel_array,xxrange,yyrange)
    print('original lenth: ', len(mean_values_over_time))
    Trace_raw=mean_values_over_time[1:]
    print('trace_raw lenth 1: ', len(Trace_raw))
    Trace_raw = np.append(Trace_raw, Trace_raw[-1])
    print('trace_raw lenth 2: ', len(Trace_raw))
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(Trace_raw,ax, fs, label="raw_data")
    
    lambd = 10e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    sig_base=fp.airPLS(Trace_raw,lambda_=lambd,porder=porder,itermax=itermax) 
    signal = (Trace_raw - sig_base)  
    z_score=(signal - np.median(signal)) / np.std(signal)
    
    fig, ax = plt.subplots(figsize=(8, 2))
    plot_trace(z_score,ax, fs, label="zscore")
    return Trace_raw,z_score

#%% Workable code, above is testing
# dpath='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240429_Day1/Atlas/Burst-RS-25200frames-840Hz_2024-04-29_12-57_1/'
# #dpath='F:/2024MScR_NORtask/1732333_pyramidal_G8f_Atlas/20240420_Day1/Atlas/Burst-RS-25200frames-840Hz_2024-04-20_11-53_2/'
# hotpixel_path='F:/SPADdata/Altas_hotpixel.csv'
# xxrange = [25, 85]
# yyrange = [30, 90]

# Trace_raw,z_score=get_zscore_from_atlas_continuous (dpath,hotpixel_path,xxrange= [25, 85],yyrange= [30, 90],fs=840)
# # Plot the image of the pixel array
# #%%
# fig, ax = plt.subplots(figsize=(8, 2))
# plot_trace(Trace_raw[800:1600],ax, fs=840, label="raw_data")
# #%%
# '''Read binary files for single ROI'''
# # Display the grayscale image
#%%
#pixel_array_all_frames,sum_pixel_array,avg_pixel_array=decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=1000)
# show_image_with_pixel_array(pixel_array_all_frames[:,:,800])
# pixel_array=pixel_array_all_frames[:,:,800]
# pixel_array_plot_hist(pixel_array_all_frames[:,:,1000], plot_min_thre=100)