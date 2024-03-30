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

def decode_atlas_folder (folderpath,photoncount_thre=2000):
    files = os.listdir(folderpath)
    # Filter out the CSV files that match the pattern 'AnimalTracking_*.csv'
    frame_files = [file for file in files if file.startswith('frame_') and file.endswith('.mat')]
    # Sort the CSV files based on the last two digits in their filenames
    sorted_mat_files = sorted(frame_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    pixel_arrays = []
    i=0
    for file in sorted_mat_files:
        #print ('framefile name', file)
        single_frame_data_path = os.path.join(folderpath, file)
        matdata = loadmat(single_frame_data_path)
        real_data=matdata['realData']
        readData=loadPCFrame(real_data) #decode data to single pixel frame
        readData=remove_hotpixel(readData,photoncount_thre)
        single_pixel_array=readData[:,:,0]
        if i>0:
            pixel_arrays.append(single_pixel_array)
        i=i+1
    pixel_array_all_frames = np.stack(pixel_arrays, axis=2)
    sum_pixel_array = np.sum(pixel_array_all_frames, axis=2)
    avg_pixel_array =np.mean(pixel_array_all_frames, axis=2)
    
    return pixel_array_all_frames,sum_pixel_array,avg_pixel_array

def show_image_with_pixel_array(pixel_array_2d,showPixel_label=True):
    plt.imshow(avg_pixel_array, cmap='gray')
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

def get_trace_from_3d_pixel_array(pixel_array_all_frames,xxrange,yyrange):
    import matplotlib.patches as patches
    plt.figure(figsize=(8, 8))
    plt.imshow(sum_pixel_array, cmap='gray')
    plt.colorbar(label='Photon count')
    plt.title('Image with Selected Region')
    plt.xlabel('Y coordinate')
    plt.ylabel('X coordinate')
    # Add a rectangle around the selected region
    rect = patches.Rectangle((yyrange[0], xxrange[0]), yyrange[1]-yyrange[0], xxrange[1]-xxrange[0], 
                             linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()
    region_pixel_array = pixel_array_all_frames[xxrange[0]:xxrange[1]+1, yyrange[0]:yyrange[1]+1, :]
    # Sum along the x and y axes to get the sum of photon counts within the specified range for each frame
    sum_values_over_time = np.sum(region_pixel_array, axis=(0, 1))
    return sum_values_over_time

def plot_trace(trace,ax, fs=1/1017, label="trace"):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    ax.plot(taxis,trace,linewidth=1,label=label)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax
#%%
dpath='C:/SPAD/18032024/Burst-RS-1017frames-1017Hz_16uW/'
pixel_array_all_frames,sum_pixel_array,avg_pixel_array=decode_atlas_folder (dpath,photoncount_thre=1200)
xxrange = [50, 80]
yyrange = [40, 70]
# Slice the pixel array to extract the desired region
sum_values_over_time=get_trace_from_3d_pixel_array(pixel_array_all_frames,xxrange,yyrange)
np.savetxt('C:/SPAD/18032024/ROI_trace_16uW.csv', sum_values_over_time, delimiter=',', header='data', comments='')
# Plot the image of the pixel array
fig, ax = plt.subplots(figsize=(12, 2.5))
plot_trace(sum_values_over_time,ax, fs=1017, label="trace")
#%%
# Display the grayscale image
show_image_with_pixel_array(pixel_array_all_frames[:,:,800])
pixel_array=pixel_array_all_frames[:,:,800]
#%%
pixel_array_plot_hist(pixel_array_all_frames[:,:,1000], plot_min_thre=100)