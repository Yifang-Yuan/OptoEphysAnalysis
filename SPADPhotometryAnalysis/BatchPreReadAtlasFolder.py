# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:53:34 2024

@author: Yifang
"""

from SPADPhotometryAnalysis import AtlasDecode
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

def replace_outliers_with_avg(data, threshold):
    # Identify the outliers
    outliers = np.abs(data) > threshold

    # Get the indices of the outliers
    outlier_indices = np.where(outliers)[0]

    # Iterate over the outlier indices
    for index in outlier_indices:
        # Find the nearest non-outlier value
        left = index
        while left > 0 and outliers[left - 1]:
            left -= 1

        right = index
        while right < len(data) - 1 and outliers[right + 1]:
            right += 1

        # If the outlier is not at the boundaries of the array, replace it with the average of its neighbors
        if left != 0 and right != len(data) - 1:
            data[index] = (data[left - 1] + data[right + 1]) / 2
        # If the outlier is at the boundaries of the array, replace it with the nearest non-outlier value
        else:
            data[index] = data[left - 1] if left != 0 else data[right + 1]

    return data

def plot_trace(trace,ax, fs=9938.4, label="trace",color='tab:blue'):
    t=(len(trace)) / fs
    taxis = np.arange(len(trace)) / fs
    ax.plot(taxis,trace,linewidth=0.5,label=label,color=color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0,t)
    ax.legend(loc="upper right", frameon=False)
    #ax.spines['left'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.xaxis.set_visible(False)  # Hide x-axis
    #ax.yaxis.set_visible(False)  # Hide x-axis
    ax.set_xlabel('Time(second)')
    ax.set_ylabel('Photon Count')
    return ax

def Remove_outliers_for_session (parent_folder,TargetfolderName='SyncRecording'):
    
    # List all files and directories in the parent folder
    all_contents = os.listdir(parent_folder)
    # Filter out directories containing the target string
    sync_recording_folders = [folder for folder in all_contents if TargetfolderName in folder]
    # Define a custom sorting key function to sort folders in numeric order
    def numeric_sort_key(folder_name):
        return int(folder_name.lstrip(TargetfolderName))
    # Sort the folders in numeric order
    sync_recording_folders.sort(key=numeric_sort_key)
    # Iterate over each sync recording folder
    for SyncRecordingName in sync_recording_folders:
        # Now you can perform operations on each folder, such as reading files inside it
        print("----Now processing folder:", SyncRecordingName)
        filename=os.path.join(parent_folder, SyncRecordingName,'Zscore_traceAll.csv')
        trace = np.genfromtxt(filename, delimiter=',')
        np.savetxt(filename, trace, delimiter=',', comments='')
        fig, ax = plt.subplots(figsize=(8,2))
        plot_trace(trace,ax, fs=840,label='z_score')
        #trace=replace_outliers_with_nearest_avg(trace, window_size=25000, z_thresh=4)
        trace=replace_outliers_with_avg(trace, threshold=4)
        fig, ax = plt.subplots(figsize=(8,2))
        plot_trace(trace,ax, fs=840,label='cleaned_z_score')
        np.savetxt(filename, trace, delimiter=',', comments='')

    return -1

def extract_timestamp(folder_name):
    # Assuming the timestamp is at the end in the format "YYYY-MM-DD_HH-MM"
    timestamp_str = folder_name.split('_')[-2]+ '_' +folder_name.split('_')[-1] 
    # Convert the string to a datetime object
    return datetime.strptime(timestamp_str,  '%Y-%m-%d_%H-%M')

def read_multiple_Atlas_bin_folder(atlas_parent_folder,day_parent_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=2000):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    # Get a list of all directories in the parent folder
    all_atlas_folders = os.listdir(atlas_parent_folder)
    # Sort by folder name
    pattern = r'^Burst-RS-\d+frames-\d+Hz_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$'
    folders = [
        item for item in all_atlas_folders
        if os.path.isdir(os.path.join(atlas_parent_folder, item)) and re.match(pattern, item)
    ]
    # Sort folders by the extracted timestamp
    sorted_atlas_folders = sorted(folders, key=extract_timestamp)
    print (sorted_atlas_folders)
    # Read the directories in sorted order
    i=0
    for foldername in sorted_atlas_folders:
        i=i+1
        directory=os.path.join(atlas_parent_folder, foldername)
        print("Folder:", directory)
        Trace_raw,dff=AtlasDecode.get_dff_from_atlas_snr_circle_mask (directory,hotpixel_path,center_x, center_y,radius,
                                                                        fs=840,snr_thresh=2,photoncount_thre=photoncount_thre)
        
        # Trace_raw,dff= AtlasDecode.get_total_photonCount_atlas_continuous_circle_mask (directory,hotpixel_path,center_x, center_y,radius,fs=840,photoncount_thre=photoncount_thre)
        folder_name = f'{new_folder_name}{i}'
        save_folder = os.path.join(day_parent_folder, folder_name)
        print ('save_folder is', save_folder)
        #z_score=replace_outliers_with_avg(z_score, threshold=4)   
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), dff, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_raw, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Red_traceAll.csv'), Trace_raw, delimiter=',', comments='')
        #np.save(os.path.join(save_folder, 'pixel_array_all_frames.npy'), pixel_array_all_frames)
    return -1


def main():
    '''Set the folder to the one-day recording folder that contain SPAD data for each trial'''
    '''IMPORTANT: Set the ROI range, this can be found by the screenshot you made during recording,
    or draw image for a single trial using DemoS ingleSPAD_folder.py'''
    
    'Reading SPAD binary data'
    #hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
    hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
    
    center_x, center_y,radius=49, 57, 17
    day_folder='F:/2025_ATLAS_SPAD/Batch1/1842515_PV_mNeon/Day10/'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=400)
    
    center_x, center_y,radius=45, 57, 17
    day_folder='F:/2025_ATLAS_SPAD/Batch1/1842515_PV_mNeon/Day11/'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=400)
    
    center_x, center_y,radius=49, 57, 17
    day_folder='F:/2025_ATLAS_SPAD/1844607_PV_mNeon_F/Day6/'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=400)
    
    center_x, center_y,radius=46, 57, 17
    day_folder='F:/2025_ATLAS_SPAD/1844607_PV_mNeon_F/Day7/'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=400)
    
    center_x, center_y,radius=49, 57, 17
    day_folder='F:/2025_ATLAS_SPAD/1842516_PV_Jedi2p/Day5/'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=400)
    
    center_x, center_y,radius=46, 58, 17
    day_folder='F:/2025_ATLAS_SPAD/1842516_PV_Jedi2p/Day6/'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=400)
    
  

if __name__ == "__main__":
    main()