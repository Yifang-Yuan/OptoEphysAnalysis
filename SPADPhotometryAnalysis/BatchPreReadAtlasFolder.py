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
from SPADPhotometryAnalysis import photometry_functions as fp

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
                                                                        fs=841.68,snr_thresh=2,photoncount_thre=photoncount_thre)
        
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



def read_multiple_Atlas_folder_twoROI(atlas_parent_folder,day_parent_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=2000):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    center_x_sig = ROI_info['center_x_sig']
    center_y_sig = ROI_info['center_y_sig']
    radius_sig   = ROI_info['radius_sig']
    center_x_ref = ROI_info['center_x_ref']
    center_y_ref = ROI_info['center_y_ref']
    radius_ref   = ROI_info['radius_ref']
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
        Trace_sig,Trace_ref,zdFF=AtlasDecode.get_trace_atlas_two_ROI (directory,hotpixel_path,
                                                                      center_x_sig, center_y_sig,radius_sig,
                                                                      center_x_ref, center_y_ref,radius_ref,
                                                                      fs=841.68,snr_thresh=10,photoncount_thre=photoncount_thre)
        
        folder_name = f'{new_folder_name}{i}'
        save_folder = os.path.join(day_parent_folder, folder_name)
        print ('save_folder is', save_folder)
        #z_score=replace_outliers_with_avg(z_score, threshold=4)   
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), zdFF, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_sig, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Red_traceAll.csv'), Trace_ref, delimiter=',', comments='')
        #np.save(os.path.join(save_folder, 'pixel_array_all_frames.npy'), pixel_array_all_frames)
    return -1

def read_multiple_Atlas_bin_folder_smallFOV(atlas_parent_folder,day_parent_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=2000):
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
        Trace_raw,dff=AtlasDecode.get_dff_from_atlas_snr_circle_mask_smallFOV (directory,hotpixel_path,center_x, center_y,radius,
                                                                        fs=1682.92,snr_thresh=4,photoncount_thre=photoncount_thre)
        
        # Trace_raw,dff= AtlasDecode.get_total_photonCount_atlas_continuous_circle_mask (directory,hotpixel_path,center_x, center_y,radius,fs=840,photoncount_thre=photoncount_thre)
        folder_name = f'{new_folder_name}{i}'
        save_folder = os.path.join(day_parent_folder, folder_name)
        print ('save_folder is', save_folder)

        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), dff, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_raw, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Red_traceAll.csv'), Trace_raw, delimiter=',', comments='')
        #np.save(os.path.join(save_folder, 'pixel_array_all_frames.npy'), pixel_array_all_frames)
    return -1

def read_multiple_Atlas_folder_twoROI_small(atlas_parent_folder,day_parent_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=2000):
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
        Trace_sig,Trace_ref,zdFF=AtlasDecode.get_trace_atlas_two_ROI_small (directory,hotpixel_path,
                                                                      ROI_info,fs=1682.92,snr_thresh=0,
                                                                      photoncount_thre=photoncount_thre)
        
        folder_name = f'{new_folder_name}{i}'
        save_folder = os.path.join(day_parent_folder, folder_name)
        print ('save_folder is', save_folder)
        #z_score=replace_outliers_with_avg(z_score, threshold=4)   
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), zdFF, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_sig, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Red_traceAll.csv'), Trace_ref, delimiter=',', comments='')
        #np.save(os.path.join(save_folder, 'pixel_array_all_frames.npy'), pixel_array_all_frames)
    return -1

def read_multiple_Atlas_folder_threeROI_small(atlas_parent_folder,day_parent_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=2000):
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
        '''
        Here,Trace_sig,Trace_ref,Trace_z are raw signal trace from the three ROIs
        Trace_sig: green cable one, 
        '''
        Trace_sig,Trace_ref,Trace_z=AtlasDecode.get_trace_atlas_three_ROI_small (directory,hotpixel_path,
                                                                      ROI_info,fs=1682.92,snr_thresh=0.5,
                                                                      photoncount_thre=photoncount_thre)
        
        folder_name = f'{new_folder_name}{i}'
        save_folder = os.path.join(day_parent_folder, folder_name)
        print ('save_folder is', save_folder)
        #z_score=replace_outliers_with_avg(z_score, threshold=4)   
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), Trace_z, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_sig, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Red_traceAll.csv'), Trace_ref, delimiter=',', comments='')
        #np.save(os.path.join(save_folder, 'pixel_array_all_frames.npy'), pixel_array_all_frames)
    return -1

def Calculate_dff (parent_folder,TargetfolderName='SyncRecording'):
    
    # List all files and directories in the parent folder
    all_contents = os.listdir(parent_folder)
    # Filter out directories containing the target string
    sync_recording_folders = [folder for folder in all_contents if TargetfolderName in folder]
    # Define a custom sorting key function to sort folders in numeric order
    def numeric_sort_key(folder_name):
        return int(folder_name.lstrip(TargetfolderName))
    # Sort the folders in numeric order
    sync_recording_folders.sort(key=numeric_sort_key)
    lambd = 5e3 # Adjust lambda to get the best fit
    porder = 1
    itermax = 50
    # Iterate over each sync recording folder
    for SyncRecordingName in sync_recording_folders:
        print("----Now processing folder:", SyncRecordingName)
        current_folder = os.path.join(parent_folder, SyncRecordingName)
    
        sig_csv_filename = os.path.join(current_folder, "Green_traceAll.csv")
        sig_data = np.genfromtxt(sig_csv_filename, delimiter=',')
    
        # Baseline (airPLS) and dF/F
        sig_base = fp.airPLS(sig_data, lambda_=lambd, porder=porder, itermax=itermax)
        sig = sig_data - sig_base
        eps = 1e-12
        dff_sig = 100.0 * sig / (sig_base + eps)
    
        # ---------- plotting ----------
        fs=841.38
        x = np.arange(sig_data.size) / fs
        xlab = "Time (s)"
     
    
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
        # Top: raw vs baseline
        axes[0].plot(x, sig_data, linewidth=1, label='sig_data')
        axes[0].plot(x, sig_base, linewidth=2, label='sig_base (airPLS)')
        axes[0].set_title(f"{SyncRecordingName} — raw & baseline")
        axes[0].set_ylabel("a.u.")
        axes[0].legend(loc="upper right", frameon=False)
    
        # Bottom: dF/F
        axes[1].plot(x, dff_sig, linewidth=1)
        axes[1].set_title("dF/F (%)")
        axes[1].set_ylabel("%")
        axes[1].set_xlabel(xlab)
    
        plt.tight_layout()
        out_png = os.path.join(current_folder, "sig_and_dff_preview.png")
        plt.savefig(out_png, dpi=150)
        plt.show()
    
        # Save your dF/F as before
        np.savetxt(os.path.join(current_folder, 'Zscore_traceAll.csv'),
                   dff_sig, delimiter=',', comments='')
    return -1       


def main():
    '''Set the folder to the one-day recording folder that contain SPAD data for each trial'''
    '''IMPORTANT: Set the ROI range, this can be found by the screenshot you made during recording,
    or draw image for a single trial using DemoS ingleSPAD_folder.py'''
    
    'Reading SPAD binary data'
    #hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
    hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'
    'READ SINGLE ROI CODES --FULL FOV'
    # center_x, center_y,radius=31, 49, 25
    # day_folder='H:/2024MScR_NORtask/1765508_Jedi2p_CompareSystem/Day5_Atlas_EphysGood/'
    # atlas_folder=os.path.join(day_folder,'Atlas')
    # read_multiple_Atlas_bin_folder(atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,new_folder_name='SyncRecording',photoncount_thre=500)
    
    'READ SINGLE ROI CODES--SMALL FOV'

    center_x, center_y,radius=52, 25, 12

    center_x, center_y,radius=52, 24, 12

    day_folder=r'G:\2025_ATLAS_SPAD\1881363_Jedi2p_CB\Day3'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder_smallFOV(
        atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,
        new_folder_name='SyncRecording',photoncount_thre=50000)
    day_folder=r'G:\2025_ATLAS_SPAD\1881363_Jedi2p_CB\Day4'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder_smallFOV(
        atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,
        new_folder_name='SyncRecording',photoncount_thre=50000)
    
    day_folder=r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_CB\Day1'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder_smallFOV(
        atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,
        new_folder_name='SyncRecording',photoncount_thre=50000)
    
    day_folder=r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_CB\Day2'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder_smallFOV(
        atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,
        new_folder_name='SyncRecording',photoncount_thre=50000)
    
    day_folder=r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_CB\Day3'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder_smallFOV(
        atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,
        new_folder_name='SyncRecording',photoncount_thre=50000)
    
    day_folder=r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_CB\Day4'
    atlas_folder=os.path.join(day_folder,'Atlas')
    read_multiple_Atlas_bin_folder_smallFOV(
        atlas_folder,day_folder,hotpixel_path,center_x, center_y,radius,
        new_folder_name='SyncRecording',photoncount_thre=50000)
    



    'READ TWO ROIs CODES ---FULL FOV'
    # ROI_info = {
    # 'center_x_sig': 44,
    # 'center_y_sig': 64,
    # 'radius_sig': 11,
    # 'center_x_ref': 66,
    # 'center_y_ref': 53,
    # 'radius_ref': 9
    # }
    
    # day_folder='F:/2025_ATLAS_SPAD/1881363_Jedi2p_mCherry/Day1/'
    # atlas_folder=os.path.join(day_folder,'Atlas')
    # read_multiple_Atlas_folder_twoROI (atlas_folder,day_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=3500)

    'READ TWO ROIs CODES---SMALL FOV'
    # ROI_info = {
    # 'center_x_sig': 64,
    # 'center_y_sig': 26,
    # 'radius_sig': 10,
    # 'center_x_ref': 46,
    # 'center_y_ref': 18,
    # 'radius_ref': 10
    # }
    
    # day_folder=r'G:\2025_ATLAS_SPAD\1881365_Jedi2p_mCherry\Day9_Cont'
    # atlas_folder=os.path.join(day_folder,'Atlas')
    # read_multiple_Atlas_folder_twoROI_small (atlas_folder,day_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=50000)
    

    '''
    READ THREE ROIs CODES---SMALL FOV
    z-Black label-Left CA3
    sig-Green label-Left CA1
    ref-Red label- Right CA1 with OEC
    '''
    
    # ROI_info = {
    # 'center_x_z': 47,
    # 'center_y_z': 35,
    # 'radius_z': 9,
    
    # 'center_x_sig': 48,
    # 'center_y_sig': 16,
    # 'radius_sig': 10,
    
    # 'center_x_ref': 64,
    # 'center_y_ref': 25,
    # 'radius_ref': 10
    # }
    
    # day_folder='F:/2025_ATLAS_SPAD/1887932_Jedi2p_Multi_ephysbad/Day2/'
    # atlas_folder=os.path.join(day_folder,'Atlas')
    # read_multiple_Atlas_folder_threeROI_small (atlas_folder,day_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=45000)
    
    
    # day_folder='F:/2025_ATLAS_SPAD/1887933_Jedi2P_Multi/Day5/'
    # atlas_folder=os.path.join(day_folder,'Atlas')
    # read_multiple_Atlas_folder_threeROI_small (atlas_folder,day_folder,hotpixel_path,ROI_info,new_folder_name='SyncRecording',photoncount_thre=55000)
    
  
    '''
    CALCULATE DFF
    '''
    # day_folder=r'G:\2025_ATLAS_SPAD\PVCre\1842516_PV_Jedi2p\Day1'
    # Calculate_dff (day_folder,TargetfolderName='SyncRecording')
    
    
if __name__ == "__main__":
    main()