# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:53:34 2024

@author: Yifang
"""

from SPADPhotometryAnalysis import AtlasDecode
import os
import numpy as np

'xxRange=[25, 85],yyRange=[30, 90]'


def read_multiple_Atlas_bin_folder(atlas_parent_folder,day_parent_folder,hotpixel_path,xxRange,yyRange,new_folder_name='SyncRecording'):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    # Get a list of all directories in the parent folder
    directories = [os.path.join(atlas_parent_folder, d) for d in os.listdir(atlas_parent_folder) if os.path.isdir(os.path.join(atlas_parent_folder, d))]
    # Sort the directories by creation time
    directories.sort(key=lambda x: os.path.getctime(x))
    # Read the directories in sorted order
    i=0
    for directory in directories:
        print("Folder:", directory)
        Trace_raw,z_score=AtlasDecode.get_zscore_from_atlas_continuous (directory,hotpixel_path,xxrange= xxRange,yyrange= yyRange,fs=840)
        
        i=i+1
        folder_name = f'{new_folder_name}{i}'
        save_folder = os.path.join(day_parent_folder, folder_name)
        print ('save_folder is', save_folder)
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savetxt(os.path.join(save_folder,'Zscore_traceAll.csv'), z_score, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Green_traceAll.csv'), Trace_raw, delimiter=',', comments='')
        np.savetxt(os.path.join(save_folder,'Red_traceAll.csv'), Trace_raw, delimiter=',', comments='')
    return -1


def main():
    '''Set the folder to the one-day recording folder that contain SPAD data for each trial'''
    '''IMPORTANT: Set the ROI range, this can be found by the screenshot you made during recording,
    or draw image for a single trial using DemoS ingleSPAD_folder.py'''
    
    'Reading SPAD binary data'
    hotpixel_path='F:/SPADdata/Altas_hotpixel.csv'
    xxRange=[25, 80]
    yyRange=[35, 90]
    
    # atlas_parent_folder='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240429_Day1/Atlas/'
    # day_parent_folder='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240429_Day1/'
    # read_multiple_Atlas_bin_folder(atlas_parent_folder,day_parent_folder,hotpixel_path,xxRange,yyRange,new_folder_name='SyncRecording')
    
    # atlas_parent_folder='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240430_Day2/Atlas/'
    # day_parent_folder='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240430_Day2/'
    # read_multiple_Atlas_bin_folder(atlas_parent_folder,day_parent_folder,hotpixel_path,xxRange,yyRange,new_folder_name='SyncRecording')
    
    # atlas_parent_folder='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240429_Day1/Atlas/'
    # day_parent_folder='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240429_Day1/'
    # read_multiple_Atlas_bin_folder(atlas_parent_folder,day_parent_folder,hotpixel_path,xxRange,yyRange,new_folder_name='SyncRecording')
    
    # atlas_parent_folder='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/Atlas/'
    # day_parent_folder='F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/'
    # read_multiple_Atlas_bin_folder(atlas_parent_folder,day_parent_folder,hotpixel_path,xxRange,yyRange,new_folder_name='SyncRecording')


if __name__ == "__main__":
    main()