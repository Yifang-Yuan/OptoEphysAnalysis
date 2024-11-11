# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:02:38 2024
@author: Yifang
Batch process multiple SPAD recording folders in an experiment.
"""
import shutil
from SPADPhotometryAnalysis import SPADreadBin
import os
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import photometry_functions as fp
import numpy as np
import matplotlib.pyplot as plt

def read_multiple_SPAD_bin_folder(parent_folder,xxRange,yyRange):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    # Get a list of all directories in the parent folder
    directories = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    # Sort the directories by creation time
    directories.sort(key=lambda x: os.path.getctime(x))
    # Read the directories in sorted order
    for directory in directories:
        print("Folder:", directory)
        SPADreadBin.readMultipleBinfiles(directory,1,xxRange=xxRange,yyRange=yyRange)
    return -1

'Following functions are used for calculate zscore for continuous recording'

def multiple_SPAD_folders_get_zscore(parent_folder):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    def sort_by_trial_index(file_name):
        trial_index = int(file_name.split("Trial")[-1])
        return trial_index
    folder_names = [folder for folder in os.listdir(parent_folder) if 'Trial' in folder]
    sorted_folders = sorted(folder_names, key=sort_by_trial_index)
    # Read the directories in sorted order
    for folder_name in sorted_folders:
        directory=os.path.join(parent_folder, folder_name)
        print("Folder name:", directory)
        'Demodulate single ROI time division recodings'
        filename=Analysis.Set_filename (directory, csv_filename="traceValueAll.csv")
        raw_signal=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=100)
        
        lambd = 10e3 # Adjust lambda to get the best fit
        porder = 1
        itermax = 15
        sig_base=fp.airPLS(raw_signal,lambda_=lambd,porder=porder,itermax=itermax) 
        signal = (raw_signal - sig_base)  
        z_score=(signal - np.median(signal)) / np.std(signal)
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(111)
        ax1 = fp.plotSingleTrace (ax1, z_score, SamplingRate=9938.4,color='black',Label='zscore_signal')

        greenfname = os.path.join(directory, "Green_traceAll.csv")
        np.savetxt(greenfname, raw_signal, delimiter=",")
        redfname = os.path.join(directory, "Red_traceAll.csv")
        np.savetxt(redfname, raw_signal, delimiter=",")
        zscorefname = os.path.join(directory, "Zscore_traceAll.csv")
        np.savetxt(zscorefname, z_score, delimiter=",")
    return -1

def copy_file(file_to_copy,source_dir,destination_dir):
    if file_to_copy in os.listdir(source_dir):
        # Construct the source and destination file paths
        source_file = os.path.join(source_dir, file_to_copy)
        destination_file = os.path.join(destination_dir, file_to_copy)
        # Copy the file to the destination directory
        shutil.copy(source_file, destination_file)
    else:
        print(f"File '{file_to_copy}' does not exist in the source directory.")
    return -1

def copy_results_to_SyncRecording (day_parent_folder,SPAD_parent_folder,new_folder_name='SyncRecording'):
    # Sort the directories by creation time
    def sort_by_trial_index(file_name):
        trial_index = int(file_name.split("Trial")[-1])
        return trial_index
    folder_names = [folder for folder in os.listdir(SPAD_parent_folder) if 'Trial' in folder]
    sorted_folders = sorted(folder_names, key=sort_by_trial_index)
    # Read the directories in sorted order
    i=0
    for folder in sorted_folders:
        i=i+1
        source_dir=os.path.join(SPAD_parent_folder, folder)
        folder_name = f'{new_folder_name}{i}'
        destination_dir = os.path.join(day_parent_folder, folder_name)
        print ('source_dir is', source_dir)
        print ('destination_dir is', destination_dir)
        # Create the folder if it doesn't exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        file_to_copy = 'Zscore_traceAll.csv'
        copy_file(file_to_copy,source_dir,destination_dir)
        file_to_copy = 'Green_traceAll.csv'
        copy_file(file_to_copy,source_dir,destination_dir)
        file_to_copy = 'Red_traceAll.csv'
        copy_file(file_to_copy,source_dir,destination_dir)
    return -1


def main():
    '''Set the folder to the one-day recording folder that contain SPAD data for each trial'''
    '''
    IMPORTANT: 
    (1)Set the ROI range, this can be found by the screenshot you made during recording,
    or draw image for a single trial using DemoSingleSPAD_folder.py
    (2)Each SPC trial folder should be named as xxx_Trial1,xxx_Trial2
    
    '''
    
    'Reading SPAD binary data'
    SPC_data_folder='E:/ATLAS_SPAD\HardwareTest/SPC_linearity/SPC/'
    read_multiple_SPAD_bin_folder(SPC_data_folder,xxRange=[135,270],yyRange=[70,205])
    multiple_SPAD_folders_get_zscore(SPC_data_folder)
    day_parent_folder='E:/ATLAS_SPAD\HardwareTest/SPC_linearity/'
 
    copy_results_to_SyncRecording (day_parent_folder,SPC_data_folder,new_folder_name='SyncRecording')

if __name__ == "__main__":
    main()