# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:28:55 2024

@author: Yifang
"""
import numpy as np
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
import pandas as pd
import os

'''This function reads multipile pyPhotometry .csv files in the same folder with file created temporal order
It will create a new folder for each .csv data and save zscore and camsync data in the new folder'''

def read_multiple_photometry_files_in_folder(pydata_folder_path,save_parent_folder,sampling_rate=130,new_folder_name='SyncRecording'):
    '''
    Assuming all pyPhotometry files in this folder are from one session of experiment, 
    this function will read and processing all of them and save results in separate new folders.
    pydata_folder_path:path for the pyPhotometry data
    save_parent_folder: parent folder to save results in new folders
    '''    
    # Get a list of all files in the folder
    all_files = os.listdir(pydata_folder_path)
    # Filter for CSV files and get their full paths
    csv_files = [os.path.join(pydata_folder_path, file) for file in all_files if file.endswith('.csv')]
    # Sort the CSV files based on their modified time
    csv_files_sorted = sorted(csv_files, key=os.path.getmtime) #sorted by modified time
    #csv_files_sorted = sorted(csv_files, key=os.path.getctime) #sorted by created time

    # Read each CSV file in order
    for i, csv_file in enumerate(csv_files_sorted, 1):
        folder_name = f'{new_folder_name}{i}'
        save_folder_path = os.path.join(save_parent_folder, folder_name)
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        
        # Extract the file name from the file path
        file_name = os.path.basename(csv_file)
        print(f"Reading {file_name}...")
        '''Read csv file and calculate zscore of the fluorescent signal'''
        PhotometryData = pd.read_csv(csv_file,index_col=False) # Adjust this line depending on your data file
        raw_reference = PhotometryData[' Analog2'][1:]
        raw_signal = PhotometryData['Analog1'][1:]
        Cam_Sync=PhotometryData[' Digital1'][1:]
        
        '''Get zdFF directly'''
        zdFF = fp.get_zdFF(raw_reference,raw_signal,smooth_win=2,remove=0,lambd=5e4,porder=1,itermax=50)
        fig = plt.figure(figsize=(16, 5))
        ax1 = fig.add_subplot(111)
        ax1 = fp.plotSingleTrace (ax1, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')
        '''Save signal'''
        greenfname = os.path.join(save_folder_path, "Green_traceAll.csv")
        np.savetxt(greenfname, raw_signal, delimiter=",")
        redfname = os.path.join(save_folder_path, "Red_traceAll.csv")
        np.savetxt(redfname, raw_reference, delimiter=",")
        zscorefname = os.path.join(save_folder_path, "Zscore_traceAll.csv")
        np.savetxt(zscorefname, zdFF, delimiter=",")
        CamSyncfname = os.path.join(save_folder_path, "CamSync_photometry.csv")
        np.savetxt(CamSyncfname, Cam_Sync, fmt='%d',delimiter=",")
        
    return -1

def main():
    pydata_folder_path='F:/2024MScR_NORtask/1748725_mdlx-G8f-py/20240409_Day2/pyPhotometry/'
    save_parent_folder='F:/2024MScR_NORtask/1748725_mdlx-G8f-py/20240409_Day2/'
    read_multiple_photometry_files_in_folder(pydata_folder_path,save_parent_folder,sampling_rate=130,new_folder_name='SyncRecording')

if __name__ == "__main__":
    main()