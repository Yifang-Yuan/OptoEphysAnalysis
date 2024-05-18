# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:25:05 2024
@author: Yifang
"""

import numpy as np
import matplotlib.pyplot as plt
import OpenEphysTools as OE
import pandas as pd
import os

def label_multiple_behaviour_files_in_folder(BehaviourData_folder_path,save_parent_folder,tracking_fs=10,new_folder_name='SyncRecording'):
    LabelFile=os.path.join(save_parent_folder,'TrailLabel.csv')
    LabelData = pd.read_csv(LabelFile)
    print(LabelData)
    files = os.listdir(BehaviourData_folder_path)
    # Filter out the CSV files that match the pattern 'AnimalTracking_*.csv'
    csv_files = [file for file in files if file.startswith('AnimalTracking_') and file.endswith('.csv')]
    # Sort the CSV files based on the last two digits in their filenames
    sorted_csv_files = sorted(csv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Read the CSV files in sorted order
    i=0
    for file in sorted_csv_files:
        file_path = os.path.join(BehaviourData_folder_path, file)
        print(f"'{file}' labelled as--:'{LabelData['TrialNum'][i]}''{LabelData['sleepState'][i]}'--'{LabelData['movingState'][i]}'--'{LabelData['TrainingState'][i]}'")
        trackingdata = pd.read_csv(file_path, names=['X', 'Y'])
        trackingdata=trackingdata.fillna(method='ffill')
        trackingdata=trackingdata/20
        if LabelData['sleepState'][i]=='sleep':
            trackingdata['speed']=0
        elif LabelData['sleepState'][i]=='awake' and LabelData['movingState'][i]=='notmoving':
            trackingdata['speed']=0
        else:
            trackingdata['speed']=((trackingdata['X'].diff()**2 + trackingdata['Y'].diff()**2)**0.5)*tracking_fs # cm per second
            trackingdata['speed'][trackingdata['speed'] > 20] = np.nan #If the speed is too fast, maybe the tracking position is wrong, delete it.
            trackingdata['speed'] = trackingdata['speed'].fillna(method='bfill')
            window_size=5
            trackingdata['speed'] = trackingdata['speed'].rolling(window=window_size, min_periods=1).median()
            trackingdata['speed'] = trackingdata['speed'].rolling(window=window_size, min_periods=1).min()
        # trackingdata['sleepState']=LabelData['sleepState'][i]
        # trackingdata['movingState']=LabelData['movingState'][i]
        # trackingdata['TrainingState']=LabelData['TrainingState'][i]
        OE.plot_animal_tracking (trackingdata)
        plt.plot(trackingdata['speed']) 
        folder_name = f'{new_folder_name}{i+1}'
        save_folder_path = os.path.join(save_parent_folder, folder_name)
        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        filepath=os.path.join(save_folder_path, f'AnimalTracking_{i+1}.pkl')
        trackingdata.to_pickle(filepath)
        i=i+1
    return -1

def plot_multiple_behaviour_files_in_folder(BehaviourData_folder_path,tracking_fs=10):
    files = os.listdir(BehaviourData_folder_path)
    # Filter out the CSV files that match the pattern 'AnimalTracking_*.csv'
    csv_files = [file for file in files if file.startswith('AnimalTracking_') and file.endswith('.csv')]
    # Sort the CSV files based on the last two digits in their filenames
    sorted_csv_files = sorted(csv_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Read the CSV files in sorted order
    for file in sorted_csv_files:
        file_path = os.path.join(BehaviourData_folder_path, file)
        print(f"Processing file '{file}' and plot tracking")
        trackingdata = pd.read_csv(file_path, names=['X', 'Y'])
        trackingdata=trackingdata.fillna(method='ffill')
        trackingdata=trackingdata/20    
        trackingdata['speed']=((trackingdata['X'].diff()**2 + trackingdata['Y'].diff()**2)**0.5)*tracking_fs # cm per second
        trackingdata['speed'][trackingdata['speed'] > 20] = np.nan #If the speed is too fast, maybe the tracking position is wrong, delete it.
        trackingdata['speed'] = trackingdata['speed'].fillna(method='bfill')
        OE.plot_animal_tracking (trackingdata)
        #plt.plot(trackingdata['speed']) 
    return -1

def main():   
    
    '''Set the folder for the Behaviour recording, defualt folder names are usually date and time'''
    '''Set the parent folder your session results, this should be the same parent folder to save optical data'''
    
    save_parent_folder='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240503_Day5/'
    BehaviourData_folder_path='F:/2024MScR_NORtask/1765507_iGlu_Atlas/20240503_Day5/Behaviour'
    label_multiple_behaviour_files_in_folder(BehaviourData_folder_path,save_parent_folder,tracking_fs=10,new_folder_name='SyncRecording')
    
    #plot_multiple_behaviour_files_in_folder(BehaviourData_folder_path,tracking_fs=10)
if __name__ == "__main__":
    main()


#%%
# save_parent_folder='F:/2024MScR_NORtask/1732333_pyPhotometry/20240212_Day1/'
# BehaviourData_folder_path='F:/2024MScR_NORtask/1732333_pyPhotometry/20240212_Day1/Behaviour/'

# #%%
# tracking_fs=10
# csv_file_path='F:/2024MScR_NORtask/1732333_pyPhotometry/20240212_Day1/Behaviour/AnimalTracking_6.csv'
# '''Read csv file and calculate zscore of the fluorescent signal'''
# trackingdata = pd.read_csv(csv_file_path, names=['X', 'Y'])
# trackingdata=trackingdata.fillna(method='ffill')
# trackingdata=trackingdata/20    
# trackingdata['speed']=((trackingdata['X'].diff()**2 + trackingdata['Y'].diff()**2)**0.5)*tracking_fs # cm per second
# trackingdata['speed'][trackingdata['speed'] > 20] = np.nan #If the speed is too fast, maybe the tracking position is wrong, delete it.
# trackingdata['speed'] = trackingdata['speed'].fillna(method='bfill')
# OE.plot_animal_tracking (trackingdata)
# #%%
# plt.plot(trackingdata['X'][0:100]) 
# plt.plot(trackingdata['Y'][0:100]) 
# #%%
# window_size=5
# trackingdata['speed'] = trackingdata['speed'].rolling(window=window_size, min_periods=1).median()
# trackingdata['speed'] = trackingdata['speed'].rolling(window=window_size, min_periods=1).min()
# #%%
# plt.plot(trackingdata['speed']) 
