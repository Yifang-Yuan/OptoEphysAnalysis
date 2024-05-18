# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:16:37 2024
@author: Yifang

This file is to concatenate processed trials data for bulk analysis of pre-learning and post-learning.
It takes all Ephys_tracking_photometry_aligned.pkl file in multiple SyncRecording* folders and make a new .pkl file for the concatenated data
Be careful to analyse concatenated traces because the animal might be at different behaviour states and threshold might be different.
"""
import os
import numpy as np
import pandas as pd
import pickle
from SyncOECPySessionClass import SyncOEpyPhotometrySession

def ConcatenateTrial (parent_folder,TargetfolderName='SyncRecording', targetFile='Ephys_tracking_photometry_aligned.pkl',
                      StartTrialIdx=1, EndTrialIdx=4, trialTag='Post'):
    ConCatenateData=pd.DataFrame()
    for i in range (StartTrialIdx,EndTrialIdx+1):
        foldername=TargetfolderName+str(i)
        filepath=os.path.join(parent_folder, foldername,"Ephys_tracking_photometry_aligned.pkl")
        TrialData = pd.read_pickle(filepath)
        ConCatenateData = pd.concat([ConCatenateData, TrialData], ignore_index=True,axis=0)
    
    
    save_folder_name = f'Saved{trialTag}Trials'
    save_folder_path = os.path.join(parent_folder, save_folder_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    filepath=os.path.join(save_folder_path, targetFile)
    
    ConCatenateData.to_pickle(filepath)
    'Rebuild the timestamp'
    time_interval=1/10000
    total_duration=len(ConCatenateData)*time_interval
    timestamps_new = np.arange(0, total_duration, time_interval)
    if len(ConCatenateData)>len(timestamps_new):
        ConCatenateData=ConCatenateData[:len(timestamps_new)]
    elif len(ConCatenateData)<len(timestamps_new):
        timestamps_new=timestamps_new[:len(ConCatenateData)]   
        
    timestamps_time = pd.to_timedelta(timestamps_new, unit='s')
    ConCatenateData.index = timestamps_time            
    ConCatenateData['timestamps']=timestamps_new             
    return -1

def main():
    'Specify the session and trial indice you want to concatenate'
    parent_folder="F:/2024MScR_NORtask/1765508_Jedi2p_Atlas/20240430_Day2/" 
    ConcatenateTrial (parent_folder,TargetfolderName='SyncRecording', targetFile='Ephys_tracking_photometry_aligned.pkl',
                                                        StartTrialIdx=1, EndTrialIdx=5, trialTag='PreAwake')

if __name__ == "__main__":
    main()
