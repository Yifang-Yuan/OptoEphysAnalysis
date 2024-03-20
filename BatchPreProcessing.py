# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:34:27 2024

@author: Yifang
"""
import PreReadBehaviourFolder
import PreReadEphysFolder
import PreReadpyPhotometryFolder

pydata_folder_path='F:/2024MScR_NORtask/1723433_pyPhotometry_mdlxG8f/20231215_Day1/pyPhotometry'
save_parent_folder='F:/2024MScR_NORtask/1723433_pyPhotometry_mdlxG8f/20231215_Day1/'
PreReadpyPhotometryFolder.read_multiple_photometry_files_in_folder(pydata_folder_path,save_parent_folder,sampling_rate=130,new_folder_name='SyncRecording')

Ephys_folder_path = "F:/2024MScR_NORtask/1723433_pyPhotometry_mdlxG8f/20231215_Day1/OpenEphys/2023-12-15_15-37-34/"   
save_parent_folder="F:/2024MScR_NORtask/1723433_pyPhotometry_mdlxG8f/20231215_Day1/" 
Ephys_fs=30000 #Ephys sampling rate
PreReadEphysFolder.read_multiple_Ephys_data_in_folder(Ephys_folder_path,save_parent_folder,mode='py',Ephys_fs=Ephys_fs,new_folder_name='SyncRecording')

save_parent_folder='F:/2024MScR_NORtask/1723433_pyPhotometry_mdlxG8f/20231215_Day1/'
BehaviourData_folder_path='F:/2024MScR_NORtask/1723433_pyPhotometry_mdlxG8f/20231215_Day1/Behaviour/'
PreReadBehaviourFolder.label_multiple_behaviour_files_in_folder(BehaviourData_folder_path,save_parent_folder,tracking_fs=10,new_folder_name='SyncRecording')