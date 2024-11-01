# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:08:41 2024

@author: Yifang
"""
'''This function reads multipile Open Ephys recordings in the same folder with file created temporal order
It will create save save data the SyncRecordingX folder.'''

import os
import OpenEphysTools as OE
from open_ephys.analysis import Session
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator

def read_multiple_Ephys_data_in_folder(Ephys_folder_path,save_parent_folder,mode='py',Ephys_fs=30000,new_folder_name='SyncRecording',recordingTime=30):
    '''
    mode: py--to read session recorded with pyPhotometry
        SPAD--to read session recorded with SPAD (I haven't code for SPAD batch processing yet')
    '''
    thisSession = Session(Ephys_folder_path)
    totalRecordingNums=len(thisSession.recordnodes[0].recordings)
    print ('processing folder:',Ephys_folder_path)
    print ('Total recording trials in this Session---',totalRecordingNums)
    
    if mode =='py' :
        print ('---The processing is done with a pyPhotometry or Camera Sync mask---')
        for i in range(totalRecordingNums):
            index=i+1
            print (f'----Processing Recording {index}----')
            EphysData=OE.readEphysChannel_withSessionInput (thisSession,recordingNum=i)
            OE.plot_trace_in_seconds(EphysData['CamSync'],Ephys_fs,title= f'Camera Sync pulse for recording {index}')
            py_mask = OE.py_sync_mask (EphysData['CamSync'], start_lim=0, end_lim=len (EphysData['CamSync']))
            OE.check_Optical_mask_length(py_mask)
            EphysData['py_mask']=py_mask
            
            folder_name = f'{new_folder_name}{index}'
            save_folder_path = os.path.join(save_parent_folder, folder_name)
            # Create the folder if it doesn't exist
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            OE.save_open_ephys_data (save_folder_path,EphysData)
    if mode =='SPAD' :
        print ('---The processing is done with a SPAD and a Camera Sync mask---')
        for i in range(totalRecordingNums):
            index=i+1
            print (f'----Processing Recording {index}----')
            EphysData=OE.readEphysChannel_withSessionInput (thisSession,recordingNum=i)
            '''This is to check the SPAD mask range and to make sure SPAD sync is correctly recorded by the Open Ephys'''
            fig, ax = plt.subplots(figsize=(15,5))
            ax.plot(EphysData['SPADSync'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            num_ticks = 20  # Adjust the number of ticks as needed
            ax.xaxis.set_major_locator(MaxNLocator(num_ticks))
            plt.show()
            'This is to find the SPAD mask based on the proxy time range of SPAD sync.  Change the start_lim and end_lim to generate the SPAD mask.'
            # Get user input for start_lim and end_lim
            start_lim = int(input("Enter the start limit: "))
            end_lim = int(input("Enter the end limit: "))
            SPAD_mask = OE.SPAD_sync_mask (EphysData['SPADSync'], start_lim=start_lim, end_lim=end_lim)
            
            '''To double check the SPAD mask'''
            fig, ax = plt.subplots(figsize=(15,5))
            ax.plot(SPAD_mask)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.show()
            '''To double check the SPAD mask'''
            OE.check_Optical_mask_length(SPAD_mask)
            # If the SPAD mask is correct, save spad mask
            EphysData['SPAD_mask'] = SPAD_mask
            OE.plot_trace_in_seconds(EphysData['CamSync'],Ephys_fs,title= f'Camera Sync pulse for recording {index}')
            cam_mask = OE.py_sync_mask (EphysData['CamSync'], start_lim=0, end_lim=len (EphysData['CamSync']))
            OE.check_Optical_mask_length(cam_mask)
            EphysData['cam_mask']=cam_mask
            'SAVE THE open ephys data as .pkl file.'        
            folder_name = f'{new_folder_name}{index}'
            save_folder_path = os.path.join(save_parent_folder, folder_name)
            # Create the folder if it doesn't exist
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            OE.save_open_ephys_data (save_folder_path,EphysData)
    if mode =='Atlas' :
        for i in range(totalRecordingNums):
            index=i+1
            print (f'----Processing Recording {index}----')
            EphysData=OE.readEphysChannel_withSessionInput (thisSession,recordingNum=i)
            fig, ax = plt.subplots(figsize=(15,5))
            ax.plot(EphysData['AtlasSync'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            num_ticks = 20  # Adjust the number of ticks as needed
            ax.xaxis.set_major_locator(MaxNLocator(num_ticks))
            plt.show()
            Atlas_mask = OE.Atlas_sync_mask (EphysData['AtlasSync'], start_lim=0, end_lim=len(EphysData['AtlasSync']),recordingTime=recordingTime)
            '''To double check the SPAD mask'''
            fig, ax = plt.subplots(figsize=(15,5))
            ax.plot(Atlas_mask)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            '''To double check the SPAD mask'''
            OE.check_Optical_mask_length(Atlas_mask)
            EphysData['SPAD_mask'] = Atlas_mask
            OE.plot_trace_in_seconds(EphysData['CamSync'],Ephys_fs,title= f'Camera Sync pulse for recording {index}')
            cam_mask = OE.py_sync_mask (EphysData['CamSync'], start_lim=0, end_lim=len (EphysData['CamSync']))
            OE.check_Optical_mask_length(cam_mask)
            EphysData['cam_mask']=cam_mask
            'SAVE THE open ephys data as .pkl file.'
            folder_name = f'{new_folder_name}{index}'
            save_folder_path = os.path.join(save_parent_folder, folder_name)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            OE.save_open_ephys_data (save_folder_path,EphysData)
    
    return -1


def main():
    '''Set the folder for the Open Ephys recording, defualt folder names are usually date and time'''
    '''Set the parent folder your session results, this should be the same parent folder to save optical data'''
    Ephys_fs=30000 #Ephys sampling rate
    '''IF ATLAS'''
    Frame_num=25200
    Fs_atlas=840
    recordingTime=Frame_num/Fs_atlas
    
    Ephys_folder_path = 'E:/ATLAS_SPAD/1820061_PVcre/Day4/Ephys/2024-10-28_15-51-19/'
    save_parent_folder='E:/ATLAS_SPAD/1820061_PVcre/Day4/'
    read_multiple_Ephys_data_in_folder(Ephys_folder_path,save_parent_folder,mode='Atlas',Ephys_fs=Ephys_fs,new_folder_name='SyncRecording',recordingTime=recordingTime)

if __name__ == "__main__":
    main()
