# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:08:41 2024

@author: Yang
"""

'''This function reads multipile Open Ephys recordings in the same folder with file created temporal order
It will create save save data the SyncRecordingX folder.'''

import os
import OpenEphysTools as OE
from open_ephys.analysis import Session


def read_multiple_Ephys_data_in_folder(Ephys_folder_path,save_parent_folder,mode='py',Ephys_fs=30000,new_folder_name='SyncRecording'):
    '''
    mode: py--to read session recorded with pyPhotometry
        SPAD--to read session recorded with SPAD (I haven't code for SPAD batch processing yet')
    '''
    thisSession = Session(Ephys_folder_path)
    totalRecordingNums=len(thisSession.recordnodes[0].recordings)
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
    
    return -1


def main():
    '''Set the folder for the Open Ephys recording, defualt folder names are usually date and time'''
    '''Set the parent folder your session results, this should be the same parent folder to save optical data'''
    
    Ephys_folder_path = "E:/YYFstudy/20240212_Day1/OpenEphys/2024-02-12_15-02-31"   
    save_parent_folder="E:/YYFstudy/20240212_Day1/" 
    Ephys_fs=30000 #Ephys sampling rate
    read_multiple_Ephys_data_in_folder(Ephys_folder_path,save_parent_folder,mode='py',Ephys_fs=Ephys_fs,new_folder_name='SyncRecording')

if __name__ == "__main__":
    main()
