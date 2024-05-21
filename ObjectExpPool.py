# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:44:44 2024

@author: Yifang
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import OpenEphysTools as OE
import pynapple as nap
import pickle
import MakePlots

def load_pickle_files (filepath):
    with open(filepath, 'rb') as file:
        data=pickle.load(file)
    return data

def SeparateTrialsByStateAndSave (parent_folder,LFP_channel='LFP_1'):
    '''
    This function pools data from all sessions and separate trial by the animals behavioural/sleeping state.
    It will go through all daily folders and trial folders to check the state label of the trial,
    then save optical transient features during ripple and theta in a dictionary with animal sleeping state as the key.
    '''
    'State label as the key to the dictionary'
    #List_template={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_triggered_optical_peak_value= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_triggered_optical_peak_time= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_triggered_zscore= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_triggered_LFP= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_event_corr= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_freq= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_numbers= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_std_values= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    ripple_duration_values= {'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    theta_triggered_optical_peak_value={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    theta_triggered_optical_peak_time={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    theta_triggered_zscore={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    theta_triggered_LFP={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    theta_event_corr={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    
    all_contents = os.listdir(parent_folder)
    # Filter out directories containing the target string
    day_recording_folders = [folder for folder in all_contents if 'Day' in folder]
    # Define a custom sorting key function to sort folders in numeric order
    sorted_folders = sorted(day_recording_folders, key=lambda x: int(x.split('Day')[-1]))
    
    for DayRecordingFolder in sorted_folders:
    # for i in range(2):
    #     if i==0:
    #         DayRecordingFolder='20240213_Day2'
    #     if i==1:
    #         DayRecordingFolder='20240216_Day5'
            
        print("Day folder:", DayRecordingFolder)
        day_folder_path= os.path.join(parent_folder, DayRecordingFolder)
        all_contents = os.listdir(day_folder_path)
        sync_recording_folders = [folder for folder in all_contents if 'SyncRecording' in folder]
        # Define a custom sorting key function to sort folders in numeric order
        def numeric_sort_key(folder_name):
            return int(folder_name.lstrip('SyncRecording'))
        sync_recording_folders.sort(key=numeric_sort_key)
        for SyncRecordingName in sync_recording_folders:
            print("Now processing folder:", SyncRecordingName)
            Classfilepath=os.path.join(day_folder_path, SyncRecordingName,SyncRecordingName+LFP_channel+'_Class.pkl')
            print ("Class file name:", Classfilepath)
            with open(Classfilepath, 'rb') as file:
                Recording1 = pickle.load(file)
                column_name=Recording1.TrainingState+'_'+Recording1.sleepState
                
                if hasattr(Recording1, 'ripple_triggered_optical_peak_values'):
                    ripple_triggered_optical_peak_value [column_name].append(Recording1.ripple_triggered_optical_peak_values)
                    ripple_triggered_optical_peak_time [column_name].append(Recording1.ripple_triggered_optical_peak_times)
                    ripple_triggered_zscore [column_name].append(Recording1.ripple_triggered_zscore_values)
                    ripple_triggered_LFP[column_name].append(Recording1.ripple_triggered_LFP_values_1)
                    ripple_event_corr[column_name].append(Recording1.ripple_event_corr_array)
                ripple_freq[column_name].append(Recording1.ripple_freq)
                ripple_numbers[column_name].append(Recording1.ripple_numbers)
                ripple_std_values[column_name].append(Recording1.ripple_std_values)
                ripple_duration_values[column_name].append(Recording1.ripple_duration_values)
                
                theta_triggered_optical_peak_value [column_name].append(Recording1.theta_triggered_optical_peak_values)
                theta_triggered_optical_peak_time [column_name].append(Recording1.theta_triggered_optical_peak_times)
                theta_triggered_zscore [column_name].append(Recording1.theta_triggered_zscore_values)
                #theta_triggered_LFP[column_name].append(Recording1.theta_triggered_LFP_values)
                theta_event_corr[column_name].append(Recording1.theta_event_corr_array)
                
    save_path = os.path.join(parent_folder, 'ripple_triggered_optical_peak_value_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_triggered_optical_peak_value, file)
    save_path = os.path.join(parent_folder, 'ripple_triggered_optical_peak_time_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_triggered_optical_peak_time, file)
    save_path = os.path.join(parent_folder, 'ripple_triggered_zscore_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_triggered_zscore, file)
    save_path = os.path.join(parent_folder, 'ripple_triggered_LFP_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_triggered_LFP, file)     
    save_path = os.path.join(parent_folder, 'ripple_event_corr_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_event_corr, file)
    save_path = os.path.join(parent_folder, 'ripple_freq'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_freq, file)
    save_path = os.path.join(parent_folder, 'ripple_numbers'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_numbers, file)
    save_path = os.path.join(parent_folder, 'ripple_std_values'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_std_values, file)     
    save_path = os.path.join(parent_folder, 'ripple_duration_values'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(ripple_duration_values, file)           
    save_path = os.path.join(parent_folder, 'theta_triggered_optical_peak_value_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(theta_triggered_optical_peak_value, file)
    save_path = os.path.join(parent_folder, 'theta_triggered_optical_peak_time_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(theta_triggered_optical_peak_time, file)
    save_path = os.path.join(parent_folder, 'theta_triggered_zscore_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(theta_triggered_zscore, file)
    save_path = os.path.join(parent_folder, 'theta_triggered_LFP_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(theta_triggered_LFP, file) 
    save_path = os.path.join(parent_folder, 'theta_event_corr_'+LFP_channel+'.pkl')
    with open(save_path, 'wb') as file:
        pickle.dump(theta_event_corr, file) 
    
    return -1

def PoolDatabyStateAndPlot (parent_folder, LFP_channel, mode='ripple'):
    '''
    This function will read the saved dictionary files from the above funcion and plot optical transient features according to state.
    '''
    if mode=='ripple':
        half_window=0.1 #seconds, for ripple
    if mode=='theta':
        half_window=0.5 #seconds, for theta
    filename = os.path.join(parent_folder, mode+'_triggered_optical_peak_value_'+LFP_channel+'.pkl')
    peak_value=load_pickle_files (filename)
    
    filename = os.path.join(parent_folder, mode+'_triggered_optical_peak_time_'+LFP_channel+'.pkl')
    time_dict=load_pickle_files (filename)

    filename = os.path.join(parent_folder, mode+'_triggered_zscore_'+LFP_channel+'.pkl')
    zscore=load_pickle_files (filename)

    filename = os.path.join(parent_folder, mode+'_triggered_LFP_'+LFP_channel+'.pkl')
    LFP=load_pickle_files (filename)
 
    filename = os.path.join(parent_folder, mode+'_event_corr_'+LFP_channel+'.pkl')
    event_corr=load_pickle_files (filename)
            
    savepath = os.path.join(parent_folder, "ResultsPooled")
    if not os.path.exists(savepath):
        os.makedirs(savepath)     
    for key in time_dict:
        print (key)
        if time_dict[key]:
            time_i = np.concatenate(time_dict[key])
            peak_value_i = np.concatenate(peak_value[key])
            zscore_i = np.concatenate(zscore[key])
            LFP_i = np.concatenate(LFP[key])
            event_corr_i=np.concatenate(event_corr[key])
            mean_z_score,std_z_score, CI_z_score=OE.calculateStatisticNumpy (zscore_i)
            mean_LFP,std_LFP, CI_LFP=OE.calculateStatisticNumpy (LFP_i)
            mean_event_corr,std_event_corr,CI_event_corr=OE.calculateStatisticNumpy (event_corr_i)
            '--plot transient and LFP---'
            x = np.linspace(-half_window, half_window, len(mean_z_score))
            fig, ax = plt.subplots(figsize=(8, 4))
            MakePlots.plot_oscillation_epoch_traces(ax,x,mean_z_score,
                                                    mean_LFP,std_z_score,std_LFP,CI_z_score,CI_LFP,mode='ripple',plotShade='CI')
            fig.suptitle(f'{key}: Mean optical transient triggered by {mode} peak in {LFP_channel}')
            figName=f'{key}_Optical_triggered_by_{mode}_{LFP_channel}.png'
            fig.savefig(os.path.join(savepath,figName))
            
            '---scatter plot optical peak and LFP---'
            fig, ax = plt.subplots(figsize=(8, 4))
            MakePlots.plot_oscillation_epoch_optical_peaks(ax,x,time_i,peak_value_i,mean_LFP,std_LFP,CI_LFP,
                                                            half_window,mode='ripple',plotShade='CI')
            fig.suptitle(f'{key}_optical_peak_triggered_by_{mode}_{LFP_channel}')
            figName=f'{key}_Optical Peak triggered by {mode}_{LFP_channel}.png'
            fig.savefig(os.path.join(savepath,figName))
            
            '---plot correlation---'
            Fs=10000
            x = np.linspace((-len(mean_event_corr)/2)/Fs, (len(mean_event_corr)/2)/Fs, len(mean_event_corr))  
            plt.figure(figsize=(8, 4))
            plt.plot(x, mean_event_corr, color='gray', label='Mean Cross-Correlation')
            plt.fill_between(x, CI_event_corr[0], CI_event_corr[1], color='gray', alpha=0.2, label='0.95 CI')
            plt.xlabel('Lags(seconds)')
            plt.ylabel('Cross-Correlation')
            plt.title(f'{key}: Mean Cross-Correlation (1-Second Window)_{LFP_channel}')
            plt.legend()
            figName=f'{key}_Optical_LFP_corr_{mode}_{LFP_channel}.png'
            plt.savefig(os.path.join(savepath,figName))
     
    return -1

def Compare_OpticalPeak_RipplePeak (parent_folder, LFP_channel,side='both', halfwindow=0.01, mode='ripple'):
    '''
    This function will plot histogram of optical peak density around a ripple peak by animal sleeping state.
    It will also count the optical peak with a given half window around the ripple peak
    
    '''
    filename = os.path.join(parent_folder, mode+'_triggered_optical_peak_time_'+LFP_channel+'.pkl')
    time_dict=load_pickle_files (filename)
    savepath = os.path.join(parent_folder, "ResultsPooled")
    peak_num_probability={'pre_sleep': [], 'pre_awake': [], 'post_sleep': [],'post_awake':[],'openfield_awake':[]}
    if not os.path.exists(savepath):
        os.makedirs(savepath)     
    for key in time_dict:
        if time_dict[key]:
            time_i = np.concatenate(time_dict[key])
            total_num=len(time_i)
            '---plot histogram----'
            # Plotting histograms
            timepoints_negative = time_i[time_i < 0]
            timepoints_positive = time_i[time_i >= 0]
            plt.figure(figsize=(8, 6))
            # Histogram for timepoints smaller than 0
            plt.hist(timepoints_negative, bins=20, color='blue', alpha=0.5, label='Timepoints < 0',density=True)
            # Histogram for timepoints larger than or equal to 0
            plt.hist(timepoints_positive, bins=20, color='red', alpha=0.5, label='Timepoints >= 0',density=True)
            plt.xlabel('Time relevant to LFP ripple peak (seconds)')
            plt.ylabel('peak numbers (density)')
            plt.title(f'{key}: Histogram of Optical peak times {LFP_channel}')
            figName=f'{key}_Optical_peaktime_hist_{mode}_{LFP_channel}.png'
            plt.savefig(os.path.join(savepath,figName))
            '----calculate ripple-optical coocurence----'
            if side=='both':
                time_i=time_i[time_i < halfwindow]
                time_i=time_i[time_i>-halfwindow]
                peak_num_probability[key]=len(time_i)/total_num
            if side =='after':
                time_i=time_i[time_i < halfwindow]
                time_i=time_i[time_i>0]
                peak_num_probability[key]=len(time_i)/total_num
            if side =='before':
                time_i=time_i[time_i < 0]
                time_i=time_i[time_i>-halfwindow]
                peak_num_probability[key]=len(time_i)/total_num
            
    fig, ax = plt.subplots(figsize=(8, 6))
    MakePlots.plot_bar_from_dict(ax,peak_num_probability,plotScatter=False)
    # Add labels and title
    ax.set_ylabel('Number of Optical Peaks',fontsize=16)
    #ax.set_xlabel('Condition',fontsize=16)
    ax.set_title(f'Number of Optical Peaks within {halfwindow} on {side} side of the ripple peak',fontsize=16)
    plt.xticks(rotation=45, ha='right',fontsize=16)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    figName=f'Number of Optical Peaks_{halfwindow}_{side}_{LFP_channel}.png'
    fig.savefig(os.path.join(savepath,figName))
    plt.show()    
    return -1
    

def Ripple_Stat_by_State_Bar_plot(parent_folder,LFP_channel,filterOF=True):
    filename = os.path.join(parent_folder, 'ripple_freq'+LFP_channel+'.pkl')
    ripple_freq=load_pickle_files (filename)

    filename = os.path.join(parent_folder, 'ripple_numbers'+LFP_channel+'.pkl')
    ripple_numbers=load_pickle_files (filename)
    filename = os.path.join(parent_folder, 'ripple_std_values'+LFP_channel+'.pkl')
    ripple_std_values=load_pickle_files (filename)
    filename = os.path.join(parent_folder, 'ripple_duration_values'+LFP_channel+'.pkl')
    ripple_duration_values=load_pickle_files (filename)

    if filterOF:
        keys_to_plot = ['pre_sleep', 'pre_awake','post_sleep','post_awake']
        ripple_freq = {key: ripple_freq[key] for key in keys_to_plot if key in ripple_freq}
        ripple_numbers={key: ripple_numbers[key] for key in keys_to_plot if key in ripple_numbers}
        ripple_std_values = {key: ripple_std_values[key] for key in keys_to_plot if key in ripple_std_values}
        ripple_duration_values={key: ripple_duration_values[key] for key in keys_to_plot if key in ripple_duration_values}
        
    savepath = os.path.join(parent_folder, "ResultsPooled")
    fig, ax = plt.subplots(figsize=(8, 6))
    MakePlots.plot_bar_from_dict(ax,ripple_freq,plotScatter=False)
    # Add labels and title
    ax.set_ylabel('Ripple Frequency (events/second)',fontsize=16)
    #ax.set_xlabel('Condition',fontsize=16)
    ax.set_title('Ripple Frequency by Condition',fontsize=16)
    plt.xticks(rotation=45, ha='right',fontsize=16)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    figName=f'Ripple Frequency by Condition_{LFP_channel}.png'
    fig.savefig(os.path.join(savepath,figName))
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    MakePlots.plot_bar_from_dict(ax,ripple_numbers,plotScatter=False)
    # Add labels and title
    ax.set_ylabel('Ripple numbers in the 3-mins trial',fontsize=16)
    #ax.set_xlabel('Condition',fontsize=16)
    ax.set_title('Ripple number by Condition',fontsize=16)
    plt.xticks(rotation=45, ha='right',fontsize=16)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    figName=f'Ripple Numbers by Condition_{LFP_channel}.png'
    fig.savefig(os.path.join(savepath,figName))
    plt.show()
    
    for key in ripple_std_values:
        print (key)
        if (ripple_std_values[key]):
            ripple_std_values[key] = np.concatenate(ripple_std_values[key])
    fig, ax = plt.subplots(figsize=(8, 6))
    MakePlots.plot_bar_from_dict(ax,ripple_std_values,plotScatter=False)
    # Add labels and title
    ax.set_ylabel('Ripple Amplitude (std)',fontsize=16)
    #ax.set_xlabel('Condition',fontsize=16)
    ax.set_title('Ripple Amplitude by Condition',fontsize=16)
    plt.xticks(rotation=45, ha='right',fontsize=16)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    figName=f'Ripple Amplitude by Condition_{LFP_channel}.png'
    fig.savefig(os.path.join(savepath,figName))
    plt.show()
    
    for key in ripple_std_values:
        print (key)
        if ripple_duration_values[key]:
            ripple_duration_values[key] = np.concatenate(ripple_duration_values[key])
    fig, ax = plt.subplots(figsize=(8, 6))
    MakePlots.plot_bar_from_dict(ax,ripple_duration_values,plotScatter=False)
    # Add labels and title
    ax.set_ylabel('Ripple Duration (ms)',fontsize=16)
    #ax.set_xlabel('Condition',fontsize=16)
    ax.set_title('Ripple Duration by Condition',fontsize=16)
    plt.xticks(rotation=45, ha='right',fontsize=16)  # Rotate x-axis labels for better visibility
    plt.tight_layout()
    figName=f'Ripple Duration by Condition_{LFP_channel}.png'
    fig.savefig(os.path.join(savepath,figName))
    plt.show()
    
    return ripple_freq,ripple_numbers,ripple_std_values
#%%
parent_folder='F:/2024MScR_NORtask/1765010_PVGCaMP8f_Atlas/'
SeparateTrialsByStateAndSave (parent_folder,LFP_channel='LFP_2')
#%%
filename = os.path.join(parent_folder, 'ripple_triggered_optical_peak_time_'+'LFP_2'+'.pkl')
time_dict=load_pickle_files (filename)

PoolDatabyStateAndPlot (parent_folder, 'LFP_2', mode='ripple')
#%%
ripple_freq,ripple_numbers,ripple_std_values=Ripple_Stat_by_State_Bar_plot(parent_folder,'LFP_2')
#%%
Compare_OpticalPeak_RipplePeak (parent_folder, 'LFP_2',side='after', halfwindow=0.1, mode='theta')















