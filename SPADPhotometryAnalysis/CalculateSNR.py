# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:15:07 2024
To calculate SNR of a simple paper test with pyPhotometry and SPAD

@author: Yifang
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import photometry_functions as fp
import scipy

def calculate_SNR_for_photometry_folder (parent_folder):
    # Iterate over all folders in the parent folder
    SNR_savename='pyPhotometry_SNR_timedivision.csv'        
    SNR_array = np.array([])
    all_files = os.listdir(parent_folder)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    print(csv_files)
    for csv_file in csv_files:
        print('111',csv_file)
        raw_signal,raw_reference=fp.read_photometry_data (parent_folder, csv_file, readCamSync=False,plot=True)
        SNR=calculate_SNR(raw_signal)
        SNR_array = np.append(SNR_array, SNR)
    csv_savename = os.path.join(parent_folder, SNR_savename)
    np.savetxt(csv_savename, SNR_array, delimiter=',')
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(SNR_array, marker='o', linestyle='-', color='b')
    plt.xlabel('Light Power (uW)')
    plt.ylabel('SNR')
    return -1

def calculate_SNR_for_SPAD_folder (parent_folder,mode='continuous'):
    # Iterate over all folders in the parent folder
    if mode=='continuous':
        csv_filename="traceValueAll.csv"
        SNR_savename='SPAD_SNR_continuous.csv'
    if mode=='timedivision':
        csv_filename="Green_traceAll.csv"
        SNR_savename='SPAD_SNR_timedivision.csv'        
    SNR_array = np.array([])
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            filename=Analysis.Set_filename (folder_path, csv_filename)
            Trace_raw=Analysis.getSignalTrace (filename, traceType='Constant',HighFreqRemoval=False,getBinTrace=False,bin_window=100)
            fig, ax = plt.subplots(figsize=(12, 2.5))
            Analysis.plot_trace(Trace_raw,ax, fs=9938.4, label="Full raw data trace")
            SNR=calculate_SNR(Trace_raw[0:9000])
            SNR_array = np.append(SNR_array, SNR)
            
    csv_savename = os.path.join(parent_folder, SNR_savename)
    np.savetxt(csv_savename, SNR_array, delimiter=',')
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(SNR_array, marker='o', linestyle='-', color='b')
    plt.xlabel('Light Power (uW)')
    plt.ylabel('SNR')
    plt.title('SPAD_SNR_timedivision')
    return -1

def calculate_SNR (data):
    sig_value=np.mean(data)
    noise_value=np.std(data)
    snr=sig_value**2/noise_value**2
    print ('SNR is', snr)
    return snr

