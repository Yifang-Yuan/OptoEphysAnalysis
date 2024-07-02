# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:09:52 2024

@author: Yang
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from SPADPhotometryAnalysis import photometry_functions as fp
import numpy as np
# Path to the folder containing the CSV files
folder_path = 'G:/pyPhotometry_data/chemoValidation/'

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Sort files by creation time
csv_files.sort(key=os.path.getctime)
sampling_rate=130
# Plot and save figures
for idx, file in enumerate(csv_files):
    # Read the CSV file
    print (file)
    PhotometryData=pd.read_csv(file)
    raw_reference = PhotometryData[' Analog2'][1:]
    raw_signal = PhotometryData['Analog1'][1:]
    smooth_win = 10
    smooth_reference = fp.smooth_signal(raw_reference, smooth_win)
    smooth_signal = fp.smooth_signal(raw_signal, smooth_win)
    
    lambd = 10e4 # Adjust lambda to get the best fit
    porder = 1
    itermax = 15
    
    r_base=fp.airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
    s_base=fp.airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)
    
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(211)
    ax1 = fp.plotSingleTrace (ax1, smooth_signal, SamplingRate=sampling_rate,color='blue',Label='smooth_signal') 
    ax1 = fp.plotSingleTrace (ax1, s_base, SamplingRate=sampling_rate,color='black',Label='baseline_signal',linewidth=2) 
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2 = fig.add_subplot(212)
    ax2 = fp.plotSingleTrace (ax2, smooth_reference, SamplingRate=sampling_rate,color='purple',Label='smooth_reference')
    ax2 = fp.plotSingleTrace (ax2, r_base, SamplingRate=sampling_rate,color='black',Label='baseline_reference',linewidth=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig.suptitle(f'Smoothed Signal-Trial {idx + 1}', fontsize=16)
    plt.tight_layout()
    # Save the plot
    save_path = os.path.join(folder_path, f'trial{idx + 1}_smoothed.png')
    plt.savefig(save_path)
    plt.show()

    # Create a plot
    remove=0
    reference = (smooth_reference[remove:] - r_base[remove:])
    signal = (smooth_signal[remove:] - s_base[remove:])  
    z_reference = (reference - np.median(reference)) / np.std(reference)
    z_signal = (signal - np.median(signal)) / np.std(signal)
    from sklearn.linear_model import Lasso
    lin = Lasso(alpha=0.001,precompute=True,max_iter=1000,
                positive=True, random_state=9999, selection='random')
    n = len(z_reference)
    '''Need to change to numpy if previous smooth window is 1'''
    # z_signal=z_signal.to_numpy()
    # z_reference=z_reference.to_numpy()
    lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))

    z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)
    zdFF = (z_signal - z_reference_fitted)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax1 = fp.plotSingleTrace (ax1, zdFF, SamplingRate=sampling_rate,color='black',Label='zscore_signal')
    # Add title and labels
    plt.title(f'Zscore Signal-Trial {idx + 1}')
    # Save the plot
    save_path = os.path.join(folder_path, f'trial{idx + 1}_zscore.png')
    plt.savefig(save_path)
    plt.show()