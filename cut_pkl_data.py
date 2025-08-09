# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 19:53:00 2025

@author: yifan
"""

import pandas as pd
import os

# Set the folder path and filename
folder_path = r'G:\2024_OEC_Atlas_main\1765508_Jedi2p_pyPhotometry\Day2\SyncRecording4'
file_name = "Ephys_tracking_photometry_aligned.pkl"
file_path = os.path.join(folder_path, file_name)

# Load the original .pkl file
df = pd.read_pickle(file_path)

# Cut the first 30 seconds of data (at 10,000 Hz sampling rate)
sampling_rate = 10000
duration_sec = 30
num_samples = sampling_rate * duration_sec
df_30s = df.iloc[300000:300000+num_samples]

# Save to new .pkl file
output_name = "Ephys_tracking_photometry_30s.pkl"
output_path = os.path.join(folder_path, output_name)
df_30s.to_pickle(output_path)

print(f"Trimmed data saved to: {output_path}")