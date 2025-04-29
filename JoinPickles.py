# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:58:27 2025

@author: yifan
"""

import os
import re
import pickle
import numpy as np

def concatenate_theta_pkls(dpath):
    """
    Reads and concatenates three types of pkl files (aligned_theta_LFP, bandpass_LFP, Zscore)
    from subfolders of dpath beginning with 'Theta1' to 'Theta9'. 
    Saves the concatenated results in dpath.
    """
    # Define expected file names
    filenames = [
        "ailgned_theta_LFP.pkl",
        "ailgned_theta_bandpass_LFP.pkl",
        "ailgned_theta_Zscore.pkl"
    ]

    # Storage for each file type
    data_store = {name: [] for name in filenames}

    # Regex pattern for folder names like 'Theta1', 'Theta2', ..., optionally with suffix
    pattern = re.compile(r"^Theta[1-9]\w*")

    # List valid subdirectories
    subfolders = [f for f in os.listdir(dpath)
                  if os.path.isdir(os.path.join(dpath, f)) and pattern.match(f)]

    print(f"Found {len(subfolders)} matching folders.")

    for folder in subfolders:
        folder_path = os.path.join(dpath, folder)
        for fname in filenames:
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path):
                with open(full_path, 'rb') as f:
                    arr = pickle.load(f)
                    data_store[fname].append(arr)
            else:
                print(f"Warning: File not found - {full_path}")

    # Concatenate and save
    for fname in filenames:
        if data_store[fname]:
            concatenated = np.concatenate(data_store[fname], axis=0)
            save_path = os.path.join(dpath, fname)
            with open(save_path, 'wb') as f:
                pickle.dump(concatenated, f)
            print(f"Saved: {save_path} with shape {concatenated.shape}")
        else:
            print(f"No data found for {fname}, skipping save.")
            
            
def concatenate_ripple_pkls(dpath):
    """
    Reads and concatenates three types of pkl files (aligned_theta_LFP, bandpass_LFP, Zscore)
    from subfolders of dpath beginning with 'Theta1' to 'Theta9'. 
    Saves the concatenated results in dpath.
    """
    # Define expected file names
    filenames = [
        "ailgned_ripple_LFP.pkl",
        "ailgned_ripple_bandpass_LFP.pkl",
        "ailgned_ripple_Zscore.pkl"
    ]

    # Storage for each file type
    data_store = {name: [] for name in filenames}

    # Regex pattern for folder names like 'Theta1', 'Theta2', ..., optionally with suffix
    pattern = re.compile(r"^Ripple[1-9]\w*")

    # List valid subdirectories
    subfolders = [f for f in os.listdir(dpath)
                  if os.path.isdir(os.path.join(dpath, f)) and pattern.match(f)]

    print(f"Found {len(subfolders)} matching folders.")

    for folder in subfolders:
        folder_path = os.path.join(dpath, folder)
        for fname in filenames:
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path):
                with open(full_path, 'rb') as f:
                    arr = pickle.load(f)
                    data_store[fname].append(arr)
            else:
                print(f"Warning: File not found - {full_path}")

    # Concatenate and save
    for fname in filenames:
        if data_store[fname]:
            concatenated = np.concatenate(data_store[fname], axis=0)
            save_path = os.path.join(dpath, fname)
            with open(save_path, 'wb') as f:
                pickle.dump(concatenated, f)
            print(f"Saved: {save_path} with shape {concatenated.shape}")
        else:
            print(f"No data found for {fname}, skipping save.")
# Example usage
dpath = "F:/2025_ATLAS_SPAD/Figure3_Pyr_ripple/RippleDataAll/"
#concatenate_theta_pkls(dpath)
concatenate_ripple_pkls(dpath)