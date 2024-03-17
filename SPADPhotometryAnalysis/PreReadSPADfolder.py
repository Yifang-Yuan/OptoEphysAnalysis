# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:02:38 2024
@author: Yifang
Batch process multiple SPAD recording folders in an experiment.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SPADPhotometryAnalysis import SPADAnalysisTools as Analysis
from SPADPhotometryAnalysis import SPADdemod
from SPADPhotometryAnalysis import SPADreadBin
import os

'xxRange=[0,160],yyRange=[40,200]'

def read_multiple_SPAD_bin_folder(parent_folder,xxRange,yyRange):
    '''When using this batch processing function, please make sure the ROI did not change for this whole experiment.'''
    # Get a list of all directories in the parent folder
    directories = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    # Sort the directories by creation time
    directories.sort(key=lambda x: os.path.getctime(x))
    # Read the directories in sorted order
    for directory in directories:
        print("Folder:", directory)
        SPADreadBin.readMultipleBinfiles(directory,9,xxRange=xxRange,yyRange=yyRange)
    return -1


parent_folder="D:/2024MScR_NORtask/1732333_SPAD_Day1/"
read_multiple_SPAD_bin_folder(parent_folder,xxRange=[0,160],yyRange=[40,200])
parent_folder="D:/2024MScR_NORtask/1732333_SPAD_Day2/"
read_multiple_SPAD_bin_folder(parent_folder,xxRange=[0,160],yyRange=[40,200])
parent_folder="D:/2024MScR_NORtask/1732333_SPAD_Day3/"
read_multiple_SPAD_bin_folder(parent_folder,xxRange=[0,160],yyRange=[40,200])
parent_folder="D:/2024MScR_NORtask/1732333_SPAD_Day4/"
read_multiple_SPAD_bin_folder(parent_folder,xxRange=[0,160],yyRange=[40,200])
parent_folder="D:/2024MScR_NORtask/1732333_SPAD_Day5/"
read_multiple_SPAD_bin_folder(parent_folder,xxRange=[0,160],yyRange=[40,200])