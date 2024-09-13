# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:34:58 2024

@author: Yifang
"""
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from SPADPhotometryAnalysis import SPADAnalysisTools as OpticalAnlaysis
from SPADPhotometryAnalysis import AtlasDecode
from SPADPhotometryAnalysis import photometry_functions as fp


Fs=840
#parent_path='G:/ATLAS_SPAD/SNR_PSD_30x30/'
parent_path='D:/ATLAS_SPAD/YellowPaperSNR_PSD_30x30/'
pattern = os.path.join(parent_path, '*uW*/')
file_list = glob.glob(pattern)
lambd = 10e3 # Adjust lambda to get the best fit
porder = 1
itermax = 15

fig, ax = plt.subplots(1, 1, figsize=(5,10))
for path in file_list:
    tag = os.path.basename(os.path.normpath(path))
    print (tag)
    raw_trace = pd.read_csv(os.path.join(path, 'Green_traceAll.csv'), header=None).squeeze().values
    #zscore = pd.read_csv(os.path.join(path, 'Zscore_traceAll.csv'), header=None).squeeze().values
    sig_base=fp.airPLS(raw_trace,lambda_=lambd,porder=porder,itermax=itermax) 
    noise = (raw_trace - sig_base)
    relatevie_noise=noise/sig_base
    
    signal_trace = (raw_trace - sig_base)  
    z_score=(signal_trace - np.median(signal_trace)) / np.std(signal_trace)
    fluctuation = ((raw_trace - sig_base)/sig_base)*100
    
    # fig, ax = plt.subplots(figsize=(8,2)) #(8,2)
    # AtlasDecode.plot_trace(raw_trace,ax, Fs, label="raw_data")
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'raw_data.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(figsize=(8,2))
    # AtlasDecode.plot_trace(sig_base,ax, Fs, label="sig_base")
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'sig_base.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(figsize=(8,2))
    # AtlasDecode.plot_trace(noise,ax, Fs, label="noise")
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'noise.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(figsize=(8,2))
    # AtlasDecode.plot_trace(z_score,ax, Fs, label="z_score")
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'z_score.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(1, 1, figsize=(3,6)) ##(3,6)
    # OpticalAnlaysis.PSD_plot (raw_trace,fs=Fs,method="welch",color='black', xlim=[0,200],linewidth=1,linestyle='-',label='raw_trace',ax=ax)
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'raw_trace_PSD.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(1, 1, figsize=(3,6))
    # OpticalAnlaysis.PSD_plot (z_score,fs=Fs,method="welch",color='tab:green', xlim=[0,200],linewidth=1,linestyle='-',label='zscore',ax=ax)
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'zscore_PSD.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(1, 1, figsize=(3,6))
    # OpticalAnlaysis.PSD_plot (noise,fs=Fs,method="welch",color='tab:blue', xlim=[0,200],linewidth=1,linestyle='-',label='noise',ax=ax)
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'noise_PSD.png')
    # fig.savefig(fig_path, transparent=False)
    
    # fig, ax = plt.subplots(1, 1, figsize=(3,6))
    # fig, ax,AA,BB=OpticalAnlaysis.PSD_plot (relatevie_noise,fs=Fs,method="welch",color='tab:orange', xlim=[0,200],linewidth=1,linestyle='-',label='relative noise',ax=ax)
    # ax.set_title(tag)
    # fig_path = os.path.join(path, tag+'relativeNoise_PSD.png')
    # fig.savefig(fig_path, transparent=False)
    
    
    sns.kdeplot(fluctuation, common_norm=True, ax=ax, label=tag, linewidth=2,alpha=0.9)
    # Add a legend
    legend=ax.legend()
    legend.get_frame().set_alpha(0.1)
    # Add a title
    fig_path = os.path.join(path, tag+'noise_density.png')
    fig.savefig(fig_path, transparent=False)

#%%