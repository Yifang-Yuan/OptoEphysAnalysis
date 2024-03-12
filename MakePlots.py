# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:58:43 2024

@author: Yifang
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_oscillation_epoch_traces(ax,x_value,data1_mean,data2_mean,data1_std,data2_std,data1_CI,data2_CI,mode='ripple',plotShade='CI'):
    ax.plot(x_value, data1_mean, color='limegreen', label='Mean z-score')
    ax.plot(x_value, data2_mean, color='royalblue', label='Mean LFP')
    plotShade = 'CI'  # Choose 'std' or 'CI'
    if plotShade == 'std':
        ax.fill_between(x_value, data1_mean - data1_std, data1_mean + data1_std, color='limegreen', alpha=0.2, label='Std')
        ax.fill_between(x_value, data2_mean - data2_std, data2_mean + data2_std, color='dodgerblue', alpha=0.1, label='Std')
    elif plotShade == 'CI':
        ax.fill_between(x_value, data1_CI[0], data1_CI[1], color='limegreen', alpha=0.2, label='0.95 CI')
        ax.fill_between(x_value, data2_CI[0], data2_CI[1], color='dodgerblue', alpha=0.1, label='0.95 CI')
    
    # Add vertical line
    ax.axvline(x=0, color='tomato', label=f'{mode} Peak')
    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('z-score')
    ax.set_title(f'Mean optical transient triggered by {mode} peak')
    # Set legend location
    ax.legend(loc='upper right')
    return ax


def plot_oscillation_epoch_optical_peaks(ax,x_value,optical_peak_times,optical_peak_values,mean_LFP,std_LFP,CI_LFP,half_window,mode='ripple',plotShade='CI'):
    # Plot peak values
    ax.scatter(optical_peak_times, optical_peak_values, color='limegreen', label='Optical-peak')
    # Plot mean LFP
    ax.plot(x_value, mean_LFP, color='royalblue', label='Mean LFP')
    # Plot shaded regions
    plotShade = 'CI'  # Choose 'std' or 'CI'
    if plotShade == 'std':
        ax.fill_between(x_value, mean_LFP - std_LFP, mean_LFP + std_LFP, color='dodgerblue', alpha=0.1, label='Std')
    elif plotShade == 'CI':
        ax.fill_between(x_value, CI_LFP[0], CI_LFP[1], color='dodgerblue', alpha=0.1, label='0.95 CI')
    # Add vertical line
    ax.axvline(x=0, color='tomato', label='Ripple Peak')
    # Set labels and title
    ax.set_xlabel('Peak Time')
    ax.set_ylabel('Normalised signal')
    ax.set_title(f'Optical peaks during {mode} Event')
    ax.set_xlim(-half_window, half_window)
    # Set legend location
    ax.legend(loc='lower right')

    return ax