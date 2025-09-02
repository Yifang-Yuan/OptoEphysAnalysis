# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:58:43 2024

@author: Yifang
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import OpenEphysTools as OE
import os
import pandas as pd


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
    #ax.axvline(x=0, color='tomato', label=f'{mode} Peak')
    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('z-score')
    # Set legend location
    #ax.legend(loc='upper right')
    return ax


def plot_oscillation_epoch_optical_peaks(ax,x_value,optical_peak_times,optical_peak_values,mean_LFP,std_LFP,CI_LFP,half_window,mode='ripple',plotShade='CI'):
    # Plot peak values
    ax.scatter(optical_peak_times, optical_peak_values, color='limegreen', label='Optical-peak',s=8)
    # Plot mean LFP
    ax.plot(x_value, mean_LFP, color='royalblue', label='Mean LFP')
    # Plot shaded regions
    plotShade = 'CI'  # Choose 'std' or 'CI'
    if plotShade == 'std':
        ax.fill_between(x_value, mean_LFP - std_LFP, mean_LFP + std_LFP, color='dodgerblue', alpha=0.1, label='Std')
    elif plotShade == 'CI':
        ax.fill_between(x_value, CI_LFP[0], CI_LFP[1], color='dodgerblue', alpha=0.1, label='0.95 CI')
    # Add vertical line
    #ax.axvline(x=0, color='tomato', label='Ripple Peak')
    # Set labels and title
    ax.set_xlabel('Peak Time')
    ax.set_ylabel('Normalised signal')
    ax.set_xlim(-half_window, half_window)
    # Set legend location
    #ax.legend(loc='lower right')

    return ax

def plot_bar_from_dict(ax,data_dict,plotScatter=True):
    # Extract labels and data
    labels = list(data_dict.keys())
    data = [data_dict[label] for label in labels]
    # Calculate the mean and standard deviation for each group
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]
    # Plotting
    # Plot bars with error bars
    ax.bar(labels, means, yerr=stds, capsize=8,alpha=0.3,color='black',)
    # Add scatter dots
    if plotScatter:
        for i, label in enumerate(labels):
            x = [i] * len(data_dict[label])  # x-coordinate for scatter points
            ax.scatter(x, data_dict[label], color='tomato', zorder=5)  # zorder to ensure scatter dots are on top of bars
    return ax


'Plots for multiple ROI'
def _slice_df_by_time(df: pd.DataFrame, fs: float, start_time: float, end_time: float) -> pd.DataFrame:
    """Slice by seconds. Uses 'time' col if present (in seconds), else index->iloc via fs."""
    if 'time' in df.columns:
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        return df.loc[mask]
    # fall back to index-based slicing
    start_idx = int(start_time * fs)
    end_idx   = int(end_time   * fs)
    end_idx   = max(end_idx, start_idx + 1)
    return df.iloc[start_idx:end_idx]

def plot_segment_feature_multiROI_independent(
        df_aligned: pd.DataFrame,
        fs: float,
        savepath: str,
        LFP_channel: str,
        start_time: float,
        end_time: float,
        SPAD_cutoff: float,
        lfp_cutoff: float,
        label_fs: int = 18,
        tick_fs: int = 16,
        text_fs: int = 16,
        five_xticks: bool = False):
    """
    Standalone version of plot_segment_feature_multiROI:
    Plots trace + wavelet for sig, ref, zscore and LFP.

    Required columns in df_aligned:
      - 'sig_raw', 'ref_raw', 'zscore_raw', and LFP_channel (e.g. 'LFP_1')
      - optional 'time' column (seconds). If absent, slices by index using fs.
    """
    # Slice the window
    data = _slice_df_by_time(df_aligned, fs, start_time, end_time)

    # --- Smooth / filter photometry (signal / control / zscore) --------------
    if 'sig_raw' not in data.columns or 'ref_raw' not in data.columns:
        raise ValueError("df_aligned must contain 'sig_raw' and 'ref_raw'.")
    if 'zscore_raw' not in data.columns:
        raise ValueError("df_aligned must contain 'zscore_raw' for z_smooth.")

    sig_smooth = OE.smooth_signal(data['sig_raw'], Fs=fs, cutoff=SPAD_cutoff)
    sig_smooth = OE.butter_filter(sig_smooth, btype='high', cutoff=2.5, fs=fs, order=3)

    ref_smooth = OE.smooth_signal(data['ref_raw'], Fs=fs, cutoff=SPAD_cutoff)
    ref_smooth = OE.butter_filter(ref_smooth, btype='high', cutoff=2.5, fs=fs, order=3)

    z_smooth = OE.smooth_signal(data['zscore_raw'], Fs=fs, cutoff=SPAD_cutoff)
    z_smooth = OE.butter_filter(z_smooth, btype='high', cutoff=2.5, fs=fs, order=3)

    # --- LFP -----------------------------------------------------------------
    if LFP_channel not in data.columns:
        raise ValueError(f"df_aligned must contain LFP channel '{LFP_channel}'.")
    lfp_lowpass = OE.butter_filter(data[LFP_channel], btype='low', cutoff=lfp_cutoff, fs=fs, order=5)
    lfp_lowpass = OE.butter_filter(lfp_lowpass, btype='high', cutoff=2.5, fs=fs, order=3)

    # Wrap as Series to preserve index for downstream plotting
    signal_data = pd.Series(sig_smooth, index=data['sig_raw'].index)
    ref_data    = pd.Series(ref_smooth, index=data['ref_raw'].index)
    z_data      = pd.Series(z_smooth,   index=data['zscore_raw'].index)
    lfp_data    = pd.Series(lfp_lowpass, index=data[LFP_channel].index)

    # --- Figure: 8 rows (sig, sig-WT, ref, ref-WT, z, z-WT, LFP, LFP-WT) ----
    fig, ax = plt.subplots(8, 1, figsize=(12, 17))

    # control trace + wavelet
    OE.plot_trace_in_seconds_ax(ax[0], ref_data, fs, label='CA1_R',
                                color=sns.color_palette("husl", 8)[0],
                                ylabel='z-score', xlabel=False)
    sst, frequency, power, global_ws = OE.Calculate_wavelet(ref_data, lowpassCutoff=100, Fs=fs, scale=40)
    OE.plot_wavelet(ax[1], sst, frequency, power, Fs=fs, colorBar=False, logbase=False)
    
    # signal trace + wavelet
    OE.plot_trace_in_seconds_ax(ax[2], signal_data, fs, label='CA1_L',
                                color=sns.color_palette("husl", 8)[3],
                                ylabel='z-score', xlabel=False)
    sst, frequency, power, global_ws = OE.Calculate_wavelet(signal_data, lowpassCutoff=100, Fs=fs, scale=40)
    OE.plot_wavelet(ax[3], sst, frequency, power, Fs=fs, colorBar=False, logbase=False)
    
    # z_smooth trace + wavelet
    OE.plot_trace_in_seconds_ax(ax[4], z_data, fs, label='CA3_L',
                                color=sns.color_palette("husl", 8)[2],
                                ylabel='z-score', xlabel=False)
    sst, frequency, power, global_ws = OE.Calculate_wavelet(z_data, lowpassCutoff=100, Fs=fs, scale=40)
    OE.plot_wavelet(ax[5], sst, frequency, power, Fs=fs, colorBar=False, logbase=False)

    # LFP (to mV) + wavelet
    lfp_mv = lfp_data / 1000.0
    OE.plot_trace_in_seconds_ax(ax[6], lfp_mv, fs, label='LFP',
                                color=sns.color_palette("husl", 8)[5],
                                ylabel='mV', xlabel=False)
    sst, frequency, power, global_ws = OE.Calculate_wavelet(lfp_mv, lowpassCutoff=500, Fs=fs, scale=40)
    OE.plot_wavelet(ax[7], sst, frequency, power, Fs=fs, colorBar=False, logbase=False)

    # Wavelet y-lims
    for wl_ax in (ax[1], ax[3], ax[5], ax[7]):
        wl_ax.set_ylim(0, 20)

    # Axis cosmetics
    ax[7].set_xlabel('Time (seconds)', fontsize=label_fs)

    # Hide legends (keep handles)
    # for i in (0, 2, 4, 6):
    #     leg = ax[i].legend()
    #     leg.set_visible(False)

    # Remove spines on wavelets & hide x-ticks where appropriate
    for a in (ax[1], ax[2], ax[3], ax[4], ax[5], ax[6]):
        a.set_xticks([])
        a.set_xlabel('')
    for wl_ax in (ax[1], ax[3], ax[5], ax[7]):
        wl_ax.spines['top'].set_visible(False)
        wl_ax.spines['right'].set_visible(False)
        wl_ax.spines['bottom'].set_visible(False)
        wl_ax.spines['left'].set_visible(False)

    # Scale fonts
    for a in ax:
        if a.get_ylabel():
            a.set_ylabel(a.get_ylabel(), fontsize=label_fs)
        a.tick_params(axis='both', labelsize=tick_fs, width=1.2)

    # Optional: exactly five x-ticks on bottom axis
    if five_xticks:
        lo, hi = ax[7].get_xlim()
        ticks = np.linspace(lo, hi, 5)
        ticks[np.isclose(ticks, 0.0)] = 0.0
        ax[7].set_xticks(ticks)
        ax[7].set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=tick_fs)

    # Save
    makefigure_path = os.path.join(savepath if savepath else os.getcwd(), 'makefigure')
    os.makedirs(makefigure_path, exist_ok=True)
    output_path = os.path.join(makefigure_path, 'example_trace_powerspectral.png')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
    plt.show()

    return fig, ax, output_path


def plot_segment_feature_multiROI_twoLFP(
        df_aligned: pd.DataFrame,
        fs: float,
        savepath: str,
        LFP_channels,                     # <- NEW: tuple/list of TWO channels, e.g. ('LFP_2','LFP_3')
        start_time: float,
        end_time: float,
        SPAD_cutoff: float,
        lfp_cutoff: float,
        label_fs: int = 18,
        tick_fs: int = 16,
        leg_fs: int = 16,                # <- NEW: legend fontsize
        five_xticks: bool = False):
    """
    Plot sig/ref/zscore traces+wavelets and TWO LFP channels (each trace+wavelet).
    Output: (fig, ax, output_path)

    Requires columns: 'sig_raw','ref_raw','zscore_raw', and both LFP channels.
    """

    # --- validate LFPs ---
    if isinstance(LFP_channels, (str,)):
        raise ValueError("LFP_channels must be a tuple/list of TWO channel names, e.g. ('LFP_2','LFP_3').")
    if len(LFP_channels) != 2:
        raise ValueError("Provide exactly TWO LFP channel names.")
    lfp1, lfp2 = LFP_channels

    # Slice the window
    data = _slice_df_by_time(df_aligned, fs, start_time, end_time)

    # --- Smooth/filter photometry ------------------------------------------------
    for col in ('sig_raw','ref_raw','zscore_raw'):
        if col not in data.columns:
            raise ValueError(f"df_aligned must contain '{col}'.")

    sig_smooth = OE.smooth_signal(data['sig_raw'],  Fs=fs, cutoff=SPAD_cutoff)
    sig_smooth = OE.butter_filter(sig_smooth, btype='high', cutoff=2.5, fs=fs, order=3)

    ref_smooth = OE.smooth_signal(data['ref_raw'],  Fs=fs, cutoff=SPAD_cutoff)
    ref_smooth = OE.butter_filter(ref_smooth, btype='high', cutoff=2.5, fs=fs, order=3)

    z_smooth   = OE.smooth_signal(data['zscore_raw'], Fs=fs, cutoff=SPAD_cutoff)
    z_smooth   = OE.butter_filter(z_smooth,   btype='high', cutoff=2.5, fs=fs, order=3)

    # --- LFPs --------------------------------------------------------------------
    for lfp_col in (lfp1, lfp2):
        if lfp_col not in data.columns:
            raise ValueError(f"df_aligned must contain LFP channel '{lfp_col}'.")
    lfp1_f = OE.butter_filter(data[lfp1], btype='low',  cutoff=lfp_cutoff, fs=fs, order=5)
    lfp1_f = OE.butter_filter(lfp1_f,     btype='high', cutoff=2.5,        fs=fs, order=3)
    lfp2_f = OE.butter_filter(data[lfp2], btype='low',  cutoff=lfp_cutoff, fs=fs, order=5)
    lfp2_f = OE.butter_filter(lfp2_f,     btype='high', cutoff=2.5,        fs=fs, order=3)

    # Wrap as Series (preserve index)
    sig_s  = pd.Series(sig_smooth, index=data['sig_raw'].index)
    ref_s  = pd.Series(ref_smooth, index=data['ref_raw'].index)
    z_s    = pd.Series(z_smooth,   index=data['zscore_raw'].index)
    lfp1_s = pd.Series(lfp1_f,     index=data[lfp1].index)
    lfp2_s = pd.Series(lfp2_f,     index=data[lfp2].index)

    # --- Figure: 10 rows (ref,WT | sig,WT | z,WT | LFP1,WT | LFP2,WT) -----------
    fig, ax = plt.subplots(10, 1, figsize=(15, 20), constrained_layout=False)

    pal = sns.color_palette("husl", 8)
    color_ref = pal[0]
    color_sig = pal[3]
    color_z   = pal[2]
    color_l1  = pal[5]
    color_l2  = pal[6]

    # Row map
    ROW = {
        'ref':  (0,1),
        'sig':  (2,3),
        'z':    (4,5),
        'lfp1': (6,7),
        'lfp2': (8,9),
    }

    # --- ref trace + wavelet
    OE.plot_trace_in_seconds_ax(ax[ROW['ref'][0]], ref_s, fs, label='CA1_R',
                                color=color_ref, ylabel='z-score', xlabel=False)
    leg = ax[ROW['ref'][0]].legend(fontsize=leg_fs, loc='upper right'); leg.set_visible(True)
    sst,freq,powr,glob = OE.Calculate_wavelet(ref_s, lowpassCutoff=100, Fs=fs, scale=40)
    OE.plot_wavelet(ax[ROW['ref'][1]], sst, freq, powr, Fs=fs, colorBar=False, logbase=False)

    # --- sig trace + wavelet
    OE.plot_trace_in_seconds_ax(ax[ROW['sig'][0]], sig_s, fs, label='CA1_L',
                                color=color_sig, ylabel='z-score', xlabel=False)
    leg = ax[ROW['sig'][0]].legend(fontsize=leg_fs, loc='upper right'); leg.set_visible(True)
    sst,freq,powr,glob = OE.Calculate_wavelet(sig_s, lowpassCutoff=100, Fs=fs, scale=40)
    OE.plot_wavelet(ax[ROW['sig'][1]], sst, freq, powr, Fs=fs, colorBar=False, logbase=False)

    # --- z trace + wavelet
    OE.plot_trace_in_seconds_ax(ax[ROW['z'][0]], z_s, fs, label='CA3_L',
                                color=color_z, ylabel='z-score', xlabel=False)
    leg = ax[ROW['z'][0]].legend(fontsize=leg_fs, loc='upper right'); leg.set_visible(True)
    sst,freq,powr,glob = OE.Calculate_wavelet(z_s, lowpassCutoff=100, Fs=fs, scale=40)
    OE.plot_wavelet(ax[ROW['z'][1]], sst, freq, powr, Fs=fs, colorBar=False, logbase=False)

    # --- LFP1 trace + wavelet (to mV)
    lfp1_mv = lfp1_s / 1000.0
    OE.plot_trace_in_seconds_ax(ax[ROW['lfp1'][0]], lfp1_mv, fs, label=lfp1,
                                color=color_l1, ylabel='mV', xlabel=False)
    leg = ax[ROW['lfp1'][0]].legend(fontsize=leg_fs, loc='upper right'); leg.set_visible(True)
    sst,freq,powr,glob = OE.Calculate_wavelet(lfp1_mv, lowpassCutoff=500, Fs=fs, scale=40)
    OE.plot_wavelet(ax[ROW['lfp1'][1]], sst, freq, powr, Fs=fs, colorBar=False, logbase=False)

    # --- LFP2 trace + wavelet (to mV)
    lfp2_mv = lfp2_s / 1000.0
    OE.plot_trace_in_seconds_ax(ax[ROW['lfp2'][0]], lfp2_mv, fs, label=lfp2,
                                color=color_l2, ylabel='mV', xlabel=False)
    leg = ax[ROW['lfp2'][0]].legend(fontsize=leg_fs, loc='upper right'); leg.set_visible(True)
    sst,freq,powr,glob = OE.Calculate_wavelet(lfp2_mv, lowpassCutoff=500, Fs=fs, scale=40)
    OE.plot_wavelet(ax[ROW['lfp2'][1]], sst, freq, powr, Fs=fs, colorBar=False, logbase=False)

    # Wavelet y-lims
    for wl_ax in (ax[1], ax[3], ax[5], ax[7], ax[9]):
        wl_ax.set_ylim(0, 20)

    # Axis cosmetics
    for a in (ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7], ax[8]):
        a.set_xticks([]); a.set_xlabel('')
    for wl_ax in (ax[1], ax[3], ax[5], ax[7], ax[9]):
        for side in ('top','right','bottom','left'):
            wl_ax.spines[side].set_visible(False)

    # Scale fonts
    for a in ax:
        if a.get_ylabel():
            a.set_ylabel(a.get_ylabel(), fontsize=label_fs)
        a.tick_params(axis='both', labelsize=tick_fs, width=1.2)

    # Bottom x-label & optional 5 ticks
    ax[9].set_xlabel('Time (seconds)', fontsize=label_fs)
    if five_xticks:
        lo, hi = ax[9].get_xlim()
        ticks = np.linspace(lo, hi, 5)
        ticks[np.isclose(ticks, 0.0)] = 0.0
        ax[9].set_xticks(ticks)
        ax[9].set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=tick_fs)

    # Save
    makefigure_path = os.path.join(savepath if savepath else os.getcwd(), 'makefigure')
    os.makedirs(makefigure_path, exist_ok=True)
    output_path = os.path.join(makefigure_path, 'example_trace_powerspectral_twoLFP.png')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
    plt.show()

    return fig, ax, output_path