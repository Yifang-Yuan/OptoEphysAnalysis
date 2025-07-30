# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 22:01:38 2025

@author: yifan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from SPADPhotometryAnalysis import AtlasDecode
from scipy.signal import spectrogram, find_peaks

def notchfilter (data,f0=50,bw=10,fs=30000):
    # Bandwidth of the notch filter (in Hz)   
    Q = f0/bw # Quality factor
    b, a = signal.iirnotch(f0, Q, fs)
    data=signal.filtfilt(b, a, data)
    return data
def plot_power_spectrum(signal, sampling_rate, title='Power Spectrogram', nperseg=256, noverlap=128, fmax=252):
    """Plots power spectrogram of a signal with large labels for publication."""
    frequencies, times, Sxx = spectrogram(
        signal,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        mode='psd'
    )
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    
    cbar = plt.colorbar()
    cbar.set_label('Power [dB]', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.ylabel('Frequency [Hz]', fontsize=20)
    plt.xlabel('Time [s]', fontsize=20)
    plt.title(title, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, fmax)
    plt.tight_layout()
    plt.show()
def auto_cut_signal_by_sweep(raw_signal, sampling_rate, f_target=250, tolerance=20,
                              nperseg=1024, noverlap=512, min_peak_prominence=0.1):
    """
    Automatically detect and cut one full chirp cycle based on peaks in high-frequency power (around f_target Hz).
    """
    # Compute spectrogram
    f, t, Sxx = spectrogram(raw_signal, fs=sampling_rate,
                            nperseg=nperseg, noverlap=noverlap,
                            scaling='density', mode='psd')

    # Select frequency range around target
    freq_mask = (f >= f_target - tolerance) & (f <= f_target + tolerance)
    if not np.any(freq_mask):
        raise ValueError("Frequency mask is empty. Adjust f_target or tolerance.")

    # Average power in high-frequency band
    high_freq_power = Sxx[freq_mask, :].mean(axis=0)

    # Plot for inspection
    plt.figure(figsize=(10, 3))
    plt.plot(t, high_freq_power, label=f'Power near {f_target} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.title('High-Frequency Power Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Debug print
    print("Max high-freq power:", np.max(high_freq_power))
    print("Power threshold for peaks (80th percentile):", np.percentile(high_freq_power, 80))

    # Detect peaks
    threshold = np.percentile(high_freq_power, 80)  # More flexible than fixed 0.5*max
    peaks, props = find_peaks(high_freq_power, height=threshold, distance=20, prominence=min_peak_prominence)

    # Debug
    print(f"Found {len(peaks)} peaks at times: {t[peaks]}")

    if len(peaks) < 2:
        # Try fallback: lower threshold
        threshold_fallback = np.percentile(high_freq_power, 60)
        print(f"Trying fallback with lower threshold: {threshold_fallback}")
        peaks, props = find_peaks(high_freq_power, height=threshold_fallback, distance=20)

    if len(peaks) < 2:
        raise ValueError("Couldn't find two distinct high-frequency peaks — try lowering threshold or check signal.")

    # Use first two peak times
    t1, t2 = t[peaks[0]], t[peaks[1]]
    start_idx = int(t1 * sampling_rate)
    end_idx = int(t2 * sampling_rate)
    cut_signal = raw_signal[start_idx:end_idx]
    cut_time = np.arange(start_idx, end_idx) / sampling_rate

    # Plot cut signal
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(cut_time, cut_signal, color='green', linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Raw Signal', fontsize=20)
    ax.set_title(f'Auto-cut Signal from {t1:.2f}s to {t2:.2f}s', fontsize=22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    plt.show()

    return cut_signal

def cut_signal_by_time(raw_signal, sampling_rate):
    """Prompts user to enter start/end time and returns sliced signal + time vector with clean plotting."""
    start_time = float(input("Enter start time (in seconds): "))
    end_time = float(input("Enter end time (in seconds): "))
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    cut_signal = raw_signal[start_idx:end_idx]
    cut_time = np.arange(start_idx, end_idx) / sampling_rate

    # Clean and minimal plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cut_time, cut_signal, color='green', linewidth=2)

    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Raw Signal', fontsize=20)
    ax.set_title(f'Signal from {start_time}s to {end_time}s', fontsize=22)

    # Remove top and right spines for clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Keep left and bottom spines slightly thicker
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Tick font size
    ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    plt.show()

    return cut_signal
#%%
'''For pyPhotometry imaging'''
# Folder with your files
# Modify it depending on where your file is located
folder =r"G:\2025_ATLAS_SPAD\ChirpSignal\Set1_noSPC\pyPhotometry/"
# File name
file_name = '50mVsweep-2025-07-05-135546.csv'
sampling_rate=1000
'''Read csv file and calculate zscore of the fluorescent signal'''
PhotometryData = pd.read_csv(folder+file_name,index_col=False) 
raw_signal = PhotometryData['Analog1'][1:]
#%%
'''ATLAS imaging'''
sampling_rate=1682.92
dpath = r'G:\2025_ATLAS_SPAD\ChirpSignal\Set2_notLinearSweep\ATLAS/'
recordingNum=1
sync_recording_str = f"SyncRecording{recordingNum}"
recording_path=file_path_optical=os.path.join(dpath,sync_recording_str)
file_path_optical=os.path.join(dpath,sync_recording_str, "Green_traceAll.csv")
trace_raw = pd.read_csv(file_path_optical, header=None)  # Adjust if there's a head
trace_raw = trace_raw.squeeze()
'fliter out 100Hz'
# raw_signal=notchfilter (trace_raw,f0=100,bw=10,fs=1682.92)
# raw_signal=notchfilter (raw_signal,f0=200,bw=5,fs=1682.92)
# raw_signal = pd.Series(raw_signal)
'Do not filter out 100Hz'
raw_signal = trace_raw
#%%
'''SPC imaging'''
sampling_rate=9938.4
dpath = r'G:\2025_ATLAS_SPAD\ChirpSignal\Set2_notLinearSweep\SPC/'
recordingNum=1
sync_recording_str = f"SyncRecording{recordingNum}"
recording_path=file_path_optical=os.path.join(dpath,sync_recording_str)
file_path_optical=os.path.join(dpath,sync_recording_str, "traceValueAll.csv")
trace_raw = pd.read_csv(file_path_optical, header=None)  # Adjust if there's a head
trace_raw = trace_raw.squeeze()
raw_signal = trace_raw
#%%
# Plot full signal
time_vector = raw_signal.index / sampling_rate
plt.figure(figsize=(10, 4))
plt.plot(time_vector, raw_signal, color='blue', linewidth=1)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Raw Signal', fontsize=12)
plt.title('Full Raw Signal', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot full spectrogram
plot_power_spectrum(raw_signal, sampling_rate, title='Spectrogram of Full Signal',
                    nperseg=256, noverlap=128, fmax=260)

# Automatically cut signal between two full sweep cycles
cut_sig=cut_signal_by_time(raw_signal, sampling_rate)
# cut_sig = auto_cut_signal_by_sweep(raw_signal, sampling_rate, f_target=200, tolerance=10)
#%%
# Plot trimmed spectrogram
plot_power_spectrum(cut_sig, sampling_rate, title='Spectrogram of Auto-Cut Signal',
                    nperseg=256, noverlap=128, fmax=252)