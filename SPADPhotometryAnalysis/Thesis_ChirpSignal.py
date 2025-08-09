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
def plot_signal_and_spectrogram_time(signal, sampling_rate, title='Power Spectrogram',
                                nperseg=256, noverlap=128, fmax=252):
    """
    Plots two subplots:
    - Left: Middle 2 seconds of the signal (bold line, left & bottom spines only).
    - Right: Full power spectrogram.
    """
    total_samples = len(signal)
    total_duration = total_samples / sampling_rate

    # Extract middle 2 seconds
    mid_start = int((total_duration / 2 - 0.5) * sampling_rate)
    mid_end = int((total_duration / 2 + 0.5) * sampling_rate)
    signal_mid = signal[mid_start:mid_end]
    time_mid = np.linspace(0, 1, len(signal_mid))

    # Prepare spectrogram
    frequencies, times, Sxx = spectrogram(signal, fs=sampling_rate,
                                          nperseg=nperseg, noverlap=noverlap,
                                          scaling='density', mode='psd')

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1]})

    # Left subplot: middle 2s signal
    axes[0].plot(time_mid, signal_mid, linewidth=2)
    axes[0].set_xlabel('Time [s]', fontsize=18)
    axes[0].set_ylabel('Photon count per pixel', fontsize=18)
    axes[0].tick_params(labelsize=18)
    axes[0].set_title('Middle 1-sec Signal', fontsize=18)

    # Remove top and right spines
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right subplot: full spectrogram
    pcm = axes[1].pcolormesh(times, frequencies, 10 * np.log10(Sxx),
                             shading='gouraud', cmap='viridis')
    axes[1].set_xlabel('Time [s]', fontsize=18)
    axes[1].set_ylabel('Frequency [Hz]', fontsize=18)
    axes[1].tick_params(labelsize=18)
    axes[1].set_title('Full Power Spectrogram', fontsize=18)
    axes[1].set_ylim(0, fmax)

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axes[1])
    cbar.set_label('Power [dB]', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    fig.suptitle(title, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
def plot_signal_and_spectrogram(signal, sampling_rate, title='Power Spectrogram',
                                nperseg=256, noverlap=128, fmax=252):
    """
    Plots two subplots:
    - Left: Middle 1 second of the signal (bold line, no x-axis).
    - Right: Full power spectrogram (no x-axis).
    """
    total_samples = len(signal)
    total_duration = total_samples / sampling_rate

    # Extract middle 1 second
    mid_start = int((total_duration / 2 - 0.5) * sampling_rate)
    mid_end = int((total_duration / 2 + 0.5) * sampling_rate)
    signal_mid = signal[mid_start:mid_end]
    time_mid = np.linspace(0, 1, len(signal_mid))

    # Prepare spectrogram
    frequencies, times, Sxx = spectrogram(signal, fs=sampling_rate,
                                          nperseg=nperseg, noverlap=noverlap,
                                          scaling='density', mode='psd')

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1]})

    # Left subplot: middle 1s signal
    axes[0].plot(time_mid, signal_mid, linewidth=2)
    #axes[0].set_ylabel('Photon count per pixel', fontsize=18)
    axes[0].set_ylabel('Total Photon Count', fontsize=18)
    axes[0].tick_params(labelsize=18)
    axes[0].set_title('Middle 1-sec Signal', fontsize=18)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Right subplot: full spectrogram
    pcm = axes[1].pcolormesh(times, frequencies, 10 * np.log10(Sxx),
                             shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency [Hz]', fontsize=18)
    axes[1].tick_params(labelsize=18)
    axes[1].set_title('Full Power Spectrogram', fontsize=18)
    axes[1].set_ylim(0, fmax)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Colorbar
    cbar = fig.colorbar(pcm, ax=axes[1])
    cbar.set_label('Power [dB]', fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    fig.suptitle(title, fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

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

from scipy.signal import spectrogram, get_window
def ridge_snr_auto(trace,
                   fs,
                   sweep_band=(0.5, 250),       # expected chirp span [Hz]
                   noise_off=30,                # gap from ridge to noise band [Hz]
                   noise_span=3,                # half-width (# bins) of noise band
                   nperseg=None,
                   window='hann',
                   debug=False):
    """
    Ridge-tracking SNR for swept-frequency signals.
    Uses median power of a ±noise_span-bin band at ±noise_off Hz
    from the ridge, giving a stable noise estimate even at high amplitude.

    Returns
    -------
    snr_lin : float   linear SNR
    snr_db  : float   SNR in dB
    """
    if nperseg is None:                   # default = 1-s windows
        nperseg = int(fs)

    f, t, Sxx = spectrogram(trace, fs,
                            window=get_window(window, nperseg),
                            nperseg=nperseg,
                            noverlap=nperseg//2,
                            scaling='density')   # V²/Hz

    # limit ridge search to the nominal sweep band
    band_msk = (f >= sweep_band[0]) & (f <= sweep_band[1])
    S_band   = Sxx[band_msk, :]
    ridge_r  = np.argmax(S_band, axis=0)         # row index inside S_band
    ridge_bin = np.where(band_msk)[0][ridge_r]   # row index inside Sxx

    # frequency resolution
    df       = f[1] - f[0]
    off_bins = max(int(noise_off / df), noise_span + 1)

    nbins = len(f)
    idx_time = np.arange(len(t))

    # build noise-band indices (shape: (2*noise_span, time))
    noise_idx = []
    for side in (+1, -1):
        for j in range(noise_span):
            noise_idx.append((ridge_bin + side*(off_bins + j)) % nbins)
    noise_idx = np.array(noise_idx)

    # power per time slice
    sig_pow   = Sxx[ridge_bin, idx_time]                   # V²/Hz
    noise_pow = np.median(Sxx[noise_idx, idx_time], axis=0)

    P_sig   = np.mean(sig_pow   * df)                      # V²
    P_noise = np.mean(noise_pow * df)                      # V²
    snr_lin = P_sig / P_noise
    snr_db  = 10 * np.log10(snr_lin)

    if debug:
        print(f"Δf bin = {df:.2f} Hz   off_bins = {off_bins}")
        print(f"Mean ridge freq  = {f[ridge_bin].mean():.1f} Hz")
        print(f"Mean noise freqs = "
              f"{f[(ridge_bin+off_bins)%nbins].mean():.1f} / "
              f"{f[(ridge_bin-off_bins)%nbins].mean():.1f} Hz")
        print(f"P_sig = {P_sig:.3e}  P_noise = {P_noise:.3e}  "
              f"SNR = {snr_db:.1f} dB")

    return snr_lin, snr_db

#%%
'''ATLAS imaging'''
sampling_rate=1682.92
dpath = r'G:\2025_ATLAS_SPAD\ChirpSignal\Set3\Atlas/'
recordingNum=6
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
# %%fake sweep: chirp + white noise
snr_lin, snr_db = ridge_snr_auto(raw_signal,
                                 fs=sampling_rate,
                                 sweep_band=(0.5, 250),
                                 noise_off=40,     # give the strong sweep more room
                                 noise_span=3,
                                 debug=True)       # prints ridge / noise info

print(f"SNR ≈ {snr_lin:.1f}  ({snr_db:.1f} dB)")
#%%
# Plot full spectrogram
plot_power_spectrum(raw_signal, sampling_rate, title='Spectrogram of Full Signal',
                    nperseg=512, noverlap=128, fmax=260)

# Automatically cut signal between two full sweep cycles
cut_sig=cut_signal_by_time(raw_signal, sampling_rate)
# Plot trimmed spectrogram
plot_signal_and_spectrogram_time(cut_sig, sampling_rate, title=None,
                    nperseg=512, noverlap=256, fmax=260)


#%%
plot_signal_and_spectrogram(cut_sig, sampling_rate, title=None,
                    nperseg=512, noverlap=256, fmax=260)
#%%
'''For pyPhotometry imaging'''
# Folder with your files
# Modify it depending on where your file is located
folder =r"G:\2025_ATLAS_SPAD\ChirpSignal\Set3\pyPhotometry/"
# File name
file_name = '2mV-2025-08-01-175542.csv'
sampling_rate=1000
'''Read csv file and calculate zscore of the fluorescent signal'''
PhotometryData = pd.read_csv(folder+file_name,index_col=False) 
raw_signal = PhotometryData['Analog1'][1:]
# from SPADPhotometryAnalysis import photometry_functions as fp
# raw_signal=fp.smooth_signal(raw_signal, 4)
snr_lin, snr_db = ridge_snr_auto(raw_signal,
                                 fs=sampling_rate,
                                 sweep_band=(0.5, 250),
                                 noise_off=40,     # give the strong sweep more room
                                 noise_span=3,
                                 debug=True)       # prints ridge / noise info

print(f"SNR ≈ {snr_lin:.1f}  ({snr_db:.1f} dB)")

#%%
# Plot full spectrogram
plot_power_spectrum(raw_signal, sampling_rate, title='Spectrogram of Full Signal',
                    nperseg=256, noverlap=128, fmax=260)

# Automatically cut signal between two full sweep cycles
cut_sig=cut_signal_by_time(raw_signal, sampling_rate)
# Plot trimmed spectrogram
plot_signal_and_spectrogram_time(cut_sig, sampling_rate, title=None,
                    nperseg=256, noverlap=128, fmax=260)
#%%
plot_signal_and_spectrogram(cut_sig, sampling_rate, title=None,
                    nperseg=256, noverlap=128, fmax=260)
#%%
'''SPC imaging'''
sampling_rate=9938.4
dpath = r'G:\2025_ATLAS_SPAD\ChirpSignal\Set3\SPC/'
recordingNum=6
sync_recording_str = f"SyncRecording{recordingNum}"
recording_path=file_path_optical=os.path.join(dpath,sync_recording_str)
file_path_optical=os.path.join(dpath,sync_recording_str, "traceValueAll.csv")
trace_raw = pd.read_csv(file_path_optical, header=None)  # Adjust if there's a head
trace_raw = trace_raw.squeeze()
raw_signal = trace_raw

snr_lin, snr_db = ridge_snr_auto(raw_signal,
                                 fs=sampling_rate,
                                 sweep_band=(0.5, 250),
                                 noise_off=40,     # give the strong sweep more room
                                 noise_span=3,
                                 debug=True)       # prints ridge / noise info

print(f"SNR ≈ {snr_lin:.1f}  ({snr_db:.1f} dB)")
#%%
# Plot full spectrogram
plot_power_spectrum(raw_signal, sampling_rate, title='Spectrogram of Full Signal',
                    nperseg=2048, noverlap=256, fmax=260)

# Automatically cut signal between two full sweep cycles
cut_sig=cut_signal_by_time(raw_signal, sampling_rate)
# Plot trimmed spectrogram
plot_signal_and_spectrogram_time(cut_sig, sampling_rate, title=None,
                    nperseg=2048, noverlap=256, fmax=260)
#%%
plot_signal_and_spectrogram(cut_sig, sampling_rate, title=None,
                    nperseg=2048, noverlap=256, fmax=260)

#%%
'PLOT SNR'
import numpy as np
import matplotlib.pyplot as plt

# ── data ──────────────────────────────────────────────────────────
amps  = np.array([80, 40, 20, 10, 5, 2])           # mV (x-axis)

snr_atlas = np.array([29251.2, 9799.1, 3389.0, 759.6, 175.4, 40.7])
snr_spc   = np.array([ 9633.5, 2072.1,  881.7, 179.2,  54.8, 13.5])
snr_py    = np.array([   65.5,   24.3,   10.3,   8.1,   5.1,  6.9])

# ── plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(amps, snr_py,    'd-', label='pyPhotometry')
ax.plot(amps, snr_spc,   's-', label='SPC')
ax.plot(amps, snr_atlas, 'o-', label='ATLAS')


ax.set_xlabel('LED modulation amplitude (mV)')
ax.set_ylabel('SNR (linear)')

# big fonts
ax.set_xlabel('LED modulation amplitude (mV)', fontsize=16)
ax.set_ylabel('SNR (Linear)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.legend(frameon=False, fontsize=14)

ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.show()
#%%
# ── data (SNR in dB) ──────────────────────────────────────────────
amps       = np.array([80, 40, 20, 10,  5,  2])          # mV
snr_db_atl = np.array([44.7, 39.9, 35.3, 28.8, 22.4, 16.1])
snr_db_spc = np.array([39.8, 33.2, 29.5, 22.5, 17.4, 11.3])
snr_db_py  = np.array([18.2, 13.9, 10.1,  9.1,  7.1,  8.4])

# ── plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(amps, snr_db_py,  'd-', label='pyPhotometry', linewidth=2)
ax.plot(amps, snr_db_spc, 's-', label='SPC',         linewidth=2)
ax.plot(amps, snr_db_atl, 'o-', label='ATLAS',        linewidth=2)

# big fonts
ax.set_xlabel('LED modulation amplitude (mV)', fontsize=16)
ax.set_ylabel('SNR (dB)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.legend(frameon=False, fontsize=14)

ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
plt.show()
