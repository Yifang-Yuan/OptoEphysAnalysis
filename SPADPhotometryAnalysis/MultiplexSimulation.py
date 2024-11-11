# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:46:12 2022

@author: Yifang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from SPADPhotometryAnalysis import SPADAnalysisTools as Ananlysis
from SPADPhotometryAnalysis import SPADdemod
#%%
x = np.linspace(0, 1000, 1000)

def f(x):
    y = 0
    result = []
    for _ in x:
        result.append(y)
        y += np.random.normal(scale=1)
    return np.abs(np.array(result))

plt.plot(x,f(x))
#%%
'''Simulate the frequency modulation'''
samples=10000
fs =9938.4
t = np.arange(samples)/fs
fc_g=1000
fc_r=2000

G_Carrier = 20*np.sin(2.0*np.pi*fc_g*t)+30
G_signal=f(t)+5
G_mod = G_Carrier*G_signal
fig, (ax0, ax1,ax2) = plt.subplots(nrows=3)
ax0=Ananlysis.plot_trace(G_signal[0:500],ax0, fs=9938.4, label="Green Signal", color='g')
ax1=Ananlysis.plot_trace(G_Carrier[0:500],ax1, fs=9938.4, label="1kHz carrier", color='g')
ax2=Ananlysis.plot_trace(G_mod[0:500],ax2, fs=9938.4, label="Green Modulated Signal", color='g')
fig.tight_layout()

R_Carrier = 5*np.sin(2.0*np.pi*fc_r*t)+10
R_signal=f(t)+2
R_mod = R_Carrier*R_signal
fig, (ax0, ax1,ax2) = plt.subplots(nrows=3)
ax0=Ananlysis.plot_trace(R_signal,ax0, fs=9938.4, label="Red Signal", color='r')
ax1=Ananlysis.plot_trace(R_Carrier[0:500],ax1, fs=9938.4, label="2KHz carrier", color='r')
ax2=Ananlysis.plot_trace(R_mod,ax2, fs=9938.4, label="Red Modulated Signal", color='r')
fig.tight_layout()

Mix_mod=G_mod+R_mod
red_recovered,green_recovered=SPADdemod.DemodFreqShift (Mix_mod,fc_g,fc_r,fs=9938.4)

fig, (ax0, ax1,ax2) = plt.subplots(nrows=3)
ax0=Ananlysis.plot_trace(Mix_mod,ax0, fs=9938.4, label="Mixed Modulated Signal",color='b')
ax1=Ananlysis.plot_trace(green_recovered,ax1, fs=9938.4, label="Green Recovered", color='g')
ax2=Ananlysis.plot_trace(red_recovered,ax2, fs=9938.4, label="Red Recovered", color='r')
fig.tight_layout()
#%%
'''Simulate the time division'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Simulate the time division
samples = 1000
fs = 9938.4
t = np.arange(samples) / fs

# Define the carriers and signals
G_Carrier = 2 * signal.square(2 * np.pi * 500 * t, duty=0.3) + 2
G_signal = f(t) + 10
G_mod = G_Carrier * G_signal

R_Carrier = signal.square(2 * np.pi * 500 * (t + 0.001), duty=0.3) + 1
R_signal = f(t) + 5
R_mod = R_Carrier * R_signal

# Plot G and R channel signals and carriers
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6, 4))

# Remove frames, ticks, and labels, and move legend to top-right
ax0 = Ananlysis.plot_trace(G_signal, ax0, fs=9938.4, label="Green Signal", color='g')
ax1 = Ananlysis.plot_trace(G_Carrier, ax1, fs=9938.4, label="Square wave 1", color='g')
ax2 = Ananlysis.plot_trace(G_mod, ax2, fs=9938.4, label="Green Modulated Signal", color='g')

for ax in (ax0, ax1, ax2):
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label

fig.tight_layout()

# Overlay Red Channel on the same plot
ax0 = Ananlysis.plot_trace(R_signal, ax0, fs=9938.4, label="Red Signal", color='r')
ax1 = Ananlysis.plot_trace(R_Carrier, ax1, fs=9938.4, label="Square wave 2", color='r')
ax2 = Ananlysis.plot_trace(R_mod, ax2, fs=9938.4, label="Red Modulated Signal", color='r')

for ax in (ax0, ax1, ax2):
    ax.legend(loc='upper right', fontsize=10, frameon=False)

fig.tight_layout()

# Plot Mixed Signal and Channels
Mix_mod = G_mod + R_mod
GreenGet = Mix_mod * G_Carrier / 4
lmin, lmax = SPADdemod.hl_envelopes_idx(GreenGet, dmin=1, dmax=1, split=False)
xnew, green_recover = SPADdemod.Interpolate_timeDiv(lmax, GreenGet)

RedGet = Mix_mod * R_Carrier
lmin, lmax = SPADdemod.hl_envelopes_idx(RedGet, dmin=1, dmax=1, split=False)
xnew, red_recover = SPADdemod.Interpolate_timeDiv(lmax, RedGet)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6, 4))

ax0 = Ananlysis.plot_trace(Mix_mod, ax0, fs=9938.4, label="Mixed Signal", color='b')
ax1 = Ananlysis.plot_trace(GreenGet, ax1, fs=9938.4, label="Green Channel", color='g')
ax2 = Ananlysis.plot_trace(RedGet, ax2, fs=9938.4, label="Red Channel", color='r')

for ax in (ax0, ax1, ax2):
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label

fig.tight_layout()

# Plot recovered signals
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6, 4))

ax0 = Ananlysis.plot_trace(Mix_mod, ax0, fs=9938.4, label="Mixed Signal", color='b')
ax1 = Ananlysis.plot_trace(green_recover, ax1, fs=9938.4, label="Green Channel Recovered", color='g')
ax2 = Ananlysis.plot_trace(red_recover, ax2, fs=9938.4, label="Red Channel Recovered", color='r')

for ax in (ax0, ax1, ax2):
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label

fig.tight_layout()
