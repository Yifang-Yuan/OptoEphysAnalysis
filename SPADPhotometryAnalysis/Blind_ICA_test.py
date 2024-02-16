# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:22:04 2022

@author: Yifang
"""

from sklearn.decomposition import FastICA, PCA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SPADdemod

#%%
dpath="C:/SPAD/SPADData/20220423/1454214_g1r2_2022_4_23_13_48_56"
filename = os.path.join(dpath, "traceValue1.csv")  #csv file is the file contain values for each frame
count_value = np.genfromtxt(filename, delimiter=',')

Red,Green= SPADdemod.DemodFreqShift (count_value,fc_g=1000,fc_r=2000,fs=9938.4)
#%% Photometry data
dpath= "C:/SPAD/SPADData/20220407"
csv_1 =  os.path.join(dpath, "1432002-2022-04-07-161522.csv")
mouse1=pd.read_csv(csv_1)  
Green=mouse1['Analog1']
Red=mouse1['Analog2']
#%%
csv_2 =  os.path.join(dpath, "1454213-2022-04-07-154749.csv")
mouse2=pd.read_csv(csv_2) 
Green=mouse2['Analog1']
Red=mouse2['Analog2']
#%%
plt.figure(figsize=(20, 6))
plt.plot(Green, color='g',linewidth=1)
plt.plot(Red, color='r',linewidth=1)
#%%
fig, ax = plt.subplots(figsize=(6, 4))
ax.psd(Green,Fs=130,linewidth=1,color='g',label="green channel")
ax.psd(Red,Fs=130,linewidth=1,color='r',label="red channel")
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(False)
#%%
plt.figure()
plt.psd(Green,Fs=130,linewidth=1,color='g')
plt.psd(Red,Fs=130,linewidth=1,color='r')


#%%
fs=130
fig, ax = plt.subplots(figsize=(8, 2))
powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(Green, Fs=fs,NFFT=1024, detrend='linear')
ax.set_xlabel('Time (Second)')
ax.set_ylabel('Frequency')


fig, ax = plt.subplots(figsize=(8, 2))
powerSpectrum, freqenciesFound, time, imageAxis = ax.specgram(Red, Fs=fs,NFFT=1024, detrend='linear')
ax.set_xlabel('Time (Second)')
ax.set_ylabel('Frequency')
#%% fastICA
channel1=Green
channel2=Red

X = np.c_[channel1,channel2]
# Compute ICA
ica = FastICA(n_components=2)
S = ica.fit_transform(X)  # Reconstruct signals
A = ica.mixing_  # Get estimated mixing matrix
#%%PSD of recovered signal
plt.figure()
plt.psd(S[:,0],Fs=130,linewidth=1,color='g')
plt.psd(S[:,1],Fs=130,linewidth=1,color='r')
#%%
fig, ax = plt.subplots(figsize=(6, 4))
ax.psd(S[:,0],Fs=130,linewidth=1,color='g',label="recovered signal 1")
ax.psd(S[:,1],Fs=130,linewidth=1,color='r',label="recovered signal 2")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
ax.grid(False)
#%%
plt.figure()

models = [X, S]
names = [
    "Observations (mixed signal)",
    "ICA recovered signals",
]
colors = ["green", "red"]

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()

#%% plot only a part of the trace

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig[49000:50000], color=color)

plt.tight_layout()
plt.show()