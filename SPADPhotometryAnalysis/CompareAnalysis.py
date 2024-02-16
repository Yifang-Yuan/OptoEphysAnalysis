# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 22:55:20 2022

@author: Yifang
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import SPADreadBin
'''barplot the mean dark count value'''

# dpath="C:/SPAD/SPADData/TianSampleData/continuous_SPAD"
# filename = os.path.join(dpath, "trace_ref1.csv")  #csv file is the file contain values for each frame
# count_value = np.genfromtxt(filename, delimiter=',')
# plt.figure(figsize=(20, 6))
# plt.plot(count_value,linewidth=1)
# plt.title("Single Cell")

dpath="C:/SPAD/SPADData/20220420/BeforeBleachingGreen100mA_2022_4_20_19_26_11"
filename = os.path.join(dpath, "traceValue1.csv")
DarkRoom= np.genfromtxt(filename, delimiter=',')
plt.figure(figsize=(20, 6))
plt.plot(DarkRoom,linewidth=1)
plt.title("BeforeBleaching")

dpath="D:/SPAD/SPADData/20220419/NoAnimalNoCannula_2022_4_19_10_14_56"
filename = os.path.join(dpath, "traceValue1.csv")
Rmlight_blackpaper = np.genfromtxt(filename, delimiter=',')
plt.figure(figsize=(20, 6))
plt.plot(Rmlight_blackpaper,linewidth=1)
plt.title("AfterBleaching")

dpath="D:/SPAD/SPADData/20220419/NoAnimalWithCannula_2022_4_19_10_19_56"
filename = os.path.join(dpath, "traceValue1.csv")
Darkroom_tissue = np.genfromtxt(filename, delimiter=',')
plt.figure(figsize=(20, 6))
plt.plot(Darkroom_tissue,linewidth=1)
plt.title("Cannula_Before")

dpath="D:/SPAD/SPADData/20220419/NoAnimalWithCannula_Post_2022_4_19_10_28_0"
filename = os.path.join(dpath, "traceValue1.csv")
Rmlight_tissue = np.genfromtxt(filename, delimiter=',')
plt.figure(figsize=(20, 6))
plt.plot(Rmlight_tissue,linewidth=1)
plt.title("Cannula_After")

#%%
MeanPhotoCount=np.zeros(4)
MeanPhotoCount[0]=DarkRoom.mean()
MeanPhotoCount[1]=Rmlight_blackpaper.mean()
MeanPhotoCount[2]=Darkroom_tissue.mean()
MeanPhotoCount[3]=Rmlight_tissue.mean()

#%%
'''Bar plot'''
d = {'Before': DarkRoom.mean(), 'After': Rmlight_blackpaper.mean(), 'Cannula_Before': Darkroom_tissue.mean(),
     'Cannula_After': Rmlight_tissue.mean()}
#ser = pd.Series(data=d, index=['DarkRoom', 'Rmlight_blackpaper', 'Darkroom_tissue','Rmlight_tissue'])
ser = pd.Series(data=d)

plt.figure(figsize=(4, 3))
ax=ser.plot(kind='bar')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Photon Count')
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() * 1.05, p.get_height() * 1.02))
plt.xticks(rotation=45)


# '''Tian sample data'''
# dpath="C:/SPAD/SPADData/TianSampleData/continuous_SPAD/real_data"
# filename = os.path.join(dpath, "spc_data3.bin")
#%%
dpath="C:/SPAD/SPADData/20220420/DarkCountBaseline_2022_4_20_19_24_57"
filename = os.path.join(dpath, "spc_data1.bin")
Bindata=SPADreadBin.SPADreadBin(filename,pyGUI=False)
SPADreadBin.ShowImage(Bindata,dpath)
#%%
PixelArrary_original=np.sum(Bindata, axis=0)
data=PixelArrary_original.flatten()
#%%
'''Histogram of dark count rate'''
fig, ax = plt.subplots(1,1)
ax.hist(data)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Photon count')
ax.set_ylabel('Pixel number')
#%%
'''xxrange=[60,180],yyrange=[40,160]'''
#Red50mA=SPADreadBin.countTraceValue(dpath,Bindata,xxrange=[60,180],yyrange=[40,160]) 
Dark=SPADreadBin.countTraceValue(dpath,Bindata,xxrange=[0,320],yyrange=[0,240]) 
#%%
#ShowImage(Bindata,dpath)
#%%
# HotPixelIdx,HotPixelNum=FindHotPixel(dpath,Bindata,thres=0.1)
# IdxFilename="C:/SPAD/SPADData/HotPixelIdx_TianPCB.csv"
# HotPixelIdx_read=np.genfromtxt(IdxFilename, delimiter=',')
# #HotPixelIdx_read =np.load(IdxFilename, allow_pickle=True)
# HotPixelIdx_read=HotPixelIdx_read.astype(int)
#%%
# plt.figure(figsize=(15, 4))
# plt.plot(count_value,linewidth=1)
# plt.title("trace4")