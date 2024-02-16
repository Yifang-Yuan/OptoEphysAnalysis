# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 18:50:34 2022

@author: Yifang
"""
import os
import numpy as np 
import pylab as plt
import pandas as pd 
from functools import partial
from matplotlib.gridspec import GridSpec
from scipy.signal import windows, butter, filtfilt
from scipy.io import loadmat

'''CV:coefficient of variation'''

def Continuous_CV(filename,fs,lowpass=False):
    Two_traces=pd.read_csv(filename)
    signal=Two_traces['Analog1']
    
    if lowpass:
        # Calculate coeficcient of variation for signal lowpassed at 20Hz.
        b, a = butter(2, lowpass/(0.5*fs), 'lowpass')
        td_sig_lowpass = filtfilt(b,a,signal)
        td_CV = np.std(td_sig_lowpass)/np.mean(td_sig_lowpass)
    else:
        td_CV = np.std(signal)/np.mean(signal)
    return td_CV

def Continuous_CV_SPAD (filename,fs,lowpass=False):
    signal= np.genfromtxt(filename, delimiter=',')
    
    if lowpass:
        # Calculate coeficcient of variation for signal lowpassed at 20Hz.
        b, a = butter(2, lowpass/(0.5*fs), 'lowpass')
        td_sig_lowpass = filtfilt(b,a,signal)
        td_CV = np.std(td_sig_lowpass)/np.mean(td_sig_lowpass)
    else:
        td_CV = np.std(signal)/np.mean(signal)
    return td_CV

def CalculateCVs(dpath,filenamelist,fs,lowpass=False):
    CVs = [Continuous_CV(os.path.join(dpath, i),fs,lowpass) for i in filenamelist]
    return CVs

def CalculateCVs_SPAD(dpath,filenamelist,fs,lowpass=False):
    CVs = [Continuous_CV_SPAD(os.path.join(dpath, i),fs,lowpass) for i in filenamelist]
    return CVs

'''Compare CV between SPAD and photometry'''
def compareSensor_CV():
    dpath="C:/SPAD/SPADData/202200606Photometry/contDoric"# PV cre
    pyPhotometry_list=["contDoric60-2022-06-07-135111.csv","ContDoric73-2022-06-07-135245.csv","contDoric85-2022-06-07-135324.csv",
                  "contDoric98-2022-06-07-135412.csv","contDoric110-2022-06-07-135438.csv","contDoric123-2022-06-07-135529.csv"]
    
    CVs_pyPhotometry=CalculateCVs(dpath,pyPhotometry_list,130,lowpass=False)
    LED_power_photometry = np.array([60,73, 85, 98,110,123])
    
    dpath="C:/SPAD/SPADData/20220606/"# PV cre
    SPADlist=["20mA_2022_6_7_15_24_20/traceValue.csv","50mA_2022_6_7_15_23_41/traceValue.csv","70mA2022_6_7_15_22_57/traceValue.csv",
                  "100mA_2022_6_7_15_22_0/traceValue.csv","175mA_65uW2022_6_7_15_21_14/traceValue.csv","201mA_72uW2022_6_7_15_20_32/traceValue.csv",
                  "252mA_85uW2022_6_7_15_19_19/traceValue.csv","310mA98uW2022_6_7_15_18_28/traceValue.csv","365mA110uW2022_6_7_15_17_38/traceValue.csv",
                  "428mA123uW2022_6_7_15_16_21/traceValue.csv"]
    
    CVs_SPAD=CalculateCVs_SPAD(dpath,SPADlist,9938.4,lowpass=False)
    
    LED_power_SPAD = np.array([12,25, 28, 43,65,72,85,98,110,123])
    
    plt.plot(LED_power_photometry, CVs_pyPhotometry,'o-', label='pyPhotometry_30Hz Lowpass')
    plt.plot(LED_power_SPAD, CVs_SPAD,'o-', label='SPAD_300Hz Lowpass')
    plt.xlabel('Continuous  LED power (uW)')
    plt.ylabel('Signal coef. of variation.')
    #plt.xticks(np.arange(0,22,4))
    plt.ylim(ymin=0)
    plt.legend()
    return -1

def compare_LED_CV():
    '''Compare CV between Doric LED driver and photometry LED driver'''
    dpath="C:/SPAD/SPADData/202200606Photometry/contDoric"# PV cre
    pyPhotometry_list=["contDoric60-2022-06-07-135111.csv","ContDoric73-2022-06-07-135245.csv","contDoric85-2022-06-07-135324.csv",
                  "contDoric98-2022-06-07-135412.csv","contDoric110-2022-06-07-135438.csv","contDoric123-2022-06-07-135529.csv"]
    
    CVs_pyPhotometry=CalculateCVs(dpath,pyPhotometry_list,130,lowpass=20)
    LED_power_Doric= np.array([60,73, 85, 98,110,123])
    
    dpath="C:/SPAD/SPADData/202200606Photometry/contPy"# PV cre
    Doriclist=["contPy5-2022-06-07-135712.csv","contPy6-2022-06-07-135727.csv","contPy7-2022-06-07-135744.csv",
                  "contPy8-2022-06-07-135758.csv","contPy9-2022-06-07-135816.csv","contPy10-2022-06-07-135830.csv"]
    
    CVs_Doric=CalculateCVs(dpath,Doriclist,130,lowpass=20)
    LED_power_photometry  = np.array([60,73, 85, 98,110,123])
    
    plt.plot(LED_power_Doric, CVs_pyPhotometry,'o-', label='Doric')
    plt.plot(LED_power_photometry, CVs_Doric,'o-', label='pyBoard')
    plt.xlabel('Continuous  LED power (uW)')
    plt.ylabel('Signal coef. of variation.')
    #plt.xticks(np.arange(0,22,4))
    plt.ylim(ymin=0)
    plt.legend()
    return -1
#%%
'''Compare photometry continuous mode and time-division mode--465nm'''
dpath="C:/SPAD/SPADData/202200606Photometry/contPyLED"
pyPhotometry_list=["contPyLED5-2022-06-07-131156.csv","contPyLED10-2022-06-07-131214.csv","contPyLED15-2022-06-07-131233.csv",
              "contPyLED20-2022-06-07-131255.csv","contPyLED25-2022-06-07-131317.csv","contPyLED30-2022-06-07-131340.csv"]

CVs_cont=CalculateCVs(dpath,pyPhotometry_list,130,lowpass=20)
LED_current_cont = np.array([20,40,60,80,100,120])

dpath="C:/SPAD/SPADData/202200606Photometry/timePyLED"
Doriclist=["timePyLED5-2022-06-07-130844.csv","timePyLED10-2022-06-07-130904.csv","timePyLED15-2022-06-07-130925.csv",
              "timePyLED20-2022-06-07-130942.csv","timePyLED25-2022-06-07-130959.csv","timePyLED30-2022-06-07-131018.csv"]

CVs_TD=CalculateCVs(dpath,Doriclist,130,lowpass=20)
LED_current_TD = np.array([20,40,60,80,100,120])

plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division 20Hz lowpass')
plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous 20Hz lowpass')

# plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous')
# plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division')
plt.xlabel('LED current (mA)')
plt.ylabel('Signal coef. of variation.')
#plt.xticks(np.arange(0,22,4))
plt.ylim(ymin=0)
plt.legend()

#%%
'''Compare photometry continuous mode and time-division mode--560nm'''
dpath="C:/SPAD/SPADData/202200606Photometry/560nmData/cont560_fs1kHz"
pyPhotometry_list=["cont560nm5-2022-06-11-144429.csv","cont560nm10-2022-06-11-144446.csv","cont560nm15-2022-06-11-144534.csv",
              "cont560nm20-2022-06-11-144552.csv","cont560nm25-2022-06-11-144609.csv","cont560nm30-2022-06-11-144632.csv"]

CVs_cont=CalculateCVs(dpath,pyPhotometry_list,1000,lowpass=20)
LED_current_cont = np.array([20,40,60,80,100,120])

dpath="C:/SPAD/SPADData/202200606Photometry/560nmData/time560"
Doriclist=["time560nm5-2022-06-11-144844.csv","time560nm10-2022-06-11-144901.csv","time560nm15-2022-06-11-144919.csv",
              "time560nm20-2022-06-11-144937.csv","time560nm25-2022-06-11-144957.csv","time560nm30-2022-06-11-145019.csv"]

CVs_TD=CalculateCVs(dpath,Doriclist,130,lowpass=20)
LED_current_TD = np.array([20,40,60,80,100,120])

plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division 560nm 20Hz lowpass')
plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous 560nm 20Hz lowpass')

# plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous')
# plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division')
plt.xlabel('LED current (mA)')
plt.ylabel('Signal coef. of variation.')
#plt.xticks(np.arange(0,22,4))
plt.ylim(ymin=0)
plt.legend()
#%%
'''Compare autofluorescence'''
def Mean_PhotonCount (filename):
    trace = np.genfromtxt(filename, delimiter=',')
    meanCount = np.mean(trace)
    return meanCount

def PhotonCountMeans(dpath,filenamelist):
    MeanCountValues = [Mean_PhotonCount(os.path.join(dpath, i, "traceValue.csv")) for i in filenamelist]
    return MeanCountValues

dpath="C:/SPAD/SPADData/20220610/"# PV cre
GreenList=["Green_17mA_2022_6_10_16_15_12","Green_18mA_2022_6_10_16_16_11","Green_20mA_2022_6_10_15_50_35",
              "Green_30mA_2022_6_10_15_51_7","Green_50mA_2022_6_10_15_51_48","Green_80mA_2022_6_10_15_52_17",
              "Green_100mA_2022_6_10_15_52_45","Green_150mA_2022_6_10_15_53_29","Green_200mA_2022_6_10_15_54_30",
              "Green_300mA_2022_6_10_15_55_28","Green_400mA_2022_6_10_15_56_11","Green_500mA_2022_6_10_15_56_52",
              "Green_600mA_2022_6_10_15_58_32","Green_800mA_2022_6_10_15_59_14","Green_1000mA_2022_6_10_15_59_41"]

Green_Auto=PhotonCountMeans(dpath,GreenList)
Green_Power = np.array([10.5,11,11.8,16.5,25,36.3,43,59,73,98,120,140,158,191,221])
#%%
RedList=["Red_20mA_2022_6_10_16_1_25","Red_60mA_2022_6_10_16_2_12","Red_100mA_2022_6_10_16_2_54",
              "Red_200mA_2022_6_10_16_3_39","Red_300mA_2022_6_10_16_4_9","Red_500mA_2022_6_10_16_4_55",
              "Red_700mA_2022_6_10_16_6_10","Red_800mA_2022_6_10_16_6_44","Red_900mA_2022_6_10_16_7_19",
              "Red_1000mA_2022_6_10_16_8_33"]

Red_Auto=PhotonCountMeans(dpath,RedList)
Red_Power = np.array([0.2,0.8,1.3,2.6,3.9,6.1,8.3,9.4,10.4,11.45])
#%%
plt.plot(Green_Power,Green_Auto,'o-',label='465nm')
plt.plot(Red_Power,Red_Auto,'o-', color='y',label='560nm')
plt.xlabel('Continuous LED power (uW)')
plt.ylabel('Average Photon Count')

plt.ylim(ymin=0)
plt.legend()
#%%
plt.plot(Red_Power,Red_Auto,'o-',color='y', label='560nm')
plt.xlabel('Continuous LED power (uW)')
plt.ylabel('Average Photon Count')

plt.ylim(ymin=0)
plt.legend()