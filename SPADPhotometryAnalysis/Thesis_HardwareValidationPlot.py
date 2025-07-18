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
    CVs = [Continuous_CV(os.path.join(dpath, i,"Green_traceAll.csv"),fs,lowpass) for i in filenamelist]
    return CVs

def CalculateCVs_SPAD(dpath,filenamelist,fs,lowpass=False):
    CVs = [Continuous_CV_SPAD(os.path.join(dpath, i,"Green_traceAll.csv"),fs,lowpass) for i in filenamelist]
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
    LED_power_Doric= np.array([2,4,6,8,10,12,14,16,18,20])
    
    dpath="C:/SPAD/SPADData/202200606Photometry/contPy"# PV cre
    Doriclist=["contPy5-2022-06-07-135712.csv","contPy6-2022-06-07-135727.csv","contPy7-2022-06-07-135744.csv",
                  "contPy8-2022-06-07-135758.csv","contPy9-2022-06-07-135816.csv","contPy10-2022-06-07-135830.csv"]
    
    CVs_Doric=CalculateCVs(dpath,Doriclist,99,lowpass=20)
    LED_power_photometry  = np.array([60,73, 85, 98,110,123])
    
    plt.plot(LED_power_Doric, CVs_pyPhotometry,'o-', label='Doric')
    plt.plot(LED_power_photometry, CVs_Doric,'o-', label='pyBoard')
    plt.xlabel('Continuous  LED power (uW)')
    plt.ylabel('Signal coef. of variation.')
    #plt.xticks(np.arange(0,22,4))
    plt.ylim(ymin=0)
    plt.legend()
    return -1

def Mean_PhotonCount (filename):
    trace = np.genfromtxt(filename, delimiter=',')
    meanCount = np.mean(trace)
    return meanCount

def PhotonCountMeans(dpath,filenamelist):
    MeanCountValues = [Mean_PhotonCount(os.path.join(dpath, i, "Green_traceAll.csv")) for i in filenamelist]
    return MeanCountValues

def PhotonCountMeans_1(dpath,filenamelist):
    MeanCountValues = [Mean_PhotonCount(os.path.join(dpath, i, "GreenValueAll.csv")) for i in filenamelist]
    return MeanCountValues

def CalculateMeans_SPAD(dpath, folder_list, some_param):
    # Placeholder function
    # Replace this with your real mean calculation function similar to CalculateCVs_SPAD
    means = []
    for folder in folder_list:
        # Calculate mean for each folder, dummy example:
        mean_val = some_param  # replace with actual mean retrieval
        means.append(mean_val)
    return np.array(means)

#%%
'''Coefficient covarience'''

folder_list=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6",
              "SyncRecording7","SyncRecording8","SyncRecording9",
              "SyncRecording10"]
dpath="H:/ThesisData/HardwareTest/pyPhotometry_linearity/"
CVs_py=CalculateCVs_SPAD(dpath,folder_list,1000,lowpass=False)

LED_current_cont = np.array([2,4,6,8,10,12,14,16,18,20])

dpath="H:/ThesisData/HardwareTest/SPC_linearity/"
CVs_SPC=CalculateCVs_SPAD(dpath,folder_list,9938.4,lowpass=False)
LED_current_cont =  np.array([2,4,6,8,10,12,14,16,18,20])

dpath="H:/ThesisData/HardwareTest/ATLAS_linearity/"
CVs_ATLAS=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)
LED_current_cont =  np.array([2,4,6,8,10,12,14,16,18,20])

means_ATLAS = CalculateMeans_SPAD("H:/ThesisData/HardwareTest/ATLAS_linearity/", folder_list, 840)
# Multiply means by 1257 pixels
ideal_means = means_ATLAS * 1257

# Calculate ideal CV for shot noise: CV = 1 / sqrt(mean)
ideal_CV = 1 / np.sqrt(ideal_means)
#%%
# === Plotting ===
plt.plot(LED_current_cont, CVs_py, 'o-', label='pyPhotometry')
plt.plot(LED_current_cont, CVs_SPC, '^-', label='SPC')
plt.plot(LED_current_cont, CVs_ATLAS, 'D-', label='ATLAS')
plt.plot(LED_current_cont, ideal_CV, 'k--', label='Ideal shot noise CV')

plt.xlabel('LED Light Power (μW)', fontsize=14)
plt.ylabel('Signal coefficient of variation', fontsize=14)
plt.ylim(ymin=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(LED_current_cont, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(frameon=False, fontsize=14)
plt.show()
#%%
plt.plot(LED_current_cont, CVs_py,'o-', label='pyPhotometry')
plt.plot(LED_current_cont, CVs_SPC,'^-', label='SPC')
plt.plot(LED_current_cont, CVs_ATLAS,'D-', label='ATLAS')

# plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous')
# plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division')
plt.xlabel('LED Light Power (μW)')
plt.ylabel('Signal coefficient of variation')
#plt.xticks(np.arange(0,22,4))
plt.ylim(ymin=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(LED_current_cont) 
plt.legend()
plt.legend(frameon=False)
plt.show()
#%%
dpath="E:/ATLAS_SPAD\HardwareTest/PSD_noise_bleached/"
GreenList=["5uW_bleached","10uW_bleached","20uW_bleached",
              "50uW_bleached","80uW_bleached"]
Green_Auto1=PhotonCountMeans(dpath,GreenList)


dpath="E:/ATLAS_SPAD\HardwareTest/PSD_noise_still/"
GreenList=["5uW","10uW","20uW",
              "50uW","80uW"]
Green_Auto2=PhotonCountMeans(dpath,GreenList)


Green_Power = np.array([5,10,20,50,80])

plt.plot(Green_Power,Green_Auto2,'D-',label='before-bleach')# # 2ca02c is green,ff7f0e is orange,1f77b4 is blue#
plt.plot(Green_Power,Green_Auto1,'o-',label='after-bleach')# # 2ca02c is green,ff7f0e is orange,1f77b4 is blue#
plt.rc('font', size=14)  # Adjust as needed
plt.xlabel('Continuous LED power (μW)',fontsize=16)
plt.ylabel('Average Photon Count',fontsize=16)
#plt.ylabel('Total Photon Count',fontsize=16)
#plt.ylabel('Voltage (mV)',fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(Green_Power) 
plt.legend()
plt.legend(frameon=False)
plt.show()
#%%
dpath="E:/ATLAS_SPAD\HardwareTest/PSD_noise_bleached/"
folder_list=["5uW_bleached","10uW_bleached","20uW_bleached",
              "50uW_bleached","80uW_bleached"]
CVs_1=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)

LED_current_cont =  np.array([5,10,20,50,80])
dpath="E:/ATLAS_SPAD\HardwareTest/PSD_noise_still/"
folder_list=["5uW","10uW","20uW",
              "50uW","80uW"]
CVs_2=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)

plt.plot(LED_current_cont, CVs_2,'D-', label='before-bleachS')
plt.plot(LED_current_cont, CVs_1,'o-', label='after-bleach')
# plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous')
# plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division')
plt.xlabel('LED Light Power (μW)')
plt.ylabel('Signal coefficient of variation')
#plt.xticks(np.arange(0,22,4))
plt.ylim(ymin=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(ymin=0)
plt.ylim(ymax=0.02)
plt.xticks(LED_current_cont) 
plt.legend()
plt.legend(frameon=False)
#%%
'''Compare Linearity LARGER VALUE'''

dpath="E:/ATLAS_SPAD/HardwareTest/ATLAS_linearity/"
GreenList=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6",
              "SyncRecording7","SyncRecording8","SyncRecording9",
              "SyncRecording10"]

Green_Auto=PhotonCountMeans(dpath,GreenList)

Green_Power = np.array([2,4,6,8,10,12,14,16,18,20])
Green_Auto =[74.78062289737453,
 138.3420015260292,
 201.518780910762,
 264.96673464433115,
 329.3816819269587,
 420.3217266673603,
 550.5136449554329,
 683.711538861721,
 759.0596096486664,
 834.0975528040786]
plt.plot(Green_Power,Green_Auto,'o-',color='#2ca02c',label='ATLAS')# # 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.rc('font', size=14)  # Adjust as needed
plt.xlabel('Continuous LED power (μW)',fontsize=16)
plt.ylabel('Average Photon Count',fontsize=16)
#plt.ylabel('Total Photon Count',fontsize=16)
#plt.ylabel('Voltage (mV)',fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(Green_Power) 
plt.legend()
plt.legend(frameon=False)
plt.show()
#%%
'''Compare Linearity SMALL VALUE'''
from scipy.stats import linregress

dpath = r"H:\ThesisData\HardwareTest\ForLinearity_5uWrange\pyPhotometry"
GreenList = ["SyncRecording1","SyncRecording2","SyncRecording3",
             "SyncRecording4","SyncRecording5","SyncRecording6",
             "SyncRecording7","SyncRecording8"]

Green_Auto = PhotonCountMeans_1(dpath, GreenList)
Green_Power = np.array([.06, .25, .5, 1, 2, 3, 4, 5])
# Linear regression (with intercept)
slope, intercept, r_value, p_value, std_err = linregress(Green_Power, Green_Auto)
r_squared = r_value**2
# Extend x-range to include 0
x_fit = np.linspace(0, Green_Power.max() * 1.05, 100)
y_fit = slope * x_fit + intercept

# Plot data
plt.plot(Green_Power, Green_Auto, 'o-', color='#1f77b4', label='pyPhotometry')

# Plot regression line (black)
plt.plot(x_fit, y_fit, '--', color='gray', label=f'Linear Fit (R² = {r_squared:.4f})')

# Aesthetics
plt.rc('font', size=16)
plt.xlabel('Continuous LED power (uW)', fontsize=16)
plt.ylabel('Voltage (mV)', fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.array([0, .5, 1, 2, 3, 4, 5])) 
plt.xlim(left=0)  # 👈 Make x-axis start at 0
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# Print regression info
print(f"Slope = {slope:.4f}, Intercept = {intercept:.4f}")
print(f"R² = {r_squared:.4f}")
#%%
dpath=r"H:\ThesisData\HardwareTest\ForLinearity_5uWrange\Atlas"
GreenList=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6",
              "SyncRecording7","SyncRecording8"]

Green_Auto=PhotonCountMeans(dpath,GreenList)

Green_Power = np.array([.06,.25,.5,1,2,3,4,5])

# Linear regression (with intercept)
slope, intercept, r_value, p_value, std_err = linregress(Green_Power, Green_Auto)
r_squared = r_value**2
# Extend x-range to include 0
x_fit = np.linspace(0, Green_Power.max() * 1.05, 100)
y_fit = slope * x_fit + intercept

# Plot data
plt.plot(Green_Power, Green_Auto, 'o-', color='#2ca02c', label='ATLAS')

# Plot regression line (black)
plt.plot(x_fit, y_fit, '--', color='gray', label=f'Linear Fit (R² = {r_squared:.4f})')

# Aesthetics
plt.rc('font', size=16)
plt.xlabel('Continuous LED power (uW)', fontsize=16)
plt.ylabel('Voltage (mV)', fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.array([0, .5, 1, 2, 3, 4, 5])) 
plt.xlim(left=0)  # 👈 Make x-axis start at 0
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# Print regression info
print(f"Slope = {slope:.4f}, Intercept = {intercept:.4f}")
print(f"R² = {r_squared:.4f}")
#%%
dpath=r"H:\ThesisData\HardwareTest\ForLinearity_5uWrange\SPC"
GreenList=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6",
              "SyncRecording7","SyncRecording8"]

Green_Auto=PhotonCountMeans_1(dpath,GreenList)

Green_Power = np.array([.06,.25,.5,1,2,3,4,5])
# Linear regression (with intercept)
slope, intercept, r_value, p_value, std_err = linregress(Green_Power, Green_Auto)
r_squared = r_value**2
# Extend x-range to include 0
x_fit = np.linspace(0, Green_Power.max() * 1.05, 100)
y_fit = slope * x_fit + intercept

# Plot data
plt.plot(Green_Power, Green_Auto, 'o-', color='#ff7f0e', label='SPC')

# Plot regression line (black)
plt.plot(x_fit, y_fit, '--', color='gray', label=f'Linear Fit (R² = {r_squared:.4f})')

# Aesthetics
plt.rc('font', size=16)
plt.xlabel('Continuous LED power (uW)', fontsize=16)
plt.ylabel('Voltage (mV)', fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.array([0, .5, 1, 2, 3, 4, 5])) 
plt.xlim(left=0)  # 👈 Make x-axis start at 0
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# Print regression info
print(f"Slope = {slope:.4f}, Intercept = {intercept:.4f}")
print(f"R² = {r_squared:.4f}")
#%%
'''Compare Linearity Collimator'''

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Collimator_600umNA0.48/LargeROI/"
GreenList=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6"]

Green_Auto=PhotonCountMeans(dpath,GreenList)
Green_Auto=[x * 4 for x in Green_Auto]
Green_Power = np.array([3,8,21,45,71,100])

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Collimator_600umNA0.48/FitROI/"
Green_Auto1=PhotonCountMeans(dpath,GreenList)
Green_Auto1=[x * 4 for x in Green_Auto1]



plt.plot(Green_Power,Green_Auto,'o-',label='Collimator-Large')# 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.plot(Green_Power,Green_Auto1,'D-',label='Collimator-Fit')# 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.rc('font', size=16)  # Adjust as needed
plt.xlabel('Continuous LED power (uW)',fontsize=16)
plt.ylabel('Total Photon Count',fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(Green_Power) 
plt.legend()
plt.legend(frameon=False)
plt.show()

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_600um_0d48NA/LargeROI/"
GreenList=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6"]

Green_Auto3=PhotonCountMeans(dpath,GreenList)

Green_Power = np.array([3,8,21,45,71,100])

plt.plot(Green_Power,Green_Auto3,'D-',color='#2ca02c',label='No Collimator')# 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.rc('font', size=16)  # Adjust as needed
plt.xlabel('Continuous LED power (uW)',fontsize=16)
plt.ylabel('Total Photon Count',fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(Green_Power) 
plt.legend()
plt.legend(frameon=False)
plt.show()

#%%
'''Coefficient covarience'''

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Collimator_600umNA0.48/LargeROI/"
folder_list=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5","SyncRecording6"]

CVs_1=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)

LED_current_cont = np.array([3,8,21,45,71,100])

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Collimator_600umNA0.48/FitROI/"
CVs_2=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)


dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_600um_0d48NA/LargeROI/"

CVs_3=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)

plt.plot(LED_current_cont, CVs_1,'o-', label='Collimator-LargeROI')
plt.plot(LED_current_cont, CVs_2,'^-', label='Collimator-FitROI')
plt.plot(LED_current_cont, CVs_3,'D-', label='No-collimator')

# plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous')
# plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division')
plt.xlabel('LED Light Power (μW)')
plt.ylabel('Signal coefficient of variation')
#plt.xticks(np.arange(0,22,4))
plt.ylim(ymin=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(LED_current_cont) 
plt.legend()
plt.legend(frameon=False)
plt.show()

#%%
dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_200um_0d57NA/LargeROI/"
GreenList=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5"]

Green_Auto=PhotonCountMeans(dpath,GreenList)
Green_Auto=[x * 1 for x in Green_Auto]
Green_Power = np.array([3,8,21,45,71])

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_400um_0d57NA/FitROI/"
Green_Auto1=PhotonCountMeans(dpath,GreenList)
Green_Auto1=[x * 1 for x in Green_Auto1]


dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_600um_0d48NA/LargeROI/"
Green_Auto2=PhotonCountMeans(dpath,GreenList)

plt.plot(Green_Power,Green_Auto,'o-',label='200um-NA0.57')# 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.plot(Green_Power,Green_Auto1,'^-',label='400um-NA0.57')# 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.plot(Green_Power,Green_Auto2,'D-',label='600um-NA0.48')# 2ca02c is green,ff7f0e is orange,1f77b4 is blue
plt.rc('font', size=16)  # Adjust as needed
plt.xlabel('Continuous LED power (uW)',fontsize=16)
plt.ylabel('Total Photon Count',fontsize=16)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(Green_Power) 
plt.legend()
plt.legend(frameon=False)
plt.show()
#%%
'''Coefficient covarience'''

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_200um_0d57NA/FitROI/"
folder_list=["SyncRecording1","SyncRecording2","SyncRecording3",
              "SyncRecording4","SyncRecording5"]

CVs_1=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)

LED_current_cont = np.array([3,8,21,45,71])

dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_400um_0d57NA/FitROI/"
CVs_2=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)


dpath="E:/ATLAS_SPAD/ColiimatorFibreNAtest/Fibre_600um_0d48NA/FitROI/"

CVs_3=CalculateCVs_SPAD(dpath,folder_list,840,lowpass=False)

plt.plot(LED_current_cont, CVs_1,'o-', label='200um-NA0.57')
plt.plot(LED_current_cont, CVs_2,'^-', label='400um-NA0.57')
plt.plot(LED_current_cont, CVs_3,'D-', label='600um-NA0.48')

# plt.plot(LED_current_cont, CVs_cont,'o-', label='Contiuous')
# plt.plot(LED_current_TD, CVs_TD,'o-', label='Time-division')
plt.xlabel('LED Light Power (μW)')
plt.ylabel('Signal coefficient of variation')
#plt.xticks(np.arange(0,22,4))
plt.ylim(ymin=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(LED_current_cont) 
plt.legend()
plt.legend(frameon=False)
plt.show()