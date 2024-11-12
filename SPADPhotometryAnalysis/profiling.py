# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:17:40 2024

@author: Yifang
"""

from SPADPhotometryAnalysis import AtlasDecode

#%% Workable code, above is testin
#ppath='D:/ATLAS_SPAD/1825505_SimCre/Day2/Atlas/'
dpath='E:/ATLAS_SPAD/ColiimatorFibreNAtest/Collimator_600umNA0.48/Atlas/Burst-RS-4200frames-840Hz_2024-11-12_11-56_100/'
#dpath='F:/SPADdata/SNR_test_2to16uW/Altas_SNR_20240318/18032024/Burst-RS-1017frames-1017Hz_4uW/'
#hotpixel_path='E:/YYFstudy/OptoEphysAnalysis/Altas_hotpixel.csv'
hotpixel_path='C:/SPAD/OptoEphysAnalysis/Altas_hotpixel.csv'

pixel_array_all_frames,_,avg_pixel_array=AtlasDecode.decode_atlas_folder (dpath,hotpixel_path,photoncount_thre=2000)