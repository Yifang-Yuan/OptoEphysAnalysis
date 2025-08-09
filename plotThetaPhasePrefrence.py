# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:41:08 2025

@author: yifan
"""

'''recordingMode: use py, Atlas, SPAD for different systems'''
def run_theta_plot_all_cycle (dpath,LFP_channel,recordingName,savename,theta_low_thres=0.5):
    save_path = os.path.join(dpath,savename)
    os.makedirs(save_path, exist_ok=True)
    Recording1=SyncOEpyPhotometrySession(dpath,recordingName,IsTracking=False,
                                         read_aligned_data_from_file=True,
                                         recordingMode='Atlas',indicator='GEVI') 
    
    Recording1.pynacollada_label_theta (LFP_channel,Low_thres=theta_low_thres,High_thres=10,save=False,plot_theta=True)
    trough_index,peak_index =Recording1.plot_theta_correlation(LFP_channel,save_path)
    theta_part=Recording1.theta_part
    #theta_part=Recording1.Ephys_tracking_spad_aligned
    theta_zscores_np,theta_lfps_np=OE.get_theta_cycle_value(theta_part, LFP_channel, trough_index, half_window=0.5, fs=Recording1.fs)
    plot_aligned_theta_phase (save_path,LFP_channel,recordingName,theta_lfps_np,theta_zscores_np,Fs=10000)
    #plot_raster_histogram_theta_phase (save_path,theta_lfps_np,theta_zscores_np,Fs=10000)
    
    return -1

def run_theta_plot_main():
    'This is to process a single or concatenated trial, with a Ephys_tracking_photometry_aligned.pkl in the recording folder'
   
    dpath=r'C:\SPAD\Data\OEC\1765508_Jedi2p_Atlas\Day3'
    recordingName='SyncRecording13'


    savename='ThetaSave_Move'
    '''You can try LFP1,2,3,4 and plot theta to find the best channel'''
    LFP_channel='LFP_1'
    run_theta_plot_all_cycle (dpath,LFP_channel,recordingName,savename,theta_low_thres=-0.7) #-0.3

def main():    
    run_theta_plot_main()
    
if __name__ == "__main__":
    main()