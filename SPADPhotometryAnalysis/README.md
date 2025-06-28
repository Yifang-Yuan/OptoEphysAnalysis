# SPADPhotometryAnalysis
Processing SPAD and pyPhotometry data to get zscore and normalised traces. This includes codes to (1) decode SPAD .bin data if the recording is made by SPC imager; (2) analyse photometry .csv raw data from the pyPhotometry system or decoded SPAD data from step (1); (3) some cheeseboard related analysis.
 
## SPAD-SPC imager data processing
For the SPC imager, more information about the SPC imager can be found in the README of this repository: 
https://github.com/MattNolanLab/SPAD_in_vivo

### Method 1. Batch pre-processing

**Step 1. Decode raw .bin files.**

`BatchPreReadSPADfolder.py` can process multiple trial folders under a one-day session folder. To run this:

(1) put all your SPAD data path in the 'main' function; (2) modify the ROI xxrange and yyrange from screenshot or the FOV image saved by `DemoSingleSPAD_folder.py`.

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/8f8a1a1b-a686-42f7-9dc6-9cb8fd16a1d0)

**Step 2. Demodulate `traceValueAll.csv`.**

`BatchPreReadSPADfolder.py` will demodulate multiple SPAD trial folders and save zscore values as .csv files. However, demodulation is a bit tricy since you need to check the raw data to set thresholds, you may also want to check the first trial and the last trial's raw trace to make sure the recorded signal is stble across that day.

After demodulation, you will get three more files in each trial folder (above screenshot).

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/ecdb177b-13d3-477d-bcfd-56cf4514bfc4)

### Method 2. Process a single SPAD folder.
`DemoSingleSPAD_folder.py` is single trial/folder demo to process data recorded by the SPC imager. This file includes lines to process binary files, demodulate time division mode recordings. Saved results are: normalised signal trace `"Green_traceAll.csv"`, reference trace `"Red_traceAll.csv"`, zscore trace `"Zscore_traceAll.csv"`

**Note:**

`SPADreadBin.py` provides decoding functions for the binary data (,bin) saved by the SPC imager. Usually, we don't need to change anything here, functions are called by the `mianAnalysisSPC.py`.

`SPADdemod.py` provides demodulating functions to demodulate signal and unmix the neural signal trace and the reference trace.----For photometry imaging, we often have two channels, one is fluorescence signal that report neural activity, the other is a reference that does not change with neural activity but may report movement artefact. Therefore, time-division modulation or frequency modulation are used to modulate the two light channels. The modulation fuctions are not inbuild in the SPAD imaging system, we modulate two LEDs for excitation and two emissions are mixed in the raw imaging data. 

## Altas imager data processing
A trial of Atlas data is saved as a folder containing multiple .mat files, each .mat file contains data for a single frame. The folder name indicates the sampling rate and total number of frames of this recording. 

![image](https://github.com/user-attachments/assets/fcd34d5e-2a47-4050-a915-cdb8b973ed70)

**Note: ATLAS sampling rate can be changed with smaller ROI size:**
Fs_atlas=841.68 when using full ROI
Fs_atlas=1682.92 when using half of the ROI

`AtlasDecode.py` contains functions that can decode ATLAS data. 
`DemoSingleAtlasSmallFOV.py` can analyse a single ATLAS folder data.
`BatchPreReadAtlasFolder.py` can batch read a parent folder, but need to change / comment codes in the main function for smaller ROI analysis.

For small ROI analysis:
`DemoSingleAtlasfolder.py` will call functions in 'AtlasDecode.py' to get raw optical trace and an average image of this recording. This file can be used to check and find the ROI of the fibre tip.
`BatchPreReadAtlasFolder.py` can batch read a parent folder with multiple Altas recording folders.

## pyPhotometry data analysis
Analysis for pyPhotometry data is modified from:
https://github.com/katemartian/Photometry_data_processing

`photometry_functions.py` provides functions to read, batch-read pyPhotometry data that saved as .csv, it also includes codes to integrate photometry analysis to Cheeseboard task.

`PhotometryAnalysisSingleTrialDemo.py` is the main file to read and process pyPhotometry data, you'll be able to save normalised signal trace `"Green_traceAll.csv"`, reference trace `"Red_traceAll.csv"`, zscore trace `"Zscore_traceAll.csv"`, and a CamSync file if synchronisation is included: `"CamSync_photometry.csv"`.

`PreReadpyPhotometryFolder.py` in the parent folder can be used to batch process multiple pyPhotometry .csv data files in a folder. See: https://github.com/Yifang-Yuan/OptoEphysAnalysis/tree/main

## Cheeseboard task with photometry recording
`pyCheese_singleTrial.py` and `pyCheese_multiTrial.py` provide photometry data analysis with cheeseboard task. 

A prerequisition is COLD pipeline (Lewis-Fallows 2024) to process cheeseboard behavioural data. 

NOTE: These two files are designed for a specific experiment, you do not need them to perform other pyPhotometry related analysis.


