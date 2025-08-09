# Analysis for simultaneous Ephys recording and photometry imaging by pyPhotometry or SPAD.
## Pre-requisition
### Packages that used for open-ephys analysis
These packages are implemented in the analysis code, but you need to install or download them first.

(1) To decode OpenEphys binary data, you need to install (or include this repository in the same project): 

https://github.com/open-ephys/open-ephys-python-tools

**NOTE:** This package has an issue on the sequence of reading `recording*` folder , i.e. if you have an experiment folder with more than 10 recordings, it will read the folders in this order: 1,10,2,3,4...,
Please refer to: https://github.com/Yifang-Yuan/OptoEphysAnalysis/tree/main/TroubleShootingOthersPackage. You can simply replace a file or replace a function in this package.

(2) To process open-ephys LTP data, install pynapple: https://github.com/pynapple-org/pynapple

**NOTE:** pynapple is somehow not compatible with the inbuild Spyder installater on Anaconda home page, i.e. if you create an environment for pynapple and want to use spyder, you have to use terminal to install and open Spyder (I'm not sure whether they've fixed this, my experience).

**IMPORTANT:** pynapple changed some data structure from Pandas to Numpy which causes errors in some functions, to install an older version of pyapple, use these in the terminal: 

`pip install pynapple==0.3.3`

(3) Pynacollada is used for ripple detection, include this repository:

https://github.com/PeyracheLab/pynacollada

Tutorial:

https://github.com/PeyracheLab/pynacollada#getting-started

https://github.com/PeyracheLab/pynacollada/tree/main/pynacollada/eeg_processing

(4) Power spectrum is analysed by Wavelet:

https://paos.colorado.edu/research/wavelets/

Functions are aleady included in my package so you don't need to install anything.

## Pre-processing

### Option 1: Pre-process optical, ephys, behavioural data in one go
Use `BatchPreProcessing.py` and change folder path according to your folder name.

Note: It works fine, but not recommended since it requires your folder to be nicely organised without missing files. Run optical, ephys and behavioural preprocessing separately will allow you to spot errors and file unmatched issues.

Your raw data should be put in a one-day session folder like this:
![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/7134ea10-b00b-4e9e-9124-b6a35ec3844f)

### Option 2: Run three pre-prpcess file separately.
Your raw data should be put in a one-day session folder like the above screenshot, but this method will allow you spot issues and missing files step by step.
#### Step1. Analysing optical data
##### NOTE: To analyse SPAD-photometry data recording from the SPC imager or the ATLAS imager, follow this document：

https://github.com/Yifang-Yuan/OptoEphysAnalysis/tree/main/SPADPhotometryAnalysis#readme

`PreReadpyPhotometryFolder.py` will process all pyPhotometry .csv data files in the same folder with a temporal order, i.e. files that created first will be read first. For each recording trial, a new folder **SyncRecording*** will be created to save .csv results (* start from 1). 

To run `PreReadpyPhotometryFolder.py`, simply change your folder path in the **def main():** function and run the file.

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/42e2cc9c-a2fb-4abc-a80a-b79990f07fc1)


With optical signal saved as a .csv file in the same folder as ephys results, you'll be able to proceed with the `CombineAnalysis.py` for single trial, or `BatchProcessRecordingClass.py` for a one-day session. 

Note: you can also demo a single trial analysis in this repository:

https://github.com/Yifang-Yuan/OptoEphysAnalysis/tree/main/SPADPhotometryAnalysis

In the above folder,`PhotometryAnalysisSingleTrialDemo.py` ----- process a single pyPhotometry recording trial.

#### Step2. Pre-processing Ephys data
`PreReadEphysFolder.py` will process all Open Ephys recordings in a same folder (usually named with data and time). If you've already processed optical data and saved them in **SyncRecording*** folders, and each optical recording is matched with an Open Ephys recording, results will be saved as `open_ephys_read_pd.pkl` in each folder with their paired optical results. 

To run `PreReadEphysFolder.py`, change your folder path in the **def main():** function and run the file.

Note: 

You can still use `DemoEphyPreProcessSingleTrial.py` to call 'OpenEphysTools.py' to analysis Open Ephys data of a single recording, it uses sync line to generate SPAD_mask and py_mask, save pickle file for each recording session.

#### Step3. Analysing behviour data
Behaviour data should also be analysed saved as .csv files with animals' coordinates in each camera frame. Bonsai tracking and DeepLabCut can both be used to provide animal's coordinates. 

Method 1. Using Bonsai saved tracking .csv files and trial labels.

`PreReadBehaviourFolder.py` will read all behavioural tracking .csv in a same folder as well as a 'TrailLabel.csv'. The result will be saved as `AnimalTracking_*.pkl` file in each **SyncRecording*** folder.

After preprocessing, you will get this in each **SyncRecording*** folder:

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/5048d453-a176-48ae-8b1c-5be5e7254802)


## Post-processing for Optical signal and LFP combine analysis

### Processing by each trials

`CombineAnalysis.py`---to create a class (`SyncOESPADSessionClass.py` or `SyncOECPySessionClass.py`) and analyse a single trial (a single SyncRecording folder) of data.  

`BatchProcessRecordingClass.py`---a bactch analysis of all trials in the same folder (data from one session/one day).

Afer combine analysis, you will get more saved pkl files in each **SyncRecording*** folder, like this:

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/90deba4c-82e5-4efe-bbae-24eeab47525b)

Figures will automatically saved in a newly created **'Results'** folder (folder created by codes) in the parent folder for that one-day session (showed in one of above screenshots).

`SyncOESPADSessionClass.py``SyncOECPySessionClass.py`--- Class with synchronised LFP,SPAD,cam data, no need to run, but it might need to be modified to achieve new functions and analysis. 

`OpenEphysTools.py`---including functions to decode the Open Ephys binary data, analysis LFP data and ripple detection, no need to run and change, unless you want to change the Open Ephys channels, band-pass filters for pre-processing, Sync pulse channels or other output signal channels. 

'ConcatenateTrials.py'----can be used to concatenate all pre-training trials and all post-training trials to get a long single trace, and then use `CombineAnalysis.py` to treat it as a single trace. But be careful to use it, since the animal may have different states in different trials.

#### Example Results

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/80fb9f71-2329-4128-9df8-680b8776c326)

### Processing a whole object experiment (5-days data)
`ObjectExpPool.py` will pool all pre-awake, pre-sleep, open-field, post-awake, post-sleep trials together for comparison.

Resulted figures will be saved in a newly created folder **'ResultsPooled'** in the parent folder that you saved multiple days data. 

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/9193b15a-9620-44d6-8be6-d98bc79f4762)

#### Example Results
![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/208da4d1-d08e-4035-8f48-04c303b51d10)

## Hippocampus network oscillation analysis
### Analyse and visualise theta, gamma and SWR
`plotTheta.py`,`plotGamma.py`,`plotRipple.py` codes have following funtions: 

(1) Generate heatmap, average trace for LFP and optical signal during theta, gamma and ripple.

(2) Run Phase modulation analysis and calculate phase modulation index, prefered phase, phase modulation depth, ripple modulation depth. 

#### Example Results
<img width="1227" height="960" alt="image" src="https://github.com/user-attachments/assets/882ddee3-0f74-468f-91d6-7e476f71e6a4" />

### Plot Example traces with short durations.
`plot_example_traces.py` run example trace plot for a trial, by cutting the trial to 3 seconds (user defined) short period, and plot all segments' LFP and optical signal trace with LFP and optical signal power spectrogram.

#### Example Results
<img width="655" height="662" alt="image" src="https://github.com/user-attachments/assets/cec9c65c-e0db-4fca-856a-3496374d9c0e" />




 
