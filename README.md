# Analysis for simultaneous Ephys recording and photometry imaging by pyPhotometry or SPAD.
## Pre-requisition
### Packages that used for open-ephys analysis
These packages are implemented in the analysis code, but you need to install or download them first.

(1) To decode OpenEphys binary data, you need to install (or include this repository in the same project): 

https://github.com/open-ephys/open-ephys-python-tools

**NOTE:** This package has an issue on the sequence of reading `recording*` folder , i.e. if you have an experiment folder with more than 10 recordings, it will read the folders in this order: 1,10,2,3,4...,
Please refer to: https://github.com/Yifang-Yuan/OptoEphysAnalysis/tree/main/TroubleShootingOthersPackage. You can simply replace a file or replace a function in this package.

(2) To process open-ephys LTP data, install pynapple:

**NOTE:** pynapple is somehow not compatible with the inbuild Spyder installater on Anaconda home page, i.e. if you create an environment for pynapple and want to use spyder, you have to use terminal to install and open Spyder (I'm not sure whether they've fixed this, my experience).

https://github.com/pynapple-org/pynapple

(3) Pynacollada is used for ripple detection, include this repository:

https://github.com/PeyracheLab/pynacollada

Tutorial:

https://github.com/PeyracheLab/pynacollada#getting-started

https://github.com/PeyracheLab/pynacollada/tree/main/pynacollada/eeg_processing

(4) Power spectrum is analysed by Wavelet:

https://paos.colorado.edu/research/wavelets/

Functions are aleady included in my package so you don't need to install anything.

### Analysing optical data
`PreReadpyPhotometryFolder.py` will process all pyPhotometry .csv data files in the same folder with a temporal order, i.e. files that created first will be read first. For each recording trial, a new folder **SyncRecording*** will be created to save .csv results (* start from 1). 

With optical signal saved as a .csv file in the same folder as ephys results, you'll be able to proceed with the `CombineAnalysis.py`.

![微信截图_20240219113507](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/b971b6aa-ffc1-42af-a69e-0b723ad6bf50)

Note: you can also demo a single trial analysis in this repository:

https://github.com/Yifang-Yuan/OptoEphysAnalysis/tree/main/SPADPhotometryAnalysis

In the above folder,`PhotometryAnalysisSingleTrialDemo.py` ----- process a single pyPhotometry recording trial.

### Analysing behviour data
Behaviour data should also be analysed saved as .csv files with animals' coordinates in each camera frame.

Bonsai tracking and DeepLabCut can both be used to provide animal's coordinates. 

### Pre-processing Ephys data
`PreReadEphysFolder.py` will process all Open Ephys recordings in a same folder (usually named with data and time). If you've already processed optical data and saved them in **SyncRecording*** folders, and each optical recording is matched with an Open Ephys recording, results will be saved as `open_ephys_read_pd.pkl` in each folder with their paired optical results. 

![image](https://github.com/Yifang-Yuan/OptoEphysAnalysis/assets/77569999/da1b83da-75c0-4f4a-9795-a453dbf18017)

Note: 

You can still use `DemoEphyPreProcessSingleTrial.py` to call 'OpenEphysTools.py' to analysis Open Ephys data of a single recording, it uses sync line to generate SPAD_mask and py_mask, save pickle file for each recording session.

## Combine analysis for Optical signal and LFP

`CombineAnalysis.py`---to create a class (`SyncOESPADSessionClass.py` or `SyncOECPySessionClass.py`) and analyse a single trial (a single SyncRecording folder) of data.  

`BatchProcessRecordingClass.py`---a bactch analysis of all trials in the same folder (data from one session/one day).

`SyncOESPADSessionClass.py``SyncOECPySessionClass.py`--- Class with synchronised LFP,SPAD,cam data, no need to run, but it might need to be modified to achieve new functions and analysis. 

`OpenEphysTools.py`---including functions to decode the Open Ephys binary data, analysis LFP data and ripple detection, no need to run and change, unless you want to change the Open Ephys channels, band-pass filters for pre-processing, Sync pulse channels or other output signal channels. 

'ConcatenateTrials.py'----can be used to concatenate all pre-training trials and all post-training trials to get a long single trace, and then use `CombineAnalysis.py` to treat it as a single trace. But be careful to use it, since the animal may have different states in different trials.
