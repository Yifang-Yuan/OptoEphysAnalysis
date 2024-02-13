# Analysis for simultaneous Ephys recording and photometry imaging by pyPhotometry or SPAD.

## Pre-requisition
Before conducting analysis in this package, optical recordings should already be analysed and normalised signal, reference and zscore data should be saved as .csv files in your file folder.

Behaviour data should also be analysed saved as .csv files with animals' coordinates in each camera frame.

### Analysing optical data
For pyPhotometry, I use 
https://github.com/katemartian/Photometry_data_processing
to save normalised signal, reference traces, and th zscore.

For SPAD data analysis, please refer to SPAD analysis (I'll make a new cleaner repository)
https://github.com/Yifang-Yuan/SPAD_in_vivo/tree/main/SPAD_Python

### Analysing behviour data
Bonsai tracking and DeepLabCut are used for my analysis. 

### Packages that used for open-ephys analysis
https://github.com/open-ephys/open-ephys-python-tools

https://github.com/pynapple-org/pynapple

https://github.com/PeyracheLab/pynacollada#getting-started

https://github.com/PeyracheLab/pynacollada/blob/main/pynacollada/PETH/Tutorial_PETH_Ripples.ipynb

## Run order
OpenEphysTools.py---including functions to decode the Open Ephys binary data, analysis LFP data and ripple detection, no need to run. 

EphyPreProcessing.py---use this file to call 'OpenEphysTools.py' to analysis Open Ephys data, adjust SPAD sync and save pickle file for each recording session.

SyncOESPADSessionClass.py--- a class with synchronised LFP,SPAD,cam data, no need to run

CombineAnalysis.py---to create a class (call SyncOESPADSessionClass.py) and analyse a session of data.  
