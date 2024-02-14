# Analysis for simultaneous Ephys recording and photometry imaging by pyPhotometry or SPAD.
## Pre-requisition
### Packages that used for open-ephys analysis
These packages are implemented in the analysis code, but you need to install or download them first.

To decode OpenEphys binary data, you need to install: 
https://github.com/open-ephys/open-ephys-python-tools

To process open-ephys LTP data:
https://github.com/pynapple-org/pynapple

Pynacollada is the package I used for ripple detection:
https://github.com/PeyracheLab/pynacollada#getting-started

https://github.com/PeyracheLab/pynacollada/blob/main/pynacollada/PETH/Tutorial_PETH_Ripples.ipynb

### Analysing optical data
Before conducting analysis in this package, optical recordings should already be analysed and normalised signal, reference and zscore data should be saved as .csv files in your file folder.

I've put codes for processing both pyPhotometry and SPAD in the same repository:
https://github.com/Yifang-Yuan/SPADPhotometryAnalysis

With zscore signal data saved as a .csv file, and saved in the same folder as ephys results, you'll be able to proceed this analysis.

### Analysing behviour data
Behaviour data should also be analysed saved as .csv files with animals' coordinates in each camera frame.

Bonsai tracking and DeepLabCut can both be used to provide animal's coordinates. 


## Run order
OpenEphysTools.py---including functions to decode the Open Ephys binary data, analysis LFP data and ripple detection, no need to run. 

EphyPreProcessing.py---use this file to call 'OpenEphysTools.py' to analysis Open Ephys data, adjust SPAD sync and save pickle file for each recording session.

SyncOESPADSessionClass.py--- a class with synchronised LFP,SPAD,cam data, no need to run

CombineAnalysis.py---to create a class (call SyncOESPADSessionClass.py) and analyse a session of data.  
