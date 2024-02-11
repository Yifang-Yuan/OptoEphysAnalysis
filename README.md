# Packages that used in this analysis
https://github.com/open-ephys/open-ephys-python-tools

https://github.com/pynapple-org/pynapple

https://github.com/PeyracheLab/pynacollada#getting-started

https://github.com/PeyracheLab/pynacollada/blob/main/pynacollada/PETH/Tutorial_PETH_Ripples.ipynb

# Run order
OpenEphysTools.py---including functions to decode the Open Ephys binary data, analysis LFP data and ripple detection, no need to run. 

EphyPreProcessing.py---use this file to call 'OpenEphysTools.py' to analysis Open Ephys data, adjust SPAD sync and save pickle file for each recording session.

SyncOESPADSessionClass.py--- a class with synchronised LFP,SPAD,cam data, no need to run

CombineAnalysis.py---to create a class (call SyncOESPADSessionClass.py) and analyse a session of data.  
