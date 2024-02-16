# SPADPhotometryAnalysis
 Processing SPAD and pyPhotometry data to get zscore and normalised traces.
## pyPhotometry data analysis
Analysis for pyPhotometry data is modified from:
https://github.com/katemartian/Photometry_data_processing

`photometry_functions.py` provides functions to read, batch-read pyPhotometry data that saved as .csv, it also includes codes to integrate photometry analysis to Cheeseboard task.

`PhotometryTraceAnalysis.py` is the main file to read and process pyPhotometry data, you'll be able to save normalised signal trace `"Green_traceAll.csv"`, reference trace `"Red_traceAll.csv"`, zscore trace `"Zscore_traceAll.csv"`, and a CamSync file if synchronisation is included: `"CamSync_photometry.csv"`.

### Cheeseboard task with photometry recording
`pyCheese_singleTrial.py` and `pyCheese_multiTrial.py` provide photometry data analysis with cheeseboard task. 

A prerequisition is COLD pipeline (Lewis-Fallows 2024) to process cheeseboard behavioural data. 

NOTE: These two files are designed for a specific experiment, you do not need them to perform other pyPhotometry related analysis.

## SPAD-SPC imager data processing
More information about the SPC imager can be found in the README of this repository: 
https://github.com/MattNolanLab/SPAD_in_vivo

`mainAnalysisSPC.py` is the main function to run to process data recorded by the SPC imager. This file includes lines to process binary files, demodulate time division mode recordings. Saved results are: normalised signal trace `"Green_traceAll.csv"`, reference trace `"Red_traceAll.csv"`, zscore trace `"Zscore_traceAll.csv"`

`SPADreadBin.py` provides decoding functions for the binary data (,bin) saved by the SPC imager. Usually, we don't need to change anything here, functions are called by the `mianAnalysisSPC.py`.

`SPADdemod.py` provides demodulating functions to demodulate signal and unmix the neural signal trace and the reference trace.----For photometry imaging, we often have two channels, one is fluorescence signal that report neural activity, the other is a reference that does not change with neural activity but may report movement artefact. Therefore, time-division modulation or frequency modulation are used to modulate the two light channels. The modulation fuctions are not inbuild in the SPAD imaging system, we modulate two LEDs for excitation and two emissions are mixed in the raw imaging data. 



