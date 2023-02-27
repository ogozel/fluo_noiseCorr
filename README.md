# fluo_noiseCorr #

This repository contains the code for the postprocessing and analysis of the NIH U19 calcium imaging and behavioral data.
A subset of this code can be used to replicate the noise correlation results from "Circuit-based framework for fine spatial scale clustering of orientation tuning in mouse V1" (in preparation).

## Organization of the datasets ##

Each recording session consists of:
* 5 files of preprocessed calcium fluorescence traces (each representing 20' of recordings), in some instances we have several sets of 5 files (each with a different neuropil factor)
* (optional) face motion file containing the eigenfaces and projections on them (defined using the whole recording video)
* (optional) pupil file containing several parameters (such as area, long/short axis length, etc) during the whole recording

NB: The sampling frequency for fluorescence recordings differs between datasets, and is not identical to the behavioral sampling frequency.

Script that summarizes all the available datasets:
* dataSpecs.py: this file generates .hdf files that summarize the main chracteristics of all available datasets (L4 cytosolic, L2/3 and L2/3-targeting thalamic boutons, L4-targeting thalamic boutons). If new datasets become available, add the corresponding characteristics in this script, and rerun to get updated .hdf files. The dataDir has to be modified appropriately.


### Postprocessing of the datasets ###

The scripts to postprocess the datastes are the following:
* main_postprocess.py : this file contains the full preprocessing pipeline, combines all the available data files, and save Pandas Dataframes into a single file to make subsequent analysis convenient. Before running the script, the folder path has to be modified appropriately and a few decisions have to be made concerning which recording session to preprocess (section "Choose parameters of the data to postprocess" in the script). Nothing else should be modified.
* globalParams.py : this file contains the global parameters for the datasets. Only the paths to data (projectDir, dataDir, and processedDataDir) have to be modified appropriately.
* functions_postprocess.py : this file contains all the functions that do the heavy-lifting for the postprocessing of the data (eg. loading of the data, selection of the ROIs, postprocessing of the behavioral data, different plotting functions to get a sense of the dataset during postprocessing). Only the user's directory has to be modified appropriately.

To posprocess a dataset, simply run main_postprocess.py after having determined the parameters of the chosen dataset. This can be done on the local machine and should take a few minutes maximum (depending on the number of ROIs for this particular dataset). While postprocessing the data, the pipeline will also plot some useful figures (such as visualization of the ROIs color-coded by spontaneously active/preferred orientation, pairwise correlation as a function of pairwise distance, etc).

