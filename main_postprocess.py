# -*- coding: utf-8 -*-
"""
@author: Olivia Gozel

Combining all datasets for each recording session, and then post-processing
fluorescence data
(1) ROIs should be at least 5um away from each others, and if a pair
    is too close, only the largest one is kept
(2) Classify ROIs as "visually evoked" or "spontaneously active" by running 
    an anova over the mean df/f0 of all blank frames versus the mean df/f0 of
    all grating frames.
(3) Determine which of the "visually evoked" ROIs are orientation-selective 
    (OS) by one-way anova (ROIs which are not responsive to the grating stimuli
    cannot be OS)
(4) Determine the orientation preference of OS ROIs (not a good tuning since 
    there are only four orientations)
    
This code can be used to postprocess different datasets ('dataType'):
- 'L4_cytosolic': simultaneous L4 cytosolic data with pupil and face motion
- 'ThalamicAxons_L23': simultaneous L2/3 data with thalamic boutons to L2/3 
                       and behavior
- 'L4_LGN_targeted_axons': simultaneous thalamic boutons to L4 and behavior
"""

import numpy as np
import pandas as pd
import os

os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_postprocess


#%% Choose parameters of the data to postprocess

# Type of dataset
dataType = 'L4_cytosolic'
# 'L4_cytosolic'
# 'L23_thalamicBoutons'
# 'L4_LGN_targeted_axons'

# Number of the dataset of interest
idxDataset = 0

# Thalamic boutons ('True') or V1 ROIs ('False')
boolBoutons = False

# Neuropil factor
dataNeuropilSub = globalParams.neuropilSub[3]


#%% Recover data specifications

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
nBlankFrames = dataSpecs.iloc[idxDataset]['nBlankFrames']
nStimFrames = dataSpecs.iloc[idxDataset]['nStimFrames']
dataFramesPerTrial = nBlankFrames + nStimFrames


#%% Load fluorescence data (V1) and classify ROIs

print("Loading fluorescence data (V1)...")
dff0, order, positionROI_3d, nTrialsPerSession = \
    functions_postprocess.loadData(dataType, dataSessions, dataDate, 
                                   dataMouse, dataDepth, dataNeuropilSub, 
                                   nBlankFrames, nStimFrames)

# Parameters for this dataset
nROI_init, nTrials, fluoPlaneWidth, fluoPlaneHeight, __, __ = \
    functions_postprocess.determine_params(dff0, positionROI_3d, nBlankFrames,
                                           nStimFrames)

print("Classifying neurons...")
charROI, charTrials, fluo, data, positionROI, distROI, idxKept = \
    functions_postprocess.selectROIs(dataType, pixelSize, dataSessions,
                                     nTrialsPerSession, order, dff0, 
                                     positionROI_3d, nBlankFrames, nStimFrames)


#%% Take care of the behavioral data if present

# Pupil data
charPupil = functions_postprocess.postprocess_pupil(dataType, dataDate,
                                                    dataMouse, dataDepth)
# Plot the average pupil area per orientation
if charPupil is not None:
    data = pd.concat([charTrials, charPupil['pupilArea']], axis=1)
    functions_postprocess.plot_avgPupilPerOri(data, dataFramesPerTrial)
    
# Motion data
motSVD = functions_postprocess.postprocess_motion(dataType, dataDate, 
                                                  dataMouse, dataDepth)


#%% Save data
# NB: To read the dataframes within the saved file:
# data = pd.read_hdf(filepath,'data')

print("Saving all the data...")
savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' \
    + dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'

fluo.to_hdf(savefilepath, key='fluo')
charROI.to_hdf(savefilepath, key='charROI')
charTrials.to_hdf(savefilepath ,key='charTrials')
positionROI.to_hdf(savefilepath, key='positionROI')
distROI = pd.DataFrame(distROI)
distROI.to_hdf(savefilepath, key='distROI')

fluoPlaneWidthHeight = np.array([fluoPlaneWidth, fluoPlaneHeight])
fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
fluoPlaneWidthHeight.to_hdf(savefilepath, key='fluoPlaneWidthHeight')

if charPupil is not None:
    charPupil.to_hdf(savefilepath, key='charPupil')

if motSVD is not None:
    motSVD.to_hdf(savefilepath, key='motSVD')


#%% Load thalamic bouton data and select good boutons for further analyses

if boolBoutons:
    
    # No neuropil subtraction for the thalamic boutons
    boutonNeuropilSub = globalParams.neuropilSub[0]
    
    print("Loading thalamic boutons fluorescence data...")
    nTrialsPerSession, dff0, order, baseline_raw, positionROI_3d = \
        functions_postprocess.loadBoutonData(dataType, dataSessions, dataDate, 
                                             dataMouse, dataDepth, 
                                             boutonNeuropilSub, nBlankFrames, 
                                             nStimFrames)
    
    # Parameters for this dataset
    nROI_init, nTrials, fluoPlaneWidth, fluoPlaneHeight, __, __ = \
        functions_postprocess.determine_params(dff0, positionROI_3d, 
                                               nBlankFrames ,nStimFrames)
    
    print("Selecting thalamic boutons...")
    charROI, charTrials, fluo, fluo_array, data, positionROI, distROI = \
        functions_postprocess.selectBoutonROIs(dataType, pixelSize, 
                                               dataSessions, nTrialsPerSession,
                                               order, dff0, positionROI_3d, 
                                               nBlankFrames, nStimFrames)
    
    print("Taking care of the behavioral data if present...")
    
    # Pupil data
    charPupil = functions_postprocess.postprocess_pupil(dataType, dataDate, 
                                                        dataMouse, dataDepth)
    # Plot the average pupil area per orientation
    # !!! Not accurate for L4 boutons!!! (because not same number of frames of 
    # fluo and behavior)
    if charPupil is not None:
        data = pd.concat([charTrials, charPupil['pupilArea']], axis=1)
        functions_postprocess.plot_avgPupilPerOri(data, dataFramesPerTrial)
    
    # Motion data
    motSVD = functions_postprocess.postprocess_motion(dataType, dataDate, 
                                                      dataMouse, dataDepth)
    
    print("Saving all the data...")
    savefilepath = globalParams.processedDataDir + dataType +'_boutons_' + \
        dataDate + '_' + dataMouse + '_' + dataDepth + '_neuropilF_' + \
        boutonNeuropilSub + '.hdf'
    
    fluo.to_hdf(savefilepath, key='fluo')
    charROI.to_hdf(savefilepath, key='charROI')
    charTrials.to_hdf(savefilepath, key='charTrials')
    positionROI.to_hdf(savefilepath, key='positionROI')
    distROI = pd.DataFrame(distROI)
    distROI.to_hdf(savefilepath, key='distROI')
    fluoPlaneWidthHeight = np.array([fluoPlaneWidth, fluoPlaneHeight])
    fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
    fluoPlaneWidthHeight.to_hdf(savefilepath, key='fluoPlaneWidthHeight')
    
    if charPupil is not None:
        charPupil.to_hdf(savefilepath, key='charPupil')

    if motSVD is not None:
        motSVD.to_hdf(savefilepath, key='motSVD')


